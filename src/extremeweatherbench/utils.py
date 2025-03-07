"""Utility functions for the ExtremeWeatherBench package that don't fit into any
other specialized package.
"""

from typing import Union, List
from collections import namedtuple
import fsspec
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
import datetime
from pathlib import Path
from importlib import resources
import yaml

#: Struct packaging latitude/longitude location definitions.
Location = namedtuple("Location", ["latitude", "longitude"])

#: Maps the ARCO ERA5 to CF conventions.
ERA5_MAPPING = {
    "air_temperature": "2m_temperature",
    "eastward_wind": "10m_u_component_of_wind",
    "northward_wind": "10m_v_component_of_wind",
    "air_pressure_at_mean_sea_level": "mean_sea_level_pressure",
    "specific_humidity": "specific_humidity",
    "valid_time": "time",
    "level": "level",
    "latitude": "latitude",
    "longitude": "longitude",
}


def convert_longitude_to_360(longitude: float) -> float:
    """Convert a longitude from the range [-180, 180) to [0, 360)."""
    return np.mod(longitude, 360)


def convert_longitude_to_180(
    dataset: Union[xr.Dataset, xr.DataArray], longitude_name: str = "longitude"
) -> Union[xr.Dataset, xr.DataArray]:
    """Coerce the longitude dimension of an xarray data structure to [-180, 180)."""
    dataset.coords[longitude_name] = (dataset.coords[longitude_name] + 180) % 360 - 180
    dataset = dataset.sortby(longitude_name)
    return dataset


def clip_dataset_to_bounding_box_degrees(
    dataset: xr.Dataset, location_center: Location, box_degrees: Union[tuple, float]
) -> xr.Dataset:
    """Clip an xarray dataset to a box around a given location in degrees latitude & longitude.

    Args:
        dataset: The input xarray dataset.
        location_center: A Location object corresponding to the center of the bounding box.
        box_degrees: The side length(s) of the bounding box in degrees, as a tuple (lat,lon) or single value.

    Returns:
        The clipped xarray dataset.
    """

    lat_center = location_center.latitude
    lon_center = location_center.longitude
    if lon_center < 0:
        lon_center = convert_longitude_to_360(lon_center)
    if isinstance(box_degrees, tuple):
        box_degrees_lat, box_degrees_lon = box_degrees
    else:
        box_degrees_lat = box_degrees
        box_degrees_lon = box_degrees
    min_lat = lat_center - box_degrees_lat / 2
    max_lat = lat_center + box_degrees_lat / 2
    min_lon = lon_center - box_degrees_lon / 2
    max_lon = lon_center + box_degrees_lon / 2
    if min_lon < 0:
        min_lon = convert_longitude_to_360(min_lon)
    if min_lon > max_lon:
        # Ensure max_lon is always the larger value and account for cyclic nature of lon
        min_lon, max_lon = max_lon, min_lon
        clipped_dataset = dataset.sel(
            latitude=(dataset.latitude > min_lat) & (dataset.latitude <= max_lat),
            longitude=(dataset.longitude < min_lon) | (dataset.longitude >= max_lon),
        )
    else:
        clipped_dataset = dataset.sel(
            latitude=(dataset.latitude > min_lat) & (dataset.latitude <= max_lat),
            longitude=(dataset.longitude > min_lon) & (dataset.longitude <= max_lon),
        )
    return clipped_dataset


def convert_day_yearofday_to_time(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Convert dayofyear and hour coordinates in an xarray Dataset to a new time
    coordinate.

    Args:
        dataset: The input xarray dataset.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f"{year}-01-01",
        periods=len(dataset["dayofyear"]) * len(dataset["hour"]),
        freq="6h",
    )
    dataset = dataset.stack(time=("dayofyear", "hour")).drop(["dayofyear", "hour"])
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(time=time_dim)

    return dataset


def remove_ocean_gridpoints(dataset: xr.Dataset) -> xr.Dataset:
    """Subset a dataset to only include land gridpoints based on a land-sea mask.

    Args:
        dataset: The input xarray dataset.

    Returns:
        The dataset masked to only land gridpoints.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_sea_mask = land.mask(dataset.longitude, dataset.latitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    dataset = dataset.where(land_mask, drop=True)

    return dataset


def _open_kerchunk_zarr_reference_jsons(
    file_list, forecast_schema_config, remote_protocol: str = "s3"
):
    """Open a dataset from a list of kerchunk JSON files."""
    xarray_datasets = []
    for json_file in file_list:
        fs_ = fsspec.filesystem(
            "reference",
            fo=json_file,
            ref_storage_args={"skip_instance_cache": True},
            remote_protocol=remote_protocol,
            remote_options={"anon": True},
        )
        m = fs_.get_mapper("")
        ds = xr.open_dataset(
            m, engine="zarr", backend_kwargs={"consolidated": False}, chunks="auto"
        )
        if "initialization_time" not in ds.attrs:
            raise ValueError(
                "Initialization time not found in dataset attributes. \
                             Please add initialization_time to the dataset attributes."
            )
        else:
            model_run_time = np.datetime64(
                pd.to_datetime(ds.attrs["initialization_time"])
            )
        ds[forecast_schema_config.init_time] = model_run_time.astype("datetime64[ns]")
        ds = ds.set_coords(forecast_schema_config.init_time)
        ds = ds.expand_dims(forecast_schema_config.init_time)
        for variable in forecast_schema_config.__dict__:
            attr_value = getattr(forecast_schema_config, variable)
            if attr_value in ds.data_vars:
                ds = ds.rename({attr_value: variable})
        ds = ds.transpose(
            forecast_schema_config.init_time,
            forecast_schema_config.valid_time,
            forecast_schema_config.latitude,
            forecast_schema_config.longitude,
            forecast_schema_config.level,
        )
        xarray_datasets.append(ds)

    return xr.concat(xarray_datasets, dim=forecast_schema_config.init_time)


def _open_mlwp_kerchunk_reference(
    file, forecast_schema_config, remote_protocol: str = "s3"
):
    """Open a dataset from a kerchunked reference file for the OAR MLWP S3 bucket."""
    if "parq" in file:
        storage_options = {
            "remote_protocol": "s3",
            "skip_instance_cache": True,
            "remote_options": {"anon": True},
            "target_protocol": "file",
            "lazy": True,
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        ds = xr.open_dataset(
            file,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
    else:
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": file,
                    "remote_protocol": remote_protocol,
                    "remote_options": {"anon": True},
                },
            },
        )
    ds = ds.rename({"time": "lead_time"})
    ds["lead_time"] = range(0, 241, 6)
    for variable in forecast_schema_config.__dict__:
        attr_value = getattr(forecast_schema_config, variable)
        if attr_value in ds.data_vars:
            ds = ds.rename({attr_value: variable})
        # Step 1: Create a meshgrid of init_time and lead_time
    lead_time_grid, init_time_grid = np.meshgrid(ds.lead_time, ds.init_time)
    # Step 2: Flatten the meshgrid and convert lead_time to timedelta
    valid_time = init_time_grid.flatten() + pd.to_timedelta(
        lead_time_grid.flatten(), unit="h"
    )
    ds.coords["time"] = valid_time
    return ds


def map_era5_vars_to_forecast(forecast_schema_config, forecast_dataset, era5_dataset):
    """Map ERA5 variable names to forecast variable names."""
    era5_subset_list = []
    for variable in forecast_schema_config.__dict__:
        if variable in forecast_dataset.data_vars:
            era5_dataset = era5_dataset.rename({ERA5_MAPPING[variable]: variable})
            era5_subset_list.append(variable)
    return era5_dataset[era5_subset_list]


def expand_lead_times_to_6_hourly(
    dataarray: xr.DataArray, max_fcst_hour: int = 240, fcst_output_cadence: int = 6
) -> xr.DataArray:
    """Makes hours in metrics output for max_fcst_hour hours at a fcst_cadence-hourly rate.
    Depending on initialization time and output cadence, there may be missing lead times
    in final output of certain metrics."""
    all_hours = np.arange(0, max_fcst_hour + 1, fcst_output_cadence)
    final_data = []
    final_times = []
    current_idx = 0
    for hour in all_hours:
        if (
            current_idx < len(dataarray.lead_time)
            and dataarray.lead_time[current_idx] == hour
        ):
            final_data.append(dataarray.values[current_idx])
            current_idx += 1
        else:
            final_data.append(None)
        final_times.append(hour)
    dataarray = xr.DataArray(
        data=final_data, dims=["lead_time"], coords={"lead_time": final_times}
    ).astype(float)
    return dataarray


def process_dataarray_for_output(da_list: List[xr.DataArray]) -> xr.DataArray:
    """Extract and format data from a list of DataArrays.

    Args:
        da_list: A list of xarray DataArrays.

    Returns:
        A DataArray with a sorted lead_time coordinate, expanded to 6-hourly intervals.
    """

    if len(da_list) == 0:
        # create dummy nan dataarray in case no applicable forecast valid times exist
        output_da = xr.DataArray(
            data=[np.nan],
            dims=["lead_time"],
            coords={"lead_time": [0]},
        )
    else:
        output_da = xr.concat(da_list, dim="lead_time")
        # Reverse the lead time so that the minimum lead time is first
        output_da = output_da.isel(lead_time=slice(None, None, -1))
    output_da = expand_lead_times_to_6_hourly(output_da)
    return output_da


def center_forecast_on_time(da: xr.DataArray, time: pd.Timestamp, hours: int):
    """Center a forecast DataArray on a given time with a given range in hours.

    Args:
        da: The forecast DataArray to center.
        time: The time to center the forecast on.
        hours: The number of hours to include in the centered forecast.
    """
    time_range = pd.date_range(
        end=pd.to_datetime(time) + pd.Timedelta(hours=hours),
        periods=hours * 2 + 1,
        freq="h",
    )
    return da.sel(time=slice(time_range[0], time_range[-1]))


def temporal_align_dataarrays(
    forecast: xr.DataArray,
    observation: xr.DataArray,
    init_time_datetime: datetime.datetime,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align the individual initialization time forecast and observation dataarrays.

    Args:
        forecast: The forecast dataarray to align.
        observation: The observation dataarray to align.
        init_time_datetime: The initialization time to subset the forecast dataarray by.

    Returns:
        A tuple containing the time-aligned forecast and observation dataarrays.
    """

    forecast = forecast.sel(init_time=init_time_datetime)
    time = np.array(
        [init_time_datetime + pd.Timedelta(hours=int(t)) for t in forecast["lead_time"]]
    )
    forecast = forecast.assign_coords(time=("lead_time", time))
    forecast = forecast.swap_dims({"lead_time": "time"})
    forecast, observation = xr.align(forecast, observation, join="inner")
    return (forecast, observation)


def align_observations_temporal_resolution(
    forecast: xr.DataArray, observation: xr.DataArray
) -> xr.DataArray:
    """Align observation dataarray on the forecast dataarray's temporal resolution.,

    Metrics which need a singular timestep from gridded obs will fail if the forecasts
    are not aligned with the observation timestamps (e.g. a 03z minimum temp in observations
    when the forecast only has 00z and 06z timesteps).

    Args:
        forecast: The forecast data which will be aligned against.
        observation: The observation data to align.

    Returns:
        The aligned observation dataarray.
    """
    obs_time_delta = pd.to_timedelta(np.diff(observation.time).mean())
    forecast_time_delta = pd.to_timedelta(np.diff(forecast.lead_time).mean(), unit="h")

    if forecast_time_delta > obs_time_delta:
        observation = observation.resample(time=forecast_time_delta).first()

    return observation


def truncate_incomplete_days(da: xr.DataArray) -> xr.DataArray:
    """Truncate a dataarray to only include full days of data."""
    # Group by dayofyear and check if each day has a complete times
    # Count how many unique hours exist per day in the data
    hours_per_day = len(np.unique(da.time.dt.hour.values))
    valid_days = da.groupby("time.dayofyear").count("time") == hours_per_day
    # Only keep days that have a full set of timestamps
    da = da.where(
        da.time.dt.dayofyear.isin(
            valid_days.where(valid_days).dropna(dim="dayofyear").dayofyear
        ),
        drop=True,
    )
    return da


def return_max_min_timestamp(da: xr.DataArray) -> pd.Timestamp:
    """Return the timestamp of the maximum minimum temperature in a DataArray."""
    return pd.Timestamp(
        da.where(
            da == da.groupby("time.dayofyear").min().max(),
            drop=True,
        ).time.values[0]
    )


def load_events_yaml():
    """Load the events yaml file."""
    import extremeweatherbench.data

    events_yaml_file = resources.files(extremeweatherbench.data).joinpath("events.yaml")
    with resources.as_file(events_yaml_file) as file:
        yaml_event_case = read_event_yaml(file)

    return yaml_event_case


def read_event_yaml(input_pth: str | Path) -> dict:
    """Read events yaml from data."""
    input_pth = Path(input_pth)
    with open(input_pth, "rb") as f:
        yaml_event_case = yaml.safe_load(f)
    return yaml_event_case
