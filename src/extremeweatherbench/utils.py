"""Utility functions for the ExtremeWeatherBench package that don't fit into any
other specialized package.
"""

import datetime
import itertools
import logging
from importlib import resources
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import regionmask
import xarray as xr
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#: Maps the ARCO ERA5 to CF conventions.
ERA5_MAPPING = {
    "surface_air_temperature": "2m_temperature",
    "surface_eastward_wind": "10m_u_component_of_wind",
    "surface_northward_wind": "10m_v_component_of_wind",
    "air_temperature": "temperature",
    "eastward_wind": "u_component_of_wind",
    "northward_wind": "v_component_of_wind",
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
    longitude: float | Union[xr.Dataset, xr.DataArray],
    longitude_name: str = "longitude",
) -> float | Union[xr.Dataset, xr.DataArray]:
    """Convert a longitude from the range [0, 360) to [-180, 180).

    Datasets are coerced to [-180, 180) and sorted by longitude."""
    if isinstance(longitude, xr.Dataset) or isinstance(longitude, xr.DataArray):
        longitude.coords[longitude_name] = (
            longitude.coords[longitude_name] + 180
        ) % 360 - 180
        longitude = longitude.sortby(longitude_name)
        return longitude
    else:
        return np.mod(longitude - 180, 360) - 180


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
    return dataset.where(land_mask)


def _open_mlwp_kerchunk_reference(
    file, forecast_schema_config, remote_protocol: str = "s3"
):
    """Open a dataset from a kerchunked reference file for the OAR MLWP S3 bucket."""
    if "parq" in file:
        storage_options = {
            "remote_protocol": remote_protocol,
            "remote_options": {"anon": True},
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        ds = xr.open_dataset(
            file,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
        ds = ds.compute()
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


def location_subset_point_obs(
    df: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
):
    """Subset a dataframe based upon maximum and minimum latitudes and longitudes.

    Arguments:
        df: dataframe with point obs.
        min_lat: minimum latitude.
        max_lat: maximum latitude.
        min_lon: minimum longitude.
        max_lon: maximum longitude.
        lat_name: name of latitude column.
        lon_name: name of longitude column.

    Returns a subset dataframe."""
    location_subset_mask = (
        (df[lat_name] >= min_lat)
        & (df[lat_name] <= max_lat)
        & (df[lon_name] >= min_lon)
        & (df[lon_name] <= max_lon)
    )
    return df[location_subset_mask]


def maybe_remove_missing_data_vars(data_var: List[str], df: pd.DataFrame):
    """Remove data variables from the list if they are not found in the dataframe.

    Args:
        data_var: The list of data variables to subset.
        df: The dataframe with accompanying variables to subset.

    Returns:
        The list of data variables without variables missing from the dataframe.
    """
    data_var_without_missing_vars = []
    for individual_data_var in data_var:
        if individual_data_var in df:
            data_var_without_missing_vars.append(individual_data_var)
        else:
            logger.warning(
                "Data variable %s not found in dataframe",
                individual_data_var,
            )
    return data_var_without_missing_vars


def align_point_obs_from_gridded(
    forecast_ds: xr.Dataset,
    case_subset_point_obs_df: pd.DataFrame,
    data_var: List[str],
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Takes in a forecast dataarray and point observation dataframe, aligning them by
    reducing dimensions. Metadata variables used are identical to those in PointObservationSchemaConfig.

    Args:
        forecast_ds: The forecast dataset.
        case_subset_point_obs_df: The point observation dataframe.
        data_var: The variable(s) to subset (e.g. "surface_air_temperature")

    Returns a tuple of aligned forecast and observation dataarrays. Will return a tuple of
    empty datasets if there is no valid data or overlap between the forecast and observation data.
    """
    if case_subset_point_obs_df.empty:
        logger.warning(
            "Observation data is empty, returning empty xarray datasets for forecast and observation."
        )
        return (xr.Dataset(), xr.Dataset())
    case_subset_point_obs_df.loc[:, "longitude"] = convert_longitude_to_360(
        case_subset_point_obs_df.loc[:, "longitude"]
    )
    case_subset_point_obs_df = location_subset_point_obs(
        case_subset_point_obs_df,
        float(forecast_ds["latitude"].min().values),
        float(forecast_ds["latitude"].max().values),
        float(forecast_ds["longitude"].min().values),
        float(forecast_ds["longitude"].max().values),
    )
    # Reset index to allow for easier modification
    case_subset_point_obs_df = case_subset_point_obs_df.reset_index()
    case_subset_point_obs_df.loc[:, "station_id"] = case_subset_point_obs_df[
        "station_id"
    ].astype("str")

    # Set up multiindex to enable slicing along individual timesteps
    case_subset_point_obs_df = case_subset_point_obs_df.set_index(
        [
            "station_id",
            "time",
        ]
    ).sort_index()

    aligned_forecast_list = []
    aligned_observation_list = []

    logger.debug(
        "number of init times: %s \n number of lead times: %s",
        len(forecast_ds.init_time),
        len(forecast_ds.lead_time),
    )
    logger.debug(
        "total pairs to analyze: %s",
        len(forecast_ds.init_time) * len(forecast_ds.lead_time),
    )
    # Loop init and lead times to prevent indexing error from duplicate valid times
    for init_time, lead_time in itertools.product(
        forecast_ds.init_time, forecast_ds.lead_time
    ):
        valid_time = pd.Timestamp(
            init_time.values + pd.to_timedelta(lead_time.values, unit="h").to_numpy()
        )
        valid_time_index = pd.IndexSlice[:, valid_time]
        if valid_time in case_subset_point_obs_df.index.get_level_values(1):
            obs_overlapping_valid_time = case_subset_point_obs_df.loc[
                valid_time_index, :
            ]
        else:
            logger.debug(
                "No valid time found in point obs for %s",
                valid_time.strftime("%Y-%m-%d %H:%M"),
            )
            continue
        obs_overlapping_valid_time = obs_overlapping_valid_time.reset_index()

        station_ids = obs_overlapping_valid_time["station_id"]
        lons = xr.DataArray(
            obs_overlapping_valid_time["longitude"].values, dims="station_id"
        )
        lats = xr.DataArray(
            obs_overlapping_valid_time["latitude"].values, dims="station_id"
        )

        forecast_at_obs_ds = forecast_ds.sel(
            init_time=init_time, lead_time=lead_time
        ).interp(latitude=lats, longitude=lons, method="nearest")
        forecast_at_obs_ds = forecast_at_obs_ds.assign_coords(
            {
                "station_id": station_ids,
                "time": valid_time,
            }
        )
        forecast_at_obs_ds.coords["lead_time"] = lead_time
        forecast_at_obs_ds.coords["init_time"] = init_time
        # Uses the dataframe attrs to apply metadata columns
        obs_overlapping_valid_time_ds = (
            obs_overlapping_valid_time.to_xarray().set_coords(
                case_subset_point_obs_df.attrs["metadata_vars"]
            )
        )

        # Subset the observation dataarray to only include the data variables of interest
        # which is checked for missing variables
        data_var = maybe_remove_missing_data_vars(data_var, case_subset_point_obs_df)
        obs_overlapping_valid_time_ds = obs_overlapping_valid_time_ds[data_var]

        obs_overlapping_valid_time_ds = obs_overlapping_valid_time_ds.swap_dims(
            {"index": "station_id"}
        )
        obs_overlapping_valid_time_ds.coords["lead_time"] = lead_time
        obs_overlapping_valid_time_ds.coords["init_time"] = init_time

        aligned_observation_list.append(obs_overlapping_valid_time_ds)
        aligned_forecast_list.append(forecast_at_obs_ds)
    # concat the dataarrays along the station_id dimension
    if len(aligned_forecast_list) == 0 or len(aligned_observation_list) == 0:
        return (xr.Dataset(), xr.Dataset())
    else:
        # Convert to Dataset before concat to ensure we always get a Dataset back
        # even if there's only one element in the list
        interpolated_forecast = xr.concat(
            [
                ds.to_dataset() if isinstance(ds, xr.DataArray) else ds
                for ds in aligned_forecast_list
            ],
            dim="station_id",
        )
        interpolated_observation = xr.concat(
            [
                ds.to_dataset() if isinstance(ds, xr.DataArray) else ds
                for ds in aligned_observation_list
            ],
            dim="station_id",
        )
        return (interpolated_forecast, interpolated_observation)


def derive_indices_from_init_time_and_lead_time(
    dataset: xr.Dataset,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> np.ndarray:
    """Derive the indices of valid times in a dataset when the dataset has init_time and lead_time coordinates.

    Args:
        dataset: The dataset to derive the indices from.
        start_date: The start date to derive the indices from.
        end_date: The end date to derive the indices from.

    Returns:
        The indices of valid times in the dataset.

    Example:
        >>> import xarray as xr
        >>> import datetime
        >>> import pandas as pd
        >>> from extremeweatherbench.utils import (
        ...     derive_indices_from_init_time_and_lead_time,
        ... )
        >>> ds = xr.Dataset(
        ...     coords={
        ...         "init_time": pd.date_range("2020-01-01", "2020-01-03"),
        ...         "lead_time": [0, 24, 48],  # hours
        ...     }
        ... )
        >>> start = datetime.datetime(2020, 1, 1)
        >>> end = datetime.datetime(2020, 1, 4)
        >>> indices = derive_indices_from_init_time_and_lead_time(ds, start, end)
        >>> print(indices)
        array([0, 0, 1, 1, 2])
    """
    lead_time_grid, init_time_grid = np.meshgrid(dataset.lead_time, dataset.init_time)
    valid_times = (
        init_time_grid.flatten()
        + pd.to_timedelta(lead_time_grid.flatten(), unit="h").to_numpy()
    )
    valid_times_reshaped = valid_times.reshape(
        (
            dataset.init_time.shape[0],
            dataset.lead_time.shape[0],
        )
    )
    valid_time_mask = (valid_times_reshaped > pd.to_datetime(start_date)) & (
        valid_times_reshaped < pd.to_datetime(end_date)
    )
    valid_time_indices = np.asarray(valid_time_mask).nonzero()

    # The first index will subset init_time based on the first valid_time_reshaped line above
    # we don't need to subset lead_time but it might be useful in the future
    init_time_subset_indices = valid_time_indices[0]

    return init_time_subset_indices


def maybe_convert_to_path(value: str | Path) -> str | Path:
    """Convert a string to a Path object if it's a local filesystem path.

    This function will:
    - Convert local filesystem paths to Path objects
    - Leave URLs and cloud storage paths as strings
    - Leave existing Path objects unchanged
    """
    if isinstance(value, str):
        # Check if it's a local filesystem path (not a URL or cloud storage path)
        if not any(
            value.startswith(prefix)
            for prefix in ["http://", "https://", "s3://", "gs://"]
        ):
            return Path(value)
    return value


# Type alias for use in type hints
PathOrStr = str | Path


def _default_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Default forecast preprocess function that does nothing."""
    return ds
