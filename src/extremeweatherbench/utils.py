"""Utility functions for the ExtremeWeatherBench package that don't fit into any
other specialized package.
"""

import logging
from typing import Optional, Union
from collections import namedtuple, defaultdict
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import ujson
import xarray as xr
from kerchunk.hdf import SingleHdf5ToZarr
from shapely.geometry import box

#: Struct packaging latitude/longitude location definitions.
Location = namedtuple("Location", ["latitude", "longitude"])

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
    # NOTE(daniel): I don't think this is actually correct :)
    return longitude % 360


def convert_longitude_to_180(
    dataset: Union[xr.Dataset, xr.DataArray], longitude_name: str = "longitude"
) -> Union[xr.Dataset, xr.DataArray]:
    """Coerce the longitude dimension of an xarray data structure to [-180, 180)."""
    dataset.coords[longitude_name] = (dataset.coords[longitude_name] + 180) % 360 - 180
    dataset = dataset.sortby(longitude_name)
    return dataset


def generate_json_from_nc(u, so, fs, fs_out, json_dir):
    """Generate a kerchunk JSON file from a NetCDF file.

    TODO(taylor): Define function signature and docstring.
    """
    with fs.open(u, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, u, inline_threshold=300)

        file_split = u.split(
            "/"
        )  # seperate file path to create a unique name for each json
        model = file_split[1].split("_")[0]
        date_string = file_split[-1].split("_")[3]
        outf = f"{json_dir}{model}_{date_string}_.json"
        print(outf)
        with fs_out.open(outf, "wb") as f:
            f.write(ujson.dumps(h5chunks.translate()).encode())


def is_jja(month: int):
    """Check if a month is in the boreal summer season (June, July, August)."""
    return (month >= 6) & (month <= 8)


def is_6_hourly(hour: int):
    """Check if an hour is evenly divisible by 6."""
    return (hour == 0) | (hour == 6) | (hour == 12) | (hour == 18)
    # return (hour % 6) == 0 and (hour >= 0) and (hour <= 18)


def clip_dataset_to_bounding_box(
    # NOTE(daniel): given its use here, "case.Location" should be moved to this module
    # or something else stand-alone; high likelihood of inadvertently introducing a
    # circular import dependency here.
    dataset: xr.Dataset,
    location_center: Location,
    length_km: float,
) -> xr.Dataset:
    """Clip an xarray dataset to a boxbox around a given location.

    Args:
        dataset: The input xarray dataset.
        location_center: A Location object corresponding to the center of the bounding box.
        length_km: The side length of the bounding box in kilometers.

    Returns:
        The clipped xarray dataset.
    """
    lat_center = location_center.latitude
    lon_center = location_center.longitude

    # Convert length from kilometers to degrees (approximation)
    length_deg = length_km / 111  # 1 degree is approximately 111 km

    # Create a bounding box
    min_lat = lat_center - (length_deg / 2)
    max_lat = lat_center + (length_deg / 2)
    min_lon = lon_center - (length_deg / 2)
    max_lon = lon_center + (length_deg / 2)

    # Create a GeoDataFrame with the bounding box
    bbox = gpd.GeoDataFrame(
        {"geometry": [box(min_lon, min_lat, max_lon, max_lat)]},
    )
    # Clip the dataset using the bounding box
    clipped_dataset = dataset.rio.write_crs("EPSG:4326").rio.clip(
        bbox.geometry, bbox.crs, drop=True
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
    # TODO(taylor): Extend this so that the user may pass their own land-sea mask,
    # best suited the dataset they're analyzing.
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_sea_mask = land.mask(dataset.longitude, dataset.latitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    dataset = dataset.where(land_mask, drop=True)

    return dataset


def _open_kerchunk_zarr_reference_jsons(file_list, forecast_schema_config):
    """Open a dataset from a list of kerchunk JSON files."""
    xarray_datasets = []
    for json_file in file_list:
        fs_ = fsspec.filesystem(
            "reference",
            fo=json_file,
            ref_storage_args={"skip_instance_cache": True},
            remote_protocol="gcs",
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
        for attr in dir(forecast_schema_config):
            if not attr.startswith("__"):
                attr_value = getattr(forecast_schema_config, attr)
                if attr_value in ds.data_vars:
                    ds = ds.rename({attr_value: attr})
        ds = ds.transpose(
            forecast_schema_config.init_time,
            forecast_schema_config.valid_time,
            forecast_schema_config.latitude,
            forecast_schema_config.longitude,
            forecast_schema_config.level,
        )
        xarray_datasets.append(ds)

    return xr.concat(xarray_datasets, dim=forecast_schema_config.init_time)


def map_era5_vars_to_forecast(forecast_schema_config, forecast_dataset, era5_dataset):
    """Map ERA5 variable names to forecast variable names."""
    era5_subset_list = []
    for attr in dir(forecast_schema_config):
        if not attr.startswith("__"):
            if attr in forecast_dataset.data_vars:
                era5_dataset = era5_dataset.rename({ERA5_MAPPING[attr]: attr})
                era5_subset_list.append(attr)
    return era5_dataset[era5_subset_list]
