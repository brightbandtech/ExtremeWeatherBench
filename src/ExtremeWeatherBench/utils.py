"""
Utility functions for other files that can apply to multiple files in the library
"""

import numpy as np
import pandas as pd
import ujson
from collections import namedtuple
from kerchunk.hdf import SingleHdf5ToZarr 
import typing as t
import xarray as xr
import rioxarray
from shapely.geometry import box
import geopandas as gpd
import regionmask
from . import config

def convert_longitude_to_360(longitude: float) -> float:
    return longitude % 360

def convert_longitude_to_180(dataset: t.Union[xr.Dataset, xr.DataArray], longitude_name: str='longitude') -> xr.Dataset:
    dataset.coords[longitude_name] = (dataset.coords[longitude_name] + 180) % 360 - 180
    dataset = dataset.sortby(longitude_name)
    return dataset

def generate_json_from_nc(u,so,fs,fs_out,json_dir):
    with fs.open(u, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, u, inline_threshold=300)

        file_split = u.split('/') # seperate file path to create a unique name for each json 
        model = file_split[1].split('_')[0]
        date_string = file_split[-1].split('_')[3]
        outf = f'{json_dir}{model}_{date_string}_.json'
        print(outf)
        with fs_out.open(outf, 'wb') as f:
            f.write(ujson.dumps(h5chunks.translate()).encode());

def is_jja(month):
    return (month >= 6) & (month <= 8)

def is_6_hourly(hour):
    return (hour == 0) | (hour == 6) | (hour == 12) | (hour == 18)


def clip_dataset_to_bounding_box(dataset: xr.Dataset, location_center: namedtuple, length_km: float) -> xr.Dataset:
    """
    Clip an xarray dataset to a bounding_box around a given location with a specified side length.

    Parameters:
    dataset (xarray.Dataset): The input xarray dataset.
    location_center (dict): A dictionary with 'latitude' and 'longitude' keys.
    length_km (float): The side length of the bounding_box in kilometers.

    Returns:
    xarray.Dataset: The clipped xarray dataset.
    """
    lat_center = location_center['latitude']
    lon_center = location_center['longitude']
    
    # Convert length from kilometers to degrees (approximation)
    length_deg = length_km / 111  # 1 degree is approximately 111 km
    
    # Create a bounding box
    min_lat = lat_center - (length_deg / 2)
    max_lat = lat_center + (length_deg / 2)
    min_lon = lon_center - (length_deg / 2)
    max_lon = lon_center + (length_deg / 2)
    
    # Create a GeoDataFrame with the bounding box
    bbox = gpd.GeoDataFrame(
        {'geometry': [box(min_lon, min_lat, max_lon, max_lat)]},
    )
    # Clip the dataset using the bounding box
    clipped_dataset = dataset.rio.write_crs('EPSG:4326').rio.clip(bbox.geometry, bbox.crs, drop=True)
    
    return clipped_dataset

def convert_day_yearofday_to_time(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """
    Convert dayofyear and hour coordinates to a new time coordinate.

    Parameters:
    dataset (xarray.Dataset): The input xarray dataset.
    year (int): The year to use for the time coordinate.

    Returns:
    xarray.Dataset: The dataset with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f'{year}-01-01', 
        periods=len(dataset['dayofyear']) * len(dataset['hour']), 
        freq='6h'
    )
    dataset = dataset.stack(time=('dayofyear', 'hour')).drop(['dayofyear', 'hour'])
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(time=time_dim)
    
    return dataset


def remove_ocean_gridpoints(dataset: xr.Dataset) -> xr.Dataset:
    """
    Subset a dataset to only include land gridpoints based on a land-sea mask.

    Parameters:
    dataset (xarray.Dataset): The input xarray dataset.
    land_sea_mask (xarray.DataArray): A land-sea mask with land gridpoints set to 1 and sea gridpoints set to 0.

    Returns:
    xarray.Dataset: The dataset with only land gridpoints.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_sea_mask = land.mask(dataset.longitude, dataset.latitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    dataset = dataset.where(land_mask, drop=True)
    
    return dataset


def _open_obs_datasets(eval_config: config.Config):
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path is not None:
        point_obs = pd.read_parquet(eval_config.point_obs_path, chunks='auto')
    if eval_config.gridded_obs_path is not None:
        gridded_obs = xr.open_zarr(eval_config.gridded_obs_path, chunks='auto')
    if point_obs is None and gridded_obs is None:
        raise ValueError("No grided or point observation data provided.")
    return point_obs, gridded_obs

#TODO simplify to one paradigm, don't use nc, zarr, AND json
def _open_forecast_dataset(eval_config: config.Config, forecast_schema_config: Optional[config.ForecastSchemaConfig] = None):
    logging.info("Opening forecast dataset")
    if eval_config.forecast_path.startswith("s3://"):
        fs = fsspec.filesystem('s3')
    elif eval_config.forecast_path.startswith("gcs://") or eval_config.forecast_path.startswith("gs://"):
        fs = fsspec.filesystem('gcs')
    else:
        fs = fsspec.filesystem('file')

    file_list = fs.ls(eval_config.forecast_path)
    file_types = set([file.split('.')[-1] for file in file_list])
    if len(file_types) > 1:
        raise ValueError("Multiple file types found in forecast path.")
    
    if 'zarr' in file_types and len(file_list) == 1:
        forecast_dataset = xr.open_zarr(file_list, chunks='auto')
    elif 'zarr' in file_types and len(file_list) > 1:
        raise ValueError("Multiple zarr files found in forecast path, please provide a single zarr file.")
    
    if 'nc' in file_types:
        logging.warning("NetCDF files are not recommended for large datasets. Consider converting to zarr.")
        forecast_dataset = xr.open_mfdataset(file_list, chunks='auto')

    if 'json' in file_types:
        forecast_dataset = _open_kerchunk_zarr_reference_jsons(file_list, forecast_schema_config)

    return forecast_dataset

def _open_kerchunk_zarr_reference_jsons(file_list, forecast_schema_config):
    xarray_datasets = []
    for json_file in file_list:
        fs_ = fsspec.filesystem("reference", fo=json_file, ref_storage_args={'skip_instance_cache':True},
                        remote_protocol='gcs', remote_options={'anon':True})
        m = fs_.get_mapper("")
        ds = xr.open_dataset(m, engine="zarr", backend_kwargs={'consolidated':False}, chunks='auto')
        if 'initialization_time' not in ds.attrs:
            raise ValueError("Initialization time not found in dataset attributes. \
                             Please add initialization_time to the dataset attributes.")
        else:
            model_run_time = np.datetime64(pd.to_datetime(ds.attrs['initialization_time']))
        ds[forecast_schema_config.init_time] = model_run_time.astype('datetime64[ns]')
        fhours = ds[forecast_schema_config.time] - model_run_time
        fhours = fhours.values / np.timedelta64(1, 'h')
        ds[forecast_schema_config.fhour] = fhours
        ds = ds.set_coords(forecast_schema_config.init_time)
        ds = ds.expand_dims(forecast_schema_config.init_time)
        for data_vars in ds.data_vars:
            if forecast_schema_config.time in ds[data_vars].dims:
                ds[data_vars] = ds[data_vars].swap_dims({forecast_schema_config.time:forecast_schema_config.fhour})
        ds = ds.transpose(forecast_schema_config.init_time, 
                          forecast_schema_config.time, 
                          forecast_schema_config.fhour, 
                          forecast_schema_config.latitude, 
                          forecast_schema_config.longitude, 
                          forecast_schema_config.level)
        xarray_datasets.append(ds)

    return xr.concat(xarray_datasets, dim=forecast_schema_config.init_time)