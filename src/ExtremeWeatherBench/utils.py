"""
Utility functions for other files that can apply to multiple files in the library
"""

import numpy as np
import pandas as pd
import ujson
from kerchunk.hdf import SingleHdf5ToZarr 
import typing as t
import xarray as xr
import rioxarray
from shapely.geometry import box
import geopandas as gpd
import regionmask

def convert_longitude_to_360(longitude):
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

# seasonal aggregation functions for max, min, and mean
def seasonal_subset_max(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.max()

def seasonal_subset_min(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.min()

def seasonal_subset_mean(df):
    df = df.where(df.index.month.isin([6,7,8]))
    return df.mean()

def is_jja(month):
    return (month >= 6) & (month <= 8)

def is_6_hourly(hour):
    return (hour == 0) | (hour == 6) | (hour == 12) | (hour == 18)


def clip_dataset_to_square(dataset: xr.Dataset, location_center: dict, length_km: float) -> xr.Dataset:
    """
    Clip an xarray dataset to a square around a given location with a specified side length.

    Parameters:
    dataset (xarray.Dataset): The input xarray dataset.
    location_center (dict): A dictionary with 'latitude' and 'longitude' keys.
    length_km (float): The side length of the square in kilometers.

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

