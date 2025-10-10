"""Handle variable extraction for xarray DataArrays."""

import datetime

import pandas as pd
import xarray as xr

from extremeweatherbench import regions, utils


def safely_pull_variables_xr_dataarray(
    dataset: xr.DataArray,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> xr.DataArray:
    """Handle variable extraction for xarray DataArray.
    
    Args:
        dataset: The xarray DataArray to extract variables from.
        variables: List of required variable names to extract.
        optional_variables: List of optional variable names to extract.
        optional_variables_mapping: Dictionary mapping optional variables to
            the required variables they replace.
    
    Returns:
        The DataArray if it matches one of the requested variables.
        
    Raises:
        KeyError: If the DataArray name doesn't match any requested variables.
    """
    # For DataArray, the variable is the DataArray itself
    # Check if the requested variable matches the DataArray name
    dataarray_name = dataset.name or "unnamed"

    # Check if any of the requested variables match this DataArray
    if (
        dataarray_name in variables
        or dataarray_name in optional_variables
        or any(
            dataarray_name in variables
            for variables in optional_variables_mapping.values()
        )
    ):
        return dataset
    else:
        available_vars = [dataarray_name]
        raise KeyError(
            f"Required variables {variables} not found in DataArray. "
            f"Available variable: {available_vars}"
        )


def check_for_valid_times_xr_dataarray(
    dataset: xr.DataArray, start_date: datetime.datetime, end_date: datetime.datetime
) -> bool:
    """Check if the DataArray has valid times in the given date range.
    
    Args:
        dataset: The xarray DataArray to check for valid times.
        start_date: The start date of the time range to check.
        end_date: The end date of the time range to check.
    
    Returns:
        True if the DataArray has any times within the specified range,
        False otherwise.
    """
    # Convert the start and end dates to pandas Timestamp objects for xarray's
    # loc indexing
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Try different time dimension names
    time_dims = ["valid_time", "time", "init_time"]
    for time_dim in time_dims:
        if time_dim in dataset.coords:
            return any(dataset[time_dim].loc[start_ts:end_ts])
    
    # If no time dimension found, return False
    return False

def check_for_spatial_data_xr_dataarray(
    dataset: xr.DataArray,
    location: "regions.Region",
) -> bool:
    """Check if the DataArray has spatial data for the given location.
    
    Args:
        dataset: The xarray DataArray to check for spatial data.
        location: The region to check for spatial overlap.
    
    Returns:
        True if the DataArray has any data within the specified region,
        False otherwise.
    """
    # Check if DataArray has latitude and longitude dimensions
    lat_dims = ["latitude", "lat"]
    lon_dims = ["longitude", "lon"]
    
    lat_dim = utils.check_for_vars(lat_dims, dataset.dims)
    lon_dim = utils.check_for_vars(lon_dims, dataset.dims)
    
    if lat_dim is None or lon_dim is None:
        return False

    coords = location.get_bounding_coordinates
    # Get location bounds
    lat_min, lat_max = coords.latitude_min, coords.latitude_max
    lon_min, lon_max = coords.longitude_min, coords.longitude_max

    # Check if reversing the latitude range still returns no data
    if len(dataset.sel({lat_dim: slice(lat_min, lat_max)})) == 0:
        if len(dataset.sel({lat_dim: slice(lat_max, lat_min)})) == 0:
            # If reversing the latitude range still returns no data, return False
            return False
        else:
            # If latitude has data, check longitude
            dataset = dataset.sel({lat_dim: slice(lat_max, lat_min), lon_dim: slice(lon_min, lon_max)})
    else:
        # Check longitude if latitude > 0
        dataset = dataset.sel({lat_dim: slice(lat_min, lat_max), lon_dim: slice(lon_min, lon_max)})
    
    # Check if any data remains after spatial filtering
    return sum(dataset.sizes.values()) > 0
