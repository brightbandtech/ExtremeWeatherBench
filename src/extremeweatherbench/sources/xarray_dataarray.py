"""Handle variable extraction for xarray DataArrays."""

import datetime
import logging

import pandas as pd
import xarray as xr

from extremeweatherbench import regions, utils

logger = logging.getLogger(__name__)


def safely_pull_variables(
    data: xr.DataArray,
    variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
) -> xr.DataArray:
    """Handle variable extraction for xarray DataArray.

    Args:
        data: The xarray DataArray to extract variables from.
        variables: List of required variable names to extract.
        alternative_variables: Dictionary mapping alternative variable names to
            the required variables they replace.
        optional_variables: List of optional variable names to extract.

    Returns:
        The DataArray if it matches one of the requested variables.

    Raises:
        KeyError: If the DataArray name doesn't match any requested variables.
    """
    # Handle None values for alternative_variables and optional_variables
    if alternative_variables is None:
        alternative_variables = {}
    if optional_variables is None:
        optional_variables = []

    # For DataArray, the variable is the DataArray itself
    # Check if the requested variable matches the DataArray name
    dataarray_name = data.name or "unnamed"
    if len(variables) > 1:
        logger.warning(
            "Multiple variables provided for DataArray. Only the first one will be "
            "used."
        )

    # Check if the DataArray name matches any required variable
    if dataarray_name in variables:
        # Found the required variable
        pass
    elif dataarray_name in optional_variables:
        # Found as optional variable
        pass
    elif variables[0] in alternative_variables:
        # Check if dataarray_name is in the alternatives
        if dataarray_name not in alternative_variables[variables[0]]:
            raise KeyError(
                f"Required variables {variables[0]} not found in DataArray. "
                f"Available variable: {dataarray_name}"
            )
    else:
        # Required variable not found and no alternatives
        raise KeyError(
            f"Required variables {variables[0]} not found in DataArray. "
            f"Available variable: {dataarray_name}"
        )

    if len(optional_variables) > 0 and dataarray_name not in optional_variables:
        logger.warning(
            f"Optional variables {optional_variables} provided for DataArray but will "
            "be ignored."
        )

    return data


def check_for_valid_times(
    data: xr.DataArray,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> bool:
    """Check if the DataArray has valid times in the given date range.

    Args:
        data: The xarray DataArray to check for valid times.
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
        if time_dim in data.coords:
            return any(data[time_dim].loc[start_ts:end_ts])

    # If no time dimension found, return False
    return False


def check_for_spatial_data(data: xr.DataArray, location: "regions.Region") -> bool:
    """Check if the DataArray has spatial data for the given location.

    Args:
        data: The xarray DataArray to check for spatial data.
        location: The region to check for spatial overlap.

    Returns:
        True if the DataArray has any data within the specified region,
        False otherwise.
    """
    # Check if DataArray has latitude and longitude dimensions
    lat_dims = ["latitude", "lat"]
    lon_dims = ["longitude", "lon"]

    lat_dim = utils.check_for_vars(lat_dims, data.dims)
    lon_dim = utils.check_for_vars(lon_dims, data.dims)

    if lat_dim is None or lon_dim is None:
        return False

    coords = location.as_geopandas().total_bounds
    # Get location bounds
    lat_min, lat_max = coords[1], coords[3]
    lon_min, lon_max = coords[0], coords[2]

    # Check if reversing the latitude range still returns no data
    if len(data.sel({lat_dim: slice(lat_min, lat_max)})) == 0:
        if len(data.sel({lat_dim: slice(lat_max, lat_min)})) == 0:
            # If reversing the latitude range still returns no data, return False
            return False
        else:
            # If latitude has data, check longitude
            data = data.sel(
                {lat_dim: slice(lat_max, lat_min), lon_dim: slice(lon_min, lon_max)}
            )
    else:
        # Check longitude if latitude > 0
        data = data.sel(
            {lat_dim: slice(lat_min, lat_max), lon_dim: slice(lon_min, lon_max)}
        )

    # Check if any data remains after spatial filtering
    return sum(data.sizes.values()) > 0
