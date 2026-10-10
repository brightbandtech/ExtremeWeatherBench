"""Handle variable extraction for xarray DataArrays."""

import datetime

import xarray as xr

from extremeweatherbench import regions, utils
from extremeweatherbench.sources import xarray_dataset


def safely_pull_variables(
    data: xr.DataArray,
    variables: list[str],
) -> xr.DataArray:
    """Handle variable extraction for xarray DataArray.

    Args:
        data: The xarray DataArray to extract variables from.
        variables: List of required variable names to extract.

    Returns:
        The DataArray if it matches one of the requested variables.

    Raises:
        KeyError: If the DataArray name doesn't match any requested variables.
    """
    # For DataArray, the variable is the DataArray itself
    # Check if the requested variable matches the DataArray name
    dataarray_name = data.name or "unnamed"

    # Check if any of the requested variables match this DataArray
    if dataarray_name in variables:
        return data
    else:
        available_vars = [dataarray_name]
        raise KeyError(
            f"Required variables {variables} not found in DataArray. "
            f"Available variable: {available_vars}"
        )


def check_for_valid_times(
    data: xr.DataArray,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> bool:
    """Check if the DataArray has any times in the given date range.

    Same rules as ``xarray_dataset.check_for_valid_times``.
    """
    return xarray_dataset.check_for_valid_times(data, start_date, end_date)


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
