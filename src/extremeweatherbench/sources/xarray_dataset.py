"""Handle variable extraction for xarray Datasets."""

import datetime
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from extremeweatherbench import utils

if TYPE_CHECKING:
    from extremeweatherbench import regions


def safely_pull_variables(
    data: xr.Dataset,
    variables: list[str],
) -> xr.Dataset:
    """Handle variable extraction for xarray Dataset.

    Args:
        data: The xarray Dataset to extract variables from.
        variables: List of required variable names to extract.

    Returns:
        The dataset containing only the found variables.

    Raises:
        KeyError: If any required variables are missing from the dataset.
    """
    # Track which variables we've found
    found_variables = []

    # Then check for required variables that weren't replaced
    missing_variables = []
    for var in variables:
        if var in data.data_vars:
            found_variables.append(var)
        else:
            missing_variables.append(var)

    # Raise error if any required variables are missing
    if missing_variables:
        available_vars = list(data.data_vars.keys())
        raise KeyError(
            f"Required variables {missing_variables} not found in dataset. "
            f"Available variables: {available_vars}"
        )

    # Return dataset with only the found variables
    return data[found_variables]


def check_for_valid_times(
    data: xr.Dataset,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> bool:
    """Check if the dataset has valid times in the given date range.

    Args:
        data: The xarray Dataset to check for valid times.
        start_date: The start date of the time range to check.
        end_date: The end date of the time range to check.

    Returns:
        True if the dataset has any times within the specified range,
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
            try:
                time_slice = data[time_dim].sel({time_dim: slice(start_ts, end_ts)})
                return len(time_slice) > 0

            # If time dim is not found, check rest of time dims just in case
            except (KeyError, ValueError):
                continue

    # If no time dimension found, return False
    return False


def check_for_spatial_data(data: xr.Dataset, location: "regions.Region") -> bool:
    """Check if the Dataset has spatial data for the given location.

    Args:
        data: The xarray Dataset to check for spatial data.
        location: The region to check for spatial overlap.

    Returns:
        True if the Dataset has any data within the specified region,
        False otherwise.
    """
    # Check if Dataset has latitude and longitude dimensions
    lat_dims = ["latitude", "lat"]
    lon_dims = ["longitude", "lon"]

    lat_dim = utils.check_for_vars(lat_dims, list(data.coords.keys()))
    lon_dim = utils.check_for_vars(lon_dims, list(data.coords.keys()))

    if lat_dim is None or lon_dim is None:
        return False
    lat = data[lat_dim].values
    lon = data[lon_dim].values % 360.0
    # explode() splits antimeridian MultiPolygons into non-wrapping lobes,
    # since their combined total_bounds would span the full globe.
    bounds = location.as_geopandas().geometry.explode(index_parts=False).bounds
    for lon_min, lat_min, lon_max, lat_max in bounds.itertuples(index=False):
        if not ((lat >= lat_min) & (lat <= lat_max)).any():
            continue
        if lon_max - lon_min >= 360.0:
            return True
        lon_min, lon_max = lon_min % 360.0, lon_max % 360.0
        lon_hit = (
            (lon >= lon_min) & (lon <= lon_max)
            if lon_min <= lon_max
            else (lon >= lon_min) | (lon <= lon_max)
        )
        if lon_hit.any():
            return True
    return False
