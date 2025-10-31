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
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> xr.Dataset:
    """Handle variable extraction for xarray Dataset.

    Args:
        data: The xarray Dataset to extract variables from.
        variables: List of required variable names to extract.
        optional_variables: List of optional variable names to extract.
        optional_variables_mapping: Dictionary mapping optional variables to
            the required variables they replace.

    Returns:
        The dataset containing only the found variables.

    Raises:
        KeyError: If any required variables are missing from the dataset.
    """
    # Track which variables we've found
    found_variables = []
    required_variables_satisfied = set()

    # First, check for optional variables and add them if present
    for opt_var in optional_variables:
        if opt_var in data.data_vars:
            found_variables.append(opt_var)
            # Check if this optional variable replaces required variables
            if opt_var in optional_variables_mapping:
                replaced_vars = optional_variables_mapping[opt_var]
                # Handle both single string and list of strings
                if isinstance(replaced_vars, str):
                    required_variables_satisfied.add(replaced_vars)
                else:
                    required_variables_satisfied.update(replaced_vars)

    # Then check for required variables that weren't replaced
    missing_variables = []
    for var in variables:
        if var in required_variables_satisfied:
            # This required variable was replaced by an optional variable
            continue
        elif var in data.data_vars:
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
    import logging

    logger = logging.getLogger(__name__)

    logger.debug(
        "Checking xarray for time columns. Available coords: %s",
        list(data.coords.keys()),
    )
    logger.debug("Date range: %s to %s", start_date, end_date)

    # Convert the start and end dates to pandas Timestamp objects for xarray's
    # loc indexing
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Try different time dimension names
    time_dims = ["valid_time", "time", "init_time"]
    for time_dim in time_dims:
        if time_dim in data.coords:
            logger.debug("Found time coord: %s", time_dim)
            try:
                result = any(data[time_dim].loc[start_ts:end_ts])
                logger.debug("Time check result: %s", result)
                return result
            except Exception as e:
                logger.debug("Time check failed: %s", e)
                continue

    # If no time dimension found, return False
    logger.debug("No time dimension found in xarray dataset")
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
    import logging

    logger = logging.getLogger(__name__)

    # Check if Dataset has latitude and longitude dimensions
    lat_dims = ["latitude", "lat"]
    lon_dims = ["longitude", "lon"]

    lat_dim = utils.check_for_vars(lat_dims, list(data.coords.keys()))
    lon_dim = utils.check_for_vars(lon_dims, list(data.coords.keys()))

    logger.debug("Xarray spatial check - lat_dim: %s, lon_dim: %s", lat_dim, lon_dim)

    if lat_dim is None or lon_dim is None:
        logger.debug("No lat/lon dims found in xarray dataset")
        return False
    coords = location.as_geopandas().total_bounds
    # Get location bounds
    lat_min, lat_max = coords[1], coords[3]
    lon_min, lon_max = coords[0], coords[2]

    logger.debug(
        "Region bounds: lat [%s, %s], lon [%s, %s]", lat_min, lat_max, lon_min, lon_max
    )

    # Check if reversing the latitude range still returns no data
    if len(data[lat_dim].sel({lat_dim: slice(lat_min, lat_max)})) == 0:
        if len(data[lat_dim].sel({lat_dim: slice(lat_max, lat_min)})) == 0:
            # If reversing the latitude range still returns no data, return False
            logger.debug("No spatial data found (both lat orderings tried)")
            return False
        else:
            # If latitude has data, check longitude
            data = data[[lat_dim, lon_dim]].sel(
                {lat_dim: slice(lat_max, lat_min), lon_dim: slice(lon_min, lon_max)}
            )
    else:
        # Check longitude if latitude > 0
        data = data[[lat_dim, lon_dim]].sel(
            {lat_dim: slice(lat_min, lat_max), lon_dim: slice(lon_min, lon_max)}
        )

    # Check if any data remains after spatial filtering
    result = sum(data.sizes.values()) > 0
    logger.debug("Xarray spatial check result: %s", result)
    return result
