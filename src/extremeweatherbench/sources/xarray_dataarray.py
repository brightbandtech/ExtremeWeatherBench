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


def safe_concat(data_objects: list[xr.DataArray]) -> xr.Dataset:
    """Safely concatenate DataArrays, filtering out empty ones.

    This function prevents FutureWarnings from pd.concat when dealing with
    empty or all-NA DataFrames by filtering them out before concatenation.
    It handles dtype mismatches by converting to object dtype only when
    necessary to prevent concatenation warnings.

    Args:
        data_objects: List of DataArrays to concatenate
        ignore_index: Whether to ignore index during concatenation

    Returns:
        Concatenated DataArray, or empty DataArray with OUTPUT_SCHEMA if all input
        DataArrays are empty. Preserves original dtypes when consistent across
        DataArrays.
    """
    # Filter out problematic DataArrays that would trigger FutureWarning
    valid_data: list[xr.DataArray] = []

    for i, data in enumerate(data_objects):
        # Skip empty DataFrames or DataArrays
        if data.empty:
            logger.debug("Skipping empty data %s", i)
            continue
        # Skip DataFrames where all values are NA
        if data.isna().all().all():
            logger.debug("Skipping all-NA data %s", i)
            continue

        # Skip DataFrames where all columns are empty/NA
        if len(data.coords) > 0 and all(data[col].isna().all() for col in data.coords):
            logger.debug("Skipping data %s with all-NA columns", i)
            continue

        valid_data.append(data)

        return xr.concat(
            objs=valid_data,
            dim="value",
            coordinates="minimal",
            data_vars="minimal",
        )

    # If all input DataArrays are empty, return an empty Dataset
    return xr.Dataset()


def ensure_output_schema(data: xr.DataArray, **metadata) -> xr.DataArray:
    """Ensure data conforms to OUTPUT_SCHEMA schema.

    This function adds any provided metadata columns to the data and validates
    that all OUTPUT_SCHEMA are present. Any missing columns will be filled with NaN
    and a warning will be logged.

    Args:
        data: Base data, typically with 'value' name from metric result DataArray
        **metadata: Key-value pairs for metadata columns (e.g., target_variable='temp')

    Returns:
        DataArray with coordinates matching OUTPUT_SCHEMA specification.

    Example:
        data = ensure_output_schema(
            metric_result,
            target_variable=target_var,
            metric=metric.name,
            case_id_number=case_id,
            event_type=event_type
        )
    """
    # Add metadata columns
    for col, value in metadata.items():
        data[col] = value

    incoming_schema = list(data.coords)
    # Check for missing columns and warn
    missing_cols = set(utils.OUTPUT_SCHEMA) - set(incoming_schema)

    # An output requires one of init_time or lead_time. If aggregating over one or the
    # other, it is expected that one will be missing. init_time will be present for a
    # metric that assesses something in an entire model run, such as the onset error of
    # an event. Lead_time will be present for a metric that assesses something at a
    # specific forecast hour, such as RMSE. If neither are present, the output is
    # invalid.
    init_time_missing = "init_time" in missing_cols
    lead_time_missing = "lead_time" in missing_cols

    # Check if exactly one of init_time or lead_time is missing
    if init_time_missing != lead_time_missing:
        missing_cols.discard("init_time")
        missing_cols.discard("lead_time")

    if missing_cols:
        logger.warning("Missing expected columns: %s.", missing_cols)

    # Ensure all OUTPUT_SCHEMA are present (missing ones will be NaN)
    # and reorder to match OUTPUT_SCHEMA specification
    return data.reindex(coords=utils.OUTPUT_SCHEMA)
