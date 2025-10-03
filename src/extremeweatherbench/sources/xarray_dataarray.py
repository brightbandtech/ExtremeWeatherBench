"""Handle variable extraction for xarray DataArrays."""

import datetime

import pandas as pd
import xarray as xr


def safely_pull_variables_xr_dataarray(
    dataset: xr.DataArray,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> xr.DataArray:
    """Handle variable extraction for xarray DataArray."""
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
    """Check if the dataset has valid times in the given date range."""
    # Convert the start and end dates to pandas Timestamp objects for xarray's
    # loc indexing
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return any(dataset["valid_time"].loc[start_ts:end_ts])
