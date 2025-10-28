"""Utility functions for variable extraction and manipulation."""

from typing import Optional, Union

import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import sources

IncomingDataInput = Union[xr.Dataset, xr.DataArray, pl.LazyFrame, pd.DataFrame]


def safely_pull_variables(
    data: IncomingDataInput,
    required_variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
    source_module: Optional["sources.base.Source"] = None,
) -> IncomingDataInput:
    """Safely extract variables from data with alternative and optional support.

    This is a convenience wrapper around the source module's safely_pull_variables
    function that automatically detects the appropriate source module.

    Args:
        data: The data to extract variables from (xr.Dataset, xr.DataArray,
            pl.LazyFrame, or pd.DataFrame).
        required_variables: List of required variable names to extract.
        alternative_variables: Dictionary mapping required variable names to
            lists of alternative variables that can replace them.
        optional_variables: List of optional variable names to extract.
        source_module: Optional pre-created source module. If None, creates one.

    Returns:
        The data subset to only the specified variables.

    Raises:
        KeyError: If any required variables are missing and no alternatives.
    """
    # Import here to avoid circular dependency
    from extremeweatherbench import sources

    # Use provided source module or get one
    if source_module is None:
        source_module = sources.get_backend_module(type(data))

    return source_module.safely_pull_variables(
        data,
        required_variables,
        alternative_variables,
        optional_variables,
    )
