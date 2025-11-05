"""Input source modules for different data types."""

import pandas as pd
import polars as pl
import xarray as xr

from . import pandas_dataframe, polars_lazyframe, xarray_dataarray, xarray_dataset
from .base import Source

# Registry mapping data types to their corresponding source modules
# Each module implements the Source Protocol at module level
DATA_BACKEND_REGISTRY: dict[type, Source] = {
    pd.DataFrame: pandas_dataframe,
    pl.LazyFrame: polars_lazyframe,
    xr.DataArray: xarray_dataarray,
    xr.Dataset: xarray_dataset,
}


def get_backend_module(data_type: type) -> Source:
    """Get the source module for a given data type.

    Args:
        data_type: The type of data (e.g., pd.DataFrame, xr.Dataset)

    Returns:
        The module that implements Source Protocol for this data type

    Raises:
        ValueError: If no source handler is registered for the data type
    """
    module = DATA_BACKEND_REGISTRY.get(data_type)
    if module is None:
        available_types = list(DATA_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"No source handler found for dataset type: {data_type}. "
            f"Available types: {available_types}"
        )

    return module
