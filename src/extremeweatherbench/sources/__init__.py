"""Input source modules for different data types."""

from .pandas_dataframe import safely_pull_variables_pandas_dataframe
from .polars_lazyframe import safely_pull_variables_polars_lazyframe
from .xarray_dataarray import safely_pull_variables_xr_dataarray
from .xarray_dataset import safely_pull_variables_xr_dataset

__all__ = [
    "safely_pull_variables_xr_dataset",
    "safely_pull_variables_xr_dataarray",
    "safely_pull_variables_polars_lazyframe",
    "safely_pull_variables_pandas_dataframe",
]
