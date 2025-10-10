"""Input source modules for different data types."""

from .pandas_dataframe import (
    check_for_valid_times_pandas_dataframe,
    safely_pull_variables_pandas_dataframe,
    check_for_spatial_data_pandas_dataframe,
)
from .polars_lazyframe import (
    check_for_valid_times_polars_lazyframe,
    safely_pull_variables_polars_lazyframe,
    check_for_spatial_data_polars_lazyframe,
)
from .xarray_dataarray import (
    check_for_valid_times_xr_dataarray,
    safely_pull_variables_xr_dataarray,
    check_for_spatial_data_xr_dataarray,
)
from .xarray_dataset import (
    check_for_valid_times_xr_dataset,
    safely_pull_variables_xr_dataset,
    check_for_spatial_data_xr_dataset,
)

__all__ = [
    "safely_pull_variables_xr_dataset",
    "safely_pull_variables_xr_dataarray",
    "safely_pull_variables_polars_lazyframe",
    "safely_pull_variables_pandas_dataframe",
    "check_for_valid_times_xr_dataset",
    "check_for_valid_times_xr_dataarray",
    "check_for_valid_times_polars_lazyframe",
    "check_for_valid_times_pandas_dataframe",
    "check_for_spatial_data_pandas_dataframe",
    "check_for_spatial_data_polars_lazyframe",
    "check_for_spatial_data_xr_dataarray",
    "check_for_spatial_data_xr_dataset",
]
