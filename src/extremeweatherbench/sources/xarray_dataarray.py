"""Handle variable extraction for xarray DataArrays."""

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
