"""Handle variable extraction for xarray DataArrays."""

import xarray as xr


def safely_pull_variables_xr_dataarray(
    dataset: xr.DataArray,
    variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
) -> xr.DataArray:
    """Handle variable extraction for xarray DataArray."""
    # For DataArray, the variable is the DataArray itself
    # Check if the requested variable matches the DataArray name
    dataarray_name = dataset.name or "unnamed"
    if len(alternative_variables) > 1 and dataarray_name not in variables:
        raise KeyError(
            f"Required variables {variables} not found in DataArray. "
            f"Available variable: {dataarray_name}"
        )
    elif len(alternative_variables) == 1:
        if dataarray_name not in alternative_variables.items():
            raise KeyError(
                f"Required variables {variables} not found in DataArray. "
                f"Available variable: {dataarray_name}"
            )
        else:
            return dataset
