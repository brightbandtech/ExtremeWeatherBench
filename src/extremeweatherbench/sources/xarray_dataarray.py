"""Handle variable extraction for xarray DataArrays."""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def safely_pull_variables_xr_dataarray(
    dataset: xr.DataArray,
    variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
) -> xr.DataArray:
    """Extract variables from an xarray DataArray by matching names.

    For DataArrays, checks if the DataArray name matches any of the
    requested variables (required or optional).

    Args:
        dataset: The DataArray to check.
        variables: List of required variable names.
        optional_variables: List of optional variable names.
        optional_variables_mapping: Dict mapping optional vars to required
            vars they can replace.

    Returns:
        The original DataArray if its name matches a requested variable.

    Raises:
        KeyError: If the DataArray name doesn't match any requested variable.
    """
    # For DataArray, the variable is the DataArray itself
    # Check if the requested variable matches the DataArray name
    dataarray_name = dataset.name or "unnamed"
    if len(variables) > 1:
        logger.warning(
            "Multiple variables provided for DataArray. Only the first one will be "
            "used."
        )
    if dataarray_name != variables[0]:
        logger.warning(
            f"Required variable {variables[0]} not found in DataArray. "
            f"Returning variable: {dataarray_name}"
        )
    if len(alternative_variables) > 1 and dataarray_name not in variables:
        raise KeyError(
            f"Required variables {variables} not found in DataArray. "
            f"Available variable: {dataarray_name}"
        )
    if len(optional_variables) > 0:
        logger.warning(
            f"Optional variables {optional_variables} provided for DataArray but will "
            "be ignored."
        )
    return dataset
