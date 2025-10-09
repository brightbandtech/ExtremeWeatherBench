"""Handle variable extraction for xarray Datasets."""

import xarray as xr


def safely_pull_variables_xr_dataset(
    dataset: xr.Dataset,
    variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
) -> xr.Dataset:
    """Handle variable extraction for xarray Dataset."""
    # Track which variables we've found
    found_variables = []

    # First, check for required variables and add them if present
    for vars in variables:
        if vars in dataset.data_vars:
            found_variables.append(vars)

    # Then, check for optional variables and add them if present
    for opt_var in optional_variables:
        if opt_var in dataset.data_vars:
            found_variables.append(opt_var)
    

    # Now, check for alternative variables and add them if present
    for req_var in alternative_variables:
        # If the required variable is not found, check if all of its alternatives are
        if req_var not in found_variables:
            if all(
                alt_var in dataset.data_vars for alt_var in alternative_variables[req_var]
                ):
                # If all of the alternatives are found, add them to the found variables
                found_variables += alternative_variables[req_var]
            else:
                # If not, raise an error
                raise KeyError(
                    f"Required variables {req_var} nor any of their alternatives "
                    f"{alternative_variables[req_var]} found in dataset. "
                )

    # Return dataset with unique found variables
    return dataset[list(set(found_variables))]
