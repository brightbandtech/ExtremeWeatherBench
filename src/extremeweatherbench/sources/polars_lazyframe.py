"""Handle variable extraction for Polars LazyFrames."""

import polars as pl


def safely_pull_variables_polars_lazyframe(
    dataset: pl.LazyFrame,
    variables: list[str],
    alternative_variables: dict[str, list[str]],
    optional_variables: list[str],
) -> pl.LazyFrame:
    """Extract variables from a Polars LazyFrame, prioritizing optionals.

    Args:
        dataset: The LazyFrame to extract variables from.
        variables: List of required variable names to extract.
        optional_variables: List of optional variable names to try first.
        optional_variables_mapping: Dict mapping optional vars to list of
            required vars they can replace.

    Returns:
        LazyFrame containing only the extracted variables.

    Raises:
        KeyError: If required variables are missing and no suitable optional
            variables are available as replacements.
    """
    # Get column names from LazyFrame
    available_columns = dataset.collect_schema().names()

    # Track which variables we've found
    found_variables = []

    # First, check for required variables and add them if present
    found_variables = [var for var in variables if var in available_columns]

    # Then, check for optional variables and add them if present
    for opt_var in optional_variables:
        if opt_var in available_columns:
            found_variables.append(opt_var)

    # Now, check for alternative variables and add them if present and required
    # variables are not found
    for req_var in alternative_variables:
        # If the required variable is not found, check if all of its alternatives are
        if req_var not in found_variables:
            if all(
                alt_var in available_columns
                for alt_var in alternative_variables[req_var]
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
    return dataset.select(list(set(found_variables)))
