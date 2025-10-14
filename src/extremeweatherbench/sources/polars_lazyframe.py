"""Handle variable extraction for Polars LazyFrames."""

import polars as pl


def safely_pull_variables_polars_lazyframe(
    dataset: pl.LazyFrame,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
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
    required_variables_satisfied = set()

    # First, check for optional variables and add them if present
    for opt_var in optional_variables:
        if opt_var in available_columns:
            found_variables.append(opt_var)
            # Check if this optional variable replaces required variables
            if opt_var in optional_variables_mapping:
                replaced_vars = optional_variables_mapping[opt_var]
                # Handle both single string and list of strings
                if isinstance(replaced_vars, str):
                    required_variables_satisfied.add(replaced_vars)
                else:
                    required_variables_satisfied.update(replaced_vars)

    # Then check for required variables that weren't replaced
    missing_variables = []
    for var in variables:
        if var in required_variables_satisfied:
            # This required variable was replaced by an optional variable
            continue
        elif var in available_columns:
            found_variables.append(var)
        else:
            missing_variables.append(var)

    # Raise error if any required variables are missing
    if missing_variables:
        raise KeyError(
            f"Required variables {missing_variables} not found in LazyFrame. "
            f"Available columns: {available_columns}"
        )

    # Return LazyFrame with only the found columns
    return dataset.select(found_variables)
