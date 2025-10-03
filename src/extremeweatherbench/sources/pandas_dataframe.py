"""Handle variable extraction for Pandas DataFrames."""

import datetime

import pandas as pd


def safely_pull_variables_pandas_dataframe(
    dataset: pd.DataFrame,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Handle variable extraction for Pandas DataFrame."""
    # Get column names from DataFrame
    available_columns = list(dataset.columns)

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
            f"Required variables {missing_variables} not found in DataFrame. "
            f"Available columns: {available_columns}"
        )

    # Return DataFrame with only the found columns
    return dataset[found_variables]


def check_for_valid_times_pandas_dataframe(
    dataset: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime
) -> bool:
    """Check if the dataset has valid times in the given date range."""
    # Filter the DataFrame to only include valid times in the given date range
    time_filtered_df = dataset[
        (dataset["valid_time"] >= start_date) & (dataset["valid_time"] <= end_date)
    ]
    # If the filtered DataFrame has any rows, return True
    return len(time_filtered_df) > 0
