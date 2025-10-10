"""Handle variable extraction for Pandas DataFrames."""

import datetime
from typing import TYPE_CHECKING
import pandas as pd
from extremeweatherbench import utils

if TYPE_CHECKING:
    from extremeweatherbench import regions


def safely_pull_variables_pandas_dataframe(
    dataset: pd.DataFrame,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Handle variable extraction for Pandas DataFrame.

    Args:
        dataset: The pandas DataFrame to extract variables from.
        variables: List of required variable names to extract.
        optional_variables: List of optional variable names to extract.
        optional_variables_mapping: Dictionary mapping optional variables to
            the required variables they replace.

    Returns:
        The DataFrame containing only the found columns.

    Raises:
        KeyError: If any required variables are missing from the DataFrame.
    """
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
    """Check if the DataFrame has valid times in the given date range.

    Args:
        dataset: The pandas DataFrame to check for valid times.
        start_date: The start date of the time range to check.
        end_date: The end date of the time range to check.

    Returns:
        True if the DataFrame has any times within the specified range,
        False otherwise.
    """
    # Try different time column names
    time_cols = ["valid_time", "time", "init_time"]
    for time_col in time_cols:
        if time_col in dataset.columns:
            # Filter the DataFrame to only include valid times in the given date range
            time_filtered_df = dataset[
                (dataset[time_col] >= start_date) & (dataset[time_col] <= end_date)
            ]
            # If the filtered DataFrame has any rows, return True
            return len(time_filtered_df) > 0

    # If no time column found, return False
    return False


def check_for_spatial_data_pandas_dataframe(
    dataset: pd.DataFrame,
    location: "regions.Region",
) -> bool:
    """Check if the DataFrame has spatial data for the given location."""
    # Check if DataFrame has latitude and longitude columns
    lat_cols = ["latitude", "lat"]
    lon_cols = ["longitude", "lon"]

    lat_col = utils.check_for_vars(lat_cols, dataset.columns)
    lon_col = utils.check_for_vars(lon_cols, dataset.columns)

    if lat_col is None or lon_col is None:
        return False
    coords = location.get_bounding_coordinates
    # Get location bounds
    lat_min, lat_max = coords.latitude_min, coords.latitude_max
    lon_min, lon_max = coords.longitude_min, coords.longitude_max

    # Check if there are any data points within the location bounds
    filtered_data = dataset[
        (dataset[lat_col] >= lat_min)
        & (dataset[lat_col] <= lat_max)
        & (dataset[lon_col] >= lon_min)
        & (dataset[lon_col] <= lon_max)
    ]

    return len(filtered_data) > 0
