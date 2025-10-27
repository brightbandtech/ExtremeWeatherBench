"""Handle variable extraction for Pandas DataFrames."""

import datetime
from typing import TYPE_CHECKING

import pandas as pd

from extremeweatherbench import defaults, utils

if TYPE_CHECKING:
    from extremeweatherbench import regions


def safely_pull_variables(
    data: pd.DataFrame,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Safely extract variables from a Pandas DataFrame with optional replacements.

    This function handles variable extraction from a Pandas DataFrame, supporting
    both required and optional variables. Optional variables can replace required
    ones based on the provided mapping, allowing for flexible data processing.

    Args:
        data: The pandas DataFrame to extract variables from.
        variables: List of required variable names to extract. These must be
            present in the DataFrame unless replaced by optional variables.
        optional_variables: List of optional variable names to extract. These
            are only included if present in the DataFrame.
        optional_variables_mapping: Dictionary mapping optional variable names
            to the required variables they can replace. Keys are optional
            variable names, values can be a single string or list of strings
            representing the required variables to replace.

    Returns:
        pd.DataFrame: A new DataFrame containing only the found columns.
            Includes all found required variables and optional variables.

    Raises:
        KeyError: If any required variables are missing from the DataFrame
            and not replaced by optional variables.

    Example:
        >>> df = pd.DataFrame({"temp": [20, 25], "pressure": [1013, 1015]})
        >>> variables = ["temperature"]
        >>> optional = ["temp"]
        >>> mapping = {"temp": "temperature"}
        >>> result = safely_pull_variables(df, variables, optional, mapping)
        >>> list(result.columns)
        ['temp']
    """
    # For pandas DataFrames, automatically add coordinate variables
    # to both required and optional variables
    variables = variables + defaults.DEFAULT_COORDINATE_VARIABLES
    optional_variables = optional_variables + defaults.DEFAULT_COORDINATE_VARIABLES

    # Get column names from DataFrame
    available_columns = list(data.columns)

    # Track which variables we've found (use set to avoid duplicates)
    found_variables = set()
    required_variables_satisfied = set()

    # First, check for optional variables and add them if present
    for opt_var in optional_variables:
        if opt_var in available_columns:
            found_variables.add(opt_var)
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
            found_variables.add(var)
        else:
            missing_variables.append(var)

    # Raise error if any required variables are missing
    if missing_variables:
        raise KeyError(
            f"Required variables {missing_variables} not found in DataFrame. "
            f"Available columns: {available_columns}"
        )

    # Return DataFrame with only the found columns
    return data[list(found_variables)]


def check_for_valid_times(
    data: pd.DataFrame,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> bool:
    """Check if the DataFrame contains any data within the specified time range.

    This function searches for time columns in the DataFrame using common
    naming conventions and checks if any data points fall within the given
    date range. It tries multiple possible column names for time data.

    Args:
        data: The pandas DataFrame to check for valid times. Should contain
            at least one time column with datetime data.
        start_date: The start date of the time range to check (inclusive).
        end_date: The end date of the time range to check (inclusive).

    Returns:
        bool: True if the DataFrame has any times within the specified range,
            False if no time column is found or no data falls within the range.

    Note:
        The function searches for time columns with names: 'valid_time',
        'time', or 'init_time'. The first matching column is used for
        the time range check.

    Example:
        >>> df = pd.DataFrame(
        ...     {
        ...         "valid_time": pd.date_range("2023-01-01", periods=5),
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> start = datetime.datetime(2023, 1, 2)
        >>> end = datetime.datetime(2023, 1, 4)
        >>> check_for_valid_times(df, start, end)
        True
    """
    # Try different time column names
    time_cols = ["valid_time", "time", "init_time"]
    for time_col in time_cols:
        if time_col in data.columns:
            # Filter the DataFrame to only include valid times in the given date
            # range
            time_filtered_df = data[
                (data[time_col] >= start_date) & (data[time_col] <= end_date)
            ]
            # If the filtered DataFrame has any rows, return True
            return len(time_filtered_df) > 0

    # If no time column found, return False
    return False


def check_for_spatial_data(data: pd.DataFrame, location: "regions.Region") -> bool:
    """Check if the DataFrame contains spatial data within the specified region.

    This function verifies that the DataFrame has latitude and longitude columns
    and contains at least one data point that falls within the bounding box
    of the specified geographic region.

    Args:
        data: The pandas DataFrame to check for spatial data. Should contain
            latitude and longitude columns with coordinate data.
        location: A Region object that defines the geographic area to check
            against.
    Returns:
        bool: True if the DataFrame has spatial data within the region bounds,
            False if no coordinate columns are found or no data points fall
            within the region.

    Note:
        The function searches for coordinate columns with names: 'latitude'
        or 'lat' for latitude, and 'longitude' or 'lon' for longitude.
        The first matching column names are used for the spatial check.

    Example:
        >>> import pandas as pd
        >>> from extremeweatherbench.regions import Region
        >>> df = pd.DataFrame(
        ...     {
        ...         "latitude": [40.0, 41.0, 42.0],
        ...         "longitude": [-74.0, -73.0, -72.0],
        ...         "value": [1, 2, 3],
        ...     }
        ... )
        >>> region = Region(...)  # Define your region
        >>> check_for_spatial_data(df, region)
        True
    """
    # Check if DataFrame has latitude and longitude columns
    lat_cols = ["latitude", "lat"]
    lon_cols = ["longitude", "lon"]

    lat_col = utils.check_for_vars(lat_cols, data.columns)
    lon_col = utils.check_for_vars(lon_cols, data.columns)

    if lat_col is None or lon_col is None:
        return False
    coords = location.as_geopandas().total_bounds
    # Get location bounds
    lat_min, lat_max = coords[1], coords[3]
    lon_min, lon_max = coords[0], coords[2]

    # Check if there are any data points within the location bounds
    filtered_data = data[
        (data[lat_col] >= lat_min)
        & (data[lat_col] <= lat_max)
        & (data[lon_col] >= lon_min)
        & (data[lon_col] <= lon_max)
    ]

    return len(filtered_data) > 0
