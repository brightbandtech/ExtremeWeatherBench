"""Handle variable extraction for Polars LazyFrames."""

import datetime
from typing import TYPE_CHECKING
import polars as pl
from extremeweatherbench import utils
if TYPE_CHECKING:
    from extremeweatherbench import regions

def safely_pull_variables_polars_lazyframe(
    dataset: pl.LazyFrame,
    variables: list[str],
    optional_variables: list[str],
    optional_variables_mapping: dict[str, list[str]],
) -> pl.LazyFrame:
    """Handle variable extraction for Polars LazyFrame.
    
    Args:
        dataset: The Polars LazyFrame to extract variables from.
        variables: List of required variable names to extract.
        optional_variables: List of optional variable names to extract.
        optional_variables_mapping: Dictionary mapping optional variables to
            the required variables they replace.
    
    Returns:
        The LazyFrame containing only the found columns.
        
    Raises:
        KeyError: If any required variables are missing from the LazyFrame.
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


def check_for_valid_times_polars_lazyframe(
    dataset: pl.LazyFrame, start_date: datetime.datetime, end_date: datetime.datetime
) -> bool:
    """Check if the LazyFrame has valid times in the given date range.
    
    Args:
        dataset: The Polars LazyFrame to check for valid times.
        start_date: The start date of the time range to check.
        end_date: The end date of the time range to check.
    
    Returns:
        True if the LazyFrame has any times within the specified range,
        False otherwise.
    """
    # Try different time column names
    time_cols = ["valid_time", "time", "init_time"]
    available_columns = dataset.collect_schema().names()
    
    for time_col in time_cols:
        if time_col in available_columns:
            # Filter the LazyFrame to only include valid times in the given date range
            time_filtered_lf = dataset.select(pl.col(time_col)).filter(
                (pl.col(time_col) >= start_date) & (pl.col(time_col) <= end_date)
            )
            # If the filtered LazyFrame has any rows, return True
            return not time_filtered_lf.collect().is_empty()
    
    # If no time column found, return False
    return False

def check_for_spatial_data_polars_lazyframe(
    dataset: pl.LazyFrame,
    location: "regions.Region",
) -> bool:
    """Check if the LazyFrame has spatial data for the given location."""
    # Check if LazyFrame has latitude and longitude columns
    lat_cols = ["latitude", "lat"]
    lon_cols = ["longitude", "lon"]
    
    available_columns = dataset.collect_schema().names()
    
    lat_col = utils.check_for_vars(lat_cols, available_columns)
    lon_col = utils.check_for_vars(lon_cols, available_columns)
    
    
    if lat_col is None or lon_col is None:
        return False
    coords = location.get_bounding_coordinates
    # Get location bounds
    lat_min, lat_max = coords.latitude_min, coords.latitude_max
    lon_min, lon_max = coords.longitude_min, coords.longitude_max
    
    # Check if there are any data points within the location bounds
    filtered_data = dataset.filter(
        (pl.col(lat_col) >= lat_min) & (pl.col(lat_col) <= lat_max) &
        (pl.col(lon_col) >= lon_min) & (pl.col(lon_col) <= lon_max)
    )
    
    return not filtered_data.collect().is_empty()
