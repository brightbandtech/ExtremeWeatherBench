"""Handle variable extraction for Polars LazyFrames."""

import datetime
from typing import TYPE_CHECKING

import polars as pl

from extremeweatherbench import utils

if TYPE_CHECKING:
    from extremeweatherbench import regions


def safely_pull_variables(
    data: pl.LazyFrame,
    variables: list[str],
) -> pl.LazyFrame:
    """Safely extract variables from a Polars LazyFrame with optional replacements.

    This function handles variable extraction from a Polars LazyFrame, supporting
    both required and optional variables. Optional variables can replace required
    ones based on the provided mapping, allowing for flexible data processing.
    The function works with LazyFrames, so operations are deferred until execution.

    Args:
        data: The Polars LazyFrame to extract variables from.
        variables: List of required variable names to extract. These must be
            present in the LazyFrame.

    Returns:
        pl.LazyFrame: A new LazyFrame containing only the found columns.
            Includes all found required variables and optional variables.
            Operations remain lazy until collect() is called.

    Raises:
        KeyError: If any required variables are missing from the LazyFrame
            and not replaced by optional variables.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"temp": [20, 25], "pressure": [1013, 1015]})
        >>> lf = df.lazy()
        >>> variables = ["temperature"]
        >>> optional = ["temp"]
        >>> mapping = {"temp": "temperature"}
        >>> result = safely_pull_variables(lf, variables, optional, mapping)
        >>> result.collect().columns
        ['temp']
    """
    from extremeweatherbench import defaults

    # Get column names from LazyFrame
    available_columns = data.collect_schema().names()

    # Track which variables we've found (use set to avoid duplicates)
    found_variables = set()

    # First, add any coordinate columns that exist
    for coord in defaults.DEFAULT_COORDINATE_VARIABLES:
        if coord in available_columns:
            found_variables.add(coord)

    # Then check for required variables
    missing_variables = []
    for var in variables:
        if var in available_columns:
            found_variables.add(var)
        else:
            missing_variables.append(var)

    # Raise error if any required variables are missing
    if missing_variables:
        raise KeyError(
            f"Required variables {missing_variables} not found in LazyFrame. "
            f"Available columns: {available_columns}"
        )

    # Return LazyFrame with only the found columns
    return data.select(list(found_variables))


def check_for_valid_times(
    data: pl.LazyFrame,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> bool:
    """Check if the LazyFrame contains any data within the specified time range.

    This function searches for time columns in the LazyFrame using common
    naming conventions and checks if any data points fall within the given
    date range. It tries multiple possible column names for time data.
    The function uses lazy evaluation for efficiency.

    Args:
        data: The Polars LazyFrame to check for valid times. Should contain
            at least one time column with datetime data.
        start_date: The start date of the time range to check (inclusive).
        end_date: The end date of the time range to check (inclusive).

    Returns:
        bool: True if the LazyFrame has any times within the specified range,
            False if no time column is found or no data falls within the range.

    Note:
        The function searches for time columns with names: 'valid_time',
        'time', or 'init_time'. The first matching column is used for
        the time range check. The check is performed lazily for efficiency.

    Example:
        >>> import polars as pl
        >>> import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "valid_time": pl.Series(pd.date_range("2023-01-01", periods=5)),
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> lf = df.lazy()
        >>> start = datetime.datetime(2023, 1, 2)
        >>> end = datetime.datetime(2023, 1, 4)
        >>> check_for_valid_times(lf, start, end)
        True
    """
    # Try different time column names
    time_cols = ["valid_time", "time", "init_time"]
    available_columns = data.collect_schema().names()

    for time_col in time_cols:
        if time_col in available_columns:
            # Check the dtype of the column
            col_dtype = data.collect_schema()[time_col]
            # Filter LazyFrame to include valid times in date range
            # Handle both datetime and string types
            if col_dtype in (pl.Datetime, pl.Date):
                # Column is already datetime, compare directly
                time_filtered_lf = data.select(pl.col(time_col)).filter(
                    (pl.col(time_col) >= pl.lit(start_date))
                    & (pl.col(time_col) <= pl.lit(end_date))
                )
            else:
                # Column is string, convert to datetime first
                time_filtered_lf = data.select(pl.col(time_col)).filter(
                    (pl.col(time_col).str.to_datetime() >= pl.lit(start_date))
                    & (pl.col(time_col).str.to_datetime() <= pl.lit(end_date))
                )
            # If the filtered LazyFrame has any rows, return True
            return not time_filtered_lf.collect().is_empty()

    # If no time column found, return False
    return False


def check_for_spatial_data(
    data: pl.LazyFrame,
    location: "regions.Region",
) -> bool:
    """Check if the LazyFrame contains spatial data within the specified region.

    This function verifies that the LazyFrame has latitude and longitude columns
    and contains at least one data point that falls within the bounding box
    of the specified geographic region. The function uses lazy evaluation for
    efficiency when checking spatial bounds.

    Args:
        data: The Polars LazyFrame to check for spatial data. Should contain
            latitude and longitude columns with coordinate data.
        location: A Region object that defines the geographic area to check
            against.

    Returns:
        bool: True if the LazyFrame has spatial data within the region bounds,
            False if no coordinate columns are found or no data points fall
            within the region.

    Note:
        The function searches for coordinate columns with names: 'latitude'
        or 'lat' for latitude, and 'longitude' or 'lon' for longitude.
        The first matching column names are used for the spatial check.
        The spatial filtering is performed lazily for efficiency.

    Example:
        >>> import polars as pl
        >>> from extremeweatherbench.regions import Region
        >>> df = pl.DataFrame(
        ...     {
        ...         "latitude": [40.0, 41.0, 42.0],
        ...         "longitude": [-74.0, -73.0, -72.0],
        ...         "value": [1, 2, 3],
        ...     }
        ... )
        >>> lf = df.lazy()
        >>> region = Region(...)  # Define your region
        >>> check_for_spatial_data(lf, region)
        True
    """
    # Check if LazyFrame has latitude and longitude columns
    lat_cols = ["latitude", "lat"]
    lon_cols = ["longitude", "lon"]

    available_columns = data.collect_schema().names()

    lat_col = utils.check_for_vars(lat_cols, available_columns)
    lon_col = utils.check_for_vars(lon_cols, available_columns)

    if lat_col is None or lon_col is None:
        return False
    coords = location.as_geopandas().total_bounds
    # Get location bounds
    lat_min, lat_max = coords[1], coords[3]
    lon_min, lon_max = coords[0], coords[2]

    # Check if there are any data points within the location bounds
    filtered_data = data.filter(
        (pl.col(lat_col) >= lat_min)
        & (pl.col(lat_col) <= lat_max)
        & (pl.col(lon_col) >= lon_min)
        & (pl.col(lon_col) <= lon_max)
    )

    return not filtered_data.collect().is_empty()
