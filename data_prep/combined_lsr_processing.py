#!/usr/bin/env python
"""Combined LSR processing script.

Combines US, Canadian, and Australian LSR data in sequence:
1. Generate US LSR data from SPC NOAA
2. Add Canadian LSR data
3. Add Australian LSR data
"""

import asyncio
import logging
from datetime import timedelta

import aiohttp
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: Generate US LSR data (from generate_lsr.py)
# ============================================================================


def _process_lsr_dataframe(
    content: str, date: pd.Timestamp, adjust_times: bool = False
) -> pd.DataFrame:
    """Process LSR CSV content into a standardized DataFrame.

    Args:
        content: Raw CSV content as string
        date: Date for the LSR data
        adjust_times: Whether to adjust times for midnight-noon range
    Returns:
        Processed DataFrame with standardized columns
    """
    try:
        from io import StringIO

        df = pd.read_csv(
            StringIO(content),
            delimiter=",",
            engine="python",
            names=[
                "Time",
                "Scale",
                "Location",
                "County",
                "State",
                "Lat",
                "Lon",
                "Comments",
            ],
        )
    except Exception as e:
        logger.error("Error parsing LSR data for %s: %s", date, e)
        return pd.DataFrame()

    # If the dataframe has 3 rows, it is empty
    if len(df) == 3:
        return pd.DataFrame()

    # Initialize report_type column
    df["report_type"] = None

    # Find rows with headers and mark subsequent rows with report type
    for i, row in df.iterrows():
        new_i = i + 1
        if "F_Scale" in row.values:
            df.loc[new_i:, "report_type"] = "tor"
        elif "Speed" in row.values:
            df.loc[new_i:, "report_type"] = "wind"
        elif "Size" in row.values:
            df.loc[new_i:, "report_type"] = "hail"

    # Keep only necessary columns
    df = df[["Lat", "Lon", "report_type", "Time", "Scale"]]
    # Remove rows that have 'Lat' in the 'Lat' column (header rows)
    df = df[df["Lat"] != "Lat"]

    if len(df) == 0:
        return pd.DataFrame()

    # Convert time and add date
    time = pd.to_datetime(df["Time"], format="%H%M").dt.time
    df["Time"] = pd.to_datetime(date.strftime("%Y-%m-%d") + " " + time.astype(str))

    # Adjust times between 00:00 and 11:59 to next date if requested
    if adjust_times:
        midnight_to_noon_mask = df["Time"].dt.time < pd.Timestamp("12:00").time()
        df.loc[midnight_to_noon_mask, "Time"] = df.loc[
            midnight_to_noon_mask, "Time"
        ] + timedelta(days=1)

    # Rename columns to standard format
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Time": "valid_time",
            "Scale": "scale",
        }
    )

    # Handle unknown scale values
    df.loc[df["scale"] == "UNK", "scale"] = np.nan
    df["scale"] = df["scale"].astype(float)

    return df


async def pull_lsr_data_async(
    session: aiohttp.ClientSession, date: pd.Timestamp
) -> pd.DataFrame:
    """Pull LSR data from SPC NOAA for a given date asynchronously.

    Args:
        session: aiohttp ClientSession for making requests
        date: Pandas Timestamp for the date to retrieve
    Returns:
        DataFrame with columns: latitude, longitude, report_type,
        valid_time, and scale
    """
    if date < pd.Timestamp("2004-02-29"):
        raise ValueError("LSR data before 2004-02-29 not available")

    # Try the filtered URL first, if it fails, try without _filtered
    url = f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts_filtered.csv"  # noqa: E501

    try:
        async with session.head(url) as response:
            if response.status == 404:
                url = f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts.csv"  # noqa: E501

        async with session.get(url) as response:
            if response.status != 200:
                logger.error(
                    "Error pulling LSR data for %s: HTTP %s", date, response.status
                )
                return pd.DataFrame()
            content = await response.text()

    except Exception as e:
        logger.error("Error pulling LSR data for %s: %s", date, e)
        return pd.DataFrame()

    return _process_lsr_dataframe(content, date, adjust_times=True)


async def download_lsr_data_range(
    start_date: pd.Timestamp, end_date: pd.Timestamp, max_concurrent: int = 100
) -> pd.DataFrame:
    """Download and combine LSR data for date range asynchronously.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        max_concurrent: Maximum number of concurrent downloads
    Returns:
        Combined DataFrame with all LSR data for the date range
    """
    # Generate list of dates
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(
        session: aiohttp.ClientSession, date: pd.Timestamp
    ):
        async with semaphore:
            return await pull_lsr_data_async(session, date)

    # Download all data concurrently
    async with aiohttp.ClientSession() as session:
        logger.info(
            "Downloading LSR data for %s dates from %s to %s",
            len(dates),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        tasks = [download_with_semaphore(session, date) for date in dates]
        dataframes = await tqdm_asyncio.gather(*tasks)

    # Filter out exceptions and empty dataframes
    valid_dfs = []
    successful_downloads = 0

    for i, result in enumerate(dataframes):
        if isinstance(result, Exception):
            logger.error(
                "Error downloading data for %s: %s",
                dates[i].strftime("%Y-%m-%d"),
                result,
            )
        elif isinstance(result, pd.DataFrame) and not result.empty:
            valid_dfs.append(result)
            successful_downloads += 1

    logger.info(
        "Successfully downloaded data for %s/%s dates", successful_downloads, len(dates)
    )

    if not valid_dfs:
        logger.error("No valid data found for the specified date range")
        return pd.DataFrame()

    # Combine all dataframes
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    logger.info("Combined dataset contains %s LSR reports", len(combined_df))

    return combined_df


async def generate_us_lsr_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download and process US LSR data from SPC NOAA for date range.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
    Returns:
        DataFrame with US LSR data
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Generating US LSR data")
    logger.info("=" * 80)
    df_range = await download_lsr_data_range(start, end, max_concurrent=5)
    logger.info("US LSR data: %s reports", len(df_range))

    # Convert to correct types
    df_range["latitude"] = df_range["latitude"].astype(float)
    df_range["longitude"] = df_range["longitude"].astype(float)
    df_range["scale"] = df_range["scale"].astype(float)
    df_range["valid_time"] = pd.to_datetime(df_range["valid_time"])
    df_range["report_type"] = df_range["report_type"].astype(str)

    return df_range


# ============================================================================
# PART 2: Add Canadian LSR data (from canada_lsr_to_spc_lsr.py)
# ============================================================================


def convert_can_lsr_to_bb_lsr(can_lsr: pd.DataFrame) -> pd.DataFrame:
    """Convert Canadian LSR data to standardized LSR format."""
    # Rename columns to align with standardized LSR format
    modified_can_lsr = can_lsr.rename(
        columns={
            "Date/Time UTC": "valid_time",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

    # Convert time to datetime type
    modified_can_lsr["valid_time"] = pd.to_datetime(modified_can_lsr["valid_time"])

    # Convert from mm to hundredths of inch
    modified_can_lsr["hail_size"] = np.round(
        modified_can_lsr["Maximum Hail Dimension (mm)"] * 0.0393701 * 100, 0
    )
    # Convert EF scale strings to numeric values
    damage_mapping = {
        "ef0": 0,
        "default_ef0": 0,
        "ef1": 1,
        "ef2": 2,
        "ef3": 3,
        "ef4": 4,
        "ef5": 5,
    }
    modified_can_lsr["Damage"] = (
        modified_can_lsr["Damage"].map(damage_mapping).astype(float)
    )
    # Merge hail size and Fujita scale into scale column
    modified_can_lsr["scale"] = modified_can_lsr["hail_size"].fillna(
        modified_can_lsr["Damage"]
    )

    modified_can_lsr = modified_can_lsr[
        ["latitude", "longitude", "report_type", "valid_time", "scale"]
    ]
    return modified_can_lsr


def add_canadian_lsr_data(us_lsr_df: pd.DataFrame) -> pd.DataFrame:
    """Add Canadian LSR data to US LSR data.

    Args:
        us_lsr_df: DataFrame containing US LSR data
    Returns:
        Combined DataFrame with US and Canadian LSR data
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Adding Canadian LSR data")
    logger.info("=" * 80)

    can_lsr = pd.read_csv(
        "gs://extremeweatherbench/deprecated/CanadaLSRData_2020-2024.csv"
    )
    logger.info("Loaded Canadian LSR data: %s reports", len(can_lsr))

    converted_can_lsr = convert_can_lsr_to_bb_lsr(can_lsr)
    combined_lsr_df = pd.concat([us_lsr_df, converted_can_lsr])
    combined_lsr_df["latitude"] = combined_lsr_df["latitude"].astype(float)
    combined_lsr_df["longitude"] = combined_lsr_df["longitude"].astype(float)
    combined_lsr_df = combined_lsr_df.sort_values(by="valid_time")

    logger.info("Combined US+Canada LSR data: %s reports", len(combined_lsr_df))
    return combined_lsr_df


# ============================================================================
# PART 3: Add Australian LSR data (from australia_lsr_to_spc_lsr.py)
# ============================================================================


def convert_aus_lsr_to_bb_lsr(aus_lsr: pd.DataFrame) -> pd.DataFrame:
    """Convert Australian LSR data to standardized LSR format."""
    # Rename columns to align with standardized LSR format
    modified_aus_lsr = aus_lsr.rename(
        columns={
            "Date/Time UTC": "valid_time",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

    # Convert time to datetime type
    modified_aus_lsr["valid_time"] = pd.to_datetime(modified_aus_lsr["valid_time"])

    # Convert from cm to hundredths of inch
    modified_aus_lsr["hail_size"] = np.round(
        modified_aus_lsr["Hail size"] * 0.393701 * 100, 0
    )

    # Merge hail size and Fujita scale into scale column
    modified_aus_lsr["scale"] = modified_aus_lsr["hail_size"].fillna(
        modified_aus_lsr["Fujita scale"]
    )

    # Drop unnecessary columns and reorder
    modified_aus_lsr = modified_aus_lsr.drop(
        columns=["hail_size", "Hail size", "Fujita scale", "Nearest town", "State"]
    )
    modified_aus_lsr = modified_aus_lsr[
        ["latitude", "longitude", "report_type", "valid_time", "scale"]
    ]
    return modified_aus_lsr


def add_australian_lsr_data(combined_lsr_df: pd.DataFrame) -> pd.DataFrame:
    """Add Australian LSR data to combined US+Canada LSR data.

    Args:
        combined_lsr_df: DataFrame with combined US and Canadian LSR data
    Returns:
        Final DataFrame with US, Canadian, and Australian LSR data
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Adding Australian LSR data")
    logger.info("=" * 80)

    aus_lsr = pd.read_csv(
        "gs://extremeweatherbench/deprecated/AustralianLSRData_2020-2024.csv"
    )
    logger.info("Loaded Australian LSR data: %s reports", len(aus_lsr))

    converted_aus_lsr = convert_aus_lsr_to_bb_lsr(aus_lsr)
    final_lsr_df = pd.concat([combined_lsr_df, converted_aus_lsr])
    final_lsr_df["latitude"] = final_lsr_df["latitude"].astype(float)
    final_lsr_df["longitude"] = final_lsr_df["longitude"].astype(float)

    logger.info("Final combined LSR data: %s reports", len(final_lsr_df))
    return final_lsr_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main() -> pd.DataFrame:
    """Execute complete LSR data processing pipeline.

    Downloads US LSR data, adds Canadian and Australian data, then
    saves combined output to parquet file.

    Returns:
        Final combined DataFrame with all LSR data
    """
    # Step 1: Generate US LSR data
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-09-27")
    us_lsr_data = await generate_us_lsr_data(start, end)

    # Step 2: Add Canadian LSR data
    us_canada_lsr_data = add_canadian_lsr_data(us_lsr_data)

    # Step 3: Add Australian LSR data
    final_lsr_data = add_australian_lsr_data(us_canada_lsr_data)

    # Save final output
    output_file = "combined_canada_australia_us_lsr_01012020_09272025.parq"
    final_lsr_data.to_parquet(output_file)
    logger.info("=" * 80)
    logger.info("Saved final output to: %s", output_file)
    logger.info("=" * 80)

    return final_lsr_data


def verify_output(df: pd.DataFrame) -> bool:
    """Verify the output meets expected criteria.

    Checks:
    1. Rows with latitude >= 49 (Canadian data)
    2. Total row count
    3. Row count for specific date range

    Args:
        df: Final combined LSR DataFrame
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    # Check 1: Rows with latitude >= 49
    valid_high_lat_count = 1581
    high_lat_count = len(df[df["latitude"] >= 49])
    logger.info("Check 1: Rows with latitude >= 49")
    logger.info("  Expected: %s", valid_high_lat_count)
    logger.info("  Actual:   %s", high_lat_count)
    logger.info(
        "  Status:   %s", "PASS" if high_lat_count == valid_high_lat_count else "FAIL"
    )

    # Check 2: Total rows
    valid_row_count = 136010
    total_rows = len(df)
    logger.info("Check 2: Total rows")
    logger.info("  Expected: %s", valid_row_count)
    logger.info("  Actual:   %s", total_rows)
    logger.info("  Status:   %s", "PASS" if total_rows == valid_row_count else "FAIL")

    # Check 3: Length between specific dates
    valid_date_range_count = 56
    start_time = pd.Timestamp("2020-09-05 12:00:00")
    end_time = pd.Timestamp("2020-09-06 11:59:00")
    mask = (df["valid_time"] >= start_time) & (df["valid_time"] <= end_time)
    date_range_count = len(df[mask])
    logger.info("Check 3: Rows between 2020-09-05 12:00:00 and 2020-09-06 11:59:00")
    logger.info("  Expected: %s", valid_date_range_count)
    logger.info("  Actual:   %s", date_range_count)
    logger.info(
        "  Status:   %s",
        "PASS" if date_range_count == valid_date_range_count else "FAIL",
    )

    logger.info("=" * 80)

    # Return boolean for all checks passing
    all_pass = (
        high_lat_count == valid_high_lat_count
        and total_rows == valid_row_count
        and date_range_count == valid_date_range_count
    )
    return all_pass


if __name__ == "__main__":
    # Run the combined processing
    final_data = asyncio.run(main())

    # Verify the output
    all_checks_passed = verify_output(final_data)

    if all_checks_passed:
        logger.info("All verification checks passed!")
    else:
        logger.warning("Some verification checks failed!")
