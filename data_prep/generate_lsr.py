import asyncio
import logging
from datetime import timedelta

import aiohttp
import numpy as np
import pandas as pd
import requests
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pull_lsr_data(date: pd.Timestamp) -> pd.DataFrame:
    """Pull the latest LSR data for a given date. A "date" for LSRs is considered the
    date starting at 12 UTC to the next day at 11:59 UTC.

    Args:
        date: A pandas Timestamp object.
    Returns:
        df: A pandas DataFrame containing the LSR data with columns lat, lon,
        report_type, time, and scale.
    """
    # Try the filtered URL first, if it fails, try without _filtered
    url = f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts_filtered.csv"  # noqa: E501
    # Check if the URL exists by attempting to open it
    response = requests.head(url)
    if date < pd.Timestamp("2004-02-29"):
        raise ValueError("LSR data before 2004-02-29 is not available in CSV format")
    if response.status_code == 404:
        # If the filtered URL doesn't exist, use the non-filtered version
        url = (
            f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts.csv"
        )
    # Read the CSV file with all columns to identify report types
    try:
        df = pd.read_csv(
            url,
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
        logger.error("Error pulling LSR data for %s: %s", date, e)
        return pd.DataFrame()
    if len(df) == 3:
        return pd.DataFrame()
    # Initialize report_type column
    df["report_type"] = None

    # Find rows with headers and mark subsequent rows with appropriate report type
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
    # Remove rows that have 'Lat' in the 'Lat' column (these are header rows)
    df = df[df["Lat"] != "Lat"]
    time = pd.to_datetime(df["Time"], format="%H%M").dt.time
    df["Time"] = pd.to_datetime(date.strftime("%Y-%m-%d") + " " + time.astype(str))
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Time": "valid_time",
            "Scale": "scale",
        }
    )
    df["scale"] = df["scale"].replace("UNK", np.nan)
    df["scale"] = df["scale"].astype(float)
    return df


async def pull_lsr_data_async(
    session: aiohttp.ClientSession, date: pd.Timestamp
) -> pd.DataFrame:
    """Async version of pull_lsr_data function.

    Args:
        session: aiohttp ClientSession for making requests
        date: A pandas Timestamp object.
    Returns:
        df: A pandas DataFrame containing the LSR data with columns
            latitude, longitude, report_type, valid_time, and scale.
    """
    if date < pd.Timestamp("2004-02-29"):
        raise ValueError("LSR data before 2004-02-29 is not available in CSV format")

    # Try the filtered URL first, if it fails, try without _filtered
    url = f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts_filtered.csv"  # noqa: E501

    try:
        async with session.head(url) as response:
            if response.status == 404:
                # If the filtered URL doesn't exist, use the non-filtered version
                url = f"https://www.spc.noaa.gov/climo/reports/{date.strftime('%y%m%d')}_rpts.csv"  # noqa: E501

        # Read the CSV file with all columns to identify report types
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

    try:
        # Parse CSV content using pandas
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

    if len(df) == 3:
        return pd.DataFrame()

    # Initialize report_type column
    df["report_type"] = None

    # Find rows with headers and mark subsequent rows with appropriate report type
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
    # Remove rows that have 'Lat' in the 'Lat' column (these are header rows)
    df = df[df["Lat"] != "Lat"]

    if len(df) == 0:
        return pd.DataFrame()

    time = (pd.to_datetime(df["Time"], format="%H%M")).dt.time
    df["Time"] = pd.to_datetime(date.strftime("%Y-%m-%d") + " " + time.astype(str))

    # Adjust times between 00:00 and 11:59 to the next date
    midnight_to_noon_mask = df["Time"].dt.time < pd.Timestamp("12:00").time()
    df.loc[midnight_to_noon_mask, "Time"] = df.loc[
        midnight_to_noon_mask, "Time"
    ] + timedelta(days=1)
    df = df.rename(
        columns={
            "Lat": "latitude",
            "Lon": "longitude",
            "Time": "valid_time",
            "Scale": "scale",
        }
    )
    df.loc[df["scale"] == "UNK", "scale"] = np.nan
    df["scale"] = df["scale"].astype(float)
    return df


async def download_lsr_data_range(
    start_date: pd.Timestamp, end_date: pd.Timestamp, max_concurrent: int = 100
) -> pd.DataFrame:
    """Download and combine LSR data for a date range asynchronously.

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


if __name__ == "__main__":

    async def download_lsr_data_range_async():
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2025-09-27")

        # Change max_concurrent to increase download rate
        df_range = await download_lsr_data_range(start, end, max_concurrent=5)
        logger.info("Async date range: %s reports", len(df_range))
        return df_range

    # Run the async code
    data = asyncio.run(download_lsr_data_range_async())
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)
    data["scale"] = data["scale"].astype(float)
    data.to_parquet("lsr_01012020_09272025.parq")
