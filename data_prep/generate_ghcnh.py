"""Streaming GHCN-H data download script that processes files immediately
to save disk space and skip already processed data.
"""

import asyncio
from pathlib import Path

import aiohttp
import nest_asyncio
import pandas as pd
import polars as pl
from tqdm.asyncio import tqdm


async def download_station_list():
    """Download and parse the official GHCN-H station list."""
    station_list_url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt"  # noqa: E501

    async with aiohttp.ClientSession() as session:
        async with session.get(station_list_url) as response:
            if response.status == 200:
                content = await response.text()
                return parse_station_list(content)
            else:
                raise Exception(f"Failed to download station list: {response.status}")


def parse_station_list(content):
    """Parse the GHCN-H station list text format."""
    stations = []
    lines = content.strip().split("\n")

    for line in lines:
        if len(line) >= 30:  # Ensure line has minimum expected length
            # Parse fixed-width format based on the station list structure
            station_id = line[0:11].strip()
            latitude = line[12:20].strip()
            longitude = line[21:30].strip()
            elevation = line[31:37].strip()
            name = line[38:].strip()

            try:
                lat = float(latitude) if latitude else None
                lon = float(longitude) if longitude else None
                elev = float(elevation) if elevation not in ["", "-999.9"] else None

                stations.append(
                    {
                        "station_id": station_id,
                        "latitude": lat,
                        "longitude": lon,
                        "elevation": elev,
                        "name": name,
                    }
                )
            except ValueError:
                # Skip stations with invalid coordinates
                continue

    return pd.DataFrame(stations)


def construct_station_download_url(station_id, year):
    """Construct download URL for a specific station and year."""
    base_url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year"  # noqa: E501
    filename = f"GHCNh_{station_id}_{year}.psv"
    return f"{base_url}/{year}/psv/{filename}"


# Cache for existing station-year combinations for O(1) lookup
_existing_station_years = None
_cache_file_path = None


def check_station_already_processed(station_id, year, main_parquet_file):
    """Check if a station-year combination is already in the main parquet file."""
    global _existing_station_years, _cache_file_path

    if not Path(main_parquet_file).exists():
        return False

    try:
        # Only read the parquet file once and build station-year set
        if _existing_station_years is None or _cache_file_path != main_parquet_file:
            print("Loading existing data for duplicate checking with Polars...")
            df = pl.read_parquet(main_parquet_file, columns=["station", "time"])

            # Extract year from time and create station-year combinations
            station_years = (
                df.with_columns(pl.col("time").dt.year().alias("year"))
                .select(["station", "year"])
                .unique()
                .to_pandas()  # Convert to pandas for set creation
            )

            # Create set of (station_id, year) tuples for O(1) lookup
            _existing_station_years = set(
                zip(station_years["station"], station_years["year"])
            )
            _cache_file_path = main_parquet_file
            print(
                f"Cached {len(_existing_station_years)} unique "
                f"station-year combinations"
            )

        # O(1) lookup in set
        return (station_id, year) in _existing_station_years
    except Exception as e:
        print(f"Error checking existing data: {e}")
        return False


def clear_existing_data_cache():
    """Clear the existing data cache (useful when parquet file is updated)."""
    global _existing_station_years, _cache_file_path
    _existing_station_years = None
    _cache_file_path = None


async def download_and_process_station_file(
    session, url, output_dir, filename, main_parquet_file, overwrite=False
):
    """Download a station file and immediately process it to parquet."""
    station_id = filename.split("_")[1]  # Extract station ID from filename
    year = filename.split("_")[2].split(".")[0]  # Extract year

    # Check if already processed
    if not overwrite and check_station_already_processed(
        station_id, year, main_parquet_file
    ):
        print(f"  ⏭ Skipping {filename} - already processed")
        return "skipped"

    temp_path = Path(output_dir) / filename

    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                # Check if it's actually data (not an error page)
                if len(content) > 100:  # Reasonable minimum for valid data
                    # Write temporary file
                    with open(temp_path, "wb") as file:
                        file.write(content)

                    # Immediately process and append to main parquet
                    success = process_file_immediately_and_append(
                        temp_path, main_parquet_file
                    )
                    return "processed" if success else "failed"

            elif response.status == 404:
                # Station data not available for this year - normal
                return "not_found"
            else:
                print(f"Unexpected status {response.status} for {filename}")
                return "failed"
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return "failed"

    return "failed"


# Global list to accumulate processed DataFrames for batch writing
_batch_data_frames = []
_batch_size = 50  # Process 50 files before writing to parquet


def process_file_immediately_and_append(file_path, main_parquet_file):
    """Process a single PSV file and store for batch append to parquet."""
    global _batch_data_frames

    try:
        # Process the PSV file
        df = process_psv_file(file_path)
        if df is None or len(df) == 0:
            print(f"    No valid data in {file_path.name}")
            file_path.unlink()  # Delete the file
            return False

        # Aggregate to hourly
        df = aggregate_to_hourly(df)

        # Apply transformations
        df = apply_data_transformations(df)

        # Add to batch instead of immediate append
        _batch_data_frames.append(df)

        # Delete the processed PSV file immediately
        file_path.unlink()
        print(f"    ✓ Processed and batched {file_path.name}: {len(df)} rows")

        # Check if batch is full - if so, write to parquet
        if len(_batch_data_frames) >= _batch_size:
            flush_batch_to_parquet(main_parquet_file)

        return True

    except Exception as e:
        print(f"    ✗ Error processing {file_path.name}: {e}")
        try:
            file_path.unlink()  # Still delete the problematic file
        except Exception:
            pass
        return False


def flush_batch_to_parquet(main_parquet_file):
    """Write accumulated batch of DataFrames to parquet file."""
    global _batch_data_frames

    if not _batch_data_frames:
        return

    try:
        print(f"📦 Flushing batch of {len(_batch_data_frames)} files to parquet...")

        # Combine all DataFrames in the batch
        batch_df = pd.concat(_batch_data_frames, ignore_index=True)

        # Convert to Polars for efficient parquet operations
        batch_polars = pl.from_pandas(batch_df)

        # Append to main parquet file
        if Path(main_parquet_file).exists():
            existing_df = pl.read_parquet(main_parquet_file)
            combined_df = pl.concat([existing_df, batch_polars])
            combined_df.write_parquet(main_parquet_file)
        else:
            batch_polars.write_parquet(main_parquet_file)

        # Clear the batch and invalidate cache
        _batch_data_frames.clear()
        clear_existing_data_cache()

        print(f"✅ Batch written: {len(batch_df)} total rows appended")

    except Exception as e:
        print(f"❌ Error writing batch to parquet: {e}")
        # Clear batch anyway to avoid memory buildup
        _batch_data_frames.clear()


def process_psv_file(file_path):
    """Process a single PSV file and return DataFrame or None."""
    try:
        df = pd.read_csv(file_path, sep="|", low_memory=False)

        # Drop rows where temperature_quality_check is not 1 or 5
        if "temperature_Quality_Code" in df.columns:
            if isinstance(df["temperature_Quality_Code"].iloc[0], dict):
                df = df[
                    df["temperature_Quality_Code"].apply(
                        lambda x: x.get("member0") in [1, 5]
                        if x is not None and isinstance(x, dict)
                        else False
                    )
                ]
            else:
                df["temperature_Quality_Code"] = df["temperature_Quality_Code"].astype(
                    str
                )
                df = df[df["temperature_Quality_Code"].isin(["1", "5"])]

        # Select required columns
        required_cols = [
            "STATION",
            "Station_name",
            "DATE",
            "LATITUDE",
            "LONGITUDE",
            "Elevation",
            "temperature",
            "station_level_pressure",
            "sea_level_pressure",
            "precipitation",
            "wind_speed",
            "wind_direction",
            "dew_point_temperature",
            "relative_humidity",
        ]

        # Only keep columns that exist
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]

        # Convert DATE to datetime
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def aggregate_to_hourly(data):
    """Aggregate rows that represent same hour to one row."""
    data["hour"] = data["DATE"].dt.round("h")
    data["hour_dist"] = (data["DATE"] - data["hour"]).dt.total_seconds().abs() // 60
    data = data.sort_values(["hour", "hour_dist"])

    if data["hour"].duplicated().any():
        data = data.groupby("hour").apply(
            lambda df: df.bfill().iloc[0], include_groups=False
        )
    data = data.reset_index(drop=True)
    data["time"] = data["DATE"].dt.round("h")
    return data


def apply_data_transformations(df):
    """Apply all data transformations to a dataframe."""
    # Extract 'member0' values from dictionaries in 'relative_humidity' column
    if "relative_humidity" in df.columns:
        mask = df["relative_humidity"].apply(lambda x: isinstance(x, dict))
        df.loc[mask, "relative_humidity"] = df.loc[mask, "relative_humidity"].apply(
            lambda x: x.get("member0")
        )

    # Apply column renaming
    df = df.rename(
        columns={
            "temperature": "surface_air_temperature",
            "dew_point_temperature": "surface_dew_point",
            "wind_speed": "surface_wind_speed",
            "wind_direction": "surface_wind_from_direction",
            "station_level_pressure": "surface_air_pressure",
            "sea_level_pressure": "air_pressure_at_mean_sea_level",
            "c": "cloud_area_fraction",
            "relative_humidity": "surface_relative_humidity",
            "precipitation": "accumulated_1_hour_precipitation",
            "STATION": "station",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "Elevation": "elevation",
            "Station_name": "name",
        }
    )

    # Drop unnecessary columns if they exist
    columns_to_drop = ["DATE", "hour"]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)

    return df


async def download_and_process_station_file_with_semaphore(
    semaphore, session, url, output_dir, filename, main_parquet_file, overwrite
):
    """Download and process a single station file with semaphore limiting."""
    async with semaphore:
        return await download_and_process_station_file(
            session, url, output_dir, filename, main_parquet_file, overwrite
        )


async def download_all_stations_streaming(
    years, output_dir, overwrite=False, max_concurrent=10
):
    """Download and immediately process GHCN-H stations to save disk space."""
    print("Downloading and processing GHCN-H stations with streaming approach...")
    station_df = await download_station_list()
    print(f"Found {len(station_df)} stations in official list")

    # Main parquet file to append to
    main_parquet_file = "ghcnh_all_2020_2024.parq"
    print(f"Will stream data to: {main_parquet_file}")

    # Pre-filter tasks to skip those already processed
    total_tasks = 0
    skipped_during_setup = 0

    # Create task list with pre-filtering
    tasks = []

    print("Checking for existing data and setting up download tasks...")

    # Calculate total combinations for progress bar
    total_combinations = len(station_df) * len(years)

    with tqdm(
        total=total_combinations, desc="Setting up tasks", unit="checks", ncols=80
    ) as setup_pbar:
        for year in years:
            for _, station in station_df.iterrows():
                station_id = station["station_id"]
                filename = f"GHCNh_{station_id}_{year}.psv"

                # Check if already processed before creating task
                if not overwrite and check_station_already_processed(
                    station_id, year, main_parquet_file
                ):
                    skipped_during_setup += 1
                    setup_pbar.set_postfix(
                        {"Skipped": skipped_during_setup, "Tasks": len(tasks)}
                    )
                else:
                    url = construct_station_download_url(station_id, year)
                    tasks.append(
                        {
                            "station_id": station_id,
                            "year": year,
                            "url": url,
                            "filename": filename,
                        }
                    )
                    total_tasks += 1
                    setup_pbar.set_postfix(
                        {"Skipped": skipped_during_setup, "Tasks": len(tasks)}
                    )

                setup_pbar.update(1)

    print(f"Pre-filtered: {skipped_during_setup} already processed")
    print(f"Will attempt to download: {len(tasks)} files")

    if not tasks:
        print("All data already exists! Nothing to download.")
        return

    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent)

    # Counters for progress tracking
    progress_counters = {
        "processed": 0,
        "skipped": 0,
        "not_found": 0,
        "failed": 0,
        "errors": 0,
    }

    async with aiohttp.ClientSession() as session:
        print(
            f"Starting streaming download with max {max_concurrent} "
            f"concurrent downloads"
        )
        print("Files will be processed and deleted immediately after download!")

        # Create coroutines for all tasks
        coroutines = [
            download_and_process_station_file_with_semaphore(
                semaphore,
                session,
                task["url"],
                output_dir,
                task["filename"],
                main_parquet_file,
                overwrite,
            )
            for task in tasks
        ]

        # Run with async tqdm progress bar
        with tqdm(
            total=len(coroutines),
            desc="Downloading & Processing",
            unit="files",
            ncols=100,
        ) as pbar:

            async def process_with_progress(coro):
                """Wrapper to update progress bar after each completion."""
                try:
                    result = await coro
                    # Update counters
                    if result in progress_counters:
                        progress_counters[result] += 1

                    # Update progress bar description with current stats
                    pbar.set_postfix(
                        {
                            "Processed": progress_counters["processed"],
                            "Skipped": progress_counters["skipped"],
                            "Not Found": progress_counters["not_found"],
                            "Failed": progress_counters["failed"],
                        }
                    )
                    pbar.update(1)
                    return result
                except Exception as e:
                    progress_counters["errors"] += 1
                    pbar.set_postfix(
                        {
                            "Processed": progress_counters["processed"],
                            "Skipped": progress_counters["skipped"],
                            "Not Found": progress_counters["not_found"],
                            "Failed": progress_counters["failed"],
                            "Errors": progress_counters["errors"],
                        }
                    )
                    pbar.update(1)
                    return e

            # Execute all tasks with progress tracking
            await asyncio.gather(
                *[process_with_progress(coro) for coro in coroutines],
                return_exceptions=True,
            )

        # Final count results (including pre-filtered)
        total_processed = progress_counters["processed"]
        total_skipped = progress_counters["skipped"] + skipped_during_setup
        total_not_found = progress_counters["not_found"]
        total_failed = progress_counters["failed"]
        total_errors = progress_counters["errors"]

        # Flush any remaining batch data
        flush_batch_to_parquet(main_parquet_file)

        print("\nStreaming download and processing complete:")
        print(f"  Processed: {total_processed}")
        print(f"  Skipped (already exists): {total_skipped}")
        print(f"  Not found (404): {total_not_found}")
        print(f"  Failed: {total_failed}")
        print(f"  Errors: {total_errors}")
        print(f"  Total files checked: {len(station_df) * len(years)}")

        if Path(main_parquet_file).exists():
            try:
                final_df = pd.read_parquet(main_parquet_file)
                print("\nFinal dataset summary:")
                print(f"  Total rows: {len(final_df)}")
                print(f"  Unique stations: {final_df['station'].nunique()}")
                if "time" in final_df.columns:
                    print(
                        f"  Date range: {final_df['time'].min()} to "
                        f"{final_df['time'].max()}"
                    )
            except Exception as e:
                print(f"Error reading final dataset: {e}")


def main():
    """Main function for streaming GHCN-H download."""
    print("GHCN-H Streaming Download Script")
    print("===============================")
    print("This script downloads, processes, and deletes files immediately!")
    print("Files are checked for existing data and skipped if already processed.")
    print()

    # Define years to download
    years = [2020, 2021, 2022, 2023, 2024]

    # Set output directory for temporary files
    output_dir = "/tmp/ghcnh_temp/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Enable nested event loop for async operations
    nest_asyncio.apply()

    # Download and process using streaming approach
    print(f"Streaming data for years: {years}")
    asyncio.run(
        download_all_stations_streaming(
            years, output_dir, overwrite=False, max_concurrent=10
        )
    )

    # Clean up temp directory
    try:
        import shutil

        shutil.rmtree(output_dir)
        print(f"Cleaned up temporary directory: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temp directory: {e}")

    print("Streaming processing complete!")


if __name__ == "__main__":
    main()
