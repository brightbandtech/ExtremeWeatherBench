# %%
# !uv pip install numpy aiohttp asyncio nest_asyncio distributed tqdm xarray
# !uv pip install geopy h5py pyarrow
# !uv pip install git+https://github.com/brightbandtech/ExtremeWeatherBench.git
# !uv pip install git+https://github.com/fsspec/kerchunk
# !uv pip install ipywidgets
# !uv pip install cartopy


# %%
import asyncio
import dataclasses
from pathlib import Path
from typing import Optional

import aiohttp
import nest_asyncio
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from extremeweatherbench.regions import map_to_create_region


@dataclasses.dataclass
class BoundingBox:
    """Simple bounding box for backward compatibility."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def wind_speed_direction_to_uv(speed, direction):
    """
    Convert wind speed and direction to u and v components
    Args:
        speed: float, wind speed
        direction: float, wind direction in degrees (0-359)
    Returns:
        u: float, eastward wind component
        v: float, northward wind component
    """
    direction = np.deg2rad(direction)
    u = -speed * np.cos(direction)
    v = -speed * np.sin(direction)
    return u, v


def subset_stations_by_lat_lon_box(df, min_lat, max_lat, min_lon, max_lon):
    return df[
        (df["LATITUDE"] >= min_lat)
        & (df["LATITUDE"] <= max_lat)
        & (df["LONGITUDE"] >= min_lon)
        & (df["LONGITUDE"] <= max_lon)
    ]


def aggregate_to_hourly(data):
    """Aggregate rows that represent same hour to one row Order the rows from same hour
    by difference from the top of the hour, then use ffill at each hour to get the
    nearest valid values for each variable Specifically, For t/td, avoid combining two
    records from different rows together."""
    data["hour"] = data["DATE"].dt.round("h")
    # Sort data by difference from the top of the hour so that bfill can be applied
    # to give priority to the closer records
    data["hour_dist"] = (data["DATE"] - data["hour"]).dt.total_seconds().abs() // 60
    data = data.sort_values(["hour", "hour_dist"])

    if data["hour"].duplicated().any():
        # For same hour, fill NaNs at the first row in the order of difference from the
        # top of the hour
        data = data.groupby("hour").apply(
            lambda df: df.bfill().iloc[0], include_groups=False
        )
    data = data.reset_index(drop=True)
    data["time"] = data["DATE"].dt.round("h")
    return data


def location_translation(
    location: dict, degrees: Optional[float] = None
) -> BoundingBox:
    """Translate the location dictionary to a BoundingBox object."""
    # Handle new YAML format with type and parameters
    if "type" in location and "parameters" in location:
        region = map_to_create_region(location)
        if hasattr(region, "latitude_min"):
            # BoundingBoxRegion
            return BoundingBox(
                min_lat=region.latitude_min,
                max_lat=region.latitude_max,
                min_lon=region.longitude_min,
                max_lon=region.longitude_max,
            )
        elif hasattr(region, "latitude") and hasattr(region, "bounding_box_degrees"):
            # CenteredRegion
            if isinstance(region.bounding_box_degrees, tuple):
                lat_degrees, lon_degrees = region.bounding_box_degrees
            else:
                lat_degrees = lon_degrees = region.bounding_box_degrees

            min_lat = region.latitude - lat_degrees / 2
            max_lat = region.latitude + lat_degrees / 2
            min_lon = region.longitude - lon_degrees / 2
            max_lon = region.longitude + lon_degrees / 2

            return BoundingBox(
                min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon
            )

    # Handle legacy format
    if "latitude" in location and "longitude" in location:
        if degrees is None:
            raise ValueError("degrees must be provided if latitude is provided")
        min_lat = location["latitude"] - degrees / 2
        max_lat = location["latitude"] + degrees / 2
        min_lon = location["longitude"] - degrees / 2
        max_lon = location["longitude"] + degrees / 2
    elif (
        "lat_min" in location
        and "lat_max" in location
        and "lon_min" in location
        and "lon_max" in location
    ):
        min_lat = location["lat_min"]
        max_lat = location["lat_max"]
        min_lon = location["lon_min"]
        max_lon = location["lon_max"]
    else:
        raise ValueError(
            "location must contain new format (type/parameters) or legacy format "
            "(latitude/longitude or lat_min/lat_max/lon_min/lon_max)"
        )
    return BoundingBox(
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon
    )


async def download_file_async(session, station, url, output_dir, overwrite=False):
    async with session.get(url + f"{station}") as response:
        final_path = Path(output_dir) / f"{station}"
        if final_path.exists() and not overwrite:
            return True
        if response.status == 200 and not response.headers.get(
            "Content-Type", ""
        ).startswith("application/xml"):
            with open(final_path, "wb") as file:
                file.write(await response.read())
            return True
        return False


async def download_async(case_stations_dict, output_dir, overwrite=False):
    """Download the data from the source for the stations in station_list."""
    URL_DATA = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year"  # noqa: E501

    async with aiohttp.ClientSession() as session:
        tasks = []
        for k, v in case_stations_dict.items():
            directory_for_case = Path(output_dir) / str(v["year"])
            if not directory_for_case.exists():
                directory_for_case.mkdir(parents=True, exist_ok=True)

            url = URL_DATA + f"/{v['year']}/psv/"
            station_list = v["stations"]
            for station in station_list:
                tasks.append(
                    download_file_async(
                        session, station, url, directory_for_case, overwrite
                    )
                )

        _ = await asyncio.gather(*tasks)


def process_file(file, subset_station_dict):
    try:
        df = pd.read_csv(file, sep="|", low_memory=False)
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
        df = df[
            [
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
        ]
        df["DATE"] = pd.to_datetime(df["DATE"])
        # Extract station ID from file name
        station_id = df["STATION"].iloc[0] if not df.empty else None
        if "stations" in subset_station_dict and station_id in [
            s.split("_")[1] for s in subset_station_dict["stations"] if "_" in s
        ]:
            return df
    except pd.errors.EmptyDataError:
        pass
    return None


# Define a function to process each dataframe
def process_dataframe(df):
    return aggregate_to_hourly(df)


def main():
    """Main function to generate GHCN data for extreme weather events."""
    # Define column specifications based on character positions
    col_specs = [
        (0, 11),  # ID
        (13, 20),  # LATITUDE
        (22, 30),  # LONGITUDE
        (33, 37),  # ELEVATION
        (39, 40),  # STATE
        (46, 71),  # NAME
        (73, 75),  # GSN FLAG
        (77, 79),  # HCN/CRN FLAG
        (81, 85),  # WMO ID
    ]
    # ID, 1-11, Character
    # LATITUDE,13-20, Real
    # LONGITUDE, 22-30, Real
    # ELEVATION, 32-37, Real
    # STATE, 39-40, Character
    # NAME, 42-71, Character
    # GSN FLAG, 73-75, Character
    # HCN/CRN FLAG, 77-79, Character
    # WMO ID, 81-85, Character
    # Read the CSV file with fixed-width formatting
    station_df = pd.read_fwf(
        "/home/taylor/data/ghcnh-station-list.csv",
        colspecs=col_specs,
        names=[
            "ID",
            "LATITUDE",
            "LONGITUDE",
            "ELEVATION",
            "STATE",
            "NAME",
            "GSN_FLAG",
            "HCN_CRN_FLAG",
            "WMO_ID",
        ],
    )

    events_file_path = (
        "/home/taylor/code/ExtremeWeatherBench/src/extremeweatherbench/data/events.yaml"
    )
    all_results = {}
    with open(events_file_path, "r") as file:
        yaml_event_case = yaml.safe_load(file)

    for case in yaml_event_case["cases"]:
        # Handle new location format - no more bounding_box_degrees at top level
        bounding_box = location_translation(case["location"])
        min_lat = bounding_box.min_lat
        max_lat = bounding_box.max_lat
        min_lon = bounding_box.min_lon
        max_lon = bounding_box.max_lon

        if min_lon > max_lon:
            # Ensure max_lon is always the larger value and account for
            # cyclic nature of lon
            min_lon, max_lon = max_lon, min_lon
        # Filter stations within bounding box
        stations_in_box = subset_stations_by_lat_lon_box(
            station_df, min_lat, max_lat, min_lon, max_lon
        )

        # Store results - use case_id_number instead of id
        all_results[case["case_id_number"]] = {
            "year": case["start_date"].year
            if case["start_date"].year == case["end_date"].year
            else np.nan,
            "num_stations": len(stations_in_box),
            "stations": (
                "GHCNh_"
                + stations_in_box["ID"]
                + "_"
                + str(case["start_date"].year)
                + ".psv"
            ).tolist(),
        }

    # %%
    nest_asyncio.apply()

    output_dir = "<path_to_data>/ghcnh/"

    # To run the async function; note this was originally
    # await download_async(all_results, output_dir, overwrite=False)
    # in a jupyter notebook; code may need to be ported into one to work

    asyncio.run(download_async(all_results, output_dir, overwrite=False))

    # %%
    base_dir = Path("<path_to_data>/ghcnh")
    all_parquet_files = list(base_dir.glob("**/*.psv"))
    datasets_df_new = pd.DataFrame()
    df_list = []

    # Create list of (file, subset_station_dict) tuples for parallel processing
    file_tasks = []
    for case in yaml_event_case["cases"]:
        subset_station_dict = all_results[case["case_id_number"]]
        for file in all_parquet_files:
            if file.name in subset_station_dict["stations"]:
                file_tasks.append((file, subset_station_dict))

    # Process files in parallel
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_file)(file, subset_station_dict)
        for file, subset_station_dict in tqdm(file_tasks)
    )

    # Filter out None results and extend df_list
    df_list.extend([df for df in results if df is not None])

    # Set up parallel processing with progress bar
    n_jobs = -1  # Use all available cores
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_dataframe)(df) for df in df_list
    )

    # Concatenate the results
    datasets_df_new = pd.concat(results, ignore_index=True)
    datasets_df_new.to_parquet("interim_ghcnh_20250630.parq")
    datasets_df_new = pd.read_parquet("interim_ghcnh.parq")
    datasets_df_new
    # Extract 'member0' values from dictionaries in 'relative_humidity' column
    mask = datasets_df_new["relative_humidity"].apply(lambda x: isinstance(x, dict))
    datasets_df_new.loc[mask, "relative_humidity"] = datasets_df_new.loc[
        mask, "relative_humidity"
    ].apply(lambda x: x.get("member0"))

    datasets_df_new = datasets_df_new.rename(
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
    ).drop(["DATE", "hour"], axis=1)

    datasets_df_new.to_parquet("ghcnh_20250630.parq")
    # # for parquets
    # datasets_df_new.rename(columns={'temperature': 'surface_air_temperature',
    # 'dew_point_temperature': 'surface_dew_point',
    # 'wind_speed': 'surface_wind_speed',
    # 'wind_direction': 'surface_wind_from_direction',
    # 'station_level_pressure': 'surface_air_pressure',
    # 'sea_level_pressure': 'air_pressure_at_mean_sea_level',
    # 'c': 'cloud_area_fraction',
    # 'relative_humidity': 'surface_relative_humidity',
    # 'precipitation': 'accumulated_1_hour_precipitation',}
    # ).drop(['DATE','hour'],axis=1).to_parquet('ghcnh.parq')

    # for psvs
    datasets_df_new.rename(
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
            "case_id": "id",
        }
    ).drop(["DATE", "hour"], axis=1).to_parquet("ghcnh.parq")


if __name__ == "__main__":
    main()
