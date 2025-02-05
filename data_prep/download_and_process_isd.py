"""
Downloads ISD data for each case and saves all data in a parquet file.
WeatherReal quality_control code (https://github.com/microsoft/WeatherReal-Benchmark)
and methodology utilized for quality control and downloading.

This script is used for parsing ISD files and converting them to hourly data
1. Parse corresponding columns of each variable
2. Aggregate / reorganize columns if needed
3. Simple unit conversion
4. Aggregate to hourly data. Rows that represent same hour are merged by rules
5. Save the processed data to a new parquet file
"""

import pandas as pd
import yaml
from extremeweatherbench import utils
import logging
import numpy as np
from pathlib import Path
import aiohttp
import asyncio
from distributed import Client
import os
import dask
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import xarray as xr
from geopy.distance import great_circle as geodist

# Quality code for erroneous values flagged by ISD
# To collect as much data as possible and avoid some values flagged by unknown codes being excluded,
# We choose the strategy "only values marked with known error tags will be rejected"
# instead of "only values marked with known correct tags will be accepted"
ERRONEOUS_FLAGS = ["3", "7"]
URL_DATA = "https://www.ncei.noaa.gov/data/global-hourly/access/"

# URL of the official ISD metadata file
URL_ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
LOCAL_ISD_HISTORY = "/Users/taylor/Downloads/isd-history.txt"
# Disable the logging from urllib3
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def subset_stations_by_lat_lon_box(df, min_lat, max_lat, min_lon, max_lon):
    return df[
        (df["LAT"] >= min_lat)
        & (df["LAT"] <= max_lat)
        & (df["LON"] >= min_lon)
        & (df["LON"] <= max_lon)
    ]


async def download_file_async(session, station, url, output_dir, overwrite=False):
    async with session.get(url + f"{station}") as response:
        final_path = Path(output_dir) / f"{station}"
        if final_path.exists() and not overwrite:
            return True
        if response.status == 200:
            with open(final_path, "wb") as file:
                file.write(await response.read())
            return True
        return False


async def download_ISD_async(case_stations_dict, output_dir, overwrite=False):
    """
    Download the data from the source for the stations in station_list.
    """
    logger.info("Start downloading")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for k, v in case_stations_dict.items():
            directory_for_case = Path(output_dir) / str(v["year"])
            if not directory_for_case.exists():
                directory_for_case.mkdir(parents=True, exist_ok=True)
            url = URL_DATA + f"/{v['year']}/"
            station_list = v["stations"]
            for station in atqdm(station_list):
                tasks.append(
                    download_file_async(
                        session, station, url, directory_for_case, overwrite
                    )
                )

        results = await asyncio.gather(*tasks)
        successful_downloads = sum(results)
        logger.info(
            f"Successfully downloaded {successful_downloads} out of {len(tasks)} files"
        )


def parse_temperature_col(data):
    """
    Process temperature and dew point temperature columns
    TMP/DEW column format: -0100,1
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in Celsius
    """
    if "TMP" in data.columns and data["TMP"].notnull().any():
        temp_data = data.copy()
        temp_data[["t", "t_qc"]] = temp_data["TMP"].str.split(",", expand=True)
        temp_data.loc[:, "t"] = temp_data["t"].where(temp_data["t"] != "+9999", pd.NA)
        temp_data.loc[:, "t"] = temp_data["t"].where(
            ~temp_data["t_qc"].isin(ERRONEOUS_FLAGS), pd.NA
        )
        # Scaling factor: 10
        temp_data.loc[:, "t"] = temp_data["t"].astype("Float32") / 10
        data = temp_data
    if "DEW" in data.columns and data["DEW"].notnull().any():
        temp_data = data.copy()
        temp_data[["td", "td_qc"]] = temp_data["DEW"].str.split(",", expand=True)
        temp_data.loc[:, "td"] = temp_data["td"].where(
            temp_data["td"] != "+9999", pd.NA
        )
        temp_data.loc[:, "td"] = temp_data["td"].where(
            ~temp_data["td_qc"].isin(ERRONEOUS_FLAGS), pd.NA
        )
        # Scaling factor: 10
        temp_data.loc[:, "td"] = temp_data["td"].astype("Float32") / 10
        data = temp_data
    data = data.drop(columns=["TMP", "DEW", "t_qc", "td_qc"], errors="ignore")
    return data


def parse_wind_col(data):
    """
    Process wind speed and direction column
    WND column format: 267,1,N,0142,1
    N indicates normal (other values include Beaufort, Calm, etc.). Not used currently
    Steps:
    1. Set values flagged as erroneous/missing to NaN
       Note that if one of ws or wd is missing, both are set to NaN
       Exception: If ws is 0 and wd is missing, wd is set to 0 (calm)
    2. Convert wd to integer and ws to float in m/s
    """
    if "WND" in data.columns and data["WND"].notnull().any():
        data[["wd", "wd_qc", "wt", "ws", "ws_qc"]] = data["WND"].str.split(
            ",", expand=True
        )
        # First, set wd to 0 if ws is valid 0 and wd is missing
        calm = (
            (data["ws"] == "0000")
            & (~data["ws_qc"].isin(ERRONEOUS_FLAGS))
            & (data["wd"] == "999")
        )
        data.loc[calm, "wd"] = "000"
        data.loc[calm, "wd_qc"] = "1"
        # After that, if one of ws or wd is missing/erroneous, both are set to NaN
        non_missing = (data["wd"] != "999") & (data["ws"] != "9999")
        non_error = (~data["wd_qc"].isin(ERRONEOUS_FLAGS)) & (
            ~data["ws_qc"].isin(ERRONEOUS_FLAGS)
        )
        valid = non_missing & non_error
        data["wd"] = data["wd"].where(valid, pd.NA)
        data["ws"] = data["ws"].where(valid, pd.NA)
        data["wd"] = data["wd"].astype("Int16")
        # Scaling factor: 10
        data["ws"] = data["ws"].astype("Float32") / 10
    data = data.drop(columns=["WND", "wd_qc", "wt", "ws_qc"], errors="ignore")
    return data


def parse_cloud_col(data):
    """
    Process total cloud cover column
    All known columns including GA1-6, GD1-6, GF1 and GG1-6 are parsed and Maximum value of them is selected
    1. GA1-6 column format: 07,1,+00800,1,06,1
       The 1st and 2nd items are c and its quality
    2. GD1-6 column format: 3,99,1,+05182,9,9
       The 1st item is cloud cover in 0-4 and is converted to octas by multiplying 2
       The 2st and 3nd items are c in octas and its quality
    3. GF1 column format: 07,99,1,07,1,99,9,01000,1,99,9,99,9
       The 1st and 3rd items are total coverage and its quality
    4. GG1-6 column format: 01,1,01200,1,06,1,99,9
       The 1st and 2nd items are c and its quality
    Cloud/sky-condition related data is very complex and worth further investigation
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Select the maximum value of all columns
    """
    num = 0
    for group in ["GA", "GG"]:
        for col in [f"{group}{i}" for i in range(1, 7)]:
            if col in data.columns and data[col].notnull().any():
                data[[f"c{num}", "c_qc", "remain"]] = data[col].str.split(
                    ",", n=2, expand=True
                )
                # 99 will be removed later
                data[f"c{num}"] = data[f"c{num}"].where(
                    ~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA
                )
                data[f"c{num}"] = data[f"c{num}"].astype("Int16")
                num += 1
            else:
                break
    for col in [f"GD{i}" for i in range(1, 7)]:
        if col in data.columns and data[col].notnull().any():
            data[[f"c{num}", f"c{num + 1}", "c_qc", "remain"]] = data[col].str.split(
                ",", n=3, expand=True
            )
            c_cols = [f"c{num}", f"c{num + 1}"]
            data[c_cols] = data[c_cols].where(
                ~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA
            )
            data[c_cols] = data[c_cols].astype("Int16")
            # The first item is 5-level cloud cover and is converted to octas by multiplying 2
            data[f"c{num}"] = data[f"c{num}"] * 2
            num += 2
        else:
            break
    if "GF1" in data.columns and data["GF1"].notnull().any():
        data[[f"c{num}", "opa", "c_qc", "remain"]] = data["GF1"].str.split(
            ",", n=3, expand=True
        )
        data[f"c{num}"] = data[f"c{num}"].where(
            ~data["c_qc"].isin(ERRONEOUS_FLAGS), pd.NA
        )
        data[f"c{num}"] = data[f"c{num}"].astype("Int16")
        num += 1
    c_cols = [f"c{i}" for i in range(num)]
    # Mask all values larger than 8 to NaN to avoid overwriting the correct values
    data[c_cols] = data[c_cols].where(data[c_cols] <= 8, pd.NA)
    # Maximum value of all columns is selected to represent the total cloud cover
    data["c"] = data[c_cols].max(axis=1)
    data = data.drop(
        columns=[
            "GF1",
            *[f"GA{i}" for i in range(1, 7)],
            *[f"GG{i}" for i in range(1, 7)],
            *[f"GD{i}" for i in range(1, 7)],
            *[f"c{i}" for i in range(num)],
            "c_5",
            "opa",
            "c_qc",
            "remain",
        ],
        errors="ignore",
    )
    return data


def parse_surface_pressure_col(data):
    """
    Process surface pressure (station-level pressure) column
    Currently MA1 column is used. Column format: 99999,9,09713,1
    The 3rd and 4th items are station pressure and its quality
    The 1st and 2nd items are altimeter setting and its quality which are not used currently
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in hPa
    """
    if "MA1" in data.columns and data["MA1"].notnull().any():
        data[["MA1_remain", "sp", "sp_qc"]] = data["MA1"].str.rsplit(
            ",", n=2, expand=True
        )
        data["sp"] = data["sp"].where(data["sp"] != "99999", pd.NA)
        data["sp"] = data["sp"].where(~data["sp_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        # Scaling factor: 10
        data["sp"] = data["sp"].astype("Float32") / 10
    data = data.drop(columns=["MA1", "MA1_remain", "sp_qc"], errors="ignore")
    return data


def parse_sea_level_pressure_col(data):
    """
    Process mean sea level pressure column
    MSL Column format: 09725,1
    Steps:
    1. Set values flagged as erroneous/missing to NaN
    2. Convert to float in hPa
    """
    if "SLP" in data.columns and data["SLP"].notnull().any():
        data[["msl", "msl_qc"]] = data["SLP"].str.rsplit(",", expand=True)
        data["msl"] = data["msl"].where(data["msl"] != "99999", pd.NA)
        data["msl"] = data["msl"].where(~data["msl_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        # Scaling factor: 10
        data["msl"] = data["msl"].astype("Float32") / 10
    data = data.drop(columns=["SLP", "msl_qc"], errors="ignore")
    return data


def parse_single_precipitation_col(data, col):
    """
    Parse one of the precipitation columns AA1-4
    """
    if data[col].isnull().all():
        return pd.DataFrame()
    datacol = data[[col]].copy()
    # Split the column to get the period first
    datacol[["period", f"{col}_remain"]] = datacol[col].str.split(",", n=1, expand=True)
    # Remove weird periods to avoid unexpected errors
    datacol = datacol[datacol["period"].isin(["01", "03", "06", "12", "24"])]
    if len(datacol) == 0:
        return pd.DataFrame()
    # Set the period as index and unstack so that different periods are converted to different columns
    datacol = datacol.set_index("period", append=True)[f"{col}_remain"]
    datacol = datacol.unstack("period")
    # Rename the columns according to the period, e.g., 03 -> ra3
    datacol.columns = [f"ra{item.lstrip('0')}" for item in datacol.columns]
    # Further split the remaining sections
    for var in datacol.columns:
        datacol[[var, f"{var}_cond", f"{var}_qc"]] = datacol[var].str.split(
            ",", expand=True
        )
    return datacol


def parse_precipitation_col(data):
    """
    Process precipitation columns
    Currently AA1-4 columns are used. Column format: 24,0073,3,1
    The items are period, depth, condition, quality. Condition is not used currently
    It is more complex than other variables as values during different periods are stored in same columns
    It is needed to separate them to different columns
    Steps:
    1. Separate and recombine columns by period
    2. Set values flagged as erroneous/missing to NaN
    3. Convert to float in mm
    """
    for col in ["AA1", "AA2", "AA3", "AA4"]:
        if col in data.columns:
            datacol = parse_single_precipitation_col(data, col)
            # Same variable (e.g., ra24) may be stored in different original columns
            # Combine_first so that same variables can be merged to the same columns
            data = data.combine_first(datacol)
        else:
            # Assuming that the remaining columns are also not present
            break
    # Quality status treated as valid records. 3/7 indicates erroneous value
    for col in [
        item for item in data.columns if item.startswith("ra") and item[2:].isdigit()
    ]:
        data[col] = data[col].where(data[col] != "9999", pd.NA)
        data[col] = data[col].where(~data[f"{col}_qc"].isin(ERRONEOUS_FLAGS), pd.NA)
        data[col] = data[col].astype("Float32") / 10
        data = data.drop(columns=[f"{col}_cond", f"{col}_qc"])
    data = data.drop(columns=["AA1", "AA2", "AA3", "AA4"], errors="ignore")
    return data


def parse_single_file(fpath, fpath_last_year):
    """
    Parse columns of each variable in a single ISD file
    """
    # Gxn for cloud cover, MA1 for surface pressure, AAn for precipitation
    cols_var = [
        "TMP",
        "DEW",
        "WND",
        "SLP",
        "MA1",
        "AA1",
        "AA2",
        "AA3",
        "AA4",
        "GF1",
        *[f"GA{i}" for i in range(1, 7)],
        *[f"GD{i}" for i in range(1, 7)],
        *[f"GG{i}" for i in range(1, 7)],
    ]
    cols = ["DATE"] + list(cols_var)

    def _load_csv(fpath):
        return pd.read_csv(
            fpath,
            parse_dates=["DATE"],
            usecols=lambda c: c in set(cols),
            low_memory=False,
        )

    data = _load_csv(fpath)
    if fpath_last_year is not None and os.path.exists(fpath_last_year):
        data_last_year = _load_csv(fpath_last_year)
        # Load the last day of the last year for better hourly aggregation
        data_last_year = data_last_year.loc[
            (data_last_year["DATE"].dt.month == 12)
            & (data_last_year["DATE"].dt.day == 31)
        ]
        data = pd.concat([data_last_year, data], ignore_index=True)
    data = data[[item for item in cols if item in data.columns]]

    data = parse_temperature_col(data)
    data = parse_wind_col(data)
    data = parse_cloud_col(data)
    data = parse_surface_pressure_col(data)
    data = parse_sea_level_pressure_col(data)
    data = parse_precipitation_col(data)

    data = data.rename(columns={"DATE": "time"})
    value_cols = [col for col in data.columns if col != "time"]
    # drop all-NaN rows
    data = (
        data[["time"] + value_cols]
        .sort_values("time")
        .dropna(how="all", subset=value_cols)
    )
    return data


def aggregate_to_hourly(data):
    """
    Aggregate rows that represent same hour to one row
    Order the rows from same hour by difference from the top of the hour,
    then use ffill at each hour to get the nearest valid values for each variable
    Specifically, For t/td, avoid combining two records from different rows together
    """
    data["hour"] = data["time"].dt.round("h")
    # Sort data by difference from the top of the hour so that bfill can be applied
    # to give priority to the closer records
    data["hour_dist"] = (data["time"] - data["hour"]).dt.total_seconds().abs() // 60
    data = data.sort_values(["hour", "hour_dist"])

    if data["hour"].duplicated().any():
        # Consruct a new column of (t, td) tuples. Values are not NaN only when both of them are valid
        data["t_td"] = data.apply(
            lambda row: (row["t"], row["td"])
            if row[["t", "td"]].notnull().all()
            else pd.NA,
            axis=1,
        )
        # For same hour, fill NaNs at the first row in the order of difference from the top of the hour
        data = data.groupby("hour").apply(
            lambda df: df.bfill().iloc[0], include_groups=False
        )

        # 1st priority: for hours that has both valid t and td originally (decided by t_td),
        # fill values to t_new and td_new
        # Specifically, for corner cases that all t_td is NaN, we need to convert pd.NA to (pd.NA, pd.NA)
        # so that to_list() will not raise an error
        data["t_td"] = data["t_td"].apply(
            lambda item: (pd.NA, pd.NA) if pd.isna(item) else item
        )
        data[["t_new", "td_new"]] = pd.DataFrame(
            data["t_td"].to_list(), index=data.index
        )
        # 2nd priority: Remaining hours can only provide at most one of t and td. Try to fill t first
        rows_to_fill = data[["t_new", "td_new"]].isnull().all(axis=1)
        data.loc[rows_to_fill, "t_new"] = data.loc[rows_to_fill, "t"]
        # 3nd priority: Remaining hours has no t during time window. Try to fill td
        rows_to_fill = data[["t_new", "td_new"]].isnull().all(axis=1)
        data.loc[rows_to_fill, "td_new"] = data.loc[rows_to_fill, "td"]

        data = data.drop(columns=["t", "td", "t_td"]).rename(
            columns={"t_new": "t", "td_new": "td"}
        )

    data = data.reset_index(drop=True)
    data["time"] = data["time"].dt.round("h")
    return data


def post_process(data, year):
    """
    Some post-processing steps after aggregation
    """
    data = data.set_index("time")
    sorted_ra_columns = sorted(
        [col for col in data.columns if col.startswith("ra")], key=lambda x: int(x[2:])
    )
    other_columns = [
        item
        for item in ["t", "td", "ws", "wd", "sp", "msl", "c"]
        if item in data.columns
    ]
    data = data[other_columns + sorted_ra_columns]
    data = data[f"{year}-01-01" : f"{year}-12-31"]  # noqa: E203
    return data


def load_metadata(station_list):
    """
    Load the metadata of ISD data
    """
    meta = pd.read_fwf(
        LOCAL_ISD_HISTORY,
        skiprows=20,
        usecols=[
            "USAF",
            "WBAN",
            "STATION NAME",
            "CTRY",
            "CALL",
            "LAT",
            "LON",
            "ELEV(M)",
        ],
        dtype={"USAF": str, "WBAN": str},
    )
    # Drop rows with missing values. These stations are considered not trustworthy
    meta = meta.dropna(
        how="any", subset=["LAT", "LON", "STATION NAME", "ELEV(M)", "CTRY"]
    )
    meta["STATION"] = meta["USAF"] + meta["WBAN"]
    meta = meta[
        ["STATION", "CALL", "STATION NAME", "CTRY", "LAT", "LON", "ELEV(M)"]
    ].set_index("STATION", drop=True)
    return meta[meta.index.isin(station_list)]


def calc_distance_similarity(latlon1, latlon2, scale_dist=25):
    distance = geodist(latlon1, latlon2).kilometers
    similarity = np.exp(-distance / scale_dist)
    return similarity


def calc_elevation_similarity(elevation1, elevation2, scale_elev=100):
    similarity = np.exp(-abs(elevation1 - elevation2) / scale_elev)
    return similarity


def calc_id_similarity(ids1, ids2):
    """
    Compare the USAF/WBAN/CALL IDs of two stations
    """
    usaf1, wban1, call1, ctry1 = ids1
    usaf2, wban2, call2, ctry2 = ids2
    if usaf1 != "999999" and usaf1 == usaf2:
        return 1
    if wban1 != "99999" and wban1 == wban2:
        return 1
    if call1 == call2:
        return 1
    # A special case for the CALL ID, e.g., KAIO and AIO are the same stations
    if isinstance(call1, str) and len(call1) == 3 and ("K" + call1) == call2:
        return 1
    if isinstance(call2, str) and len(call2) == 3 and ("K" + call2) == call1:
        return 1
    # For a special case in Germany, 09xxxx and 10xxxx are the same stations
    # See https://gi.copernicus.org/articles/5/473/2016/gi-5-473-2016.html
    if (
        usaf1.startswith("09")
        and usaf2.startswith("10")
        and usaf1[2:] == usaf2[2:]
        and ctry1 == ctry2 == "DE"
    ):
        return 1
    return 0


def calc_name_similarity(name1, name2):
    """
    Jaccard Index for calculating name similarity
    """
    set1 = set(name1)
    set2 = set(name2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def post_process_similarity(similarity, ids):
    """
    Post-process the similarity matrix
    """
    # Fill the lower triangle of the matrix
    similarity = similarity + similarity.T
    # Set the diagonal (self-similarity) to 1
    np.fill_diagonal(similarity, 1)
    similarity = xr.DataArray(
        similarity,
        dims=["station1", "station2"],
        coords={"station1": ids, "station2": ids},
    )
    return similarity


def calc_similarity(meta, scale_dist=25, scale_elev=100):
    """
    Calculate pairwise similarity between stations
    1. Distance similarity: great circle distance between two stations using an exponential decay
    2. Elevation similarity: absolute difference of elevation between two stations using an exponential decay
    3. ID similarity: whether the USAF/WBAN/CALL IDs of two stations are the same
    """
    latlon = meta[["LAT", "LON"]].apply(tuple, axis=1).values
    elev = meta["ELEV(M)"].values
    usaf = meta.index.str[:6].values
    wban = meta.index.str[6:].values
    name = meta["STATION NAME"].values
    ids = list(zip(usaf, wban, meta["CALL"].values, meta["CTRY"].values))
    num = len(meta)
    dist_similarity = np.zeros((num, num))
    elev_similarity = np.zeros((num, num))
    id_similarity = np.zeros((num, num))
    name_similarity = np.zeros((num, num))
    for idx1 in tqdm(range(num - 1), desc="Calculating similarity"):
        for idx2 in range(idx1 + 1, num):
            dist_similarity[idx1, idx2] = calc_distance_similarity(
                latlon[idx1], latlon[idx2], scale_dist
            )
            elev_similarity[idx1, idx2] = calc_elevation_similarity(
                elev[idx1], elev[idx2], scale_elev
            )
            id_similarity[idx1, idx2] = calc_id_similarity(ids[idx1], ids[idx2])
            name_similarity[idx1, idx2] = calc_name_similarity(name[idx1], name[idx2])
    dist_similarity = post_process_similarity(dist_similarity, meta.index.values)
    elev_similarity = post_process_similarity(elev_similarity, meta.index.values)
    id_similarity = post_process_similarity(id_similarity, meta.index.values)
    name_similarity = post_process_similarity(name_similarity, meta.index.values)
    similarity = xr.merge(
        [
            dist_similarity.rename("dist"),
            elev_similarity.rename("elev"),
            id_similarity.rename("id"),
            name_similarity.rename("name"),
        ]
    )
    return similarity


def load_raw_data(data_dir, station_list):
    """
    Load raw hourly ISD data in csv files
    """
    data = []
    for stn in tqdm(station_list, desc="Loading data"):
        df = pd.read_csv(
            os.path.join(data_dir, f"{stn}.csv"), index_col="time", parse_dates=["time"]
        )
        df["station"] = stn
        df = df.set_index("station", append=True)
        data.append(df.to_xarray())
    data = xr.concat(data, dim="station")
    data["time"] = pd.to_datetime(data["time"])
    return data


def calc_meta_similarity(similarity):
    """
    Calculate the metadata similarity according to horizontal distance, elevation, and name similarity
    """
    meta_simi = (
        similarity["dist"] * 9 + similarity["elev"] * 1 + similarity["name"] * 5
    ) / 15
    # meta_simi is set to 1 if IDs are the same,
    # or it is set to a weighted sum of distance, elevation, and name similarity
    meta_simi = np.maximum(meta_simi, similarity["id"])
    # set the diagonal and lower triangle to NaN to avoid duplicated pairs
    rows, cols = np.indices(meta_simi.shape)
    meta_simi.values[rows >= cols] = np.nan
    return meta_simi


def need_merge(da, stn_source, stn_target, threshold=0.7):
    """
    Distinguish whether two stations need to be merged based on the similarity of their data
    It is possible that one of them only has few data points
    In this case, it can be treated as removing low-quality stations
    """
    ts1 = da.sel(station=stn_source)
    ts2 = da.sel(station=stn_target)
    diff = np.abs(ts1 - ts2)
    if (ts1.dropna(dim="time") % 1 < 1e-3).all() or (
        ts2.dropna(dim="time") % 1 < 1e-3
    ).all():
        max_diff = 0.5
    else:
        max_diff = 0.1
    data_simi = (diff <= max_diff).sum() / diff.notnull().sum()
    return data_simi.item() >= threshold


def merge_pairs(ds1, ds2):
    """
    Merge two stations. Each of the two ds should have only one station
    If there are only one variable, fill the missing values in ds1 with ds2
    If there are more than one variables, to ensure that all variables are from the same station,
    for each timestep, the ds with more valid variables will be selected
    """
    if len(ds1.data_vars) == 1:
        return ds1.fillna(ds2)
    da1 = ds1.to_array()
    da2 = ds2.to_array()
    mask = da1.count(dim="variable") >= da2.count(dim="variable")
    return xr.where(mask, da1, da2).to_dataset(dim="variable")


def merge_stations(
    ds, meta_simi, main_var, appendant_var=None, meta_simi_th=0.35, data_simi_th=0.7
):
    """
    For ds, merge stations based on metadata similarity and data similarity
    """
    result = []
    # Flags to avoid duplications
    is_merged = xr.DataArray(
        np.full(ds["station"].size, False),
        dims=["station"],
        coords={"station": ds["station"].values},
    )
    for station in tqdm(ds["station"].values, desc=f"Merging {main_var}"):
        if is_merged.sel(station=station).item():
            continue
        # Station list to be merged
        merged_stations = [station]
        # Candidates that pass the metadata similarity threshold
        candidates = meta_simi["station2"][
            meta_simi.sel(station1=station) >= meta_simi_th
        ].values
        # Stack to store the station pairs to be checked
        stack = [(station, item) for item in candidates]
        # Search for all stations that need to be merged
        # If A and B should be merged, and B and C should be merged, then all of them are merged together
        while stack:
            stn_source, stn_target = stack.pop()
            if stn_target in merged_stations:
                continue
            if need_merge(ds[main_var], stn_source, stn_target, threshold=data_simi_th):
                is_merged.loc[stn_target] = True
                merged_stations.append(stn_target)
                candidates = meta_simi["station2"][
                    meta_simi.sel(station1=stn_target) >= meta_simi_th
                ].values
                stack.extend([(stn_target, item) for item in candidates])
        # Merge stations according to the number of valid data points
        num_valid = ds[main_var].sel(station=merged_stations).notnull().sum(dim="time")
        sorted_stns = num_valid["station"].sortby(num_valid).values
        variables = (
            [main_var] + appendant_var if appendant_var is not None else [main_var]
        )
        stn_data = ds[variables].sel(station=sorted_stns[0])
        for target_stn in sorted_stns[1:]:
            stn_data = merge_pairs(stn_data, ds[variables].sel(station=target_stn))
        stn_data = stn_data.assign_coords(station=station)
        result.append(stn_data)
    result = xr.concat(result, dim="station")
    return result


def merge_all_variables(data, meta_simi, meta_simi_th=0.35, data_simi_th=0.7):
    """
    Merge stations for each variable
    The key is the main variable used to compare, and the value is the list of appendant variables
    """
    variables = {
        "t": ["td"],
        "ws": ["wd"],
        "sp": [],
        "msl": [],
        "c": [],
        "ra1": ["ra3", "ra6", "ra12", "ra24"],
    }
    merged = []
    for var, app_vars in variables.items():
        if var not in data.data_vars:
            continue
        else:
            ret = merge_stations(
                data,
                meta_simi,
                var,
                app_vars,
                meta_simi_th=meta_simi_th,
                data_simi_th=data_simi_th,
            )
            merged.append(ret)
    merged = xr.merge(merged).dropna(dim="station", how="all")
    return merged


def assign_meta_coords(ds, meta):
    meta = meta.loc[ds["station"].values]
    ds = ds.assign_coords(
        call=("station", meta["CALL"].values),
        name=("station", meta["STATION NAME"].values),
        lat=("station", meta["LAT"].values.astype(np.float32)),
        lon=("station", meta["LON"].values.astype(np.float32)),
        elev=("station", meta["ELEV(M)"].values.astype(np.float32)),
    )
    return ds


def pipeline(input_path, output_dir, year, overwrite=True):
    """
    The pipeline function for processing a single ISD file
    """
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    if not overwrite and os.path.exists(output_path):
        return
    input_dir = os.path.dirname(input_path)
    if input_dir.endswith(str(year)):
        input_path_last_year = os.path.join(
            input_dir[:-4] + str(year - 1), os.path.basename(input_path)
        )
    else:
        input_path_last_year = None
    data = parse_single_file(input_path, input_path_last_year)
    data = aggregate_to_hourly(data)
    data = post_process(data, year)
    data.astype("Float32").to_csv(output_path, float_format="%.1f")


def run_isd_generation(
    output_dir: str, events_file_path: str, use_dask: bool = True, n_workers=4
):
    """Runs the generation of ISD to parquet files for EWB cases.

    Args:
        output_dir: directory to save the final dataframe
        events_file_path: path to the events yaml file
        use_dask: whether to use dask for parallel processing
        n_workers: number of workers for dask client

    Returns a pandas dataframe of data for all stations within the bounding box
    of the case.
    """

    _ = Client(n_workers=n_workers)
    df = pd.read_csv(URL_ISD_HISTORY, dtype={"USAF": str, "WBAN": str})
    df["id"] = df["USAF"] + df["WBAN"]
    df["BEGIN"] = pd.to_datetime(df["BEGIN"], format="%Y%m%d")
    df["END"] = pd.to_datetime(df["END"], format="%Y%m%d")
    all_results = {}
    with open(events_file_path, "r") as file:
        yaml_event_case = yaml.safe_load(file)
    for k, v in yaml_event_case.items():
        if k == "cases":
            for individual_case in v:
                if "location" in individual_case:
                    individual_case["location"] = utils.Location(
                        **individual_case["location"]
                    )
    for case in yaml_event_case["cases"]:
        min_lat, max_lat, min_lon, max_lon = utils.get_bounding_corners(
            case["location"], case["bounding_box_km"], convert_to_360=False
        )

        # Filter stations within bounding box
        stations_in_box = subset_stations_by_lat_lon_box(
            df, min_lat, max_lat, min_lon, max_lon
        )

        # Filter stations with data in the specified date range
        stations_in_time_range = stations_in_box[
            (stations_in_box["BEGIN"] <= case["end_date"])
            & (stations_in_box["END"] >= case["start_date"])
        ]

        # Store results
        all_results[case["id"]] = {
            "year": case["start_date"].year
            if case["start_date"].year == case["end_date"].year
            else np.nan,
            "num_stations": len(stations_in_time_range),
            "stations": (stations_in_time_range["id"] + ".csv").tolist(),
        }

    # To run the async function
    def run_async():
        async def asyncfunc():
            await download_ISD_async(all_results, output_dir, overwrite=False)

        asyncio.run(asyncfunc())

    run_async()

    second_step_output_dir = f"{output_dir}/ISD_second_stage/"
    os.makedirs(second_step_output_dir, exist_ok=True)
    third_step_output_dir = f"{output_dir}/ISD_third_stage/"
    os.makedirs(third_step_output_dir, exist_ok=True)
    results_dict = {}
    for year in [n for n in os.listdir(output_dir) if "." not in n if "ISD" not in n]:
        second_step_output_dir_year = os.path.join(second_step_output_dir, str(year))
        os.makedirs(second_step_output_dir_year, exist_ok=True)

        third_step_output_dir_year = os.path.join(third_step_output_dir, str(year))
        os.makedirs(third_step_output_dir_year, exist_ok=True)
        output_dir_by_year = os.path.join(output_dir, str(year))
        input_list = [
            os.path.join(output_dir_by_year, f)
            for f in os.listdir(output_dir_by_year)
            if f.endswith(".csv")
        ]
        if use_dask:
            tasks_fs = [
                dask.delayed(pipeline)(
                    fpath,
                    output_dir=second_step_output_dir_year,
                    year=int(year),
                    overwrite=False,
                )
                for fpath in input_list
            ]
            _ = list(dask.compute(*tasks_fs))
        else:
            for fpath in input_list:
                pipeline(
                    fpath,
                    output_dir=second_step_output_dir_year,
                    year=int(year),
                    overwrite=False,
                )
        station_list = [
            item.rsplit(".", 1)[0]
            for item in os.listdir(second_step_output_dir_year)
            if item.endswith(".csv")
        ]
        meta = load_metadata(station_list)
        logger.info("calculating similarity")
        similarity = calc_similarity(meta)
        similarity_path = os.path.join(
            third_step_output_dir_year, f"similarity_{25}km_{100}m.nc"
        )
        similarity.astype("float32").to_netcdf(similarity_path)
        logger.info(f"Saved similarity to {similarity_path}")
        data = load_raw_data(second_step_output_dir_year, meta.index.values)
        # some stations have no data, they have already been removed in load_raw_data
        meta = meta.loc[data["station"].values]

        similarity = similarity.sel(
            station1=meta.index.values, station2=meta.index.values
        )
        meta_simi = calc_meta_similarity(similarity)
        merged = merge_all_variables(
            data, meta_simi, meta_simi_th=0.35, data_simi_th=0.7
        )
        # Save all metadata information in the NetCDF file
        merged = assign_meta_coords(merged, meta)
        results_dict[merged["time.year"][0].item()] = merged

    dataset_list = []
    for case in yaml_event_case["cases"]:
        year = case["start_date"].year
        case_ds = results_dict[year]
        min_lat, max_lat, min_lon, max_lon = utils.get_bounding_corners(
            case["location"], case["bounding_box_km"], convert_to_360=False
        )
        station_indices = np.where(
            (case_ds["lat"] > min_lat)
            & (case_ds["lat"] < max_lat)
            & (case_ds["lon"] > min_lon)
            & (case_ds["lon"] < max_lon)
        )[0]

        case_ds_subset = case_ds.isel(station=station_indices)
        case_ds_subset = case_ds_subset.sel(
            time=slice(case["start_date"], case["end_date"])
        )
        case_ds_subset["id"] = case["id"]
        dataset_list.append(case_ds_subset)
    datasets_df = pd.DataFrame()
    for ds in dataset_list:
        ds_df = ds.to_dataframe().reset_index()
        datasets_df = pd.concat([datasets_df, ds_df], ignore_index=True)
    datasets_df.rename(
        columns={
            "t": "surface_temperature",
            "td": "surface_dew_point",
            "ws": "wind_speed",
            "wd": "wind_from_direction",
            "sp": "surface_air_pressure",
            "msl": "air_pressure_at_mean_sea_level",
            "c": "cloud_area_fraction",
            "ra1": "accumulated_1_hour_precipitation",
            "ra3": "accumulated_3_hour_precipitation",
            "ra6": "accumulated_6_hour_precipitation",
            "ra12": "accumulated_12_hour_precipitation",
            "ra24": "accumulated_24_hour_precipitation",
        },
        inplace=True,
    )
    return datasets_df


if __name__ == "__main__":
    datasets_df = run_isd_generation(
        output_dir=".",
        events_file_path=".",
        use_dask=True,
        n_workers=10,
    )
