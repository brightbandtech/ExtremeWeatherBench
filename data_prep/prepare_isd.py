import glob
from pathlib import Path

import pandas as pd
import tqdm
from joblib import Parallel, delayed

COLUMNS = [
    "obs_year",
    "obs_month",
    "obs_day",
    "obs_hour",
    "air_temp",
    "dew_point_temp",
    "sea_level_pressure",
    "wind_dir",
    "wind_speed",
    "sky_condition",
    "precip_hrly",
    "precip_6hrly",
]

def process_gz_file(pth: Path) -> pd.DataFrame:
    """Read a GZ file and return a DataFrame."""
    usaf, wban, _ = pth.name.split("-")
    df = pd.read_table(
        pth, header=None, sep="\\s+", names=COLUMNS, compression="gzip", na_values=-9999
    )
    df = df.rename(
        {
            "obs_year": "year",
            "obs_month": "month",
            "obs_day": "day",
            "obs_hour": "hour",
        },
        axis=1,
    )
    df["obs_timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]], utc=True)
    df["usaf"] = usaf
    df["wban"] = wban
    df = df.drop(columns=["year", "month", "day", "hour"])
    return df

if __name__ == "__main__":
    fns = list(glob.glob("data/isd-lite/2021/*.gz"))
    n = len(fns)

    dfs = Parallel(n_jobs=9)(
        delayed(process_gz_file)(Path(fn)) for fn in tqdm.tqdm(fns)
    )
    data = pd.concat(dfs)
    data.to_parquet("data/isd-lite-2021.parquet")