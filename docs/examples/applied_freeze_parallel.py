import logging
import multiprocessing
import os

import joblib
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import evaluate, inputs, metrics, utils


def configure_root_logger():
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("joblib.log")
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)
    return root


# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load events yaml
case_yaml = utils.load_events_yaml()


# Preprocess function for CIRA data using Brightband kerchunk parquets
def _preprocess_bb_cira_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """An example preprocess function that renames the time coordinate to lead_time,
    creates a valid_time coordinate, and sets the lead time range and resolution not
    present in the original dataset.

    Args:
        ds: The forecast dataset to rename.

    Returns:
        The renamed forecast dataset.
    """
    ds = ds.rename({"time": "lead_time"})
    # The evaluation configuration is used to set the lead time range and resolution.
    ds["lead_time"] = np.array(
        [i for i in range(0, 241, 6)], dtype="timedelta64[h]"
    ).astype("timedelta64[ns]")
    ds["surface_wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    ds["surface_wind_from_direction"] = (
        np.degrees(np.arctan2(ds["u10"], ds["v10"])) % 360
    )
    return ds


# Define targets
# ERA5 target
era5_freeze_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)

# GHCN target
ghcn_freeze_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=[
        "surface_air_temperature",
        "surface_wind_speed",
        "surface_wind_from_direction",
    ],
)

# Define forecast (FCNv2 CIRA Virtualizarr)
fcnv2_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        "surface_air_temperature",
        "surface_wind_speed",
        "surface_wind_from_direction",
    ],
    variable_mapping={
        "t2": "surface_air_temperature",
        "u10": "surface_eastward_wind",
        "v10": "surface_northward_wind",
    },
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)


# Create a list of evaluation objects for freeze
freeze_evaluation_object = [
    inputs.EvaluationObject(
        event_type="freeze",
        metric_list=[
            metrics.RMSE,
            metrics.MinimumMAE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=ghcn_freeze_target,
        forecast=fcnv2_forecast,
    ),
    inputs.EvaluationObject(
        event_type="freeze",
        metric_list=[
            metrics.RMSE,
            metrics.MinimumMAE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=era5_freeze_target,
        forecast=fcnv2_forecast,
    ),
]

# Initialize ExtremeWeatherBench
ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=freeze_evaluation_object,
)

# Get the number of available CPUs for determining n_processes and divide by 4 threads
# per process; this is optional. Leaving n_jobs blank will use the joblib backend
# default (which, if loky, is the number of available CPUs).
n_threads_per_process = 4
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)

results = ewb.run(n_jobs=n_processes, pre_compute=True)
results.to_csv("heatwave_evaluation_results.csv", index=False)
