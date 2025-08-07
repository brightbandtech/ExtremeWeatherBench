import logging
import multiprocessing
import os
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from random import random
from time import sleep

import joblib
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import config, evaluate, inputs, metrics, utils

log = logging.getLogger(__name__)


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


def configure_worker_logger(log_name="Default", log_level="DEBUG"):
    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            worker_logger = logging.getLogger(log_name)
            if not worker_logger.hasHandlers():
                h = QueueHandler(kwargs["log_queue"])
                worker_logger.addHandler(h)
            worker_logger.setLevel(log_level)
            kwargs["log_queue"] = worker_logger
            return func(*args, **kwargs)

        return wrapper


logger = logging.getLogger(__name__)

case_yaml = utils.read_event_yaml(
    "/home/taylor/ExtremeWeatherBench/src/extremeweatherbench/data/events.yaml"
)
test_yaml = {"cases": case_yaml["cases"][:]}


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
    return ds


era5_target_config = config.TargetConfig(
    target=inputs.ERA5,
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)
ghcn_target_config = config.TargetConfig(
    target=inputs.GHCN,
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)
cira_forecast_config = config.ForecastConfig(
    forecast=inputs.KerchunkForecast,
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

# just one for now
heatwave_metric_list = [
    config.MetricEvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target_config=era5_target_config,
        forecast_config=cira_forecast_config,
    ),
    # rs.MetricEvaluationObject(
    #     event_type="heat_wave",
    #     metric=[
    #         crs.MaximumMAE,
    #         crs.RMSE,
    #         crs.OnsetME,
    #         crs.DurationME,
    #         crs.MaxMinMAE,
    #     ],
    #     target_config=ghcn_target_config,
    #     forecast_config=cira_forecast_config,
    # ),
]

test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=heatwave_metric_list,
)
logger.info("Starting EWB run")

case_operators = test_ewb.case_operators
# Get the number of available CPUs for determining n_processes
n_threads_per_process = 10
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)

# Set environment variable to control threads per process
os.environ["OMP_NUM_THREADS"] = str(n_threads_per_process)
os.environ["MKL_NUM_THREADS"] = str(n_threads_per_process)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads_per_process)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads_per_process)

joblib_config = joblib.parallel_config(n_jobs=n_processes, prefer="processes")
# Use joblib to parallelize with n processes, each with 10 threads
with logging_redirect_tqdm():
    joblib.Parallel(n_jobs=n_processes, prefer="processes")(
        joblib.delayed(test_ewb.compute_case_operator)(
            case_op, tolerance_range=48, pre_compute=True
        )
        for case_op in tqdm(case_operators)
    )
