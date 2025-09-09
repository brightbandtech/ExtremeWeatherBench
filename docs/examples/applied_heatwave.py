import logging

import numpy as np
import xarray as xr

from extremeweatherbench import evaluate, inputs, metrics, utils

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
    return ds


# Define targets
# ERA5 target
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)

# GHCN target
ghcn_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
)

# Define forecast (HRES)
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# Example of one evaluation object for heatwave
heatwave_evaluation_object = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=ghcn_target,
        forecast=hres_forecast,
    )
]

# Initialize ExtremeWeatherBench
test_ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=heatwave_evaluation_object,
)

# Run the workflow
outputs = test_ewb.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)

# Print the outputs; can be saved if desired
print(outputs.head())
