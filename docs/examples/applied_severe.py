import logging

import numpy as np
import pandas as pd
import xarray as xr

from extremeweatherbench import derived, evaluate, inputs, metrics, utils

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

case_yaml = utils.load_events_yaml()
test_yaml = {
    "cases": [
        n
        for n in case_yaml["cases"]
        if n["event_type"] == "severe_convection"
        and pd.to_datetime(n["start_date"]).year <= 2022
    ][0:1]
}


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


pph_target = inputs.PPH(
    source=inputs.PPH_URI,
    variables=["practically_perfect_hindcast"],
    variable_mapping={},
    storage_options={"remote_options": {"anon": True}},
)

hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.CravenBrooksSignificantSevere],
    variable_mapping={
        "temperature": "air_temperature",
        "dewpoint": "dewpoint_temperature",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "time": "init_time",
        "prediction_timedelta": "lead_time",
        "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
        "specific_humidity": "specific_humidity",
    },
    storage_options={"remote_options": {"anon": True}},
)

# Option 1: Use the cached factory functions directly (simplest approach)
# These will automatically share the global cache for the same thresholds
simple_metrics = [
    metrics.CSI(forecast_threshold=15000, target_threshold=0.3),
    metrics.FAR(forecast_threshold=15000, target_threshold=0.3),
    metrics.TP(forecast_threshold=15000, target_threshold=0.3),
    metrics.FP(forecast_threshold=15000, target_threshold=0.3),
    metrics.TN(forecast_threshold=15000, target_threshold=0.3),
    metrics.FN(forecast_threshold=15000, target_threshold=0.3),
]

# Option 2: Create cached metrics that share the same transformed contingency manager
cached_metrics = metrics.create_threshold_metrics(
    forecast_threshold=15000,
    target_threshold=0.3,
    functions=[metrics.csi_function, metrics.far_function, metrics.accuracy_function],
    instances=[metrics.tp, metrics.fp, metrics.tn, metrics.fn],
)

# just one for now
severe_convection_metric_list = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric=simple_metrics,  # These will use global cache automatically
        target=pph_target,
        forecast=hres_forecast,
    ),
]

test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=severe_convection_metric_list,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(pre_compute=True)
outputs.to_csv("applied_severe_results.csv")
