#!/usr/bin/env python3
"""
Miniaturized heatwave evaluation script for fast testing.
Reduced data scope for quick verification of the EWB pipeline.
"""

import logging

from extremeweatherbench import evaluate, inputs, metrics, utils

# Configure logging to suppress verbose output
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load events and use only the first heat wave case
case_yaml = utils.load_events_yaml()
heatwave_cases = [
    case for case in case_yaml["cases"] if case["event_type"] == "heat_wave"
]

# Use only the first case for speed
test_yaml = {"cases": heatwave_cases[:1]}

logger.info(f"Testing with {len(test_yaml['cases'])} heat wave case(s)")

# GHCN target
ghcn_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "surface_air_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={},
)

# HRES forecast with chunking for speed
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks={"prediction_timedelta": 10, "latitude": 180, "longitude": 360},
)

# Use only essential metrics for speed
heatwave_metric_list = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
        ],
        target=ghcn_target,
        forecast=hres_forecast,
    ),
]

# Run evaluation
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=heatwave_metric_list,
)

logger.info("Starting miniaturized EWB heatwave evaluation")
outputs = test_ewb.run(
    tolerance_range=24,  # Reduced tolerance for speed
    pre_compute=True,
)

logger.info("Heatwave evaluation completed successfully")
print("Sample results:")
print(outputs.head())
