#!/usr/bin/env python3
"""
Miniaturized atmospheric river evaluation script for fast testing.
Reduced data scope for quick verification of the EWB pipeline.
"""

import logging

from extremeweatherbench import evaluate, inputs, metrics, utils

# Configure logging to suppress verbose output
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load events and select only 1 atmospheric river case before 2023
case_yaml = utils.load_events_yaml()
ar_cases = [
    case
    for case in case_yaml["cases"]
    if case["start_date"].year < 2023 and case["event_type"] == "atmospheric_river"
]

# Use only the first AR case for speed
test_yaml = {"cases": ar_cases[:1]}

logger.info(f"Testing with {len(test_yaml['cases'])} atmospheric river case(s)")

# ERA5 target with minimal chunking for speed
# Note: Using basic variables since IntegratedVaporTransport is not yet implemented
era5_target = inputs.ERA5(
    variables=[
        "specific_humidity",
        "eastward_wind",
        "northward_wind",
        "surface_air_temperature",
    ],
    chunks=None,
)

# HRES forecast with reduced lead times for speed
# Note: Using basic variables since IntegratedVaporTransport is not yet implemented
hres_forecast = inputs.ZarrForecast(
    name="HRES",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
        "eastward_wind",
        "northward_wind",
    ],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    chunks="auto",
)

# Use only MAE metric for speed
ar_metric_list = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[metrics.MAE],
        target=era5_target,
        forecast=hres_forecast,
    ),
]

# Run evaluation
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    evaluation_objects=ar_metric_list,
)

logger.info("Starting miniaturized EWB atmospheric river evaluation")
outputs = test_ewb.run(
    n_jobs=40,
)

logger.info("Atmospheric river evaluation completed successfully")
print("Sample results:")
print(outputs.head())
