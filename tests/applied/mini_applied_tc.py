#!/usr/bin/env python3
"""
Miniaturized tropical cyclone evaluation script for fast testing.
Reduced data scope for quick verification of the EWB pipeline.
"""

import logging

from extremeweatherbench import derived, evaluate, inputs, metrics, utils

# Configure logging to suppress verbose output
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load events and select one tropical cyclone case
case_yaml = utils.load_events_yaml()
tc_cases = [
    case for case in case_yaml["cases"] if case["event_type"] == "tropical_cyclone"
]

# Use only one TC case for speed (case 200 from original was a good example)
test_yaml = {"cases": tc_cases[:1] if tc_cases else []}

if not test_yaml["cases"]:
    logger.warning("No tropical cyclone cases found, using case 200 index")
    # Fallback to specific case index if available
    if len(case_yaml["cases"]) > 200:
        test_yaml = {"cases": [case_yaml["cases"][200]]}
    else:
        test_yaml = {"cases": case_yaml["cases"][-1:]}  # Use last case

logger.info(f"Testing with {len(test_yaml['cases'])} tropical cyclone case(s)")

# IBTrACS target
ibtracs_target = inputs.IBTrACS(
    source=inputs.IBTRACS_URI,
    variable_mapping=inputs.IBTrACS_metadata_variable_mapping,
)

# HRES forecast with chunking and TC track variables
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.TropicalCycloneTrackVariables],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
        "lead_time": "prediction_timedelta",
        "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
        "10m_wind_speed": "surface_wind_speed",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks={"prediction_timedelta": 10, "latitude": 180, "longitude": 360},
)

# Use only essential TC metrics for speed
tc_metric_list = [
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric=[
            metrics.LandfallTimeME,
            metrics.LandfallDisplacement,
        ],
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
]

# Run evaluation
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=tc_metric_list,
)

logger.info("Starting miniaturized EWB tropical cyclone evaluation")
outputs = test_ewb.run(
    tolerance_range=24,  # Reduced tolerance for speed
    pre_compute=False,  # TC processing can be memory intensive
)

logger.info("Tropical cyclone evaluation completed successfully")
print("Sample results:")
print(outputs.head())
