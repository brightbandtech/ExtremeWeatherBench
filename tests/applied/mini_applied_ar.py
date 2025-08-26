#!/usr/bin/env python3
"""
Miniaturized atmospheric river evaluation script for fast testing.
Reduced data scope for quick verification of the EWB pipeline.
"""

import logging

from extremeweatherbench import derived, evaluate, inputs, metrics, utils

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
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        "specific_humidity",
        "eastward_wind",
        "northward_wind",
        "surface_air_temperature",
    ],
    variable_mapping={
        "specific_humidity": "specific_humidity",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks={"time": 24, "latitude": 180, "longitude": 360},  # Smaller chunks
)

# HRES forecast with reduced lead times for speed
# Note: Using basic variables since IntegratedVaporTransport is not yet implemented
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
        "eastward_wind",
        "northward_wind",
    ],
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

# Use only MAE metric for speed
ar_metric_list = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric=[metrics.MAE],
        target=era5_target,
        forecast=hres_forecast,
    ),
]

# Run evaluation
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=ar_metric_list,
)

logger.info("Starting miniaturized EWB atmospheric river evaluation")
outputs = test_ewb.run(
    tolerance_range=24,  # Reduced tolerance for speed
    pre_compute=True,
)

logger.info("Atmospheric river evaluation completed successfully")
print("Sample results:")
print(outputs.head())
