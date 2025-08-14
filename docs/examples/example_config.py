"""
Example configuration file for ExtremeWeatherBench CLI.

This file demonstrates how to create custom evaluation objects and case data
for use with the --config-file option.

Usage:
    ewb --config-file example_config.py
"""

from extremeweatherbench import derived, inputs, metrics
from extremeweatherbench.utils import load_events_yaml

# Define targets (observation data)
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# Define forecasts
custom_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

# Define evaluation objects
evaluation_objects = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=era5_heatwave_target,
        forecast=custom_forecast,
    ),
]

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
cases_dict = load_events_yaml()

# Alternatively, users could define custom cases like this:
# cases_dict = {
#     "cases": [
#         {
#             "case_id_number": 1,
#             "title": "Custom Heat Wave Case",
#             "start_date": "2021-06-01T00:00:00",
#             "end_date": "2021-06-15T00:00:00",
#             "location": {
#                 "type": "centered_region",
#                 "parameters": {
#                     "latitude": 40.0,
#                     "longitude": -100.0,
#                     "bounding_box_degrees": 5.0,
#                 },
#             },
#             "event_type": "heat_wave",
#         },
#     ]
# }
