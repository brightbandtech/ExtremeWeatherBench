"""Example configuration file for ExtremeWeatherBench CLI.

This file demonstrates how to create custom evaluation objects and case data
for use with the --config-file option.

Usage:
    ewb --config-file example_config.py
"""

from extremeweatherbench import inputs, metrics
from extremeweatherbench.cases import load_ewb_events_yaml_into_case_collection

# Define targets (observation data)
era5_heatwave_target = inputs.ERA5(
    variables=["surface_air_temperature"],
    chunks=None,
)

# Define forecasts
fcnv2_forecast = inputs.KerchunkForecast(
    name="fcnv2_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
)

# Define evaluation objects
evaluation_objects = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=era5_heatwave_target,
        forecast=fcnv2_forecast,
    ),
]

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
cases_dict = load_ewb_events_yaml_into_case_collection()

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
