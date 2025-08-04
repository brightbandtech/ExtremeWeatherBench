from email import utils

import concrete_refactor_script as crs
import refactor_scripts as rs

case_yaml = utils.read_event_yaml(
    "/Users/taylor/code/ExtremeWeatherBench/src/extremeweatherbench/data/events.yaml"
)
test_yaml = {"cases": [case_yaml["cases"][0]]}

incoming_forecast = crs.KerchunkForecast(
    forecast_source="gs://extremeweatherbench/datasets/FOUR_v200_GFS.parq"
)

# just one for now
heatwave_metric_list = [
    rs.MetricEvaluationObject(
        metric=crs.MaximumMAE,
        target=crs.ERA5,
        forecast=incoming_forecast,
        target_variables=["surface_air_temperature"],
        forecast_variables=["surface_air_temperature"],
        target_storage_options={"remote_options": {"anon": True}},
        forecast_storage_options={
            "remote_protocol": "s3",
            "remote_options": {"anon": True},
        },
        target_variable_mapping={"surface_air_temperature": "2m_temperature"},
        forecast_variable_mapping={"surface_air_temperature": "t2"},
    )
]

test_heat_wave = crs.HeatWave(
    case_metadata=test_yaml, metric_evaluation_objects=heatwave_metric_list
)

test_ewb = rs.ExtremeWeatherBench(events=[test_heat_wave], forecast=incoming_forecast)
test_ewb.run()
