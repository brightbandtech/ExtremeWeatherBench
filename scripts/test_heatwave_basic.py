import datetime

from extremeweatherbench import evaluate, inputs, metrics
from extremeweatherbench.defaults import preprocess_bb_cira_forecast_dataset

test_case = {
    "cases": [
        {
            "case_id_number": 1,
            "title": "2024 New York City Test",
            "start_date": datetime.datetime(2024, 7, 10, 0, 0, 0),
            "end_date": datetime.datetime(2024, 7, 18, 0, 0, 0),
            "location": {
                "type": "bounded_region",
                "parameters": {
                    "latitude_min": 40,
                    "latitude_max": 42,
                    "longitude_min": -75,
                    "longitude_max": -72.25,
                },
            },
            "event_type": "heat_wave",
        }
    ]
}

era5_heatwave_target = inputs.ERA5(
    variables=["surface_air_temperature"],
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)

era5_ghcn_target = inputs.GHCN(
    variables=["surface_air_temperature"],
    storage_options={},
)

fcnv2_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=preprocess_bb_cira_forecast_dataset,
)

gc_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/GRAP_v100_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=preprocess_bb_cira_forecast_dataset,
)

heatwave_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
        ],
        target=era5_heatwave_target,
        forecast=fcnv2_forecast,
    ),
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
        ],
        target=era5_ghcn_target,
        forecast=gc_forecast,
    ),
]


test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_case,
    evaluation_objects=heatwave_evaluation_objects,
)

test_ewb.run(
    pre_compute=True,
)

print(test_ewb.outputs)
