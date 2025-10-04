import logging

from extremeweatherbench import derived, evaluate, inputs, metrics, utils

# %%
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

case_yaml = utils.load_events_yaml()
ar_cases = {
    "cases": [
        n
        for n in case_yaml["cases"]
        if n["start_date"].year < 2023 and n["event_type"] == "atmospheric_river"
    ][5:6]
}

era5_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        derived.AtmosphericRiverMask,
    ],
    variable_mapping={
        "specific_humidity": "specific_humidity",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[
        derived.AtmosphericRiverMask,
    ],
    variable_mapping={
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

ar_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[metrics.SpatialDisplacement],
        target=era5_target,
        forecast=hres_forecast,
    ),
]

ar_ewb = evaluate.ExtremeWeatherBench(
    cases=ar_cases,
    evaluation_objects=ar_evaluation_objects,
)
logger.info("Starting EWB run")
outputs = ar_ewb.run()

outputs.to_csv("ar_outputs.csv")
