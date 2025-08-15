import logging

from extremeweatherbench import evaluate, inputs, metrics, utils

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

case_yaml = utils.load_events_yaml()
test_yaml = {"cases": case_yaml["cases"][:1]}


ghcn_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "surface_air_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={},
)
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# just one for now
heatwave_metric_list = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=ghcn_target,
        forecast=hres_forecast,
    ),
]

test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=heatwave_metric_list,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)

print(outputs.head())
