import logging

from dask.distributed import Client

from extremeweatherbench import cases, derived, evaluate, inputs, metrics

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

case_yaml = cases.load_ewb_events_yaml_into_case_collection()
case_yaml.select_cases("case_id_number", 305, inplace=True)

pph_target = inputs.PPH(
    variables=["practically_perfect_hindcast"],
)

hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.CravenBrooksSignificantSevere(layer_depth=100)],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

threshold_metrics = [
    metrics.ThresholdMetric(
        metrics=[
            metrics.CSI,
            metrics.FAR,
            metrics.TP,
            metrics.FP,
            metrics.TN,
            metrics.FN,
        ],
        forecast_threshold=15000,
        target_threshold=0.3,
    )
]


# just one for now
severe_convection_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=threshold_metrics,
        target=pph_target,
        forecast=hres_forecast,
    ),
]
if __name__ == "__main__":
    with Client() as client:
        test_ewb = evaluate.ExtremeWeatherBench(
            case_metadata=case_yaml,
            evaluation_objects=severe_convection_evaluation_objects,
        )
        logger.info("Starting EWB run")
        outputs = test_ewb.run(parallel_config={"backend": "dask"})
        outputs.to_csv("applied_severe_results.csv")
