import logging

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

# Option 1: Use the cached factory functions directly (simplest approach)
# These will automatically share the global cache for the same thresholds
simple_metrics = [
    metrics.CSI(forecast_threshold=15000, target_threshold=0.3),
    metrics.FAR(forecast_threshold=15000, target_threshold=0.3),
    metrics.TP(forecast_threshold=15000, target_threshold=0.3),
    metrics.FP(forecast_threshold=15000, target_threshold=0.3),
    metrics.TN(forecast_threshold=15000, target_threshold=0.3),
    metrics.FN(forecast_threshold=15000, target_threshold=0.3),
]

# just one for now
severe_convection_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=simple_metrics,  # These will use global cache automatically
        target=pph_target,
        forecast=hres_forecast,
    ),
]

test_ewb = evaluate.ExtremeWeatherBench(
    case_metadata=case_yaml, evaluation_objects=severe_convection_evaluation_objects
)
logger.info("Starting EWB run")
outputs = test_ewb.run(n_jobs=1)
outputs.to_csv("applied_severe_results.csv")
