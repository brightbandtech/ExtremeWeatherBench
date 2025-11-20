import logging

from dask.distributed import Client

from extremeweatherbench import cases, derived, evaluate, inputs, metrics

# Set loggers that might be noisy during the evaluation
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load case data from the default events.yaml
case_yaml = cases.load_ewb_events_yaml_into_case_collection()
case_yaml.select_cases("case_id_number", 305, inplace=True)

# Define PPH target
pph_target = inputs.PPH(
    variables=["practically_perfect_hindcast"],
)

# Define LSR target
lsr_target = inputs.LSR()

# Define HRES forecast
hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.CravenBrooksSignificantSevere(layer_depth=100)],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

# Define pph metrics as thresholdmetric to share scores contingency table
pph_metrics = [
    metrics.ThresholdMetric(
        metrics=[
            metrics.CriticalSuccessIndex,
            metrics.FalseAlarmRatio,
        ],
        forecast_threshold=15000,
        target_threshold=0.3,
    ),
    metrics.EarlySignal(threshold=15000),
]

# Define LSR metrics as thresholdmetric to share scores contingency table
lsr_metrics = [
    metrics.ThresholdMetric(
        metrics=[
            metrics.TP,
            metrics.FN,
        ],
        forecast_threshold=15000,
        target_threshold=0.5,
    )
]

# Define evaluation objects for severe convection:
# One evaluation object for PPH
pph_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=pph_target,
        forecast=hres_forecast,
    ),
]

# One evaluation object for LSR
lsr_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=lsr_target,
        forecast=hres_forecast,
    ),
]

if __name__ == "__main__":
    # Set up dask client for parallel execution
    with Client() as client:
        # Initialize ExtremeWeatherBench with both evaluation objects
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=case_yaml,
            evaluation_objects=lsr_evaluation_objects + pph_evaluation_objects,
        )
        logger.info("Starting EWB run")

        # Run the workflow with parllel_config backend set to dask
        outputs = ewb.run(parallel_config={"backend": "dask"})

        # Save the results to a CSV file
        outputs.to_csv("applied_severe_convection_results.csv")
