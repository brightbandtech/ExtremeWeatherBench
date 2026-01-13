import logging

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Load case data from the default events.yaml
case_yaml = ewb.load_cases()
case_yaml.select_cases("case_id_number", 305, inplace=True)

# Define PPH target
pph_target = ewb.targets.PPH(
    variables=["practically_perfect_hindcast"],
)

# Define LSR target
lsr_target = ewb.targets.LSR()

# Define HRES forecast
hres_forecast = ewb.forecasts.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[ewb.derived.CravenBrooksSignificantSevere(layer_depth=100)],
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

# Define pph metrics as thresholdmetric to share scores contingency table
pph_metrics = [
    ewb.metrics.ThresholdMetric(
        metrics=[
            ewb.metrics.CriticalSuccessIndex,
            ewb.metrics.FalseAlarmRatio,
        ],
        forecast_threshold=15000,
        target_threshold=0.3,
    ),
    ewb.metrics.EarlySignal(threshold=15000),
]

# Define LSR metrics as thresholdmetric to share scores contingency table
lsr_metrics = [
    ewb.metrics.ThresholdMetric(
        metrics=[
            ewb.metrics.TruePositives,
            ewb.metrics.FalseNegatives,
        ],
        forecast_threshold=15000,
        target_threshold=0.5,
    )
]

# Define evaluation objects for severe convection:
# One evaluation object for PPH
pph_evaluation_objects = [
    ewb.EvaluationObject(
        event_type="severe_convection",
        metric_list=pph_metrics,
        target=pph_target,
        forecast=hres_forecast,
    ),
]

# One evaluation object for LSR
lsr_evaluation_objects = [
    ewb.EvaluationObject(
        event_type="severe_convection",
        metric_list=lsr_metrics,
        target=lsr_target,
        forecast=hres_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench with both evaluation objects
    severe_ewb = ewb.evaluation(
        case_metadata=case_yaml,
        evaluation_objects=lsr_evaluation_objects + pph_evaluation_objects,
    )
    logger.info("Starting EWB run")

    # Run the workflow with parllel_config backend set to dask
    outputs = severe_ewb.run(parallel_config={"backend": "loky", "n_jobs": 3})

    # Save the results to a CSV file
    outputs.to_csv("applied_severe_convection_results.csv")
