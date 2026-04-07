import logging
import operator

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = ewb.cases.load_ewb_events_yaml_into_case_list()


# ERA5 target
era5_freeze_target = ewb.inputs.ERA5(
    variables=["surface_air_temperature"],
    chunks=None,
)

# GHCN target
ghcn_freeze_target = ewb.inputs.GHCN(variables=["surface_air_temperature"])

# Forecast (FCNv2) using helper in defaults
fcnv2_forecast = ewb.defaults.cira_fcnv2_freeze_forecast

# Load the climatology for DurationMeanError
climatology = ewb.defaults.get_climatology(quantile=0.15)

# Define the metrics
metrics_list = [
    ewb.metrics.RootMeanSquaredError(),
    ewb.metrics.MinimumMeanAbsoluteError(),
    ewb.metrics.DurationMeanError(threshold_criteria=climatology, op_func=operator.le),
]

# Create a list of evaluation objects for freeze
freeze_evaluation_object = [
    ewb.inputs.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=ghcn_freeze_target,
        forecast=fcnv2_forecast,
    ),
    ewb.inputs.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=era5_freeze_target,
        forecast=fcnv2_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench runner instance
    freeze_ewb = ewb.evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=freeze_evaluation_object,
    )

    # Run the workflow
    outputs = freeze_ewb.run_evaluation(
        parallel_config={"backend": "loky", "n_jobs": 1}
    )

    # Print the outputs; can be saved if desired
    outputs.to_csv("freeze_outputs.csv")
