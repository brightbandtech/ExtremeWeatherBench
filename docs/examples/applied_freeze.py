import logging
import operator

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = ewb.load_cases()

# Define targets
# ERA5 target
era5_freeze_target = ewb.targets.ERA5(
    variables=["surface_air_temperature"],
    chunks=None,
)

# GHCN target
ghcn_freeze_target = ewb.targets.GHCN(variables=["surface_air_temperature"])

# Define forecast (FCNv2 CIRA Virtualizarr)
fcnv2_forecast = ewb.forecasts.KerchunkForecast(
    name="fcnv2_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=ewb.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=ewb.defaults._preprocess_bb_cira_forecast_dataset,
)

# Load the climatology for DurationMeanError
climatology = ewb.get_climatology(quantile=0.85)

# Define the metrics
metrics_list = [
    ewb.metrics.RootMeanSquaredError(),
    ewb.metrics.MinimumMeanAbsoluteError(),
    ewb.metrics.DurationMeanError(threshold_criteria=climatology, op_func=operator.le),
]

# Create a list of evaluation objects for freeze
freeze_evaluation_object = [
    ewb.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=ghcn_freeze_target,
        forecast=fcnv2_forecast,
    ),
    ewb.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=era5_freeze_target,
        forecast=fcnv2_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench runner instance
    freeze_ewb = ewb.evaluation(
        case_metadata=case_yaml,
        evaluation_objects=freeze_evaluation_object,
    )

    # Run the workflow
    outputs = freeze_ewb.run(parallel_config={"backend": "loky", "n_jobs": 1})

    # Print the outputs; can be saved if desired
    outputs.to_csv("freeze_outputs.csv")
