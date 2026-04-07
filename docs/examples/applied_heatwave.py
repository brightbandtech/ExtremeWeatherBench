import logging
import operator

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = ewb.cases.load_ewb_events_yaml_into_case_list()

# Define targets
# ERA5 target
era5_heatwave_target = ewb.inputs.ERA5(
    variables=["surface_air_temperature"],
    chunks=None,
)

# GHCN target
ghcn_heatwave_target = ewb.inputs.GHCN(
    variables=["surface_air_temperature"],
)

# Define forecast (HRES)
hres_forecast = ewb.inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping=ewb.inputs.HRES_metadata_variable_mapping,
    preprocess=ewb.defaults.preprocess_heatwave_forecast_dataset,
)

# Load the climatology for DurationMeanError
climatology = ewb.defaults.get_climatology(quantile=0.85)

# Define the metrics
metrics_list = [
    ewb.metrics.MaximumMeanAbsoluteError(),
    ewb.metrics.RootMeanSquaredError(),
    ewb.metrics.DurationMeanError(threshold_criteria=climatology, op_func=operator.ge),
    ewb.metrics.MaximumLowestMeanAbsoluteError(),
]

# Create a list of evaluation objects for heatwave
heatwave_evaluation_object = [
    ewb.inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=ghcn_heatwave_target,
        forecast=hres_forecast,
    ),
    ewb.inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=era5_heatwave_target,
        forecast=hres_forecast,
    ),
]
if __name__ == "__main__":
    # Initialize ExtremeWeatherBench
    heatwave_ewb = ewb.evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=heatwave_evaluation_object,
    )

    # Run the workflow
    outputs = heatwave_ewb.run_evaluation(
        parallel_config={"backend": "loky", "n_jobs": 2}
    )
    outputs.to_csv("applied_heatwave_outputs.csv")
