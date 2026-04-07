import logging

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = ewb.cases.load_all_cases()


# ERA5 target
era5_target = ewb.inputs.ERA5(
    variables=[
        ewb.derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
)

# Forecast (HRES)
hres_forecast = ewb.inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variables=[
        ewb.derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    variable_mapping=ewb.inputs.HRES_metadata_variable_mapping,
)

grap_forecast = ewb.inputs.get_cira_icechunk(
    model_name="GRAP_v100_GFS",
    variables=[
        ewb.derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    name="Graphcast",
    preprocess=ewb.defaults.preprocess_cira_icechunk_ar_forecast_dataset,
)

pang_forecast = ewb.inputs.get_cira_icechunk(
    model_name="PANG_v100_GFS",
    variables=[
        ewb.derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    name="Pangu",
    preprocess=ewb.defaults.preprocess_cira_icechunk_ar_forecast_dataset,
)
# Create a list of evaluation objects for atmospheric river
ar_evaluation_objects = [
    ewb.inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            ewb.metrics.CriticalSuccessIndex(),
            ewb.metrics.EarlySignal(),
            ewb.metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=hres_forecast,
    ),
    ewb.inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            ewb.metrics.CriticalSuccessIndex(),
            ewb.metrics.EarlySignal(),
            ewb.metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=grap_forecast,
    ),
    ewb.inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            ewb.metrics.CriticalSuccessIndex(),
            ewb.metrics.EarlySignal(),
            ewb.metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=pang_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench; will only run on cases with event_type
    # atmospheric_river
    ar_ewb = ewb.evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=ar_evaluation_objects,
    )

    # Run the workflow using 3 jobs
    outputs = ar_ewb.run_evaluation(parallel_config={"backend": "loky", "n_jobs": 3})

    # Save the evaluation outputs to a csv file
    outputs.to_csv("ar_signal_outputs.csv")
