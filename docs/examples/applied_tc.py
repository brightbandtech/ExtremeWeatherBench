import logging

from extremeweatherbench import cases, defaults, derived, evaluate, inputs, metrics

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Load the case collection from the YAML file
case_yaml = cases.load_ewb_events_yaml_into_case_collection()

# Select single case (TC Ida)
case_yaml.select_cases(by="case_id_number", value=220, inplace=True)

# Define IBTrACS target, no arguments needed as defaults are sufficient
ibtracs_target = inputs.IBTrACS()

# Define HRES forecast
hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    # Define tropical cyclone track derivedvariable to include in the forecast
    variables=[derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for HRES forecast
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    # Preprocess the HRES forecast to include geopotential thickness calculation
    preprocess=defaults._preprocess_hres_tc_forecast_dataset,
)

# Define FCNv2 forecast
fcnv2_forecast = inputs.KerchunkForecast(
    name="fcn_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for FCNv2 forecast
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    # Preprocess the FCNv2 forecast to include geopotential thickness calculation
    preprocess=defaults._preprocess_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

# Define Pangu forecast
pangu_forecast = inputs.KerchunkForecast(
    name="pangu_forecast",
    source="gs://extremeweatherbench/PANG_v100_GFS.parq",
    variables=[derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for Pangu forecast
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    # Preprocess the Pangu forecast to include geopotential thickness calculation
    # which uses the same preprocessing function as the FCNv2 forecast
    preprocess=defaults._preprocess_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


# Define composite metric for tropical cyclone track metrics. Using a composite metric
# prevents recomputation of landfalls, saving significant time. approach="next" sets
# the evaluation to occur, in the case of multiple landfalls, for the next landfall in
# time to be evaluated against
composite_landfall_metrics = [
    metrics.LandfallMetric(
        metrics=[
            metrics.LandfallIntensityMeanAbsoluteError,
            metrics.LandfallTimeMeanError,
            metrics.LandfallDisplacement,
        ],
        approach="next",
        # Set the intensity variable to use for the metric
        forecast_variable="air_pressure_at_mean_sea_level",
        target_variable="air_pressure_at_mean_sea_level",
    )
]

# Define evaluation objects for tropical cyclone metrics. Setting event type subsets
# the relevant cases inside the events YAML file
tc_evaluation_object = [
    # HRES forecast
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
    # Pangu forecast
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=pangu_forecast,
    ),
    # FCNv2 forecast
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=fcnv2_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench
    ewb = evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=tc_evaluation_object,
    )
    logger.info("Starting EWB run")
    # Run the workflow with parallel_config backend set to dask
    outputs = ewb.run_evaluation(
        parallel_config={"backend": "loky", "n_jobs": 3},
    )
    outputs.to_csv("tc_metric_test_results.csv")
