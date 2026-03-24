import logging

import extremeweatherbench as ewb

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Load the case collection from the YAML file
case_yaml = ewb.load_cases()

# Select single case (TC Ida)
case_list = [n for n in case_yaml if n.case_id_number == 220]

# Define IBTrACS target, no arguments needed as defaults are sufficient
ibtracs_target = ewb.targets.IBTrACS()

# # Define HRES forecast
hres_forecast = ewb.forecasts.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    # Define tropical cyclone track derived variable to include in the forecast
    variables=[ewb.derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for HRES forecast
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    # Preprocess the HRES forecast to include geopotential thickness calculation
    preprocess=ewb.defaults.preprocess_hres_tc_forecast_dataset,
)


# Define FCNv2 forecast, using provided defaults helper
fcnv2_forecast = ewb.defaults.cira_fcnv2_tropical_cyclone_forecast

# Define Pangu forecast, using icechunk helper. Preprocesser available in defaults
pangu_forecast = ewb.inputs.get_cira_icechunk(
    model_name="PANG_v100_GFS",
    variables=[ewb.derived.TropicalCycloneTrackVariables()],
    name="Pangu",
    preprocess=ewb.defaults.preprocess_cira_icechunk_tc_forecast_dataset,
)

# Define composite metric for tropical cyclone track metrics. Using a composite metric
# prevents recomputation of landfalls, saving significant time. approach="next" sets
# the evaluation to occur, in the case of multiple landfalls, for the next landfall in
# time to be evaluated against
composite_landfall_metrics = [
    ewb.metrics.LandfallMetric(
        metrics=[
            ewb.metrics.LandfallIntensityMeanAbsoluteError,
            ewb.metrics.LandfallTimeMeanError,
            ewb.metrics.LandfallDisplacement,
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
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
    # Pangu forecast
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=pangu_forecast,
    ),
    # FCNv2 forecast
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=fcnv2_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench
    tc_ewb = ewb.evaluation(
        case_metadata=case_list,
        evaluation_objects=tc_evaluation_object,
    )
    logger.info("Starting EWB run")
    outputs = tc_ewb.run_evaluation(
        parallel_config={"backend": "loky", "n_jobs": 1},
    )
    outputs.to_csv("tc_metric_test_results.csv")
