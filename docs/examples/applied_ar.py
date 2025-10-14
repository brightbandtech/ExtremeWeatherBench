import logging

from extremeweatherbench import cases, derived, evaluate, inputs, metrics

# %%

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = cases.load_ewb_events_yaml_into_case_collection()

# Define ERA5 target
era5_target = inputs.ERA5(
    variables=[derived.AtmosphericRiverMask],
)

# Define forecast (HRES)
hres_forecast = inputs.ZarrForecast(
    variables=[derived.AtmosphericRiverMask],
)

# Create a list of evaluation objects for atmospheric river
ar_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[metrics.SpatialDisplacement],
        target=era5_target,
        forecast=hres_forecast,
    ),
]

# Initialize ExtremeWeatherBench; will only run on cases with event_type
# atmospheric_river
ar_ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=ar_evaluation_objects,
)

# Run the workflow using all available CPUs
outputs = ar_ewb.run(n_jobs=-1)

# Save the evaluationoutputs to a csv file
outputs.to_csv("ar_outputs.csv")
