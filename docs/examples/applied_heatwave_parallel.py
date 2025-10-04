import logging
import multiprocessing

from extremeweatherbench import cases, evaluate, inputs, metrics

logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = cases.load_ewb_events_yaml_into_case_collection()

# ERA5 target
era5_target = inputs.ERA5(
    variables=["surface_air_temperature"],
    chunks=None,
)

# GHCN target
ghcn_target = inputs.GHCN(
    variables=["surface_air_temperature"],
)

# Define forecast (HRES)
hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
)

# Example of two evaluation objects for heatwaves, one for GHCN and one for ERA5
# evaluated against HRES
heatwave_evaluation_object = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=ghcn_target,
        forecast=hres_forecast,
    ),
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=era5_target,
        forecast=hres_forecast,
    ),
]

# Initialize ExtremeWeatherBench
ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=heatwave_evaluation_object,
)

# Get the number of available CPUs for determining n_processes and divide by 4 threads
# per process; this is optional. Leaving n_jobs blank will use the joblib backend
# default (which, if loky, is the number of available CPUs).
n_threads_per_process = 4
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)

# Run the evaluation using pre_compute to avoid recomputing the datasets for each metric
results = ewb.run(n_jobs=n_processes, pre_compute=True)

# Save the results to a csv file
results.to_csv("heatwave_evaluation_results.csv", index=False)
