import logging
import multiprocessing

from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import cases, evaluate, inputs, metrics

logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

case_yaml = cases.load_ewb_events_yaml_into_case_collection()


era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)

ghcn_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# just one for now
heatwave_evaluation_list = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=era5_heatwave_target,
        forecast=hres_forecast,
    ),
]

ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=heatwave_evaluation_list,
)
# Get the number of available CPUs for determining n_processes
n_threads_per_process = 10
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)
with logging_redirect_tqdm(loggers=[logger]):
    results = ewb.run(parallel=True, n_jobs=n_processes, pre_compute=True)
results.to_csv("heatwave_evaluation_results.csv", index=False)
