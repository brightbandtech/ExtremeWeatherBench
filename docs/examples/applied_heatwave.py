import logging
import time

import xarray as xr

from extremeweatherbench import cases, evaluate, inputs, metrics

start_time = time.time()
# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)

# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = cases.load_ewb_events_yaml_into_case_collection()
case_yaml.select_cases(by="case_id_number", value=1, inplace=True)

# Define targets
# ERA5 target
era5_heatwave_target = inputs.ERA5(
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

climatology = xr.open_zarr(
    "gs://extremeweatherbench/datasets/surface_air_temperature_1990_2019_climatology.zarr",  # noqa: E501
    storage_options={"anon": True},
    chunks="auto",
)
climatology = climatology["2m_temperature"].sel(quantile=0.85)
metrics_list = [
    metrics.HeatwaveDurationME(climatology),
    metrics.MaximumMAE,
    metrics.RMSE,
]
# Create a list of evaluation objects for heatwave
heatwave_evaluation_object = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=ghcn_target,
        forecast=hres_forecast,
    ),
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=era5_heatwave_target,
        forecast=hres_forecast,
    ),
]

# Initialize ExtremeWeatherBench
ewb = evaluate.ExtremeWeatherBench(
    case_metadata=case_yaml,
    evaluation_objects=heatwave_evaluation_object,
)

# Run the workflow
outputs = ewb.run(
    n_jobs=1,
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts for certain metrics
    # such as minimum temperature MAE
    tolerance_range=48,
    # precompute the datasets before metrics are calculated, to avoid IO costs loading
    # them into memory for each metric
    pre_compute=True,
)
outputs.to_csv("outputs.csv")
# Print the outputs; can be saved if desired
logger.info(f"Time taken: {time.time() - start_time} seconds")
print(f"Time taken: {time.time() - start_time} seconds")
