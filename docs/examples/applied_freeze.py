import logging

import numpy as np
import xarray as xr

from extremeweatherbench import cases, evaluate, inputs, metrics

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Preprocess function for CIRA data using Brightband kerchunk parquets
def _preprocess_bb_cira_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """An example preprocess function that renames the time coordinate to lead_time,
    creates a valid_time coordinate, and sets the lead time range and resolution not
    present in the original dataset.

    Args:
        ds: The forecast dataset to rename.

    Returns:
        The renamed forecast dataset.
    """
    ds = ds.rename({"time": "lead_time"})
    # The evaluation configuration is used to set the lead time range and resolution.
    ds["lead_time"] = np.array(
        [i for i in range(0, 241, 6)], dtype="timedelta64[h]"
    ).astype("timedelta64[ns]")
    ds["surface_wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    ds["surface_wind_from_direction"] = (
        np.degrees(np.arctan2(ds["u10"], ds["v10"])) % 360
    )
    return ds


# Load case data from the default events.yaml
# Users can also define their own cases_dict structure
case_yaml = cases.load_ewb_events_yaml_into_case_collection()

# Define targets
# ERA5 target
era5_freeze_target = inputs.ERA5(
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    chunks=None,
)

# GHCN target
ghcn_freeze_target = inputs.GHCN(
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
)

# Define forecast (FCNv2 CIRA Virtualizarr)
fcnv2_forecast = inputs.KerchunkForecast(
    name="fcnv2_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

metrics_list = [
    metrics.RMSE,
    metrics.MinimumMAE,
    metrics.OnsetME,
    metrics.DurationME,
]
# Create a list of evaluation objects for freeze
freeze_evaluation_object = [
    inputs.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=ghcn_freeze_target,
        forecast=fcnv2_forecast,
    ),
    inputs.EvaluationObject(
        event_type="freeze",
        metric_list=metrics_list,
        target=era5_freeze_target,
        forecast=fcnv2_forecast,
    ),
]

# Initialize ExtremeWeatherBench runner instance
ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=freeze_evaluation_object,
)

# Run the workflow
outputs = ewb.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts for certain metrics
    # such as minimum temperature MAE
    tolerance_range=48,
    # precompute is false by default, but can be set to True to avoid IO costs loading
    # the datasets into memory for each metric
    pre_compute=False,
)

# Print the outputs; can be saved if desired
logger.info(outputs.head())
