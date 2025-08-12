# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import logging

# %%
import numpy as np
import xarray as xr

# %%
from extremeweatherbench import derived, evaluate, inputs, metrics, utils

# %%
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
case_yaml = utils.load_events_yaml()
test_yaml = {"cases": [case_yaml["cases"][200]]}

# %%
test_yaml


# %%
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
    return ds


# %%
ibtracs_target = inputs.IBTrACS(
    source=inputs.IBTRACS_URI,
    variables=[],
    variable_mapping=inputs.IBTrACS_metadata_variable_mapping,
    storage_options={"anon": True},
)

# %%
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.TCTrack],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
        "lead_time": "prediction_timedelta",
        "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
        "10m_wind_speed": "surface_wind_speed",
    },
    storage_options={"remote_options": {"anon": True}},
)

# %%
# just one for now
tc_metric_list = [
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric=[
            metrics.MAE,
        ],
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
]

# %%
hres_forecast

# %%
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=tc_metric_list,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)

# %%
