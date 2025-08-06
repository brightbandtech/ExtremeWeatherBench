import logging

import concrete_refactor_script as crs
import numpy as np
import refactor_scripts as rs
import xarray as xr

from extremeweatherbench import utils

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

case_yaml = utils.read_event_yaml(
    "/home/taylor/ExtremeWeatherBench/src/extremeweatherbench/data/events.yaml"
)
test_yaml = {"cases": [case_yaml["cases"][0], case_yaml["cases"][40]]}

incoming_forecast = crs.KerchunkForecast(
    forecast_source="gs://extremeweatherbench/FOUR_v200_GFS.parq"
)


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


# multiple events
heatwave_metric_list = [
    rs.MetricEvaluationObject(
        event_type="heat_wave",
        metric=[
            crs.MaximumMAE,
            crs.RMSE,
            crs.OnsetME,
            crs.DurationME,
            crs.MaxMinMAE,
        ],
        target=crs.ERA5,
        forecast=incoming_forecast,
        target_variables=["surface_air_temperature"],
        forecast_variables=["surface_air_temperature"],
        target_storage_options={"remote_options": {"anon": True}},
        forecast_storage_options={
            "remote_protocol": "s3",
            "remote_options": {"anon": True},
        },
        target_variable_mapping={
            "2m_temperature": "surface_air_temperature",
            "time": "valid_time",
        },
        forecast_variable_mapping={"t2": "surface_air_temperature"},
    )
]

freeze_metric_list = [
    rs.MetricEvaluationObject(
        event_type="freeze",
        metric=[crs.RMSE, crs.OnsetME],
        target=crs.LSR,
        forecast=incoming_forecast,
        target_variables=[
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
        ],
        forecast_variables=[
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
        ],
        target_storage_options={"remote_options": {"anon": True}},
        forecast_storage_options={
            "remote_protocol": "s3",
            "remote_options": {"anon": True},
        },
        target_variable_mapping={
            "2m_temperature": "surface_air_temperature",
            "10m_u_component_of_wind": "surface_eastward_wind",
            "10m_v_component_of_wind": "surface_northward_wind",
        },
        forecast_variable_mapping={
            "t2": "surface_air_temperature",
            "u10": "surface_eastward_wind",
            "v10": "surface_northward_wind",
        },
    )
]

test_ewb = rs.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=heatwave_metric_list + freeze_metric_list,
    forecast=incoming_forecast,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(
    forecast_preprocess_function=_preprocess_bb_cira_forecast_dataset,
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)
