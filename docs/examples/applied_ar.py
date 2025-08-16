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
from extremeweatherbench import derived, evaluate, inputs, metrics, utils

# %%
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
case_yaml = utils.load_events_yaml()
test_yaml = {
    "cases": [
        n
        for n in case_yaml["cases"]
        if n["start_date"].year < 2023 and n["event_type"] == "atmospheric_river"
    ][5:6]
}

# %%
era5_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[derived.IntegratedVaporTransport],
    variable_mapping={
        "specific_humidity": "specific_humidity",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# %%
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.IntegratedVaporTransport],
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
ar_metric_list = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric=[
            metrics.MAE,
        ],
        target=era5_target,
        forecast=hres_forecast,
    ),
]
# %%
test_ewb = evaluate.ExtremeWeatherBench(
    cases=test_yaml,
    metrics=ar_metric_list,
)
logger.info("Starting EWB run")
outputs = test_ewb.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)
