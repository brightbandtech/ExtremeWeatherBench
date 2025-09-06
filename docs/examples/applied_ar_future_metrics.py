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
    variables=[
        derived.IntegratedVaporTransport,
        derived.IntegratedVaporTransportLaplacian,
        derived.AtmosphericRiverMask,
        derived.AtmosphericRiverLandIntersection,
    ],
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
    variables=[
        derived.IntegratedVaporTransport,
        derived.IntegratedVaporTransportLaplacian,
        derived.AtmosphericRiverMask,
        derived.AtmosphericRiverLandIntersection,
    ],
    variable_mapping={
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# %%
# just one for now
ar_metric_list = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric=[
            metrics.SpatialDisplacement(
                forecast_variable=derived.IntegratedVaporTransport.name,
                target_variable=derived.IntegratedVaporTransport.name,
                forecast_mask_variable=derived.AtmosphericRiverLandIntersection.name,
                target_mask_variable=derived.AtmosphericRiverLandIntersection.name,
            ),
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
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)
