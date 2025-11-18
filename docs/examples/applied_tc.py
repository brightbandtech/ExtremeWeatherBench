import logging

import numpy as np
import xarray as xr
from dask.distributed import Client

from extremeweatherbench import calc, cases, derived, evaluate, inputs, metrics

logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig()
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.DEBUG)


def _preprocess_bb_cira_tc_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
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
    ds["geopotential_thickness"] = calc.geopotential_thickness(
        ds["z"], top_level_value=300, bottom_level_value=500
    )
    return ds


def _preprocess_hres_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """An example preprocess function that renames the time coordinate to lead_time,
    creates a valid_time coordinate, and sets the lead time range and resolution not
    present in the original dataset.

    Args:
        ds: The forecast dataset to rename.
    """
    ds["geopotential_thickness"] = calc.geopotential_thickness(
        ds["geopotential"],
        top_level_value=300,
        bottom_level_value=500,
        geopotential=True,
    )
    return ds


case_yaml = cases.load_ewb_events_yaml_into_case_collection()
case_yaml.select_cases(by="case_id_number", value=220, inplace=True)


ibtracs_target = inputs.IBTrACS()


hres_forecast = inputs.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=[derived.TropicalCycloneTrackVariables()],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    preprocess=_preprocess_hres_forecast_dataset,
)

fcn_forecast = inputs.KerchunkForecast(
    name="fcn_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.TropicalCycloneTrackVariables()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    preprocess=_preprocess_bb_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

pangu_forecast = inputs.KerchunkForecast(
    name="pangu_forecast",
    source="gs://extremeweatherbench/PANG_v100_GFS.parq",
    variables=[derived.TropicalCycloneTrackVariables()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    preprocess=_preprocess_bb_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


composite = metrics.LandfallMetric(
    metrics=[
        metrics.LandfallIntensityMAE,
        metrics.LandfallTimeME,
        metrics.LandfallDisplacement,
    ],
    approach="next",
    forecast_variable="air_pressure_at_mean_sea_level",
    target_variable="air_pressure_at_mean_sea_level",
)
# Note: compute_metric() requires DataArrays, not InputBase objects.
# Use the evaluation pipeline below to properly extract DataArrays and compute metrics.
# Evaluation objects for tropical cyclone metrics
# Note: Landfall metrics work with DataArrays that have lat/lon/time coords
# For intensity metrics, specify variables explicitly to evaluate different
# intensity measures (e.g., wind speed vs. pressure)
tc_evaluation_object = [
    # HRES forecast
    inputs.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=[composite],
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
    # # Pangu forecast
    # inputs.EvaluationObject(
    #     event_type="tropical_cyclone",
    #     metric_list=[
    #         metrics.LandfallTimeME(),
    #         metrics.LandfallDisplacement(),
    #         metrics.LandfallIntensityMAE(),
    #     ],
    #     target=ibtracs_target,
    #     forecast=pangu_forecast,
    # ),
    # # FCN forecast
    # inputs.EvaluationObject(
    #     event_type="tropical_cyclone",
    #     metric_list=[
    #         metrics.LandfallTimeME(),
    #         metrics.LandfallDisplacement(),
    #         metrics.LandfallIntensityMAE(),
    #     ],
    #     target=ibtracs_target,
    #     forecast=fcn_forecast,
    # ),
]
if __name__ == "__main__":
    test_ewb = evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=tc_evaluation_object,
    )
    with Client():
        logger.info("Starting EWB run")
        outputs = test_ewb.run(
            parallel_config={"backend": "dask", "n_jobs": 1},
        )
        outputs.to_csv("tc_metric_test_results.csv")
