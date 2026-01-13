import logging

import numpy as np
import xarray as xr

import extremeweatherbench as ewb
from extremeweatherbench import calc

# Set the logger level to INFO
logger = logging.getLogger("extremeweatherbench")
logger.setLevel(logging.INFO)


# Preprocessing function for CIRA data that includes geopotential thickness calculation
# required for tropical cyclone tracks
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


# Preprocessing function for HRES data that includes geopotential thickness calculation
# required for tropical cyclone tracks
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


# Load the case collection from the YAML file
case_yaml = ewb.load_cases()

# Select single case (TC Ida)
case_yaml.select_cases(by="case_id_number", value=220, inplace=True)

# Define IBTrACS target, no arguments needed as defaults are sufficient
ibtracs_target = ewb.targets.IBTrACS()

# Define HRES forecast
hres_forecast = ewb.forecasts.ZarrForecast(
    name="hres_forecast",
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    # Define tropical cyclone track derivedvariable to include in the forecast
    variables=[ewb.derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for HRES forecast
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
    # Preprocess the HRES forecast to include geopotential thickness calculation
    preprocess=_preprocess_hres_forecast_dataset,
)

# Define FCNv2 forecast
fcnv2_forecast = ewb.forecasts.KerchunkForecast(
    name="fcn_forecast",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[ewb.derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for FCNv2 forecast
    variable_mapping=ewb.CIRA_metadata_variable_mapping,
    # Preprocess the FCNv2 forecast to include geopotential thickness calculation
    preprocess=_preprocess_bb_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

# Define Pangu forecast
pangu_forecast = ewb.forecasts.KerchunkForecast(
    name="pangu_forecast",
    source="gs://extremeweatherbench/PANG_v100_GFS.parq",
    variables=[ewb.derived.TropicalCycloneTrackVariables()],
    # Define metadata variable mapping for Pangu forecast
    variable_mapping=ewb.CIRA_metadata_variable_mapping,
    # Preprocess the Pangu forecast to include geopotential thickness calculation
    # which uses the same preprocessing function as the FCNv2 forecast
    preprocess=_preprocess_bb_cira_tc_forecast_dataset,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


# Define composite metric for tropical cyclone track metrics. Using a composite metric
# prevents recomputation of landfalls, saving significant time. approach="next" sets
# the evaluation to occur, in the case of multiple landfalls, for the next landfall in
# time to be evaluated against
composite_landfall_metrics = [
    ewb.metrics.LandfallMetric(
        metrics=[
            ewb.metrics.LandfallIntensityMeanAbsoluteError,
            ewb.metrics.LandfallTimeMeanError,
            ewb.metrics.LandfallDisplacement,
        ],
        approach="next",
        # Set the intensity variable to use for the metric
        forecast_variable="air_pressure_at_mean_sea_level",
        target_variable="air_pressure_at_mean_sea_level",
    )
]

# Define evaluation objects for tropical cyclone metrics. Setting event type subsets
# the relevant cases inside the events YAML file
tc_evaluation_object = [
    # HRES forecast
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=hres_forecast,
    ),
    # Pangu forecast
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=pangu_forecast,
    ),
    # FCNv2 forecast
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=composite_landfall_metrics,
        target=ibtracs_target,
        forecast=fcnv2_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench
    tc_ewb = ewb.evaluation(
        case_metadata=case_yaml,
        evaluation_objects=tc_evaluation_object,
    )
    logger.info("Starting EWB run")
    # Run the workflow with parallel_config backend set to dask
    outputs = tc_ewb.run(
        parallel_config={"backend": "loky", "n_jobs": 3},
    )
    outputs.to_csv("tc_metric_test_results.csv")
