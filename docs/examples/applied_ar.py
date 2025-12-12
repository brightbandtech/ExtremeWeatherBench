import datetime
import logging

import numpy as np
import xarray as xr

from extremeweatherbench import cases, derived, evaluate, inputs, metrics

# %%

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
case_yaml = case_yaml.select_cases(by="case_id_number", value=114)

case_yaml.cases[0].start_date = datetime.datetime(2022, 12, 27, 11, 0, 0)
case_yaml.cases[0].end_date = datetime.datetime(2022, 12, 27, 13, 0, 0)
# Define ERA5 target
era5_target = inputs.ERA5(
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
)

# Define forecast (HRES)
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    variable_mapping=inputs.HRES_metadata_variable_mapping,
)

grap_forecast = inputs.KerchunkForecast(
    name="Graphcast",
    source="gs://extremeweatherbench/GRAP_v100_IFS.parq",
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

pang_forecast = inputs.KerchunkForecast(
    name="Pangu",
    source="gs://extremeweatherbench/PANG_v100_IFS.parq",
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)
# Create a list of evaluation objects for atmospheric river
ar_evaluation_objects = [
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            metrics.CriticalSuccessIndex(),
            metrics.EarlySignal(),
            metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=hres_forecast,
    ),
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            metrics.CriticalSuccessIndex(),
            metrics.EarlySignal(),
            metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=grap_forecast,
    ),
    inputs.EvaluationObject(
        event_type="atmospheric_river",
        metric_list=[
            metrics.CriticalSuccessIndex(),
            metrics.EarlySignal(),
            metrics.SpatialDisplacement(),
        ],
        target=era5_target,
        forecast=pang_forecast,
    ),
]

if __name__ == "__main__":
    # Initialize ExtremeWeatherBench; will only run on cases with event_type
    # atmospheric_river
    ar_ewb = evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=ar_evaluation_objects,
    )

    # Run the workflow using 3 jobs
    outputs = ar_ewb.run(parallel_config={"backend": "loky", "n_jobs": 3})

    # Save the evaluation outputs to a csv file
    outputs.to_csv("ar_signal_outputs.csv")
