import logging
import operator

import numpy as np
import xarray as xr

from extremeweatherbench import derived, inputs

# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)


# The core coordinate variables that are always required, even if not dimensions
# (e.g. latitude and longitude for xarray datasets)
DEFAULT_COORDINATE_VARIABLES = [
    "valid_time",
    "lead_time",
    "init_time",
    "latitude",
    "longitude",
]


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


# ERA5 targets
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    storage_options={"remote_options": {"anon": True}},
)

era5_freeze_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        "surface_air_temperature",
    ],
    storage_options={"remote_options": {"anon": True}},
)


era5_atmospheric_river_target = inputs.ERA5(
    variables=[derived.AtmosphericRiverVariables()],
)

# GHCN targets
ghcn_heatwave_target = inputs.GHCN(
    variables=["surface_air_temperature"],
)

ghcn_freeze_target = inputs.GHCN(
    variables=[
        "surface_air_temperature",
    ],
    storage_options={},
)

# LSR/PPH target
lsr_target = inputs.LSR(
    variables=["local_storm_reports"],
    storage_options={"remote_options": {"anon": True}},
)

pph_target = inputs.PPH(
    variables=["practically_perfect_hindcast"],
    variable_mapping={},
    storage_options={"remote_options": {"anon": True}},
)

# IBTrACS target

# TODO: Re-enable when IBTrACS target is implemented
# ibtracs_target = inputs.IBTrACS(
#     source=inputs.IBTRACS_URI,
#     variables=[derived.TCTrackVariables()],
#     variable_mapping={
#         "vmax": "surface_wind_speed",
#         "slp": "air_pressure_at_mean_sea_level",
#     },
#     storage_options={"remote_options": {"anon": True}},
# )

# Forecast Examples

cira_heatwave_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_freeze_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_atmospheric_river_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.AtmosphericRiverVariables()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_severe_convection_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.CravenSignificantSevereParameter()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


def get_brightband_evaluation_objects() -> list[inputs.EvaluationObject]:
    """Get the default Brightband list of evaluation objects.

    This is a function which will update as new event types are added to the project
    prior to feature completion.

    Returns:
        A list of EvaluationObject instances matching the complete Brightband eval
        routine.
    """
    # Import metrics here to avoid circular import
    from extremeweatherbench import metrics

    heatwave_metric_list: list[metrics.BaseMetric] = [
        metrics.MaximumMAE(),
        metrics.RMSE(),
        metrics.OnsetME(),
        metrics.DurationME(),
        metrics.MaxMinMAE(),
    ]
    freeze_metric_list: list[metrics.BaseMetric] = [
        metrics.MinimumMAE(),
        metrics.RMSE(),
        metrics.OnsetME(),
        metrics.DurationME(),
    ]

    return [
        inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=heatwave_metric_list,
            target=era5_heatwave_target,
            forecast=cira_heatwave_forecast,
        ),
        inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=heatwave_metric_list,
            target=ghcn_heatwave_target,
            forecast=cira_heatwave_forecast,
        ),
        inputs.EvaluationObject(
            event_type="freeze",
            metric_list=freeze_metric_list,
            target=era5_freeze_target,
            forecast=cira_freeze_forecast,
        ),
        inputs.EvaluationObject(
            event_type="freeze",
            metric_list=freeze_metric_list,
            target=ghcn_freeze_target,
            forecast=cira_freeze_forecast,
        ),
        inputs.EvaluationObject(
            event_type="severe_convection",
            metric_list=[
                metrics.CSI(),
                metrics.FAR(),
                # Need to add regional hits/misses and hits/misses metrics
                # metrics.RegionalHitsMisses(),
                # metrics.HitsMisses(),
            ],
            target=lsr_target,
            forecast=cira_severe_convection_forecast,
        ),
        inputs.EvaluationObject(
            event_type="atmospheric_river",
            metric_list=[
                metrics.CSI(),
                metrics.SpatialDisplacement(),
                metrics.EarlySignal(
                    comparison_operator=operator.ge,
                    threshold=0.5,
                    spatial_aggregation="any",
                ),
            ],
            target=era5_atmospheric_river_target,
            forecast=cira_atmospheric_river_forecast,
        ),
        # TODO: Re-enable when tropical cyclone forecast is implemented
        # inputs.EvaluationObject(
        #     event_type="tropical_cyclone",
        #     metric_list=[
        #         metrics.EarlySignal(),
        #         metrics.LandfallDisplacement(),
        #         metrics.LandfallTimeME(),
        #         metrics.LandfallIntensityMAE(),
        #     ],
        #     target=ibtracs_target,
        #     forecast=cira_tropical_cyclone_forecast,
        # ),
    ]
