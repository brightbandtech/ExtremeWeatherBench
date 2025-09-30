import logging

import numpy as np
import xarray as xr

from extremeweatherbench import derived, inputs

# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)

# Columns for the evaluation output dataframe
OUTPUT_COLUMNS = [
    "value",
    "lead_time",
    "init_time",
    "target_variable",
    "metric",
    "case_id_number",
    "event_type",
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
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

era5_freeze_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

era5_atmospheric_river_target = inputs.ERA5(
    variables=[
        derived.AtmosphericRiverMask,
    ],
)

# GHCN targets
ghcn_heatwave_target = inputs.GHCN(
    variables=["surface_air_temperature"],
)

ghcn_freeze_target = inputs.GHCN(
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
)

# LSR/PPH target
lsr_target = inputs.LSR()

pph_target = inputs.PPH()

# IBTrACS target
ibtracs_target = inputs.IBTrACS()

# Forecast Examples

cira_heatwave_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_freeze_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_atmospheric_river_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        derived.AtmosphericRiverMask,
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

cira_tropical_cyclone_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        derived.TropicalCycloneTrackVariables,
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)
cira_severe_convection_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.CravenBrooksSignificantSevere],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)


def get_brightband_evaluation_objects() -> list[inputs.EvaluationObject]:
    """Get the default Brightband list of evaluation objects.

    This function defers the import of metrics to avoid circular imports.
    """
    # Import metrics here to avoid circular import
    from extremeweatherbench import metrics

    return [
        inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=[
                metrics.MaximumMAE,
                metrics.RMSE,
                metrics.OnsetME,
                metrics.DurationME,
                metrics.MaxMinMAE,
            ],
            target=era5_heatwave_target,
            forecast=cira_heatwave_forecast,
        ),
        inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=[
                metrics.MaximumMAE,
                metrics.RMSE,
                metrics.MaxMinMAE,
            ],
            target=ghcn_heatwave_target,
            forecast=cira_heatwave_forecast,
        ),
        inputs.EvaluationObject(
            event_type="freeze",
            metric_list=[
                metrics.MinimumMAE,
                metrics.RMSE,
                metrics.OnsetME,
                metrics.DurationME,
            ],
            target=era5_freeze_target,
            forecast=cira_freeze_forecast,
        ),
        inputs.EvaluationObject(
            event_type="freeze",
            metric_list=[
                metrics.MinimumMAE,
                metrics.RMSE,
            ],
            target=ghcn_freeze_target,
            forecast=cira_freeze_forecast,
        ),
        # TODO: Re-enable when severe convection forecast is implemented
        # inputs.EvaluationObject(
        #     event_type="severe_convection",
        #     metric_list=[
        #         metrics.CSI,
        #         metrics.FAR,
        #         metrics.RegionalHitsMisses,
        #         metrics.HitsMisses,
        #     ],
        #     target=lsr_target,
        #     forecast=cira_severe_convection_forecast,
        # ),
        inputs.EvaluationObject(
            event_type="atmospheric_river",
            metric_list=[metrics.CSI, metrics.SpatialDisplacement, metrics.EarlySignal],
            target=era5_atmospheric_river_target,
            forecast=cira_atmospheric_river_forecast,
        ),
        inputs.EvaluationObject(
            event_type="tropical_cyclone",
            metric_list=[
                metrics.EarlySignal,
                metrics.LandfallDisplacement,
                metrics.LandfallTimeME,
                metrics.LandfallIntensityMAE,
            ],
            target=ibtracs_target,
            forecast=cira_tropical_cyclone_forecast,
        ),
    ]
