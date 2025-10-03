import itertools
import logging
from pathlib import Path
from typing import Callable, Union

import numpy as np
import xarray as xr

from extremeweatherbench import derived, inputs, utils

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

# TODO: Re-enable when atmospheric river target is implemented
# era5_atmospheric_river_target = inputs.ERA5(
#     source=inputs.ARCO_ERA5_FULL_URI,
#     variables=[
#         derived.IntegratedVaporTransport,
#         derived.AtmosphericRiverMask,
#     ],
#     variable_mapping={
#         "u_component_of_wind": "eastward_wind",
#         "v_component_of_wind": "northward_wind",
#         "temperature": "air_temperature",
#         "vertical_integral_of_northward_water_vapour_flux":
#             "northward_water_vapour_flux",
#         "vertical_integral_of_eastward_water_vapour_flux":
#             "eastward_water_vapour_flux",
#     },
#     storage_options={"remote_options": {"anon": True}},
# )

# GHCN targets
ghcn_heatwave_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={},
)

ghcn_freeze_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "surface_temperature": "surface_air_temperature",
        "surface_eastward_wind": "surface_eastward_wind",
        "surface_northward_wind": "surface_northward_wind",
    },
    storage_options={},
)

# LSR/PPH target
# TODO: Re-enable when severe convection is implemented
# lsr_target = inputs.LSR(
#     source=inputs.LSR_URI,
#     variables=["local_storm_reports"],
#     variable_mapping={},
#     storage_options={"remote_options": {"anon": True}},
# )

# pph_target = inputs.PPH(
#     source=inputs.PPH_URI,
#     variables=["practically_perfect_hindcast"],
#     variable_mapping={},
#     storage_options={"remote_options": {"anon": True}},
# )

# IBTrACS target

# TODO: Re-enable when IBTrACS target is implemented
# ibtracs_target = inputs.IBTrACS(
#     source=inputs.IBTRACS_URI,
#     variables=[derived.TCTrackVariables],
#     variable_mapping={
#         "vmax": "surface_wind_speed",
#         "slp": "air_pressure_at_mean_sea_level",
#     },
#     storage_options={"remote_options": {"anon": True}},
# )

# Forecast Examples

cira_heatwave_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
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
    variable_mapping={
        "t2": "surface_air_temperature",
        "10u": "surface_eastward_wind",
        "10v": "surface_northward_wind",
    },
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
)

# TODO: Re-enable when atmospheric river forecast is implemented
# cira_atmospheric_river_forecast = inputs.KerchunkForecast(
#     source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
#     variables=[
#         derived.IntegratedVaporTransport,
#         derived.AtmosphericRiverMask,
#     ],
#     variable_mapping={
#         "u_component_of_wind": "eastward_wind",
#         "v_component_of_wind": "northward_wind",
#         "specific_humidity": "specific_humidity",
#         "temperature": "air_temperature",
#         "vertical_integral_of_northward_water_vapour_flux":
#             "northward_water_vapour_flux",
#         "vertical_integral_of_eastward_water_vapour_flux":
#             "eastward_water_vapour_flux",
#     },
#     storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
# )

# TODO: Re-enable when CravenSignificantSevereParameter is implemented
# cira_severe_convection_forecast = inputs.KerchunkForecast(
#     source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
#     variables=[derived.CravenSignificantSevereParameter],
#     variable_mapping={
#         "t": "air_temperature",
#         "t2": "surface_air_temperature",
#         "z": "geopotential",
#         "r": "relative_humidity",
#         "u": "eastward_wind",
#         "v": "northward_wind",
#         "10u": "surface_eastward_wind",
#         "10v": "surface_northward_wind",
#     },
#     storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
# )


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
                metrics.OnsetME,
                metrics.DurationME,
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
                metrics.OnsetME,
                metrics.DurationME,
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
        # TODO: Re-enable when atmospheric river forecast is implemented
        # inputs.EvaluationObject(
        #     event_type="atmospheric_river",
        #     metric_list=[metrics.CSI, metrics.SpatialDisplacement,
        #  metrics.EarlySignal],
        #     target=era5_atmospheric_river_target,
        #     forecast=cira_atmospheric_river_forecast,
        # ),
        # TODO: Re-enable when tropical cyclone forecast is implemented
        # inputs.EvaluationObject(
        #     event_type="tropical_cyclone",
        #     metric_list=[
        #         metrics.EarlySignal,
        #         metrics.LandfallDisplacement,
        #         metrics.LandfallTimeME,
        #         metrics.LandfallIntensityMAE,
        #     ],
        #     target=ibtracs_target,
        #     forecast=cira_tropical_cyclone_forecast,
        # ),
    ]


def build_default_forecast_object(
    forecast_source: Union[Path, str],
    event_type: str,
    variable_mapping: dict,
    storage_options: dict = {},
    preprocess: Callable = utils._default_preprocess,
) -> inputs.ForecastBase:
    """Build a forecast object from a given source.

    Args:
        forecast_source: The forecast source to use for the forecast object. Can be
            a path to a local file or a remote URI.
        variables: The variables to use for the forecast object.
        variable_mapping: The variable mapping to use for the forecast object.
        storage_options: The storage options to use for the forecast object.
        preprocess: The preprocess function to use for the forecast object.
    """
    match event_type:
        case "heat_wave":
            variables = ["surface_air_temperature"]
        case "freeze":
            variables = ["surface_air_temperature"]
        case "severe_convection":
            variables = [derived.CravenSignificantSevereParameter]
        case "atmospheric_river":
            variables = [derived.AtmosphericRiverMask]
        case "tropical_cyclone":
            variables = [derived.TropicalCycloneTrackVariables]
        case _:
            raise ValueError(f"Unknown event type: {event_type}")
    # Convert to string if Path, check file type, and build the forecast object
    if isinstance(forecast_source, Path):
        forecast_source = forecast_source.as_posix()

    # Build the forecast object based on the file type
    if (
        forecast_source.endswith(".parq")
        or forecast_source.endswith(".parquet")
        or forecast_source.endswith(".json")
    ):
        return inputs.KerchunkForecast(
            source=forecast_source,
            variables=variables,
            variable_mapping=variable_mapping,
            storage_options=storage_options,
            preprocess=preprocess,
        )
    elif forecast_source.endswith(".zarr"):
        return inputs.ZarrForecast(
            source=forecast_source,
            variables=variables,
            variable_mapping=variable_mapping,
            storage_options=storage_options,
            preprocess=preprocess,
        )
    else:
        raise ValueError(
            f"Unknown forecast file type found in forecast path, only "
            f"parquet, json, and zarr are supported. Found {forecast_source}"
        )


def build_default_evaluation_objects_for_forecast(
    forecast_source: Union[Path, str],
    event_types: Union[list[str], str],
    variable_mapping: dict,
    storage_options: dict = {},
    preprocess: Callable = utils._default_preprocess,
) -> list[inputs.EvaluationObject]:
    """Get the default evaluation objects for a given forecast source and event
    types.

    Args:
        forecast_source: The forecast source to use for the evaluation objects. Can be
            a path to a local file or a remote URI.
        event_types: The event types to use for the evaluation objects.
        variable_mapping: The variable mapping to use for the forecast object. Suggested
        to use one single mapping dictionary for all variables in the forecast.
        storage_options: The storage options to use for the forecast object.
        preprocess: The preprocess function to use for the forecast object.
    Returns:
        A list of evaluation objects.
    """

    # Convert to list if string
    if isinstance(event_types, str):
        event_types = [event_types]

    # Convert to string if Path
    if isinstance(forecast_source, Path):
        forecast_source = forecast_source.as_posix()

    # Build the evaluation objects by event type. For the default evaluation objects,
    # check if the event type matches the chosen event types that have already been used
    # to build the forecast objects.
    default_evaluation_objects = get_brightband_evaluation_objects()
    evaluation_objects = []
    for eval_obj, event_type in itertools.product(
        default_evaluation_objects, event_types
    ):
        if event_type == eval_obj.event_type:
            # Inject the new forecast object. variable mapping is required between your
            # forecast and the variables in EWB
            eval_obj.forecast = build_default_forecast_object(
                forecast_source=forecast_source,
                event_type=event_type,
                variable_mapping=variable_mapping,
                storage_options=storage_options,
                preprocess=preprocess,
            )
            evaluation_objects.append(eval_obj)
    return evaluation_objects
