import logging
import operator

import numpy as np
import xarray as xr

from extremeweatherbench import calc, derived, inputs

# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)


# The core coordinate variables that are selected if they exist, even if not dimensions
# (e.g. latitude and longitude for xarray datasets). These are used currently for
# selecting columns in pandas dataframes and polars lazyframes.
DEFAULT_COORDINATE_VARIABLES = [
    "valid_time",
    "lead_time",
    "init_time",
    "latitude",
    "longitude",
    "season",  # for tropical cyclone data
    "tc_name",  # for tropical cyclone data
    "number",  # for tropical cyclone data
]

DEFAULT_VARIABLE_NAMES = [
    "100m_eastward_wind",  # 100-meter u component of wind, m/s
    "100m_northward_wind",  # 100-meter v component of wind, m/s
    "air_pressure",  # Pa
    "air_pressure_at_mean_sea_level",  # mean sea level pressure, Pa
    "air_temperature",  # K
    "dewpoint_temperature",  # K
    "eastward_wind",  # u component of wind, m/s
    "geopotential",  # m^2/s^2
    "geopotential_height",  # m
    "init_time",  # initialization time of the forecast model; t0
    "latitude",  # degrees
    "lead_time",  # lead time of the forecast
    "level",  # pressure level of the data, hPa
    "longitude",  # degrees
    "northward_wind",  # v component of wind, m/s
    "relative_humidity",  # %
    "specific_humidity",  # kg/kg
    "storm_id",  # storm identifier for data such as tropical cyclones
    "surface_air_temperature",  # 2-meter temperature
    "surface_dewpoint_temperature",  # 2-meter dewpoint temperature, K
    "surface_eastward_wind",  # 10-meter u component of wind, m/s
    "surface_northward_wind",  # 10-meter v component of wind, m/s
    "surface_relative_humidity",  # 2-meter relative humidity, %
    "surface_specific_humidity",  # 2-meter specific humidity, kg/kg
    "surface_wind_from_direction",  # 10-meter wind direction, degrees (from, not to)
    "surface_wind_speed",  # 10-meter wind speed, m/s
    "valid_time",  # valid time of the data slice
    "wind_from_direction",  # degrees (from, not to)
    "wind_speed",  # m/s
]


def _preprocess_bb_cira_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """A preprocess function for CIRA data that renames the time coordinate to
    lead_time, creates a valid_time coordinate, and sets the lead time range and
    resolution not present in the original dataset.

    Args:
        ds: The forecast dataset to preprocess.

    Returns:
        The preprocessed forecast dataset.
    """
    ds = ds.rename({"time": "lead_time"})
    # The evaluation configuration is used to set the lead time range and resolution.
    ds["lead_time"] = np.array(
        [i for i in range(0, 241, 6)], dtype="timedelta64[h]"
    ).astype("timedelta64[ns]")
    return ds


# Preprocessing function for CIRA data that includes geopotential thickness calculation
# required for tropical cyclone tracks
def _preprocess_bb_cira_tc_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """A preprocess function for CIRA data that includes geopotential thickness
    calculation required for tropical cyclone tracks.

    This function renames the time coordinate to lead_time,
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

    # Calculate the geopotential thickness required for tropical cyclone tracks
    ds["geopotential_thickness"] = calc.geopotential_thickness(
        ds["z"], top_level_value=300, bottom_level_value=500
    )
    return ds


# Preprocessing function for HRES data that includes geopotential thickness calculation
# required for tropical cyclone tracks
def _preprocess_bb_hres_tc_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
    """A preprocess function for CIRA data that includes geopotential thickness
    calculation required for tropical cyclone tracks.

    This function renames the time coordinate to lead_time,
    creates a valid_time coordinate, and sets the lead time range and resolution not
    present in the original dataset.

    Args:
        ds: The forecast dataset to rename.

    Returns:
        The renamed forecast dataset.
    """

    # Calculate the geopotential thickness required for tropical cyclone tracks
    ds["geopotential_thickness"] = calc.geopotential_thickness(
        ds["geopotential"], top_level_value=300, bottom_level_value=500
    )
    return ds


# Preprocess function for CIRA data using Brightband kerchunk parquets
def _preprocess_bb_ar_cira_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
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
    if "q" not in ds.variables:
        # Calculate specific humidity from relative humidity and air temperature
        ds["specific_humidity"] = calc.specific_humidity_from_relative_humidity(
            air_temperature=ds["t"],
            relative_humidity=ds["r"],
            levels=ds["level"],
        )
    return ds


# Preprocess function for CIRA data using Brightband kerchunk parquets
def _preprocess_bb_severe_cira_forecast_dataset(ds: xr.Dataset) -> xr.Dataset:
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
    if "q" not in ds.variables:
        # Calculate specific humidity from relative humidity and air temperature
        ds["specific_humidity"] = calc.specific_humidity_from_relative_humidity(
            air_temperature=ds["t"],
            relative_humidity=ds["r"],
            levels=ds["level"],
        )
    ds["geopotential"] = ds["z"] * calc.g0
    return ds


def get_climatology(quantile: float = 0.85) -> xr.DataArray:
    """Get the climatology dataset for the heatwave criteria."""
    if quantile not in [0.15, 0.85]:
        raise ValueError("Quantile must be 0.15 or 0.85")
    return xr.open_zarr(
        "gs://extremeweatherbench/datasets/surface_air_temperature_1990_2019_climatology.zarr",  # noqa: E501
        storage_options={"anon": True},
        chunks="auto",
    )["2m_temperature"].sel(quantile=quantile)


# ERA5 targets
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    storage_options={"remote_options": {"anon": True}},
)

era5_freeze_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    storage_options={"remote_options": {"anon": True}},
)

era5_atmospheric_river_target = inputs.ERA5(
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    storage_options={"remote_options": {"anon": True}},
)

# GHCN targets
ghcn_heatwave_target = inputs.GHCN(
    variables=["surface_air_temperature"],
)

ghcn_freeze_target = inputs.GHCN(
    variables=["surface_air_temperature"],
    storage_options={},
)

# LSR/PPH target
lsr_target = inputs.LSR(
    storage_options={"remote_options": {"anon": True}},
)

pph_target = inputs.PPH(
    storage_options={"remote_options": {"anon": True}},
)

# IBTrACS target

ibtracs_target = inputs.IBTrACS()

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

cira_tropical_cyclone_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.TropicalCycloneTrackVariables()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_tc_forecast_dataset,
)
cira_atmospheric_river_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        derived.AtmosphericRiverVariables(
            output_variables=["atmospheric_river_land_intersection"]
        )
    ],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_ar_cira_forecast_dataset,
)

cira_severe_convection_forecast = inputs.KerchunkForecast(
    name="FourCastNetv2",
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[derived.CravenBrooksSignificantSevere()],
    variable_mapping=inputs.CIRA_metadata_variable_mapping,
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    preprocess=_preprocess_bb_cira_forecast_dataset,
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
        metrics.MaximumMeanAbsoluteError(),
        metrics.RootMeanSquaredError(),
        metrics.DurationMeanError(
            threshold_criteria=get_climatology(0.85), op_func=operator.ge
        ),
        metrics.MaximumLowestMeanAbsoluteError(),
    ]
    freeze_metric_list: list[metrics.BaseMetric] = [
        metrics.MinimumMeanAbsoluteError(),
        metrics.RootMeanSquaredError(),
        metrics.DurationMeanError(
            threshold_criteria=get_climatology(0.15), op_func=operator.le
        ),
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
                metrics.CriticalSuccessIndex,
                metrics.FalseAlarmRatio,
            ],
            target=lsr_target,
            forecast=cira_severe_convection_forecast,
        ),
        inputs.EvaluationObject(
            event_type="atmospheric_river",
            metric_list=[
                metrics.CriticalSuccessIndex,
                metrics.SpatialDisplacement,
                metrics.EarlySignal,
            ],
            target=era5_atmospheric_river_target,
            forecast=cira_atmospheric_river_forecast,
        ),
        inputs.EvaluationObject(
            event_type="tropical_cyclone",
            metric_list=[
                metrics.LandfallDisplacement,
                metrics.LandfallTimeMeanError,
                metrics.LandfallIntensityMeanAbsoluteError,
            ],
            target=ibtracs_target,
            forecast=cira_tropical_cyclone_forecast,
        ),
    ]
