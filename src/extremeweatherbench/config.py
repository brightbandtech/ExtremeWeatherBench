"""Module defining congifuration objects for driving ExtremeWeatherBench analyses."""

import dataclasses
from typing import List, Optional

from extremeweatherbench import events

DEFAULT_OUTPUT_DIR = "assets/data/outputs/"
DEFAULT_FORECAST_DIR = "assets/data/forecasts/"
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ISD_POINT_OBS_URI = "gs://extremeweatherbench/isd_no_wr_qc.parquet"


@dataclasses.dataclass
class Config:
    """High-level configuration for an ExtremeWeatherBench analysis run.

    The Config class defines data ranges, output directories, and any custom
    behavior that should used when running an end-to-end ExtremeWeatherBench
    analysis. We prescribe sensible defaults for consistency, but these can be
    extended as desired by the user.

    Attributes:
        output_dir: A directory where outputs generated by the analysis should be saved.
        forecast_dir: A directory where forecast data to be analyzed by EWB is stored.
        gridded_obs_path: A URI or filepath where a gridded observation dataset that can
            be used for evaluation is stored. Defaults to ARCO-ERA5 on Google Cloud
            Storage.
        point_obs_path: (optional) A URI or filepath where a point observation dataset
            that be used for evaluation is stored.
        event_types: A list of event types to evaluate.
        cache: Enable caching of intermediate data and artifacts generated by an EWB
            analysis. Defaults to "False".
    """

    event_types: List[events.EventContainer]
    output_dir: str = DEFAULT_OUTPUT_DIR
    forecast_dir: str = DEFAULT_FORECAST_DIR
    gridded_obs_path: str = ARCO_ERA5_FULL_URI
    point_obs_path: str = ISD_POINT_OBS_URI
    cache: bool = False


@dataclasses.dataclass
class ForecastSchemaConfig:
    """A mapping between standard variable names used across EWB, and their counterpart
    in a forecast dataset.

    Allows users to insert custom schemas for decoding forecast data. Defaults are
    suggested based on the CF Conventions.
    """

    air_temperature: Optional[str] = "t2"
    eastward_wind: Optional[str] = "u10"
    northward_wind: Optional[str] = "v10"
    air_pressure_at_mean_sea_level: Optional[str] = "msl"
    lead_time: Optional[str] = "time"
    init_time: Optional[str] = "init_time"
    fhour: Optional[str] = "fhour"
    level: Optional[str] = "level"
    latitude: Optional[str] = "latitude"
    longitude: Optional[str] = "longitude"
