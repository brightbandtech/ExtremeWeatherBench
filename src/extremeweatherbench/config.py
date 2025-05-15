"""Module defining congifuration objects for driving ExtremeWeatherBench analyses."""

import dataclasses
from pathlib import Path
from typing import Callable, List, Optional, Union

import xarray as xr

from extremeweatherbench import events

DATA_DIR = Path("./data")
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"
DEFAULT_FORECAST_DIR = DATA_DIR / "forecasts"
DEFAULT_CACHE_DIR = DATA_DIR / "cache"

#: Storage/access options for gridded observation datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point observation dataset.
DEFAULT_POINT_OBS_URI = "gs://extremeweatherbench/ghcnh.parq"

#: Storage/access options for point observation datasets.
POINT_OBS_STORAGE_OPTIONS = dict(token="anon")

#: Storage/access options for gridded observation datasets.
GRIDDED_OBS_STORAGE_OPTIONS = dict(token="anon")


def _default_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Default forecast preprocess function that does nothing."""
    return ds


@dataclasses.dataclass
class ForecastSchemaConfig:
    """A mapping between standard variable names used across EWB, and their counterpart
    in a forecast dataset.

    Allows users to insert custom schemas for decoding forecast data. Defaults are
    suggested based on the CF Conventions.
    """

    surface_air_temperature: Optional[str] = "t2"
    surface_eastward_wind: Optional[str] = "u10"
    surface_northward_wind: Optional[str] = "v10"
    air_temperature: Optional[str] = "t"
    eastward_wind: Optional[str] = "u"
    northward_wind: Optional[str] = "v"
    air_pressure_at_mean_sea_level: Optional[str] = "msl"
    lead_time: Optional[str] = "time"
    init_time: Optional[str] = "init_time"
    fhour: Optional[str] = "fhour"
    level: Optional[str] = "level"
    latitude: Optional[str] = "latitude"
    longitude: Optional[str] = "longitude"


@dataclasses.dataclass
class PointObservationSchemaConfig:
    """A mapping between standard variable names used across EWB, and their counterpart
    in a provided point observation dataset.

    Allows users to insert custom schemas for decoding default or custom point observation data.
    Defaults are suggested based on the CF Conventions.
    A successful configuration does not need to include all variables, only the required metadata variables
    (station, id, latitude, longitude, time) and the data variables desired for evaluation if not already named
    identically to the default values. Others will be ignored.

    Attributes:
        air_pressure_at_mean_sea_level: The variable name in the point observation dataset
        for the mean sea level pressure. Defaults to "air_pressure_at_mean_sea_level".
        surface_air_pressure: The surface air pressure. Defaults to "surface_air_pressure".
        surface_wind_speed: The surface wind speed. Defaults to "surface_wind_speed".
        surface_wind_from_direction: The surface wind from direction. Defaults to "surface_wind_from_direction".
        surface_air_temperature: The surface air temperature. Defaults to "surface_air_temperature".
        surface_dew_point_temperature: The surface dew point temperature. Defaults to "surface_dew_point_temperature".
        surface_relative_humidity: The surface relative humidity. Defaults to "surface_relative_humidity".
        accumulated_1_hour_precipitation: The accumulated 1 hour precipitation.
        Defaults to "accumulated_1_hour_precipitation".
        time: The time. Defaults to "time".
        latitude: The latitude. Defaults to "latitude".
        longitude: The longitude. Defaults to "longitude".
        elevation: The elevation. Defaults to "elevation".
        station_id: The station id. Defaults to "station".
        station_long_name: The station name. Defaults to "name".
        case_id: The case id in the event yaml file.
        Defaults to "id".
        metadata_vars: A list of metadata variables to include in the point observation dataset.
        Default is ["station", "id", "latitude", "longitude", "time"]. All five required for successful configuration.
    """

    air_pressure_at_mean_sea_level: Optional[str] = "air_pressure_at_mean_sea_level"
    surface_air_pressure: Optional[str] = "surface_air_pressure"
    surface_wind_speed: Optional[str] = "surface_wind_speed"
    surface_wind_from_direction: Optional[str] = "surface_wind_from_direction"
    surface_air_temperature: Optional[str] = "surface_air_temperature"
    surface_dew_point_temperature: Optional[str] = "surface_dew_point"
    surface_relative_humidity: Optional[str] = "surface_relative_humidity"
    accumulated_1_hour_precipitation: Optional[str] = "accumulated_1_hour_precipitation"
    time: Optional[str] = "time"
    latitude: Optional[str] = "latitude"
    longitude: Optional[str] = "longitude"
    elevation: Optional[str] = "elevation"
    station_id: Optional[str] = "station"
    station_long_name: Optional[str] = "name"
    case_id: Optional[str] = "id"
    metadata_vars: List[str] = dataclasses.field(
        default_factory=lambda: [
            "station",
            "id",
            "latitude",
            "longitude",
            "time",
        ]
    )

    def extend_metadata_vars(self, input_metadata_variables: List[str]):
        """Extend the metadata variables for the schema config if more than the default are needed."""
        self.metadata_vars.extend(input_metadata_variables)

    @property
    def mapped_metadata_vars(self):
        """Returns the dataclass field names mapped to the given input variable names."""
        return [
            field.name
            for field in dataclasses.fields(self)
            if field.default in self.metadata_vars
        ]


@dataclasses.dataclass
class Config:
    """High-level configuration for an ExtremeWeatherBench analysis run.

    The Config class defines data ranges, output directories, and any custom
    behavior that should used when running an end-to-end ExtremeWeatherBench
    analysis. We prescribe sensible defaults for consistency, but these can be
    extended as desired by the user.

    Attributes:
        event_types: A list of EventContainer event types to evaluate.
            Base options currently include: HeatWave and Freeze,
            Expanded options will include: SevereDay, AtmosphericRiver,
            and TropicalCyclone with others to follow.
        output_dir: A directory where outputs generated by the analysis should be saved.
        forecast_dir: A directory where forecast data to be analyzed by EWB is stored.
        cache_dir: A directory where intermediate data generated by an EWB analysis are stored.
               Set to None to disable caching.
        gridded_obs_path: A URI or filepath where a gridded observation dataset that can
            be used for evaluation is stored. Defaults to ARCO-ERA5 on Google Cloud
            Storage.
        point_obs_path: A URI or filepath where a point observation dataset
            that be used for evaluation is stored.
        remote_protocol: The storage protocol which the forecast data is stored on.
            Defaults to "s3".
        forecast_preprocess: A function that preprocesses the forecast dataset into a
            format expected by the evaluation metrics.
        init_forecast_hour: The first forecast hour to include in the evaluation.
            Defaults to 0.
        temporal_resolution_hours: The resolution of the forecast data in hours.
            Defaults to 6.
        output_timesteps: The number of timesteps to include in the evaluation.
            Defaults to 41.
        gridded_obs_storage_options: A dictionary of cloud storage options for the gridded
            observation dataset.
        point_obs_storage_options: A dictionary of cloud storage options for the point
            observation dataset.
    """

    event_types: List[events.EventContainer]
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR
    forecast_dir: Union[str, Path] = DEFAULT_FORECAST_DIR
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR
    gridded_obs_path: Union[str, Path] = ARCO_ERA5_FULL_URI
    point_obs_path: Union[str, Path] = DEFAULT_POINT_OBS_URI
    remote_protocol: str = "s3"
    forecast_preprocess: Callable[[xr.Dataset], xr.Dataset] = _default_preprocess
    init_forecast_hour: int = 0  # The first forecast hour (e.g. Graphcast starts at 6).
    temporal_resolution_hours: int = 6
    output_timesteps: int = 41
    gridded_obs_storage_options: dict = dataclasses.field(
        default_factory=lambda: GRIDDED_OBS_STORAGE_OPTIONS
    )
    point_obs_storage_options: dict = dataclasses.field(
        default_factory=lambda: POINT_OBS_STORAGE_OPTIONS
    )
    forecast_schema_config: ForecastSchemaConfig = dataclasses.field(
        default_factory=lambda: ForecastSchemaConfig()
    )
    point_obs_schema_config: PointObservationSchemaConfig = dataclasses.field(
        default_factory=lambda: PointObservationSchemaConfig()
    )
