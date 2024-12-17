import xarray as xr

import dataclasses
from typing import List, Optional
import pandas as pd
from . import events


@dataclasses.dataclass
class Config:
    """
    Config class for EWB. Date ranges, output directories, and changes to
    the default behavior of the EWB can be set here. Defaults are prescribed
    for consistency but can be extended as desired.
    Attributes:
        output_path: (optional) the directory to store output files
        forecast_path: (optional) the directory where forecast files are located
        era5_link: (optional) the link to the ARCO ERA5 dataset
        event_types: a list of event types to evaluate
        isd_obs_parquet_link: link to the ISD observations used for evaluation
        cache: (optional) whether to cache the data or not. If so, creates tmp folder in output directory
    """
    output_path: str = 'assets/data/outputs/'
    forecast_path: str = 'assets/data/forecasts/'
    gridded_obs_path: str = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2" 
    point_obs_path: Optional[str] = None
    event_types: List[events.Event] = dataclasses.field(default_factory=lambda: [events.HeatWave(), events.Freeze()])
    cache: bool = False


@dataclasses.dataclass
class ForecastSchemaConfig:
    """
    Config class to insert custom schema for forecast data. Defaults are prescribed
    based on the CIRA AI model schema (https://aiweather.cira.colostate.edu).
    Attributes:
    """
    t2: Optional[str] = 't2'
    u10: Optional[str] = 'u10'
    v10: Optional[str] = 'v10'
    msl: Optional[str] = 'msl'
    q: Optional[str] = 'q'
    time: Optional[str] = 'time'
    init_time: Optional[str] = 'init_time'
    fhour: Optional[str] = 'fhour'
    level: Optional[str] = 'level'
    latitude: Optional[str] = 'latitude'
    longitude: Optional[str] = 'longitude'