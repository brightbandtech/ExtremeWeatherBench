import xarray as xr
import utils
import dataclasses
import typing as t
import pandas as pd
import os
import events as e

@dataclasses.dataclass
class Config:
    """
    Config class for EWB. Date ranges, output directories, and changes to
    the default behavior of the EWB can be set here. Defaults are prescribed
    for consistency but can be extended as desired.
    """
    output_dir: str = 'assets/data/outputs/'
    forecast_dir: str = 'assets/data/forecasts/'
    model_init_time_freq: str = '6h'
    era5_link: str = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    events: t.List[e._Event] = dataclasses.field(default_factory=lambda: [e.HeatWave(), e.Freeze()])

    def __post_init__(self):
        forecast_extensions = set()
        if not os.path.exists(self.forecast_dir):
            raise ValueError('Forecast directory does not exist.')
        for _, _, files in os.walk(self.forecast_dir):
            for file in files:
                ext = os.path.splitext(file)[1]
                forecast_extensions.add(ext)
        if len(forecast_extensions) > 1:
            raise ValueError('Multiple forecast file extensions found in forecast directory.')
        self.forecast_extension = self.get_forecast_extension(forecast_extensions)
    
    def get_forecast_extension(self, forecast_extensions):
        return forecast_extensions.pop()
    

# To enable both lower and uppercase imports of config    
config = Config