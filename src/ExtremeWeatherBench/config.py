import xarray as xr

import dataclasses
import typing as t
import pandas as pd
import os
import logging

from . import utils
from . import events


@dataclasses.dataclass
class Config:
    """
    Config class for EWB. Date ranges, output directories, and changes to
    the default behavior of the EWB can be set here. Defaults are prescribed
    for consistency but can be extended as desired.
    Attributes:
        output_dir: (optional) the directory to store output files
        forecast_dir: (optional) the directory where forecast files are located
        era5_link: (optional) the link to the ERA5 dataset
        event_types: t.List[events._Event]: a list of event types to evaluate
    """
    output_dir: str = 'assets/data/outputs/'
    forecast_dir: str = 'assets/data/forecasts/'
    era5_link: str = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2" 
    event_types: t.List[events._Event] = dataclasses.field(default_factory=lambda: [events.HeatWave(), events.Freeze()])
    obs_parquet_link: str = "tbd"

    def evaluate(self):
        self.era5 = xr.open_zarr(self.era5_link, chunks='auto')
        self.parquet = pd.read_parquet(self.obs_parquet_link)
        for event in self.event_types:
            _evaluate_event_loop(event, self.era5)


def _evaluate(case: pd.core.frame):
    cases = case.generate_cases()
    event.metrics()
    return event

def _evaluate_event_loop(event: events._Event, era5: xr.Dataset):
    
    for case in event.case_df.itertuples():
        _evaluate(case)