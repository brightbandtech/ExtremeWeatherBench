'''
Core methods for various validation benchmarks
'''

from dataclasses import dataclass
import typing
import numpy as np
import pandas as pd
import xarray as xr

@dataclass
class ForecastData:
    forecast_ds: xr.Dataset

@dataclass
class AnalysisData:
    analysis_ds: xr.Dataset

class Metric:
    def __init__(self, date_range: pd.date_range):
        self.date_range = date_range
        self.date_begin = self.date_range[0]
        self.date_end = self.date_range[-1]
    




