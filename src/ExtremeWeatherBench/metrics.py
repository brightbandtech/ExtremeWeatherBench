import xarray as xr
from typing import Union
import numpy as np
import scores
import dataclasses
from sklearn.metrics import mean_squared_error

from . import utils

@dataclasses.dataclass
class Metric:
    """
    A parent class for a metric to evaluate a forecast. 
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """
        raise NotImplementedError


@dataclasses.dataclass
class DurationME(Metric):
    """
    The mean error in the duration of an event.
    """
    threshold: float
    threshold_tolerance: float

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class RegionalRMSE(Metric): 
    """
    The root mean squared error of the forecasted regional mean value.
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """

        raise NotImplementedError
    
@dataclasses.dataclass
class MaximumMAE(Metric):
    """
    The mean absolute error of the forecasted maximum value.
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """
        print(forecast)
        print(observation)
        
        max_t2_times = merged_df.reset_index().groupby('init_time').apply(lambda x: x.loc[x['t2'].idxmax()])
        max_t2_times['model'] = 'PanguWeather'
        max_t2_times = max_t2_times[max_t2_times.index < era5_dataset.case_analysis_ds['time'][era5_dataset.case_analysis_ds['2m_temperature'].mean(['latitude','longitude']).argmax().values].values]
        max_t2_times['time_error'] = abs(max_t2_times['time'] - era5_dataset.case_analysis_ds['time'][era5_dataset.case_analysis_ds['2m_temperature'].mean(['latitude','longitude']).argmax().values].values) / np.timedelta64(1, 'h')
        max_t2_times['t2_mae'] = abs(max_t2_times['t2'] - era5_dataset.case_analysis_ds['2m_temperature'].mean(['latitude','longitude']).max().values)
        merged_pivot = max_t2_times.pivot(index='model', columns='init_time', values='t2_mae')        
        raise NotImplementedError
    
@dataclasses.dataclass
class MaxMinMAE(Metric):
    """
    The mean absolute error of the forecasted highest minimum value,
    rolled up by a predefined time interval (e.g. daily).
    """
    time_interval: str

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """
        raise NotImplementedError

    
@dataclasses.dataclass
class OnsetME(Metric):
    """
    The mean error in the onset of an event, in hours.
    Attributes:
        endpoint_extension_criteria: float: the number of hours beyond the event window
        to include
    """
    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """
        Compute the metric.
        """

        raise NotImplementedError