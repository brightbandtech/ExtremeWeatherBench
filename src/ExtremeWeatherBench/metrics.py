import xarray as xr
import typing as t
import numpy as np
import scores
import dataclasses

from . import utils

@dataclasses.dataclass
class ERA5Metric:
    """
    A parent class for a metric based on ERA5 to evaluate a forecast. 
    Attributes:
        forecast: xr.Dataset: the forecast dataset
        observation: xr.Dataset: the ERA5 dataset of observations
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError



@dataclasses.dataclass
class ObsMetric:
    """
    A parent class for a metric based on observations to evaluate a forecast. 
    The metric is defined by a function that takes
    forecast and observation data arrays and returns some kind of score.
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError


@dataclasses.dataclass
class DurationME(ERA5Metric):
    """
    The mean error in the duration of an event.
    """
    threshold: float
    threshold_tolerance: float

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class RegionalRMSE(ERA5Metric): 
    """
    The root mean squared error of the forecasted regional mean value.
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class MaximumMAE(ERA5Metric):
    """
    The mean absolute error of the forecasted maximum value.
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class MaxMinMAE(ERA5Metric):
    """
    The mean absolute error of the forecasted highest minimum value,
    rolled up by a predefined time interval (e.g. daily).
    """
    time_interval: str

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError

    
@dataclasses.dataclass
class OnsetME(ERA5Metric):
    """
    The mean error in the onset of an event, in hours.
    Attributes:
        endpoint_relaxation_criteria: float: the number of hours beyond the event window
        to include
    """
    endpoint_extension_criteria: float = 48
    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError