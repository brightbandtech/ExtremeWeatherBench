import xarray as xr
from typing import Union
import numpy as np
import scores
import dataclasses

from . import utils

@dataclasses.dataclass
class Metric:
    """
    A parent class for a metric to evaluate a forecast. 
    """

    def compute(self):
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

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class RegionalRMSE(Metric): 
    """
    The root mean squared error of the forecasted regional mean value.
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class MaximumMAE(Metric):
    """
    The mean absolute error of the forecasted maximum value.
    """

    def compute(self):
        """
        Compute the metric.
        """
        raise NotImplementedError
    
@dataclasses.dataclass
class MaxMinMAE(Metric):
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
class OnsetME(Metric):
    """
    The mean error in the onset of an event, in hours.
    Attributes:
        endpoint_extension_criteria: float: the number of hours beyond the event window
        to include
    """
    def compute(self):
        """
        Compute the metric.
        """

        raise NotImplementedError