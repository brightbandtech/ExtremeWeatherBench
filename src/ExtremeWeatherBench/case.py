import dataclasses
import xarray as xr
from typing import Optional
import pandas as pd
from collections import namedtuple
from . import utils
from . import events
from . import config

@dataclasses.dataclass
class Case:
    """
    Case holds the event and climatology metadata for a given case.
    It also holds the metadata for the location and box length width in km. 
    Attributes:
        location_centroid: dict: the latitude and longitude of the center of the location
        box_length_width_in_km: int: the side length of the square in kilometers
        analysis_variables: t.Union[None, t.List[str]] = None: variable names for the analysis dataset, optional
        forecast_variables: t.Union[None, t.List[str]] = None: variable names for the forecast dataset, optional
    """
    event_type: events._Event
    case_info: pd.core.frame.pandas
    gridded_obs_ds: Optional[xr.Dataset]
    point_obs_ds: Optional[pd.DataFrame]
    forecast_ds: xr.Dataset

    def compute_event_gridded_metrics(self):
        """
        Process the metrics for the event.
        """
        gridded_metric_ds = xr.Dataset()
        for metric in self.event_type.metrics:
            metric.compute(observation=self.gridded_obs_ds,
                           forecast=self.forecast_ds)
            gridded_metric_ds = xr.merge([gridded_metric_ds, metric.metric_ds])
        return gridded_metric_ds
    
    def compute_event_point_metrics(self):
        """
        Process the metrics for the event.
        """
        point_metric_ds = xr.Dataset()
        for metric in self.event_type.metrics:
            metric.compute(observation=self.point_obs_ds,
                           forecast=self.forecast_ds)
            point_metric_ds = xr.merge([point_metric_ds, metric.metric_ds])
        return point_metric_ds