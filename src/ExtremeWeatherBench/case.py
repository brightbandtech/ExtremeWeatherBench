import dataclasses
import xarray as xr
from typing import Optional
import pandas as pd
from collections import namedtuple
import logging
from . import utils
from . import events
import datetime
import yaml
import dacite

Location = namedtuple('Location', ['latitude', 'longitude'])

@dataclasses.dataclass
class IndividualCase:
    """
    IndividualCase holds the metadata for a single case
    based on the events.yaml metadata.
    Attributes:
        id: the numerical identifier for the event
        start_date: datetime.date: the start date of the case
        end_date: datetime.date: the end date of the case
        location: Location: the latitude and longitude of the center of the location
        bounding_box_km: int: the side length of the square in kilometers
    """

    id: int
    start_date: datetime.date
    end_date: datetime.date
    location: Location
    bounding_box_km: int

# @dataclasses.dataclass
# class Case:
#     """
#     Case holds the event and climatology metadata for a given case.
#     It also holds the metadata for the location and box length width in km. 
#     Attributes:
#         location_centroid: dict: the latitude and longitude of the center of the location
#         box_length_width_in_km: int: the side length of the square in kilometers
#         analysis_variables: t.Union[None, t.List[str]] = None: variable names for the analysis dataset, optional
#         forecast_variables: t.Union[None, t.List[str]] = None: variable names for the forecast dataset, optional
#     """
#     event: events._Event
#     case_info: pd.core.frame.pandas
#     gridded_obs_ds: Optional[xr.Dataset]
#     point_obs_ds: Optional[pd.DataFrame]
#     forecast_ds: xr.Dataset
#     subset_forecast_ds: Optional[xr.Dataset] = None
#     subset_gridded_obs_ds: Optional[xr.Dataset] = None

#     def compute_event_gridded_metrics(self):
#         """
#         Process the metrics for the event.
#         """
#         gridded_metric_ds = xr.Dataset()
#         if self.subset_forecast_ds is None or self.subset_gridded_obs_ds is None:
#             logging.warning('Subset forecast and observation datasets will be computed first.')
#             self.compute_subset()
#         for metric in self.event.metrics:
#             result = metric.compute(forecast=self.subset_forecast_ds,
#                            observation=self.subset_gridded_obs_ds
#                            )
#             gridded_metric_ds = xr.merge([gridded_metric_ds, result])
#         return gridded_metric_ds
    
#     def compute_event_point_metrics(self):
#         """
#         Process the metrics for the event.
#         """
#         point_metric_ds = xr.Dataset()
#         if self.subset_forecast_ds is None or self.subset_point_obs_ds is None:
#             logging.warning('Subset forecast and observation datasets will be computed first.')
#             self.compute_subset()
#         for metric in self.event.metrics:
#             result = metric.compute(forecast=self.subset_forecast_ds,
#                            observation=self.subset_point_obs_ds
#                            )
#             point_metric_ds = xr.merge([point_metric_ds, result])
#         return point_metric_ds
    
#     def compute_subset(self):
#         """
#         Subset the region and datetimes for the case.
#         """
#         print(self.case_info)
#         print(self.case_info[4])
#         self.subset_forecast_ds = utils.clip_dataset_to_bounding_box(self.forecast_ds,
#                                                                      self.case_info['location'],
#                                                                      self.case_info['box_length_width_in_km'])
#         if self.gridded_obs_ds is not None:
#             self.subset_gridded_obs_ds = utils.clip_dataset_to_bounding_box(self.gridded_obs_ds,
#                                                                      self.case_info['location'],
#                                                                      self.case_info['box_length_width_in_km'])
#         if self.point_obs_ds is not None:
#             self.subset_point_obs_ds = utils.clip_dataset_to_bounding_box(self.point_obs_ds,
#                                                                      self.case_info['location'],
#                                                                      self.case_info['box_length_width_in_km'])
            
# @dataclasses.dataclass
# class IndividualCase:
#     """
#     IndividualCase holds the metadata for a single case.
#     Attributes:
#         case_id: str: the unique identifier for the case
#         location: namedtuple: the latitude and longitude of the center of the location
#         box_length_width_in_km: int: the side length of the square in kilometers
#         start_date: str: the start date of the case
#         end_date: str: the end date of the case
#     """
#     case_id: str
#     location: dict
#     box_length_width_in_km: int
#     start_date: pd.Timestamp
#     end_date: pd.Timestamp