import yaml
import xarray as xr
import pandas as pd
import dacite
import typing as t
import os
from . import metrics
from . import utils
from . import case
from collections import namedtuple
import dataclasses


#TODO: public bucket link
CLIMATOLOGY_LINK = '/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr' 

@dataclasses.dataclass
class Event:
    """
    Event holds the cases for a specific event type.
    Attributes:
        event: list[case.IndividualCase]: the list of cases for the event
    """

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as file:
            yaml_event_case = yaml.safe_load(file)['events']
            for individual_case in yaml_event_case:
                for event_type in yaml_event_case[individual_case]:
                    event_type['location'] = case.Location(**event_type['location'])
        return dacite.from_dict(cls, yaml_event_case)   


@dataclasses.dataclass
class HeatWave(Event):
    """
    HeatWave holds the cases for extreme heat wave events.
    Attributes:
        heat_wave: list[IndividualCase]: the list of cases for the event
    """
    heat_wave: list[case.IndividualCase]


@dataclasses.dataclass
class Freeze(Event):
    """
    Freeze holds the cases for extreme freeze events.
    Attributes:
        freeze: list[IndividualCase]: the list of cases for the event
    """
    freeze: list[case.IndividualCase]


# class _Event:
#     """
#     Event holds the metadata that extends to all cases of a given event type.
#     """

#     def __init__(self):
#         self.events_data = self.load_config()

#     def count_event_ids(self):
#         if self.event_type not in self.events_data:
#             raise KeyError(f"Event type '{self.event_type}' not found in events data.")
#         return len(self.events_data[self.event_type])

#     def create_case_dataframe(self):
#         case_list = self.events_data[self.event_type]
#         df = pd.DataFrame(case_list).set_index('id')
#         df['start_date'] = pd.to_datetime(df['start_date'])
#         df['end_date'] = pd.to_datetime(df['end_date'])
#         Location = namedtuple('Location', ['latitude', 'longitude'])
#         df['location'] = df['location'].apply(lambda loc: Location(**loc))
#         return df
    
#     def load_config(self):
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         events_file_path = os.path.join(base_dir, '../../assets/data/events.yaml')
#         with open(events_file_path, 'r') as file:
#             events_data = yaml.safe_load(file)['events']
#         return events_data


# class HeatWave(_Event):
#     """
#     HeatWave holds the event datasets for extreme heat wave events. 
#     It also holds the metadata for the location and box length width in km. 
#     params:
#     analysis_variables: t.Union[None, t.List[str]] = None: variable names for the analysis dataset, optional
#     forecast_variables: t.Union[None, t.List[str]] = None: variable names for the forecast dataset, optional
#     """
   
#     def __init__(self):
#         super().__init__()
#         self.event_type = 'heat_wave'
#         self.count = self.count_event_ids()
#         self.case_df = self.create_case_dataframe()
#         self.analysis_variables = ['2m_temperature']
#         self.endpoint_extension_criteria: int = 48
#         self.define_heatwave_metrics()
#         self.climatology_85th_percentile_ds = xr.open_zarr(CLIMATOLOGY_LINK)
#         self.subset_procedure = [
#                 utils.remove_ocean_gridpoints,
#                 utils.clip_dataset_to_bounding_box
#             ]
        
#     def define_heatwave_metrics(self):
#         self.metrics = [metrics.MaximumMAE,
#                         metrics.DurationME,
#                         metrics.RegionalRMSE,
#                         metrics.MaxMinMAE,
#                         metrics.OnsetME]

# class Freeze(_Event):
#     """
#     Freeze holds the event datasets for extreme freezing events. 
#     It also holds the metadata for the location and box length width in km. 
#     """

#     def __init__(self):
#         super().__init__()
#         self.event_type = 'freeze'
#         self.count = self.count_event_ids()
#         self.case_df = self.create_case_dataframe()
#         self.analysis_variables = ['2m_temperature']
#         self.define_freeze_metrics()
#         self.subset_procedure = [
#                 utils.clip_dataset_to_bounding_box
#             ]
        
#     def define_freeze_metrics(self):
#         self.metrics = [metrics.MaximumMAE,
#                         metrics.DurationME,
#                         metrics.RegionalRMSE,
#                         metrics.OnsetME]


    