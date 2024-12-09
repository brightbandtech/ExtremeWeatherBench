import yaml
import xarray as xr
import pandas as pd
import numpy as np
import dataclasses
import ExtremeWeatherBench.utils as utils
import typing as t

@dataclasses.dataclass
class Event:
    """
    Event holds the metadata that extends to all cases of a given event type.
    """
    event_type: str
    
    def __post_init__(self):
        with open('/home/taylor/code/ExtremeWeatherBench/assets/data/events.yaml', 'r') as file:
            events_data = yaml.safe_load(file)
        self.events = [event for event in events_data if event['event_type'] == self.event_type]

@dataclasses.dataclass
class Case(Event):
    """
    Case holds the event and climatology datasets for a given case. 
    It also holds the metadata for the location and box length width in km. 
    """
    
    case_analysis_ds: xr.Dataset
    climatology_ds: xr.Dataset
    location_center: dict
    box_length_width_in_km: int

    def __post_init__(self):
        self.convert_longitude()
        self.clip_and_remove_ocean()
        self.align_climatology_to_event_analysis_time()
        
    def convert_longitude(self):
        self.event_analysis_ds = utils.convert_longitude_to_180(self.event_analysis_ds)
        self.climatology_ds = utils.convert_longitude_to_180(self.climatology_ds)

    def clip_and_remove_ocean(self):
        self.event_analysis_ds = self.process_dataset_to_spec(self.event_analysis_ds)
        self.climatology_ds = self.process_dataset_to_spec(self.climatology_ds)
    
    def align_climatology_to_event_analysis_time(self):
        self.climatology_ds = self.climatology_ds.sel(time=self.event_analysis_ds.time)
    
    def process_dataset_to_spec(self, xarray_data_object: t.Union[xr.Dataset,xr.DataArray]):
        interim_xarray_data_object = utils.clip_dataset_to_square(xarray_data_object, self.location_center, self.box_length_width_in_km)
        output_xarray_data_object = utils.remove_ocean_gridpoints(interim_xarray_data_object)
        return output_xarray_data_object
