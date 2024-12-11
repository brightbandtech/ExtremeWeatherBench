import yaml
import xarray as xr
import pandas as pd
import dataclasses
import utils as utils
import typing as t

@dataclasses.dataclass
class _Event:
    """
    Event holds the metadata that extends to all cases of a given event type.
    params:
    event_type: str
    """

    def __post_init__(self):
        with open('/home/taylor/code/ExtremeWeatherBench/assets/data/events.yaml', 'r') as file:
            self.events_data = yaml.safe_load(file)['events']

    def count_event_ids(self):
        return len(self.events_data[self.event_type])

    def create_events_dataframe(self):
        events_list = self.events_data[self.event_type]
        df = pd.DataFrame(events_list).set_index('id')
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        return df

@dataclasses.dataclass
class _Case:
    """
    Case holds the event and climatology datasets for a given case. 
    It also holds the metadata for the location and box length width in km. 
    """
    case_analysis_ds: xr.Dataset
    climatology_ds: xr.Dataset
    location_center: dict
    box_length_width_in_km: int
    analysis_variables: t.Union[None, t.List[str]] = None
    forecast_variables: t.Union[None, t.List[str]] = None 

    def __post_init__(self):
        self.convert_longitude()
        self.process_data_to_spec()
        self.align_climatology_to_case_analysis_time()
        #TODO: include obs data calls here from ISD
        
    def convert_longitude(self):
        self.case_analysis_ds = utils.convert_longitude_to_180(self.case_analysis_ds)
        self.climatology_ds = utils.convert_longitude_to_180(self.climatology_ds)

    def process_data_to_spec(self):
        self.case_analysis_ds = self.clip_and_remove_ocean(self.case_analysis_ds)
        self.climatology_ds = self.clip_and_remove_ocean(self.climatology_ds)
    
    def align_climatology_to_case_analysis_time(self):
        self.climatology_ds = self.climatology_ds.sel(time=self.case_analysis_ds.time)
    
    def clip_and_remove_ocean(self, xarray_data_object: t.Union[xr.Dataset,xr.DataArray]):
        interim_xarray_data_object = utils.clip_dataset_to_square(xarray_data_object, self.location_center, self.box_length_width_in_km)
        output_xarray_data_object = utils.remove_ocean_gridpoints(interim_xarray_data_object)
        return output_xarray_data_object

@dataclasses.dataclass
class HeatWave(_Event):
    """
    HeatWave holds the event datasets for extreme heat wave events. 
    It also holds the metadata for the location and box length width in km. 
    params:
    analysis_variables: t.Union[None, t.List[str]] = None: variable names for the analysis dataset, optional
    forecast_variables: t.Union[None, t.List[str]] = None: variable names for the forecast dataset, optional
    """

    def __post_init__(self):
        super().__post_init__()
        self.event_type = 'heat_wave'
        self.count = self.count_event_ids()
        self.events_df = self.create_events_dataframe()
        self.analysis_variables = ['2m_temperature'] if self.analysis_variables is None else self.analysis_variables
        self.forecast_variables = ['t2'] if self.forecast_variables is None else self.forecast_variables
    
    def get_case(self, case_id: int):
        case_data = self.events_df.loc[case_id]
        case_analysis_ds = utils.load_case_data(case_data['case_analysis'])
        climatology_ds = utils.load_case_data(case_data['climatology'])
        return _Case(case_analysis_ds, climatology_ds, case_data['location_center'], case_data['box_length_width_in_km'])
    


@dataclasses.dataclass
class Freeze(_Event):
    """
    Freeze holds the event datasets for extreme freezing events. 
    It also holds the metadata for the location and box length width in km. 
    """

    def __post_init__(self):
        super().__post_init__()
        self.event_type = 'freeze'
        self.count = self.count_event_ids()
        self.events_df = self.create_events_dataframe()


    