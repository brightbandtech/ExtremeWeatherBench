import yaml
import xarray as xr
import pandas as pd
import dataclasses
import utils as utils
import typing as t
import metrics
import case

@dataclasses.dataclass
class _Event:
    """
    Event holds the metadata that extends to all cases of a given event type.
    params:
    """
    analysis_variables: t.Union[None, t.List[str]] = None
    forecast_variables: t.Union[None, t.List[str]] = None
 
    def __post_init__(self):
        with open('/home/taylor/code/ExtremeWeatherBench/assets/data/events.yaml', 'r') as file:
            self.events_data = yaml.safe_load(file)['events']

    def count_event_ids(self):
        if self.event_type not in self.events_data:
            raise KeyError(f"Event type '{self.event_type}' not found in events data.")
        return len(self.events_data[self.event_type])

    def create_case_dataframe(self):
        case_list = self.events_data[self.event_type]
        df = pd.DataFrame(case_list).set_index('id')
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        return df
    
    def metrics(self):
        self._metrics: t.Dict[str, metrics.Metric] = {}

    def modify_metrics(self, key: str, value: t.Any):
        self._metrics[key] = value
    
    def generate_case_files(self):
        if not hasattr(self, 'case_df'):
            raise RuntimeError("create_case_dataframe must be called before generate_case_files.")
        for case_id in self.case_df.index:
            case.Case()
            #TODO: create a list/dict of case objects for each case_id

        
        



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
        self.case_df = self.create_case_dataframe()
        self.analysis_variables = ['2m_temperature'] if self.analysis_variables is None else self.analysis_variables
        self.forecast_variables = ['t2'] if self.forecast_variables is None else self.forecast_variables
    
    def heatwave_metrics(self):
        self.metrics()
        self.modify_metrics('threshold_weighted_rmse', metrics.threshold_weighted_rmse)
        self.modify_metrics('mae_max_of_max_temperatures', metrics.mae_max_of_max_temperatures)
        self.modify_metrics('mae_max_of_min_temperatures', metrics.mae_max_of_min_temperatures)
        self.modify_metrics('onset_above_85th_percentile', metrics.onset_above_85th_percentile)
        self.modify_metrics('mae_onset_and_end_above_85th_percentile', metrics.mae_onset_and_end_above_85th_percentile)


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
        self.case_df = self.create_case_dataframe()


    