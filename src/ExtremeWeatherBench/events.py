"""Primary module to contain different event types and their associated metrics."""
import yaml
import dacite

#TODO remove relative imports
from . import metrics
from . import case
import dataclasses

#TODO: public bucket link
CLIMATOLOGY_LINK = '/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr' 

@dataclasses.dataclass
class Event:
    """
    Event holds the cases for a specific event type.
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
        heat_wave: the list of cases for the event type
    """
    heat_wave: list[case.IndividualCase]

    def __post_init__(self):
        self.cases = self.heat_wave

    def build_metrics(self):
        self.metrics = [
            metrics.MaximumMAE,
            metrics.DurationME,
            metrics.RegionalRMSE,
            metrics.MaxMinMAE,
            metrics.OnsetME
            ]


@dataclasses.dataclass
class Freeze(Event):
    """
    Freeze holds the cases for extreme freeze events.
    Attributes:
        freeze: list[IndividualCase]: the list of cases for the event type
    """
    freeze: list[case.IndividualCase]

    def __post_init__(self):
        self.cases = self.freeze

    def build_metrics(self):
        self.metrics = [
            metrics.DurationME,
            metrics.RegionalRMSE,
            metrics.MaxMinMAE,
            metrics.OnsetME
            ]
    