"""Definitions for different type of extreme weather Events for analysis."""

import dataclasses

import dacite
import yaml
from extremeweatherbench import case, metrics

# TODO(taylor): Cache in a bucket in brightband-public project and link here.
CLIMATOLOGY_LINK = "/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr"


@dataclasses.dataclass
class Event:
    """A generic base class for extending with details for specific extreme weather
    events.
    """

    path: str
    cases

    def __post_init__(self):
        self.cases = []
        self.metrics = []

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as file:
            yaml_event_case = yaml.safe_load(file)["events"]
            for individual_case in yaml_event_case:
                for event_type in yaml_event_case[individual_case]:
                    event_type["location"] = case.Location(**event_type["location"])
        return dacite.from_dict(cls, yaml_event_case)


@dataclasses.dataclass
class HeatWave(Event):
    """A class for defining heat wave events.

    Attributes:
        heat_wave: The list of cases to analyze for this Event.
        # NOTE(daniel): all attributes established anywhere in the class template
        # should be listed here.
        cases: The list of cases to analyze for this Event.
        metrics: The list of metrics to use for analyzing each case for this Event.

    """

    # NOTE(daniel): It would make the most sense for this to simply be called "cases",
    # and make it a standard attribute of the base Event class. The construction here
    # which unnecessarily saves two copies of the same data. We can revisit this after
    # the first end-to-end analyses are done.
    heat_wave: list[case.IndividualCase]

    def __post_init__(self):
        self.cases = self.heat_wave

    # NOTE(daniel): This is a good candidate for refactoring into the base Event class.
    # Then, each subclass would internally define a list of potential metrics, and a
    # user can provide an over-ride list of metrics they'd like to use for events.
    # Part of the process of instantiating the specialized Event object, then, would
    # be reconciling that list of valid metrics and the user-provided list, and saving
    # in the "metrics" attribute of the (base) Event object.
    def build_metrics(self):
        self.metrics = [
            metrics.MaximumMAE(),
            metrics.DurationME(),
            metrics.RegionalRMSE(),
            metrics.MaxMinMAE(),
            metrics.OnsetME(),
        ]


@dataclasses.dataclass
class Freeze(Event):
    """A class for defining freeze events.

    Attributes:
        freeze: The list of cases to analyze for this Event.
        cases: The list of cases to analyze for this Event.
        metrics: The list of metrics to use for analyzing each case for this Event.
    """

    freeze: list[case.IndividualCase]

    def __post_init__(self):
        self.cases = self.freeze

    def build_metrics(self):
        self.metrics = [
            metrics.DurationME(),
            metrics.RegionalRMSE(),
            metrics.MaxMinMAE(),
            metrics.OnsetME(),
        ]
