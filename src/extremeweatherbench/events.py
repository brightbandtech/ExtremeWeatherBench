"""Definitions for different type of extreme weather Events for analysis."""

import dataclasses
from typing import List
from extremeweatherbench import case, metrics

# TODO(taylor): Cache in a bucket in brightband-public project and link here.
CLIMATOLOGY_LINK = "/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr"


@dataclasses.dataclass
class Event:
    """A generic base class for extending with details for specific extreme weather
    events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
        metrics: A list of metrics to be used in the analysis of the cases.
    """

    cases: List[case.IndividualCase]
    metrics: List[metrics.Metric]


@dataclasses.dataclass
class HeatWave(Event):
    """A class for defining heat wave events.

    Attributes:
        cases: The list of cases to analyze for this Event.
        metrics: The list of metrics to use for analyzing each case for this Event.

    """

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
