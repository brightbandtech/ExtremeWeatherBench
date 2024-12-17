import dataclasses
from typing import Literal

import pandas as pd

from ExtremeWeatherBench.events import Freeze, HeatWave
from ExtremeWeatherBench.metrics import DurationME, MaxMinMAE, MaximumMAE, OnsetME, RegionalRMSE

@dataclasses.dataclass
class Location:
    latitude: float
    longitude: float

    def __post_init__(self):
        # TODO: validate latitude and longitude
        pass


@dataclasses.dataclass
class MetricConfig:
    metric_type: Literal["maximum_mae", "duration_me", "regional_rmse", "max_min_mae", "onset_me"]

    def build(self) -> RegionalRMSE | MaximumMAE | DurationME | MaxMinMAE | OnsetME:
        # TODO define a dict mapping ids to metric classes
        pass
    


@dataclasses.dataclass
class EventConfig:
    id: int
    title: str
    start_date: str
    end_date: str
    location: Location
    bounding_box_km: int
    event_type: Literal["freeze", "heat_wave"]
    metrics: list[MetricConfig]

    def __post_init__(self):
        # TODO: flesh out validation here
        try:
            pd.to_datetime(self.start_date)
            pd.to_datetime(self.end_date)
        except ValueError:
            raise ValueError("start_date and end_date must be in YYYY-MM-DD format")

    def build(self) -> Freeze | HeatWave:
        # TODO define a mapping from event_type to event class
        pass


@dataclasses.dataclass
class EventsConfig:
    events: list[EventConfig]

    def build(self) -> list[Freeze | HeatWave]:
        events = [e.build() for e in self.events]
        # TODO: do stuff with events
        return events