"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

import dataclasses
from typing import List
from extremeweatherbench import case

# TODO(taylor): Cache in a bucket in brightband-public project and link here.
CLIMATOLOGY_LINK = "/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr"


@dataclasses.dataclass
class EventContainer:
    """A container class to hold a list of cases of varying event types.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    cases: List[case.IndividualCase]

    def subset_cases(self, subset) -> List[case.IndividualCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is a specific type."""

        return [
            case.get_case_event_dataclass(c.event_type)(**dataclasses.asdict(c))
            for c in self.cases
            if subset in c.event_type
        ]

    def __post_init__(self):
        self.cases = self.subset_cases(self.event_type)


@dataclasses.dataclass
class HeatWave(EventContainer):
    """A container class to hold a list of cases of heat wave events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "heat_wave"


@dataclasses.dataclass
class Freeze(EventContainer):
    """A container class to hold a list of cases of freeze events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "freeze"
