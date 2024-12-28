"""Definitions for different type of extreme weather Events for analysis."""

import dataclasses
from typing import List
from extremeweatherbench import case, metrics

# TODO(taylor): Cache in a bucket in brightband-public project and link here.
CLIMATOLOGY_LINK = "/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr"


@dataclasses.dataclass
class EventContainer:
    """A container class to hold a list of cases of varying event types.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    cases: List[case.IndividualCase]


@dataclasses.dataclass
class HeatWave(EventContainer):
    """A container class to hold a list of cases of heat wave events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    def __post_init__(self):
        self.cases = self.subset_heatwave_cases()

    def subset_heatwave_cases(self) -> List[case.IndividualHeatWaveCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is IndividualHeatWaveCase."""
        return [c for c in self.cases if "heat_wave" in c.event_type]


@dataclasses.dataclass
class Freeze(EventContainer):
    """A container class to hold a list of cases of freeze events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    def __post_init__(self):
        self.cases = self.subset_freeze_cases()

    def subset_freeze_cases(self) -> List[case.IndividualFreezeCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is IndividualFreezeCase."""
        return [c for c in self.cases if "freeze" in c.event_type]
