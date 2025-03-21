"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

import dataclasses
from typing import List, Optional
from extremeweatherbench import case


@dataclasses.dataclass
class EventContainer:
    """A container class to hold a list of cases of varying event types.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    cases: List[case.IndividualCase]
    event_type: Optional[str] = None

    def subset_cases(self, subset) -> List[case.IndividualCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is a specific type."""
        assert self.event_type is not None, "Event type must be defined."
        case_subset = [
            case.get_case_event_dataclass(c.event_type)(**dataclasses.asdict(c))
            for c in self.cases
            if subset in c.event_type
        ]
        return case_subset

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
