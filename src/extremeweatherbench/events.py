"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

import dataclasses
from typing import List, Optional

from extremeweatherbench import case
from extremeweatherbench.observations import (
    ObservationHandler,
    create_observation_handler,
)


# TODO: looks more like a regular class than dataclass at this point; convert it
@dataclasses.dataclass
class EventContainer:
    """A container class to hold a list of cases of varying event types.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    cases: List[case.IndividualCase]
    event_type: str
    observation_types: List[str] = dataclasses.field(default_factory=list)
    _observation_handlers: Optional[List[ObservationHandler]] = None

    def __post_init__(self):
        self.cases = self.subset_cases(self.event_type)
        # Convert observation type strings to handler instances
        self._observation_handlers = [
            create_observation_handler(obs_type, self.event_type)
            for obs_type in self.observation_types
        ]

    @property
    def observation_handlers(self) -> List[ObservationHandler]:
        """Get the observation handlers for this event container."""
        if self._observation_handlers is None:
            self._observation_handlers = [
                create_observation_handler(obs_type, self.event_type)
                for obs_type in self.observation_types
            ]
        return self._observation_handlers

    def subset_cases(self, subset) -> List[case.IndividualCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is a specific type."""
        assert self.event_type is not None, "Event type must be defined."
        case_subset = [
            case.get_case_event_dataclass(c.event_type)(**dataclasses.asdict(c))
            for c in self.cases
            if subset in c.event_type
        ]
        return case_subset


# TODO: refactor the event/case relationship; determine if possible to consolidate and simplify
@dataclasses.dataclass
class HeatWave(EventContainer):
    """A container class to hold a list of cases of heat wave events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "heat_wave"
    observation_types: List[str] = dataclasses.field(
        default_factory=lambda: ["era5", "ghcn"]
    )


@dataclasses.dataclass
class Freeze(EventContainer):
    """A container class to hold a list of cases of freeze events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "freeze"
    observation_types: List[str] = dataclasses.field(
        default_factory=lambda: ["era5", "ghcn"]
    )
