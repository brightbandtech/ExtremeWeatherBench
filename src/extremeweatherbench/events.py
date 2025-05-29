"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

import abc
import dataclasses
from typing import Dict, List, Optional, Type

from extremeweatherbench import case
from extremeweatherbench.observations import ObservationHandler

# Registry to map event type strings to their corresponding classes
EVENT_REGISTRY: Dict[str, Type["EventContainer"]] = {}


def register_event_type(event_type: str, event_class: Type["EventContainer"]) -> None:
    """Register a new event type and its corresponding class.

    Args:
        event_type: The string identifier for the event type
        event_class: The class that implements this event type
    """
    EVENT_REGISTRY[event_type] = event_class


def get_event_class(event_type: str) -> Optional[Type["EventContainer"]]:
    """Get the class corresponding to an event type string.

    Args:
        event_type: The string identifier for the event type

    Returns:
        The corresponding event class if found, None otherwise
    """
    return EVENT_REGISTRY.get(event_type)


class EventContainer(abc.ABC):
    """A container class to hold a list of cases of varying event types.
    Attributes:
        cases: A list of cases that is defined by events.yaml
        event_type: The type of event
        observation_types: The types of observations to use
        metrics: The metrics to use
        variables: The variables to use
    """

    def __init__(
        self,
        cases: List[case.IndividualCase],
        event_type: str,
        observation_types: List[str],
        metrics: List[str],
        variables: List[str],
    ):
        self.cases = self.subset_cases(cases)
        self.event_type = event_type
        self.observation_types = observation_types
        self.metrics = metrics
        self.variables = variables

    def build(self):
        """Build the event container."""
        self.observation_handlers = self.get_observation_handlers()
        self.cases = self.subset_cases(self.cases)

    def subset_cases(self, subset) -> List[case.IndividualCase]:
        """Subset all IndividualCases inside EventContainer where _case_event_type is a specific type."""
        assert self.event_type is not None, "Event type must be defined."
        case_subset = [
            case.get_case_event_dataclass(c.event_type)(**dataclasses.asdict(c))
            for c in self.cases
            if subset in c.event_type
        ]
        return case_subset

    def get_observation_handlers(self) -> List[ObservationHandler]:
        """Get the observation handlers for the event type."""
        return [
            ObservationHandler(observation_type, self.event_type)
            for observation_type in self.observation_types
        ]


@dataclasses.dataclass
class HeatWave(EventContainer):
    """A container class to hold a list of cases of heat wave events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "heatwave"
    observation_types = ["era5", "ghcn"]


register_event_type("heatwave", HeatWave)


@dataclasses.dataclass
class Freeze(EventContainer):
    """A container class to hold a list of cases of freeze events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "freeze"
    observation_types = ["era5", "ghcn"]


register_event_type("freeze", Freeze)


@dataclasses.dataclass
class SevereConvection(EventContainer):
    """A container class to hold a list of cases of severe convection events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "severe_convection"
    observation_types = ["storm_report"]


register_event_type("severe_convection", SevereConvection)


@dataclasses.dataclass
class TropicalCyclone(EventContainer):
    """A container class to hold a list of cases of tropical cyclone events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "tropical_cyclone"
    observation_types = ["ibtracs", "era5", "ghcn"]


register_event_type("tropical_cyclone", TropicalCyclone)


@dataclasses.dataclass
class AtmosphericRiver(EventContainer):
    """A container class to hold a list of cases of atmospheric river events.
    Attributes:
        cases: A list of cases that is defined by events.yaml
    """

    event_type: str = "atmospheric_river"
    observation_types = ["era5", "ghcn"]


register_event_type("atmospheric_river", AtmosphericRiver)
