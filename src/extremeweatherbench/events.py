"""Definitions and abstract base classes for different type of extreme weather Events for analysis."""

import dataclasses
from abc import ABC
from typing import Any, List

import dacite

from extremeweatherbench import calc, case, metrics, observations


def maybe_expand_variable_lists(
    variable_list: List[str | DerivedVariable],
) -> List[str]:
    """Build a list of core variables for the event, given the forecast and observation variables."""

    def iterator(variables: List[str | DerivedVariable]) -> List[str]:
        for variable in variables:
            if isinstance(variable, str):
                pass
            elif issubclass(variable, DerivedVariable):
                variables.extend([n for n in variable().input_variables])
        return variables

    return iterator(variable_list)


class EventType(ABC):
    """A base class defining the interface for ExtremeWeatherBench event types.

    An Event in ExtremeWeatherBench defines a specific weather event type, such as a heat wave,
    severe convective weather, or atmospheric rivers. These events encapsulate a set of cases and
    derived behavior for evaluating those cases. These cases will share common metrics, observations,
    and variables while each having unique dates and locations.

    Attributes:
        event_type: The type of event.
        forecast_variables: A list of variables that are used to forecast the event.
        observation_variables: A list of variables that are used to observe the event.
        case_metadata: A dictionary or yaml file with guiding metadata.
        metrics: A list of Metrics that are used to evaluate the cases.
        observations: A list of Observations that are used as targets for the metrics.
    """

    def __init__(
        self,
        event_type: str,
        forecast_variables: List[str | DerivedVariable],
        observation_variables: List[str | DerivedVariable],
        case_metadata: dict[str, Any],
        metrics: List[metrics.Metric],
        observations: List[Observation],
    ):
        self.event_type = event_type
        self.forecast_variables = maybe_expand_variable_lists(forecast_variables)
        self.observation_variables = maybe_expand_variable_lists(observation_variables)
        self.case_metadata = case_metadata
        self.metrics = metrics
        self.observations = observations

    def _build_base_case_metadata_collection(self) -> BaseCaseMetadataCollection:
        """Build a list of IndividualCases from the case_metadata."""
        cases = dacite.from_dict(
            data_class=BaseCaseMetadataCollection,
            data=self.case_metadata,
            config=dacite.Config(
                type_hooks={regions.Region: regions.map_to_create_region},
            ),
        )
        cases = BaseCaseMetadataCollection(
            cases=[c for c in cases.cases if c.event_type == self.event_type]
        )
        return cases

    def build_case_operator(self) -> list[CaseOperator]:
        """Build a CaseOperator from the event type."""
        case_metadata_collection = self._build_base_case_metadata_collection()
        case_operators = [
            CaseOperator(
                case=case,
                metrics=self.metrics,
                observations=self.observations,
            )
            for case in case_metadata_collection.cases
        ]
        return case_operators


class HeatWave(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        forecast_variables: List[str | DerivedVariable] = ["surface_air_temperature"],
        observation_variables: List[str | DerivedVariable] = [
            "surface_air_temperature"
        ],
        metrics: List[metrics.Metric] = [
            MaximumMAE,
            MaxMinMAE,
            RegionalRMSE,
            OnsetME,
            DurationME,
        ],
        observations: List[observations.Observation] = [observations.ERA5],
    ):
        super().__init__(
            event_type="heat_wave",
            forecast_variables=forecast_variables,
            observation_variables=observation_variables,
            case_metadata=case_metadata,
            metrics=metrics,
            observations=observations,
        )


class SevereConvection(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        forecast_variables: List[str | DerivedVariable] = [
            CravenSignificantSevereParameter,
        ],
        observation_variables: List[str | DerivedVariable] = [
            PracticallyPerfectHindcast,
        ],
        metrics: List[metrics.Metric] = [
            CSI,
            LeadTimeDetection,
            RegionalHitsMisses,
            HitsMisses,
        ],
        observations: List[observations.Observation] = [observations.LSR],
    ):
        super().__init__(
            event_type="severe_convection",
            forecast_variables=forecast_variables,
            observation_variables=observation_variables,
            case_metadata=case_metadata,
            metrics=metrics,
            observations=observations,
        )


class AtmosphericRiver(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        forecast_variables: List[str | DerivedVariable] = [],
        observation_variables: List[str | DerivedVariable] = [],
        metrics: List[metrics.Metric] = [
            CSI,
            LeadTimeDetection,
        ],
        observations: List[observations.Observation] = [observations.ERA5],
    ):
        super().__init__(
            event_type="atmospheric_river",
            forecast_variables=forecast_variables,
            observation_variables=observation_variables,
            case_metadata=case_metadata,
            metrics=metrics,
            observations=observations,
        )


@dataclasses.dataclass
class EventOperator:
    events: List[EventType]
    pre_composed_metrics: dict[EventType, List[metrics.Metric]] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )
    pre_composed_observations: dict[EventType, List[observations.Observation]] = (
        dataclasses.field(default_factory=list, init=False, repr=True)
    )
    pre_composed_forecast_variables: dict[EventType, List[str | DerivedVariable]] = (
        dataclasses.field(default_factory=list, init=False, repr=True)
    )
    pre_composed_observation_variables: dict[EventType, List[str | DerivedVariable]] = (
        dataclasses.field(default_factory=list, init=False, repr=True)
    )
    pre_composed_case_operators: List[CaseOperator] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )

    def __post_init__(self):
        # Unravel attributes from composed event types
        self.pre_composed_metrics = {}
        self.pre_composed_observations = {}
        self.pre_composed_forecast_variables = {}
        self.pre_composed_observation_variables = {}
        self.pre_composed_case_operators = []

        # Collect attributes from each event type
        for event in self.events:
            self.pre_composed_metrics[event.event_type] = event.metrics
            self.pre_composed_observations[event.event_type] = event.observations
            self.pre_composed_forecast_variables[event.event_type] = (
                event.forecast_variables
            )
            self.pre_composed_observation_variables[event.event_type] = (
                event.observation_variables
            )
            self.pre_composed_case_operators.extend(event.build_case_operator())
