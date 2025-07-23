"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

import dataclasses
from abc import ABC
from typing import Any, List

import dacite

from extremeweatherbench import case, metrics, observations, variables


class EventType(ABC):
    """A base class defining the interface for ExtremeWeatherBench event types.

    An Event in ExtremeWeatherBench defines a specific weather event type, such as a heat wave,
    severe convective weather, or atmospheric rivers. These events encapsulate a set of cases and
    derived behavior for evaluating those cases. These cases will share common metrics, observations,
    and variables while each having unique dates and locations.

    Attributes:
        case_metadata: A dictionary or yaml file with guiding metadata.
        metrics: A list of Metrics that are used to evaluate the cases.
        observations: A list of Observations that are used as targets for the metrics.
    """

    def __init__(
        self,
        event_type: str,
        evaluation_variables: List[str | variables.DerivedVariable],
        case_metadata: dict[str, Any],
        metrics: List[metrics.Metric],
        evaluation_observations: List[observations.Observation],
    ):
        self.event_type = event_type
        self.evaluation_variables = evaluation_variables
        self.case_metadata = case_metadata
        self.metrics = metrics
        self.evaluation_observations = evaluation_observations

    def _build_base_case_metadata_collection(self) -> case.BaseCaseMetadataCollection:
        """Build a list of IndividualCases from the case_metadata."""
        cases = dacite.from_dict(
            data_class=case.BaseCaseMetadataCollection, data=self.case_metadata
        )
        cases = case.BaseCaseMetadataCollection(
            cases=[c for c in cases.cases if c.event_type == self.event_type]
        )
        return cases

    def build_case_operator(self) -> list[case.CaseOperator]:
        """Build a CaseOperator from the event type."""
        case_metadata_collection = self._build_base_case_metadata_collection()
        case_operators = [
            case.CaseOperator(
                case=c,
                metrics=self.metrics,
                observations=self.evaluation_observations,
            )
            for c in case_metadata_collection.cases
        ]
        return case_operators

    def _maybe_expand_variable_list(self) -> List[str]:
        """Build a list of core variables for the event, given"""
        evaluation_variables = []
        for variable in self.evaluation_variables:
            if isinstance(variable, str):
                pass
            elif isinstance(variable, variables.DerivedVariable):
                # DerivedVariables include a list of input variables needed to construct themselves
                evaluation_variables.append([n for n in variable.input_variables])
        return evaluation_variables


class HeatWave(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        evaluation_variables: List[str | variables.DerivedVariable] = [
            "surface_air_temperature"
        ],
        metrics: List[metrics.Metric] = [
            metrics.MaximumMAE,
            metrics.MaxMinMAE,
            metrics.RegionalRMSE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        evaluation_observations: List[observations.Observation] = [observations.ERA5],
    ):
        super().__init__(
            event_type="heat_wave",
            evaluation_variables=evaluation_variables,
            case_metadata=case_metadata,
            metrics=metrics,
            evaluation_observations=evaluation_observations,
        )


class SevereConvection(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        evaluation_variables: List[str | variables.DerivedVariable] = [
            variables.CravenSignificantSevereParameter,
            variables.PracticallyPerfectHindcast,
        ],
        metrics: List[metrics.Metric] = [
            metrics.CSI,
            metrics.LeadTimeDetection,
            metrics.RegionalHitsMisses,
            metrics.HitsMisses,
        ],
        evaluation_observations: List[observations.Observation] = [observations.LSR],
    ):
        super().__init__(
            event_type="severe_day",
            case_metadata=case_metadata,
            evaluation_variables=evaluation_variables,
            metrics=metrics,
            evaluation_observations=evaluation_observations,
        )


class AtmosphericRiver(EventType):
    def __init__(
        self,
        case_metadata: dict[str, Any],
        evaluation_variables: List[str | variables.DerivedVariable] = [],
        metrics: List[metrics.Metric] = [
            metrics.CSI,
            metrics.LeadTimeDetection,
        ],
        evaluation_observations: List[observations.Observation] = [observations.ERA5],
    ):
        super().__init__(
            event_type="atmospheric_river",
            evaluation_variables=evaluation_variables,
            case_metadata=case_metadata,
            metrics=metrics,
            evaluation_observations=evaluation_observations,
        )


@dataclasses.dataclass
class EventOperator:
    events: List[EventType]
    composed_metrics: List[metrics.Metric] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )
    composed_observations: List[observations.Observation] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )
    composed_variable_mappings: List[dict] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )
    composed_case_operators: List[case.CaseOperator] = dataclasses.field(
        default_factory=list, init=False, repr=True
    )

    def __post_init__(self):
        # Unravel attributes from composed event types
        self.composed_metrics = []
        self.composed_observations = []
        self.composed_variable_mappings = []
        self.composed_case_operators = []

        # Collect attributes from each event type
        for event in self.events:
            self.composed_metrics.extend(event.metrics)
            self.composed_observations.extend(event.evaluation_observations)
            self.composed_case_operators.extend(event.build_case_operator())
