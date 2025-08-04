"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

from abc import ABC, abstractmethod
from typing import Any, List

import dacite

from extremeweatherbench import (
    case,
    derived,
    forecasts,
    metrics,
    regions,
    targets,
)  # noqa: F401


class EventType(ABC):
    """A base class defining the interface for ExtremeWeatherBench event types.

    An Event in ExtremeWeatherBench defines a specific weather event type, such as a heat wave,
    severe convective weather, or atmospheric rivers. These events encapsulate a set of cases and
    derived behavior for evaluating those cases. These cases will share common metrics, observations,
    and variables while each having unique dates and locations.

    Attributes:
        event_type: The type of event.
        forecast_variables: A list of variables that are used to forecast the event.
        target_variables: A list of variables that are used to observe the event.
        case_metadata: A dictionary or yaml file with guiding metadata.
        metric_list: A list of Metrics that are used to evaluate the cases.
        target_list: A list of Targets that are used as targets for the metrics.
    """

    def __init__(
        self,
        case_metadata: dict[str, Any],
    ):
        """Initialize the EventType.

        Args:
            case_metadata: A dictionary with case metadata; EWB uses a YAML file to define the cases upstream.
        """
        self.case_metadata = case_metadata
        self.expanded_forecast_variables, self.expanded_target_variables = (
            self._maybe_expand_not_derived_variables()
        )

    @property
    @abstractmethod
    def event_type(self) -> str:
        pass

    @property
    @abstractmethod
    def forecast_variables(self) -> List[str | "derived.DerivedVariable"]:
        pass

    @property
    @abstractmethod
    def target_variables(self) -> List[str | "derived.DerivedVariable"]:
        pass

    @property
    @abstractmethod
    def metric_list(self) -> List["metrics.BaseMetric"]:
        pass

    @property
    @abstractmethod
    def target_list(self) -> List["targets.TargetBase"]:
        pass

    def _maybe_expand_not_derived_variables(
        self,
    ) -> tuple[
        List[str | "derived.DerivedVariable"], List[str | "derived.DerivedVariable"]
    ]:
        """Expand the variables to include the input variables of any derived variables.

        This private method checks if there are variables in the DerivedVariable(s) not already
        present in the forecast_variables or observation_variables. If so, it adds them to a new
        list to ensure they are subset along with the other variables for evaluation.
        """
        expanded_forecast_variables = []
        expanded_target_variables = []
        for v in self.forecast_variables:
            if hasattr(v, "input_variables"):
                expanded_forecast_variables = (
                    expanded_forecast_variables + v.input_variables
                )

        for v in self.target_variables:
            if hasattr(v, "input_variables"):
                expanded_target_variables = (
                    expanded_target_variables + v.input_variables
                )
        return expanded_forecast_variables, expanded_target_variables

    def _build_base_case_metadata_collection(self) -> "case.BaseCaseMetadataCollection":
        """Build a list of IndividualCases from the case_metadata."""
        cases = dacite.from_dict(
            data_class="case.BaseCaseMetadataCollection",
            data=self.case_metadata,
            config=dacite.Config(
                type_hooks={regions.Region: regions.map_to_create_region},
            ),
        )
        cases = "case.BaseCaseMetadataCollection"(
            cases=[c for c in cases.cases if c.event_type == self.event_type]
        )
        return cases

    def build_case_operators(
        self,
        forecast_source: "forecasts.ForecastSource",
    ) -> list["case.CaseOperator"]:
        """Build a CaseOperator from the event type.

        Args:
            forecast_source: The forecast source to use for the case operators.

        Returns:
            A list of CaseOperator objects.
        """
        case_metadata_collection = self._build_base_case_metadata_collection()
        case_operators = [
            "case.CaseOperator"(
                case=case,
                metrics=self.metric_list,
                targets=self.target_list,
                forecast_source=forecast_source,
                # if the expanded variables exist, use them, otherwise use the original variables
                target_variables=(
                    self.expanded_target_variables
                    if self.expanded_target_variables
                    else self.target_variables
                ),
                forecast_variables=(
                    self.expanded_forecast_variables
                    if self.expanded_forecast_variables
                    else self.forecast_variables
                ),
            )
            for case in case_metadata_collection.cases
        ]
        return case_operators
