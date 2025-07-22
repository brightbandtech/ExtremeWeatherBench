"""Definitions for different type of extreme weather Events for analysis.
Logic for the dataclasses here largely to handle the logic of parsing the events."""

from abc import ABC
from typing import Any, List, Optional

import dacite

from extremeweatherbench import case, metrics, observations, variables


class EventType(ABC):
    """A base class defining the interface for ExtremeWeatherBench event types.

    An Event in ExtremeWeatherBench defines a specific weather event type, such as a heat wave,
    severe convective weather, or atmospheric rivers. These events encapsulate a set of cases and
    defined behavior for evaluating those cases. These cases will share common metrics, observations,
    and variables while each having unique dates and locations.

    Attributes:
        case_metadata: A dictionary or yaml file with guiding metadata.
        metrics: A list of Metrics that are used to evaluate the cases.
        observations: A list of Observations that are used as targets for the metrics.
    """

    def __init__(
        self,
        case_metadata: dict[str, Any],
        metrics: List[metrics.Metric],
        observations: List[observations.Observation],
        variable_mapping: Optional[
            dict[str | variables.DerivedVariable, str | variables.DerivedVariable]
        ] = None,
    ):
        self.case_metadata = case_metadata
        self.metrics = metrics
        self.observations = observations
        self.variable_mapping = variable_mapping

    @property
    def build_cases(self) -> List[case.IndividualCase]:
        """Build a list of IndividualCases from the case_metadata."""
        cases = dacite.from_dict(
            data_class=case.IndividualCase, data=self.case_metadata
        )
        return cases
