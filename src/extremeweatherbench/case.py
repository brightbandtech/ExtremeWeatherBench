"""Classes for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBenchX (Rasp et al.)."""

import dataclasses
import datetime
import logging
from typing import List

import xarray as xr

from extremeweatherbench import derived, forecasts, regions, targets  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single or individual case.

    An IndividualCase defines the relevant metadata for a single case study for a
    given extreme weather event; it is designed to be easily instantiable through a
    simple YAML-based configuration file.

    Attributes:
        case_id_number: A unique numerical identifier for the event.
        start_date: The start date of the case, for use in subsetting data for analysis.
        end_date: The end date of the case, for use in subsetting data for analysis.
        location: A Location dataclass representing the location of a case.
        event_type: A string representing the type of extreme weather event.
    """

    case_id_number: int
    title: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    location: regions.Region
    event_type: str


@dataclasses.dataclass
class BaseCaseMetadataCollection:
    """A collection of IndividualCases.

    This class is used to store a collection of IndividualCases, which can be used to
    subset the cases by event type.

    Attributes:
        cases: A list of IndividualCases.

    Methods:
        subset_cases_by_event_type: Subset the cases in the collection by event type.
    """

    cases: List[IndividualCase]

    def subset_cases_by_event_type(self, event_type: str) -> List[IndividualCase]:
        """Subset the cases in the collection by event type."""

        case_list = [c for c in self.cases if c.event_type == event_type]

        # raises error if no cases (empty list) are found for the event type
        if case_list:
            return case_list
        else:
            raise ValueError(f"No cases found for event type {event_type}")


@dataclasses.dataclass
class CaseOperator:
    """A class which stores the graph to process an individual case.

    This class is used to store the graph to process an individual case. The purpose of
    this class is to be a one-stop-shop for the evaluation of a single case. Multiple
    CaseOperators can be run in parallel to evaluate multiple cases, or run through the
    ExtremeWeatherBench.run() method to evaluate all cases in an evaluation in serial.

    Attributes:
        case: IndividualCase metadata
        metrics: A list of metrics that are intended to be evaluated for the case
        targets: A list of targets to evaluate against the forecast
        forecast: The incoming forecast data
        target_variables: Names of the variables present in the target data relevant to the evaluation
        forecast_variables: Names of the variables present in the forecast data relevant to the evaluation

    Methods:
        evaluate_case: Process a case's metrics
        process_metrics: Process the metrics.
        build_targets: Build target xarray Datasets from the target sources.
    """

    case: IndividualCase
    metrics: list["metrics.BaseMetric"]
    targets: list["targets.TargetBase"]
    forecast: "forecasts.Forecast"
    target_variables: list[str | "derived.DerivedVariable"]
    forecast_variables: list[str | "derived.DerivedVariable"]

    def evaluate_case(self, forecast: xr.Dataset):
        """Process a case."""
        self.process_metrics(forecast)

    def process_metrics(self, forecast: xr.Dataset):
        """Process the metrics."""
        for metric in self.metrics:
            metric.process_metric(forecast, self.targets)

    def build_targets(
        self, target_storage_options: dict, target_variable_mapping: dict
    ) -> list[xr.Dataset]:
        """Build target xarray Datasets from the target sources."""
        target_datasets = []
        for target in self.targets:
            target_dataset = target().run_pipeline(
                case=self,
                target_variables=self.target_variables,
                storage_options=target_storage_options,
                target_variable_mapping=target_variable_mapping,
            )
            target_datasets.append(target_dataset)

        return target_datasets
