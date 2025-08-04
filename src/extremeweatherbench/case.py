"""Classes for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBench (Rasp et al.)."""

import dataclasses
import datetime
import logging
from typing import List, Optional

import numpy as np
import xarray as xr

from extremeweatherbench import derived, metrics, regions, targets, utils

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
        cross_listed: A list of other event types that this case study is cross-listed under.
    """

    case_id_number: int
    title: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    location: regions.Region
    event_type: str
    data_vars: Optional[List[str]] = None
    cross_listed: Optional[List[str]] = None

    def subset_region(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to the region specified in the location.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        return self.location.mask(dataset, drop=True)

    def _subset_data_vars(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include the variables specified in data_vars.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        subset_dataset = dataset
        if self.data_vars is not None:
            subset_dataset = subset_dataset[self.data_vars]
        return subset_dataset

    def _subset_valid_times(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include init times with valid times within the case period.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        indices = utils.derive_indices_from_init_time_and_lead_time(
            dataset, self.start_date, self.end_date
        )
        return dataset.isel(init_time=np.unique(indices))

    def _check_for_forecast_data_availability(
        self,
        forecast_dataset: xr.Dataset,
    ) -> bool:
        """Check if the forecast and observation datasets have overlapping time periods.

        Args:
            forecast_dataset: The forecast dataset.
            gridded_obs: The gridded observation dataset.

        Returns:
            True if the datasets have overlapping time periods, False otherwise.
        """
        lead_time_len = len(forecast_dataset.init_time)

        if lead_time_len == 0:
            logger.warning(
                "No forecast data available for case %s, skipping", self.case_id_number
            )
            return False
        elif lead_time_len < (self.end_date - self.start_date).days:
            logger.warning(
                "Fewer valid times in forecast than days in case %s, results likely unreliable",
                self.case_id_number,
            )
        else:
            logger.info("Forecast data available for case %s", self.case_id_number)
        logger.info(
            "Lead time length for case %s: %s", self.case_id_number, lead_time_len
        )
        return True


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
        return [c for c in self.cases if c.event_type == event_type]


@dataclasses.dataclass
class CaseOperator:
    """A class which stores the graph to process an individual case.

    This class is used to store the graph to process an individual case. The purpose of
    this class is to be a one-stop-shop for the evaluation of a single case. Multiple
    CaseOperators can be run in parallel to evaluate multiple cases, or run through the
    ExtremeWeatherBench.run() method to evaluate all cases in an evaluation in serial.

    Attributes:
        case: The case to process.
        metrics: A list of metrics to process.
        targets: A list of targets to process.
        target_variables: A list of target variables to process.
        forecast_variables: A list of forecast variables to process.

    Methods:
        evaluate_case: Process a case's metrics
        process_metrics: Process the metrics.
        build_targets: Build target xarray Datasets from the target sources.
    """

    case: IndividualCase
    metrics: list["metrics.BaseMetric"]
    targets: list["targets.TargetBase"]
    target_variables: list[str | "derived.DerivedVariable"]
    forecast_variables: list[str | "derived.DerivedVariable"]

    def evaluate_case(self, forecast: xr.Dataset):
        """Process a case."""
        self.process_metrics(forecast)

    def process_metrics(self, forecast: xr.Dataset):
        """Process the metrics."""
        for metric in self.metrics:
            metric.process_metric(forecast, self.targets)

    def build_targets(self, **kwargs) -> list[xr.Dataset]:
        """Build target xarray Datasets from the target sources."""
        target_storage_options = kwargs.get(
            "target_storage_options",
            {"remote_protocol": "s3", "remote_options": {"anon": True}},
        )
        target_variable_mapping = kwargs.get("target_variable_mapping", {"anon": True})
        target_datasets = []
        for target in self.targets:
            # TODO: need to pipe in storage options here
            target_dataset = target().run_pipeline(
                case=self.case,
                storage_options=target_storage_options,
                target_variables=self.target_variables,
                target_variable_mapping=target_variable_mapping,
            )
            target_datasets.append(target_dataset)

        return target_datasets
