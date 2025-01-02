"""Utiltiies for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBench (Rasp et al.)."""

import dataclasses
import datetime
from extremeweatherbench.utils import Location
from typing import List, Optional
from extremeweatherbench import metrics, utils
import xarray as xr
from enum import StrEnum
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single or individual case.

    An IndividualCase defines the relevant metadata for a single case study for a
    given extreme weather event; it is designed to be easily instantiable through a
    simple YAML-based configuration file.

    Attributes:
        id: A unique numerical identifier for the event.
        start_date: A datetime.date object representing the start date of the case, for
            use in subsetting data for analysis.
        end_date: A datetime.date object representing the end date of the case, for use
            in subsetting data for analysis.
        location: A Location object representing the latitude and longitude of the event
            center or  focus.
        bounding_box_km: int: The side length of a bounding box centered on location, in
            kilometers.
        event_type: str: A string representing the type of extreme weather event.
        cross_listed: Optional[List[str]]: A list of other event types that this case
            study is cross-listed under.
    """

    id: int
    title: str
    start_date: datetime.date
    end_date: datetime.date
    location: dict
    bounding_box_km: float
    event_type: str
    cross_listed: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.location, dict):
            self.location = Location(**self.location)

    def perform_subsetting_procedure(self, dataset: xr.Dataset) -> xr.Dataset:
        """Perform any necessary subsetting procedures on the input dataset.

        This method is designed to be overridden by subclasses to provide custom
        subsetting procedures for specific event types.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        raise NotImplementedError

    def _subset_data_vars(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include the variables specified in data_vars.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        subset_dataset = dataset.copy()
        if self.data_vars is not None:
            subset_dataset = subset_dataset[self.data_vars]
        return subset_dataset

    def subset_valid_times(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include the valid times within the case period.
        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        time_subset_ds = dataset.sel(time=slice(self.start_date, self.end_date))
        return time_subset_ds

    def check_for_forecast_data_availability(
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
        valid_time_len = len(forecast_dataset.time)

        if valid_time_len == 0:
            logger.warning(f"No forecast data available for case {self.id}, skipping")
            return False
        elif valid_time_len < (self.end_date - self.start_date).days:
            logger.warning(
                f"Fewer valid times in forecast than days in case {self.id}, results likely unreliable"
            )
        else:
            logger.info(f"Forecast data available for case {self.id}")
        logger.info(f"Lead time length for case {self.id}: {lead_time_len}")
        logger.info(
            f"Total time step count (valid times by forecasr hour) for case: {lead_time_len*valid_time_len}"
        )
        return True


@dataclasses.dataclass
class IndividualHeatWaveCase(IndividualCase):
    """Container for metadata defining a single or individual case of a heat wave.

    An IndividualHeatWaveCase is a subclass of IndividualCase that is designed to
    provide additional metadata specific to heat wave events.

    Attributes:
        metrics_list: A list of Metrics to be used in the evaluation
    """

    metrics_list: List[metrics.Metric] = dataclasses.field(
        default_factory=lambda: [metrics.RegionalRMSE]
    )
    data_vars: List[str] = dataclasses.field(
        default_factory=lambda: ["air_temperature"]
    )

    def perform_subsetting_procedure(self, dataset: xr.Dataset) -> xr.Dataset:
        modified_ds = self._subset_data_vars(dataset)
        modified_ds = utils.clip_dataset_to_bounding_box(
            modified_ds, self.location, self.bounding_box_km
        )
        modified_ds = utils.remove_ocean_gridpoints(modified_ds)
        return modified_ds


@dataclasses.dataclass
class IndividualFreezeCase(IndividualCase):
    """Container for metadata defining a single or individual case of a freeze event.

    An IndividualFreezeCase is a subclass of IndividualCase that is designed to
    provide additional metadata specific to freeze events.

    Attributes:
        freeze_type: str: A string representing the type of freeze event.
    """

    metrics_list: List[metrics.Metric] = dataclasses.field(
        default_factory=lambda: [metrics.RegionalRMSE]
    )
    data_vars: List[str] = dataclasses.field(
        default_factory=lambda: ["air_temperature", "eastward_wind", "northward_wind"]
    )

    def perform_subsetting_procedure(self, dataset) -> xr.Dataset:
        modified_ds = utils.clip_dataset_to_bounding_box(
            dataset, self.location, self.bounding_box_km
        )
        return modified_ds


# maps the case event type to the corresponding dataclass
# additional event types need to be added here and
# CASE_EVENT_TYPE_MATCHER, which maps the metadata case event type
# to the corresponding case dataclass.
class CaseEventType(StrEnum):
    """Enum class for the different types of extreme weather events."""

    HEAT_WAVE = "heat_wave"
    FREEZE = "freeze"


CASE_EVENT_TYPE_MATCHER: dict[CaseEventType, IndividualCase] = {
    CaseEventType.HEAT_WAVE: IndividualHeatWaveCase,
    CaseEventType.FREEZE: IndividualFreezeCase,
}


def get_case_event_dataclass(case_type: str) -> IndividualCase:
    event_dataclass = CASE_EVENT_TYPE_MATCHER.get(case_type)
    if event_dataclass is None:
        raise ValueError(f"Unknown case event type {case_type}")
    return event_dataclass
