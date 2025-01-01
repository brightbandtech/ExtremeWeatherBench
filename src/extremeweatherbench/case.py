"""Utiltiies for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBench (Rasp et al.)."""

import dataclasses
import datetime
from extremeweatherbench.utils import Location
from typing import List, Optional, Dict, Type
from extremeweatherbench import metrics, utils
import xarray as xr
from enum import StrEnum


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

    def subset_data_vars(self, dataset):
        """Subset the input dataset to only include the variables specified in data_vars.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        raise NotImplementedError


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
        modified_ds = dataset.sel(time=slice(self.start_date, self.end_date))
        modified_ds = self._subset_data_vars(modified_ds)
        modified_ds = utils.convert_longitude_to_180(modified_ds)
        modified_ds = utils.clip_dataset_to_bounding_box(
            modified_ds, self.location, self.bounding_box_km
        )
        modified_ds = utils.remove_ocean_gridpoints(modified_ds)
        return modified_ds

    def _subset_data_vars(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include the variables specified in data_vars.

        Args:
            dataset: xr.Dataset: The input dataset to subset.
            data_vars: the variables within the ForecastSchemaConfig to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        if self.data_vars is not None:
            return dataset[self.data_vars]
        return dataset


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
        modified_ds = dataset.sel(time=slice(self.start_date, self.end_date))
        modified_ds = utils.convert_longitude_to_180(dataset)
        modified_ds = utils.clip_dataset_to_bounding_box(
            dataset, self.location, self.bounding_box_km
        )
        return modified_ds

    def subset_data_vars(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to only include the variables specified in data_vars.

        Args:
            dataset: xr.Dataset: The input dataset to subset.
            data_vars: the variables within the ForecastSchemaConfig to subset.
        Returns:
            xr.Dataset: The subset dataset.
        """
        if self.data_vars is not None:
            return dataset[self.data_vars]
        return dataset


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
