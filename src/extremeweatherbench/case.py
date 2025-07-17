"""Classes for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBench (Rasp et al.)."""

import dataclasses
import datetime
import logging
from enum import StrEnum
from typing import List, Optional, Type

import numpy as np
import regionmask
import xarray as xr

from extremeweatherbench import metrics, utils

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
    location: utils.Region
    event_type: str
    data_vars: Optional[List[str]] = None
    cross_listed: Optional[List[str]] = None

    # TODO: unit tests and fix shapefile region loader (shouldn't work right now, boilerplate code)
    def subset_region(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset the input dataset to the region specified in the location.

        Args:
            dataset: xr.Dataset: The input dataset to subset.

        Returns:
            xr.Dataset: The subset dataset.
        """
        if isinstance(self.location, utils.CenteredRegion):
            modified_ds = utils.clip_dataset_to_bounding_box_degrees(
                dataset, self.location
            )
        elif isinstance(self.location, utils.BoundingBoxRegion):
            modified_ds = dataset.sel(
                latitude=slice(self.location.latitude_min, self.location.latitude_max),
                longitude=slice(
                    self.location.longitude_min, self.location.longitude_max
                ),
            )
        elif isinstance(self.location, utils.ShapefileRegion):
            shapefile_path = self.location.shapefile_path
            mask = regionmask.from_geopandas(shapefile_path, names="region")
            region_mask = mask.mask(dataset, lon_name="longitude", lat_name="latitude")
            modified_ds = dataset.where(~np.isnan(region_mask))
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")
        return modified_ds

    def perform_subsetting_procedure(self, dataset: xr.Dataset) -> xr.Dataset:
        """Perform any necessary subsetting procedures on the input dataset.

        This method is designed to be overridden by subclasses to provide custom
        subsetting procedures for specific event types.

        This method is deprecated and will be removed in a future version.

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
class IndividualHeatWaveCase(IndividualCase):
    """Container for metadata defining a single or individual case of a heat wave.

    An IndividualHeatWaveCase is a subclass of IndividualCase that is designed to
    provide additional metadata specific to heat wave events.

    Attributes:
        metrics_list: A list of Metrics to be used in the evaluation
    """

    metrics_list: List[Type[metrics.Metric]] = dataclasses.field(
        default_factory=lambda: [
            metrics.MaxOfMinTempMAE,
            metrics.RegionalRMSE,
            metrics.MaximumMAE,
        ]
    )
    data_vars: List[str] = dataclasses.field(
        default_factory=lambda: ["surface_air_temperature"]
    )

    def __post_init__(self):
        self.data_vars = ["surface_air_temperature"]

    def perform_subsetting_procedure(self, dataset: xr.Dataset) -> xr.Dataset:
        if isinstance(self.location, utils.CenteredRegion):
            modified_ds = utils.clip_dataset_to_bounding_box_degrees(
                dataset, self.location
            )
        elif isinstance(self.location, utils.BoundingBoxRegion):
            modified_ds = dataset.sel(
                latitude=slice(self.location.latitude_min, self.location.latitude_max),
                longitude=slice(
                    self.location.longitude_min, self.location.longitude_max
                ),
            )
        elif isinstance(self.location, utils.ShapefileRegion):
            # Create a mask from the shapefile
            shapefile_path = self.location.shapefile_path
            mask = regionmask.from_geopandas(shapefile_path, names="region")
            # Apply the mask to the dataset
            region_mask = mask.mask(dataset, lon_name="longitude", lat_name="latitude")
            # Select only points within the region (where mask value is not NaN)
            modified_ds = dataset.where(~np.isnan(region_mask), drop=True)
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")
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

    metrics_list: List[Type[metrics.Metric]] = dataclasses.field(
        default_factory=lambda: [metrics.RegionalRMSE]
    )
    data_vars: List[str] = dataclasses.field(
        default_factory=lambda: [
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
    )

    def __post_init__(self):
        self.data_vars = [
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

    def perform_subsetting_procedure(self, dataset: xr.Dataset) -> xr.Dataset:
        if isinstance(self.location, utils.CenteredRegion):
            modified_ds = utils.clip_dataset_to_bounding_box_degrees(
                dataset, self.location
            )
        elif isinstance(self.location, utils.BoundingBoxRegion):
            modified_ds = dataset.sel(
                latitude=slice(self.location.latitude_min, self.location.latitude_max),
                longitude=slice(
                    self.location.longitude_min, self.location.longitude_max
                ),
            )
        elif isinstance(self.location, utils.ShapefileRegion):
            # Create a mask from the shapefile
            shapefile_path = self.location.shapefile_path
            mask = regionmask.from_geopandas(shapefile_path, names="region")
            # Apply the mask to the dataset
            region_mask = mask.mask(dataset, lon_name="longitude", lat_name="latitude")
            # Select only points within the region (where mask value is not NaN)
            modified_ds = dataset.where(~np.isnan(region_mask), drop=True)
        else:
            raise ValueError(f"Unsupported location type: {type(self.location)}")
        return modified_ds


# maps the case event type to the corresponding dataclass
# additional event types need to be added here and
# CASE_EVENT_TYPE_MATCHER, which maps the metadata case event type
# to the corresponding case dataclass.
class CaseEventType(StrEnum):
    """Enum class for the different types of extreme weather events."""

    HEAT_WAVE = "heat_wave"
    FREEZE = "freeze"


CASE_EVENT_TYPE_MATCHER: dict[CaseEventType, type[IndividualCase]] = {
    CaseEventType.HEAT_WAVE: IndividualHeatWaveCase,
    CaseEventType.FREEZE: IndividualFreezeCase,
}


def get_case_event_dataclass(case_type: str) -> Type[IndividualCase]:
    event_dataclass = CASE_EVENT_TYPE_MATCHER.get(CaseEventType(case_type))
    if event_dataclass is None:
        raise ValueError(f"Unknown case event type {case_type}")
    return event_dataclass
