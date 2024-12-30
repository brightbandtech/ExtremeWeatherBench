"""Utiltiies for defining individual units of case studies for analysis.
Some code similarly structured to WeatherBench (Rasp et al.)."""

import dataclasses
import datetime
from extremeweatherbench.utils import Location
from typing import List, Optional, Dict, Type
from extremeweatherbench import metrics, utils
import xarray as xr


@dataclasses.dataclass
class CaseEventTypeMatcher:
    """A container for defining a mapping between event types and their corresponding
    case study classes.

    Attributes:

    """

    mapping: Dict[str, Type] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Automatically populate the mapping with string -> child dataclass pairs
        self.mapping = {
            "heat_wave": IndividualHeatWaveCase,
            "freeze": IndividualFreezeCase,
        }

    def get_dataclass(self, key: str):
        """
        Retrieve an instance of the corresponding dataclass based on the key.
        kwargs will be passed to the constructor of the dataclass.
        """
        dataclass_type = self.mapping.get(key)
        if dataclass_type is None:
            raise ValueError(f"No mapping found for key '{key}'")
        return dataclass_type


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
    bounding_box_km: int
    event_type: str
    cross_listed: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.location, dict):
            self.location = Location(**self.location)

    def get_event_specific_case_type(self):
        mapping = CaseEventTypeMatcher()
        _case_event_type = mapping.get_dataclass(self.event_type)

        return _case_event_type(**vars(self))

    def perform_subsetting_procedure(self, dataset) -> xr.Dataset:
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

    def perform_subsetting_procedure(self, dataset) -> xr.Dataset:
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
