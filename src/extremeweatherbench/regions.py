"""Region classes and utilities for the ExtremeWeatherBench package."""

import abc
import logging
import pathlib
import warnings
from typing import TYPE_CHECKING, Literal, Mapping, Type, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import shapely
import xarray as xr

from extremeweatherbench import utils

if TYPE_CHECKING:
    from extremeweatherbench import cases

logger = logging.getLogger(__name__)


class Region(abc.ABC):
    """Base class for different region representations."""

    @classmethod
    @abc.abstractmethod
    def create_region(cls, *args, **kwargs) -> "Region":
        """Abstract factory method to create a region; subclasses must implement with
        their own, specialized arguments."""
        pass

    @abc.abstractmethod
    def as_geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame."""
        pass

    def get_adjusted_bounds(
        self, dataset: xr.Dataset
    ) -> tuple[float, float, float, float]:
        """Get region bounds adjusted to dataset's longitude convention.

        Args:
            dataset: The dataset to match longitude convention with

        Returns:
            Tuple of (lon_min, lat_min, lon_max, lat_max) adjusted to
            match the dataset's longitude convention
        """
        region_bounds = self.as_geopandas().total_bounds
        return _adjust_bounds_to_dataset_convention(region_bounds, dataset)

    def mask(self, dataset: xr.Dataset, drop: bool = False) -> xr.Dataset:
        """Mask a dataset to the region.

        Args:
            dataset: The dataset to mask.
            drop: Whether to drop coordinates outside the region bounds.

        Returns:
            The subset dataset.
        """
        if drop:
            warnings.warn(
                "`drop` is no longer used and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Get region bounds adjusted to dataset's longitude convention
        (
            region_longitude_min,
            region_latitude_min,
            region_longitude_max,
            region_latitude_max,
        ) = self.get_adjusted_bounds(dataset)

        # Avoids slice() which is susceptible to differences in coord order
        latitude_da = dataset.latitude.where(
            np.logical_and(
                dataset.latitude >= region_latitude_min,
                dataset.latitude <= region_latitude_max,
            ),
            drop=True,
        )

        # Detect if region wraps around 0/360 or -180/180 boundary
        # This happens either from:
        # 1. True antimeridian crossing (MultiPolygon geometry)
        # 2. Prime meridian crossing converted to 0-360 (lon_min > lon_max)
        gdf = self.as_geopandas()
        geometry = gdf.geometry.iloc[0]
        crosses_boundary = isinstance(geometry, shapely.MultiPolygon) or (
            region_longitude_min > region_longitude_max
        )

        if crosses_boundary:
            # Use OR condition: include lons >= min OR lons <= max
            longitude_da = dataset.longitude.where(
                np.logical_or(
                    dataset.longitude >= region_longitude_min,
                    dataset.longitude <= region_longitude_max,
                ),
                drop=True,
            )
        else:
            longitude_da = dataset.longitude.where(
                np.logical_and(
                    dataset.longitude >= region_longitude_min,
                    dataset.longitude <= region_longitude_max,
                ),
                drop=True,
            )
        dataset = dataset.sel(
            latitude=latitude_da,
            longitude=longitude_da,
        )

        return dataset

    def intersects(self, other: "Region") -> bool:
        """Check if this region intersects with another region."""
        return self.as_geopandas().intersects(other.as_geopandas()).any().any()

    def contains(self, other: "Region") -> bool:
        """Check if this region completely contains another region."""
        return self.as_geopandas().contains(other.as_geopandas()).any().any()

    def area_overlap_fraction(self, other: "Region") -> float:
        """Calculate fraction of other region's area that overlaps with this region.

        Args:
            other: The other region (case region) to check

        Returns:
            Fraction of other region's area within this region (0.0 to 1.0)
        """
        # Calculate intersection area between regions
        intersection = self.as_geopandas().overlay(
            other.as_geopandas(), how="intersection"
        )
        if intersection.empty:
            return 0.0

        # Convert to equal area projection for accurate area calculation
        # Using World Mollweide projection (ESRI:54009)
        intersection_projected = intersection.to_crs("ESRI:54009")
        other_projected = other.as_geopandas().to_crs("ESRI:54009")

        # Calculate areas in square meters
        intersection_area = intersection_projected.geometry.area.sum()
        other_area = other_projected.geometry.area.sum()

        if other_area == 0:
            return 0.0

        return intersection_area / other_area


class CenteredRegion(Region):
    """A region defined by a center point and a bounding box.

    bounding_box_degrees is the width (length) of one or all sides, not half size;
    e.g., bounding_box_degrees=10.0 means a 10 degree by 10 degree box around
    the center point.

    Attributes:
        latitude: Center latitude
        longitude: Center longitude
        bounding_box_degrees: Size of bounding box in degrees or tuple of
            (lat_degrees, lon_degrees)
    """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(latitude={self.latitude}, "
            f"longitude={self.longitude}, "
            f"bounding_box_degrees={self.bounding_box_degrees})"
        )

    def __init__(
        self, latitude: float, longitude: float, bounding_box_degrees: float | tuple
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.bounding_box_degrees = bounding_box_degrees

    @classmethod
    def create_region(
        cls, latitude: float, longitude: float, bounding_box_degrees: float | tuple
    ) -> "CenteredRegion":
        """Create a CenteredRegion with the given parameters.

        Args:
            latitude: The latitude of the center point.
            longitude: The longitude of the center point.
            bounding_box_degrees: The size of the bounding box in degrees or tuple of
                (lat_degrees, lon_degrees).
        """

        return cls(
            latitude=latitude,
            longitude=longitude,
            bounding_box_degrees=bounding_box_degrees,
        )

    def as_geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame.

        Returns:
            A GeoDataFrame representing the region.
        """
        if isinstance(self.bounding_box_degrees, tuple):
            bounding_box_degrees = tuple(self.bounding_box_degrees)
            latitude_min = self.latitude - bounding_box_degrees[0] / 2
            latitude_max = self.latitude + bounding_box_degrees[0] / 2
            longitude_min = self.longitude - bounding_box_degrees[1] / 2
            longitude_max = self.longitude + bounding_box_degrees[1] / 2
        else:
            latitude_min = self.latitude - self.bounding_box_degrees / 2
            latitude_max = self.latitude + self.bounding_box_degrees / 2
            longitude_min = self.longitude - self.bounding_box_degrees / 2
            longitude_max = self.longitude + self.bounding_box_degrees / 2

        return _create_geopandas_from_bounds(
            longitude_min, longitude_max, latitude_min, latitude_max
        )


class BoundingBoxRegion(Region):
    """A region defined by explicit latitude and longitude bounds.

    Attributes:
        latitude_min: Minimum latitude bound
        latitude_max: Maximum latitude bound
        longitude_min: Minimum longitude bound
        longitude_max: Maximum longitude bound
    """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(latitude_min={self.latitude_min}, "
            f"latitude_max={self.latitude_max}, "
            f"longitude_min={self.longitude_min}, "
            f"longitude_max={self.longitude_max})"
        )

    def __init__(
        self,
        latitude_min: float,
        latitude_max: float,
        longitude_min: float,
        longitude_max: float,
    ):
        self.latitude_min = latitude_min
        self.latitude_max = latitude_max
        self.longitude_min = longitude_min
        self.longitude_max = longitude_max

    @classmethod
    def create_region(
        cls,
        latitude_min: float,
        latitude_max: float,
        longitude_min: float,
        longitude_max: float,
    ) -> "BoundingBoxRegion":
        """Create a BoundingBoxRegion with the given parameters."""
        return cls(
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
        )

    def as_geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame.

        Returns:
            A GeoDataFrame representing the region.
        """
        return _create_geopandas_from_bounds(
            self.longitude_min, self.longitude_max, self.latitude_min, self.latitude_max
        )


class ShapefileRegion(Region):
    """A region defined by a shapefile.

    A geopandas object shapefile is read in and stored as an attribute
    on instantiation.

    Attributes:
        shapefile_path: Local or remote path to the .shp shapefile
    """

    def __repr__(self):
        return f"{self.__class__.__name__}(shapefile_path={self.shapefile_path})"

    def __init__(self, shapefile_path: str | pathlib.Path):
        self.shapefile_path = pathlib.Path(shapefile_path)

    @classmethod
    def create_region(cls, shapefile_path: str | pathlib.Path) -> "ShapefileRegion":
        """Create a ShapefileRegion with the given parameters."""
        return cls(shapefile_path=str(shapefile_path))

    def as_geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame.

        Returns:
            A GeoDataFrame representing the region.
        """
        try:
            return gpd.read_file(self.shapefile_path)
        except Exception as e:
            logger.error(f"Error reading shapefile: {e}")
            raise ValueError(f"Error reading shapefile: {e}")

    def mask(self, dataset: xr.Dataset, drop: bool = False) -> xr.Dataset:
        """Mask a dataset to the region.

        Args:
            dataset: The dataset to mask.
            drop: Whether to drop NaN values outside the region. Defaults to False.

        Returns:
            The subset dataset.
        """
        # Get region bounds adjusted to dataset's longitude convention
        (
            longitude_min,
            latitude_min,
            longitude_max,
            latitude_max,
        ) = self.get_adjusted_bounds(dataset)

        # Note: ShapefileRegion.mask uses slice which doesn't support
        # prime/antimeridian crossing with OR logic, but regionmask handles it
        dataset = dataset.sel(
            latitude=slice(latitude_max, latitude_min),
            longitude=slice(longitude_min, longitude_max),
            drop=drop,
        )
        # Subset dataset after cutting out a box to minimize memory pressure
        mask = regionmask.mask_geopandas(
            self.as_geopandas(), dataset.longitude, dataset.latitude
        )
        return dataset.where(~np.isnan(mask), drop=drop)


# Registry of region types that can be extended by users
REGION_TYPES: dict[str, Type[Region]] = {
    "centered_region": CenteredRegion,
    "bounded_region": BoundingBoxRegion,
    "shapefile_region": ShapefileRegion,
}


def map_to_create_region(region_input: Region | dict) -> Region:
    """Map a dictionary of keyword arguments to a Region object.

    This is used to map the Region objects from the yaml file to the
    create_region function
    with dacite.from_dict and type_hooks.

    Args:
        region_input: Either a Region object or a dictionary of parameters.

    Returns:
        A Region object.
    """
    if isinstance(region_input, Region):
        return region_input

    region_type = region_input.get("type")
    region_parameters = region_input.get("parameters")

    if region_type not in REGION_TYPES:
        raise KeyError(
            f"Region type '{region_type}' not registered. Available types: "
            f"{list(REGION_TYPES.keys())}"
        )

    region_class = REGION_TYPES[region_type]
    if region_parameters is None:
        region_parameters = {}
    return region_class.create_region(**region_parameters)


def _create_geopandas_from_bounds(
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from bounding box coordinates with antimeridian handling.

    Args:
        longitude_min: Minimum longitude
        longitude_max: Maximum longitude
        latitude_min: Minimum latitude
        latitude_max: Maximum latitude

    Returns:
        GeoDataFrame with proper geometry handling antimeridian crossing
    """
    # Check if the original coordinates cross the antimeridian before conversion
    # This happens when the longitude range naturally crosses 180°/-180°
    original_crosses_antimeridian = (
        longitude_max > 180 and longitude_min < longitude_max - 360
    )

    # Convert longitude coordinates to -180 to 180 range
    lon_min = utils.convert_longitude_to_180(longitude_min)
    lon_max = utils.convert_longitude_to_180(longitude_max)

    # Special case: if original coordinates went exactly to 180°, keep it as
    # 180° instead of -180°
    if longitude_max == 180:
        lon_max = 180

    # Handle antimeridian crossing
    # Check if we have a true antimeridian crossing (not just a conversion artifact)
    crosses_antimeridian = original_crosses_antimeridian or (
        lon_min > lon_max and not (longitude_max == 180 and lon_max == 180)
    )

    if crosses_antimeridian:
        # Create two polygons: one for each side of the antimeridian
        polygon1 = shapely.Polygon(
            [
                (lon_min, latitude_min),
                (180, latitude_min),
                (180, latitude_max),
                (lon_min, latitude_max),
                (lon_min, latitude_min),
            ]
        )
        polygon2 = shapely.Polygon(
            [
                (-180, latitude_min),
                (lon_max, latitude_min),
                (lon_max, latitude_max),
                (-180, latitude_max),
                (-180, latitude_min),
            ]
        )
        # Use a MultiPolygon or combine the geometries
        multi_polygon = shapely.MultiPolygon([polygon1, polygon2])
        return gpd.GeoDataFrame(geometry=[multi_polygon], crs="EPSG:4326")
    else:
        # Normal case - no antimeridian crossing
        polygon = shapely.Polygon(
            [
                (lon_min, latitude_min),
                (lon_max, latitude_min),
                (lon_max, latitude_max),
                (lon_min, latitude_max),
                (lon_min, latitude_min),
            ]
        )
        return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


class RegionSubsetter:
    """A utility class for subsetting ExtremeWeatherBench objects by region.

    Attributes:
        region: The region to subset to. Can be a Region object or a
            dictionary of bounds with keys "latitude_min", "latitude_max",
            "longitude_min", and "longitude_max".
        method: The method to use for subsetting. Options:
            - "intersects": Include cases where ANY part of a case intersects region
            - "percent": Include cases where percent of case area overlaps with region.
            - "all": Only include cases where entirety of a case is within region
        percent_threshold: Threshold for percent overlap (0.0 to 1.0)
    """

    region: Region
    method: str
    percent_threshold: float

    def __init__(
        self,
        region: Union[Region, Mapping[str, float]],
        method: Literal["intersects", "percent", "all"] = "intersects",
        percent_threshold: float = 0.5,
    ):
        """Initialize the RegionSubsetter.

        Args:
            region: The region to subset to. Can be a Region object or a
                dictionary of bounds with keys "latitude_min", "latitude_max",
                "longitude_min", and "longitude_max".
            method: The method to use for subsetting. Options:
                - "intersects": Include cases where ANY part of a case intersects region
                - "percent": Include cases where percent of case area overlaps with
                region
                - "all": Only include cases where entirety of a case is within region
            percent_threshold: Threshold for percent overlap (0.0 to 1.0)
        """
        # Convert dictionary input to BoundingBoxRegion if needed
        if isinstance(region, Mapping):
            self.region = BoundingBoxRegion.create_region(
                latitude_min=region["latitude_min"],
                latitude_max=region["latitude_max"],
                longitude_min=region["longitude_min"],
                longitude_max=region["longitude_max"],
            )
        else:
            self.region = region

        self.method = method
        self.percent_threshold = percent_threshold

    def subset_case_collection(
        self, case_collection: "list[cases.IndividualCase]"
    ) -> "list[cases.IndividualCase]":
        """Subset a list of IndividualCases by region.

        Args:
            case_collection: The list of IndividualCases to subset

        Returns:
            A new list of IndividualCases with cases subset to the region
        """
        # Avoid circular import

        filtered_cases = []

        for case in case_collection:
            if self._should_include_case(case):
                filtered_cases.append(case)

        return filtered_cases

    def _should_include_case(self, case: "cases.IndividualCase") -> bool:
        """Determine if a case should be included based on the subsetting criteria."""
        case_location = case.location

        if self.method == "intersects":
            return self.region.intersects(case_location)

        elif self.method == "all":
            return self.region.contains(case_location)

        elif self.method == "percent":
            # Calculate fraction of case area within the region
            overlap_fraction = self.region.area_overlap_fraction(case_location)
            return overlap_fraction >= self.percent_threshold

        else:
            raise ValueError(f"Unknown method: {self.method}")


# Convenience functions for direct usage
def subset_cases_to_region(
    case_collection: "list[cases.IndividualCase]",
    region: Union[Region, Mapping[str, float]],
    method: Literal["intersects", "percent", "all"] = "intersects",
    percent_threshold: float = 0.5,
) -> "list[cases.IndividualCase]":
    """Subset a list of IndividualCases to a region.

    This is a convenience function that creates a RegionSubsetter and applies it to
    a list of IndividualCases.

    Args:
        case_collection: The list of IndividualCases to subset
        region: The region to subset to. Can be a Region object or a
            dictionary of bounds with keys "latitude_min", "latitude_max",
            "longitude_min", and "longitude_max".
        method: The subsetting method for RegionSubsetter
        percent_threshold: Threshold for percent overlap

    Returns:
        A new list of IndividualCases with cases subset to the region
    """
    subsetter = RegionSubsetter(region, method, percent_threshold)
    return subsetter.subset_case_collection(case_collection)


def subset_results_to_region(
    region: RegionSubsetter,
    results_df: pd.DataFrame,
    case_collection: "list[cases.IndividualCase]",
) -> pd.DataFrame:
    """Subset results DataFrame by region using case_id_number.

    This is a convenience function that creates a RegionSubsetter and applies it to
    the results DataFrame that is output from ExtremeWeatherBench.run().

    Args:
        region: The region to subset to. Can be a Region object or a
            dictionary of bounds with keys "latitude_min", "latitude_max",
            "longitude_min", and "longitude_max".
        results_df: DataFrame with results from ExtremeWeatherBench.run()
        case_collection: The original case collection to determine which
            case_id_numbers correspond to cases in the region

    Returns:
        Subset DataFrame containing only results for cases in the region
    """
    # Get the case IDs that should be included
    subset_cases = region.subset_case_collection(case_collection)
    included_case_ids = {case.case_id_number for case in subset_cases}

    # Filter the results DataFrame
    return results_df[results_df["case_id_number"].isin(included_case_ids)]


def _adjust_bounds_to_dataset_convention(
    region_bounds: tuple[float, float, float, float],
    dataset: xr.Dataset,
    longitude_coord: str = "longitude",
) -> tuple[float, float, float, float]:
    """Adjust region bounds to match dataset longitude convention.

    Uses existing utils functions to convert between 0-360 and -180/+180.

    Args:
        region_bounds: Tuple of (lon_min, lat_min, lon_max, lat_max)
        dataset: The dataset to match conventions with
        longitude_coord: Name of the longitude coordinate

    Returns:
        Adjusted bounds as (lon_min, lat_min, lon_max, lat_max)
    """
    lon_min, lat_min, lon_max, lat_max = region_bounds

    # Detect if dataset uses 0-360 convention (has values > 180)
    ds_uses_360 = np.any(dataset[longitude_coord].values > 180)

    if ds_uses_360:
        # Convert region bounds to 0-360 to match dataset
        lon_min = utils.convert_longitude_to_360(lon_min)
        lon_max = utils.convert_longitude_to_360(lon_max)
    # If dataset uses -180/+180, region bounds are already in that
    # convention from _create_geopandas_from_bounds

    return (lon_min, lat_min, lon_max, lat_max)
