"""Region classes and utilities for the ExtremeWeatherBench package."""

import dataclasses
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import regionmask
import xarray as xr
from shapely import Polygon  # type: ignore[import-untyped]

from extremeweatherbench import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Region(ABC):
    """Base class for different region representations."""

    @classmethod
    @abstractmethod
    def create_region(cls, *args, **kwargs) -> "Region":
        """Abstract factory method to create a region;
        subclasses must implement with their own, specialized arguments."""
        pass

    @property
    @abstractmethod
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame"""
        pass

    def mask(self, dataset: xr.Dataset, drop: bool = False) -> xr.Dataset:
        """Mask a dataset to the region."""
        mask = regionmask.mask_geopandas(
            self.geopandas, dataset.longitude, dataset.latitude
        )
        mask_array = ~np.isnan(mask)
        return dataset.where(mask_array, drop=drop)


class CenteredRegion(Region):
    """A region defined by a center point and a bounding box.

    bounding_box_degrees is the width (length) of one or all sides, not half size;
    e.g., bounding_box_degrees=10.0 means a 10 degree by 10 degree box around the center point.

    Attributes:
        latitude: Center latitude
        longitude: Center longitude
        bounding_box_degrees: Size of bounding box in degrees or tuple of (lat_degrees, lon_degrees)
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
        """Create a CenteredRegion with the given parameters."""

        return cls(
            latitude=latitude,
            longitude=longitude,
            bounding_box_degrees=bounding_box_degrees,
        )

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame"""
        # Build the region coordinates
        longitude = self.longitude
        if longitude < 0:
            longitude = utils.convert_longitude_to_360(longitude)

        if isinstance(self.bounding_box_degrees, tuple):
            bounding_box_degrees = tuple(self.bounding_box_degrees)
            latitude_min = self.latitude - bounding_box_degrees[0] / 2
            latitude_max = self.latitude + bounding_box_degrees[0] / 2
            longitude_min = longitude - bounding_box_degrees[1] / 2
            longitude_max = longitude + bounding_box_degrees[1] / 2
        else:
            latitude_min = self.latitude - self.bounding_box_degrees / 2
            latitude_max = self.latitude + self.bounding_box_degrees / 2
            longitude_min = longitude - self.bounding_box_degrees / 2
            longitude_max = longitude + self.bounding_box_degrees / 2

        longitude_min, longitude_max = _check_longitude_bounds(
            longitude_min, longitude_max
        )

        # Create polygon and geodataframe from bounding box coordinates
        polygon = Polygon(
            [
                (longitude_min, latitude_min),
                (longitude_max, latitude_min),
                (longitude_max, latitude_max),
                (longitude_min, latitude_max),
                (longitude_min, latitude_min),
            ]
        )
        return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


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

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame"""
        # Build the region coordinates

        longitude_min, longitude_max = _check_longitude_bounds(
            self.longitude_min, self.longitude_max
        )

        # Create polygon and geodataframe from bounding box coordinates
        polygon = Polygon(
            [
                (longitude_min, self.latitude_min),
                (longitude_max, self.latitude_min),
                (longitude_max, self.latitude_max),
                (longitude_min, self.latitude_max),
                (longitude_min, self.latitude_min),
            ]
        )
        return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


@dataclasses.dataclass
class ShapefileRegion(Region):
    """A region defined by a shapefile.

    A geopandas object shapefile is read in and stored as an attribute
    on instantiation.

    Attributes:
        shapefile_path: Path to the shapefile
    """

    def __repr__(self):
        return f"{self.__class__.__name__}(shapefile_path={self.shapefile_path})"

    def __init__(self, shapefile_path: str | Path):
        self.shapefile_path = Path(shapefile_path)

    @classmethod
    def create_region(cls, shapefile_path: str | Path) -> "ShapefileRegion":
        """Create a ShapefileRegion with the given parameters."""
        return cls(shapefile_path=str(shapefile_path))

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame"""
        try:
            return gpd.read_file(self.shapefile_path)
        except Exception as e:
            logger.error(f"Error reading shapefile: {e}")
            raise ValueError(f"Error reading shapefile: {e}")


def map_to_create_region(region_input: Region | dict) -> Region:
    """Map a dictionary of keyword arguments to a Region object.

    This is used to map the Region objects from the yaml file to the create_region function
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
            f"Region type '{region_type}' not registered. Available types: {list(REGION_TYPES.keys())}"
        )

    region_class = REGION_TYPES[region_type]
    if region_parameters is None:
        region_parameters = {}
    return region_class.create_region(**region_parameters)


def _check_longitude_bounds(
    longitude_min: float, longitude_max: float
) -> tuple[float, float]:
    """Check that the longitude bounds are valid."""
    if longitude_min < 0:
        longitude_min = utils.convert_longitude_to_360(longitude_min)
    if longitude_max < 0:
        longitude_max = utils.convert_longitude_to_360(longitude_max)
    if longitude_min > longitude_max:
        # Ensure max_lon is always the larger value
        longitude_min, longitude_max = (
            longitude_max,
            longitude_min,
        )
    return longitude_min, longitude_max


# Registry of region types that can be extended by users
REGION_TYPES: dict[str, Type[Region]] = {
    "centered_region": CenteredRegion,
    "bounded_region": BoundingBoxRegion,
    "shapefile_region": ShapefileRegion,
}
