"""Region classes and utilities for the ExtremeWeatherBench package."""

import abc
import dataclasses
import logging
import pathlib
from collections import namedtuple
from typing import Any, Type

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import regionmask
import shapely  # type: ignore[import-untyped]
import xarray as xr

from extremeweatherbench import utils

logger = logging.getLogger(__name__)


class Region(abc.ABC):
    """Base class for different region representations."""

    @classmethod
    @abc.abstractmethod
    def create_region(cls, *args, **kwargs) -> "Region":
        """Abstract factory method to create a region; subclasses must implement with
        their own, specialized arguments."""
        pass

    @property
    @abc.abstractmethod
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame."""
        pass

    @property
    def get_bounding_coordinates(self) -> tuple[Any, ...]:
        """Get the bounding coordinates of the region.

        Returns:
            A named tuple with longitude_min, latitude_min, longitude_max, and
            latitude_max.
        """
        return namedtuple(
            "BoundingCoordinates",
            ["longitude_min", "latitude_min", "longitude_max", "latitude_max"],
        )(*self.geopandas.total_bounds)

    def mask(self, dataset: xr.Dataset, drop: bool = False) -> xr.Dataset:
        """Mask a dataset to the region.

        Args:
            dataset: The dataset to mask.

        Returns:
            The subset dataset.
        """
        if drop:
            logger.info(
                "drop is no longer used and will be removed in a future version."
            )
        # If the lats are monotonically decreasing, reverse the slice (max, min)
        longitude_min, latitude_min, longitude_max, latitude_max = (
            self.geopandas.total_bounds
        )
        latitude_order = dataset.latitude.values.argsort()
        if latitude_order[0] > latitude_order[-1]:
            dataset = dataset.sel(
                latitude=slice(latitude_max, latitude_min),
                longitude=slice(longitude_min, longitude_max),
            )
        else:
            # If monotonically increasing, slice normally (min, max)
            dataset = dataset.sel(
                latitude=slice(latitude_min, latitude_max),
                longitude=slice(longitude_min, longitude_max),
            )

        return dataset


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

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
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

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
        """Return representation of this Region as a GeoDataFrame.

        Returns:
            A GeoDataFrame representing the region.
        """
        return _create_geopandas_from_bounds(
            self.longitude_min, self.longitude_max, self.latitude_min, self.latitude_max
        )


@dataclasses.dataclass
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

    @property
    def geopandas(self) -> gpd.GeoDataFrame:
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
        longitude_min, latitude_min, longitude_max, latitude_max = (
            self.geopandas.total_bounds
        )
        try:
            dataset = dataset.sel(
                latitude=slice(latitude_min, latitude_max),
                longitude=slice(longitude_min, longitude_max),
                drop=drop,
            )
        except Exception:
            # If the latitude slice fails, try the reverse
            dataset = dataset.sel(
                latitude=slice(latitude_max, latitude_min),
                longitude=slice(longitude_min, longitude_max),
                drop=drop,
            )
        # Subset dataset after cutting out a box to minimize memory pressure
        mask = regionmask.mask_geopandas(
            self.geopandas, dataset.longitude, dataset.latitude
        )
        mask_array = ~np.isnan(mask)
        return dataset.where(mask_array, drop=drop)


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
