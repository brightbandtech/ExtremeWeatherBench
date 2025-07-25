"""Region classes and utilities for the ExtremeWeatherBench package."""

import dataclasses
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import regionmask
import xarray as xr
from shapely import Polygon

from extremeweatherbench import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Region:
    """Base class for different region representations."""

    def to_geopandas(self):
        """Convert the region to a geopandas object."""
        if isinstance(self, ShapefileRegion):
            return self.region_gdf
        elif isinstance(self, BoundingBoxRegion) or isinstance(self, CenteredRegion):
            # Convert longitudes to 0-360 range
            lon_min_360 = utils.convert_longitude_to_360(self.longitude_min)
            lon_max_360 = utils.convert_longitude_to_360(self.longitude_max)

            # Create polygon and geodataframe from bounding box coordinates
            polygon = Polygon(
                [
                    (lon_min_360, self.latitude_min),
                    (lon_max_360, self.latitude_min),
                    (lon_max_360, self.latitude_max),
                    (lon_min_360, self.latitude_max),
                    (lon_min_360, self.latitude_min),
                ]
            )
            gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
            return gdf
        else:
            raise NotImplementedError(f"to_geopandas not implemented for {type(self)}")

    def mask(self, dataset: xr.Dataset, drop: bool = False) -> xr.Dataset:
        """Mask a dataset to the region."""
        mask = regionmask.mask_geopandas(
            self.to_geopandas(), dataset.longitude, dataset.latitude
        )
        mask = ~np.isnan(mask)
        return dataset.where(mask, drop=drop)


@dataclasses.dataclass
class CenteredRegion(Region):
    """A region defined by a center point and a bounding box.

    Attributes:
        latitude: Center latitude
        longitude: Center longitude
        bounding_box_degrees: Size of bounding box in degrees or tuple of (lat_degrees, lon_degrees)
    """

    latitude: float
    longitude: float
    bounding_box_degrees: Union[float, Tuple[float, float]]

    def __post_init__(self):
        if isinstance(self.bounding_box_degrees, tuple):
            self.bounding_box_degrees = tuple(self.bounding_box_degrees)
            self.latitude_min = self.latitude - self.bounding_box_degrees[0] / 2
            self.latitude_max = self.latitude + self.bounding_box_degrees[0] / 2
            self.longitude_min = self.longitude - self.bounding_box_degrees[1] / 2
            self.longitude_max = self.longitude + self.bounding_box_degrees[1] / 2
        else:
            self.latitude_min = self.latitude - self.bounding_box_degrees / 2
            self.latitude_max = self.latitude + self.bounding_box_degrees / 2
            self.longitude_min = self.longitude - self.bounding_box_degrees / 2
            self.longitude_max = self.longitude + self.bounding_box_degrees / 2


@dataclasses.dataclass
class BoundingBoxRegion(Region):
    """A region defined by explicit latitude and longitude bounds.

    Attributes:
        latitude_min: Minimum latitude bound
        latitude_max: Maximum latitude bound
        longitude_min: Minimum longitude bound
        longitude_max: Maximum longitude bound
    """

    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


@dataclasses.dataclass
class ShapefileRegion(Region):
    """A region defined by a shapefile.

    A geopandas object shapefile is read in and stored as an attribute
    on instantiation.

    Attributes:
        shapefile_path: Path to the shapefile
    """

    shapefile_path: Union[str, Path]

    def __post_init__(self):
        self.shapefile_path = Path(self.shapefile_path)
        try:
            self.region_gdf = gpd.read_file(self.shapefile_path)
        except Exception as e:
            logger.error(f"Error reading shapefile: {e}")
            raise ValueError(f"Error reading shapefile: {e}")


def map_to_create_region(kwargs: dict) -> Region:
    """Map a dictionary of keyword arguments to a Region object.

    This is used to map the Region objects from the yaml file to the create_region function
    with dacite.from_dict and type_hooks.

    Args:
        kwargs: A dictionary of keyword arguments to pass to the create_region function.

    Returns:
        A Region object.
    """
    return create_region(**kwargs)


def create_region(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    bounding_box_degrees: Optional[Union[float, Tuple[float, float]]] = None,
    latitude_min: Optional[float] = None,
    latitude_max: Optional[float] = None,
    longitude_min: Optional[float] = None,
    longitude_max: Optional[float] = None,
    shapefile_path: Optional[Union[str, Path]] = None,
) -> Region:
    """Factory function to create the appropriate Region subclass based on provided parameters.

    Args:
        latitude: Center latitude
        longitude: Center longitude
        bounding_box_degrees: Size of bounding box in degrees
        latitude_min: Minimum latitude bound
        latitude_max: Maximum latitude bound
        longitude_min: Minimum longitude bound
        longitude_max: Maximum longitude bound
        shapefile_path: Path to shapefile

    Returns:
        An instance of the appropriate Region subclass

    Raises:
        ValueError: If the provided parameters don't match any of the region types
    """
    if shapefile_path is not None:
        return ShapefileRegion(shapefile_path=shapefile_path)
    elif (
        latitude is not None
        and longitude is not None
        and bounding_box_degrees is not None
    ):
        return CenteredRegion(
            latitude=latitude,
            longitude=longitude,
            bounding_box_degrees=bounding_box_degrees,
        )
    elif all(
        x is not None
        for x in [latitude_min, latitude_max, longitude_min, longitude_max]
    ):
        # Type checkers can't infer that the values are not None after the all() check
        # so we need to assert or cast them
        assert latitude_min is not None
        assert latitude_max is not None
        assert longitude_min is not None
        assert longitude_max is not None
        return BoundingBoxRegion(
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
        )
    else:
        raise ValueError(
            "Invalid parameters. Must provide either (latitude, longitude, bounding_box_degrees) "
            "or (latitude_min, latitude_max, longitude_min, longitude_max) or shapefile_path."
        )


def clip_dataset_to_bounding_box_degrees(
    dataset: xr.Dataset, location: CenteredRegion
) -> xr.Dataset:
    """Clip an xarray dataset to a box around a given location in degrees latitude & longitude.

    Args:
        dataset: The input xarray dataset.
        location_center: A Location object corresponding to the center of the bounding box.
        box_degrees: The side length(s) of the bounding box in degrees, as a tuple (lat,lon) or single value.

    Returns:
        The clipped xarray dataset.
    """

    lat_center = location.latitude
    lon_center = location.longitude
    if lon_center < 0:
        lon_center = utils.convert_longitude_to_360(lon_center)
    if isinstance(location.bounding_box_degrees, tuple):
        box_degrees_lat, box_degrees_lon = location.bounding_box_degrees
    else:
        box_degrees_lat = location.bounding_box_degrees
        box_degrees_lon = location.bounding_box_degrees
    min_lat = lat_center - box_degrees_lat / 2
    max_lat = lat_center + box_degrees_lat / 2
    min_lon = lon_center - box_degrees_lon / 2
    max_lon = lon_center + box_degrees_lon / 2
    if min_lon < 0:
        min_lon = utils.convert_longitude_to_360(min_lon)
    if min_lon > max_lon:
        # Ensure max_lon is always the larger value and account for cyclic nature of lon
        min_lon, max_lon = max_lon, min_lon
        clipped_dataset = dataset.sel(
            latitude=(dataset.latitude > min_lat) & (dataset.latitude <= max_lat),
            longitude=(dataset.longitude < min_lon) | (dataset.longitude >= max_lon),
        )
    else:
        clipped_dataset = dataset.sel(
            latitude=(dataset.latitude > min_lat) & (dataset.latitude <= max_lat),
            longitude=(dataset.longitude > min_lon) & (dataset.longitude <= max_lon),
        )
    return clipped_dataset
