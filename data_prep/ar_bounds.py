#!/usr/bin/env python3
"""Calculate case bounds from AR mask data."""

import logging
import pickle
from typing import Dict, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
from dask.distributed import Client
from matplotlib.patches import Rectangle

import extremeweatherbench.cases as cases
import extremeweatherbench.derived as derived
import extremeweatherbench.inputs as inputs
import extremeweatherbench.regions as regions
import extremeweatherbench.utils as utils
from extremeweatherbench.events import atmospheric_river as ar

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants for AR object identification and filtering
MIN_AR_OBJECT_SIZE = 50  # Min pixels for AR object to be considered
MIN_LAND_PIXELS_FOR_PEAK = 10  # Min land pixels required for peak selection


def calculate_end_point(
    start_lat: float, start_lon: float, bearing: float, distance_km: float
) -> tuple[float, float]:
    """Calculate the end point (latitude, longitude) given a starting point,
    bearing, and distance.

    Args:
        start_lat: Starting latitude in degrees.
        start_lon: Starting longitude in degrees.
        bearing: Bearing in degrees (0-360, where 0 is north, 90 is east).
        distance_km: Distance in kilometers.

    Returns:
        End point as (latitude, longitude) in degrees.
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1 = np.radians(start_lat)
    lon1 = np.radians(start_lon)
    bearing_rad = np.radians(bearing)

    # Calculate end point
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(distance_km / R)
        + np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing_rad)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_km / R) * np.cos(lat1),
        np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2),
    )

    # Convert back to degrees
    end_lat = np.degrees(lat2)
    end_lon = np.degrees(lon2)

    return end_lat, end_lon


def calculate_extent_bounds(
    left_lon: float,
    right_lon: float,
    bottom_lat: float,
    top_lat: float,
    extent_buffer: float = 250,
) -> regions.Region:
    """Calculate extent bounds with buffer for atmospheric river regions.

    Args:
        left_lon: Western longitude boundary.
        right_lon: Eastern longitude boundary.
        bottom_lat: Southern latitude boundary.
        top_lat: Northern latitude boundary.
        extent_buffer: Buffer distance in km or degrees.

    Returns:
        Bounding box region with buffer applied.
    """
    new_bottom_lat, _ = np.round(
        calculate_end_point(bottom_lat, left_lon, 180, extent_buffer), 1
    )
    new_top_lat, _ = np.round(
        calculate_end_point(top_lat, right_lon, 0, extent_buffer), 1
    )
    _, new_left_lon = np.round(
        calculate_end_point(bottom_lat, left_lon, 270, extent_buffer), 1
    )
    _, new_right_lon = np.round(
        calculate_end_point(bottom_lat, right_lon, 90, extent_buffer), 1
    )

    new_left_lon = np.round(utils.convert_longitude_to_360(new_left_lon), 1)
    new_right_lon = np.round(utils.convert_longitude_to_360(new_right_lon), 1)
    new_box = regions.BoundingBoxRegion(
        new_bottom_lat, new_top_lat, new_left_lon, new_right_lon
    )
    return new_box


def identify_ar_objects(
    ar_mask: xr.DataArray,
    min_area_gridpoints: int = 500,
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """Identify separate atmospheric river objects using connected components.

    Args:
        ar_mask: Binary mask where 1 indicates atmospheric river presence.
        min_area_gridpoints: Minimum area in grid points for valid AR object.

    Returns:
        Labeled array and dictionary of object properties.
    """
    # Convert to numpy array for processing
    mask_data = ar_mask.values.astype(bool)

    # Label connected components
    labeled_array, num_objects = ndimage.label(mask_data)

    # Analyze each object
    object_properties = {}

    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_array == obj_id

        # Calculate basic properties
        area = np.sum(obj_mask)

        if area < min_area_gridpoints:
            # Remove small objects
            labeled_array[obj_mask] = 0
            continue

        # Find object bounds
        obj_coords = np.where(obj_mask)
        lat_indices, lon_indices = obj_coords

        min_lat_idx, max_lat_idx = np.min(lat_indices), np.max(lat_indices)
        min_lon_idx, max_lon_idx = np.min(lon_indices), np.max(lon_indices)

        max_lat_idx_add = max_lat_idx + 1
        max_lon_idx_add = max_lon_idx + 1
        # Convert to actual coordinates
        lats = ar_mask.latitude.values[min_lat_idx:max_lat_idx_add]
        lons = ar_mask.longitude.values[min_lon_idx:max_lon_idx_add]

        lat_span = np.max(lats) - np.min(lats)
        lon_span = np.max(lons) - np.min(lons)

        # Calculate centroid
        centroid_lat_idx, centroid_lon_idx = ndimage.center_of_mass(obj_mask)
        centroid_lat = ar_mask.latitude.values[int(centroid_lat_idx)]
        centroid_lon = ar_mask.longitude.values[int(centroid_lon_idx)]

        object_properties[obj_id] = {
            "area": area,
            "lat_span": lat_span,
            "lon_span": lon_span,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "bounds": {
                "lat_min": np.min(lats),
                "lat_max": np.max(lats),
                "lon_min": np.min(lons),
                "lon_max": np.max(lons),
            },
        }

    return labeled_array, object_properties


def find_largest_ar_object(ar_mask: xr.DataArray, **object_kwargs) -> Optional[Dict]:
    """Find the largest valid atmospheric river object.

    Args:
        ar_mask: Binary mask where 1 indicates atmospheric river presence.
        **object_kwargs: Additional arguments for identify_ar_objects.

    Returns:
        Properties of the largest AR object, or None if no valid objects found.
    """
    labeled_array, object_properties = identify_ar_objects(ar_mask, **object_kwargs)

    if not object_properties:
        return None

    # Find the largest object by area
    largest_obj_id = max(
        object_properties.keys(), key=lambda x: object_properties[x]["area"]
    )

    largest_obj = object_properties[largest_obj_id].copy()
    largest_obj["object_id"] = largest_obj_id
    largest_obj["total_objects"] = len(object_properties)
    largest_obj["labeled_array"] = labeled_array

    return largest_obj


def find_central_ar_object(
    ar_slice: xr.DataArray,
    center_lat: float,
    center_lon: float,
    min_object_size: int = MIN_AR_OBJECT_SIZE,
) -> Optional[int]:
    """Find AR object label closest to map center.

    Args:
        ar_slice: AR mask slice for single timestep (2D).
        center_lat: Map center latitude for distance calculation.
        center_lon: Map center longitude for distance calculation.
        min_object_size: Minimum size in pixels for valid AR object.

    Returns:
        Label of closest AR object, or None if no valid objects found.
    """
    if ar_slice.sum() == 0:
        return None

    labeled_array, num_objects = ndimage.label(ar_slice.values > 0)

    if num_objects == 0:
        return None

    min_distance = float("inf")
    closest_object_label = None

    for obj_label in range(1, num_objects + 1):
        # Get coordinates of this AR object
        obj_coords = np.where(labeled_array == obj_label)
        lat_indices, lon_indices = obj_coords

        # Apply size filter first
        obj_size = len(lat_indices)
        if obj_size < min_object_size:
            continue  # Skip tiny objects

        # Convert to actual coordinates
        obj_lats = ar_slice.latitude.values[lat_indices]
        obj_lons = ar_slice.longitude.values[lon_indices]

        # Calculate centroid of this AR object
        obj_center_lat = np.mean(obj_lats)
        obj_center_lon = np.mean(obj_lons)

        # Calculate distance to map center
        lat_diff = obj_center_lat - center_lat
        lon_diff = obj_center_lon - center_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2)

        if distance < min_distance:
            min_distance = distance
            closest_object_label = obj_label

    return closest_object_label


def find_timestamp_peak_field(
    ivt_data: xr.DataArray,
    ar_mask: xr.DataArray,
    land_mask: Optional[xr.DataArray] = None,
) -> Tuple[int, float]:
    """Find timestamp with highest aggregate IVT within AR regions closest to center.

    Args:
        ivt_data: Integrated vapor transport data with time dimension.
        ar_mask: Binary AR mask with time dimension.
        land_mask: Land mask (1 for land, 0 for ocean). If None, uses all points.

    Returns:
        Time index of peak IVT and the peak IVT value (only from central AR regions).
    """
    time_dim = "valid_time" if "valid_time" in ivt_data.dims else "time"

    # Calculate map center coordinates
    center_lat = float(ar_mask.latitude.mean())
    center_lon = float(ar_mask.longitude.mean())

    aggregate_ivt_values = []

    for t in range(len(ivt_data[time_dim])):
        # Get IVT and AR mask at this timestep
        ivt_slice = ivt_data.isel({time_dim: t})
        ar_slice = ar_mask.isel({time_dim: t})

        # Find central AR object (closest to map center)
        closest_object_label = find_central_ar_object(ar_slice, center_lat, center_lon)

        if closest_object_label is not None:
            labeled_array, _ = ndimage.label(ar_slice.values > 0)
            closest_mask_array = labeled_array == closest_object_label
            central_ar_mask = xr.where(
                xr.DataArray(
                    closest_mask_array,
                    dims=ar_slice.dims,
                    coords=ar_slice.coords,
                ),
                1,
                0,
            )
        else:
            central_ar_mask = xr.zeros_like(ar_slice)

        # Only consider IVT values where central AR mask is active
        ivt_in_central_ar = ivt_slice.where(central_ar_mask > 0)
        aggregate_ivt = 0.0
        # Check if this central AR has significant land coverage
        if land_mask is not None:
            # Count AR pixels over land
            ar_over_land = central_ar_mask.where(land_mask > 0)
            land_ar_pixels = ar_over_land.sum().values

            # Only consider this timestamp if AR has reasonable land coverage
            if land_ar_pixels >= MIN_LAND_PIXELS_FOR_PEAK:
                # Calculate aggregate IVT only over land areas
                ivt_land_only = ivt_in_central_ar.where(land_mask > 0)
                aggregate_ivt = ivt_land_only.sum().values
        else:
            # No land mask provided, use all central AR pixels
            aggregate_ivt = ivt_in_central_ar.sum().values

        if not np.isnan(aggregate_ivt) and aggregate_ivt > 0:
            aggregate_ivt_values.append(float(aggregate_ivt))
        else:
            aggregate_ivt_values.append(0.0)

    # Find timestamp with maximum aggregate IVT (only from central AR regions)
    if all(val == 0 for val in aggregate_ivt_values):
        logger.warning("    Warning: No valid timestamps found with AR over land.")

        fallback_ivt_values = []

        for t in range(len(ivt_data[time_dim])):
            ivt_slice = ivt_data.isel({time_dim: t})
            ar_slice = ar_mask.isel({time_dim: t})

            # Find central AR object using helper function
            closest_obj = find_central_ar_object(ar_slice, center_lat, center_lon)

            if closest_obj is not None:
                labeled_array, _ = ndimage.label(ar_slice.values > 0)
                central_mask = labeled_array == closest_obj
                ivt_central = ivt_slice.where(
                    xr.DataArray(
                        central_mask, dims=ar_slice.dims, coords=ar_slice.coords
                    )
                    > 0
                )
                fallback_ivt = ivt_central.sum().values
                fallback_ivt_values.append(
                    float(fallback_ivt) if not np.isnan(fallback_ivt) else 0.0
                )
            else:
                fallback_ivt_values.append(0.0)

        if all(val == 0 for val in fallback_ivt_values):
            logger.warning("    No AR found at any timestamp - using first timestamp")
            peak_time_idx = 0
            peak_ivt_value = 0.0
        else:
            peak_time_idx = np.argmax(fallback_ivt_values)
            peak_ivt_value = fallback_ivt_values[peak_time_idx]
            logger.info(
                "    Selected fallback timestamp %s with total AR IVT=%.0f",
                peak_time_idx,
                peak_ivt_value,
            )
    else:
        peak_time_idx = np.argmax(aggregate_ivt_values)
        peak_ivt_value = aggregate_ivt_values[peak_time_idx]

    # Check if peak timestamp has valid land coverage
    if land_mask is not None and peak_ivt_value > 0:
        logger.info(
            "    Peak IVT value: %.0f kg/m/s at time %s", peak_ivt_value, peak_time_idx
        )
        # Check land coverage at peak time
        peak_ar_slice = ar_mask.isel({time_dim: peak_time_idx})
        closest_object_label = find_central_ar_object(
            peak_ar_slice, center_lat, center_lon
        )

        if closest_object_label is not None:
            labeled_array, _ = ndimage.label(peak_ar_slice.values > 0)
            peak_central_mask = labeled_array == closest_object_label
            peak_ar_over_land = peak_central_mask & (land_mask.values > 0)
            land_pixels = np.sum(peak_ar_over_land)
            total_pixels = np.sum(peak_central_mask)
            land_pct = 100 * land_pixels / total_pixels if total_pixels > 0 else 0
            logger.info(
                "    Peak time AR land coverage: %s/%s pixels (%.1f%%)",
                land_pixels,
                total_pixels,
                land_pct,
            )

    return peak_time_idx, peak_ivt_value


def create_composite_ar_mask(
    ar_mask: xr.DataArray,
    land_intersection: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """Create composite AR masks by taking maximum over time.

    Args:
        ar_mask: Binary AR mask with time dimension.
        land_intersection: Optional land intersection mask with time dim.

    Returns:
        Tuple of (composite_ar_mask, composite_land_intersection).
    """
    time_dim = "valid_time" if "valid_time" in ar_mask.dims else "time"

    # Create composite by taking max over time dimension
    composite_mask = ar_mask.max(dim=time_dim)

    composite_land = None
    if land_intersection is not None:
        composite_land = land_intersection.max(dim=time_dim)

    return composite_mask, composite_land


def expand_bounds_to_contiguous_ar(
    ar_mask: xr.DataArray,
    land_intersection: xr.DataArray,
    initial_bounds: Dict[str, float],
    object_id: int,
    labeled_array: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Expand bounds to include all contiguous AR over land.

    Args:
        ar_mask: Composite AR mask (2D, no time dimension).
        land_intersection: Composite land intersection (2D).
        initial_bounds: Initial bounds from largest object.
        object_id: Label ID of the main AR object.
        labeled_array: Labeled array identifying separate AR objects.

    Returns:
        Expanded (left_lon, right_lon, bottom_lat, top_lat).
    """
    lats = ar_mask.latitude.values
    lons = ar_mask.longitude.values

    # Start with initial bounds
    left_lon = initial_bounds["lon_min"]
    right_lon = initial_bounds["lon_max"]
    bottom_lat = initial_bounds["lat_min"]
    top_lat = initial_bounds["lat_max"]

    # Convert to indices
    lat_indices = np.where((lats >= bottom_lat) & (lats <= top_lat))[0]
    lon_indices = np.where((lons >= left_lon) & (lons <= right_lon))[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return left_lon, right_lon, bottom_lat, top_lat

    min_lat_idx = lat_indices[0]
    max_lat_idx = lat_indices[-1]
    min_lon_idx = lon_indices[0]
    max_lon_idx = lon_indices[-1]

    # Expand in each direction while AR-land is contiguous
    land_data = land_intersection.values

    # Expand East (increase longitude)
    for lon_idx in range(max_lon_idx + 1, len(lons)):
        # Check if this column has AR over land
        expanded = max_lat_idx + 1
        column = land_data[min_lat_idx:expanded, lon_idx]
        if column.sum() > 0:
            # Check if contiguous (labeled same as main object)
            column_labels = labeled_array[min_lat_idx:expanded, lon_idx]
            if np.any(column_labels == object_id):
                max_lon_idx = lon_idx
            else:
                break
        else:
            break

    # Expand West (decrease longitude)
    for lon_idx in range(min_lon_idx - 1, -1, -1):
        expanded = max_lat_idx + 1
        column = land_data[min_lat_idx:expanded, lon_idx]
        if column.sum() > 0:
            column_labels = labeled_array[min_lat_idx:expanded, lon_idx]
            if np.any(column_labels == object_id):
                min_lon_idx = lon_idx
            else:
                break
        else:
            break

    # Expand North (increase latitude index, but lat decreases)
    for lat_idx in range(max_lat_idx + 1, len(lats)):
        expanded = max_lon_idx + 1
        row = land_data[lat_idx, min_lon_idx:expanded]
        if row.sum() > 0:
            row_labels = labeled_array[lat_idx, min_lon_idx:expanded]
            if np.any(row_labels == object_id):
                max_lat_idx = lat_idx
            else:
                break
        else:
            break

    # Expand South (decrease latitude index, but lat increases)
    for lat_idx in range(min_lat_idx - 1, -1, -1):
        expanded = max_lon_idx + 1
        row = land_data[lat_idx, min_lon_idx:expanded]
        if row.sum() > 0:
            row_labels = labeled_array[lat_idx, min_lon_idx:expanded]
            if np.any(row_labels == object_id):
                min_lat_idx = lat_idx
            else:
                break
        else:
            break

    # Convert back to coordinates
    left_lon = lons[min_lon_idx]
    right_lon = lons[max_lon_idx]

    # Latitude arrays can be ordered N->S or S->N, ensure correct order
    lat_values = [lats[min_lat_idx], lats[max_lat_idx]]
    bottom_lat = min(lat_values)
    top_lat = max(lat_values)

    return left_lon, right_lon, bottom_lat, top_lat


def find_ar_bounds_from_largest_object(
    ar_mask: xr.DataArray,
    min_area_gridpoints: float = 500,
    land_intersection: Optional[xr.DataArray] = None,
    expand_to_contiguous: bool = True,
) -> Tuple[float, float, float, float, Optional[Dict]]:
    """Find geographical bounds of largest AR object.

    Args:
        ar_mask: Binary mask where 1 indicates AR presence.
        min_area_gridpoints: Minimum area in grid points.
        land_intersection: Optional land intersection for expansion.
        expand_to_contiguous: If True, expand to capture contiguous AR.

    Returns:
        left_lon, right_lon, bottom_lat, top_lat bounds & metadata.
    """
    largest_obj = find_largest_ar_object(
        ar_mask,
        min_area_gridpoints=min_area_gridpoints,
    )

    if largest_obj is None:
        # No valid AR objects found
        return np.nan, np.nan, np.nan, np.nan, None

    bounds = largest_obj["bounds"]
    left_lon = bounds["lon_min"]
    right_lon = bounds["lon_max"]
    bottom_lat = bounds["lat_min"]
    top_lat = bounds["lat_max"]

    # Expand bounds to capture contiguous AR over land
    if expand_to_contiguous and land_intersection is not None:
        left_lon, right_lon, bottom_lat, top_lat = expand_bounds_to_contiguous_ar(
            ar_mask,
            land_intersection,
            bounds,
            largest_obj["object_id"],
            largest_obj["labeled_array"],
        )

        # Update metadata with expanded bounds
        largest_obj["expanded_bounds"] = {
            "lat_min": bottom_lat,
            "lat_max": top_lat,
            "lon_min": left_lon,
            "lon_max": right_lon,
        }

    return (
        left_lon,
        right_lon,
        bottom_lat,
        top_lat,
        largest_obj,
    )


def create_case_summary_plot(
    case_id: int,
    title: str,
    ivt_data: xr.DataArray,
    composite_ar_mask: xr.DataArray,
    composite_land_intersection: xr.DataArray,
    peak_time_idx: int,
    peak_ivt_value: float,
    ar_bounds: Dict,
    buffered_bounds: regions.Region,
    largest_obj_metadata: Optional[Dict] = None,
    extent_modifier_degrees: float = 5,
) -> None:
    """Create summary plot showing composite AR approach and bounds.

    Args:
        case_id: Case ID number.
        title: Event title.
        ivt_data: Integrated vapor transport data.
        composite_ar_mask: Composite AR mask (max over time).
        composite_land_intersection: Composite land intersection.
        peak_time_idx: Time index of peak IVT.
        peak_ivt_value: Peak IVT value.
        ar_bounds: AR bounds dictionary (expanded bounds).
        buffered_bounds: Buffered bounds region.
        largest_obj_metadata: Metadata about largest object.
    """
    # Get the time dimension name
    time_dim = "valid_time" if "valid_time" in ivt_data.dims else "time"

    # Extract data at peak time
    ivt_peak = ivt_data.isel({time_dim: peak_time_idx})
    peak_time = ivt_data[time_dim].isel({time_dim: peak_time_idx}).values

    # Create figure with cartopy subplots
    fig = plt.figure(figsize=(20, 8))

    # Plot 1: IVT at peak time
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax1.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax1.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Plot IVT with colormap
    ivt_plot = ivt_peak.plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        add_colorbar=False,
        vmin=0,
        vmax=1000,
    )
    plt.colorbar(ivt_plot, ax=ax1, label="IVT (kg/m/s)", shrink=0.8)

    ax1.set_title(
        f"IVT at Peak Time\n{pd.to_datetime(peak_time).strftime('%Y-%m-%d %H:%M')}"
    )
    ax1.set_extent(
        [
            float(ivt_peak.longitude.min()) - extent_modifier_degrees,
            float(ivt_peak.longitude.max()) + extent_modifier_degrees,
            float(ivt_peak.latitude.min()) - extent_modifier_degrees,
            float(ivt_peak.latitude.max()) + extent_modifier_degrees,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Plot 2: Composite AR mask (max over time)
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax2.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax2.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Plot composite AR mask
    composite_ar_mask.plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        add_colorbar=False,
        vmin=0,
        vmax=1,
    )

    ax2.set_title("Composite AR Mask\n(Max over time)")
    ax2.set_extent(
        [
            float(composite_ar_mask.longitude.min()) - extent_modifier_degrees,
            float(composite_ar_mask.longitude.max()) + extent_modifier_degrees,
            float(composite_ar_mask.latitude.min()) - extent_modifier_degrees,
            float(composite_ar_mask.latitude.max()) + extent_modifier_degrees,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Plot 3: Composite land intersection with bounds overlay
    ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax3.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax3.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Plot land intersection
    composite_land_intersection.plot(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap="Oranges",
        add_colorbar=False,
        vmin=0,
        vmax=1,
    )

    # Plot initial bounds (if available, dashed blue)
    if largest_obj_metadata and "bounds" in largest_obj_metadata:
        init_bounds = largest_obj_metadata["bounds"]
        init_width = init_bounds["lon_max"] - init_bounds["lon_min"]
        init_height = init_bounds["lat_max"] - init_bounds["lat_min"]

        init_rect = Rectangle(
            (init_bounds["lon_min"], init_bounds["lat_min"]),
            init_width,
            init_height,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle=":",
            alpha=0.7,
            transform=ccrs.PlateCarree(),
        )
        ax3.add_patch(init_rect)

    # Plot expanded AR bounds (solid blue)
    ar_width = ar_bounds["longitude_max"] - ar_bounds["longitude_min"]
    ar_height = ar_bounds["latitude_max"] - ar_bounds["latitude_min"]

    ar_rect = Rectangle(
        (ar_bounds["longitude_min"], ar_bounds["latitude_min"]),
        ar_width,
        ar_height,
        linewidth=3,
        edgecolor="blue",
        facecolor="blue",
        alpha=0.3,
        transform=ccrs.PlateCarree(),
    )
    ax3.add_patch(ar_rect)
    buffered_bounds_edges = buffered_bounds.as_geopandas().total_bounds
    # Plot buffered bounds (dashed green)
    buff_width = buffered_bounds_edges[2] - buffered_bounds_edges[0]
    buff_height = buffered_bounds_edges[3] - buffered_bounds_edges[1]

    buff_rect = Rectangle(
        (buffered_bounds_edges[0], buffered_bounds_edges[1]),
        buff_width,
        buff_height,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )
    ax3.add_patch(buff_rect)

    ax3.set_title("AR-Land Intersection\nwith Bounds")
    lon_min = float(composite_land_intersection.longitude.min())
    lon_max = float(composite_land_intersection.longitude.max())
    lat_min = float(composite_land_intersection.latitude.min())
    lat_max = float(composite_land_intersection.latitude.max())
    ax3.set_extent(
        [
            lon_min - extent_modifier_degrees,
            lon_max + extent_modifier_degrees,
            lat_min - extent_modifier_degrees,
            lat_max + extent_modifier_degrees,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add overall title with statistics
    stats_text = f"Peak IVT: {peak_ivt_value:.0f} kg/m/s"
    if largest_obj_metadata:
        stats_text += f" | Objects: {largest_obj_metadata.get('total_objects', 'N/A')}"
        stats_text += f" | Area: {largest_obj_metadata.get('area', 'N/A')} pts"

    fig.suptitle(f"Case {case_id}: {title}\n{stats_text}", fontsize=14, y=0.95)

    plt.tight_layout()

    # Save plot
    plot_filename = f"case_{case_id:03d}_summary.png"
    plt.savefig(plot_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)  # Close to save memory
    logger.info("    Saved summary plot: %s", plot_filename)


def process_ar_event(
    single_case: cases.IndividualCase,
    era5_ar: inputs.ERA5,
    AR_OBJECT_CONFIG: Dict,
    extent_modifier_degrees: float = 5,
) -> dict:
    """Process an atmospheric river event."""
    logger.info(
        "\nProcessing: %s (Case %s)", single_case.title, single_case.case_id_number
    )
    # Create a case object for this event
    case_list = cases.load_individual_cases([single_case])
    case = case_list[0]
    case.start_date = case.start_date - pd.Timedelta(days=3)
    case.end_date = case.end_date + pd.Timedelta(days=3)

    # Expand the case location by 10 degrees on all edges
    original_location = case.location.as_geopandas().total_bounds
    expanded_location = regions.BoundingBoxRegion(
        latitude_min=original_location[1] - extent_modifier_degrees,
        latitude_max=original_location[3] + extent_modifier_degrees,
        longitude_min=original_location[0] - extent_modifier_degrees,
        longitude_max=original_location[2] + extent_modifier_degrees,
    )
    case.location = expanded_location

    # Load ERA5 data for this case
    era5_data = era5_ar.open_and_maybe_preprocess_data_from_source()
    era5_data = era5_ar.maybe_map_variable_names(era5_data)
    era5_data = era5_data.sel(
        valid_time=era5_data.valid_time.dt.hour.isin([0, 6, 12, 18])
    )
    era5_data = inputs.maybe_subset_variables(era5_data, variables=era5_ar.variables)
    era5_subset = era5_ar.subset_data_to_case(era5_data, case)
    era5_subset = era5_subset.chunk()
    # Generate IVT
    logger.info("  Computing IVT...")
    ivt_da = ar.integrated_vapor_transport(
        specific_humidity=era5_subset["specific_humidity"],
        eastward_wind=era5_subset["eastward_wind"],
        northward_wind=era5_subset["northward_wind"],
        levels=era5_subset["adjusted_level"],
    )
    ivt_da.name = "integrated_vapor_transport"
    # Compute IVT Laplacian
    ivt_laplacian = ar.integrated_vapor_transport_laplacian(ivt_da)
    ivt_laplacian.name = "integrated_vapor_transport_laplacian"

    # Compute AR mask
    ar_mask = ar.atmospheric_river_mask(
        ivt=ivt_da,
        ivt_laplacian=ivt_laplacian,
        min_size_gridpoints=AR_OBJECT_CONFIG["min_area_gridpoints"],
    )

    # Generate land mask for peak time finding
    logger.info("  Generating land mask...")
    if ar_mask.longitude.size > 0 and ar_mask.latitude.size > 0:
        mask_parent = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
            ar_mask.longitude, ar_mask.latitude
        )
        land_mask = mask_parent.where(np.isnan(mask_parent), 1).where(
            mask_parent == 0, 0
        )
        total_land_pixels = land_mask.sum().values
        if total_land_pixels > 0:
            total_pixels = land_mask.size
            land_percentage = 100 * total_land_pixels / total_pixels

            logger.info(
                "    Generated land mask: %s/%s land pixels (%.1f%%)",
                total_land_pixels,
                total_pixels,
                land_percentage,
            )

    else:
        logger.warning("    Warning: Could not generate land mask; AR mask empty")
        land_mask = None

    # Find timestamp with highest aggregate IVT (for plotting/reference)
    logger.info("  Finding peak IVT timestamp...")

    peak_time_idx, peak_ivt_value = find_timestamp_peak_field(
        ivt_da,
        ar_mask,
        land_mask=land_mask,
    )

    peak_time = ar_mask.valid_time.isel(valid_time=peak_time_idx).values
    logger.info(
        "  Peak IVT at time index %s: %.0f kg/m/s",
        peak_time_idx,
        peak_ivt_value,
    )
    logger.info("  Peak time: %s", pd.to_datetime(peak_time).strftime("%Y-%m-%d %H:%M"))

    # Create composite AR mask over entire time range (max)
    logger.info("  Creating composite AR mask over time...")
    composite_ar_mask, composite_land_intersection = create_composite_ar_mask(
        ar_mask, land_intersection=land_mask
    )

    logger.info(
        "  Composite AR mask has %s grid points", composite_ar_mask.sum().values
    )
    logger.info(
        "  Composite land intersection has %s grid points",
        composite_land_intersection.sum().values
        if composite_land_intersection is not None
        else 0,
    )

    # Find bounds using composite mask & expand to contiguous AR
    left_lon, right_lon, bottom_lat, top_lat, largest_obj_metadata = (
        find_ar_bounds_from_largest_object(
            composite_ar_mask,
            AR_OBJECT_CONFIG["min_area_gridpoints"],
            land_intersection=composite_land_intersection,
            expand_to_contiguous=True,
        )
    )

    if np.isnan(left_lon):
        logger.warning(
            "  Warning: No valid AR objects detected for %s", single_case.title
        )
        logger.info("  Composite AR pixels: %s", composite_ar_mask.sum().values)
        logger.info(
            "  Object filtering criteria: min_area_gridpoints=%s",
            AR_OBJECT_CONFIG["min_area_gridpoints"],
        )
        return {
            "case_id": single_case.case_id_number,
            "title": single_case.title,
            "start_date": single_case.start_date,
            "end_date": single_case.end_date,
            "original_bounds": single_case.location,
        }

    logger.info(
        "  Expanded AR bounds: %.1f-%.1f°E, %.1f-%.1f°N",
        left_lon,
        right_lon,
        bottom_lat,
        top_lat,
    )

    if largest_obj_metadata:
        area = largest_obj_metadata["area"]
        total = largest_obj_metadata["total_objects"]
        logger.info(
            "  Object properties: area=%s gridpoints, total_objects=%s",
            area,
            total,
        )

    # Calculate bounds with 250km buffer (using largest object bounds)
    bounds_with_buffer = calculate_extent_bounds(
        left_lon, right_lon, bottom_lat, top_lat, extent_buffer=250
    )
    bounds_with_buffer_edges = bounds_with_buffer.as_geopandas().total_bounds
    lon_min = bounds_with_buffer_edges[0]
    lon_max = bounds_with_buffer_edges[2]
    lat_min = bounds_with_buffer_edges[1]
    lat_max = bounds_with_buffer_edges[3]
    logger.info(
        "  Buffered bounds: %.1f-%.1f°, %.1f-%.1f°",
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )

    # Create summary plot for this case
    logger.info("  Creating summary plot...")
    ar_bounds_dict = {
        "latitude_min": bottom_lat,
        "latitude_max": top_lat,
        "longitude_min": left_lon,
        "longitude_max": right_lon,
    }

    create_case_summary_plot(
        case_id=single_case.case_id_number,
        title=single_case.title,
        ivt_data=ivt_da,
        composite_ar_mask=composite_ar_mask,
        composite_land_intersection=composite_land_intersection,
        peak_time_idx=peak_time_idx,
        peak_ivt_value=peak_ivt_value,
        ar_bounds=ar_bounds_dict,
        buffered_bounds=bounds_with_buffer,
        largest_obj_metadata=largest_obj_metadata,
    )
    return {
        "case_id": single_case.case_id_number,
        "title": single_case.title,
        "start_date": single_case.start_date,
        "end_date": single_case.end_date,
        "original_bounds": single_case.location,
        "ar_largest_object_bounds": {
            "latitude_min": bottom_lat,
            "latitude_max": top_lat,
            "longitude_min": left_lon,
            "longitude_max": right_lon,
        },
        "buffered_bounds": {
            "latitude_min": bounds_with_buffer_edges[1],
            "latitude_max": bounds_with_buffer_edges[3],
            "longitude_min": bounds_with_buffer_edges[0],
            "longitude_max": bounds_with_buffer_edges[2],
        },
        "bounds_region": bounds_with_buffer,
        "largest_object_metadata": largest_obj_metadata,
        "peak_time_idx": peak_time_idx,
        "peak_ivt_value": peak_ivt_value,
        "peak_timestamp": ar_mask.valid_time.isel(valid_time=peak_time_idx).values,
        "ar_config": AR_OBJECT_CONFIG,
    }


def main():
    """Main execution function for AR bounds processing."""
    client = Client()

    # In case the client progress is useful to view
    logger.info(client)
    logger.info(client.dashboard_link)

    # Setup ERA5 data source for atmospheric river detection
    era5_ar = inputs.ERA5(variables=[derived.AtmosphericRiverVariables])
    parallel = True

    # Load atmospheric river events from the events.yaml file
    events_yaml = cases.load_ewb_events_yaml_into_case_list()
    ar_events = [n for n in events_yaml if n.event_type == "atmospheric_river"]
    logger.info("Found %s atmospheric river events in events.yaml", len(ar_events))

    # Process each atmospheric river event with enhanced object-based bounds calculation
    ar_bounds_results_enhanced = []

    # Configuration for AR object filtering
    AR_OBJECT_CONFIG = {
        "min_area_gridpoints": 500,  # Minimum size in grid points
        # Removed shape constraints (aspect ratio and circularity) to allow more AR
        # shapes
    }
    if parallel:
        with joblib.parallel_backend("dask"):
            ar_bounds_results_enhanced = joblib.Parallel(n_jobs=len(ar_events))(
                joblib.delayed(process_ar_event)(single_case, era5_ar, AR_OBJECT_CONFIG)
                for single_case in ar_events
            )
    else:
        # Run in serial using a list comprehension
        ar_bounds_results_enhanced = [
            process_ar_event(single_case, era5_ar, AR_OBJECT_CONFIG)
            for single_case in ar_events
        ]
    logger.info(
        "\nSuccessfully processed %s events",
        len(ar_bounds_results_enhanced),
    )

    # Save the enhanced atmospheric river bounds results; "bounded_region" is the final
    # region that is used for the events coordinates
    pickle_file_path = "ar_bounds_results_enhanced.pkl"
    with open(pickle_file_path, "wb") as f:
        pickle.dump(ar_bounds_results_enhanced, f)


if __name__ == "__main__":
    main()
