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
import xarray as xr
from dask.distributed import Client
from matplotlib.patches import Rectangle
from scipy.ndimage import center_of_mass, label

from extremeweatherbench import cases, derived, inputs, regions, utils
from extremeweatherbench.events import atmospheric_river as ar

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    labeled_array, num_objects = label(mask_data)

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
        centroid_lat_idx, centroid_lon_idx = center_of_mass(obj_mask)
        centroid_lat = ar_mask.latitude.values[int(centroid_lat_idx)]
        centroid_lon = ar_mask.longitude.values[int(centroid_lon_idx)]

        # Only apply size filter - no shape constraints
        if area < min_area_gridpoints:
            # Remove objects that are too small
            labeled_array[obj_mask] = 0
            continue

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


def find_peak_ivt_timestamp(
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

        # Find AR objects and filter to the one closest to center
        if ar_slice.sum() > 0:  # Only process if there are AR pixels
            labeled_array, num_objects = label(ar_slice.values > 0)

            if num_objects > 0:
                # Find which AR object is closest to map center
                # Apply minimum size filter first to avoid selecting tiny fragments
                min_distance = float("inf")
                closest_object_label = None
                min_object_size = 50  # Minimum 50 pixels to be considered

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

                # Create mask for only the closest AR object
                if closest_object_label is not None:
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
            else:
                central_ar_mask = xr.zeros_like(ar_slice)
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
            central_pixels = central_ar_mask.sum().values
            land_pct = (
                100 * land_ar_pixels / central_pixels if central_pixels > 0 else 0
            )

            # Only consider this timestamp if AR has reasonable land coverage
            # (at least 10 pixels over land)
            if land_ar_pixels >= 10:
                # Calculate aggregate IVT only over land areas
                ivt_land_only = ivt_in_central_ar.where(land_mask > 0)
                aggregate_ivt = ivt_land_only.sum().values
        else:
            # No land mask provided, use all central AR pixels
            aggregate_ivt = ivt_in_central_ar.sum().values
            central_pixels = central_ar_mask.sum().values

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

            if ar_slice.sum() > 0:
                labeled_array, num_objects = label(ar_slice.values > 0)
                if num_objects > 0:
                    # Find central AR again
                    min_distance = float("inf")
                    closest_obj = None
                    center_lat = float(ar_mask.latitude.mean())
                    center_lon = float(ar_mask.longitude.mean())

                    for obj_label in range(1, num_objects + 1):
                        obj_coords = np.where(labeled_array == obj_label)
                        lat_indices, lon_indices = obj_coords
                        obj_lats = ar_slice.latitude.values[lat_indices]
                        obj_lons = ar_slice.longitude.values[lon_indices]
                        obj_center_lat = np.mean(obj_lats)
                        obj_center_lon = np.mean(obj_lons)
                        lat_diff = obj_center_lat - center_lat
                        lon_diff = obj_center_lon - center_lon
                        distance = np.sqrt(lat_diff**2 + lon_diff**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_obj = obj_label

                    if closest_obj is not None:
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
                else:
                    fallback_ivt_values.append(0.0)
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
        if peak_ar_slice.sum() > 0:
            labeled_array, _ = label(peak_ar_slice.values > 0)
            # Find central AR again for verification
            center_lat = float(ar_mask.latitude.mean())
            center_lon = float(ar_mask.longitude.mean())
            min_distance = float("inf")
            closest_object_label = None

            for obj_label in range(1, labeled_array.max() + 1):
                obj_coords = np.where(labeled_array == obj_label)
                lat_indices, lon_indices = obj_coords
                obj_lats = peak_ar_slice.latitude.values[lat_indices]
                obj_lons = peak_ar_slice.longitude.values[lon_indices]
                obj_center_lat = np.mean(obj_lats)
                obj_center_lon = np.mean(obj_lons)
                lat_diff = obj_center_lat - center_lat
                lon_diff = obj_center_lon - center_lon
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_object_label = obj_label

            if closest_object_label is not None:
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


def find_ar_bounds_from_largest_object(
    ar_mask: xr.DataArray,
    min_area_gridpoints: float = 500,
) -> Tuple[float, float, float, float, Optional[Dict]]:
    """Find the geographical bounds of the largest atmospheric river object.

    Args:
        ar_mask: Binary mask where 1 indicates atmospheric river presence.
        min_area_gridpoints: Minimum area in grid points for valid AR object.

    Returns:
        left_lon, right_lon, bottom_lat, top_lat bounds and object metadata.
    """
    largest_obj = find_largest_ar_object(
        ar_mask,
        min_area_gridpoints=min_area_gridpoints,
    )

    if largest_obj is None:
        # No valid AR objects found
        return np.nan, np.nan, np.nan, np.nan, None

    bounds = largest_obj["bounds"]
    return (
        bounds["lon_min"],
        bounds["lon_max"],
        bounds["lat_min"],
        bounds["lat_max"],
        largest_obj,
    )


def create_case_summary_plot(
    case_id: int,
    title: str,
    ivt_data: xr.DataArray,
    ar_mask: xr.DataArray,
    peak_time_idx: int,
    peak_ivt_value: float,
    ar_bounds: Dict,
    buffered_bounds: regions.Region,
    largest_obj_metadata: Optional[Dict] = None,
) -> None:
    """Create a summary plot for each AR case showing IVT, mask, and bounds.

    Args:
        case_id: Case ID number.
        title: Event title.
        ivt_data: Integrated vapor transport data.
        ar_mask: AR mask data.
        peak_time_idx: Time index of peak IVT.
        peak_ivt_value: Peak IVT value.
        ar_bounds: AR bounds dictionary.
        buffered_bounds: Buffered bounds region.
        largest_obj_metadata: Metadata about the largest object.
    """
    # Get the time dimension name
    time_dim = "valid_time" if "valid_time" in ivt_data.dims else "time"

    # Extract data at peak time
    ivt_peak = ivt_data.isel({time_dim: peak_time_idx})
    ar_peak = ar_mask.isel({time_dim: peak_time_idx})
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
            float(ivt_peak.longitude.min()) - 5,
            float(ivt_peak.longitude.max()) + 5,
            float(ivt_peak.latitude.min()) - 5,
            float(ivt_peak.latitude.max()) + 5,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Plot 2: AR mask at peak time
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax2.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax2.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Plot AR mask
    ar_peak.plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        add_colorbar=False,
        vmin=0,
        vmax=1,
    )

    ax2.set_title("AR Mask at Peak Time")
    ax2.set_extent(
        [
            float(ar_peak.longitude.min()) - 5,
            float(ar_peak.longitude.max()) + 5,
            float(ar_peak.latitude.min()) - 5,
            float(ar_peak.latitude.max()) + 5,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Plot AR mask as background
    ar_peak.plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        alpha=0.3,
        add_colorbar=False,
        vmin=0,
        vmax=1,
    )

    # Plot AR bounds (solid blue)
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
    ax2.add_patch(ar_rect)

    # Plot buffered bounds (dashed green)
    buff_width = buffered_bounds.longitude_max - buffered_bounds.longitude_min
    buff_height = buffered_bounds.latitude_max - buffered_bounds.latitude_min

    buff_rect = Rectangle(
        (buffered_bounds.longitude_min, buffered_bounds.latitude_min),
        buff_width,
        buff_height,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )
    ax2.add_patch(buff_rect)
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
    single_case: cases.IndividualCase, era5_ar: inputs.ERA5, AR_OBJECT_CONFIG: Dict
) -> dict:
    """Process an atmospheric river event."""
    logger.info(
        "\nProcessing: %s (Case %s)", single_case.title, single_case.case_id_number
    )
    # Create a case object for this event
    case_collection = cases.load_individual_cases({"cases": [single_case]})
    case = case_collection.cases[0]

    # Expand the case location by 10 degrees on all edges
    original_location = case.location
    expanded_location = regions.BoundingBoxRegion(
        latitude_min=original_location.latitude_min - 10,
        latitude_max=original_location.latitude_max + 10,
        longitude_min=original_location.longitude_min - 10,
        longitude_max=original_location.longitude_max + 10,
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
    # Compute IVT first
    logger.info("  Computing IVT...")
    ivt_da = ar.compute_ivt(era5_subset)
    ivt_da.name = "integrated_vapor_transport"
    # Compute IVT Laplacian
    ivt_laplacian = ar.compute_ivt_laplacian(ivt_da)
    ivt_laplacian.name = "integrated_vapor_transport_laplacian"

    # Merge all data
    full_data = xr.merge([era5_subset, ivt_da, ivt_laplacian])

    # Compute AR mask
    ar_mask = ar.atmospheric_river_mask(full_data)

    logger.info("  AR mask shape: %s", ar_mask.shape)
    logger.info("  Total AR grid points across all time: %s", ar_mask.sum().values)

    # Generate land mask using the same approach as find_land_intersection
    logger.info("  Generating land mask...")
    try:
        # Use the same approach as find_land_intersection but just get the land
        # mask
        mask_parent = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
            ar_mask.longitude, ar_mask.latitude
        )
        land_mask = mask_parent.where(np.isnan(mask_parent), 1).where(
            mask_parent == 0, 0
        )

        total_land_pixels = land_mask.sum().values
        total_pixels = land_mask.size
        land_percentage = 100 * total_land_pixels / total_pixels

        logger.info(
            "    Generated land mask: %s/%s land pixels (%.1f%%)",
            total_land_pixels,
            total_pixels,
            land_percentage,
        )

    except Exception as e:
        logger.warning("    Warning: Could not generate land mask: %s", e)
        land_mask = None

    # Method 1: Find timestamp with highest aggregate IVT over land
    logger.info("  Finding peak IVT timestamp...")

    peak_time_idx, peak_ivt_value = find_peak_ivt_timestamp(
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

    # Use AR mask at peak time for bounds calculation
    ar_mask_at_peak = ar_mask.isel(valid_time=peak_time_idx)

    # Extract minimum gridpoints parameter
    min_gridpoints = AR_OBJECT_CONFIG["min_area_gridpoints"]

    left_lon, right_lon, bottom_lat, top_lat, largest_obj_metadata = (
        find_ar_bounds_from_largest_object(ar_mask_at_peak, min_gridpoints)
    )

    if np.isnan(left_lon):
        logger.warning(
            "  Warning: No valid AR objects detected for %s", single_case.title
        )
        logger.info("  AR pixels at peak timestamp: %s", ar_mask_at_peak.sum().values)
        logger.info(
            "  Object filtering criteria: min_area_gridpoints=%s",
            min_gridpoints,
        )
        return {
            "case_id": single_case.case_id_number,
            "title": single_case.title,
            "start_date": single_case.start_date,
            "end_date": single_case.end_date,
            "original_bounds": single_case.location,
        }

    logger.info(
        "  Largest object bounds: %.1f-%.1f°, %.1f-%.1f°",
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

    lon_min = bounds_with_buffer.longitude_min
    lon_max = bounds_with_buffer.longitude_max
    lat_min = bounds_with_buffer.latitude_min
    lat_max = bounds_with_buffer.latitude_max
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
        ar_mask=ar_mask,
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
            "latitude_min": bounds_with_buffer.latitude_min,
            "latitude_max": bounds_with_buffer.latitude_max,
            "longitude_min": bounds_with_buffer.longitude_min,
            "longitude_max": bounds_with_buffer.longitude_max,
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
    # Setup ERA5 data source for atmospheric river detection
    client = Client()
    print(client)
    print(client.dashboard_link)
    era5_ar = inputs.ERA5(variables=[derived.AtmosphericRiverMask])
    parallel = True
    # Load atmospheric river events from the events.yaml file
    events_yaml = cases.load_ewb_events_yaml_into_case_collection()
    ar_events = events_yaml.select_cases(by="event_type", value="atmospheric_river")
    logger.info("Found %s atmospheric river events in events.yaml", len(ar_events))

    # Process each atmospheric river event with enhanced object-based bounds calculation
    ar_bounds_results_enhanced = []

    # Configuration for AR object filtering
    AR_OBJECT_CONFIG = {
        "min_area_gridpoints": 300,  # Minimum size in grid points
        # Removed shape constraints (aspect ratio and circularity) to allow more AR
        # shapes
        "consistency_weight": 0.3,  # Weight for temporal consistency vs size
    }
    if parallel:
        with joblib.parallel_backend("dask"):
            ar_bounds_results_enhanced = joblib.Parallel(n_jobs=len(ar_events))(
                joblib.delayed(process_ar_event)(single_case, era5_ar, AR_OBJECT_CONFIG)
                for single_case in ar_events
            )
    else:
        ar_bounds_results_enhanced = [
            process_ar_event(single_case, era5_ar, AR_OBJECT_CONFIG)
            for single_case in ar_events
        ]

    logger.info(
        "\nSuccessfully processed %s events",
        len(ar_bounds_results_enhanced),
    )

    # Save the enhanced atmospheric river bounds results
    if ar_bounds_results_enhanced:
        pickle_file_path = "ar_bounds_results_enhanced.pkl"
        with open(pickle_file_path, "wb") as f:
            pickle.dump(ar_bounds_results_enhanced, f)


if __name__ == "__main__":
    main()
