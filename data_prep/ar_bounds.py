# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pickle
from typing import Dict, Literal, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.ndimage import center_of_mass, label
from tqdm.auto import tqdm

# %%
from extremeweatherbench import cases, derived, inputs, regions, utils

# %% [markdown]
# Enhanced functions for calculating bounds from AR mask data with object separation

# %%


def calculate_end_point(
    start_lat: float, start_lon: float, bearing: float, distance_km: float
) -> tuple[float, float]:
    """Calculate the end point (latitude, longitude) given a starting point,
    bearing, and distance.

    Parameters:
    -----------
    start_lat : float
        Starting latitude in degrees
    start_lon : float
        Starting longitude in degrees
    bearing : float
        Bearing in degrees (0-360, where 0 is north, 90 is east)
    distance_km : float
        Distance in kilometers

    Returns:
    --------
    tuple[float, float]
        End point as (latitude, longitude) in degrees
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
    extent_units: Literal["degrees", "km"] = "km",
) -> regions.Region:
    """Calculate extent bounds with buffer for atmospheric river regions.

    Parameters:
    -----------
    left_lon : float
        Western longitude boundary
    right_lon : float
        Eastern longitude boundary
    bottom_lat : float
        Southern latitude boundary
    top_lat : float
        Northern latitude boundary
    extent_buffer : float
        Buffer distance in km or degrees
    extent_units : Literal["degrees", "km"]
        Units for the buffer

    Returns:
    --------
    regions.Region
        Bounding box region with buffer applied
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

    Parameters:
    -----------
    ar_mask : xr.DataArray
        Binary mask where 1 indicates atmospheric river presence
    min_area_gridpoints : int
        Minimum area in grid points for valid AR object

    Returns:
    --------
    Tuple[np.ndarray, Dict[int, Dict]]
        Labeled array and dictionary of object properties
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

        # Convert to actual coordinates
        lats = ar_mask.latitude.values[min_lat_idx : (max_lat_idx + 1)]
        lons = ar_mask.longitude.values[min_lon_idx : (max_lon_idx + 1)]

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

    Parameters:
    -----------
    ar_mask : xr.DataArray
        Binary mask where 1 indicates atmospheric river presence
    **object_kwargs : dict
        Additional arguments for identify_ar_objects

    Returns:
    --------
    Optional[Dict]
        Properties of the largest AR object, or None if no valid objects found
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
    debug_output: bool = False,
) -> Tuple[int, float]:
    """Find timestamp with highest aggregate IVT within AR regions closest to center.

    Parameters:
    -----------
    ivt_data : xr.DataArray
        Integrated vapor transport data with time dimension
    ar_mask : xr.DataArray
        Binary AR mask with time dimension
    land_mask : Optional[xr.DataArray]
        Land mask (1 for land, 0 for ocean). If None, uses all points.

    Returns:
    --------
    Tuple[int, float]
        Time index of peak IVT and the peak IVT value (only from central AR regions)
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

        # Debug output for each timestamp (only when enabled)
        if debug_output:
            time_val = ivt_data[time_dim].isel({time_dim: t}).values
            time_str = pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")
            total_ar_pixels = ar_slice.sum().values
            print(
                f"    Time {t:2d} ({time_str}): {total_ar_pixels:4.0f} total AR pixels"
            )

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

        # Check if this central AR has significant land coverage
        if land_mask is not None:
            # Count AR pixels over land
            ar_over_land = central_ar_mask.where(land_mask > 0)
            land_ar_pixels = ar_over_land.sum().values
            central_pixels = central_ar_mask.sum().values
            land_pct = (
                100 * land_ar_pixels / central_pixels if central_pixels > 0 else 0
            )

            if debug_output:
                print(
                    f"      Central AR: {central_pixels:4.0f} pixels, "
                    f"{land_ar_pixels:3.0f} over land ({land_pct:4.1f}%)"
                )

            # Only consider this timestamp if AR has reasonable land coverage
            # (at least 10 pixels over land)
            if land_ar_pixels >= 10:
                # Calculate aggregate IVT only over land areas
                ivt_land_only = ivt_in_central_ar.where(land_mask > 0)
                aggregate_ivt = ivt_land_only.sum().values
                if debug_output:
                    print(
                        "      ✓ Meets threshold: Land IVT = {aggregate_ivt:.0f} kg/m/s"
                    )
            else:
                # Skip this timestamp - no significant land coverage
                aggregate_ivt = 0.0
                if debug_output:
                    print("      ✗ Below threshold: Skipping this timestamp")
        else:
            # No land mask provided, use all central AR pixels
            aggregate_ivt = ivt_in_central_ar.sum().values
            central_pixels = central_ar_mask.sum().values
            if debug_output:
                print(
                    f"      Central AR: {central_pixels:4.0f} pixels, "
                    f"Total IVT = {aggregate_ivt:.0f} kg/m/s (no land mask)"
                )

        if not np.isnan(aggregate_ivt) and aggregate_ivt > 0:
            aggregate_ivt_values.append(float(aggregate_ivt))
        else:
            aggregate_ivt_values.append(0.0)

    # Find timestamp with maximum aggregate IVT (only from central AR regions)
    if all(val == 0 for val in aggregate_ivt_values):
        print("    Warning: No valid timestamps found with AR over land.")

        # Fallback strategy: find timestamp with highest total central AR IVT
        # (ignoring land mask requirement)
        print("    Fallback: Using timestamp with highest total central AR IVT")
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
            print("    No AR found at any timestamp - using first timestamp")
            peak_time_idx = 0
            peak_ivt_value = 0.0
        else:
            peak_time_idx = np.argmax(fallback_ivt_values)
            peak_ivt_value = fallback_ivt_values[peak_time_idx]
            print(
                f"    Selected fallback timestamp {peak_time_idx} "
                f"with total AR IVT={peak_ivt_value:.0f}"
            )
    else:
        peak_time_idx = np.argmax(aggregate_ivt_values)
        peak_ivt_value = aggregate_ivt_values[peak_time_idx]

    # Debug info: check if peak timestamp has valid land coverage
    if land_mask is not None and peak_ivt_value > 0:
        print(
            f"    Peak IVT value: {peak_ivt_value:.0f} kg/m/s at time {peak_time_idx}"
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
                print(
                    f"    Peak time AR land coverage: "
                    f"{land_pixels}/{total_pixels} pixels ({land_pct:.1f}%)"
                )

    return peak_time_idx, peak_ivt_value


def find_ar_bounds_from_largest_object(
    ar_mask: xr.DataArray,
    min_area_gridpoints: float = 500,
) -> Tuple[float, float, float, float, Optional[Dict]]:
    """Find the geographical bounds of the largest atmospheric river object.

    Parameters:
    -----------
    ar_mask : xr.DataArray
        Binary mask where 1 indicates atmospheric river presence
    min_area_gridpoints : int
        Minimum area in grid points for valid AR object

    Returns:
    --------
    Tuple[float, float, float, float, Optional[Dict]]
        left_lon, right_lon, bottom_lat, top_lat bounds and object metadata
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

    Parameters:
    -----------
    case_id : int
        Case ID number
    title : str
        Event title
    ivt_data : xr.DataArray
        Integrated vapor transport data
    ar_mask : xr.DataArray
        AR mask data
    peak_time_idx : int
        Time index of peak IVT
    peak_ivt_value : float
        Peak IVT value
    ar_bounds : Dict
        AR bounds dictionary
    buffered_bounds : regions.Region
        Buffered bounds region
    largest_obj_metadata : Optional[Dict]
        Metadata about the largest object
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

    # Plot 3: Bounds comparison
    ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax3.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax3.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Plot AR mask as background
    ar_peak.plot(
        ax=ax3,
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
    ax3.add_patch(ar_rect)

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
    ax3.add_patch(buff_rect)

    ax3.set_title("AR Bounds\nBlue: AR Object, Green: +250km Buffer")
    ax3.set_extent(
        [
            min(ar_bounds["longitude_min"], buffered_bounds.longitude_min) - 5,
            max(ar_bounds["longitude_max"], buffered_bounds.longitude_max) + 5,
            min(ar_bounds["latitude_min"], buffered_bounds.latitude_min) - 5,
            max(ar_bounds["latitude_max"], buffered_bounds.latitude_max) + 5,
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
    plt.close()  # Close to save memory

    print(f"    Saved summary plot: {plot_filename}")


def find_dominant_ar_across_time(
    ar_mask_timeseries: xr.DataArray, consistency_weight: float = 0.3, **object_kwargs
) -> Tuple[float, float, float, float, Optional[Dict]]:
    """Find bounds based on the dominant AR object across the time period.

    This function attempts to maintain consistency by finding the AR object
    that appears most frequently and with the largest average size.

    Parameters:
    -----------
    ar_mask_timeseries : xr.DataArray
        AR mask with time dimension
    consistency_weight : float
        Weight for temporal consistency vs size (0=only size, 1=only consistency)
    **object_kwargs : dict
        Additional arguments for AR object identification

    Returns:
    --------
    Tuple[float, float, float, float, Optional[Dict]]
        Bounds and metadata for the dominant AR object
    """
    time_dim = "valid_time" if "valid_time" in ar_mask_timeseries.dims else "time"

    # Track AR objects across time
    time_objects = {}
    spatial_consistency: dict = {}

    for t, time_val in enumerate(ar_mask_timeseries[time_dim]):
        time_slice = ar_mask_timeseries.isel({time_dim: t})

        if time_slice.sum() == 0:
            continue

        labeled_array, object_properties = identify_ar_objects(
            time_slice, **object_kwargs
        )

        if not object_properties:
            continue

        time_objects[t] = object_properties

        # Track spatial consistency by centroid proximity
        for obj_id, props in object_properties.items():
            centroid = (props["centroid_lat"], props["centroid_lon"])

            # Find if this object is close to any previous objects
            matched = False
            for prev_centroid, consistency_data in spatial_consistency.items():
                # Calculate distance between centroids (simple Euclidean)
                dist = np.sqrt(
                    (centroid[0] - prev_centroid[0]) ** 2
                    + (centroid[1] - prev_centroid[1]) ** 2
                )

                # If within ~5 degrees, consider it the same AR system
                if dist < 5.0:
                    consistency_data["appearances"] += 1
                    consistency_data["total_area"] += props["area"]
                    consistency_data["time_steps"].append(t)
                    matched = True
                    break

            if not matched:
                spatial_consistency[centroid] = {
                    "appearances": 1,
                    "total_area": props["area"],
                    "time_steps": [t],
                    "properties": props,
                }

    if not spatial_consistency:
        return np.nan, np.nan, np.nan, np.nan, None

    # Score each AR system by combination of size and temporal consistency
    best_score = -1
    best_ar = None

    for centroid, consistency_data in spatial_consistency.items():
        avg_area = consistency_data["total_area"] / consistency_data["appearances"]
        consistency_score = consistency_data["appearances"] / len(
            ar_mask_timeseries[time_dim]
        )

        # Combine size and consistency
        score = (
            1 - consistency_weight
        ) * avg_area + consistency_weight * consistency_score * 10000

        if score > best_score:
            best_score = score
            best_ar = consistency_data

    if best_ar is None:
        return np.nan, np.nan, np.nan, np.nan, None

    # Use the bounds from the representative properties
    bounds = best_ar["properties"]["bounds"]
    metadata = best_ar["properties"].copy()
    metadata.update(
        {
            "appearances": best_ar["appearances"],
            "consistency_score": best_ar["appearances"]
            / len(ar_mask_timeseries[time_dim]),
            "time_steps": best_ar["time_steps"],
        }
    )

    return (
        bounds["lon_min"],
        bounds["lon_max"],
        bounds["lat_min"],
        bounds["lat_max"],
        metadata,
    )


# %% [markdown]
# Setup ERA5 data source for atmospheric river detection

# %%
ERA5_AR = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[derived.AtmosphericRiverMask],
    variable_mapping={
        "specific_humidity": "specific_humidity",
        "u_component_of_wind": "eastward_wind",
        "v_component_of_wind": "northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# %% [markdown]
# Load atmospheric river events from the events.yaml file

# %%
events_yaml = utils.load_events_yaml()
ar_events = [
    event
    for event in events_yaml["cases"]
    if event["event_type"] == "atmospheric_river"
]

print(f"Found {len(ar_events)} atmospheric river events in events.yaml")

# Show a few examples
for i, event in enumerate(ar_events[:3]):
    print(f"Event {i + 1}: {event['title']}")
    print(f"  Date: {event['start_date']} to {event['end_date']}")
    print(f"  Location: {event['location']['parameters']}")
    print()

# %% [markdown]
# Process each atmospheric river event with enhanced object-based bounds calculation

# %%
ar_bounds_results_enhanced = []

# Configuration for AR object filtering
AR_OBJECT_CONFIG = {
    "min_area_gridpoints": 300,  # Minimum size in grid points
    # Removed shape constraints (aspect ratio and circularity) to allow more AR shapes
    "consistency_weight": 0.3,  # Weight for temporal consistency vs size
}

for event in tqdm(ar_events):  # Process all atmospheric river events
    try:
        case_id = event.get("case_id_number", "unknown")
        is_debug_case = case_id == 104

        # Process all cases (debug mode removed)

        print(f"\nProcessing: {event['title']} (Case {case_id})")
        if is_debug_case:
            print("*** DEBUG MODE: Case 104 ***")

        # Create a case object for this event
        case_collection = cases.load_individual_cases({"cases": [event]})
        case = case_collection.cases[0]

        # Expand the case location by 20 degrees on all edges
        original_location = case.location
        expanded_location = regions.BoundingBoxRegion(
            latitude_min=original_location.latitude_min - 20,
            latitude_max=original_location.latitude_max + 20,
            longitude_min=original_location.longitude_min - 20,
            longitude_max=original_location.longitude_max + 20,
        )
        case.location = expanded_location

        # Load ERA5 data for this case
        era5_data = ERA5_AR.open_and_maybe_preprocess_data_from_source()
        era5_data = ERA5_AR.maybe_map_variable_names(era5_data)
        era5_data = era5_data.sel(
            valid_time=era5_data.valid_time.dt.hour.isin([0, 6, 12, 18])
        )
        era5_data = inputs.maybe_subset_variables(
            era5_data, variables=ERA5_AR.variables
        )
        era5_subset = ERA5_AR.subset_data_to_case(era5_data, case)

        # Import the atmospheric river module to access IVT computation directly
        from extremeweatherbench.events import atmospheric_river as ar

        # Compute IVT first
        print("  Computing IVT...")
        ivt_dataset = ar.compute_ivt(era5_subset)

        # Compute IVT Laplacian
        ivt_laplacian = ar.compute_ivt_laplacian(
            ivt_dataset["integrated_vapor_transport"]
        )

        # Merge all data
        full_data = xr.merge([era5_subset, ivt_dataset, ivt_laplacian])

        # Compute AR mask
        ar_mask = ar.ar_mask(full_data)

        # Get IVT data
        ivt_data = ivt_dataset["integrated_vapor_transport"]

        print(f"  AR mask shape: {ar_mask.shape}")
        print(f"  Total AR grid points across all time: {ar_mask.sum().values}")

        # Generate land mask using the same approach as find_land_intersection
        print("  Generating land mask...")
        try:
            import regionmask

            # Use the same approach as find_land_intersection but just get the land mask
            mask_parent = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
                ar_mask.longitude, ar_mask.latitude
            )
            land_mask = mask_parent.where(np.isnan(mask_parent), 1).where(
                mask_parent == 0, 0
            )

            total_land_pixels = land_mask.sum().values
            total_pixels = land_mask.size
            land_percentage = 100 * total_land_pixels / total_pixels

            print(
                f"    Generated land mask: {total_land_pixels}/{total_pixels} "
                f"land pixels ({land_percentage:.1f}%)"
            )

        except ImportError as e:
            print(f"    Warning: Required modules not available: {e}")
            land_mask = None

        except Exception as e:
            print(f"    Warning: Could not generate land mask: {e}")
            land_mask = None

        # Method 1: Find timestamp with highest aggregate IVT over land
        print("  Finding peak IVT timestamp...")
        if is_debug_case:
            print("  DEBUG: Calling find_peak_ivt_timestamp for case 104")
            print(f"  DEBUG: IVT data shape: {ivt_data.shape}")
            print(f"  DEBUG: AR mask shape: {ar_mask.shape}")
            print(f"  DEBUG: Land mask available: {land_mask is not None}")
            if land_mask is not None:
                print(f"  DEBUG: Land mask shape: {land_mask.shape}")
                print(f"  DEBUG: Land pixels: {land_mask.sum().values}")

        peak_time_idx, peak_ivt_value = find_peak_ivt_timestamp(
            ivt_data,
            ar_mask,
            land_mask=land_mask,
            debug_output=is_debug_case,
        )

        peak_time = ar_mask.valid_time.isel(valid_time=peak_time_idx).values
        print(f"  Peak IVT at time index {peak_time_idx}: {peak_ivt_value:.0f} kg/m/s")
        print(f"  Peak time: {pd.to_datetime(peak_time).strftime('%Y-%m-%d %H:%M')}")

        # Use AR mask at peak time for bounds calculation
        ar_mask_at_peak = ar_mask.isel(valid_time=peak_time_idx)

        # Extract only the parameters that apply to this function
        object_params = {
            k: v for k, v in AR_OBJECT_CONFIG.items() if k in ["min_area_gridpoints"]
        }
        min_gridpoints = object_params["min_area_gridpoints"]
        # Only show debug info for case 104 if needed
        if is_debug_case:
            print(f"  AR pixels at peak time: {ar_mask_at_peak.sum().values}")

        left_lon, right_lon, bottom_lat, top_lat, largest_obj_metadata = (
            find_ar_bounds_from_largest_object(ar_mask_at_peak, min_gridpoints)
        )

        if np.isnan(left_lon):
            print(f"  Warning: No valid AR objects detected for {event['title']}")
            print(f"  AR pixels at peak timestamp: {ar_mask_at_peak.sum().values}")
            print(f"  Object filtering criteria: {object_params}")

            # The default criteria are now relaxed, so no fallback needed

            continue

        print(
            f"  Largest object bounds: {left_lon:.1f}-{right_lon:.1f}°, "
            f"{bottom_lat:.1f}-{top_lat:.1f}°"
        )

        if largest_obj_metadata:
            area = largest_obj_metadata["area"]
            total = largest_obj_metadata["total_objects"]
            print(f"  Object properties: area={area} gridpoints, total_objects={total}")

        # Method 2: Find dominant AR across time (for comparison/future use)
        dom_left_lon, dom_right_lon, dom_bottom_lat, dom_top_lat, dominant_metadata = (
            find_dominant_ar_across_time(ar_mask, **AR_OBJECT_CONFIG)
        )

        dominant_bounds_info = {}
        if not np.isnan(dom_left_lon):
            dominant_bounds_info = {
                "latitude_min": dom_bottom_lat,
                "latitude_max": dom_top_lat,
                "longitude_min": dom_left_lon,
                "longitude_max": dom_right_lon,
            }
            if dominant_metadata:
                consistency = dominant_metadata.get("consistency_score", 0)
                appearances = dominant_metadata.get("appearances", 0)
                print(
                    f"  Dominant AR: consistency={consistency:.2f}, "
                    f"appearances={appearances}"
                )

        # Calculate bounds with 250km buffer (using largest object bounds)
        bounds_with_buffer = calculate_extent_bounds(
            left_lon, right_lon, bottom_lat, top_lat, extent_buffer=250
        )

        lon_min = bounds_with_buffer.longitude_min
        lon_max = bounds_with_buffer.longitude_max
        lat_min = bounds_with_buffer.latitude_min
        lat_max = bounds_with_buffer.latitude_max
        print(
            f"  Buffered bounds: {lon_min:.1f}-{lon_max:.1f}°, "
            f"{lat_min:.1f}-{lat_max:.1f}°"
        )

        # Create summary plot for this case
        print("  Creating summary plot...")
        ar_bounds_dict = {
            "latitude_min": bottom_lat,
            "latitude_max": top_lat,
            "longitude_min": left_lon,
            "longitude_max": right_lon,
        }

        create_case_summary_plot(
            case_id=event["case_id_number"],
            title=event["title"],
            ivt_data=ivt_data,
            ar_mask=ar_mask,
            peak_time_idx=peak_time_idx,
            peak_ivt_value=peak_ivt_value,
            ar_bounds=ar_bounds_dict,
            buffered_bounds=bounds_with_buffer,
            largest_obj_metadata=largest_obj_metadata,
        )

        ar_bounds_results_enhanced.append(
            {
                "case_id": event["case_id_number"],
                "title": event["title"],
                "start_date": event["start_date"],
                "end_date": event["end_date"],
                "original_bounds": event["location"]["parameters"],
                "ar_largest_object_bounds": {
                    "latitude_min": bottom_lat,
                    "latitude_max": top_lat,
                    "longitude_min": left_lon,
                    "longitude_max": right_lon,
                },
                "ar_dominant_object_bounds": dominant_bounds_info,
                "buffered_bounds": {
                    "latitude_min": bounds_with_buffer.latitude_min,
                    "latitude_max": bounds_with_buffer.latitude_max,
                    "longitude_min": bounds_with_buffer.longitude_min,
                    "longitude_max": bounds_with_buffer.longitude_max,
                },
                "bounds_region": bounds_with_buffer,
                "largest_object_metadata": largest_obj_metadata,
                "dominant_object_metadata": dominant_metadata,
                "peak_time_idx": peak_time_idx,
                "peak_ivt_value": peak_ivt_value,
                "peak_timestamp": ar_mask.valid_time.isel(
                    valid_time=peak_time_idx
                ).values,
                "ar_config": AR_OBJECT_CONFIG,
            }
        )

    except Exception as e:
        print(f"  Error processing {event['title']}: {e}")
        import traceback

        traceback.print_exc()
        continue

print(
    f"\nSuccessfully processed {len(ar_bounds_results_enhanced)} "
    f"events with enhanced method"
)

# %% [markdown]
# Save the enhanced atmospheric river bounds results

# %%
if ar_bounds_results_enhanced:
    pickle_file_path = "ar_bounds_results_enhanced.pkl"
    with open(pickle_file_path, "wb") as f:
        pickle.dump(ar_bounds_results_enhanced, f)

    print(
        f"Saved {len(ar_bounds_results_enhanced)} enhanced AR bounds "
        f"results to {pickle_file_path}"
    )

# %% [markdown]
# Visualize the enhanced results and compare with original method

# %%
if ar_bounds_results_enhanced:
    # Create comparison visualization
    fig = plt.figure(figsize=(20, 12))

    # Original method subplot
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax1.add_feature(cfeature.LAND, color="white", alpha=0.8)

    # Enhanced method subplot
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax2.add_feature(cfeature.LAND, color="white", alpha=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(ar_bounds_results_enhanced)))

    for i, result in enumerate(ar_bounds_results_enhanced):
        color = colors[i]

        # Plot original bounds (for reference)
        orig_bounds = result["original_bounds"]
        orig_rect = Rectangle(
            (orig_bounds["longitude_min"], orig_bounds["latitude_min"]),
            orig_bounds["longitude_max"] - orig_bounds["longitude_min"],
            orig_bounds["latitude_max"] - orig_bounds["latitude_min"],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
            linestyle=":",
        )
        ax1.add_patch(orig_rect)
        ax2.add_patch(
            Rectangle(
                (orig_bounds["longitude_min"], orig_bounds["latitude_min"]),
                orig_bounds["longitude_max"] - orig_bounds["longitude_min"],
                orig_bounds["latitude_max"] - orig_bounds["latitude_min"],
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                alpha=0.7,
                linestyle=":",
            )
        )

        # Plot enhanced bounds (largest object)
        ar_bounds = result["ar_largest_object_bounds"]
        enhanced_rect = Rectangle(
            (ar_bounds["longitude_min"], ar_bounds["latitude_min"]),
            ar_bounds["longitude_max"] - ar_bounds["longitude_min"],
            ar_bounds["latitude_max"] - ar_bounds["latitude_min"],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            transform=ccrs.PlateCarree(),
            alpha=0.3,
        )
        ax2.add_patch(enhanced_rect)

        # Plot buffered bounds
        buff_bounds = result["buffered_bounds"]
        buffered_rect = Rectangle(
            (buff_bounds["longitude_min"], buff_bounds["latitude_min"]),
            buff_bounds["longitude_max"] - buff_bounds["longitude_min"],
            buff_bounds["latitude_max"] - buff_bounds["latitude_min"],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            alpha=1.0,
        )
        ax1.add_patch(buffered_rect)
        ax2.add_patch(
            Rectangle(
                (buff_bounds["longitude_min"], buff_bounds["latitude_min"]),
                buff_bounds["longitude_max"] - buff_bounds["longitude_min"],
                buff_bounds["latitude_max"] - buff_bounds["latitude_min"],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                alpha=1.0,
            )
        )

    # Set extents
    for ax in [ax1, ax2]:
        ax.set_extent([180, 300, 10, 80], crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    ax1.set_title(
        "Original Method:\nOriginal bounds (dotted) + 250km buffer (solid)", fontsize=12
    )
    ax2.set_title(
        "Enhanced Method:\nLargest AR object (filled) + 250km buffer (solid)",
        fontsize=12,
    )

    # Add summary statistics
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis("off")

    summary_text = "Enhanced AR Bounds Results Summary:\n\n"
    for i, result in enumerate(ar_bounds_results_enhanced):
        metadata = result.get("largest_object_metadata", {})
        summary_text += f"{i + 1}. {result['title'][:40]}...\n"
        summary_text += f"   Objects found: {metadata.get('total_objects', 'N/A')}\n"
        summary_text += f"   Largest area: {metadata.get('area', 'N/A')} gridpoints\n"
        summary_text += "\n\n"

    ax3.text(
        0.05,
        0.95,
        summary_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Add configuration info
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    config_text = "AR Object Filtering Configuration:\n\n"
    for key, value in AR_OBJECT_CONFIG.items():
        config_text += f"{key}: {value}\n"

    config_text += "\nFiltering Criteria:\n"
    config_text += "• Minimum area: 300 grid points\n"
    config_text += "• Shape constraints: Removed (area only)\n"
    config_text += "• Bounds based on largest valid object\n"

    ax4.text(
        0.05,
        0.95,
        config_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig("ar_bounds_enhanced_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
