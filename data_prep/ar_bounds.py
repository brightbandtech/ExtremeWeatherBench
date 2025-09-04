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
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

# %%
from extremeweatherbench import cases, derived, inputs, regions, utils

# %% [markdown]
# Functions for calculating bounds from AR mask data

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


def find_ar_bounds_from_mask(
    ar_mask: xr.DataArray,
) -> tuple[float, float, float, float]:
    """Find the geographical bounds of atmospheric river mask.

    Parameters:
    -----------
    ar_mask : xr.DataArray
        Binary mask where 1 indicates atmospheric river presence

    Returns:
    --------
    tuple[float, float, float, float]
        left_lon, right_lon, bottom_lat, top_lat bounds
    """
    # Find where atmospheric river is detected (mask == 1)
    ar_locations = ar_mask.where(ar_mask == 1)

    if ar_locations.sum() == 0:
        # No AR detected, return NaN bounds
        return np.nan, np.nan, np.nan, np.nan

    # Get coordinates where AR is present
    ar_coords = ar_locations.stack(points=["latitude", "longitude"]).dropna("points")

    if len(ar_coords.points) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Extract lat/lon coordinates
    lats = ar_coords.latitude.values
    lons = ar_coords.longitude.values

    # Find bounds
    left_lon = float(np.min(lons))
    right_lon = float(np.max(lons))
    bottom_lat = float(np.min(lats))
    top_lat = float(np.max(lats))

    return left_lon, right_lon, bottom_lat, top_lat


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
# Process each atmospheric river event to calculate refined bounds

# %%
ar_bounds_results = []

for event in tqdm(ar_events):  # Process first 5 for testing
    try:
        # Create a case object for this event

        # Load the individual case using load_individual_cases
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

        # Derive the atmospheric river mask
        ar_dataset = derived.maybe_derive_variables(
            era5_subset, [derived.AtmosphericRiverMask]
        )

        if "atmospheric_river_mask" not in ar_dataset:
            print(f"  Warning: No AR mask generated for {event['title']}")
            continue

        # Get the AR mask
        ar_mask = ar_dataset["atmospheric_river_mask"]

        # Sum across time to get overall AR coverage for the event
        ar_mask_total = ar_mask.sum(dim="valid_time")

        # Find bounds from the mask
        left_lon, right_lon, bottom_lat, top_lat = find_ar_bounds_from_mask(
            ar_mask_total > 0  # Any time AR was present
        )

        if np.isnan(left_lon):
            print(f"  Warning: No AR detected for {event['title']}")
            continue

        print(
            f"  Raw bounds: {left_lon:.1f}-{right_lon:.1f}°, "
            f"{bottom_lat:.1f}-{top_lat:.1f}°"
        )

        # Calculate bounds with 250km buffer
        bounds_with_buffer = calculate_extent_bounds(
            left_lon, right_lon, bottom_lat, top_lat, extent_buffer=250
        )

        print(
            f"  Buffered bounds: {bounds_with_buffer.longitude_min:.1f}-"
            f"{bounds_with_buffer.longitude_max:.1f}°, "
            f"{bounds_with_buffer.latitude_min:.1f}-"
            f"{bounds_with_buffer.latitude_max:.1f}°"
        )

        ar_bounds_results.append(
            {
                "case_id": event["case_id_number"],
                "title": event["title"],
                "start_date": event["start_date"],
                "end_date": event["end_date"],
                "original_bounds": event["location"]["parameters"],
                "ar_mask_bounds": {
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
            }
        )

    except Exception as e:
        print(f"  Error processing {event['title']}: {e}")
        continue

print(f"\nSuccessfully processed {len(ar_bounds_results)} events")

# %% [markdown]
# Save the atmospheric river bounds results to a pickle file for later use

# %%

# Save ar_bounds_results to a pickle file
pickle_file_path = "ar_bounds_results.pkl"
with open(pickle_file_path, "wb") as f:
    pickle.dump(ar_bounds_results, f)

print(f"Saved {len(ar_bounds_results)} AR bounds results to {pickle_file_path}")


# %% [markdown]
# Visualize the results

# %%
# Create a figure with cartopy projection
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)

# Plot the AR mask
ar_mask.isel(valid_time=5).plot(
    ax=ax, transform=ccrs.PlateCarree(), cmap="Reds", add_colorbar=True
)

# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

plt.title("Atmospheric River Mask")

# %%
if ar_bounds_results:
    # Create a figure with cartopy projection
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)

    # Remove longitude labels from the top and latitude labels from the right
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Plot each AR event's bounding box
    colors = plt.cm.tab20(np.linspace(0, 1, len(ar_bounds_results)))

    for i, result in enumerate(ar_bounds_results):
        # Plot original bounds with +20 degree buffer as dotted line
        orig_bounds = result["original_bounds"]
        orig_lon_min = orig_bounds["longitude_min"]
        orig_lon_max = orig_bounds["longitude_max"]
        orig_lat_min = orig_bounds["latitude_min"]
        orig_lat_max = orig_bounds["latitude_max"]

        # Apply +20 degree buffer to original bounds
        buffer_orig_lon_min = orig_lon_min - 20
        buffer_orig_lon_max = orig_lon_max + 20
        buffer_orig_lat_min = max(orig_lat_min - 20, -90)  # Clamp to valid lat
        buffer_orig_lat_max = min(orig_lat_max + 20, 90)  # Clamp to valid lat

        # Create original +20 degree buffer rectangle patch
        orig_width = buffer_orig_lon_max - buffer_orig_lon_min
        orig_height = buffer_orig_lat_max - buffer_orig_lat_min

        # Handle longitude wrapping for original
        if buffer_orig_lon_max < buffer_orig_lon_min:
            buffer_orig_lon_min = buffer_orig_lon_min - 360
            orig_width = buffer_orig_lon_max - buffer_orig_lon_min

        orig_rect = Rectangle(
            (buffer_orig_lon_min, buffer_orig_lat_min),
            orig_width,
            orig_height,
            linewidth=1,
            edgecolor=colors[i],
            facecolor="none",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
            linestyle="--",
        )
        ax.add_patch(orig_rect)

        # Plot buffered bounds as solid line
        bounds = result["buffered_bounds"]

        # Extract bounds
        lon_min = bounds["longitude_min"]
        lon_max = bounds["longitude_max"]
        lat_min = bounds["latitude_min"]
        lat_max = bounds["latitude_max"]

        # Create rectangle patch
        width = lon_max - lon_min
        height = lat_max - lat_min

        # Handle longitude wrapping
        if lon_max < lon_min:
            print(f"Longitude wrapping for {result['title']}")
            lon_min = lon_min - 360
            width = lon_max - lon_min

        rect = Rectangle(
            (lon_min, lat_min),
            width,
            height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor="none",
            transform=ccrs.PlateCarree(),
            alpha=1,
            label=result["title"][:30],  # Truncate long titles
        )
        ax.add_patch(rect)

    # Set extent to North America and Pacific
    ax.set_extent([180, 300, 10, 80], crs=ccrs.PlateCarree())

    plt.title("AR Bounds: Original +20° (dashed) vs AR Mask +250km (solid)", loc="left")
    plt.tight_layout()
    plt.show()

    # Display the results
    print("\nAR Bounds Results:")
    for result in ar_bounds_results:
        print(f"\n{result['title']}:")
        print(f"  Original: {result['original_bounds']}")
        print(f"  AR Mask:  {result['ar_mask_bounds']}")
        print(f"  Buffered: {result['buffered_bounds']}")

# %%
# Convert results to DataFrame for easier analysis
if ar_bounds_results:
    df_results = pd.DataFrame(
        [
            {
                "case_id": r["case_id"],
                "title": r["title"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "orig_lat_min": r["original_bounds"]["latitude_min"],
                "orig_lat_max": r["original_bounds"]["latitude_max"],
                "orig_lon_min": r["original_bounds"]["longitude_min"],
                "orig_lon_max": r["original_bounds"]["longitude_max"],
                "mask_lat_min": r["ar_mask_bounds"]["latitude_min"],
                "mask_lat_max": r["ar_mask_bounds"]["latitude_max"],
                "mask_lon_min": r["ar_mask_bounds"]["longitude_min"],
                "mask_lon_max": r["ar_mask_bounds"]["longitude_max"],
                "buf_lat_min": r["buffered_bounds"]["latitude_min"],
                "buf_lat_max": r["buffered_bounds"]["latitude_max"],
                "buf_lon_min": r["buffered_bounds"]["longitude_min"],
                "buf_lon_max": r["buffered_bounds"]["longitude_max"],
            }
            for r in ar_bounds_results
        ]
    )

    print("\nComparison of bounds:")
    print(df_results.to_string())

# %%
