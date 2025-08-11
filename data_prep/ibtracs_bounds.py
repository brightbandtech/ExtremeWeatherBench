# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib.patches import Rectangle

# %%
from extremeweatherbench import inputs, regions, utils

# %% [markdown]
# Sample calculation replacing CaseOperator data with hardcoded values:


# %%
def calculate_end_point(
    start_lat: float, start_lon: float, bearing: float, distance_km: float
) -> tuple[float, float]:
    """
    Calculate the end point (latitude, longitude) given a starting point, bearing, and distance.

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


# %% [markdown]
# instantiate the data for processing

# %%
IBTRACS = inputs.IBTrACS(
    source=inputs.IBTRACS_URI,
    variables=["vmax", "slp"],
    variable_mapping={
        "vmax": "surface_wind_speed",
        "slp": "air_pressure_at_mean_sea_level",
    },
    storage_options={"anon": True},
)
IBTRACS_lf = IBTRACS.open_and_maybe_preprocess_data_from_source()

IBTrACS_metadata_variable_mapping = {
    "ISO_TIME": "valid_time",
    "NAME": "tc_name",
    "LAT": "latitude",
    "LON": "longitude",
    "WMO_WIND": "wmo_surface_wind_speed",
    "WMO_PRES": "wmo_air_pressure_at_mean_sea_level",
    "USA_WIND": "usa_surface_wind_speed",
    "USA_PRES": "usa_air_pressure_at_mean_sea_level",
    "NEUMANN_WIND": "neumann_surface_wind_speed",
    "NEUMANN_PRES": "neumann_air_pressure_at_mean_sea_level",
    "TOKYO_WIND": "tokyo_surface_wind_speed",
    "TOKYO_PRES": "tokyo_air_pressure_at_mean_sea_level",
    "CMA_WIND": "cma_surface_wind_speed",
    "CMA_PRES": "cma_air_pressure_at_mean_sea_level",
    "HKO_WIND": "hko_surface_wind_speed",
    "KMA_WIND": "kma_surface_wind_speed",
    "KMA_PRES": "kma_air_pressure_at_mean_sea_level",
    "NEWDELHI_WIND": "newdelhi_surface_wind_speed",
    "NEWDELHI_PRES": "newdelhi_air_pressure_at_mean_sea_level",
    "REUNION_WIND": "reunion_surface_wind_speed",
    "REUNION_PRES": "reunion_air_pressure_at_mean_sea_level",
    "BOM_WIND": "bom_surface_wind_speed",
    "BOM_PRES": "bom_air_pressure_at_mean_sea_level",
    "NADI_WIND": "nadi_surface_wind_speed",
    "NADI_PRES": "nadi_air_pressure_at_mean_sea_level",
    "WELLINGTON_WIND": "wellington_surface_wind_speed",
    "WELLINGTON_PRES": "wellington_air_pressure_at_mean_sea_level",
    "DS824_WIND": "ds824_surface_wind_speed",
    "DS824_PRES": "ds824_air_pressure_at_mean_sea_level",
    "MLC_WIND": "mlc_surface_wind_speed",
    "MLC_PRES": "mlc_air_pressure_at_mean_sea_level",
}

IBTRACS_lf = utils.maybe_map_variable_names(
    IBTRACS_lf, IBTrACS_metadata_variable_mapping
)

# %% [markdown]
# Get all storms from 2020 - 2025 seasons:

# %%
# Keep the data as a LazyFrame and do all operations lazily
all_storms_2020_2025_lf = IBTRACS_lf.filter(
    (pl.col("SEASON").cast(pl.Int32) >= 2020)
).select(IBTrACS_metadata_variable_mapping.values())

schema = all_storms_2020_2025_lf.collect_schema()
# Convert pressure and surface wind columns to float, replacing " " with null
# Get column names that contain "pressure" or "wind"
pressure_cols = [col for col in schema if "pressure" in col.lower()]
wind_cols = [col for col in schema if "wind" in col.lower()]

# Apply transformations to convert " " to null and cast to float
all_storms_lf = all_storms_2020_2025_lf.with_columns(
    [
        pl.when(pl.col(col) == " ")
        .then(None)
        .otherwise(pl.col(col))
        .cast(pl.Float64, strict=False)
        .alias(col)
        for col in pressure_cols + wind_cols
    ]
)

# Drop rows where ALL columns are null (equivalent to pandas dropna(how="all"))
all_storms_lf = all_storms_lf.filter(~pl.all_horizontal(pl.all().is_null()))

# Create unified pressure and wind columns by preferring USA and WMO data
# For surface wind speed
wind_columns = [col for col in schema if "surface_wind_speed" in col]
wind_priority = ["usa_surface_wind_speed", "wmo_surface_wind_speed"] + [
    col
    for col in wind_columns
    if col not in ["usa_surface_wind_speed", "wmo_surface_wind_speed"]
]

# For pressure at mean sea level
pressure_columns = [col for col in schema if "air_pressure_at_mean_sea_level" in col]
pressure_priority = [
    "usa_air_pressure_at_mean_sea_level",
    "wmo_air_pressure_at_mean_sea_level",
] + [
    col
    for col in pressure_columns
    if col
    not in ["usa_air_pressure_at_mean_sea_level", "wmo_air_pressure_at_mean_sea_level"]
]

# Create unified columns using coalesce (equivalent to pandas bfill)
all_storms_lf = all_storms_lf.with_columns(
    [
        pl.coalesce(wind_priority).alias("surface_wind_speed"),
        pl.coalesce(pressure_priority).alias("air_pressure_at_mean_sea_level"),
    ]
)

# Select only the columns to keep
columns_to_keep = [
    "valid_time",
    "tc_name",
    "latitude",
    "longitude",
    "surface_wind_speed",
    "air_pressure_at_mean_sea_level",
]

all_storms_lf = all_storms_lf.select(columns_to_keep)

# Drop rows where wind speed OR pressure are null (equivalent to pandas dropna with how="any")
all_storms_lf = all_storms_lf.filter(
    pl.col("surface_wind_speed").is_not_null()
    & pl.col("air_pressure_at_mean_sea_level").is_not_null()
)

# Only collect when you need the actual data for operations that require pandas
# For checking null counts, you can collect just the null counts:
print("Missing values per column:")
null_counts = all_storms_lf.select(
    [pl.col(col).null_count().alias(f"{col}_nulls") for col in columns_to_keep]
).collect()
print(null_counts)

print(f"Total rows after filtering: {all_storms_lf.select(pl.len()).collect().item()}")

# When you need pandas DataFrame for the groupby operation with your custom function:
all_storms_df = all_storms_lf.collect().to_pandas()

# %%
all_storms_df

# %%
# Check for missing values in each column
print(all_storms_df.isnull().sum())

# Create unified pressure and wind columns by preferring USA and WMO data
# Priority order: USA -> WMO -> Other agencies

# For surface wind speed
wind_columns = [col for col in all_storms_df.columns if "surface_wind_speed" in col]
wind_priority = ["usa_surface_wind_speed", "wmo_surface_wind_speed"] + [
    col
    for col in wind_columns
    if col not in ["usa_surface_wind_speed", "wmo_surface_wind_speed"]
]

all_storms_df["surface_wind_speed"] = (
    all_storms_df[wind_priority].bfill(axis=1).iloc[:, 0]
)

# For pressure at mean sea level
pressure_columns = [
    col for col in all_storms_df.columns if "air_pressure_at_mean_sea_level" in col
]
pressure_priority = [
    "usa_air_pressure_at_mean_sea_level",
    "wmo_air_pressure_at_mean_sea_level",
] + [
    col
    for col in pressure_columns
    if col
    not in ["usa_air_pressure_at_mean_sea_level", "wmo_air_pressure_at_mean_sea_level"]
]

all_storms_df["air_pressure_at_mean_sea_level"] = (
    all_storms_df[pressure_priority].bfill(axis=1).iloc[:, 0]
)

# Drop the individual agency columns and keep only the unified columns
columns_to_keep = [
    "valid_time",
    "tc_name",
    "latitude",
    "longitude",
    "surface_wind_speed",
    "air_pressure_at_mean_sea_level",
]
all_storms_df = all_storms_df[columns_to_keep]

print("\nAfter merging columns:")
print(all_storms_df.isnull().sum())


# %%
# Drop rows where both wind speed and pressure are NaN
all_storms_df = all_storms_df.dropna(
    subset=["surface_wind_speed", "air_pressure_at_mean_sea_level"], how="any"
)

print(
    f"After dropping rows with missing wind speed and pressure: {len(all_storms_df)} rows remaining"
)
print("Missing values per column:")
print(all_storms_df.isnull().sum())


# %%
# Group by tc_name and calculate extent bounds for each storm
storm_bounds = all_storms_df.groupby("tc_name").apply(
    lambda group: calculate_extent_bounds(
        left_lon=group["longitude"].min(),
        right_lon=group["longitude"].max(),
        bottom_lat=group["latitude"].min(),
        top_lat=group["latitude"].max(),
    )
)

# Create a figure with cartopy projection
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Plot each storm's bounding box
colors = plt.cm.tab20(np.linspace(0, 1, len(storm_bounds)))
for i, (storm_name, bounds) in enumerate(storm_bounds.items()):
    # Extract bounds
    lon_min = bounds.longitude_min
    lon_max = bounds.longitude_max
    lat_min = bounds.latitude_min
    lat_max = bounds.latitude_max

    # Create rectangle patch
    width = lon_max - lon_min
    height = lat_max - lat_min

    rect = Rectangle(
        (lon_min, lat_min),
        width,
        height,
        linewidth=1,
        edgecolor=colors[i],
        facecolor="none",
        transform=ccrs.PlateCarree(),
        alpha=1,
    )
    ax.add_patch(rect)

# Set global extent
ax.set_global()

plt.title("Storm Bounding Boxes from IBTrACS Data +250km Buffer")
plt.tight_layout()
plt.show()

# Also display the storm_bounds data
storm_bounds

# %%
cases_old = utils.load_events_yaml()["cases"]
names = [n["title"].upper() for n in cases_old if n["event_type"] == "tropical_cyclone"]
cases_new = cases_old.copy()
# Update the yaml cases with storm bounds from IBTrACS data
for single_case in cases_new:
    if single_case["event_type"] == "tropical_cyclone":
        storm_name = single_case["title"].upper()

        # Check if storm name has parentheses and extract both versions
        found_bounds = None
        if storm_name in storm_bounds:
            found_bounds = storm_bounds[storm_name]
        elif "(" in storm_name and ")" in storm_name:
            # Extract name before parentheses
            name_before = storm_name.split("(")[0].strip()
            # Extract name inside parentheses
            name_in_parens = storm_name.split("(")[1].split(")")[0].strip()

            if name_before in storm_bounds:
                found_bounds = storm_bounds[name_before]
            elif name_in_parens in storm_bounds:
                found_bounds = storm_bounds[name_in_parens]

        # Check if storm name contains 'AND' and try to find combined name with ':'
        if found_bounds is None and " AND " in storm_name:
            # Split the names by 'AND' and search for each individually first
            names_parts = storm_name.split(" AND ")
            if len(names_parts) == 2:
                name1 = names_parts[0].strip()
                name2 = names_parts[1].strip()

                # Try to find each name individually
                bounds1 = storm_bounds.get(name1)
                bounds2 = storm_bounds.get(name2)

                # If we found both, merge them by taking the bounding box that encompasses both
                if bounds1 and bounds2:
                    from types import SimpleNamespace

                    merged_bounds = SimpleNamespace(
                        latitude_min=min(bounds1.latitude_min, bounds2.latitude_min),
                        latitude_max=max(bounds1.latitude_max, bounds2.latitude_max),
                        longitude_min=min(bounds1.longitude_min, bounds2.longitude_min),
                        longitude_max=max(bounds1.longitude_max, bounds2.longitude_max),
                    )
                    found_bounds = merged_bounds
                # If only one found, use that one
                elif bounds1:
                    found_bounds = bounds1
                elif bounds2:
                    found_bounds = bounds2
                else:
                    # Fall back to trying combined name formats
                    combined_name = f"{name1}:{name2}"
                    if combined_name in storm_bounds:
                        found_bounds = storm_bounds[combined_name]
                    # Also try with hyphen format
                    combined_name_hyphen = f"{name1}-{name2}"
                    if found_bounds is None and combined_name_hyphen in storm_bounds:
                        found_bounds = storm_bounds[combined_name_hyphen]

        if found_bounds:
            # Get storm data for this storm to find first and last valid times
            storm_data = all_storms_df[all_storms_df["tc_name"] == storm_name]
            if len(storm_data) == 0:
                # Try to find with different name formats
                for key in storm_bounds.keys():
                    if (
                        storm_name in key
                        or (name_before and name_before in key)
                        or (name_in_parens and name_in_parens in key)
                    ):
                        storm_data = all_storms_df[all_storms_df["tc_name"] == key]
                        if len(storm_data) > 0:
                            break

            # Update the case with IBTrACS bounding box coordinates
            single_case["location"]["parameters"]["latitude_min"] = float(
                found_bounds.latitude_min
            )
            single_case["location"]["parameters"]["latitude_max"] = float(
                found_bounds.latitude_max
            )
            single_case["location"]["parameters"]["longitude_min"] = float(
                found_bounds.longitude_min
            )
            single_case["location"]["parameters"]["longitude_max"] = float(
                found_bounds.longitude_max
            )

            # Update start and end dates based on storm valid times +/- 48 hours
            if len(storm_data) > 0:
                first_time = storm_data["valid_time"].min()
                last_time = storm_data["valid_time"].max()

                # Add/subtract 48 hours (2 days)
                start_date = (
                    pd.to_datetime(first_time) - pd.Timedelta(hours=48)
                ).strftime("%Y-%m-%d")
                end_date = (
                    pd.to_datetime(last_time) + pd.Timedelta(hours=48)
                ).strftime("%Y-%m-%d")

                single_case["start_date"] = start_date
                single_case["end_date"] = end_date

                print(f"Updated {storm_name} with bounds")
                print(f"  Start date: {start_date}, End date: {end_date}")
            else:
                print(f"Updated {storm_name} with bounds: {found_bounds}")
                print("Warning: Could not find storm data to update dates")
        else:
            print(f"NOT updated: Storm {storm_name} not found in IBTrACS data")


# %%
# Load the original events.yaml file
with open("src/extremeweatherbench/data/events.yaml", "r") as f:
    events_data = yaml.safe_load(f)

cases_old = events_data["cases"]

# Find changes between old and new cases
updated_cases = []
for i, case_new in enumerate(cases_new):
    case_old = cases_old[i] if i < len(cases_old) else None

    if case_old is None or case_new != case_old:
        updated_cases.append((i, case_new))
        print(
            f"Case {case_new['case_id_number']} ({case_new['title']}) has been updated"
        )

# Replace only the updated cases in the original data
for case_index, updated_case in updated_cases:
    if case_index < len(cases_old):
        cases_old[case_index] = updated_case

# Write the updated events.yaml file
events_data["cases"] = cases_old
with open(
    "/home/taylor/code/ExtremeWeatherBench/src/extremeweatherbench/data/events.yaml",
    "w",
) as f:
    yaml.dump(events_data, f, default_flow_style=False, sort_keys=False, indent=2)

print(f"\nUpdated {len(updated_cases)} cases in events.yaml")
