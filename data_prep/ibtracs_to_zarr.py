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
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sparse
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from typing import Sequence, Literal

# %%
from extremeweatherbench import inputs, utils, regions

# %%
import importlib

# %% [markdown]
# Sample calculation replacing CaseOperator data with hardcoded values:

# %%
IBTRACS = inputs.IBTrACS(
    source=inputs.IBTRACS_URI,
    variables=["vmax","slp"],
    variable_mapping={"vmax": "surface_wind_speed", "slp": "pressure_at_mean_sea_level"},
    storage_options={"anon": True},
)
IBTRACS_lf = IBTRACS.open_and_maybe_preprocess_data_from_source()

IBTrACS_metadata_variable_mapping = {
    "ISO_TIME": "valid_time",
    "NAME": "tc_name",
    "LAT": "latitude",
    "LON": "longitude",
    "USA_WIND": "surface_wind_speed",
    "USA_PRES": "pressure_at_mean_sea_level",
}

IBTRACS_lf = utils.maybe_map_variable_names(IBTRACS_lf, IBTrACS_metadata_variable_mapping)

# %%
year = 2024

# Get the season (year) from the case start date, cast as string as polars is interpreting the schema as strings
season = str(year)

# Create a subquery to find all storm numbers in the same season
matching_numbers = (
    IBTRACS_lf.filter(pl.col("SEASON") == season)
    .select("NUMBER")
    .unique()
)

# Apply the filter to get all data for storms with the same number in the same season
# This maintains the lazy evaluation
subset_target_data = IBTRACS_lf.join(
    matching_numbers, on="NUMBER", how="inner"
).filter(
    (pl.col("tc_name") == 'HELENE')
    & (pl.col("SEASON") == season)
)

# check that the variables are in the target data
schema_fields = [field for field in subset_target_data.collect_schema()]
target_variables = [
    v for v in ['vmax','slp'] if isinstance(v, str)
]
# subset the variables
if target_variables:
    subset_target_data = subset_target_data.select(IBTrACS_metadata_variable_mapping.values())


# %%
def calculate_haversine_distance(input_a: Sequence[float], input_b: Sequence[float], output_units: Literal["degrees", "km"] = "degrees") -> float:
    """Calculate the great-circle distance between two points on the Earth's surface.
    
    Args:
        input_a: The first point, represented as an ndarray of shape (2,n) in degrees lat/lon.
        input_b: The second point(s), represented as an ndarray of shape (2,n) in degrees lat/lon.

    Returns:
        The great-circle distance between the two points in degrees.
    """
    # Convert to radians for calculations
    lat1 = np.radians(input_a[0])
    lon1 = np.radians(input_a[1])
    lat2 = np.radians(input_b[0])
    lon2 = np.radians(input_b[1])
    
    # Haversine formula for great circle distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    if output_units == "degrees":
        distance = np.degrees(c)  # Convert back to degrees
    elif output_units == "km":
        distance = c * 6371  # Earth's radius in km
    else:
        raise ValueError(f"Invalid output units: {output_units}")
    return distance



def create_great_circle_mask(ds: xr.Dataset, latlon_point: tuple[float, float], radius_degrees: float) -> xr.DataArray:
    """
    Create a circular mask based on great circle distance for an xarray dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with 'latitude' and 'longitude' coordinates
    center_lat : float
        Latitude of the center point
    center_lon : float
        Longitude of the center point
    radius_degrees : float
        Radius in degrees of great circle distance
    
    Returns:
    --------
    mask : xarray.DataArray
        Boolean mask where True indicates points within the radius
    """
    
    distance = calculate_haversine_distance(latlon_point, (ds.latitude, ds.longitude), output_units="degrees")
    # Create mask as xarray DataArray
    mask = distance <= radius_degrees
    
    return mask


# %%
def calculate_end_point(start_lat: float, start_lon: float, bearing: float, distance_km: float) -> tuple[float, float]:
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
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_km / R) +
                     np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing_rad))
    
    lon2 = lon1 + np.arctan2(np.sin(bearing_rad) * np.sin(distance_km / R) * np.cos(lat1),
                             np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2))
    
    # Convert back to degrees
    end_lat = np.degrees(lat2)
    end_lon = np.degrees(lon2)
    
    return end_lat, end_lon

def calculate_extent_bounds(left_lon: float, right_lon: float, bottom_lat: float, top_lat: float, extent_buffer: float = 250, extent_units: Literal["degrees", "km"] = "km") -> regions.Region:

    new_left_lat, new_bottom_lon = calculate_end_point(bottom_lat, left_lon, 235, extent_buffer)
    new_right_lat, new_top_lon = calculate_end_point(top_lat, right_lon, 45, extent_buffer)
    new_box = regions.BoundingBoxRegion(new_left_lat, new_right_lat, new_bottom_lon, new_top_lon)
    old_box = regions.BoundingBoxRegion(bottom_lat, top_lat, left_lon, right_lon)
    return new_box, old_box


# %%
collected_df = subset_target_data.collect().to_pandas()
collected_df['latitude'] = collected_df['latitude'].astype(float)
collected_df['longitude'] = collected_df['longitude'].astype(float)

new_box, old_box = calculate_extent_bounds(left_lon=170, right_lon=180, bottom_lat=-10, top_lat=10)

fig, ax = plt.subplots(figsize=(10, 5))
old_box.geopandas.plot(ax=ax, alpha=0.5, color='red')
new_box.geopandas.plot(ax=ax, alpha=0.5, color='blue')
ax.set_title("IBTrACS Bounding Box")
plt.show()

# %%
xr.Dataset.from_dataframe(collected_df.set_index(["valid_time","latitude","longitude"]),sparse=True)

# %%
# IBTRACS_ds = IBTRACS._custom_convert_to_dataset(IBTRACS_df)
