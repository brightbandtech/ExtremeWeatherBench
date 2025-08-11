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
def calculate_extent_bounds(df: pd.DataFrame, extent_buffer: float = 250, extent_units: Literal["degrees", "km"] = "km") -> regions.Region:

    min_lat = df["latitude"].min()
    max_lat = df["latitude"].max()
    min_lon = df["longitude"].min()
    max_lon = df["longitude"].max()

    min_lat = min_lat - extent_buffer
    max_lat = max_lat + extent_buffer
    min_lon = min_lon - extent_buffer
    max_lon = max_lon + extent_buffer

    return regions.Region(
        min_lat=df["latitude"].min(),
        max_lat=df["latitude"].max(),
        min_lon=df["longitude"].min(),
        max_lon=df["longitude"].max(),
    )



# %%
collected_df = subset_target_data.collect().to_pandas()
collected_df

# %%
xr.Dataset.from_dataframe(collected_df.set_index(["valid_time","latitude","longitude"]),sparse=True)

# %%
# IBTRACS_ds = IBTRACS._custom_convert_to_dataset(IBTRACS_df)
