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
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from extremeweatherbench import utils, inputs
import sparse
import pandas as pd
from tqdm.auto import tqdm


# %%
def sparse_practically_perfect_hindcast(
    da: xr.DataArray,
    resolution: float = 0.25,
    # 1 is wind, 2 is hail, 3 is tornado
    sigma: float = 1.5,
) -> xr.Dataset:
    """Compute the Practically Perfect Hindcast (PPH) using storm report data using latitude/longitude grid spacing
    instead of the NCEP 212 Eta Lambert Conformal projection; based on the method described in Hitchens et al 2013,
    https://doi.org/10.1175/WAF-D-12-00113.1

    Args:
        ds: An xarray Dataset containing the storm report data as a sparse (COO) array.
        resolution: The resolution of the grid in degrees to use. Default is 0.25 degrees.
        sigma: The standard deviation of the gaussian filter to use. Default is 1.5.
    Returns:
        pph: An xarray Dataset containing the PPH and storm report data.
    """

    # Create a global grid with 0.25 degree resolution (721 x 1440)
    min_lat_fixed = -90.0  # Start at -90 degrees
    max_lat_fixed = 90.0  # End at 90 degrees
    min_lon_fixed = 0.0  # Start at 0 degrees
    max_lon_fixed = 359.75  # End at 359.75 degrees (360-0.25)

    # Create the grid coordinates
    grid_lats = np.arange(min_lat_fixed, max_lat_fixed + resolution, resolution)
    grid_lons = np.arange(min_lon_fixed, max_lon_fixed + resolution, resolution)

    # Create target coordinates for regridding
    target_coords = {
        "latitude": xr.DataArray(grid_lats, dims=["latitude"]),
        "longitude": xr.DataArray(grid_lons, dims=["longitude"]),
    }

    da_dense = da.to_numpy()

    # Find where we have non-NaN values - the sparse array might have NaNs instead of zeros
    valid_mask = ~np.isnan(da_dense)
    valid_indices = np.where(valid_mask)
    lat_indices, lon_indices = valid_indices

    # Only keep the locations that actually have data, not the zeros or NaNs
    if len(lat_indices) > 0:
        # Get the actual lat/lon coordinates for these spots
        lats = da.latitude.values[lat_indices]
        lons = da.longitude.values[lon_indices]
        
        # Get the report_type values at these locations
        report_values = da_dense[valid_indices]
        
        # Put it all together in a dataframe
        coords_df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'report_type': report_values.flatten() if hasattr(report_values, 'flatten') else report_values
        })
    else:
        coords_df = pd.DataFrame(columns=['latitude', 'longitude', 'report_type'])
    # 0 is no report, 1 is wind, 2 is hail, 3 is tornado. Drop no reports or wind reports.
    if set(np.unique(coords_df['report_type'])).issubset({0, 1}):
        return None
    # First, ensure longitude is in 0-360 range
    coords_df['longitude'] = utils.convert_longitude_to_360(coords_df['longitude'])
    mapped_coords_df = coords_df.copy()
    # If any longitude is less than 180, we need to multiply by 10 to get over underreporting bias
    if any(coords_df.longitude < 180):
        # Australia is a special case, we need to multiply by 10 to get over underreporting bias
        mapped_coords_df['report_type'] = coords_df['report_type'].map({0:0, 1:1, 2: 50, 3: 10})
    else:
        mapped_coords_df['report_type'] = coords_df['report_type'].map({0:0, 1:1, 2:2, 3:3})


    coords_da = mapped_coords_df.set_index(['latitude', 'longitude']).to_xarray()['report_type']

    # Regrid the sparse data to the target grid
    # Interpolate the sparse data to the target grid
    # Handle case where there's only one data point - interpolation doesn't work
    if len(coords_da.latitude) <= 2 or len(coords_da.longitude) <= 2:
        # Create a full grid filled with zeros
        regridded_da = xr.DataArray(
            np.zeros((len(target_coords["latitude"]), len(target_coords["longitude"]))),
            dims=["latitude", "longitude"],
            coords={"latitude": target_coords["latitude"], "longitude": target_coords["longitude"]}
        )
        
        # Handle multiple data points when we have sparse data
        for i in range(len(coords_da.latitude)):
            for j in range(len(coords_da.longitude)):
                lat_val = coords_da.latitude.values[i]
                lon_val = coords_da.longitude.values[j]
                report_val = coords_da.values[i, j]
                
                # Skip if this is a NaN value or zero (no report)
                if np.isnan(report_val) or report_val == 0:
                    continue
                    
                lat_idx = np.argmin(np.abs(target_coords["latitude"].values - lat_val))
                lon_idx = np.argmin(np.abs(target_coords["longitude"].values - lon_val))
                
                # Use the maximum value if there's already a value at this grid point
                regridded_da.values[lat_idx, lon_idx] = max(regridded_da.values[lat_idx, lon_idx], report_val)
    else:
        regridded_da = coords_da.interp(
            latitude=target_coords["latitude"],
            longitude=target_coords["longitude"],
            method="nearest", 
        )

    # Fill NaN values with np.nan for the reports
    regridded_da = regridded_da.fillna(0)


    smoothed_reports = gaussian_filter(regridded_da, sigma=sigma)
    pph = xr.DataArray(
        smoothed_reports,
        dims=["latitude", "longitude"],
        coords={"latitude": grid_lats, "longitude": grid_lons},
    )
    pph_sparse = pph.copy(data=sparse.COO.from_numpy(pph.to_numpy()))
    if pph_sparse.data.nnz == 0:
        return None
    return pph_sparse

# %%
lsr = inputs.LSR(source=inputs.LSR_URI, variables=['report'],variable_mapping={'report': 'reports'}, storage_options={'anon': True})
lsr_df = lsr.open_and_maybe_preprocess_data_from_source()
lsr_ds = lsr._custom_convert_to_dataset(lsr_df)

# %%
unique_valid_times = np.unique(lsr_ds['valid_time'].values)

# %% [markdown]
# parallel:

# %%
import joblib

pph_sparse_list = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(sparse_practically_perfect_hindcast)(lsr_ds['report_type'].sel(valid_time=time)) 
    for time in tqdm(unique_valid_times)
)
# Find indices where pph_sparse_list is None and filter them out
valid_indices = [i for i, item in enumerate(pph_sparse_list) if item is not None]
filtered_valid_times = [unique_valid_times[i] for i in valid_indices]

pph_sparse = xr.concat([n for n in pph_sparse_list if n is not None], pd.Index(filtered_valid_times, name='valid_time'))
pph_sparse

# %% [markdown]
# serial:

# %%
pph_sparse = [sparse_practically_perfect_hindcast(lsr_ds['report_type'].sel(valid_time=time)) for time in tqdm(unique_valid_times)]
pph_sparse = xr.concat(pph_sparse, unique_valid_times, name='valid_time')
pph_sparse

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

def plot_pph_contours(dt: datetime, da: xr.DataArray):
    """Make a cartopy contour plot of the practically perfect hindcast data.
    
    Args:
        dt: The datetime to plot
        dataset: The xarray dataset containing PPH data
    """
    # Select the data for the given datetime
    data_slice = da.sel(valid_time=dt, method='nearest')
    n_points = data_slice.data.nnz
    
    # Convert sparse array to dense if needed
    if hasattr(data_slice.data, 'todense'):
        data_slice = data_slice.copy(data=data_slice.data.todense())
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.set_extent([-125, -66.5, 20, 50])
    
    # Create contour levels from 0 to 1 every 0.1
    levels = np.arange(0.1, 1.1, 0.1)
    
    # Make the contour plot
    contours = ax.contour(
        data_slice['longitude'], 
        data_slice['latitude'], 
        data_slice, 
        levels=levels, 
        transform=ccrs.PlateCarree(),
        colors='red',
        linewidths=1.5
    )
    
    # Add contour labels
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
    
    # Set extent to focus on data region
    # ax.set_global()
    
    # Add title
    plt.title(f'Practically Perfect Hindcast - {pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M UTC")}, N={n_points}')
    
    plt.tight_layout()
    return fig, ax



# %%
plot_pph_contours(filtered_valid_times[40], pph_sparse)

# %%
