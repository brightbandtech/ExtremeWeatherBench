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
from typing import Union, Literal
import sparse
import pandas as pd
from tqdm.auto import tqdm


# %%
def sparse_practically_perfect_hindcast(
    da: xr.DataArray,
    resolution: float = 0.25,
    # TODO: add report type back in
    report_type: Union[Literal["all"], list[Literal["tor", "hail", "wind"]]] = ['tor','hail'],
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

    coords_df['report_type'] = (coords_df['report_type'] > 0).astype(int)

    # First, ensure longitude is in 0-360 range
    coords_df['longitude'] = utils.convert_longitude_to_360(coords_df['longitude'])

    if len(coords_df['report_type']) == 0:
        return xr.DataArray(
        sparse.COO.from_numpy(np.zeros((len(grid_lats), len(grid_lons)))),
        dims=["latitude", "longitude"],
        coords={"latitude": grid_lats, "longitude": grid_lons},
    )
    coords_da = coords_df.set_index(['latitude', 'longitude']).to_xarray()['report_type']

    # Regrid the sparse data to the target grid
    # Interpolate the sparse data to the target grid
    regridded_da = coords_da.interp(
        latitude=target_coords["latitude"],
        longitude=target_coords["longitude"],
        method="nearest",  # Use nearest neighbor for sparse data
    )

    # Fill NaN values with np.nan for the reports
    regridded_da = regridded_da.fillna(0)


    smoothed_reports = gaussian_filter(regridded_da, sigma=sigma)
    pph = xr.DataArray(
        smoothed_reports,
        dims=["latitude", "longitude"],
        coords={"latitude": grid_lats, "longitude": grid_lons},
    )
    if any(coords_df.longitude < 180):
        # Australia is a special case, we need to multiply by 10 to get over underreporting bias
        pph = pph * 10
    pph_sparse = pph.copy(data=sparse.COO.from_numpy(pph.to_numpy()))
    return pph_sparse

# %%
lsr = inputs.LSR(source=inputs.LSR_URI, variables=['report'],variable_mapping={'report': 'reports'}, storage_options={'anon': True})
lsr_df = lsr.open_and_maybe_preprocess_data_from_source()
lsr_ds = lsr._custom_convert_to_dataset(lsr_df)

# %%
unique_valid_times = np.unique(lsr_ds['valid_time'].values)

# %%
import joblib

pph_sparse = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(sparse_practically_perfect_hindcast)(lsr_ds['report_type'].sel(valid_time=time)) 
    for time in tqdm(unique_valid_times)
)
pph_sparse = xr.concat(pph_sparse, dim='valid_time')
pph_sparse

# %%
pph_sparse = [sparse_practically_perfect_hindcast(lsr_ds['report_type'].sel(valid_time=time)) for time in tqdm(unique_valid_times)]
pph_sparse = xr.concat(pph_sparse, dim='valid_time')
pph_sparse
