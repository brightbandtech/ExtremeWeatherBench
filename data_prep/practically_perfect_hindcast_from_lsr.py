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


# %%
def practically_perfect_hindcast(
    ds: xr.Dataset,
    resolution: float = 0.25,
    # TODO: add report type back in
    # report_type: Union[Literal["all"], list[Literal["tor", "hail", "wind"]]] = "all",
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

    # Regrid the sparse dataset to the fixed global grid
    # First, ensure longitude is in 0-360 range
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=utils.convert_longitude_to_360(ds.longitude))

    # Interpolate the sparse data to the target grid
    regridded_ds = ds.interp(
        latitude=target_coords["latitude"],
        longitude=target_coords["longitude"],
        method="nearest",  # Use nearest neighbor for sparse data
    )

    # Fill NaN values with 0 for the reports
    if "reports" in regridded_ds.data_vars:
        regridded_ds["reports"] = regridded_ds["reports"].fillna(0)

    # Apply gaussian smoothing to the regridded data
    if "reports" in regridded_ds.data_vars:
        reports_data = regridded_ds["reports"].values
        smoothed_reports = gaussian_filter(reports_data, sigma=sigma)
        regridded_ds["practically_perfect"] = xr.DataArray(
            smoothed_reports,
            dims=["latitude", "longitude"],
            coords={"latitude": grid_lats, "longitude": grid_lons},
        )

    return regridded_ds


# %%
lsr = inputs.LSR(source=inputs.LSR_URI, variables=['report'],variable_mapping={'report': 'reports'}, storage_options={'anon': True})
lsr_df = lsr.open_and_maybe_preprocess_data_from_source()
lsr_ds = lsr._custom_convert_to_dataset(lsr_df)

# %%
lsr_ds
# next step is to run the practically perfect hindcast for each unique valid_time
# then we can save the results to a new dataset
