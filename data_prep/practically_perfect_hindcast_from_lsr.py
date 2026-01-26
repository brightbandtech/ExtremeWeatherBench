from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

import extremeweatherbench.inputs as inputs
import extremeweatherbench.utils as utils


def sparse_practically_perfect_hindcast(
    da: xr.DataArray,
    resolution: float = 0.25,
    # 1 is wind, 2 is hail, 3 is tornado
    sigma: float = 5,
) -> xr.Dataset:
    """Compute the Practically Perfect Hindcast (PPH) using storm report data
    using latitude/longitude grid spacing instead of the NCEP 212 Eta Lambert
    Conformal projection; based on the method described in Hitchens et al 2013,
    https://doi.org/10.1175/WAF-D-12-00113.1

    Args:
        da: An xarray DataArray containing the storm report data as a sparse
            (COO) array.
        resolution: The resolution of the grid in degrees to use. Default is
            0.25 degrees.
        sigma: The standard deviation of the gaussian filter to use. Default
            is 5.

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
    valid_mask = ~np.isnan(da_dense)
    lat_indices, lon_indices = np.where(valid_mask)

    if len(lat_indices) > 0:
        lats = da.latitude.values[lat_indices]
        lons = da.longitude.values[lon_indices]
        report_values = da_dense[valid_mask]
        coords_df = pd.DataFrame(
            {
                "latitude": lats,
                "longitude": lons,
                "report_type": report_values.flatten()
                if hasattr(report_values, "flatten")
                else report_values,
            }
        )
    else:
        coords_df = pd.DataFrame(columns=["latitude", "longitude", "report_type"])

    # 0 is no report, 1 is wind, 2 is hail, 3 is tornado. Drop if only no report/wind.
    if set(np.unique(coords_df["report_type"])).issubset({0, 1}):
        return None

    # Normalize longitudes to [0, 360)
    coords_df["longitude"] = utils.convert_longitude_to_360(coords_df["longitude"])

    # Underreporting adjustment; 10 for hail and tornado globally
    mapped_coords_df = coords_df.copy()
    mapped_coords_df["report_type"] = coords_df["report_type"].map(
        {0: 0, 1: 0, 2: 10, 3: 10}
    )

    if len(mapped_coords_df) == 0:
        return None

    lat_vals = mapped_coords_df["latitude"].to_numpy()
    lon_vals = mapped_coords_df["longitude"].to_numpy()
    vals = mapped_coords_df["report_type"].to_numpy()

    # Only keep nonzero, finite entries
    good = np.isfinite(vals) & (vals != 0)
    if not np.any(good):
        return None

    lat_vals = lat_vals[good]
    lon_vals = lon_vals[good]
    vals = vals[good]

    nlat = grid_lats.size
    nlon = grid_lons.size

    # Compute nearest index per point; clip to bounds
    lat_idx = np.rint((lat_vals - min_lat_fixed) / resolution).astype(int)
    lon_idx = np.rint((lon_vals - min_lon_fixed) / resolution).astype(int)
    lat_idx = np.clip(lat_idx, 0, nlat - 1)
    lon_idx = np.clip(lon_idx, 0, nlon - 1)

    # Scatter with max reduction (handles multiple points per cell)
    regrid_values = np.zeros((nlat, nlon), dtype=float)
    np.maximum.at(regrid_values, (lat_idx, lon_idx), vals)

    regridded_da = xr.DataArray(
        regrid_values,
        dims=["latitude", "longitude"],
        coords={
            "latitude": target_coords["latitude"],
            "longitude": target_coords["longitude"],
        },
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


def plot_pph_contours(dt: datetime, da: xr.DataArray):
    """Make a cartopy contour plot of the practically perfect hindcast data.

    Args:
        dt: The datetime to plot
        dataset: The xarray dataset containing PPH data
    """
    # Select the data for the given datetime
    data_slice = da.sel(valid_time=dt, method="nearest")
    n_points = data_slice.data.nnz

    # Convert sparse array to dense if needed
    if hasattr(data_slice.data, "todense"):
        data_slice = data_slice.copy(data=data_slice.data.todense())
    # Create the plot
    _ = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.set_extent([-125, -66.5, 20, 50])

    levels = [0.01, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75]  # 10 levels between 0 and 1

    # Create the colormap with alpha=0 for values below 0.05
    # Create a mask for values below 0.05
    mask = np.ma.masked_less(data_slice, 0.01)
    cmap_with_alpha = plt.cm.viridis.copy()
    cmap_with_alpha.set_bad("none", alpha=0)  # Set masked values to transparent

    _ = ax.contour(
        data_slice.longitude,
        data_slice.latitude,
        mask,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap=cmap_with_alpha,
        extend="both",
    )

    # Make the contour plot
    contours = ax.contour(
        data_slice["longitude"],
        data_slice["latitude"],
        data_slice,
        levels=levels,
        transform=ccrs.PlateCarree(),
        colors="black",
        linewidths=0.75,
    )

    # Add contour labels
    ax.clabel(contours, inline=True, fontsize=10, fmt="%.1f")

    # Add title
    plt.title(
        f"Practically Perfect Hindcast - {pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M UTC')}, N={n_points}"  # noqa: E501
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lsr = inputs.LSR(
        storage_options={"anon": True},
    )
    lsr_df = lsr.open_and_maybe_preprocess_data_from_source()
    lsr_ds = lsr._custom_convert_to_dataset(lsr_df)
    unique_valid_times = np.unique(lsr_ds["valid_time"].values)

    pph_sparse_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(sparse_practically_perfect_hindcast)(
            lsr_ds["report_type"].sel(valid_time=time)
        )
        for time in tqdm(unique_valid_times)
    )
    # Find indices where pph_sparse_list is None and filter them out
    valid_indices = [i for i, item in enumerate(pph_sparse_list) if item is not None]
    filtered_valid_times = [unique_valid_times[i] for i in valid_indices]

    pph_sparse = xr.concat(
        [n for n in pph_sparse_list if n is not None],
        pd.Index(filtered_valid_times, name="valid_time"),
    )

    # Optionally plot a PPH example (uncomment to plot)
    # plot_pph_contours(filtered_valid_times[865], pph_sparse)

    # Convert sparse array to dense and back to DataArray
    pph_dense = pph_sparse.copy()
    pph_dense.data = pph_sparse.data.todense()
    pph_dense.name = "practically_perfect_hindcast"

    # will become dataset on load
    pph_dense.to_zarr("practically_perfect_hindcast_20200104_20250927.zarr", mode="w")
