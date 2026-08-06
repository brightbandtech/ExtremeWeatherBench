"""Synthetic lat/lon grids and time series with no event semantics."""

import numpy as np
import pandas as pd
import sparse
import xarray as xr


def make_chunked_global_grid_dataset(n_time=8, n_lat=181, n_lon=360):
    """Chunked global grid for region subsetting."""
    return xr.Dataset(
        {
            "t": (
                ["valid_time", "latitude", "longitude"],
                np.zeros((n_time, n_lat, n_lon), dtype="float32"),
            )
        },
        coords={
            "valid_time": pd.date_range("2021-06-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(-90, 90, n_lat),
            "longitude": np.linspace(-180, 180, n_lon, endpoint=False),
        },
    ).chunk({"valid_time": 2})


def make_coarse_global_grid_dataset():
    """Coarse global grid for region masking."""
    return xr.Dataset(
        {"t": (["latitude", "longitude"], np.zeros((73, 144)))},
        coords={
            "latitude": np.linspace(-90, 90, 73),
            "longitude": np.linspace(-180, 177.5, 144),
        },
    )


def make_global_grid_coords(n_lat=181, n_lon=360):
    """Global-ish grid, so a local storm covers a small part of it."""
    return (
        np.linspace(-90.0, 90.0, n_lat),
        np.linspace(0.0, 359.0, n_lon),
    )


def make_daily_series_dataarray(n_days, timesteps_per_day=4, chunk=True, drop_last=0):
    """Hourly-ish series over whole days, optionally with a truncated last day."""
    n_steps = n_days * timesteps_per_day - drop_last
    freq = f"{24 // timesteps_per_day}h"
    valid_time = pd.date_range("2021-06-01", periods=n_steps, freq=freq)
    values = np.arange(float(n_steps))
    da = xr.DataArray(values, dims=["valid_time"], coords={"valid_time": valid_time})
    if chunk:
        da = da.chunk({"valid_time": timesteps_per_day})
    return da


def make_spatial_dataarray(n_time=6, n_lat=8, n_lon=8, chunk=True):
    """Small (valid_time, latitude, longitude) array for reduction tests."""
    rng = np.random.default_rng(3)
    da = xr.DataArray(
        rng.uniform(280.0, 310.0, (n_time, n_lat, n_lon)),
        dims=["valid_time", "latitude", "longitude"],
        coords={
            "valid_time": pd.date_range("2023-01-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(30.0, 30.0 + n_lat - 1, n_lat),
            "longitude": np.linspace(-120.0, -120.0 + n_lon - 1, n_lon),
        },
    )
    return da.chunk({"valid_time": 1}) if chunk else da


def make_sample_sparse_target_dataarray() -> xr.DataArray:
    # Create a simple sparse array with known coordinates
    coords = ([0, 1, 2], [0, 1, 0])  # (lat_indices, lon_indices)
    data = [1.0, 2.0, 3.0]  # values at those coordinates
    shape = (3, 2)  # (lat, lon)

    sparse_array = sparse.COO(coords, data, shape=shape)

    # Create xarray DataArray with sparse data
    da = xr.DataArray(
        sparse_array,
        dims=["latitude", "longitude"],
        coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
    )
    return da


def make_sparse_grid_dataarray():
    """DataArray backed by a sparse.COO array."""
    data = sparse.COO(
        coords=np.array([[0, 1, 2], [1, 2, 0]]),
        data=np.array([1.0, 2.0, 3.0]),
        shape=(4, 4),
    )
    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": np.linspace(0.0, 3.0, 4),
            "longitude": np.linspace(10.0, 13.0, 4),
        },
    )
