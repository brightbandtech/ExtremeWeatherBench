"""Data with a vertical dimension, for the column integrals."""

import numpy as np
import pandas as pd
import xarray as xr


def make_pressure_column_dataarray(
    n_time=4, n_lat=3, n_lon=3, levels=(1000.0, 850.0, 700.0, 500.0)
):
    """Array with a level dimension, as the vertical integrals expect."""
    rng = np.random.default_rng(7)
    shape = (n_time, len(levels), n_lat, n_lon)
    return xr.DataArray(
        rng.uniform(0.001, 0.02, shape),
        dims=["valid_time", "level", "latitude", "longitude"],
        coords={
            "valid_time": pd.date_range("2023-01-01", periods=n_time, freq="6h"),
            "level": np.array(levels),
            "latitude": np.linspace(30.0, 40.0, n_lat),
            "longitude": np.linspace(-120.0, -110.0, n_lon),
        },
    )


def make_pressure_level_dataset(chunk=True):
    """Pressure-level dataset shaped like the AR derived-variable input."""
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            name: (
                ["valid_time", "level", "latitude", "longitude"],
                rng.random((6, levels.size, 12, 16), dtype="float32"),
            )
            for name in ("specific_humidity", "eastward_wind", "northward_wind")
        },
        coords={
            "valid_time": pd.date_range("2021-01-01", periods=6, freq="6h"),
            "level": levels,
            "latitude": np.linspace(20.0, 60.0, 12),
            "longitude": np.linspace(-160.0, -120.0, 16),
        },
    )
    return ds.chunk({"valid_time": 2, "level": -1}) if chunk else ds
