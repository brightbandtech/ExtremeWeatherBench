"""Cyclone tracks, tracker inputs, and landfall-shaped arrays."""

import numpy as np
import pandas as pd
import xarray as xr


def make_gulf_coast_track(n_points):
    """Track running north out of the Gulf of Mexico onto Louisiana."""
    lons = np.linspace(-91.5, -91.0, n_points)
    lats = np.linspace(24.0, 32.0, n_points)
    return lons, lats


def make_track_dataarray(lons, lats, chunk=False):
    """Track-shaped DataArray with latitude/longitude along valid_time."""
    valid_time = pd.date_range("2021-08-28", periods=len(lons), freq="6h")
    da = xr.DataArray(
        np.arange(float(len(lons))),
        dims=["valid_time"],
        coords={
            "valid_time": valid_time,
            "latitude": ("valid_time", np.asarray(lats, dtype=float)),
            "longitude": ("valid_time", np.asarray(lons, dtype=float)),
        },
    )
    return da.chunk({"valid_time": 4}) if chunk else da


def make_landfall_dataarray(n_init=64, chunk=True, all_nan=False):
    """Landfall-shaped DataArray indexed by init_time."""
    values = np.full(n_init, np.nan) if all_nan else np.arange(float(n_init))
    da = xr.DataArray(
        values,
        dims=["init_time"],
        coords={"init_time": pd.date_range("2021-08-20", periods=n_init, freq="6h")},
    )
    return da.chunk({"init_time": 8}) if chunk else da


def make_tc_track_frame(n_rows=4, latitude=15.0, longitude=140.0):
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2021-08-01", periods=n_rows, freq="6h"),
            "latitude": np.full(n_rows, latitude),
            "longitude": np.full(n_rows, longitude),
        }
    )


def make_tc_track_target_dataset(n_time=4):
    """Observed-track dataset used to filter forecast candidates."""
    return xr.Dataset(
        {"intensity": ("valid_time", np.arange(float(n_time)))},
        coords={
            "valid_time": pd.date_range("2021-09-01", periods=n_time, freq="6h"),
            "latitude": ("valid_time", np.linspace(15.0, 25.0, n_time)),
            "longitude": ("valid_time", np.linspace(-75.0, -65.0, n_time)),
        },
    )


def make_tc_tracker_forecast_dataset(n_time=4, seed=0):
    """Minimal forecast fields the TC tracker asks for."""
    rng = np.random.default_rng(seed)
    dims = ["valid_time", "latitude", "longitude"]
    shape = (n_time, 8, 10)
    return xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (dims, rng.random(shape) + 1000.0),
            "surface_wind_speed": (dims, rng.random(shape) * 30.0),
        },
        coords={
            "valid_time": pd.date_range("2021-09-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(10.0, 30.0, 8),
            "longitude": np.linspace(-80.0, -60.0, 10),
        },
    )


def make_single_init_tc_dataset(n_lead: int = 12, n_wind_strong: int = 10):
    """Build a minimal synthetic TC dataset with one clear init_time.

    Grid: 1° resolution, 21×21 points (lat 10–30°, lon -80 to -60°).
    Storm: SLP=98 000 Pa at centre (lat=20, lon=-70) every (lead, valid) pair
    on the diagonal (same init_time T0).
    Wind: 20 m/s at one gridpoint east of centre for the first
    ``n_wind_strong`` diagonal pairs; 2 m/s everywhere else.
    Contour validation is OFF so only the wind filter is exercised.

    With a 1° grid ``_degrees_to_gridpoints(2.0, ...)`` = 2 gridpoints, so
    the ±2-pt neighbourhood around the centre includes the +1-pt east cell.

    Expected: n_wind_strong detections have neighbourhood wind ≥ 10 m/s.
    """
    lat = np.arange(10.0, 31.0, 1.0)  # 21 pts, 1 ° spacing
    lon = np.arange(-80.0, -59.0, 1.0)  # 21 pts, 1 ° spacing
    n_lat, n_lon = len(lat), len(lon)
    c_lat, c_lon = 10, 10  # centre indices → lat=20°, lon=-70°

    T0 = pd.Timestamp("2023-09-10")
    lead_h = np.arange(n_lead) * 6  # hours
    lead_td = (lead_h * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    valid_times = pd.date_range(T0, periods=n_lead, freq="6h")

    # init_time[lt, vt] = valid_time[vt] - lead_time[lt]
    init_2d = np.array(
        [
            [valid_times[vt].to_datetime64() - lead_td[lt] for vt in range(n_lead)]
            for lt in range(n_lead)
        ]
    )

    # SLP: 98 000 Pa at centre for every (lt, vt) pair; 102 000 elsewhere
    slp = np.full((n_lead, n_lead, n_lat, n_lon), 102000.0)
    for k in range(n_lead):
        slp[k, k, c_lat, c_lon] = 98000.0

    # Wind: 20 m/s east of centre for the first n_wind_strong diagonal pairs
    wind = np.full((n_lead, n_lead, n_lat, n_lon), 2.0)
    for k in range(n_wind_strong):
        wind[k, k, c_lat, c_lon + 1] = 20.0

    # Geopotential thickness: zeros (contour validation disabled)
    dz = np.zeros((n_lead, n_lead, n_lat, n_lon))

    ds = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                slp,
            ),
            "surface_wind_speed": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                wind,
            ),
            "geopotential_thickness": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                dz,
            ),
        },
        coords={
            "lead_time": lead_td,
            "valid_time": valid_times,
            "latitude": lat,
            "longitude": lon,
            "init_time": (["lead_time", "valid_time"], init_2d),
        },
    )
    return ds


def make_ibtracs_frame_for_dataset(ds: xr.Dataset) -> xr.Dataset:
    """IBTrACS stub matching the storm centre in ``make_single_init_tc_dataset``."""
    valid_times = ds.valid_time.values
    return xr.Dataset(
        {
            "latitude": (["valid_time"], np.full(len(valid_times), 20.0)),
            "longitude": (["valid_time"], np.full(len(valid_times), -70.0)),
        },
        coords={"valid_time": valid_times},
    )


def make_sample_ibtracs_dataframe():
    """Create a sample IBTrACS-like polars DataFrame."""
    import polars as pl

    data = {
        "valid_time": [
            "2021-06-20 00:00:00",
            "2021-06-20 06:00:00",
            "2021-06-20 12:00:00",
        ],
        "tc_name": ["TESTCYCLONE", "TESTCYCLONE", "TESTCYCLONE"],
        "latitude": [25.0, 26.0, 27.0],
        "longitude": [280.0, 281.0, 282.0],
        "surface_wind_speed": [30.0, 35.0, 40.0],
        "air_pressure_at_mean_sea_level": [1010.0, 1005.0, 1000.0],
    }
    return pl.DataFrame(data)
