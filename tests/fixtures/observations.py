"""Observed and analysis data, both gridded and station-based."""

import numpy as np
import pandas as pd
import xarray as xr


def make_sample_gridded_obs_dataset():
    time = pd.date_range("2021-06-20", freq="3h", periods=200)
    data = np.random.RandomState(21897820).standard_normal(size=(len(time), 181, 360))
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], 20 + 5 * data),
            "tp": (["time", "latitude", "longitude"], data),
            "10m_u_component_of_wind": (["time", "latitude", "longitude"], data),
            "10m_v_component_of_wind": (["time", "latitude", "longitude"], data),
        },
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["2m_temperature"].loc[
        dict(
            time="2021-06-21 18:00",
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 25
    return dataset


def make_sample_gridded_obs_dataarray():
    """Create a sample gridded observations DataArray."""
    dataset = make_sample_gridded_obs_dataset()
    return dataset["2m_temperature"]


def make_sample_era5_dataset():
    """Create a sample ERA5-like dataset with time dimension."""
    time = pd.date_range("2021-06-20", freq="6h", periods=50)
    data = np.random.RandomState(12345).standard_normal(size=(len(time), 91, 180))
    latitudes = np.linspace(-90, 90, 91)
    longitudes = np.linspace(0, 359, 180)

    dataset = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], 273.15 + 10 * data),
            "mean_sea_level_pressure": (
                ["time", "latitude", "longitude"],
                101325 + 1000 * data,
            ),
        },
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )
    return dataset


def make_unchunked_target_dataset(n_time=40, n_lat=64, n_lon=64):
    """Numpy-backed target dataset, as an unchunked zarr source produces."""
    rng = np.random.default_rng(5)
    return xr.Dataset(
        {
            "surface_air_temperature": (
                ["valid_time", "latitude", "longitude"],
                rng.uniform(280.0, 310.0, (n_time, n_lat, n_lon)),
            )
        },
        coords={
            "valid_time": pd.date_range("2021-06-20", periods=n_time, freq="6h"),
            "latitude": np.linspace(30.0, 60.0, n_lat),
            "longitude": np.linspace(-130.0, -100.0, n_lon),
        },
    )


def make_sample_point_obs_df():
    # Create sample point observations DataFrame
    data = {
        "time": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 06:00"]),
        "station": ["A100", "B200"],
        "call": ["KWEW", "KBCE"],
        "name": ["WEST CENTRAL", "EAST CENTRAL"],
        "latitude": [40.5, 41.8],
        "longitude": [-99.5, -99.8],
        "elev": [1000, 1100],
        "id": [1, 2],
        "surface_air_temperature": [20.0, 21.0],
    }
    df = pd.DataFrame(data)
    return df


def make_sample_point_obs_df_with_attrs():
    """Create sample point observations DataFrame with attributes."""
    df = make_sample_point_obs_df()
    df.attrs = {
        "metadata_vars": ["station", "call", "name", "elev", "id"],
    }
    return df


def make_sample_ghcn_dataframe():
    """Create a sample GHCN-like polars DataFrame."""
    import polars as pl

    dates = pd.date_range("2021-06-20", periods=100, freq="6h")
    n_stations = 5

    # Create combinations of stations and times
    station_ids = [f"STATION_{i:03d}" for i in range(n_stations)]

    data = []
    for station_id in station_ids:
        for date in dates:
            lat = 40 + np.random.normal(0, 5)
            lon = -100 + np.random.normal(0, 10)
            temp = 273.15 + np.random.normal(20, 5)

            data.append(
                {
                    "valid_time": date,
                    "station_id": station_id,
                    "latitude": lat,
                    "longitude": lon,
                    "surface_air_temperature": temp,
                }
            )

    return pl.DataFrame(data)


def make_sample_lsr_dataframe():
    """Create a sample Local Storm Report DataFrame."""
    data = {
        "valid_time": pd.date_range("2021-06-20", periods=20, freq="1h"),
        "latitude": np.random.uniform(30, 50, 20),
        "longitude": np.random.uniform(-110, -90, 20),
        "report_type": np.random.choice(["wind", "hail", "tor"], 20),
        "magnitude": np.random.uniform(0, 100, 20),
    }
    return pd.DataFrame(data)
