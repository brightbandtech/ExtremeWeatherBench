import pytest
import xarray as xr
import pandas as pd
import numpy as np
from extremeweatherbench import config, events


@pytest.fixture
def mock_forecast_dataset():
    init_time = pd.date_range("2020-01-01", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.rand(len(init_time), 180, 360, len(lead_time))
    latitudes = np.linspace(-90, 90, 180)
    longitudes = np.linspace(-180, 179, 360)
    dataset = xr.Dataset(
        {
            "t2": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "eastward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "northward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
        },
        coords={
            "init_time": init_time,
            "latitude": latitudes,
            "longitude": longitudes,
            "lead_time": lead_time,
        },
    )
    lead_time_grid, init_time_grid = np.meshgrid(dataset.lead_time, dataset.init_time)
    # Step 2: Flatten the meshgrid and convert lead_time to timedelta
    valid_time = init_time_grid.flatten() + pd.to_timedelta(
        lead_time_grid.flatten(), unit="h"
    )
    dataset.coords["time"] = valid_time
    return dataset


@pytest.fixture
def mock_config():
    return config.Config(
        event_types=[events.HeatWave],
        forecast_dir="test/forecast/path",
        gridded_obs_path="test/obs/path",
    )


@pytest.fixture
def mock_gridded_obs_dataset():
    time = pd.date_range("2020-01-01", freq="3h", periods=200)
    data = np.random.rand(len(time), 180, 360)
    latitudes = np.linspace(-90, 90, 180)
    longitudes = np.linspace(-180, 179, 360)
    dataset = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], data),
            "tp": (["time", "latitude", "longitude"], data),
            "10m_u_component_of_wind": (["time", "latitude", "longitude"], data),
            "10m_v_component_of_wind": (["time", "latitude", "longitude"], data),
        },
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )
    return dataset
