import pytest
import xarray as xr
import pandas as pd
import numpy as np
from extremeweatherbench import config, events


@pytest.fixture
def mock_forecast_dataset():
    init_time = pd.date_range("2021-06-20", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.RandomState(21897820).standard_normal(
        size=(len(init_time), 181, 360, len(lead_time)),
    )
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                20 + 5 * data,
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
    dataset["2m_temperature"].loc[
        dict(
            time="2021-06-21 18:00",
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 25
    return dataset


@pytest.fixture
def mock_forecast_dataarray(mock_forecast_dataset):
    return dataset_to_dataarray(mock_forecast_dataset)


@pytest.fixture
def mock_subset_forecast_dataarray(mock_forecast_dataset):
    return dataset_to_dataarray(mock_forecast_dataset).sel(
        latitude=slice(40, 45), longitude=slice(100, 105)
    )


@pytest.fixture
def mock_gridded_obs_dataarray(mock_gridded_obs_dataset):
    return dataset_to_dataarray(mock_gridded_obs_dataset)


@pytest.fixture
def mock_subset_gridded_obs_dataarray(mock_gridded_obs_dataset):
    return dataset_to_dataarray(mock_gridded_obs_dataset).sel(
        latitude=slice(40, 45), longitude=slice(100, 105)
    )


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]
