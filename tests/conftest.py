import pytest
import xarray as xr
import pandas as pd
import numpy as np
from extremeweatherbench import config, events


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


def make_sample_forecast_dataset():
    init_time = pd.date_range("2021-06-20", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.RandomState(21897820).standard_normal(
        size=(len(init_time), 181, 360, len(lead_time)),
    )
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                20 + 5 * data,
            ),
            "surface_eastward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "surface_northward_wind": (
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
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-21 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 24
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-20 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 23
    return dataset


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]


@pytest.fixture
def sample_forecast_dataset():
    sample_forecast_dataset = make_sample_forecast_dataset()
    return sample_forecast_dataset


@pytest.fixture
def sample_config():
    return config.Config(
        event_types=[events.HeatWave],
        forecast_dir="test/forecast/path",
        gridded_obs_path="test/obs/path",
    )


@pytest.fixture
def default_forecast_config():
    return config.ForecastSchemaConfig()


@pytest.fixture
def sample_gridded_obs_dataset():
    sample_gridded_obs_dataset = make_sample_gridded_obs_dataset()
    return sample_gridded_obs_dataset


@pytest.fixture
def sample_forecast_dataarray():
    sample_forecast_dataarray = dataset_to_dataarray(make_sample_forecast_dataset())
    return sample_forecast_dataarray


@pytest.fixture
def sample_gridded_obs_dataarray():
    sample_gridded_obs_dataarray = dataset_to_dataarray(
        make_sample_gridded_obs_dataset()
    )
    return sample_gridded_obs_dataarray


@pytest.fixture
def sample_subset_forecast_dataarray():
    sample_forecast_dataset = dataset_to_dataarray(make_sample_forecast_dataset())
    subset_sample_forecast_dataset = sample_forecast_dataset.sel(
        latitude=slice(40, 45), longitude=slice(100, 105)
    )
    return subset_sample_forecast_dataset


@pytest.fixture
def sample_subset_gridded_obs_dataarray():
    sample_gridded_obs_dataarray = dataset_to_dataarray(
        make_sample_gridded_obs_dataset()
    )
    subset_sample_gridded_obs_dataarray = sample_gridded_obs_dataarray.sel(
        latitude=slice(40, 45), longitude=slice(100, 105)
    )
    return subset_sample_gridded_obs_dataarray


@pytest.fixture
def sample_results_dataarray_list():
    results_da_list = [
        xr.DataArray(
            data=[5],
            dims=["lead_time"],
            coords={"lead_time": [0]},
        ),
        xr.DataArray(
            data=[6],
            dims=["lead_time"],
            coords={"lead_time": [6]},
        ),
    ]
    return results_da_list


@pytest.fixture
def sample_point_obs_df():
    # Create sample point observations DataFrame
    data = {
        "time": ["2023-01-01 00:00", "2023-01-01 06:00"],
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
