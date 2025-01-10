import pytest
import xarray as xr
import pandas as pd
import datetime
from extremeweatherbench import config, events, case, evaluate
from pathlib import Path
from .test_datasets import mock_forecast_dataset


@pytest.fixture
def mock_gridded_obs():
    return xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], [[[18.0]]]),
            "tp": (["time", "latitude", "longitude"], [[[0.8]]]),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=1),
            "latitude": [45.0],
            "longitude": [10.0],
        },
    )


@pytest.fixture
def mock_config():
    return config.Config(
        event_types=[events.HeatWave],
        forecast_dir="test/forecast/path",
        gridded_obs_path="test/obs/path",
    )


def test_evaluate_no_computation(mock_config):
    result = evaluate.evaluate(mock_config, no_computation=True)
    assert isinstance(result, events.EventContainer)


def test_open_forecast_dataset_invalid_path():
    invalid_config = config.Config(
        event_types=[events.HeatWave],
        forecast_dir="invalid/path",
        gridded_obs_path="test/path",
    )
    with pytest.raises(FileNotFoundError):
        evaluate._open_forecast_dataset(invalid_config)


def test_open_obs_datasets_no_obs_paths():
    invalid_config = config.Config(
        event_types=[events.HeatWave], forecast_dir="test/path", gridded_obs_path=None
    )
    with pytest.raises(
        ValueError, match="No gridded or point observation data provided"
    ):
        evaluate._open_obs_datasets(invalid_config)


def test_open_obs_datasets_no_forecast_paths():
    invalid_config = config.Config(event_types=[events.HeatWave], forecast_dir=None)
    with pytest.raises(
        AttributeError, match="'NoneType' object has no attribute 'startswith'"
    ):
        evaluate._open_forecast_dataset(invalid_config)


def test_evaluate_case(mock_forecast_dataset, mock_gridded_obs):
    test_case = case.IndividualCase(
        id=1,
        title="test_case",
        start_date=pd.Timestamp(2020, 1, 1),
        end_date=pd.Timestamp(2020, 1, 12),
        bounding_box_km=500,
        location={"latitude": 45.0, "longitude": -100.0},
        event_type="heat_wave",
    )
    result = evaluate._evaluate_case(
        test_case, mock_forecast_dataset, mock_gridded_obs, None
    )
    assert isinstance(result, xr.Dataset)
