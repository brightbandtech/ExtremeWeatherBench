import pytest
import xarray as xr
from extremeweatherbench import config, events, case, evaluate, utils
import datetime
import numpy as np
import pandas as pd


def test_evaluate_no_computation(mock_config):
    result = evaluate.evaluate(mock_config, dry_run=True, dry_run_event_type="HeatWave")
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
        event_types=[events.HeatWave],
        forecast_dir="test/path",
        gridded_obs_path=None,
        point_obs_path=None,
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


def test_evaluate_base_case(sample_forecast_dataset, sample_gridded_obs_dataset):
    base_case = case.IndividualCase(
        id=1,
        title="test_case",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 7, 3),
        bounding_box_degrees=500,
        location={"latitude": 45.0, "longitude": -100.0},
        event_type="heat_wave",
    )
    with pytest.raises(NotImplementedError):
        evaluate._evaluate_case(
            individual_case=base_case,
            forecast_dataset=sample_forecast_dataset,
            gridded_obs=sample_gridded_obs_dataset,
            point_obs=None,
        )


def test_evaluate_full_workflow(
    mocker, mock_config, sample_gridded_obs_dataset, sample_forecast_dataset
):
    # The return func will have the forecast dataset's data vars names switched already
    mocker.patch(
        "extremeweatherbench.evaluate._open_forecast_dataset",
        return_value=sample_forecast_dataset,
    )
    mocker.patch(
        "extremeweatherbench.evaluate._open_obs_datasets",
        return_value=(None, sample_gridded_obs_dataset),
    )
    result = evaluate.evaluate(mock_config)
    assert isinstance(result, dict)
    assert "heat_wave" in result
    assert isinstance(result["heat_wave"], dict)
    assert isinstance(result["heat_wave"][1], dict)
    for _, v in result["heat_wave"].items():
        if v is not None:
            assert isinstance(v, dict)  # gridded or point
            for _, v2 in v.items():
                assert isinstance(v2, dict)  # data var
                for _, v3 in v2.items():
                    assert isinstance(v3, dict)  # metric name
                    for _, v4 in v3.items():
                        assert isinstance(v4, xr.DataArray)  # metric value


# Test when no forecast data is available (i.e., empty "init_time").
def test_evaluate_case_no_forecast_data(
    sample_empty_forecast_dataset, sample_individual_case
):
    with pytest.raises(ValueError):
        evaluate._evaluate_case(
            sample_individual_case,
            sample_empty_forecast_dataset,
            None,
            None,
        )


# Test evaluation when gridded observations are provided.
def test_evaluate_case_gridded(sample_individual_case):
    sample_individual_case_heatwave = case.IndividualHeatWaveCase(
        **sample_individual_case.__dict__
    )
    start = sample_individual_case_heatwave.start_date
    end = sample_individual_case_heatwave.end_date
    # Create a forecast dataset with 4 time points.
    times = pd.date_range(start, periods=4)
    forecast_ds = xr.Dataset(
        {
            "surface_air_temperature": xr.DataArray(
                np.arange(16).reshape(4, 4, 1, 1),
                dims=["init_time", "lead_time", "latitude", "longitude"],
            )
        },
        coords={
            "init_time": times,
            "lead_time": np.arange(4),
            "latitude": [45.0],
            "longitude": [260.0],
        },
    )
    # Create a gridded observation dataset with a 'time' coordinate.
    obs_times = pd.date_range(start, end)
    gridded_obs = xr.Dataset(
        {
            "surface_air_temperature": xr.DataArray(
                np.arange(len(obs_times) * 1 * 1).reshape(len(obs_times), 1, 1),
                dims=["time", "latitude", "longitude"],
            )
        },
        coords={"time": obs_times, "latitude": [45.0], "longitude": [260.0]},
    )
    result = evaluate._evaluate_case(
        sample_individual_case_heatwave,
        forecast_ds,
        gridded_obs,
        None,
    )
    assert "gridded" in result
    assert "surface_air_temperature" in result["gridded"]
    assert "MaxOfMinTempMAE" in result["gridded"]["surface_air_temperature"]
    assert isinstance(
        result["gridded"]["surface_air_temperature"]["MaxOfMinTempMAE"], xr.DataArray
    )


# Test evaluation when point observations are provided.
def test_evaluate_case_point(mocker, sample_individual_case, sample_forecast_dataset):
    sample_individual_case_heatwave = case.IndividualHeatWaveCase(
        **sample_individual_case.__dict__
    )
    # Create a minimal point observations DataFrame.
    point_obs = pd.DataFrame(
        {
            "id": [
                1,
            ],
            "latitude": [45.0],
            "longitude": [260.0],
            "station": ["A"],
            "elev": [100],
            "name": ["Station_A"],
            "call": ["X"],
            "time": [sample_individual_case_heatwave.start_date],
            "surface_air_temperature": [10],
        }
    )
    mocker.patch(
        "extremeweatherbench.metrics.MaximumMAE.compute",
        return_value=xr.DataArray(42),
    )
    mocker.patch(
        "extremeweatherbench.metrics.MaxOfMinTempMAE.compute",
        return_value=xr.DataArray(42),
    )
    mocker.patch(
        "extremeweatherbench.metrics.RegionalRMSE.compute",
        return_value=xr.DataArray(42),
    )
    # Ensure ISD_MAPPING exists.
    if not hasattr(utils, "ISD_MAPPING"):
        utils.ISD_MAPPING = {}
    result = evaluate._evaluate_case(
        sample_individual_case_heatwave,
        sample_forecast_dataset,
        None,
        point_obs,
    )
    assert "point" in result
    assert "surface_air_temperature" in result["point"]
    assert "RegionalRMSE" in result["point"]["surface_air_temperature"]
    assert isinstance(
        result["point"]["surface_air_temperature"]["RegionalRMSE"], xr.DataArray
    )
