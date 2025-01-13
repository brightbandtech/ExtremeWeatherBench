import pytest
import xarray as xr
from extremeweatherbench import config, events, case, evaluate
import datetime


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


def test_evaluate_base_case(mock_forecast_dataset, mock_gridded_obs_dataset):
    base_case = case.IndividualCase(
        id=1,
        title="test_case",
        start_date=datetime.datetime(2020, 1, 1),
        end_date=datetime.datetime(2020, 1, 12),
        bounding_box_km=500,
        location={"latitude": 45.0, "longitude": -100.0},
        event_type="heat_wave",
    )
    with pytest.raises(NotImplementedError):
        evaluate._evaluate_case(
            individual_case=base_case,
            forecast_dataset=mock_forecast_dataset,
            gridded_obs=mock_gridded_obs_dataset,
            point_obs=None,
        )


def test_evaluate_full_workflow(
    mocker, mock_config, mock_gridded_obs_dataset, mock_forecast_dataset
):
    # The return func will have the forecast dataset's data vars names switched already
    mocker.patch(
        "extremeweatherbench.evaluate._open_forecast_dataset",
        return_value=mock_forecast_dataset,
    )
    mocker.patch(
        "extremeweatherbench.evaluate._open_obs_datasets",
        return_value=(None, mock_gridded_obs_dataset),
    )
    mock_metric = mocker.Mock()
    mock_metric.name = "HeatWave"
    mock_metric.calculate = mocker.Mock(return_value=None)
    mocker.patch("extremeweatherbench.metrics.Metric", return_value=mock_metric)
    result = evaluate.evaluate(mock_config)

    assert isinstance(result, dict)
    assert "HeatWave" in result
    assert isinstance(result["HeatWave"], dict)
    assert all(
        isinstance(ds, xr.Dataset) or ds is None for ds in result["HeatWave"].values()
    )
