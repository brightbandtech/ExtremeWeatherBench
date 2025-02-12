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


def test_evaluate_base_case(
    sample_forecast_dataset, mock_config, sample_gridded_obs_dataset
):
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
            eval_config=mock_config,
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
