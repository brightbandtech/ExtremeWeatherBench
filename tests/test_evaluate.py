import pytest
from extremeweatherbench import config, events, case, evaluate
import datetime
import xarray as xr
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


def test_evaluate_individualcase(sample_forecast_dataset, sample_gridded_obs_dataset):
    base_case = case.IndividualCase(
        id=1,
        title="test_case",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 7, 3),
        bounding_box_degrees=500,
        location={"latitude": 45.0, "longitude": -100.0},
        event_type="heat_wave",
        data_vars=["2m_temperature"],
    )
    with pytest.raises(NotImplementedError):
        evaluate._maybe_evaluate_individual_case(
            individual_case=base_case,
            forecast_dataset=sample_forecast_dataset,
            gridded_obs=sample_gridded_obs_dataset,
            point_obs=None,
        )


def test_case_evaluation_input_init():
    """Test initialization of CaseEvaluationInput."""
    eval_input = evaluate.CaseEvaluationInput(observation_type="gridded")
    assert eval_input.observation_type == "gridded"
    assert eval_input.observation is None
    assert eval_input.forecast is None


def test_case_evaluation_input_load_data(
    sample_gridded_obs_dataarray, sample_forecast_dataarray
):
    """Test load_data method of CaseEvaluationInput."""
    # Create lazy arrays
    lazy_obs = sample_gridded_obs_dataarray.chunk()
    lazy_forecast = sample_forecast_dataarray.chunk()

    eval_input = evaluate.CaseEvaluationInput(
        observation_type="gridded", observation=lazy_obs, forecast=lazy_forecast
    )

    # Verify arrays are lazy before load_data
    assert isinstance(eval_input.observation.data, type(lazy_obs.data))
    assert isinstance(eval_input.forecast.data, type(lazy_forecast.data))

    # Call load_data
    eval_input.load_data()

    # Verify arrays are now in memory
    assert isinstance(eval_input.observation.data, np.ndarray)
    assert isinstance(eval_input.forecast.data, np.ndarray)


def test_case_evaluation_data_init(mocker):
    """Test initialization of CaseEvaluationData."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.data_vars = "2m_temperature"
    mock_forecast = mocker.MagicMock(spec=xr.Dataset)
    mock_obs = mocker.MagicMock(spec=xr.Dataset)

    eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=mock_forecast,
        observation=mock_obs,
    )

    assert eval_data.individual_case == mock_case
    assert eval_data.observation_type == "gridded"
    assert eval_data.forecast == mock_forecast
    assert eval_data.observation == mock_obs


def test_check_and_subset_forecast_availability(mocker):
    """Test _check_and_subset_forecast_availability function."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = "99"
    mock_case._subset_valid_times.return_value = mocker.MagicMock(spec=xr.Dataset)
    mock_case._subset_valid_times.return_value.init_time = range(10)
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)

    mock_forecast = mocker.MagicMock(spec=xr.Dataset)
    mock_forecast.lead_time = range(5)
    mock_forecast.init_time = range(5)

    # Create a mock self object with the required attributes
    mock_self = mocker.MagicMock()
    mock_self.individual_case = mock_case
    mock_self.forecast = mock_forecast

    result = evaluate._check_and_subset_forecast_availability(mock_self)
    assert result is not None


def test_check_and_subset_forecast_availability_empty(mocker):
    """Test _check_and_subset_forecast_availability function with empty forecast."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = "test_case"
    mock_case._subset_valid_times.return_value = mocker.MagicMock(spec=xr.Dataset)
    mock_case._subset_valid_times.return_value.init_time = []
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)

    mock_forecast = mocker.MagicMock(spec=xr.Dataset)
    mock_forecast.lead_time = range(5)
    mock_forecast.init_time = range(5)

    # Create a mock self object with the required attributes
    mock_self = mocker.MagicMock()
    mock_self.individual_case = mock_case
    mock_self.forecast = mock_forecast

    result = evaluate._check_and_subset_forecast_availability(mock_self)
    assert result is None


def test_build_dataarray_subsets_no_forecast(mocker):
    """Test build_dataarray_subsets with no forecast data."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = "test_case"
    mock_case.data_vars = "2m_temperature"

    mock_forecast = mocker.MagicMock(spec=xr.Dataset)

    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=mock_forecast,
        observation=None,
    )

    # Mock _check_and_subset_forecast_availability to return None
    mocker.patch(
        "extremeweatherbench.evaluate._check_and_subset_forecast_availability",
        return_value=None,
    )

    result = evaluate.build_dataarray_subsets(case_eval_data)

    assert result.observation_type == "gridded"
    assert result.observation is None
    assert result.forecast is None


def test_build_dataarray_subsets_gridded(
    mocker, sample_gridded_obs_dataset, sample_forecast_dataset
):
    """Test build_dataarray_subsets with gridded observation data."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = "test_case"
    mock_case.data_vars = "2m_temperature"
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)
    mock_case.perform_subsetting_procedure.return_value = sample_gridded_obs_dataset[
        "2m_temperature"
    ]

    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=sample_forecast_dataset,
        observation=sample_gridded_obs_dataset,
    )

    # Mock _check_and_subset_forecast_availability to return the forecast dataset
    mocker.patch(
        "extremeweatherbench.evaluate._check_and_subset_forecast_availability",
        return_value=sample_forecast_dataset,
    )

    # Mock _subset_gridded_obs to return a CaseEvaluationInput
    mock_result = evaluate.CaseEvaluationInput(
        observation_type="gridded",
        observation=sample_gridded_obs_dataset["2m_temperature"],
        forecast=sample_forecast_dataset["surface_air_temperature"],
    )
    mocker.patch(
        "extremeweatherbench.evaluate._subset_gridded_obs", return_value=mock_result
    )

    result = evaluate.build_dataarray_subsets(case_eval_data, compute=False)

    assert result.observation_type == "gridded"
    assert result.observation is not None
    assert result.forecast is not None


def test_build_dataarray_subsets_point(
    mocker, sample_point_obs_df, sample_forecast_dataset
):
    """Test build_dataarray_subsets with point observation data."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = 1  # Match the ID in sample_point_obs_df
    mock_case.data_vars = "surface_air_temperature"

    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="point",
        forecast=sample_forecast_dataset,
        observation=sample_point_obs_df,
    )

    # Mock _check_and_subset_forecast_availability to return the forecast dataset
    mocker.patch(
        "extremeweatherbench.evaluate._check_and_subset_forecast_availability",
        return_value=sample_forecast_dataset,
    )

    # Mock _subset_point_obs to return a CaseEvaluationInput
    mock_obs = xr.DataArray(
        data=np.array([20.0]), dims=["time"], coords={"time": ["2023-01-01"]}
    )
    mock_forecast = xr.DataArray(
        data=np.array([21.0]), dims=["init_time"], coords={"init_time": ["2023-01-01"]}
    )
    mock_result = evaluate.CaseEvaluationInput(
        observation_type="point", observation=mock_obs, forecast=mock_forecast
    )
    mocker.patch(
        "extremeweatherbench.evaluate._subset_point_obs", return_value=mock_result
    )

    result = evaluate.build_dataarray_subsets(case_eval_data, compute=False)

    assert result.observation_type == "point"
    assert result.observation is not None
    assert result.forecast is not None


def test_subset_gridded_obs(
    mocker, sample_gridded_obs_dataset, sample_forecast_dataset
):
    """Test _subset_gridded_obs function."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = 99
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)
    mock_case.data_vars = "2m_temperature"
    mock_case.perform_subsetting_procedure.return_value = sample_gridded_obs_dataset[
        "2m_temperature"
    ]

    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=sample_forecast_dataset,
        observation=sample_gridded_obs_dataset,
    )

    result = evaluate._subset_gridded_obs(case_eval_data)

    assert isinstance(result, evaluate.CaseEvaluationInput)
    assert result.observation_type == "gridded"
    assert isinstance(result.observation, xr.DataArray)
    assert isinstance(result.forecast, xr.Dataset)


def test_subset_point_obs(mocker, sample_point_obs_df, sample_forecast_dataset):
    """Test _subset_point_obs function."""
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = 1
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)
    mock_case.data_vars = ["surface_air_temperature"]
    valid_time = pd.Timestamp(
        sample_forecast_dataset.init_time[0].values
    ) + pd.Timedelta(hours=6)
    sample_point_obs_df.iloc[0, sample_point_obs_df.columns.get_loc("time")] = (
        valid_time
    )
    sample_point_obs_df.iloc[1, sample_point_obs_df.columns.get_loc("time")] = (
        valid_time
    )
    # mocker.patch(
    #     'point_forecast_da.groupby(["init_time", "lead_time", "latitude", "longitude"]).mean()',
    #     return_value=sample_forecast_dataset,
    # )
    # mocker.patch(
    #     'subset_point_obs_da.groupby(["time", "latitude", "longitude"]).first()',
    #     return_value=sample_point_obs_df,
    # )
    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="point",
        forecast=sample_forecast_dataset,
        observation=sample_point_obs_df,
    )

    result = evaluate._subset_point_obs(case_eval_data)

    assert isinstance(result, evaluate.CaseEvaluationInput)
    assert result.observation_type == "point"
    assert result.observation is not None
    assert result.forecast is not None


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
    for _, v in result["heat_wave"].items():
        if v is not None:
            assert isinstance(v, dict)
            for _, v2 in v.items():
                assert isinstance(v2, dict)
                for _, v3 in v2.items():
                    assert isinstance(v3, dict) or v3 is None
