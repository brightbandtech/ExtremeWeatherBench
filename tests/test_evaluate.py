import pytest
from extremeweatherbench import events, case, evaluate
import datetime
import xarray as xr
import numpy as np
import pandas as pd


def test_evaluate_no_computation(sample_config):
    result = evaluate.evaluate(
        sample_config, dry_run=True, dry_run_event_type="HeatWave"
    )
    assert isinstance(result, events.EventContainer)


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
    mock_case.data_vars = ["surface_air_temperature"]
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)

    # Setup the return value for _subset_valid_times
    time_subset_mock = mocker.MagicMock(spec=xr.Dataset)
    mock_case._subset_valid_times.return_value = time_subset_mock

    # Setup the return value for perform_subsetting_procedure
    spatial_subset_mock = mocker.MagicMock(spec=xr.Dataset)
    mock_case.perform_subsetting_procedure.return_value = spatial_subset_mock

    # Create a mock for the data variable subset
    var_subset_mock = mocker.MagicMock(spec=xr.Dataset)
    var_subset_mock.init_time = range(10)
    spatial_subset_mock.__getitem__.return_value = var_subset_mock

    # Create the forecast mock
    mock_forecast = mocker.MagicMock(spec=xr.Dataset)
    mock_forecast.lead_time = range(5)
    mock_forecast.init_time = range(5)

    # Create the case evaluation data
    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=mock_forecast,
        observation=mocker.MagicMock(spec=xr.Dataset),
    )

    result = evaluate._check_and_subset_forecast_availability(case_eval_data)

    # Verify the function calls and result
    mock_case._subset_valid_times.assert_called_once_with(mock_forecast)
    mock_case.perform_subsetting_procedure.assert_called_once_with(time_subset_mock)
    spatial_subset_mock.__getitem__.assert_called_once_with(mock_case.data_vars)
    assert result is var_subset_mock


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


def test_build_dataset_subsets_with_existing_forecast(
    mocker, sample_gridded_obs_dataset, sample_forecast_dataset
):
    """Test build_dataset_subsets with an existing forecast dataset."""
    # Create a mock case
    mock_case = mocker.MagicMock(spec=case.IndividualCase)
    mock_case.id = "test_case"
    mock_case.data_vars = "2m_temperature"
    mock_case.start_date = datetime.datetime(2021, 6, 20)
    mock_case.end_date = datetime.datetime(2021, 6, 25)

    # Create a mock existing forecast dataset
    existing_forecast = sample_forecast_dataset.copy(deep=True)

    # Create the case evaluation data
    case_eval_data = evaluate.CaseEvaluationData(
        individual_case=mock_case,
        observation_type="gridded",
        forecast=sample_forecast_dataset,  # This should be replaced by existing_forecast
        observation=sample_gridded_obs_dataset,
    )

    # Mock _subset_gridded_obs to return a CaseEvaluationInput
    mock_result = evaluate.CaseEvaluationInput(
        observation_type="gridded",
        observation=sample_gridded_obs_dataset["2m_temperature"],
        forecast=existing_forecast["surface_air_temperature"],
    )
    mocker.patch(
        "extremeweatherbench.evaluate._subset_gridded_obs", return_value=mock_result
    )

    # Spy on _check_and_subset_forecast_availability to verify it's not called
    check_forecast_spy = mocker.patch(
        "extremeweatherbench.evaluate._check_and_subset_forecast_availability"
    )

    # Call the function with existing_forecast
    result = evaluate.build_dataset_subsets(
        case_eval_data, compute=False, existing_forecast=existing_forecast
    )

    # Verify that _check_and_subset_forecast_availability was not called
    check_forecast_spy.assert_not_called()

    # Verify that the forecast in case_eval_data was replaced with existing_forecast
    assert case_eval_data.forecast is existing_forecast

    # Verify the result
    assert result.observation_type == "gridded"
    assert result.observation is not None
    assert result.forecast is not None


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

    result = evaluate.build_dataset_subsets(case_eval_data)

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

    result = evaluate.build_dataset_subsets(case_eval_data, compute=False)

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

    result = evaluate.build_dataset_subsets(case_eval_data, compute=False)

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
    mocker, sample_config, sample_gridded_obs_dataset, sample_forecast_dataset
):
    # The return func will have the forecast dataset's data vars names switched already
    mocker.patch(
        "extremeweatherbench.data_loader.open_and_preprocess_forecast_dataset",
        return_value=sample_forecast_dataset,
    )
    mocker.patch(
        "extremeweatherbench.data_loader.open_obs_datasets",
        return_value=(None, sample_gridded_obs_dataset),
    )
    result = evaluate.evaluate(sample_config)
    assert isinstance(result, pd.DataFrame)
    # Check that the result DataFrame contains all the expected columns
    expected_columns = [
        "lead_time",
        "value",
        "variable",
        "metric",
        "observation_type",
        "case_id",
        "event_type",
    ]
    assert all(col in result.columns for col in expected_columns)
