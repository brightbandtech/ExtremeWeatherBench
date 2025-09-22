"""Comprehensive test suite for evaluate.py module.

This test suite covers all the main functionality of the ExtremeWeatherBench evaluation
workflow, including the ExtremeWeatherBench class, pipeline functions, and error
handling.
"""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, evaluate, inputs, metrics
from extremeweatherbench.defaults import OUTPUT_COLUMNS
from extremeweatherbench.regions import CenteredRegion


@pytest.fixture
def sample_individual_case():
    """Create a sample IndividualCase for testing."""
    return cases.IndividualCase(
        case_id_number=1,
        title="Test Heat Wave",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 6, 25),
        location=CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=5.0
        ),
        event_type="heat_wave",
    )


@pytest.fixture
def sample_cases_dict(sample_individual_case):
    """Create a sample cases dictionary."""
    return {
        "cases": [
            {
                "case_id_number": 1,
                "title": "Test Heat Wave",
                "start_date": datetime.datetime(2021, 6, 20),
                "end_date": datetime.datetime(2021, 6, 25),
                "location": {
                    "type": "centered_region",
                    "parameters": {
                        "latitude": 45.0,
                        "longitude": -120.0,
                        "bounding_box_degrees": 5.0,
                    },
                },
                "event_type": "heat_wave",
            }
        ]
    }


@pytest.fixture
def mock_target_base():
    """Create a mock TargetBase object."""
    mock_target = Mock(spec=inputs.TargetBase)
    mock_target.name = "MockTarget"
    mock_target.variables = ["2m_temperature"]
    mock_target.open_and_maybe_preprocess_data_from_source.return_value = xr.Dataset()
    mock_target.maybe_map_variable_names.return_value = xr.Dataset()
    mock_target.subset_data_to_case.return_value = xr.Dataset()
    mock_target.maybe_convert_to_dataset.return_value = xr.Dataset()
    mock_target.add_source_to_dataset_attrs.return_value = xr.Dataset(
        attrs={"source": "mock_target"}
    )
    mock_target.maybe_align_forecast_to_target.return_value = (
        xr.Dataset(),
        xr.Dataset(),
    )
    return mock_target


@pytest.fixture
def mock_forecast_base():
    """Create a mock ForecastBase object."""
    mock_forecast = Mock(spec=inputs.ForecastBase)
    mock_forecast.name = "MockForecast"
    mock_forecast.variables = ["surface_air_temperature"]
    mock_forecast.open_and_maybe_preprocess_data_from_source.return_value = xr.Dataset()
    mock_forecast.maybe_map_variable_names.return_value = xr.Dataset()
    mock_forecast.subset_data_to_case.return_value = xr.Dataset()
    mock_forecast.maybe_convert_to_dataset.return_value = xr.Dataset()
    mock_forecast.add_source_to_dataset_attrs.return_value = xr.Dataset(
        attrs={"source": "mock_forecast"}
    )
    return mock_forecast


@pytest.fixture
def mock_base_metric():
    """Create a mock BaseMetric object."""
    mock_metric = Mock(spec=metrics.BaseMetric)
    mock_metric.name = "MockMetric"
    mock_metric.compute_metric.return_value = xr.DataArray(
        data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
    )
    return mock_metric


@pytest.fixture
def sample_evaluation_object(mock_target_base, mock_forecast_base, mock_base_metric):
    """Create a sample EvaluationObject."""
    return inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[mock_base_metric],
        target=mock_target_base,
        forecast=mock_forecast_base,
    )


@pytest.fixture
def sample_case_operator(
    sample_individual_case, mock_target_base, mock_forecast_base, mock_base_metric
):
    """Create a sample CaseOperator."""
    return cases.CaseOperator(
        case_metadata=sample_individual_case,
        metric_list=mock_base_metric,
        target=mock_target_base,
        forecast=mock_forecast_base,
    )


@pytest.fixture
def sample_forecast_dataset():
    """Create a sample forecast dataset."""
    init_time = pd.date_range("2021-06-20", periods=3)
    lead_time = [0, 6, 12]
    latitudes = np.linspace(40, 50, 11)
    longitudes = np.linspace(-125, -115, 11)

    data = np.random.random(
        (len(init_time), len(latitudes), len(longitudes), len(lead_time))
    )

    return xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data + 273.15,
            )
        },
        coords={
            "init_time": init_time,
            "latitude": latitudes,
            "longitude": longitudes,
            "lead_time": lead_time,
        },
        attrs={"source": "test_forecast"},
    )


@pytest.fixture
def sample_target_dataset():
    """Create a sample target dataset."""
    time = pd.date_range("2021-06-20", periods=20, freq="6h")
    latitudes = np.linspace(40, 50, 11)
    longitudes = np.linspace(-125, -115, 11)

    data = np.random.random((len(time), len(latitudes), len(longitudes)))

    return xr.Dataset(
        {
            "2m_temperature": (
                ["time", "latitude", "longitude"],
                data + 273.15,
            )
        },
        coords={
            "time": time,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={"source": "test_target"},
    )


class TestExtremeWeatherBench:
    """Test the ExtremeWeatherBench class."""

    def test_initialization(self, sample_cases_dict, sample_evaluation_object):
        """Test ExtremeWeatherBench initialization."""
        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        assert ewb.cases == sample_cases_dict
        assert ewb.evaluation_objects == [sample_evaluation_object]
        assert ewb.cache_dir is None

    def test_initialization_with_cache_dir(
        self, sample_cases_dict, sample_evaluation_object
    ):
        """Test ExtremeWeatherBench initialization with cache directory."""
        cache_dir = "/tmp/test_cache"
        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
            cache_dir=cache_dir,
        )

        # Cache dir should be converted to Path object
        assert ewb.cache_dir == Path(cache_dir)

    def test_initialization_with_path_cache_dir(
        self, sample_cases_dict, sample_evaluation_object
    ):
        """Test ExtremeWeatherBench initialization with Path cache directory."""
        cache_dir = Path("/tmp/test_cache")
        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
            cache_dir=cache_dir,
        )

        assert ewb.cache_dir == cache_dir

    @patch("extremeweatherbench.cases.build_case_operators")
    def test_case_operators_property(
        self,
        mock_build_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test case_operators property."""
        mock_build_case_operators.return_value = [sample_case_operator]

        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        result = ewb.case_operators

        mock_build_case_operators.assert_called_once_with(
            sample_cases_dict, [sample_evaluation_object]
        )
        assert result == [sample_case_operator]

    @patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_serial(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes serially."""
        # Mock the case operators property
        with patch.object(
            evaluate.ExtremeWeatherBench, "case_operators", new=[sample_case_operator]
        ):
            # Mock _run_case_operators to return a list of DataFrames
            mock_result = [
                pd.DataFrame(
                    {
                        "value": [1.0],
                        "metric": ["MockMetric"],
                        "case_id_number": [1],
                    }
                )
            ]
            mock_run_case_operators.return_value = mock_result

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(parallel=False)

            mock_run_case_operators.assert_called_once_with(
                [sample_case_operator], False, None, None
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_parallel(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes in parallel."""
        with patch.object(
            evaluate.ExtremeWeatherBench, "case_operators", new=[sample_case_operator]
        ):
            mock_result = [
                pd.DataFrame(
                    {
                        "value": [1.0],
                        "metric": ["MockMetric"],
                        "case_id_number": [1],
                    }
                )
            ]
            mock_run_case_operators.return_value = mock_result

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(parallel=True, n_jobs=2)

            mock_run_case_operators.assert_called_once_with(
                [sample_case_operator], True, 2, None
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_with_kwargs(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method passes kwargs correctly."""
        with patch.object(
            evaluate.ExtremeWeatherBench, "case_operators", new=[sample_case_operator]
        ):
            mock_result = [pd.DataFrame({"value": [1.0]})]
            mock_run_case_operators.return_value = mock_result

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(parallel=False, threshold=0.5, pre_compute=True)

            # Check that kwargs were passed through
            call_args = mock_run_case_operators.call_args
            assert call_args[1]["threshold"] == 0.5
            assert call_args[1]["pre_compute"] is True
            assert isinstance(result, pd.DataFrame)

    @patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_empty_results(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
    ):
        """Test the run method handles empty results."""
        with patch.object(evaluate.ExtremeWeatherBench, "case_operators", new=[]):
            mock_run_case_operators.return_value = []

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == OUTPUT_COLUMNS

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_run_with_caching(
        self,
        mock_compute_case_operator,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method with caching enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            with patch.object(
                evaluate.ExtremeWeatherBench,
                "case_operators",
                new=[sample_case_operator],
            ):
                mock_result = pd.DataFrame(
                    {
                        "value": [1.0],
                        "metric": ["MockMetric"],
                        "case_id_number": [1],
                    }
                )

                # Make the mock also perform caching like the real function would
                def mock_compute_with_caching(case_operator, cache_dir_arg, **kwargs):
                    if cache_dir_arg:
                        cache_path = (
                            Path(cache_dir_arg)
                            if isinstance(cache_dir_arg, str)
                            else cache_dir_arg
                        )
                        mock_result.to_pickle(cache_path / "case_results.pkl")
                    return mock_result

                mock_compute_case_operator.side_effect = mock_compute_with_caching

                ewb = evaluate.ExtremeWeatherBench(
                    cases=sample_cases_dict,
                    evaluation_objects=[sample_evaluation_object],
                    cache_dir=cache_dir,
                )

                ewb.run()

                # Check that cache directory was created
                assert cache_dir.exists()

                # Check that results were cached
                cache_file = cache_dir / "case_results.pkl"
                assert cache_file.exists()

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_run_multiple_cases(
        self, mock_compute_case_operator, sample_cases_dict, sample_evaluation_object
    ):
        """Test the run method with multiple case operators."""
        # Create multiple case operators
        case_operator_1 = Mock()
        case_operator_2 = Mock()

        with patch.object(
            evaluate.ExtremeWeatherBench,
            "case_operators",
            new=[case_operator_1, case_operator_2],
        ):
            # Mock compute_case_operator to return different DataFrames
            mock_compute_case_operator.side_effect = [
                pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
                pd.DataFrame({"value": [2.0], "case_id_number": [2]}),
            ]

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

            assert mock_compute_case_operator.call_count == 2
            assert len(result) == 2
            assert result["case_id_number"].tolist() == [1, 2]


class TestRunCaseOperators:
    """Test the _run_case_operators function."""

    @patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_serial(self, mock_run_serial, sample_case_operator):
        """Test _run_case_operators routes to serial execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_serial.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator], parallel=False, n_jobs=None
        )

        mock_run_serial.assert_called_once_with([sample_case_operator], None)
        assert result == mock_results

    @patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel(self, mock_run_parallel, sample_case_operator):
        """Test _run_case_operators routes to parallel execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_parallel.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator], parallel=True, n_jobs=4
        )

        mock_run_parallel.assert_called_once_with([sample_case_operator], 4)
        assert result == mock_results

    @patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_with_kwargs(
        self, mock_run_serial, sample_case_operator
    ):
        """Test _run_case_operators passes kwargs correctly."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_serial.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator],
            parallel=False,
            n_jobs=None,
            threshold=0.5,
            pre_compute=True,
        )

        call_args = mock_run_serial.call_args
        assert call_args[0][0] == [sample_case_operator]
        assert call_args[1]["threshold"] == 0.5
        assert call_args[1]["pre_compute"] is True
        assert isinstance(result, list)

    @patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel_with_kwargs(
        self, mock_run_parallel, sample_case_operator
    ):
        """Test _run_case_operators passes kwargs to parallel execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_parallel.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator], parallel=True, n_jobs=2, custom_param="test_value"
        )

        call_args = mock_run_parallel.call_args
        assert call_args[0][0] == [sample_case_operator]
        assert call_args[0][1] == 2  # n_jobs
        assert call_args[1]["custom_param"] == "test_value"
        assert isinstance(result, list)

    def test_run_case_operators_empty_list(self):
        """Test _run_case_operators with empty case operator list."""
        with patch("extremeweatherbench.evaluate._run_serial") as mock_serial:
            mock_serial.return_value = []

            result = evaluate._run_case_operators([], parallel=False, n_jobs=None)

            mock_serial.assert_called_once_with([], None)
            assert result == []


class TestRunSerial:
    """Test the _run_serial function."""

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    @patch("tqdm.auto.tqdm")
    def test_run_serial_basic(
        self, mock_tqdm, mock_compute_case_operator, sample_case_operator
    ):
        """Test basic _run_serial functionality."""
        # Setup mocks
        mock_tqdm.return_value = [sample_case_operator]  # tqdm returns iterable
        mock_result = pd.DataFrame({"value": [1.0], "case_id_number": [1]})
        mock_compute_case_operator.return_value = mock_result

        result = evaluate._run_serial([sample_case_operator])

        mock_compute_case_operator.assert_called_once_with(sample_case_operator, None)
        assert len(result) == 1
        assert result[0].equals(mock_result)

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    @patch("tqdm.auto.tqdm")
    def test_run_serial_multiple_cases(self, mock_tqdm, mock_compute_case_operator):
        """Test _run_serial with multiple case operators."""
        case_op_1 = Mock()
        case_op_2 = Mock()
        case_operators = [case_op_1, case_op_2]

        mock_tqdm.return_value = case_operators
        mock_compute_case_operator.side_effect = [
            pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
            pd.DataFrame({"value": [2.0], "case_id_number": [2]}),
        ]

        result = evaluate._run_serial(case_operators)

        assert mock_compute_case_operator.call_count == 2
        assert len(result) == 2
        assert result[0]["case_id_number"].iloc[0] == 1
        assert result[1]["case_id_number"].iloc[0] == 2

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    @patch("tqdm.auto.tqdm")
    def test_run_serial_with_kwargs(
        self, mock_tqdm, mock_compute_case_operator, sample_case_operator
    ):
        """Test _run_serial passes kwargs to compute_case_operator."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_result = pd.DataFrame({"value": [1.0]})
        mock_compute_case_operator.return_value = mock_result

        result = evaluate._run_serial(
            [sample_case_operator], threshold=0.7, custom_param="test"
        )

        call_args = mock_compute_case_operator.call_args
        assert call_args[0][0] == sample_case_operator
        assert call_args[1]["threshold"] == 0.7
        assert call_args[1]["custom_param"] == "test"
        assert isinstance(result, list)

    def test_run_serial_empty_list(self):
        """Test _run_serial with empty case operator list."""
        result = evaluate._run_serial([])
        assert result == []


class TestRunParallel:
    """Test the _run_parallel function."""

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_basic(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test basic _run_parallel functionality."""
        # Setup mocks
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0], "case_id_number": [1]})]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel([sample_case_operator], n_jobs=2)

        # Verify Parallel was called with correct n_jobs
        mock_parallel_class.assert_called_once_with(n_jobs=2)

        # Verify the parallel instance was called (generator consumed)
        mock_parallel_instance.assert_called_once()

        assert result == mock_result

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_with_none_n_jobs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel with n_jobs=None (should use all CPUs)."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0]})]
        mock_parallel_instance.return_value = mock_result

        with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
            result = evaluate._run_parallel([sample_case_operator], n_jobs=None)

            # Should warn about using all CPUs
            mock_warning.assert_called_once_with(
                "No number of jobs provided, using all available CPUs."
            )

            # Verify Parallel was called with n_jobs=None
            mock_parallel_class.assert_called_once_with(n_jobs=None)
            assert isinstance(result, list)

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_multiple_cases(
        self, mock_tqdm, mock_delayed, mock_parallel_class
    ):
        """Test _run_parallel with multiple case operators."""
        case_op_1 = Mock()
        case_op_2 = Mock()
        case_operators = [case_op_1, case_op_2]

        mock_tqdm.return_value = case_operators
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [
            pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
            pd.DataFrame({"value": [2.0], "case_id_number": [2]}),
        ]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel(case_operators, n_jobs=4)

        assert len(result) == 2
        assert result[0]["case_id_number"].iloc[0] == 1
        assert result[1]["case_id_number"].iloc[0] == 2

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_with_kwargs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel passes kwargs correctly."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0]})]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel(
            [sample_case_operator],
            n_jobs=2,
            threshold=0.8,
            custom_param="parallel_test",
        )

        # Check that the parallel instance was called with the delayed functions
        mock_parallel_instance.assert_called_once()

        # The call should have created delayed functions with kwargs
        call_args = mock_parallel_instance.call_args[0][0]  # Generator passed
        # Convert generator to list to check
        delayed_calls = list(call_args)
        assert len(delayed_calls) == 1
        assert isinstance(result, list)

    def test_run_parallel_empty_list(self):
        """Test _run_parallel with empty case operator list."""
        with patch("extremeweatherbench.evaluate.Parallel") as mock_parallel_class:
            with patch("tqdm.auto.tqdm") as mock_tqdm:
                mock_tqdm.return_value = []
                mock_parallel_instance = Mock()
                mock_parallel_class.return_value = mock_parallel_instance
                mock_parallel_instance.return_value = []

                result = evaluate._run_parallel([], n_jobs=2)

                assert result == []


class TestComputeCaseOperator:
    """Test the compute_case_operator function."""

    @patch("extremeweatherbench.evaluate._build_datasets")
    @patch("extremeweatherbench.derived.maybe_derive_variables")
    @patch("extremeweatherbench.evaluate._evaluate_metric_and_return_df")
    def test_compute_case_operator_basic(
        self,
        mock_evaluate_metric,
        mock_derive_variables,
        mock_build_datasets,
        sample_case_operator,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test basic compute_case_operator functionality."""
        # Setup mocks
        mock_build_datasets.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )
        mock_derive_variables.side_effect = (
            lambda ds, variables: ds  # Return unchanged
        )

        mock_result = pd.DataFrame(
            {
                "value": [1.0],
                "metric": ["MockMetric"],
                "case_id_number": [1],
            }
        )
        mock_evaluate_metric.return_value = mock_result

        # Setup the case operator mocks - metric should be a list for iteration
        sample_case_operator.metric_list = [mock_base_metric]
        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        result = evaluate.compute_case_operator(sample_case_operator)

        mock_build_datasets.assert_called_once_with(sample_case_operator)
        assert isinstance(result, pd.DataFrame)

    @patch("extremeweatherbench.evaluate._build_datasets")
    @patch("extremeweatherbench.derived.maybe_derive_variables")
    def test_compute_case_operator_with_precompute(
        self,
        mock_derive_variables,
        mock_build_datasets,
        sample_case_operator,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test compute_case_operator with pre_compute=True."""
        mock_build_datasets.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )
        mock_derive_variables.side_effect = lambda ds, variables: ds

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )
        sample_case_operator.metric_list = [Mock()]

        with patch(
            "extremeweatherbench.evaluate._compute_and_maybe_cache"
        ) as mock_compute_cache:
            mock_compute_cache.return_value = [
                sample_forecast_dataset,
                sample_target_dataset,
            ]

            with patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_evaluate:
                mock_evaluate.return_value = pd.DataFrame({"value": [1.0]})

                result = evaluate.compute_case_operator(
                    sample_case_operator, pre_compute=True
                )

                mock_compute_cache.assert_called_once()
                assert isinstance(result, pd.DataFrame)

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_multiple_metrics(
        self,
        mock_build_datasets,
        sample_case_operator,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test compute_case_operator with multiple metrics."""
        mock_build_datasets.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Create multiple metrics
        metric_1 = Mock()
        metric_2 = Mock()
        sample_case_operator.metric_list = [metric_1, metric_2]

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        with patch("extremeweatherbench.derived.maybe_derive_variables") as mock_derive:
            mock_derive.side_effect = lambda ds, variables: ds

            with patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_evaluate:
                mock_evaluate.return_value = pd.DataFrame({"value": [1.0]})

                result = evaluate.compute_case_operator(sample_case_operator)

                # Should be called twice (once for each metric)
                assert mock_evaluate.call_count == 2
                assert len(result) == 2

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_zero_length_forecast_dataset(
        self, mock_build_datasets, sample_case_operator
    ):
        """Test compute_case_operator when _build_datasets returns empty forecast
        dataset."""
        # Mock _build_datasets to return empty datasets (simulating zero valid times)
        empty_forecast_ds = xr.Dataset(coords={"valid_time": pd.DatetimeIndex([])})
        empty_target_ds = xr.Dataset(coords={"valid_time": pd.DatetimeIndex([])})
        mock_build_datasets.return_value = (empty_forecast_ds, empty_target_ds)

        result = evaluate.compute_case_operator(sample_case_operator)

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

        # _build_datasets should be called, but no further processing should occur
        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_zero_length_target_dataset(
        self, mock_build_datasets, sample_case_operator, sample_forecast_dataset
    ):
        """Test compute_case_operator when _build_datasets returns empty
        target dataset."""
        # Mock _build_datasets to return empty target dataset
        empty_target_ds = xr.Dataset(coords={"valid_time": pd.DatetimeIndex([])})
        mock_build_datasets.return_value = (sample_forecast_dataset, empty_target_ds)

        result = evaluate.compute_case_operator(sample_case_operator)

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_empty_forecast_dataset(
        self, mock_build_datasets, sample_case_operator
    ):
        """Test compute_case_operator when _build_datasets returns empty forecast
        dataset."""
        # Mock _build_datasets to return empty datasets (simulating zero valid times)
        empty_forecast_ds = xr.Dataset()
        empty_target_ds = xr.Dataset()
        mock_build_datasets.return_value = (empty_forecast_ds, empty_target_ds)

        result = evaluate.compute_case_operator(sample_case_operator)

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

        # _build_datasets should be called, but no further processing should occur
        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_empty_target_dataset(
        self, mock_build_datasets, sample_case_operator, sample_forecast_dataset
    ):
        """Test compute_case_operator when _build_datasets returns empty
        target dataset."""
        # Mock _build_datasets to return empty target dataset
        empty_target_ds = xr.Dataset()
        mock_build_datasets.return_value = (sample_forecast_dataset, empty_target_ds)

        result = evaluate.compute_case_operator(sample_case_operator)

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLUMNS

        mock_build_datasets.assert_called_once_with(sample_case_operator)


class TestPipelineFunctions:
    """Test the pipeline functions."""

    def test_build_datasets(self, sample_case_operator):
        """Test _build_datasets function."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            mock_forecast_ds = xr.Dataset(attrs={"source": "forecast"})
            mock_target_ds = xr.Dataset(attrs={"source": "target"})
            mock_run_pipeline.side_effect = [mock_forecast_ds, mock_target_ds]

            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            assert mock_run_pipeline.call_count == 2
            assert forecast_ds.attrs["source"] == "forecast"
            assert target_ds.attrs["source"] == "target"

    def test_build_datasets_zero_length_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has zero-length dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create a forecast dataset with zero-length valid_time dimension
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": []},  # Empty valid_time coordinate
                attrs={"source": "forecast"},
            )
            mock_target_ds = xr.Dataset(attrs={"source": "target"})
            mock_run_pipeline.side_effect = [mock_forecast_ds, mock_target_ds]

            with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Should return empty datasets
                assert len(forecast_ds) == 0
                assert len(target_ds) == 0
                assert isinstance(forecast_ds, xr.Dataset)
                assert isinstance(target_ds, xr.Dataset)

                # Should log a warning
                mock_warning.assert_called_once()
                warning_message = mock_warning.call_args[0][0]
                assert "zero-length dimensions" in warning_message
                assert "['valid_time']" in warning_message
                assert (
                    str(sample_case_operator.case_metadata.case_id_number)
                    in warning_message
                )

                # Should only call run_pipeline once (for forecast), not for target
                assert mock_run_pipeline.call_count == 1

    def test_build_datasets_zero_length_warning_content(self, sample_case_operator):
        """Test _build_datasets warning message content when forecast has
        zero-length dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create a forecast dataset with zero-length dimension
            mock_forecast_ds = xr.Dataset(
                coords={"lead_time": []}, attrs={"source": "forecast"}
            )
            mock_run_pipeline.return_value = mock_forecast_ds

            with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Verify warning message contains expected information
                mock_warning.assert_called_once()
                warning_message = mock_warning.call_args[0][0]

                # Check all expected components are in the warning message
                assert (
                    f"case {sample_case_operator.case_metadata.case_id_number}"
                    in warning_message
                )
                assert "zero-length dimensions" in warning_message
                assert "['lead_time']" in warning_message
                assert (
                    str(sample_case_operator.case_metadata.start_date)
                    in warning_message
                )
                assert (
                    str(sample_case_operator.case_metadata.end_date) in warning_message
                )

    def test_build_datasets_multiple_zero_length_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has multiple zero-length dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create a forecast dataset with multiple zero-length dimensions
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": [], "latitude": []}, attrs={"source": "forecast"}
            )
            mock_run_pipeline.return_value = mock_forecast_ds

            with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Should return empty datasets
                assert len(forecast_ds) == 0
                assert len(target_ds) == 0

                # Should log a warning with both dimensions
                mock_warning.assert_called_once()
                warning_message = mock_warning.call_args[0][0]
                assert "zero-length dimensions" in warning_message
                # Check that both dimensions are mentioned (order may vary)
                assert "valid_time" in warning_message
                assert "latitude" in warning_message

    def test_build_datasets_normal_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has normal (non-zero) dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create datasets with normal dimensions
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": [1, 2, 3], "latitude": [40, 45, 50]},
                attrs={"source": "forecast"},
            )
            mock_target_ds = xr.Dataset(attrs={"source": "target"})
            mock_run_pipeline.side_effect = [mock_forecast_ds, mock_target_ds]

            with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Should return the actual datasets
                assert forecast_ds.attrs["source"] == "forecast"
                assert target_ds.attrs["source"] == "target"

                # Should not log any warning
                mock_warning.assert_not_called()

                # Should call run_pipeline twice (for both forecast and target)
                assert mock_run_pipeline.call_count == 2

    def test_run_pipeline_forecast(self, sample_case_operator, sample_forecast_dataset):
        """Test run_pipeline function for forecast data."""
        # Mock the pipeline methods
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = sample_forecast_dataset  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.return_value = (
            sample_forecast_dataset
        )
        sample_case_operator.forecast.subset_data_to_case.return_value = (
            sample_forecast_dataset
        )
        sample_case_operator.forecast.maybe_convert_to_dataset.return_value = (
            sample_forecast_dataset
        )
        sample_case_operator.forecast.add_source_to_dataset_attrs.return_value = (
            sample_forecast_dataset
        )

        result = evaluate.run_pipeline(sample_case_operator, "forecast")

        assert isinstance(result, xr.Dataset)
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.assert_called_once()  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.assert_called_once()
        # The pipe() method passes the dataset, then case_operator as kwarg
        assert sample_case_operator.forecast.subset_data_to_case.call_count == 1
        call_args = sample_case_operator.forecast.subset_data_to_case.call_args
        assert call_args[1]["case_operator"] == sample_case_operator
        sample_case_operator.forecast.maybe_convert_to_dataset.assert_called_once()
        sample_case_operator.forecast.add_source_to_dataset_attrs.assert_called_once()

    def test_run_pipeline_target(self, sample_case_operator, sample_target_dataset):
        """Test run_pipeline function for target data."""
        # Mock the pipeline methods
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.return_value = sample_target_dataset  # noqa: E501
        sample_case_operator.target.maybe_map_variable_names.return_value = (
            sample_target_dataset
        )
        sample_case_operator.target.subset_data_to_case.return_value = (
            sample_target_dataset
        )
        sample_case_operator.target.maybe_convert_to_dataset.return_value = (
            sample_target_dataset
        )
        sample_case_operator.target.add_source_to_dataset_attrs.return_value = (
            sample_target_dataset
        )

        result = evaluate.run_pipeline(sample_case_operator, "target")

        assert isinstance(result, xr.Dataset)
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.assert_called_once()  # noqa: E501

    def test_run_pipeline_invalid_source(self, sample_case_operator):
        """Test run_pipeline function with invalid input source."""
        with pytest.raises(ValueError, match="Invalid input source"):
            evaluate.run_pipeline(sample_case_operator, "invalid")

    def test_compute_and_maybe_cache(
        self, sample_forecast_dataset, sample_target_dataset
    ):
        """Test _compute_and_maybe_cache function."""
        # Create lazy datasets
        lazy_forecast = sample_forecast_dataset.chunk()
        lazy_target = sample_target_dataset.chunk()

        result = evaluate._compute_and_maybe_cache(
            lazy_forecast, lazy_target, cache_dir=None
        )

        assert len(result) == 2
        assert isinstance(result[0], xr.Dataset)
        assert isinstance(result[1], xr.Dataset)

    def test_compute_and_maybe_cache_with_cache_dir(
        self, sample_forecast_dataset, sample_target_dataset
    ):
        """Test _compute_and_maybe_cache with cache directory (should raise
        NotImplementedError)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            with pytest.raises(
                NotImplementedError, match="Caching is not implemented yet"
            ):
                evaluate._compute_and_maybe_cache(
                    sample_forecast_dataset, sample_target_dataset, cache_dir=cache_dir
                )


class TestMetricEvaluation:
    """Test metric evaluation functionality."""

    def test_evaluate_metric_and_return_df(
        self, sample_forecast_dataset, sample_target_dataset, mock_base_metric
    ):
        """Test _evaluate_metric_and_return_df function."""
        # Setup the metric mock
        mock_result = xr.DataArray(
            data=[1.5], dims=["lead_time"], coords={"lead_time": [0]}
        )
        mock_metric_instance = Mock()
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = mock_result
        mock_base_metric.return_value = mock_metric_instance

        result = evaluate._evaluate_metric_and_return_df(
            forecast_ds=sample_forecast_dataset,
            target_ds=sample_target_dataset,
            forecast_variable="surface_air_temperature",
            target_variable="2m_temperature",
            metric=mock_base_metric,
            case_id_number=1,
            event_type="heat_wave",
        )

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "metric" in result.columns
        assert "case_id_number" in result.columns
        assert "event_type" in result.columns
        assert result["metric"].iloc[0] == "TestMetric"
        assert result["case_id_number"].iloc[0] == 1
        assert result["event_type"].iloc[0] == "heat_wave"

    def test_evaluate_metric_and_return_df_with_kwargs(
        self, sample_forecast_dataset, sample_target_dataset, mock_base_metric
    ):
        """Test _evaluate_metric_and_return_df with additional kwargs."""
        mock_result = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [6]}
        )
        mock_metric_instance = Mock()
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = mock_result
        mock_base_metric.return_value = mock_metric_instance

        evaluate._evaluate_metric_and_return_df(
            forecast_ds=sample_forecast_dataset,
            target_ds=sample_target_dataset,
            forecast_variable="surface_air_temperature",
            target_variable="2m_temperature",
            metric=mock_base_metric,
            case_id_number=2,
            event_type="freeze",
            threshold=0.5,  # Additional kwarg
        )

        # Verify that kwargs were passed to compute_metric
        mock_metric_instance.compute_metric.assert_called_once()
        call_kwargs = mock_metric_instance.compute_metric.call_args[1]
        assert "threshold" in call_kwargs
        assert call_kwargs["threshold"] == 0.5


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_extremeweatherbench_empty_cases(self, sample_evaluation_object):
        """Test ExtremeWeatherBench with empty cases."""
        empty_cases = {"cases": []}

        ewb = evaluate.ExtremeWeatherBench(
            cases=empty_cases,
            evaluation_objects=[sample_evaluation_object],
        )

        with patch("extremeweatherbench.cases.build_case_operators") as mock_build:
            mock_build.return_value = []

            result = ewb.run()

            # Should return empty DataFrame when no cases
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_compute_case_operator_exception_handling(self, sample_case_operator):
        """Test exception handling in compute_case_operator."""
        with patch("extremeweatherbench.evaluate._build_datasets") as mock_build:
            mock_build.side_effect = Exception("Data loading failed")

            with pytest.raises(Exception, match="Data loading failed"):
                evaluate.compute_case_operator(sample_case_operator)

    def test_run_pipeline_missing_method(self, sample_case_operator):
        """Test run_pipeline when a required method is missing."""
        # Remove a required method
        del sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source

        with pytest.raises(AttributeError):
            evaluate.run_pipeline(sample_case_operator, "forecast")

    def test_evaluate_metric_computation_failure(
        self, sample_forecast_dataset, sample_target_dataset, mock_base_metric
    ):
        """Test metric evaluation when computation fails."""
        mock_metric_instance = Mock()
        mock_metric_instance.name = "FailingMetric"
        mock_metric_instance.compute_metric.side_effect = Exception(
            "Metric computation failed"
        )
        mock_base_metric.return_value = mock_metric_instance

        with pytest.raises(Exception, match="Metric computation failed"):
            evaluate._evaluate_metric_and_return_df(
                forecast_ds=sample_forecast_dataset,
                target_ds=sample_target_dataset,
                forecast_variable="surface_air_temperature",
                target_variable="2m_temperature",
                metric=mock_base_metric,
                case_id_number=1,
                event_type="heat_wave",
            )

    @patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_serial_exception(
        self, mock_run_serial, sample_case_operator
    ):
        """Test _run_case_operators handles exceptions in serial execution."""
        mock_run_serial.side_effect = Exception("Serial execution failed")

        with pytest.raises(Exception, match="Serial execution failed"):
            evaluate._run_case_operators(
                [sample_case_operator], parallel=False, n_jobs=None
            )

    @patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel_exception(
        self, mock_run_parallel, sample_case_operator
    ):
        """Test _run_case_operators handles exceptions in parallel execution."""
        mock_run_parallel.side_effect = Exception("Parallel execution failed")

        with pytest.raises(Exception, match="Parallel execution failed"):
            evaluate._run_case_operators(
                [sample_case_operator], parallel=True, n_jobs=2
            )

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    @patch("tqdm.auto.tqdm")
    def test_run_serial_case_operator_exception(
        self, mock_tqdm, mock_compute_case_operator, sample_case_operator
    ):
        """Test _run_serial handles exceptions from individual case operators."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_compute_case_operator.side_effect = Exception("Case operator failed")

        with pytest.raises(Exception, match="Case operator failed"):
            evaluate._run_serial([sample_case_operator])

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_joblib_exception(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel handles joblib Parallel exceptions."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_parallel_instance.side_effect = Exception("Joblib parallel failed")

        with pytest.raises(Exception, match="Joblib parallel failed"):
            evaluate._run_parallel([sample_case_operator], n_jobs=2)

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_delayed_function_exception(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel handles exceptions in delayed functions."""
        mock_tqdm.return_value = [sample_case_operator]

        # Mock delayed to raise an exception
        mock_delayed.side_effect = Exception("Delayed function creation failed")

        # Set up the Parallel mock to actually consume the generator and trigger delayed
        def consume_generator(generator):
            # This will consume the generator and trigger the delayed call
            list(generator)

        mock_parallel_instance = Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_parallel_instance.side_effect = consume_generator

        with pytest.raises(Exception, match="Delayed function creation failed"):
            evaluate._run_parallel([sample_case_operator], n_jobs=2)

    @patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_method_exception_propagation(
        self, mock_run_case_operators, sample_cases_dict, sample_evaluation_object
    ):
        """Test that ExtremeWeatherBench.run() propagates exceptions correctly."""
        mock_run_case_operators.side_effect = Exception("Execution failed")

        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        with pytest.raises(Exception, match="Execution failed"):
            ewb.run()

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    @patch("tqdm.auto.tqdm")
    def test_run_serial_partial_failure(self, mock_tqdm, mock_compute_case_operator):
        """Test _run_serial behavior when some case operators fail."""
        case_op_1 = Mock()
        case_op_2 = Mock()
        case_op_3 = Mock()
        case_operators = [case_op_1, case_op_2, case_op_3]

        mock_tqdm.return_value = case_operators

        # First succeeds, second fails, third never reached
        mock_compute_case_operator.side_effect = [
            pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
            Exception("Case operator 2 failed"),
            pd.DataFrame({"value": [3.0], "case_id_number": [3]}),
        ]

        # Should fail on the second case operator
        with pytest.raises(Exception, match="Case operator 2 failed"):
            evaluate._run_serial(case_operators)

        # Should have tried only the first two
        assert mock_compute_case_operator.call_count == 2

    @patch("extremeweatherbench.evaluate.Parallel")
    @patch("extremeweatherbench.evaluate.delayed")
    @patch("tqdm.auto.tqdm")
    def test_run_parallel_invalid_n_jobs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel with invalid n_jobs parameter."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = Mock()
        mock_delayed.return_value = mock_delayed_func

        # Mock Parallel to raise ValueError for invalid n_jobs
        mock_parallel_class.side_effect = ValueError("Invalid n_jobs parameter")

        with pytest.raises(ValueError, match="Invalid n_jobs parameter"):
            evaluate._run_parallel([sample_case_operator], n_jobs=-5)

    def test_run_case_operators_invalid_parallel_parameter(self, sample_case_operator):
        """Test _run_case_operators with invalid parallel parameter type."""
        # This should work fine since Python is dynamically typed,
        # but let's test edge case where parallel is not boolean
        with patch("extremeweatherbench.evaluate._run_serial") as mock_serial:
            mock_serial.return_value = []

            # Should still route to serial when parallel is falsy
            result = evaluate._run_case_operators(
                [sample_case_operator], parallel=0, n_jobs=None
            )

            mock_serial.assert_called_once()

        with patch("extremeweatherbench.evaluate._run_parallel") as mock_parallel:
            mock_parallel.return_value = []

            # Should route to parallel when parallel is truthy
            result = evaluate._run_case_operators(
                [sample_case_operator], parallel=1, n_jobs=None
            )

            mock_parallel.assert_called_once()
            assert isinstance(result, list)


class TestIntegration:
    """Test integration scenarios with real-like data."""

    @patch("extremeweatherbench.derived.maybe_derive_variables")
    def test_end_to_end_workflow(
        self,
        mock_derive_variables,
        sample_cases_dict,
        sample_evaluation_object,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test a complete end-to-end workflow."""
        mock_derive_variables.side_effect = lambda ds, variables: ds

        # Setup the evaluation object methods
        sample_evaluation_object.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Mock the pipeline methods to return our test datasets
        sample_evaluation_object.forecast.open_and_maybe_preprocess_data_from_source.return_value = (  # noqa: E501
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.maybe_map_variable_names.return_value = (
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.subset_data_to_case.return_value = (
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.maybe_convert_to_dataset.return_value = (
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.add_source_to_dataset_attrs.return_value = (
            sample_forecast_dataset
        )

        sample_evaluation_object.target.open_and_maybe_preprocess_data_from_source.return_value = (  # noqa: E501
            sample_target_dataset
        )
        sample_evaluation_object.target.maybe_map_variable_names.return_value = (
            sample_target_dataset
        )
        sample_evaluation_object.target.subset_data_to_case.return_value = (
            sample_target_dataset
        )
        sample_evaluation_object.target.maybe_convert_to_dataset.return_value = (
            sample_target_dataset
        )
        sample_evaluation_object.target.add_source_to_dataset_attrs.return_value = (
            sample_target_dataset
        )

        # Mock the metric evaluation to return a proper DataFrame
        mock_result_df = pd.DataFrame(
            {
                "value": [1.0],
                "target_variable": ["2m_temperature"],
                "metric": ["MockMetric"],
                "target_source": ["test_target"],
                "forecast_source": ["test_forecast"],
                "case_id_number": [1],
                "event_type": ["heat_wave"],
            }
        )

        with patch(
            "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
        ) as mock_eval:
            mock_eval.return_value = mock_result_df

            # Create and run the workflow
            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

        # Verify the result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "value" in result.columns
        assert "metric" in result.columns
        assert "case_id_number" in result.columns
        assert "event_type" in result.columns

    def test_multiple_variables_and_metrics(
        self, sample_cases_dict, sample_forecast_dataset, sample_target_dataset
    ):
        """Test workflow with multiple variables and metrics."""
        # Create multiple metrics
        metric_1 = Mock(spec=metrics.BaseMetric)
        metric_1.name = "Metric1"
        metric_1.return_value.name = "Metric1"
        metric_1.return_value.compute_metric.return_value = xr.DataArray(
            data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        metric_2 = Mock(spec=metrics.BaseMetric)
        metric_2.name = "Metric2"
        metric_2.return_value.name = "Metric2"
        metric_2.return_value.compute_metric.return_value = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Create evaluation object with multiple metrics and variables
        eval_obj = Mock(spec=inputs.EvaluationObject)
        eval_obj.event_type = "heat_wave"
        eval_obj.metric_list = [metric_1, metric_2]

        # Mock target and forecast with variables that match our datasets
        eval_obj.target = Mock(spec=inputs.TargetBase)
        eval_obj.target.name = "MultiTarget"
        eval_obj.target.variables = [
            "2m_temperature"
        ]  # Only include variables that exist

        eval_obj.forecast = Mock(spec=inputs.ForecastBase)
        eval_obj.forecast.name = "MultiForecast"
        eval_obj.forecast.variables = [
            "surface_air_temperature"
        ]  # Only include variables that exist

        # Setup pipeline mocks
        for obj in [eval_obj.target, eval_obj.forecast]:
            obj.open_and_maybe_preprocess_data_from_source.return_value = (
                sample_forecast_dataset
            )
            obj.maybe_map_variable_names.return_value = sample_forecast_dataset
            obj.subset_data_to_case.return_value = sample_forecast_dataset
            obj.maybe_convert_to_dataset.return_value = sample_forecast_dataset
            obj.add_source_to_dataset_attrs.return_value = sample_forecast_dataset

        eval_obj.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Mock the metric evaluation to return proper DataFrames
        mock_result_df = pd.DataFrame(
            {
                "value": [1.0],
                "target_variable": ["2m_temperature"],
                "metric": ["TestMetric"],
                "target_source": ["test_target"],
                "forecast_source": ["test_forecast"],
                "case_id_number": [1],
                "event_type": ["heat_wave"],
            }
        )

        with patch("extremeweatherbench.derived.maybe_derive_variables") as mock_derive:
            mock_derive.side_effect = lambda ds, variables: ds

            with patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_eval:
                mock_eval.return_value = mock_result_df

                ewb = evaluate.ExtremeWeatherBench(
                    cases=sample_cases_dict,
                    evaluation_objects=[eval_obj],
                )

                result = ewb.run()

                # Should have results for each metric combination
                assert len(result) >= 2  # At least 2 metrics * 1 case

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_serial_vs_parallel_results_consistency(
        self, mock_compute_case_operator, sample_cases_dict, sample_evaluation_object
    ):
        """Test that serial and parallel execution produce identical results."""
        # Setup mock case operators
        case_op_1 = Mock()
        case_op_2 = Mock()
        case_operators = [case_op_1, case_op_2]

        # Define consistent results
        result_1 = pd.DataFrame(
            {
                "value": [1.5],
                "metric": ["TestMetric"],
                "case_id_number": [1],
                "event_type": ["heat_wave"],
            }
        )
        result_2 = pd.DataFrame(
            {
                "value": [2.3],
                "metric": ["TestMetric"],
                "case_id_number": [2],
                "event_type": ["heat_wave"],
            }
        )

        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        with patch("extremeweatherbench.cases.build_case_operators") as mock_build:
            mock_build.return_value = case_operators

            # Test serial execution
            mock_compute_case_operator.side_effect = [result_1, result_2]
            serial_result = ewb.run(parallel=False)

            # Reset mock and test parallel execution
            mock_compute_case_operator.reset_mock()
            mock_compute_case_operator.side_effect = [result_1, result_2]
            parallel_result = ewb.run(parallel=True, n_jobs=2)

            # Both should produce valid DataFrames with same structure
            assert isinstance(serial_result, pd.DataFrame)
            assert isinstance(parallel_result, pd.DataFrame)
            assert len(serial_result) == 2
            assert len(parallel_result) == 2
            assert list(serial_result.columns) == list(parallel_result.columns)

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_execution_method_performance_comparison(self, mock_compute_case_operator):
        """Test that both execution methods handle the same workload."""
        import time

        # Create many case operators to simulate realistic workload
        case_operators = [Mock() for _ in range(10)]

        # Mock results
        mock_results = [
            pd.DataFrame(
                {
                    "value": [i * 0.1],
                    "metric": ["TestMetric"],
                    "case_id_number": [i],
                    "event_type": ["heat_wave"],
                }
            )
            for i in range(10)
        ]

        # Test serial execution timing - call _run_serial directly
        mock_compute_case_operator.side_effect = mock_results
        start_time = time.time()
        serial_result = evaluate._run_serial(case_operators)
        serial_time = time.time() - start_time

        # Test parallel execution timing - call _run_parallel directly with mocked
        # Parallel
        serial_call_count = mock_compute_case_operator.call_count
        mock_compute_case_operator.side_effect = mock_results

        with patch("extremeweatherbench.evaluate.Parallel") as mock_parallel_class:
            mock_parallel_instance = Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = mock_results

            start_time = time.time()
            parallel_result = evaluate._run_parallel(case_operators, n_jobs=2)
            parallel_time = time.time() - start_time

        # Both should produce the same number of results
        assert len(serial_result) == len(parallel_result) == 10

        # Serial execution should have called compute_case_operator
        assert serial_call_count == 10  # Serial execution
        # Parallel execution is mocked, so the call count doesn't increase
        assert mock_compute_case_operator.call_count == 10  # Only serial calls
        # Verify timing variables are used (avoid unused variable warnings)
        assert serial_time >= 0
        assert parallel_time >= 0

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_mixed_execution_parameters(self, mock_compute_case_operator):
        """Test various parameter combinations for execution methods."""
        case_operators = [Mock(), Mock()]
        mock_results = [
            pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
            pd.DataFrame({"value": [2.0], "case_id_number": [2]}),
        ]

        # Test different execution methods directly
        test_configs = [
            {"method": "serial", "args": [case_operators]},
            {"method": "parallel", "args": [case_operators], "kwargs": {"n_jobs": 1}},
            {"method": "parallel", "args": [case_operators], "kwargs": {"n_jobs": 2}},
            {
                "method": "parallel",
                "args": [case_operators],
                "kwargs": {"n_jobs": None},
            },
        ]

        for config in test_configs:
            mock_compute_case_operator.reset_mock()
            mock_compute_case_operator.side_effect = mock_results

            if config["method"] == "serial":
                result = evaluate._run_serial(*config["args"])
                # All configurations should produce valid results
                assert isinstance(result, list)
                assert len(result) == 2
                assert mock_compute_case_operator.call_count == 2
            else:
                # Mock parallel execution to avoid serialization issues
                with patch(
                    "extremeweatherbench.evaluate.Parallel"
                ) as mock_parallel_class:
                    mock_parallel_instance = Mock()
                    mock_parallel_class.return_value = mock_parallel_instance
                    mock_parallel_instance.return_value = mock_results

                    result = evaluate._run_parallel(
                        *config["args"], **config.get("kwargs", {})
                    )

                    # All configurations should produce valid results
                    assert isinstance(result, list)
                    assert len(result) == 2
                    # Parallel execution is mocked, so compute_case_operator is not
                    # called
                    assert mock_compute_case_operator.call_count == 0

    def test_execution_method_kwargs_propagation(self):
        """Test that kwargs are properly propagated through execution methods."""
        case_operator = Mock()

        # Mock compute_case_operator to capture kwargs
        def mock_compute_with_kwargs(case_op, cache_dir, **kwargs):
            # Store kwargs for verification
            mock_compute_with_kwargs.captured_kwargs = kwargs
            return pd.DataFrame({"value": [1.0]})

        mock_compute_with_kwargs.captured_kwargs = {}

        with patch(
            "extremeweatherbench.evaluate.compute_case_operator",
            side_effect=mock_compute_with_kwargs,
        ):
            # Test serial kwargs propagation
            result = evaluate._run_serial(
                [case_operator], custom_param="serial_test", threshold=0.9
            )

            captured = mock_compute_with_kwargs.captured_kwargs
            assert captured["custom_param"] == "serial_test"
            assert captured["threshold"] == 0.9
            assert isinstance(result, list)

            # Test parallel kwargs propagation
            with patch("extremeweatherbench.evaluate.Parallel") as mock_parallel_class:
                with patch("extremeweatherbench.evaluate.delayed") as mock_delayed:
                    mock_delayed.return_value = mock_compute_with_kwargs
                    mock_parallel_instance = Mock()
                    mock_parallel_class.return_value = mock_parallel_instance
                    mock_parallel_instance.return_value = [
                        pd.DataFrame({"value": [1.0]})
                    ]

                    # Reset captured kwargs
                    mock_compute_with_kwargs.captured_kwargs = {}

                    result = evaluate._run_parallel(
                        [case_operator],
                        n_jobs=2,
                        custom_param="parallel_test",
                        threshold=0.8,
                    )

                    # Verify parallel execution was set up correctly
                    mock_parallel_class.assert_called_once_with(n_jobs=2)
                    assert isinstance(result, list)

    def test_empty_case_operators_all_methods(self):
        """Test that all execution methods handle empty case operator lists."""
        # Test _run_case_operators
        result = evaluate._run_case_operators([], parallel=False, n_jobs=None)
        assert result == []

        result = evaluate._run_case_operators([], parallel=True, n_jobs=2)
        assert result == []

        # Test _run_serial
        result = evaluate._run_serial([])
        assert result == []

        # Test _run_parallel
        with patch("extremeweatherbench.evaluate.Parallel") as mock_parallel_class:
            mock_parallel_instance = Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = []

            result = evaluate._run_parallel([], n_jobs=2)
            assert result == []

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_large_case_operator_list_handling(self, mock_compute_case_operator):
        """Test handling of large numbers of case operators."""
        # Create a large list of case operators
        num_cases = 100
        case_operators = [Mock() for _ in range(num_cases)]

        # Create mock results
        mock_results = [
            pd.DataFrame(
                {"value": [i * 0.01], "case_id_number": [i], "metric": ["TestMetric"]}
            )
            for i in range(num_cases)
        ]

        # Test serial execution
        mock_compute_case_operator.side_effect = mock_results
        serial_results = evaluate._run_serial(case_operators)

        assert len(serial_results) == num_cases
        assert mock_compute_case_operator.call_count == num_cases

        # Test parallel execution
        mock_compute_case_operator.reset_mock()
        mock_compute_case_operator.side_effect = mock_results

        with patch("extremeweatherbench.evaluate.Parallel") as mock_parallel_class:
            mock_parallel_instance = Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = mock_results

            parallel_results = evaluate._run_parallel(case_operators, n_jobs=4)

            assert len(parallel_results) == num_cases
            mock_parallel_class.assert_called_once_with(n_jobs=4)


if __name__ == "__main__":
    pytest.main([__file__])
