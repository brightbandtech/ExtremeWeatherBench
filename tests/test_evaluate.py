"""Tests for evaluate module."""

import datetime
import logging
import pathlib
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import (
    cases,
    derived,
    evaluate,
    inputs,
    metrics,
    regions,
)

# Check if dask.distributed is available
try:
    import dask.distributed  # noqa: F401

    HAS_DASK_DISTRIBUTED = True
except ImportError:
    HAS_DASK_DISTRIBUTED = False


@pytest.fixture
def sample_individual_case():
    """Create a sample IndividualCase for testing."""
    return cases.IndividualCase(
        case_id_number=1,
        title="Test Heat Wave",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 6, 25),
        location=regions.CenteredRegion(
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
    mock_target = mock.Mock(spec=inputs.TargetBase)
    mock_target.name = "MockTarget"
    mock_target.variables = ["2m_temperature"]

    # Create a dataset with time coordinate for valid_times check
    time_coords = pd.date_range("2021-06-20", periods=5, freq="6h")
    mock_dataset = xr.Dataset(
        coords={"time": time_coords}, attrs={"source": "mock_target"}
    )

    mock_target.open_and_maybe_preprocess_data_from_source.return_value = mock_dataset
    mock_target.maybe_map_variable_names.return_value = mock_dataset
    mock_target.subset_data_to_case.return_value = mock_dataset
    mock_target.maybe_convert_to_dataset.return_value = mock_dataset
    mock_target.add_source_to_dataset_attrs.return_value = mock_dataset
    mock_target.maybe_align_forecast_to_target.return_value = (
        mock_dataset,
        mock_dataset,
    )
    return mock_target


@pytest.fixture
def mock_forecast_base():
    """Create a mock ForecastBase object."""
    mock_forecast = mock.Mock(spec=inputs.ForecastBase)
    mock_forecast.name = "MockForecast"
    mock_forecast.variables = ["surface_air_temperature"]

    # Create a dataset with init_time coordinate for valid_times check
    init_time_coords = pd.date_range("2021-06-20", periods=3, freq="24h")
    lead_time_coords = [0, 6, 12, 18]
    mock_dataset = xr.Dataset(
        coords={"init_time": init_time_coords, "lead_time": lead_time_coords},
        attrs={"source": "mock_forecast"},
    )

    mock_forecast.open_and_maybe_preprocess_data_from_source.return_value = mock_dataset
    mock_forecast.maybe_map_variable_names.return_value = mock_dataset
    mock_forecast.subset_data_to_case.return_value = mock_dataset
    mock_forecast.maybe_convert_to_dataset.return_value = mock_dataset
    mock_forecast.add_source_to_dataset_attrs.return_value = mock_dataset
    return mock_forecast


@pytest.fixture
def mock_base_metric():
    """Create a mock BaseMetric object."""
    mock_metric = mock.Mock(spec=metrics.BaseMetric)
    mock_metric.name = "MockMetric"
    mock_metric.forecast_variable = None
    mock_metric.target_variable = None
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
    mock_base_metric.forecast_variable = None
    mock_base_metric.target_variable = None
    """Create a sample CaseOperator."""
    # Ensure metric has forecast_variable and target_variable attributes
    mock_base_metric.forecast_variable = None
    mock_base_metric.target_variable = None
    return cases.CaseOperator(
        case_metadata=sample_individual_case,
        metric_list=[mock_base_metric],
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


class TestOutputColumns:
    """Test the OUTPUT_COLUMNS constant."""

    def test_output_columns_exists(self):
        """Test that OUTPUT_COLUMNS is defined and contains expected columns."""
        expected_columns = [
            "value",
            "lead_time",
            "init_time",
            "target_variable",
            "metric",
            "forecast_source",
            "target_source",
            "case_id_number",
            "event_type",
        ]
        assert hasattr(evaluate, "OUTPUT_COLUMNS")
        assert evaluate.OUTPUT_COLUMNS == expected_columns


class TestExtremeWeatherBench:
    """Test the ExtremeWeatherBench class."""

    def assert_cases_equal(self, actual, expected):
        """Assert that two IndividualCaseCollection instances are equal."""
        assert len(actual.cases) == len(expected.cases)

        for actual_case, expected_case in zip(actual.cases, expected.cases):
            assert actual_case.case_id_number == expected_case.case_id_number
            assert actual_case.title == expected_case.title
            assert actual_case.start_date == expected_case.start_date
            assert actual_case.end_date == expected_case.end_date
            assert actual_case.event_type == expected_case.event_type

            # Compare region attributes instead of objects
            assert type(actual_case.location) is type(expected_case.location)
            if hasattr(actual_case.location, "latitude"):
                assert actual_case.location.latitude == expected_case.location.latitude
                assert (
                    actual_case.location.longitude == expected_case.location.longitude
                )
                assert (
                    actual_case.location.bounding_box_degrees
                    == expected_case.location.bounding_box_degrees
                )

    def test_initialization(self, sample_cases_dict, sample_evaluation_object):
        """Test ExtremeWeatherBench initialization."""
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        self.assert_cases_equal(
            ewb.case_metadata, cases.load_individual_cases(sample_cases_dict)
        )
        assert ewb.evaluation_objects == [sample_evaluation_object]
        assert ewb.cache_dir is None

    def test_initialization_with_cache_dir(
        self, sample_cases_dict, sample_evaluation_object
    ):
        """Test ExtremeWeatherBench initialization with cache directory."""
        cache_dir = "/tmp/test_cache"
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
            cache_dir=cache_dir,
        )

        # Cache dir should be converted to Path object
        assert ewb.cache_dir == pathlib.Path(cache_dir)

    def test_initialization_with_path_cache_dir(
        self, sample_cases_dict, sample_evaluation_object
    ):
        """Test ExtremeWeatherBench initialization with Path cache directory."""
        cache_dir = pathlib.Path("/tmp/test_cache")
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
            cache_dir=cache_dir,
        )

        assert ewb.cache_dir == cache_dir

    @mock.patch("extremeweatherbench.cases.build_case_operators")
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
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        result = ewb.case_operators

        # Verify that build_case_operators was called correctly
        mock_build_case_operators.assert_called_once()
        call_args = mock_build_case_operators.call_args[0]

        # Check that the first argument (case collection) has the right structure
        passed_case_collection = call_args[0]
        self.assert_cases_equal(
            passed_case_collection, cases.load_individual_cases(sample_cases_dict)
        )

        # Check that the second argument (evaluation objects) is correct
        assert call_args[1] == [sample_evaluation_object]

        # Check that the result is what the mock returned
        assert result == [sample_case_operator]

    @mock.patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_serial(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes serially."""
        # Mock the case operators property
        with mock.patch.object(
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
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(n_jobs=1)

            # Serial mode should not pass parallel_config
            mock_run_case_operators.assert_called_once_with(
                [sample_case_operator],
                None,
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @mock.patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_parallel(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes in parallel."""
        with mock.patch.object(
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
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(n_jobs=2)

            mock_run_case_operators.assert_called_once_with(
                [sample_case_operator],
                None,
                parallel_config={"backend": "threading", "n_jobs": 2},
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @mock.patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_with_kwargs(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method passes kwargs correctly."""
        with mock.patch.object(
            evaluate.ExtremeWeatherBench, "case_operators", new=[sample_case_operator]
        ):
            mock_result = [pd.DataFrame({"value": [1.0]})]
            mock_run_case_operators.return_value = mock_result

            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run(n_jobs=1, threshold=0.5, pre_compute=True)

            # Check that kwargs were passed through
            call_args = mock_run_case_operators.call_args
            assert call_args[1]["threshold"] == 0.5
            assert call_args[1]["pre_compute"] is True
            assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_empty_results(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
    ):
        """Test the run method handles empty results."""
        with mock.patch.object(evaluate.ExtremeWeatherBench, "case_operators", new=[]):
            mock_run_case_operators.return_value = []

            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == evaluate.OUTPUT_COLUMNS

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_run_with_caching(
        self,
        mock_compute_case_operator,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method with caching enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = pathlib.Path(temp_dir)

            with mock.patch.object(
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
                            pathlib.Path(cache_dir_arg)
                            if isinstance(cache_dir_arg, str)
                            else cache_dir_arg
                        )
                        mock_result.to_pickle(cache_path / "case_results.pkl")
                    return mock_result

                mock_compute_case_operator.side_effect = mock_compute_with_caching

                ewb = evaluate.ExtremeWeatherBench(
                    case_metadata=sample_cases_dict,
                    evaluation_objects=[sample_evaluation_object],
                    cache_dir=cache_dir,
                )

                ewb.run(n_jobs=1)

                # Check that cache directory was created
                assert cache_dir.exists()

                # Check that results were cached
                cache_file = cache_dir / "case_results.pkl"
                assert cache_file.exists()

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_run_multiple_cases(
        self, mock_compute_case_operator, sample_cases_dict, sample_evaluation_object
    ):
        """Test the run method with multiple case operators."""
        # Create multiple case operators
        case_operator_1 = mock.Mock()
        case_operator_2 = mock.Mock()

        with mock.patch.object(
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
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

            assert mock_compute_case_operator.call_count == 2
            assert len(result) == 2
            assert result["case_id_number"].tolist() == [1, 2]


class TestRunCaseOperators:
    """Test the _run_case_operators function."""

    @mock.patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_serial(self, mock_run_serial, sample_case_operator):
        """Test _run_case_operators routes to serial execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_serial.return_value = mock_results

        # Serial mode: don't pass parallel_config
        result = evaluate._run_case_operators([sample_case_operator], None)

        mock_run_serial.assert_called_once_with([sample_case_operator], None)
        assert result == mock_results

    @mock.patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel(self, mock_run_parallel, sample_case_operator):
        """Test _run_case_operators routes to parallel execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_parallel.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 4},
        )

        mock_run_parallel.assert_called_once_with(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 4},
        )
        assert result == mock_results

    @mock.patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_with_kwargs(
        self, mock_run_serial, sample_case_operator
    ):
        """Test _run_case_operators passes kwargs correctly."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_serial.return_value = mock_results

        # Serial mode: don't pass parallel_config
        result = evaluate._run_case_operators(
            [sample_case_operator],
            None,
            threshold=0.5,
            pre_compute=True,
        )

        call_args = mock_run_serial.call_args
        assert call_args[0][0] == [sample_case_operator]
        assert call_args[0][1] is None  # cache_dir
        assert call_args[1]["threshold"] == 0.5
        assert call_args[1]["pre_compute"] is True
        assert isinstance(result, list)

    @mock.patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel_with_kwargs(
        self, mock_run_parallel, sample_case_operator
    ):
        """Test _run_case_operators passes kwargs to parallel execution."""
        mock_results = [pd.DataFrame({"value": [1.0]})]
        mock_run_parallel.return_value = mock_results

        result = evaluate._run_case_operators(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 2},
            custom_param="test_value",
        )

        call_args = mock_run_parallel.call_args
        assert call_args[0][0] == [sample_case_operator]
        assert call_args[1]["parallel_config"] == {"backend": "threading", "n_jobs": 2}
        assert call_args[1]["custom_param"] == "test_value"
        assert isinstance(result, list)

    def test_run_case_operators_empty_list(self):
        """Test _run_case_operators with empty case operator list."""
        with mock.patch("extremeweatherbench.evaluate._run_serial") as mock_serial:
            mock_serial.return_value = []

            # Serial mode: don't pass parallel_config
            result = evaluate._run_case_operators([], None)

            mock_serial.assert_called_once_with([], None)
            assert result == []


class TestRunSerial:
    """Test the _run_serial function."""

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    @mock.patch("tqdm.auto.tqdm")
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

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_serial_multiple_cases(self, mock_tqdm, mock_compute_case_operator):
        """Test _run_serial with multiple case operators."""
        case_op_1 = mock.Mock()
        case_op_2 = mock.Mock()
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

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    @mock.patch("tqdm.auto.tqdm")
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

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_basic(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test basic _run_parallel functionality."""
        # Setup mocks
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0], "case_id_number": [1]})]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 2},
        )

        # Verify Parallel was called with total_tasks (n_jobs via parallel_config)
        mock_parallel_class.assert_called_once_with(total_tasks=1)

        # Verify the parallel instance was called (generator consumed)
        mock_parallel_instance.assert_called_once()

        assert result == mock_result

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_with_none_n_jobs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel with n_jobs=None (should use all CPUs)."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0]})]
        mock_parallel_instance.return_value = mock_result

        with mock.patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
            result = evaluate._run_parallel(
                [sample_case_operator],
                parallel_config={"backend": "threading", "n_jobs": None},
            )

            # Should warn about using all CPUs
            mock_warning.assert_called_once_with(
                "No number of jobs provided, using joblib backend default."
            )

            # Verify Parallel was called with total_tasks (n_jobs via parallel_config)
            mock_parallel_class.assert_called_once_with(total_tasks=1)
            assert isinstance(result, list)

    @mock.patch("joblib.parallel_config")
    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    def test_run_parallel_n_jobs_in_config(
        self, mock_parallel_class, mock_parallel_config
    ):
        """Test that n_jobs is passed through parallel_config, not directly."""
        sample_case_operator = mock.Mock()
        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0]})]
        mock_parallel_instance.return_value = mock_result

        # Create a context manager mock
        mock_context = mock.MagicMock()
        mock_parallel_config.return_value.__enter__ = mock.Mock(
            return_value=mock_context
        )
        mock_parallel_config.return_value.__exit__ = mock.Mock(return_value=False)

        result = evaluate._run_parallel(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 4},
        )

        # Verify parallel_config was called with n_jobs in the config
        mock_parallel_config.assert_called_once()
        call_kwargs = mock_parallel_config.call_args[1]
        assert call_kwargs["backend"] == "threading"
        assert call_kwargs["n_jobs"] == 4

        # Verify ParallelTqdm was called WITHOUT n_jobs
        mock_parallel_class.assert_called_once_with(total_tasks=1)
        assert isinstance(result, list)

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_multiple_cases(
        self, mock_tqdm, mock_delayed, mock_parallel_class
    ):
        """Test _run_parallel with multiple case operators."""
        case_op_1 = mock.Mock()
        case_op_2 = mock.Mock()
        case_operators = [case_op_1, case_op_2]

        mock_tqdm.return_value = case_operators
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [
            pd.DataFrame({"value": [1.0], "case_id_number": [1]}),
            pd.DataFrame({"value": [2.0], "case_id_number": [2]}),
        ]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel(
            case_operators, parallel_config={"backend": "threading", "n_jobs": 4}
        )

        assert len(result) == 2
        assert result[0]["case_id_number"].iloc[0] == 1
        assert result[1]["case_id_number"].iloc[0] == 2

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_with_kwargs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel passes kwargs correctly."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_result = [pd.DataFrame({"value": [1.0]})]
        mock_parallel_instance.return_value = mock_result

        result = evaluate._run_parallel(
            [sample_case_operator],
            parallel_config={"backend": "threading", "n_jobs": 2},
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
        with mock.patch(
            "extremeweatherbench.utils.ParallelTqdm"
        ) as mock_parallel_class:
            with mock.patch("tqdm.auto.tqdm") as mock_tqdm:
                mock_tqdm.return_value = []
                mock_parallel_instance = mock.Mock()
                mock_parallel_class.return_value = mock_parallel_instance
                mock_parallel_instance.return_value = []

                result = evaluate._run_parallel(
                    [], parallel_config={"backend": "threading", "n_jobs": 2}
                )

                assert result == []

    @pytest.mark.skipif(
        not HAS_DASK_DISTRIBUTED, reason="dask.distributed not installed"
    )
    @mock.patch("dask.distributed.Client")
    @mock.patch("dask.distributed.LocalCluster")
    def test_run_parallel_dask_backend_auto_client(
        self, mock_local_cluster, mock_client_class, sample_case_operator
    ):
        """Test _run_parallel with dask backend automatically creates client."""
        # Mock Client.current() to raise ValueError (no existing client)
        mock_client_class.current.side_effect = ValueError("No client found")

        # Mock the client instance
        mock_client = mock.Mock()
        mock_client_class.return_value = mock_client

        # Mock LocalCluster
        mock_cluster = mock.Mock()
        mock_local_cluster.return_value = mock_cluster

        # Mock the parallel execution
        with mock.patch("extremeweatherbench.utils.ParallelTqdm") as mock_parallel:
            mock_parallel_instance = mock.Mock()
            mock_parallel.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = [pd.DataFrame({"test": [1]})]

            with mock.patch("joblib.parallel_config"):
                result = evaluate._run_parallel(
                    [sample_case_operator],
                    parallel_config={"backend": "dask", "n_jobs": 2},
                )

        # Verify client was created and closed
        mock_client_class.assert_called_once_with(mock_cluster)
        mock_client.close.assert_called_once()
        assert isinstance(result, list)

    @pytest.mark.skipif(
        not HAS_DASK_DISTRIBUTED, reason="dask.distributed not installed"
    )
    @mock.patch("dask.distributed.Client")
    def test_run_parallel_dask_backend_existing_client(
        self, mock_client_class, sample_case_operator
    ):
        """Test _run_parallel with dask backend uses existing client."""
        # Mock existing client
        mock_existing_client = mock.Mock()
        mock_client_class.current.return_value = mock_existing_client

        # Mock the parallel execution
        with mock.patch("extremeweatherbench.utils.ParallelTqdm") as mock_parallel:
            mock_parallel_instance = mock.Mock()
            mock_parallel.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = [pd.DataFrame({"test": [1]})]

            with mock.patch("joblib.parallel_config"):
                result = evaluate._run_parallel(
                    [sample_case_operator],
                    parallel_config={"backend": "dask", "n_jobs": 2},
                )

        # Verify no new client was created and existing wasn't closed
        mock_client_class.assert_not_called()
        mock_existing_client.close.assert_not_called()
        assert isinstance(result, list)


class TestComputeCaseOperator:
    """Test the compute_case_operator function."""

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
    @mock.patch("extremeweatherbench.derived.maybe_derive_variables")
    @mock.patch("extremeweatherbench.evaluate._evaluate_metric_and_return_df")
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
            lambda ds, variables, **kwargs: ds  # Return unchanged
        )

        mock_result = pd.DataFrame(
            {
                "value": [1.0],
                "metric": ["MockMetric"],
                "case_id_number": [1],
            }
        )
        mock_evaluate_metric.return_value = mock_result

        # Setup the case operator mocks
        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        result = evaluate.compute_case_operator(sample_case_operator)

        mock_build_datasets.assert_called_once_with(sample_case_operator)
        assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
    @mock.patch("extremeweatherbench.derived.maybe_derive_variables")
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
        mock_derive_variables.side_effect = lambda ds, variables, **kwargs: ds

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )
        mock_metric = mock.Mock(spec=metrics.BaseMetric)
        mock_metric.forecast_variable = None
        mock_metric.target_variable = None
        sample_case_operator.metric_list = [mock_metric]

        with mock.patch(
            "extremeweatherbench.evaluate._compute_and_maybe_cache"
        ) as mock_compute_cache:
            mock_compute_cache.return_value = [
                sample_forecast_dataset,
                sample_target_dataset,
            ]

            with mock.patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_evaluate:
                mock_evaluate.return_value = pd.DataFrame({"value": [1.0]})

                result = evaluate.compute_case_operator(
                    sample_case_operator, pre_compute=True
                )

                mock_compute_cache.assert_called_once()
                assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
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
        metric_1 = mock.Mock(spec=metrics.BaseMetric)
        metric_1.forecast_variable = None
        metric_1.target_variable = None
        metric_2 = mock.Mock(spec=metrics.BaseMetric)
        metric_2.forecast_variable = None
        metric_2.target_variable = None
        sample_case_operator.metric_list = [metric_1, metric_2]

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        with mock.patch(
            "extremeweatherbench.derived.maybe_derive_variables"
        ) as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            with mock.patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_evaluate:
                mock_evaluate.return_value = pd.DataFrame({"value": [1.0]})

                result = evaluate.compute_case_operator(sample_case_operator)

                # Should be called twice (once for each metric)
                assert mock_evaluate.call_count == 2
                assert len(result) == 2

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
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
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

        # _build_datasets should be called, but no further processing should occur
        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
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
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
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
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

        # _build_datasets should be called, but no further processing should occur
        mock_build_datasets.assert_called_once_with(sample_case_operator)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
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
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

        mock_build_datasets.assert_called_once_with(sample_case_operator)


class TestPipelineFunctions:
    """Test the pipeline functions."""

    def test_build_datasets(self, sample_case_operator):
        """Test _build_datasets function."""
        with mock.patch(
            "extremeweatherbench.evaluate.run_pipeline"
        ) as mock_run_pipeline:
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": [1, 2, 3]}, attrs={"name": "forecast_source"}
            )
            mock_target_ds = xr.Dataset(
                coords={"time": [1, 2, 3]}, attrs={"name": "target_source"}
            )
            mock_run_pipeline.side_effect = [mock_target_ds, mock_forecast_ds]

            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            assert mock_run_pipeline.call_count == 2
            assert forecast_ds.attrs["name"] == "forecast_source"
            assert target_ds.attrs["name"] == "target_source"

    def test_build_datasets_zero_length_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has zero-length dimensions."""
        # Set up the mock to return a dataset that will trigger the warning
        # by having no valid times in the date range
        empty_dataset = xr.Dataset()
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = empty_dataset  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.return_value = (
            empty_dataset
        )

        with mock.patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            # Should return empty datasets
            assert len(forecast_ds) == 0
            assert len(target_ds) == 0
            assert isinstance(forecast_ds, xr.Dataset)
            assert isinstance(target_ds, xr.Dataset)

            # Should log a warning
            mock_warning.assert_called()
            warning_message = mock_warning.call_args[0][0]
            assert "has no data for case time range" in warning_message
            assert (
                str(sample_case_operator.case_metadata.case_id_number)
                in warning_message
            )

    def test_build_datasets_zero_length_warning_content(self, sample_case_operator):
        """Test _build_datasets warning message content when forecast has
        zero-length dimensions."""
        # Set up the mock to return a dataset that will trigger the warning
        empty_dataset = xr.Dataset()
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = empty_dataset  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.return_value = (
            empty_dataset
        )

        with mock.patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            # Verify warning message contains expected information
            mock_warning.assert_called()
            warning_message = mock_warning.call_args[0][0]

            # Check all expected components are in the warning message
            assert (
                f"case {sample_case_operator.case_metadata.case_id_number}"
                in warning_message
            )
            assert "has no data for case time range" in warning_message
            assert str(sample_case_operator.case_metadata.start_date) in warning_message
            assert str(sample_case_operator.case_metadata.end_date) in warning_message

    def test_build_datasets_multiple_zero_length_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has multiple zero-length dimensions."""
        # Set up the mock to return a dataset that will trigger the warning
        empty_dataset = xr.Dataset()
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = empty_dataset  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.return_value = (
            empty_dataset
        )

        with mock.patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            # Should return empty datasets
            assert len(forecast_ds) == 0
            assert len(target_ds) == 0

            # Should log a warning
            mock_warning.assert_called()
            warning_message = mock_warning.call_args[0][0]
            assert "has no data for case time range" in warning_message
            assert (
                str(sample_case_operator.case_metadata.case_id_number)
                in warning_message
            )

    def test_build_datasets_normal_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has normal (non-zero) dimensions."""
        with mock.patch(
            "extremeweatherbench.evaluate.run_pipeline"
        ) as mock_run_pipeline:
            # Create datasets with normal dimensions
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": [1, 2, 3], "latitude": [40, 45, 50]},
                attrs={"source": "forecast"},
            )
            mock_target_ds = xr.Dataset(
                coords={"time": [1, 2, 3], "latitude": [40, 45, 50]},
                attrs={"source": "target"},
            )
            mock_run_pipeline.side_effect = [mock_target_ds, mock_forecast_ds]

            with mock.patch(
                "extremeweatherbench.evaluate.logger.warning"
            ) as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Should return the actual datasets
                assert forecast_ds.attrs["source"] == "forecast"
                assert target_ds.attrs["source"] == "target"

                # Should not log any warning
                mock_warning.assert_not_called()

                # Should call run_pipeline twice (for both forecast and target)
                assert mock_run_pipeline.call_count == 2

    @mock.patch("extremeweatherbench.derived.maybe_derive_variables")
    @mock.patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
    def test_run_pipeline_forecast(
        self,
        mock_maybe_subset_variables,
        mock_derived,
        sample_case_operator,
        sample_forecast_dataset,
    ):
        """Test run_pipeline function for forecast data."""
        # Mock the pipeline methods
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = sample_forecast_dataset  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.return_value = (
            sample_forecast_dataset
        )
        mock_maybe_subset_variables.return_value = sample_forecast_dataset
        sample_case_operator.forecast.subset_data_to_case.return_value = (
            sample_forecast_dataset
        )
        sample_case_operator.forecast.maybe_convert_to_dataset.return_value = (
            sample_forecast_dataset
        )
        sample_case_operator.forecast.add_source_to_dataset_attrs.return_value = (
            sample_forecast_dataset
        )
        mock_derived.return_value = sample_forecast_dataset

        result = evaluate.run_pipeline(
            sample_case_operator.case_metadata, sample_case_operator.forecast
        )

        assert isinstance(result, xr.Dataset)
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.assert_called_once()  # noqa: E501
        sample_case_operator.forecast.maybe_map_variable_names.assert_called_once()
        mock_maybe_subset_variables.assert_called_once()
        # The method is called with data as first arg, case_metadata as second arg
        assert sample_case_operator.forecast.subset_data_to_case.call_count == 1
        call_args = sample_case_operator.forecast.subset_data_to_case.call_args
        assert call_args[0][1] == sample_case_operator.case_metadata
        sample_case_operator.forecast.maybe_convert_to_dataset.assert_called_once()
        sample_case_operator.forecast.add_source_to_dataset_attrs.assert_called_once()

    @mock.patch("extremeweatherbench.derived.maybe_derive_variables")
    @mock.patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
    def test_run_pipeline_target(
        self,
        mock_maybe_subset_variables,
        mock_derived,
        sample_case_operator,
        sample_target_dataset,
    ):
        """Test run_pipeline function for target data."""
        # Mock the pipeline methods
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.return_value = sample_target_dataset  # noqa: E501
        sample_case_operator.target.maybe_map_variable_names.return_value = (
            sample_target_dataset
        )
        mock_maybe_subset_variables.return_value = sample_target_dataset
        sample_case_operator.target.subset_data_to_case.return_value = (
            sample_target_dataset
        )
        sample_case_operator.target.maybe_convert_to_dataset.return_value = (
            sample_target_dataset
        )
        sample_case_operator.target.add_source_to_dataset_attrs.return_value = (
            sample_target_dataset
        )
        mock_derived.return_value = sample_target_dataset

        result = evaluate.run_pipeline(
            sample_case_operator.case_metadata, sample_case_operator.target
        )

        assert isinstance(result, xr.Dataset)
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.assert_called_once()  # noqa: E501

    def test_run_pipeline_invalid_source(self, sample_case_operator):
        """Test run_pipeline function with invalid input source."""
        with pytest.raises(AttributeError, match="'str' object has no attribute"):
            evaluate.run_pipeline(sample_case_operator.case_metadata, "invalid")

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
            cache_dir = pathlib.Path(temp_dir)

            with pytest.raises(
                NotImplementedError, match="Caching is not implemented yet"
            ):
                evaluate._compute_and_maybe_cache(
                    sample_forecast_dataset, sample_target_dataset, cache_dir=cache_dir
                )


class TestMetricEvaluation:
    """Test metric evaluation functionality."""

    def test_evaluate_metric_and_return_df(
        self,
        sample_forecast_dataset,
        sample_target_dataset,
        sample_case_operator,
        mock_base_metric,
    ):
        """Test _evaluate_metric_and_return_df function."""
        # Setup the metric mock
        mock_result = xr.DataArray(
            data=[1.5], dims=["lead_time"], coords={"lead_time": [0]}
        )
        mock_base_metric.name = "TestMetric"
        mock_base_metric.compute_metric.return_value = mock_result
        result = evaluate._evaluate_metric_and_return_df(
            forecast_ds=sample_forecast_dataset,
            target_ds=sample_target_dataset,
            forecast_variable="surface_air_temperature",
            target_variable="2m_temperature",
            metric=mock_base_metric,
            case_operator=sample_case_operator,
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
        self,
        sample_forecast_dataset,
        sample_target_dataset,
        sample_case_operator,
        mock_base_metric,
    ):
        """Test _evaluate_metric_and_return_df with additional kwargs."""
        mock_result = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [6]}
        )
        mock_base_metric.name = "TestMetric"
        mock_base_metric.compute_metric.return_value = mock_result

        evaluate._evaluate_metric_and_return_df(
            forecast_ds=sample_forecast_dataset,
            target_ds=sample_target_dataset,
            forecast_variable="surface_air_temperature",
            target_variable="2m_temperature",
            metric=mock_base_metric,
            case_operator=sample_case_operator,
            threshold=0.5,  # Additional kwarg
        )

        # Verify that kwargs were passed to compute_metric
        mock_base_metric.compute_metric.assert_called_once()
        call_kwargs = mock_base_metric.compute_metric.call_args[1]
        assert "threshold" in call_kwargs
        assert call_kwargs["threshold"] == 0.5

    def test_evaluate_metric_and_return_df_with_derived_variables(
        self, mock_base_metric, sample_case_operator
    ):
        """Test _evaluate_metric_and_return_df with derived variables."""
        # Create datasets with derived variables included
        forecast_ds = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["init_time", "lead_time", "latitude", "longitude"],
                    np.random.randn(2, 3, 4, 5) + 280,
                ),
                "derived_forecast_var": (
                    ["init_time", "lead_time", "latitude", "longitude"],
                    np.random.randn(2, 3, 4, 5) + 285,
                ),
            },
            coords={
                "init_time": pd.date_range("2021-06-20", periods=2, freq="D"),
                "lead_time": [0, 6, 12],
                "latitude": np.linspace(30, 50, 4),
                "longitude": np.linspace(-120, -90, 5),
            },
            attrs={"source": "test_forecast", "forecast_source": "test_forecast"},
        )

        target_ds = xr.Dataset(
            {
                "2m_temperature": (
                    ["time", "latitude", "longitude"],
                    np.random.randn(3, 4, 5) + 275,
                ),
                "derived_target_var": (
                    ["time", "latitude", "longitude"],
                    np.random.randn(3, 4, 5) + 280,
                ),
            },
            coords={
                "time": pd.date_range("2021-06-20", periods=3, freq="6h"),
                "latitude": np.linspace(30, 50, 4),
                "longitude": np.linspace(-120, -90, 5),
            },
            attrs={"source": "test_target", "target_source": "test_target"},
        )

        # Setup the metric mock
        mock_result = xr.DataArray(
            data=[2.5], dims=["lead_time"], coords={"lead_time": [0]}
        )
        mock_base_metric.name = "TestDerivedMetric"
        mock_base_metric.compute_metric.return_value = mock_result

        result = evaluate._evaluate_metric_and_return_df(
            forecast_ds=forecast_ds,
            target_ds=target_ds,
            forecast_variable=ForecastDerivedVariable,
            target_variable=TargetDerivedVariable,
            metric=mock_base_metric,
            case_operator=sample_case_operator,
        )

        # Verify the result structure
        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "metric" in result.columns
        assert "target_variable" in result.columns
        assert "case_id_number" in result.columns
        assert "event_type" in result.columns

        # Check the values
        assert result["metric"].iloc[0] == "TestDerivedMetric"
        assert result["case_id_number"].iloc[0] == 1
        assert result["event_type"].iloc[0] == "heat_wave"
        assert result["value"].iloc[0] == 2.5

        # Verify that compute_metric was called with the derived variables
        mock_base_metric.compute_metric.assert_called_once()
        call_args = mock_base_metric.compute_metric.call_args[0]

        # The variables should be passed as derived variable instances
        assert isinstance(call_args[0], xr.DataArray)  # forecast data
        assert isinstance(call_args[1], xr.DataArray)  # target data


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_extremeweatherbench_empty_cases(self, sample_evaluation_object):
        """Test ExtremeWeatherBench with empty cases."""
        empty_cases = {"cases": []}

        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=empty_cases,
            evaluation_objects=[sample_evaluation_object],
        )

        with mock.patch("extremeweatherbench.cases.build_case_operators") as mock_build:
            mock_build.return_value = []

            result = ewb.run()

            # Should return empty DataFrame when no cases
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_compute_case_operator_exception_handling(self, sample_case_operator):
        """Test exception handling in compute_case_operator."""
        with mock.patch("extremeweatherbench.evaluate._build_datasets") as mock_build:
            mock_build.side_effect = Exception("Data loading failed")

            with pytest.raises(Exception, match="Data loading failed"):
                evaluate.compute_case_operator(sample_case_operator)

    def test_run_pipeline_missing_method(self, sample_case_operator):
        """Test run_pipeline when a required method is missing."""
        # Remove a required method
        del sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source

        with pytest.raises(AttributeError):
            evaluate.run_pipeline(
                sample_case_operator.case_metadata, sample_case_operator.forecast
            )

    def test_evaluate_metric_computation_failure(
        self,
        sample_forecast_dataset,
        sample_target_dataset,
        sample_case_operator,
        mock_base_metric,
    ):
        """Test metric evaluation when computation fails."""
        mock_base_metric.name = "FailingMetric"
        mock_base_metric.compute_metric.side_effect = Exception(
            "Metric computation failed"
        )

        with pytest.raises(Exception, match="Metric computation failed"):
            evaluate._evaluate_metric_and_return_df(
                forecast_ds=sample_forecast_dataset,
                target_ds=sample_target_dataset,
                forecast_variable="surface_air_temperature",
                target_variable="2m_temperature",
                metric=mock_base_metric,
                case_operator=sample_case_operator,
            )

    @mock.patch("extremeweatherbench.evaluate._run_serial")
    def test_run_case_operators_serial_exception(
        self, mock_run_serial, sample_case_operator
    ):
        """Test _run_case_operators handles exceptions in serial execution."""
        mock_run_serial.side_effect = Exception("Serial execution failed")

        with pytest.raises(Exception, match="Serial execution failed"):
            # Serial mode: don't pass parallel_config
            evaluate._run_case_operators([sample_case_operator], None)

    @mock.patch("extremeweatherbench.evaluate._run_parallel")
    def test_run_case_operators_parallel_exception(
        self, mock_run_parallel, sample_case_operator
    ):
        """Test _run_case_operators handles exceptions in parallel execution."""
        mock_run_parallel.side_effect = Exception("Parallel execution failed")

        with pytest.raises(Exception, match="Parallel execution failed"):
            evaluate._run_case_operators(
                [sample_case_operator],
                parallel_config={"backend": "threading", "n_jobs": 2},
            )

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_serial_case_operator_exception(
        self, mock_tqdm, mock_compute_case_operator, sample_case_operator
    ):
        """Test _run_serial handles exceptions from individual case operators."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_compute_case_operator.side_effect = Exception("Case operator failed")

        with pytest.raises(Exception, match="Case operator failed"):
            evaluate._run_serial([sample_case_operator])

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_joblib_exception(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel handles joblib Parallel exceptions."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_parallel_instance.side_effect = Exception("Joblib parallel failed")

        with pytest.raises(Exception, match="Joblib parallel failed"):
            evaluate._run_parallel(
                [sample_case_operator],
                parallel_config={"backend": "threading", "n_jobs": 2},
            )

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
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

        mock_parallel_instance = mock.Mock()
        mock_parallel_class.return_value = mock_parallel_instance
        mock_parallel_instance.side_effect = consume_generator

        with pytest.raises(Exception, match="Delayed function creation failed"):
            evaluate._run_parallel(
                [sample_case_operator],
                parallel_config={"backend": "threading", "n_jobs": 2},
            )

    @mock.patch("extremeweatherbench.evaluate._run_case_operators")
    def test_run_method_exception_propagation(
        self, mock_run_case_operators, sample_cases_dict, sample_evaluation_object
    ):
        """Test that ExtremeWeatherBench.run() propagates exceptions correctly."""
        mock_run_case_operators.side_effect = Exception("Execution failed")

        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        with pytest.raises(Exception, match="Execution failed"):
            ewb.run()

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_serial_partial_failure(self, mock_tqdm, mock_compute_case_operator):
        """Test _run_serial behavior when some case operators fail."""
        case_op_1 = mock.Mock()
        case_op_2 = mock.Mock()
        case_op_3 = mock.Mock()
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

    @mock.patch("extremeweatherbench.utils.ParallelTqdm")
    @mock.patch("joblib.delayed")
    @mock.patch("tqdm.auto.tqdm")
    def test_run_parallel_invalid_n_jobs(
        self, mock_tqdm, mock_delayed, mock_parallel_class, sample_case_operator
    ):
        """Test _run_parallel with invalid n_jobs parameter."""
        mock_tqdm.return_value = [sample_case_operator]
        mock_delayed_func = mock.Mock()
        mock_delayed.return_value = mock_delayed_func

        # Mock Parallel to raise ValueError for invalid n_jobs
        mock_parallel_class.side_effect = ValueError("Invalid n_jobs parameter")

        with pytest.raises(ValueError, match="Invalid n_jobs parameter"):
            evaluate._run_parallel(
                [sample_case_operator],
                parallel_config={"backend": "threading", "n_jobs": -5},
            )


class TestIntegration:
    """Test integration scenarios with real-like data."""

    @mock.patch("extremeweatherbench.derived.maybe_derive_variables")
    @mock.patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
    def test_end_to_end_workflow(
        self,
        mock_maybe_subset_variables,
        mock_derive_variables,
        sample_cases_dict,
        sample_evaluation_object,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test a complete end-to-end workflow."""
        mock_derive_variables.side_effect = lambda ds, variables, **kwargs: ds

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
        mock_maybe_subset_variables.return_value = sample_forecast_dataset
        sample_evaluation_object.forecast.subset_data_to_case.return_value = (
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.maybe_convert_to_dataset.return_value = (
            sample_forecast_dataset
        )
        sample_evaluation_object.forecast.add_source_to_dataset_attrs.return_value = (
            sample_forecast_dataset
        )

        sample_evaluation_object.target.open_and_maybe_preprocess_data_from_source.return_value = sample_target_dataset  # noqa: E501
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

        with mock.patch(
            "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
        ) as mock_eval:
            mock_eval.return_value = mock_result_df

            # Create and run the workflow
            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=sample_cases_dict,
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

    @mock.patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
    def test_multiple_variables_and_metrics(
        self,
        mock_maybe_subset_variables,
        sample_cases_dict,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test workflow with multiple variables and metrics."""
        # Create multiple metrics
        metric_1 = mock.Mock(spec=metrics.BaseMetric)
        metric_1.name = "Metric1"
        metric_1.forecast_variable = None
        metric_1.target_variable = None
        metric_1.return_value.name = "Metric1"
        metric_1.return_value.compute_metric.return_value = xr.DataArray(
            data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        metric_2 = mock.Mock(spec=metrics.BaseMetric)
        metric_2.name = "Metric2"
        metric_2.forecast_variable = None
        metric_2.target_variable = None
        metric_2.return_value.name = "Metric2"
        metric_2.return_value.compute_metric.return_value = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Create evaluation object with multiple metrics and variables
        eval_obj = mock.Mock(spec=inputs.EvaluationObject)
        eval_obj.event_type = "heat_wave"
        eval_obj.metric_list = [metric_1, metric_2]

        # Mock target and forecast with variables that match our datasets
        eval_obj.target = mock.Mock(spec=inputs.TargetBase)
        eval_obj.target.name = "MultiTarget"
        eval_obj.target.variables = [
            "2m_temperature"
        ]  # Only include variables that exist

        eval_obj.forecast = mock.Mock(spec=inputs.ForecastBase)
        eval_obj.forecast.name = "MultiForecast"
        eval_obj.forecast.variables = [
            "surface_air_temperature"
        ]  # Only include variables that exist

        # Setup pipeline mocks
        mock_maybe_subset_variables.return_value = sample_forecast_dataset
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

        with mock.patch(
            "extremeweatherbench.derived.maybe_derive_variables"
        ) as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            with mock.patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_eval:
                mock_eval.return_value = mock_result_df

                ewb = evaluate.ExtremeWeatherBench(
                    case_metadata=sample_cases_dict,
                    evaluation_objects=[eval_obj],
                )

                result = ewb.run()

                # Should have results for each metric combination
                assert len(result) >= 2  # At least 2 metrics * 1 case

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_serial_vs_parallel_results_consistency(
        self, mock_compute_case_operator, sample_cases_dict, sample_evaluation_object
    ):
        """Test that serial and parallel execution produce identical results."""
        # Setup mock case operators
        case_op_1 = mock.Mock()
        case_op_2 = mock.Mock()
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
            case_metadata=sample_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        with mock.patch("extremeweatherbench.cases.build_case_operators") as mock_build:
            mock_build.return_value = case_operators

            # Test serial execution
            mock_compute_case_operator.side_effect = [result_1, result_2]
            serial_result = ewb.run(n_jobs=1)

            # Reset mock and test parallel execution
            mock_compute_case_operator.reset_mock()
            mock_compute_case_operator.side_effect = [result_1, result_2]
            parallel_result = ewb.run(n_jobs=2)

            # Both should produce valid DataFrames with same structure
            assert isinstance(serial_result, pd.DataFrame)
            assert isinstance(parallel_result, pd.DataFrame)
            assert len(serial_result) == 2
            assert len(parallel_result) == 2
            assert list(serial_result.columns) == list(parallel_result.columns)

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_execution_method_performance_comparison(self, mock_compute_case_operator):
        """Test that both execution methods handle the same workload."""
        import time

        # Create many case operators to simulate realistic workload
        case_operators = [mock.Mock() for _ in range(10)]

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

        with mock.patch(
            "extremeweatherbench.utils.ParallelTqdm"
        ) as mock_parallel_class:
            mock_parallel_instance = mock.Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = mock_results

            start_time = time.time()
            parallel_result = evaluate._run_parallel(
                case_operators, parallel_config={"backend": "threading", "n_jobs": 2}
            )
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

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_mixed_execution_parameters(self, mock_compute_case_operator):
        """Test various parameter combinations for execution methods."""
        case_operators = [mock.Mock(), mock.Mock()]
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
                with mock.patch(
                    "extremeweatherbench.utils.ParallelTqdm"
                ) as mock_parallel_class:
                    mock_parallel_instance = mock.Mock()
                    mock_parallel_class.return_value = mock_parallel_instance
                    mock_parallel_instance.return_value = mock_results

                    # Add parallel_config to kwargs
                    kwargs = config.get("kwargs", {})
                    if "n_jobs" in kwargs:
                        n_jobs = kwargs.pop("n_jobs")
                        kwargs["parallel_config"] = {
                            "backend": "threading",
                            "n_jobs": n_jobs,
                        }

                    result = evaluate._run_parallel(*config["args"], **kwargs)

                    # All configurations should produce valid results
                    assert isinstance(result, list)
                    assert len(result) == 2
                    # Parallel execution is mocked, so compute_case_operator is not
                    # called
                    assert mock_compute_case_operator.call_count == 0

    def test_execution_method_kwargs_propagation(self):
        """Test that kwargs are properly propagated through execution methods."""
        case_operator = mock.Mock()

        # Mock compute_case_operator to capture kwargs
        def mock_compute_with_kwargs(case_op, cache_dir, **kwargs):
            # Store kwargs for verification
            mock_compute_with_kwargs.captured_kwargs = kwargs
            return pd.DataFrame({"value": [1.0]})

        mock_compute_with_kwargs.captured_kwargs = {}

        with mock.patch(
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
            with mock.patch(
                "extremeweatherbench.utils.ParallelTqdm"
            ) as mock_parallel_class:
                with mock.patch("joblib.delayed") as mock_delayed:
                    mock_delayed.return_value = mock_compute_with_kwargs
                    mock_parallel_instance = mock.Mock()
                    mock_parallel_class.return_value = mock_parallel_instance
                    mock_parallel_instance.return_value = [
                        pd.DataFrame({"value": [1.0]})
                    ]

                    # Reset captured kwargs
                    mock_compute_with_kwargs.captured_kwargs = {}

                    result = evaluate._run_parallel(
                        [case_operator],
                        parallel_config={"backend": "threading", "n_jobs": 2},
                        custom_param="parallel_test",
                        threshold=0.8,
                    )

                    # Verify parallel execution was set up correctly
                    mock_parallel_class.assert_called_once_with(total_tasks=1)
                    assert isinstance(result, list)

    def test_empty_case_operators_all_methods(self):
        """Test that all execution methods handle empty case operator lists."""
        # Test _run_case_operators
        result = evaluate._run_case_operators([], parallel_config={"n_jobs": 1})
        assert result == []

        result = evaluate._run_case_operators(
            [], parallel_config={"backend": "threading", "n_jobs": 2}
        )
        assert result == []

        # Test _run_serial
        result = evaluate._run_serial([])
        assert result == []

        # Test _run_parallel
        with mock.patch(
            "extremeweatherbench.utils.ParallelTqdm"
        ) as mock_parallel_class:
            mock_parallel_instance = mock.Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = []

            result = evaluate._run_parallel(
                [], parallel_config={"backend": "threading", "n_jobs": 2}
            )
            assert result == []

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_large_case_operator_list_handling(self, mock_compute_case_operator):
        """Test handling of large numbers of case operators."""
        # Create a large list of case operators
        num_cases = 100
        case_operators = [mock.Mock() for _ in range(num_cases)]

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

        with mock.patch(
            "extremeweatherbench.utils.ParallelTqdm"
        ) as mock_parallel_class:
            mock_parallel_instance = mock.Mock()
            mock_parallel_class.return_value = mock_parallel_instance
            mock_parallel_instance.return_value = mock_results

            parallel_results = evaluate._run_parallel(
                case_operators, parallel_config={"backend": "threading", "n_jobs": 4}
            )

            assert len(parallel_results) == num_cases
            mock_parallel_class.assert_called_once_with(total_tasks=100)


class TestEnsureOutputSchema:
    """Test the _ensure_output_schema function."""

    def test_ensure_output_schema_init_time_valid_time(self):
        """Test _ensure_output_schema with init_time and valid_time columns.

        init_time is now in evaluate.OUTPUT_COLUMNS, valid_time is not and will be
        dropped.
        """
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0],
                "init_time": pd.to_datetime(["2021-06-20", "2021-06-21"]),
                "valid_time": pd.to_datetime(["2021-06-21", "2021-06-22"]),
            }
        )

        result = evaluate._ensure_output_schema(
            df,
            target_variable="temperature",
            metric="TestMetric",
            case_id_number=1,
            event_type="heat_wave",
        )

        # Check all evaluate.OUTPUT_COLUMNS are present
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        # Check metadata was added
        assert all(result["target_variable"] == "temperature")
        assert all(result["metric"] == "TestMetric")
        assert all(result["case_id_number"] == 1)
        assert all(result["event_type"] == "heat_wave")
        # Check original columns preserved for those in evaluate.OUTPUT_COLUMNS
        assert len(result) == 2
        assert list(result["value"]) == [1.0, 2.0]
        # init_time should be preserved (now in evaluate.OUTPUT_COLUMNS)
        assert "init_time" in result.columns
        assert list(result["init_time"]) == [
            pd.to_datetime("2021-06-20"),
            pd.to_datetime("2021-06-21"),
        ]
        # valid_time should be dropped (not in evaluate.OUTPUT_COLUMNS)
        assert "valid_time" not in result.columns

    def test_ensure_output_schema_init_time_only(self):
        """Test _ensure_output_schema with init_time only.

        init_time is now in evaluate.OUTPUT_COLUMNS so will be preserved.
        """
        df = pd.DataFrame({"value": [1.5], "init_time": pd.to_datetime(["2021-06-20"])})

        result = evaluate._ensure_output_schema(
            df,
            target_variable="wind_speed",
            metric="RMSE",
            case_id_number=2,
            event_type="storm",
        )

        # Check all columns present
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        # lead_time should be NaN since not provided and is in evaluate.OUTPUT_COLUMNS
        assert pd.isna(result["lead_time"].iloc[0])
        # init_time should be preserved (now in evaluate.OUTPUT_COLUMNS)
        assert "init_time" in result.columns
        assert result["init_time"].iloc[0] == pd.to_datetime("2021-06-20")

    def test_ensure_output_schema_lead_time_only(self):
        """Test _ensure_output_schema with lead_time only.

        lead_time is in evaluate.OUTPUT_COLUMNS so will be preserved.
        """
        df = pd.DataFrame({"value": [2.5, 3.0], "lead_time": [6, 12]})

        result = evaluate._ensure_output_schema(
            df,
            target_variable="pressure",
            metric="MAE",
            case_id_number=3,
            event_type="freeze",
        )

        # Check all columns present
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        # lead_time should be preserved (it's in evaluate.OUTPUT_COLUMNS)
        assert list(result["lead_time"]) == [6, 12]
        # init_time should be NaN since not provided and is in evaluate.OUTPUT_COLUMNS
        assert pd.isna(result["init_time"].iloc[0])

    def test_ensure_output_schema_lead_time_valid_time(self):
        """Test _ensure_output_schema with lead_time and valid_time.

        lead_time is in evaluate.OUTPUT_COLUMNS, valid_time is not and will be dropped.
        """
        df = pd.DataFrame(
            {
                "value": [0.8],
                "lead_time": [24],
                "valid_time": pd.to_datetime(["2021-06-22"]),
            }
        )

        result = evaluate._ensure_output_schema(
            df,
            target_variable="humidity",
            metric="Bias",
            case_id_number=4,
            event_type="drought",
        )

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        assert result["lead_time"].iloc[0] == 24
        # init_time should be NaN since not provided and is in evaluate.OUTPUT_COLUMNS
        assert pd.isna(result["init_time"].iloc[0])
        # valid_time should be dropped (not in evaluate.OUTPUT_COLUMNS)
        assert "valid_time" not in result.columns

    def test_ensure_output_schema_init_time_temperature(self):
        """Test _ensure_output_schema with init_time and temperature."""
        df = pd.DataFrame(
            {
                "value": [15.5, 16.2],
                "init_time": pd.to_datetime(["2021-06-20", "2021-06-21"]),
                "temperature": [298.15, 299.15],
            }
        )

        result = evaluate._ensure_output_schema(
            df,
            target_variable="air_temp",
            metric="Correlation",
            case_id_number=5,
            event_type="heat_wave",
        )

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        # Custom column should be preserved but not part of evaluate.OUTPUT_COLUMNS
        # temperature column should not appear in final result
        assert "temperature" not in result.columns
        assert len(result) == 2

    def test_ensure_output_schema_init_time_multiple_variables(self):
        """Test _ensure_output_schema with init_time and multiple variables."""
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0],
                "init_time": pd.to_datetime(["2021-06-20", "2021-06-21", "2021-06-22"]),
                "variable1": [10, 11, 12],
                "variable2": [20, 21, 22],
                "variable3": [30, 31, 32],
            }
        )

        result = evaluate._ensure_output_schema(
            df,
            target_variable="composite",
            metric="MultiMetric",
            case_id_number=6,
            event_type="complex",
        )

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        # Extra variables should not appear in final result
        assert "variable1" not in result.columns
        assert "variable2" not in result.columns
        assert "variable3" not in result.columns
        assert len(result) == 3

    def test_ensure_output_schema_lead_time_multiple_variables(self):
        """Test _ensure_output_schema with lead_time and multiple variables."""
        df = pd.DataFrame(
            {
                "value": [5.0, 6.0],
                "lead_time": [48, 72],
                "variable1": [100, 101],
                "variable2": [200, 201],
                "variable3": [300, 301],
            }
        )

        result = evaluate._ensure_output_schema(
            df,
            target_variable="multi_var",
            metric="EnsembleMetric",
            case_id_number=7,
            event_type="ensemble",
        )

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        assert list(result["lead_time"]) == [48, 72]
        assert "variable1" not in result.columns
        assert len(result) == 2

    def test_ensure_output_schema_missing_columns_warning(self, caplog):
        """Test that missing columns generate appropriate warnings."""
        df = pd.DataFrame({"value": [1.0], "some_other_column": [42]})

        with caplog.at_level(logging.WARNING):
            result = evaluate._ensure_output_schema(
                df,
                target_variable="test_var",
                metric="TestMetric",
                case_id_number=1,
                event_type="test",
            )

        # Should warn about missing columns
        assert "Missing expected columns" in caplog.text
        # But still return properly structured DataFrame
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        assert result["value"].iloc[0] == 1.0

    def test_ensure_output_schema_no_warning_init_time_when_lead_time_present(
        self, caplog
    ):
        """Test no warning when init_time missing but lead_time present."""
        df = pd.DataFrame({"value": [1.0], "lead_time": [6]})

        with caplog.at_level(logging.WARNING):
            result = evaluate._ensure_output_schema(
                df,
                target_variable="test_var",
                metric="TestMetric",
                case_id_number=1,
                event_type="test",
            )

        # Should not warn about missing init_time when lead_time present
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        init_time_warnings = [msg for msg in warning_messages if "init_time" in msg]
        assert len(init_time_warnings) == 0

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

    def test_ensure_output_schema_no_warning_lead_time_when_init_time_present(
        self, caplog
    ):
        """Test no warning when lead_time missing but init_time present."""
        df = pd.DataFrame({"value": [1.0], "init_time": pd.to_datetime(["2021-06-20"])})

        with caplog.at_level(logging.WARNING):
            result = evaluate._ensure_output_schema(
                df,
                target_variable="test_var",
                metric="TestMetric",
                case_id_number=1,
                event_type="test",
            )

        # Should not warn about missing lead_time when init_time present
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        lead_time_warnings = [msg for msg in warning_messages if "lead_time" in msg]
        assert len(lead_time_warnings) == 0

        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

    def test_ensure_output_schema_no_missing_variables(self):
        """Test _ensure_output_schema when no variables are missing."""
        # Create a dataframe with all required evaluate.OUTPUT_COLUMNS already present
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0],
                "lead_time": [1, 2],
                "target_variable": ["temperature", "temperature"],
                "metric": ["TestMetric", "TestMetric"],
                "target_source": ["test_target", "test_target"],
                "forecast_source": ["test_forecast", "test_forecast"],
                "case_id_number": [1, 1],
                "event_type": ["heat_wave", "heat_wave"],
            }
        )

        # Call _ensure_output_schema without any additional metadata
        result = evaluate._ensure_output_schema(df)

        # Should work without warnings and preserve all columns
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        assert len(result) == 2
        assert result["value"].tolist() == [1.0, 2.0]
        assert result["lead_time"].tolist() == [1, 2]

    def test_ensure_output_schema_no_missing_with_metadata(self, caplog):
        """Test _ensure_output_schema when no variables are missing with metadata."""
        # Create a dataframe with all required evaluate.OUTPUT_COLUMNS already present
        df = pd.DataFrame(
            {
                "value": [1.5, 2.5],
                "init_time": pd.to_datetime(["2021-06-20", "2021-06-21"]),
                "target_variable": ["pressure", "pressure"],
                "metric": ["NewMetric", "NewMetric"],
                "target_source": ["obs_target", "obs_target"],
                "forecast_source": ["model_forecast", "model_forecast"],
                "case_id_number": [2, 2],
                "event_type": ["cold_wave", "cold_wave"],
            }
        )

        # Add some additional metadata that should overwrite existing values
        result = evaluate._ensure_output_schema(
            df,
            target_variable="updated_pressure",
            metric="UpdatedMetric",
            case_id_number=3,
            event_type="updated_event",
        )

        # Should work without warnings since no columns are missing
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        missing_warnings = [msg for msg in warning_messages if "Missing" in msg]
        assert len(missing_warnings) == 0

        # Should preserve structure and update metadata
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS
        assert len(result) == 2
        assert result["value"].tolist() == [1.5, 2.5]
        assert all(result["target_variable"] == "updated_pressure")
        assert all(result["metric"] == "UpdatedMetric")
        assert all(result["case_id_number"] == 3)
        assert all(result["event_type"] == "updated_event")


class ForecastDerivedVariable(derived.DerivedVariable):
    """Test derived variable for forecast data."""

    name = "derived_forecast_var"
    variables = ["surface_air_temperature"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation - just return the temperature variable."""
        return data["surface_air_temperature"]


class TargetDerivedVariable(derived.DerivedVariable):
    """Test derived variable for target data."""

    name = "derived_target_var"
    variables = ["2m_temperature"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation - just return the temperature variable."""
        return data["2m_temperature"]


class TestRegionSubsettingIntegration:
    """Test integration of region subsetting with ExtremeWeatherBench evaluation."""

    @pytest.fixture
    def multi_case_dict(self):
        """Create a cases dictionary with multiple cases."""
        return {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Heat Wave California",
                    "start_date": datetime.datetime(2021, 6, 20),
                    "end_date": datetime.datetime(2021, 6, 25),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 35.0,
                            "latitude_max": 40.0,
                            "longitude_min": -125.0,
                            "longitude_max": -120.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 2,
                    "title": "Heat Wave Texas",
                    "start_date": datetime.datetime(2021, 7, 15),
                    "end_date": datetime.datetime(2021, 7, 20),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 28.0,
                            "latitude_max": 33.0,
                            "longitude_min": -105.0,
                            "longitude_max": -95.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 3,
                    "title": "Cold Wave Canada",
                    "start_date": datetime.datetime(2021, 12, 10),
                    "end_date": datetime.datetime(2021, 12, 15),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 50.0,
                            "latitude_max": 55.0,
                            "longitude_min": -115.0,
                            "longitude_max": -105.0,
                        },
                    },
                    "event_type": "cold_wave",
                },
            ]
        }

    def test_region_filtered_evaluation_setup(
        self, multi_case_dict, sample_evaluation_object
    ):
        """Test that ExtremeWeatherBench with RegionSubsetter filters cases
        correctly."""
        # Create region subsetter for west coast only
        west_coast_region = regions.BoundingBoxRegion.create_region(
            latitude_min=30.0,
            latitude_max=45.0,
            longitude_min=-130.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(
            region=west_coast_region, method="intersects"
        )

        # Create evaluation WITH the region subsetter
        ewb_with_region = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
            region_subsetter=subsetter,
        )

        # Create evaluation WITHOUT region subsetter for comparison
        ewb_without_region = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        # Access case_operators to trigger region subsetting
        filtered_operators = ewb_with_region.case_operators
        all_operators = ewb_without_region.case_operators

        # The filtered evaluation should have fewer or equal case operators
        assert len(filtered_operators) <= len(all_operators)

        # Verify that the California case (case_id_number=1) is included
        # since it intersects with the west coast region
        filtered_case_ids = {
            op.case_metadata.case_id_number for op in filtered_operators
        }
        all_case_ids = {op.case_metadata.case_id_number for op in all_operators}

        # California case should be in filtered results
        assert 1 in filtered_case_ids

        # All filtered case IDs should be a subset of all case IDs
        assert filtered_case_ids.issubset(all_case_ids)

        # Verify that region subsetting actually happened
        # (unless all cases happen to be in the region)
        if len(multi_case_dict["cases"]) > 1:
            # At least verify the subsetter was applied
            assert ewb_with_region.region_subsetter is not None
            assert ewb_without_region.region_subsetter is None

    def test_region_subsetter_actually_filters_cases(
        self, multi_case_dict, sample_evaluation_object
    ):
        """Test that RegionSubsetter actually filters out cases outside the region."""
        # Create a very restrictive region that should only include California case
        california_only_region = regions.BoundingBoxRegion.create_region(
            latitude_min=35.0,
            latitude_max=40.0,
            longitude_min=-125.0,
            longitude_max=-120.0,
        )

        subsetter = regions.RegionSubsetter(
            region=california_only_region,
            method="all",  # Use "all" method to be more restrictive
        )

        # Create evaluation with restrictive region subsetter
        ewb_filtered = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
            region_subsetter=subsetter,
        )

        # Create evaluation without filtering
        ewb_unfiltered = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        # Get case operators
        filtered_operators = ewb_filtered.case_operators
        unfiltered_operators = ewb_unfiltered.case_operators

        # Should have fewer filtered operators than unfiltered
        assert len(filtered_operators) < len(unfiltered_operators)

        # Should have exactly 1 case (California) or 0 cases if none match
        assert len(filtered_operators) <= 1

        # If we have any filtered cases, they should be the California case
        if len(filtered_operators) > 0:
            filtered_case_ids = {
                op.case_metadata.case_id_number for op in filtered_operators
            }
            assert filtered_case_ids == {1}  # Only California case

    def test_region_subsetter_in_ewb_with_run(
        self, multi_case_dict, sample_evaluation_object
    ):
        """Test complete workflow with RegionSubsetter in ExtremeWeatherBench."""
        # Create region subsetter for west coast only
        west_coast_region = regions.BoundingBoxRegion.create_region(
            latitude_min=30.0,
            latitude_max=45.0,
            longitude_min=-130.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(
            region=west_coast_region, method="intersects"
        )

        # Create evaluation WITH region subsetter
        ewb_with_region = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
            region_subsetter=subsetter,
        )

        # Create evaluation WITHOUT region subsetter for comparison
        ewb_without_region = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        # Compare the case operators (should be fewer with region subsetter)
        with_region_operators = ewb_with_region.case_operators
        without_region_operators = ewb_without_region.case_operators

        # With region subsetting should have fewer or equal case operators
        assert len(with_region_operators) <= len(without_region_operators)

        # Both should be valid case operator lists
        assert isinstance(with_region_operators, list)
        assert isinstance(without_region_operators, list)

    @mock.patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_region_subset_evaluation_results(
        self, mock_compute_operator, multi_case_dict, sample_evaluation_object
    ):
        """Test that region subsetting produces expected evaluation results."""

        # Mock the compute_case_operator to return predictable results
        def mock_compute_side_effect(case_operator, *args, **kwargs):
            case_id = case_operator.case_metadata.case_id_number
            return pd.DataFrame(
                {
                    "value": [0.1 * case_id],
                    "metric": ["TestMetric"],
                    "case_id_number": [case_id],
                    "event_type": [case_operator.case_metadata.event_type],
                    "target_variable": ["temperature"],
                    "forecast_source": ["test_forecast"],
                    "target_source": ["test_target"],
                }
            )

        mock_compute_operator.side_effect = mock_compute_side_effect

        # Create evaluation with all cases
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=multi_case_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        # Create region subsetter
        west_coast_region = regions.BoundingBoxRegion.create_region(
            latitude_min=30.0,
            latitude_max=45.0,
            longitude_min=-130.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(
            region=west_coast_region, method="intersects"
        )

        # Load cases and apply subsetting
        original_cases = cases.load_individual_cases(multi_case_dict)
        subset_cases = subsetter.subset_case_collection(original_cases)
        # Create new evaluation with subset cases
        subset_cases_dict = {
            "cases": [
                {
                    "case_id_number": case.case_id_number,
                    "title": case.title,
                    "start_date": case.start_date,
                    "end_date": case.end_date,
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            key: bound
                            for key, bound in zip(
                                [
                                    "latitude_min",
                                    "latitude_max",
                                    "longitude_min",
                                    "longitude_max",
                                ],
                                case.location.as_geopandas().total_bounds,
                            )
                        },
                    },
                    "event_type": case.event_type,
                }
                for case in subset_cases.cases
            ]
        }

        subset_ewb = evaluate.ExtremeWeatherBench(
            case_metadata=subset_cases_dict,
            evaluation_objects=[sample_evaluation_object],
        )

        # Run both evaluations
        all_results = ewb.run(n_jobs=1)
        subset_results = subset_ewb.run(n_jobs=1)

        # Subset results should have fewer or equal cases
        assert len(subset_results) <= len(all_results)

        # All case IDs in subset should also be in full results
        subset_case_ids = set(subset_results["case_id_number"])
        all_case_ids = set(all_results["case_id_number"])
        assert subset_case_ids.issubset(all_case_ids)

    def test_region_subsetting_with_results_dataframe(
        self, multi_case_dict, sample_evaluation_object
    ):
        """Test subsetting of results DataFrame after evaluation."""
        # Create mock results that would come from evaluation
        mock_results = pd.DataFrame(
            {
                "case_id_number": [1, 1, 2, 2, 3, 3],
                "metric": ["mae", "rmse", "mae", "rmse", "mae", "rmse"],
                "value": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                "event_type": [
                    "heat_wave",
                    "heat_wave",
                    "heat_wave",
                    "heat_wave",
                    "cold_wave",
                    "cold_wave",
                ],
                "target_variable": ["temperature"] * 6,
                "forecast_source": ["test_forecast"] * 6,
                "target_source": ["test_target"] * 6,
            }
        )

        # Create region subsetter for west coast
        west_coast_region = regions.BoundingBoxRegion.create_region(
            latitude_min=30.0,
            latitude_max=45.0,
            longitude_min=-130.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(
            region=west_coast_region, method="intersects"
        )

        # Load original cases for reference
        original_cases = cases.load_individual_cases(multi_case_dict)

        # Subset the results
        from extremeweatherbench.regions import subset_results_to_region

        subset_results = subset_results_to_region(
            subsetter, mock_results, original_cases
        )

        # Should have fewer results (only for cases in the region)
        assert len(subset_results) <= len(mock_results)

        # All remaining case IDs should be from the original results
        subset_case_ids = set(subset_results["case_id_number"])
        original_case_ids = set(mock_results["case_id_number"])
        assert subset_case_ids.issubset(original_case_ids)

    def test_different_subsetting_methods_produce_different_results(
        self, multi_case_dict
    ):
        """Test that different subsetting methods produce different results."""
        # Load cases
        case_collection = cases.load_individual_cases(multi_case_dict)

        # Create a region that partially overlaps with cases
        partial_region = regions.BoundingBoxRegion.create_region(
            latitude_min=32.0,
            latitude_max=37.0,
            longitude_min=-122.0,
            longitude_max=-100.0,
        )

        # Test different methods
        intersects_subsetter = regions.RegionSubsetter(
            region=partial_region, method="intersects"
        )

        all_subsetter = regions.RegionSubsetter(region=partial_region, method="all")

        percent_low_subsetter = regions.RegionSubsetter(
            region=partial_region, method="percent", percent_threshold=0.1
        )

        percent_high_subsetter = regions.RegionSubsetter(
            region=partial_region, method="percent", percent_threshold=0.9
        )

        # Apply different methods
        intersects_cases = intersects_subsetter.subset_case_collection(case_collection)
        all_cases = all_subsetter.subset_case_collection(case_collection)
        percent_low_cases = percent_low_subsetter.subset_case_collection(
            case_collection
        )
        percent_high_cases = percent_high_subsetter.subset_case_collection(
            case_collection
        )

        # "all" should be most restrictive
        assert len(all_cases.cases) <= len(intersects_cases.cases)
        assert len(all_cases.cases) <= len(percent_low_cases.cases)
        assert len(all_cases.cases) <= len(percent_high_cases.cases)

        # High percent threshold should be more restrictive than low
        assert len(percent_high_cases.cases) <= len(percent_low_cases.cases)

    def test_region_subsetting_with_centered_regions(self, multi_case_dict):
        """Test region subsetting works with CenteredRegion targets."""
        case_collection = cases.load_individual_cases(multi_case_dict)

        # Create a centered region in Texas area
        texas_region = regions.CenteredRegion.create_region(
            latitude=30.0, longitude=-100.0, bounding_box_degrees=8.0
        )

        subsetter = regions.RegionSubsetter(region=texas_region, method="intersects")

        subset_cases = subsetter.subset_case_collection(case_collection)

        # Should work without errors
        assert isinstance(subset_cases, cases.IndividualCaseCollection)

    def test_region_subsetting_preserves_case_metadata(self, multi_case_dict):
        """Test that region subsetting preserves all case metadata."""
        case_collection = cases.load_individual_cases(multi_case_dict)

        # Create subsetter
        region = regions.BoundingBoxRegion.create_region(
            latitude_min=20.0,
            latitude_max=60.0,
            longitude_min=-130.0,
            longitude_max=-90.0,
        )

        subsetter = regions.RegionSubsetter(region=region, method="intersects")

        subset_cases = subsetter.subset_case_collection(case_collection)

        # Check that all metadata is preserved for included cases
        for case in subset_cases.cases:
            # Find the original case
            original_case = next(
                c
                for c in case_collection.cases
                if c.case_id_number == case.case_id_number
            )

            # All attributes should be identical
            assert case.title == original_case.title
            assert case.start_date == original_case.start_date
            assert case.end_date == original_case.end_date
            assert case.event_type == original_case.event_type
            assert case.case_id_number == original_case.case_id_number
            # Location regions should be equivalent (but may be different objects)
            assert isinstance(case.location, type(original_case.location))


class TestSafeConcat:
    """Test the _safe_concat helper function."""

    def test_safe_concat_with_non_empty_dataframes(self):
        """Test _safe_concat with non-empty DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes)

        # Should concatenate normally
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ["a", "b"]
        assert result["a"].tolist() == [1, 2, 5, 6]

    def test_safe_concat_with_all_empty_dataframes(self):
        """Test _safe_concat when all DataFrames are empty."""

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes)

        # Should return empty DataFrame with OUTPUT_COLUMNS
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

    def test_safe_concat_with_mixed_empty_and_non_empty(self):
        """Test _safe_concat with mix of empty and non-empty DataFrames."""
        df1 = pd.DataFrame()  # Empty
        df2 = pd.DataFrame({"value": [1.0], "metric": ["test"]})
        df3 = pd.DataFrame()  # Empty
        df4 = pd.DataFrame({"value": [2.0], "metric": ["test2"]})
        dataframes = [df1, df2, df3, df4]

        result = evaluate._safe_concat(dataframes)

        # Should only concatenate non-empty DataFrames
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["value"].tolist() == [1.0, 2.0]
        assert result["metric"].tolist() == ["test", "test2"]

    def test_safe_concat_with_ignore_index_false(self):
        """Test _safe_concat with ignore_index=False."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [3, 4]}, index=[0, 1])
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes, ignore_index=False)

        # Should preserve original indices
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.index) == [0, 1, 0, 1]

    def test_safe_concat_with_ignore_index_true(self):
        """Test _safe_concat with ignore_index=True (default)."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
        df2 = pd.DataFrame({"a": [3, 4]}, index=[20, 21])
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes, ignore_index=True)

        # Should create new sequential index
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.index) == [0, 1, 2, 3]

    def test_safe_concat_with_empty_list(self):
        """Test _safe_concat with empty list of DataFrames."""

        dataframes = []

        result = evaluate._safe_concat(dataframes)

        # Should return empty DataFrame with OUTPUT_COLUMNS
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == evaluate.OUTPUT_COLUMNS

    def test_safe_concat_prevents_future_warning(self):
        """Test that _safe_concat prevents the specific pandas FutureWarning."""
        import warnings

        # Create DataFrames that would trigger the specific FutureWarning
        # about empty or all-NA entries
        df1 = pd.DataFrame()  # Empty DataFrame
        df2 = pd.DataFrame({"a": [1], "b": [2]})
        df3 = pd.DataFrame({"a": [None], "b": [None]})  # All-NA DataFrame
        dataframes = [df1, df2, df3]

        # Test that our _safe_concat prevents the warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = evaluate._safe_concat(dataframes)

            # Check that no FutureWarnings about concatenation were raised
            future_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "DataFrame concatenation with empty or all-NA entries"
                in str(warning.message)
            ]
            assert len(future_warnings) == 0, (
                f"FutureWarning was raised: {future_warnings}"
            )

        # Should successfully concatenate without warnings
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1  # Should have at least the non-empty DataFrame

    def test_safe_concat_preserves_dtypes_when_consistent(self):
        """Test that _safe_concat preserves dtypes when they are consistent."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})  # int64, float64
        df2 = pd.DataFrame({"a": [5, 6], "b": [7.0, 8.0]})  # int64, float64
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes)

        # Should preserve original dtypes
        assert result["a"].dtype == "int64"
        assert result["b"].dtype == "float64"
        assert len(result) == 4

    def test_safe_concat_converts_to_object_when_mismatched(self):
        """Test that _safe_concat converts to object dtype when dtypes mismatch."""

        df1 = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})  # int64, float64
        df2 = pd.DataFrame(
            {"a": [pd.Timestamp("2021-01-01")], "b": ["text"]}
        )  # datetime, object
        dataframes = [df1, df2]

        result = evaluate._safe_concat(dataframes)

        # Should convert to object dtypes due to mismatches
        assert result["a"].dtype == "object"
        assert result["b"].dtype == "object"
        assert len(result) == 3


class TestMetricVariableHandling:
    """Test handling of metric-specific variables."""

    def test_collect_metric_variables_with_variables(self):
        """Test _collect_metric_variables with metrics that have variables."""
        # Create metrics with variables
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.forecast_variable = "temp"
        metric1.target_variable = "temp_obs"

        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.forecast_variable = "precip"
        metric2.target_variable = "precip_obs"

        forecast_vars, target_vars = evaluate._collect_metric_variables(
            [metric1, metric2]
        )

        assert forecast_vars == {"temp", "precip"}
        assert target_vars == {"temp_obs", "precip_obs"}

    def test_collect_metric_variables_without_variables(self):
        """Test _collect_metric_variables with metrics without variables."""
        # Create metrics without variables
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.forecast_variable = None
        metric1.target_variable = None

        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.forecast_variable = None
        metric2.target_variable = None

        forecast_vars, target_vars = evaluate._collect_metric_variables(
            [metric1, metric2]
        )

        assert forecast_vars == set()
        assert target_vars == set()

    def test_collect_metric_variables_mixed(self):
        """Test _collect_metric_variables with mixed metrics."""
        # Create mixed metrics
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.forecast_variable = "temp"
        metric1.target_variable = "temp_obs"

        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.forecast_variable = None
        metric2.target_variable = None

        forecast_vars, target_vars = evaluate._collect_metric_variables(
            [metric1, metric2]
        )

        assert forecast_vars == {"temp"}
        assert target_vars == {"temp_obs"}

    def test_collect_metric_variables_duplicates(self):
        """Test that duplicate variables are handled correctly."""
        # Create metrics with duplicate variables
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.forecast_variable = "temp"
        metric1.target_variable = "temp_obs"

        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.forecast_variable = "temp"
        metric2.target_variable = "temp_obs"

        forecast_vars, target_vars = evaluate._collect_metric_variables(
            [metric1, metric2]
        )

        # Sets should contain unique values
        assert forecast_vars == {"temp"}
        assert target_vars == {"temp_obs"}

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_with_metric_variables(
        self,
        mock_build_datasets,
        sample_individual_case,
        mock_target_base,
        mock_forecast_base,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test compute_case_operator with metrics that have variables."""
        # Setup datasets
        mock_build_datasets.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Create metric with its own variables
        metric = mock.Mock(spec=metrics.BaseMetric)
        metric.name = "MetricWithVars"
        metric.forecast_variable = "surface_air_temperature"
        metric.target_variable = "2m_temperature"
        metric.compute_metric.return_value = xr.DataArray(
            data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Create case operator
        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target_base,
            forecast=mock_forecast_base,
        )

        mock_target_base.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        with mock.patch(
            "extremeweatherbench.derived.maybe_derive_variables"
        ) as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            result = evaluate.compute_case_operator(case_operator)

            # Should call compute_metric once with metric's variables
            assert metric.compute_metric.call_count == 1
            assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate._build_datasets")
    def test_compute_case_operator_mixed_metrics(
        self,
        mock_build_datasets,
        sample_individual_case,
        sample_forecast_dataset,
        sample_target_dataset,
    ):
        """Test with mixed metrics (some with vars, some without)."""
        mock_build_datasets.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Metric with variables
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.name = "MetricWithVars"
        metric1.forecast_variable = "surface_air_temperature"
        metric1.target_variable = "2m_temperature"
        metric1.compute_metric.return_value = xr.DataArray(
            data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Metric without variables
        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.name = "MetricWithoutVars"
        metric2.forecast_variable = None
        metric2.target_variable = None
        metric2.compute_metric.return_value = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Setup InputBase with variables
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = ["2m_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = ["surface_air_temperature"]

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric1, metric2],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch(
            "extremeweatherbench.derived.maybe_derive_variables"
        ) as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            result = evaluate.compute_case_operator(case_operator)

            # metric1 called once (own vars)
            # metric2 NOT called because metric1 claims all available variables
            # This is the expected behavior: metrics with explicit variables
            # claim those variables exclusively
            assert metric1.compute_metric.call_count == 1
            assert metric2.compute_metric.call_count == 0
            assert isinstance(result, pd.DataFrame)

    def test_compute_case_operator_claimed_variables_excluded(
        self, sample_individual_case
    ):
        """Test that variables claimed by metrics with explicit vars are
        excluded from metrics without explicit vars."""
        # Create datasets with multiple variables
        time = pd.date_range("2021-06-20", freq="6h", periods=2)
        lat = [45.0]
        lon = [-120.0]

        ds = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["lead_time", "latitude", "longitude"],
                    [[[20.0]]],
                ),
                "2m_temperature": (
                    ["lead_time", "latitude", "longitude"],
                    [[[18.0]]],
                ),
                "total_precipitation": (
                    ["lead_time", "latitude", "longitude"],
                    [[[5.0]]],
                ),
                "precipitation": (
                    ["lead_time", "latitude", "longitude"],
                    [[[4.5]]],
                ),
            },
            coords={
                "lead_time": [0],
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time[:1]),
            },
        )

        # Metric with specific variable (claims "surface_air_temperature")
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.name = "MetricWithVars"
        metric1.forecast_variable = "surface_air_temperature"
        metric1.target_variable = "2m_temperature"
        metric1.compute_metric.return_value = xr.DataArray(
            data=[1.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Metric without variables (should only use unclaimed variables)
        metric2 = mock.Mock(spec=metrics.BaseMetric)
        metric2.name = "MetricWithoutVars"
        metric2.forecast_variable = None
        metric2.target_variable = None
        metric2.compute_metric.return_value = xr.DataArray(
            data=[2.0], dims=["lead_time"], coords={"lead_time": [0]}
        )

        # Setup InputBase with TWO variables
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = ["2m_temperature", "precipitation"]
        mock_target.maybe_align_forecast_to_target.return_value = (ds, ds)

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = ["surface_air_temperature", "total_precipitation"]

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric1, metric2],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = ds

            result = evaluate.compute_case_operator(case_operator)

            # metric1 called once with its specific variables
            assert metric1.compute_metric.call_count == 1

            # metric2 called once with the unclaimed variable pair
            # (total_precipitation, precipitation)
            assert metric2.compute_metric.call_count == 1

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2  # Two metric evaluations

    def test_compute_case_operator_instantiates_uninstantiated_metrics(
        self, sample_individual_case, mock_target_base, mock_forecast_base, caplog
    ):
        """Test that uninstantiated metrics are instantiated with warning."""

        # Create a metric CLASS (not instance) with required methods
        class TestMetric(metrics.BaseMetric):
            name = "TestMetric"

            def __init__(self):
                super().__init__("TestMetric")

            def _compute_metric(self, forecast, target, **kwargs):
                return xr.DataArray([1.0])

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[TestMetric],  # Pass class, not instance
            target=mock_target_base,
            forecast=mock_forecast_base,
        )

        with mock.patch(
            "extremeweatherbench.derived.maybe_derive_variables"
        ) as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            result = evaluate.compute_case_operator(case_operator)

        # Check that warning was logged about instantiation
        assert "instantiated with default parameters" in caplog.text
        assert "TestMetric" in caplog.text

        # Verify the function completed successfully
        assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate.run_pipeline")
    def test_build_datasets_augments_variables(
        self, mock_run_pipeline, sample_individual_case
    ):
        """Test that _build_datasets augments InputBase with metric vars."""
        # Create mock datasets
        mock_dataset = xr.Dataset(
            {"var1": (["time"], [1, 2, 3])},
            coords={"time": pd.date_range("2021-01-01", periods=3)},
        )
        mock_run_pipeline.return_value = mock_dataset

        # Create metrics with variables
        metric1 = mock.Mock(spec=metrics.BaseMetric)
        metric1.forecast_variable = "metric_forecast_var"
        metric1.target_variable = "metric_target_var"

        # Create InputBase objects with their own variables
        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = ["base_forecast_var"]

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = ["base_target_var"]

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric1],
            target=mock_target,
            forecast=mock_forecast,
        )

        evaluate._build_datasets(case_operator)

        # Check that run_pipeline was called twice (forecast and target)
        assert mock_run_pipeline.call_count == 2

        # Get the augmented InputBase objects passed to run_pipeline
        forecast_call = mock_run_pipeline.call_args_list[1]
        target_call = mock_run_pipeline.call_args_list[0]

        # Verify variables were augmented
        augmented_target = target_call[0][1]
        augmented_forecast = forecast_call[0][1]

        assert "base_target_var" in augmented_target.variables
        assert "metric_target_var" in augmented_target.variables
        assert "base_forecast_var" in augmented_forecast.variables
        assert "metric_forecast_var" in augmented_forecast.variables


class MockDerivedVariableWithOutputs(derived.DerivedVariable):
    """Mock DerivedVariable for testing output_variables."""

    variables = ["input_var"]

    def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Return a dataset with multiple output variables."""
        return xr.Dataset(
            {
                "output_1": data["input_var"] * 1,
                "output_2": data["input_var"] * 2,
                "output_3": data["input_var"] * 3,
            }
        )


class TestExpandDerivedVariableToOutputVariables:
    """Test _expand_derived_variable_to_output_variables function."""

    def test_expand_string_variable(self):
        """String variables return as single-item list."""
        result = evaluate._maybe_expand_derived_variable_to_output_variables("temp")
        assert result == ["temp"]
        assert isinstance(result, list)

    def test_expand_derived_variable_with_output_variables(self):
        """DerivedVariable with output_variables returns those names."""
        derived_var = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2"]
        )
        result = evaluate._maybe_expand_derived_variable_to_output_variables(
            derived_var
        )
        assert result == ["output_1", "output_2"]

    def test_expand_derived_variable_without_output_variables(self):
        """DerivedVariable without output_variables returns its name."""
        derived_var = MockDerivedVariableWithOutputs()
        result = evaluate._maybe_expand_derived_variable_to_output_variables(
            derived_var
        )
        assert result == ["MockDerivedVariableWithOutputs"]

    def test_expand_derived_variable_empty_output_variables(self):
        """DerivedVariable with empty output_variables returns its name."""
        derived_var = MockDerivedVariableWithOutputs(output_variables=[])
        result = evaluate._maybe_expand_derived_variable_to_output_variables(
            derived_var
        )
        assert result == ["MockDerivedVariableWithOutputs"]

    def test_expand_derived_variable_single_output(self):
        """DerivedVariable with single output_variable returns list."""
        derived_var = MockDerivedVariableWithOutputs(output_variables=["output_1"])
        result = evaluate._maybe_expand_derived_variable_to_output_variables(
            derived_var
        )
        assert result == ["output_1"]


class TestMetricWithOutputVariables:
    """Test metric evaluation with DerivedVariable output_variables."""

    def test_compute_case_operator_with_matching_output_variables(
        self, sample_individual_case
    ):
        """Test metrics with matching number of output_variables."""
        # Create datasets
        time = pd.date_range("2021-06-20", freq="6h", periods=10)
        lat = np.linspace(42.5, 47.5, 5)
        lon = np.linspace(-122.5, -117.5, 5)

        # Create with lead_time as a dimension
        output_1_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )
        output_2_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )

        forecast_ds = xr.Dataset(
            {
                "output_1": output_1_data,
                "output_2": output_2_data,
            }
        )

        target_ds = forecast_ds.copy(deep=True)

        # Create derived variables
        forecast_derived = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2"]
        )
        target_derived = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2"]
        )

        # Create metric with derived variables
        metric = metrics.RMSE(
            forecast_variable=forecast_derived,
            target_variable=target_derived,
        )

        # Create case operator
        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = []
        mock_forecast.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = []
        mock_target.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_run:
            mock_run.side_effect = [target_ds, forecast_ds]
            result = evaluate.compute_case_operator(case_operator)

        # Should have 20 rows (2 output variables * 10 lead_times)
        assert len(result) == 20
        assert "target_variable" in result.columns
        # Check that both output variables were evaluated
        target_vars = result["target_variable"].unique()
        assert set(target_vars) == {"output_1", "output_2"}
        # Each output variable should have 10 lead_time values
        assert len(result[result["target_variable"] == "output_1"]) == 10
        assert len(result[result["target_variable"] == "output_2"]) == 10

    def test_compute_case_operator_mismatched_output_variables(
        self, sample_individual_case
    ):
        """Test metrics with different numbers of output_variables."""
        time = pd.date_range("2021-06-20", freq="6h", periods=10)
        lat = np.linspace(42.5, 47.5, 5)
        lon = np.linspace(-122.5, -117.5, 5)

        # Create with lead_time as a dimension
        output_1_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )
        output_2_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )
        output_3_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )

        forecast_ds = xr.Dataset(
            {
                "output_1": output_1_data,
                "output_2": output_2_data,
                "output_3": output_3_data,
            }
        )

        target_ds = forecast_ds.copy(deep=True)

        # Forecast has 3 outputs, target has 2
        forecast_derived = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2", "output_3"]
        )
        target_derived = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2"]
        )

        metric = metrics.RMSE(
            forecast_variable=forecast_derived,
            target_variable=target_derived,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = []
        mock_forecast.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = []
        mock_target.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_run:
            mock_run.side_effect = [target_ds, forecast_ds]
            result = evaluate.compute_case_operator(case_operator)

        # Should have 20 rows (2 output variables * 10 lead_times)
        # Limited by target's 2 outputs
        assert len(result) == 20
        target_vars = result["target_variable"].unique()
        assert set(target_vars) == {"output_1", "output_2"}
        # Each output variable should have 10 lead_time values
        assert len(result[result["target_variable"] == "output_1"]) == 10
        assert len(result[result["target_variable"] == "output_2"]) == 10

    def test_compute_case_operator_one_string_one_derived(self, sample_individual_case):
        """Test metric with one string and one DerivedVariable."""
        time = pd.date_range("2021-06-20", freq="6h", periods=10)
        lat = np.linspace(42.5, 47.5, 5)
        lon = np.linspace(-122.5, -117.5, 5)

        # Create with lead_time as a dimension
        output_1_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )
        output_2_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )
        temp_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )

        forecast_ds = xr.Dataset(
            {
                "output_1": output_1_data,
                "output_2": output_2_data,
                "temp": temp_data,
            }
        )

        target_ds = forecast_ds.copy(deep=True)

        # Forecast is string, target is derived with 2 outputs
        target_derived = MockDerivedVariableWithOutputs(
            output_variables=["output_1", "output_2"]
        )

        metric = metrics.RMSE(forecast_variable="temp", target_variable=target_derived)

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = []
        mock_forecast.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = []
        mock_target.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_run:
            mock_run.side_effect = [target_ds, forecast_ds]
            # This should fail because we're pairing 1 forecast var with 2
            # target vars - they need to match
            result = evaluate.compute_case_operator(case_operator)

        # Should create 10 rows (1 pair: "temp" with first output * 10 lead_times)
        assert len(result) == 10

    def test_compute_case_operator_single_output_each(self, sample_individual_case):
        """Test metrics with single output_variable on each side."""
        time = pd.date_range("2021-06-20", freq="6h", periods=10)
        lat = np.linspace(42.5, 47.5, 5)
        lon = np.linspace(-122.5, -117.5, 5)

        # Create as data arrays with lead_time as a dimension
        output_1_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )

        forecast_ds = xr.Dataset({"output_1": output_1_data})

        target_ds = forecast_ds.copy(deep=True)

        forecast_derived = MockDerivedVariableWithOutputs(output_variables=["output_1"])
        target_derived = MockDerivedVariableWithOutputs(output_variables=["output_1"])

        metric = metrics.RMSE(
            forecast_variable=forecast_derived,
            target_variable=target_derived,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = []
        mock_forecast.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = []
        mock_target.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_run:
            mock_run.side_effect = [target_ds, forecast_ds]
            result = evaluate.compute_case_operator(case_operator)

        # Should have 10 rows (1 output variable * 10 lead_times)
        assert len(result) == 10
        assert all(result["target_variable"] == "output_1")

    def test_compute_case_operator_no_output_variables(self, sample_individual_case):
        """Test DerivedVariable without output_variables specified."""
        time = pd.date_range("2021-06-20", freq="6h", periods=10)
        lat = np.linspace(42.5, 47.5, 5)
        lon = np.linspace(-122.5, -117.5, 5)

        # Create with lead_time as a dimension
        mock_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=["lead_time", "latitude", "longitude"],
            coords={
                "lead_time": np.arange(0, 60, 6),
                "latitude": lat,
                "longitude": lon,
                "valid_time": ("lead_time", time),
            },
        )

        forecast_ds = xr.Dataset({"MockDerivedVariableWithOutputs": mock_data})

        target_ds = forecast_ds.copy(deep=True)

        # No output_variables specified - uses derived variable name
        forecast_derived = MockDerivedVariableWithOutputs()
        target_derived = MockDerivedVariableWithOutputs()

        metric = metrics.RMSE(
            forecast_variable=forecast_derived,
            target_variable=target_derived,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "MockForecast"
        mock_forecast.variables = []
        mock_forecast.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "MockTarget"
        mock_target.variables = []
        mock_target.maybe_align_forecast_to_target = mock.Mock(
            return_value=(forecast_ds, target_ds)
        )

        case_operator = cases.CaseOperator(
            case_metadata=sample_individual_case,
            metric_list=[metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch("extremeweatherbench.evaluate.run_pipeline") as mock_run:
            mock_run.side_effect = [target_ds, forecast_ds]
            result = evaluate.compute_case_operator(case_operator)

        # Should use the derived variable name, 10 rows (10 lead_times)
        assert len(result) == 10
        assert all(result["target_variable"] == "MockDerivedVariableWithOutputs")


if __name__ == "__main__":
    pytest.main([__file__])
