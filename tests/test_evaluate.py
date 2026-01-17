"""Tests for evaluate module (ExtremeWeatherBench class)."""

import datetime
import pathlib
import tempfile
from unittest import mock

import pandas as pd
import pytest

from extremeweatherbench import cases, evaluate, inputs, metrics, regions


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
def sample_cases_dict():
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
    return mock_target


@pytest.fixture
def mock_forecast_base():
    """Create a mock ForecastBase object."""
    mock_forecast = mock.Mock(spec=inputs.ForecastBase)
    mock_forecast.name = "MockForecast"
    mock_forecast.variables = ["surface_air_temperature"]
    return mock_forecast


@pytest.fixture
def mock_base_metric():
    """Create a mock BaseMetric object."""
    mock_metric = mock.Mock(spec=metrics.BaseMetric)
    mock_metric.name = "MockMetric"
    mock_metric.forecast_variable = None
    mock_metric.target_variable = None
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
        metric_list=[mock_base_metric],
        target=mock_target_base,
        forecast=mock_forecast_base,
    )


class TestOutputColumns:
    """Test the OUTPUT_COLUMNS constant."""

    def test_output_columns_exists(self):
        """Test that OUTPUT_COLUMNS is defined and contains expected columns."""
        from extremeweatherbench.evaluate_tools import OUTPUT_COLUMNS

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
        assert OUTPUT_COLUMNS == expected_columns


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

        mock_build_case_operators.assert_called_once()
        assert result == [sample_case_operator]

    @mock.patch("extremeweatherbench.evaluate.run_case_operators")
    def test_run_serial(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes serially."""
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

            result = ewb.run(n_jobs=1)

            mock_run_case_operators.assert_called_once_with(
                [sample_case_operator],
                cache_dir=None,
                parallel_config=None,
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    @mock.patch("extremeweatherbench.evaluate.run_case_operators")
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
                cache_dir=None,
                parallel_config={"backend": "loky", "n_jobs": 2},
            )
            assert isinstance(result, pd.DataFrame)

    @mock.patch("extremeweatherbench.evaluate.run_case_operators")
    def test_run_empty_results(
        self,
        mock_run_case_operators,
        sample_cases_dict,
        sample_evaluation_object,
    ):
        """Test the run method handles empty results."""
        from extremeweatherbench.evaluate_tools import OUTPUT_COLUMNS

        with mock.patch.object(evaluate.ExtremeWeatherBench, "case_operators", new=[]):
            mock_run_case_operators.return_value = []

            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=sample_cases_dict,
                evaluation_objects=[sample_evaluation_object],
            )

            result = ewb.run()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == OUTPUT_COLUMNS

    @mock.patch("extremeweatherbench.evaluate.run_case_operators")
    def test_run_with_caching(
        self,
        mock_run_case_operators,
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
                mock_result = [pd.DataFrame({"value": [1.0]})]
                mock_run_case_operators.return_value = mock_result

                ewb = evaluate.ExtremeWeatherBench(
                    case_metadata=sample_cases_dict,
                    evaluation_objects=[sample_evaluation_object],
                    cache_dir=cache_dir,
                )

                ewb.run(n_jobs=1)

                # Check that cache directory exists
                assert cache_dir.exists()
