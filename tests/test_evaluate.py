"""Comprehensive test suite for evaluate.py module.

This test suite covers all the main functionality of the ExtremeWeatherBench evaluation
workflow, including the ExtremeWeatherBench class, pipeline functions, and error
handling.
"""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, derived, evaluate, inputs, metrics
from extremeweatherbench.events import tropical_cyclone
from extremeweatherbench.regions import CenteredRegion

# flake8: noqa: E501

# =============================================================================
# Test Fixtures
# =============================================================================


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
        metric=[mock_base_metric],
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
        metric=mock_base_metric,
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


# =============================================================================
# Test ExtremeWeatherBench Class
# =============================================================================


class TestExtremeWeatherBench:
    """Test the ExtremeWeatherBench class."""

    def test_initialization(self, sample_cases_dict, sample_evaluation_object):
        """Test ExtremeWeatherBench initialization."""
        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            metrics=[sample_evaluation_object],
        )

        assert ewb.cases == sample_cases_dict
        assert ewb.metrics == [sample_evaluation_object]
        assert ewb.cache_dir is None

    def test_initialization_with_cache_dir(
        self, sample_cases_dict, sample_evaluation_object
    ):
        """Test ExtremeWeatherBench initialization with cache directory."""
        cache_dir = "/tmp/test_cache"
        ewb = evaluate.ExtremeWeatherBench(
            cases=sample_cases_dict,
            metrics=[sample_evaluation_object],
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
            metrics=[sample_evaluation_object],
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
            metrics=[sample_evaluation_object],
        )

        result = ewb.case_operators

        mock_build_case_operators.assert_called_once_with(
            sample_cases_dict, [sample_evaluation_object]
        )
        assert result == [sample_case_operator]

    @patch("extremeweatherbench.evaluate.compute_case_operator")
    def test_run_serial(
        self,
        mock_compute_case_operator,
        sample_cases_dict,
        sample_evaluation_object,
        sample_case_operator,
    ):
        """Test the run method executes serially."""
        # Mock the case operators property
        with patch.object(
            evaluate.ExtremeWeatherBench, "case_operators", new=[sample_case_operator]
        ):
            # Mock compute_case_operator to return a DataFrame
            mock_result = pd.DataFrame(
                {
                    "value": [1.0],
                    "metric": ["MockMetric"],
                    "case_id_number": [1],
                }
            )
            mock_compute_case_operator.return_value = mock_result

            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                metrics=[sample_evaluation_object],
            )

            result = ewb.run()

            mock_compute_case_operator.assert_called_once_with(sample_case_operator)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

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
                mock_compute_case_operator.return_value = mock_result

                ewb = evaluate.ExtremeWeatherBench(
                    cases=sample_cases_dict,
                    metrics=[sample_evaluation_object],
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
                metrics=[sample_evaluation_object],
            )

            result = ewb.run()

            assert mock_compute_case_operator.call_count == 2
            assert len(result) == 2
            assert result["case_id_number"].tolist() == [1, 2]


# =============================================================================
# Test compute_case_operator Function
# =============================================================================


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
            lambda ds, vars, **kwargs: ds
        )  # Return unchanged

        mock_result = pd.DataFrame(
            {
                "value": [1.0],
                "metric": ["MockMetric"],
                "case_id_number": [1],
            }
        )
        mock_evaluate_metric.return_value = mock_result

        # Setup the case operator mocks - metric should be a list for iteration
        sample_case_operator.metric = [mock_base_metric]
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
        mock_derive_variables.side_effect = lambda ds, vars, **kwargs: ds

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )
        sample_case_operator.metric = [Mock()]

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
        sample_case_operator.metric = [metric_1, metric_2]

        sample_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        with patch("extremeweatherbench.derived.maybe_derive_variables") as mock_derive:
            mock_derive.side_effect = lambda ds, vars, **kwargs: ds

            with patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_evaluate:
                mock_evaluate.return_value = pd.DataFrame({"value": [1.0]})

                result = evaluate.compute_case_operator(sample_case_operator)

                # Should be called twice (once for each metric)
                assert mock_evaluate.call_count == 2
                assert len(result) == 2


# =============================================================================
# Test Pipeline Functions
# =============================================================================


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

    def test_run_pipeline_forecast(self, sample_case_operator, sample_forecast_dataset):
        """Test run_pipeline function for forecast data."""
        # Mock the pipeline methods
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = sample_forecast_dataset
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
        sample_case_operator.forecast.open_and_maybe_preprocess_data_from_source.assert_called_once()
        sample_case_operator.forecast.maybe_map_variable_names.assert_called_once()
        # The pipe() method passes the dataset as first argument, then case_operator as kwarg
        assert sample_case_operator.forecast.subset_data_to_case.call_count == 1
        call_args = sample_case_operator.forecast.subset_data_to_case.call_args
        assert call_args[1]["case_operator"] == sample_case_operator
        sample_case_operator.forecast.maybe_convert_to_dataset.assert_called_once()
        sample_case_operator.forecast.add_source_to_dataset_attrs.assert_called_once()

    def test_run_pipeline_target(self, sample_case_operator, sample_target_dataset):
        """Test run_pipeline function for target data."""
        # Mock the pipeline methods
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.return_value = sample_target_dataset
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
        sample_case_operator.target.open_and_maybe_preprocess_data_from_source.assert_called_once()

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


# =============================================================================
# Test Metric Evaluation
# =============================================================================


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


# =============================================================================
# Test Error Handling and Edge Cases
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_extremeweatherbench_empty_cases(self, sample_evaluation_object):
        """Test ExtremeWeatherBench with empty cases."""
        empty_cases = {"cases": []}

        ewb = evaluate.ExtremeWeatherBench(
            cases=empty_cases,
            metrics=[sample_evaluation_object],
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


# =============================================================================
# Test Integration Scenarios
# =============================================================================


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
        mock_derive_variables.side_effect = lambda ds, vars, **kwargs: ds

        # Setup the evaluation object methods
        sample_evaluation_object.target.maybe_align_forecast_to_target.return_value = (
            sample_forecast_dataset,
            sample_target_dataset,
        )

        # Mock the pipeline methods to return our test datasets
        sample_evaluation_object.forecast.open_and_maybe_preprocess_data_from_source.return_value = sample_forecast_dataset  # noqa: E501
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

        with patch(
            "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
        ) as mock_eval:
            mock_eval.return_value = mock_result_df

            # Create and run the workflow
            ewb = evaluate.ExtremeWeatherBench(
                cases=sample_cases_dict,
                metrics=[sample_evaluation_object],
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
        eval_obj.metric = [metric_1, metric_2]

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
            mock_derive.side_effect = lambda ds, vars, **kwargs: ds

            with patch(
                "extremeweatherbench.evaluate._evaluate_metric_and_return_df"
            ) as mock_eval:
                mock_eval.return_value = mock_result_df

                ewb = evaluate.ExtremeWeatherBench(
                    cases=sample_cases_dict,
                    metrics=[eval_obj],
                )

                result = ewb.run()

                # Should have results for each metric combination
                assert len(result) >= 2  # At least 2 metrics * 1 case


@pytest.fixture
def sample_tc_case_operator():
    """Create a sample case operator for TC evaluation."""
    # Create sample case metadata
    from extremeweatherbench.regions import CenteredRegion

    case_metadata = cases.IndividualCase(
        case_id_number=156,
        title="Test TC Case",
        start_date=pd.Timestamp("2023-09-01"),
        end_date=pd.Timestamp("2023-09-05"),
        location=CenteredRegion(25.0, -75.0, 5.0),
        event_type="tropical_cyclone",
    )

    # Create mock target (IBTrACS)
    mock_target = MagicMock(spec=inputs.IBTrACS)
    mock_target.__class__.__name__ = "IBTrACS"
    mock_target.variables = [derived.TrackSeaLevelPressure]

    # Create mock forecast
    mock_forecast = MagicMock(spec=inputs.KerchunkForecast)
    mock_forecast.variables = [derived.TrackSeaLevelPressure]

    # Create mock metric
    mock_metric = MagicMock(spec=metrics.BaseMetric)

    case_operator = cases.CaseOperator(
        case_metadata=case_metadata,
        metric=[mock_metric],
        target=mock_target,
        forecast=mock_forecast,
    )

    return case_operator


@pytest.fixture
def sample_ibtracs_dataset():
    """Create a sample IBTrACS dataset for testing."""
    valid_time = pd.date_range("2023-09-01", periods=10, freq="6h")

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["valid_time"],
                np.random.uniform(950, 1010, len(valid_time)),
            ),
            "latitude": (["valid_time"], np.linspace(15, 30, len(valid_time))),
            "longitude": (["valid_time"], np.linspace(-80, -70, len(valid_time))),
        },
        coords={"valid_time": valid_time},
    )
    dataset.attrs["source"] = "IBTrACS"

    return dataset


@pytest.fixture
def sample_tc_forecast_dataset():
    """Create a sample TC forecast dataset."""
    time = pd.date_range("2023-09-01", periods=2, freq="12h")
    prediction_timedelta = np.array([0, 12, 24, 36], dtype="timedelta64[h]")

    # Create sample TC track data
    data_shape = (len(time), len(prediction_timedelta))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "prediction_timedelta"],
                np.random.normal(101000, 1000, data_shape),
            ),
        },
        coords={
            "time": time,
            "prediction_timedelta": prediction_timedelta,
            "latitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(15, 35, data_shape),
            ),
            "longitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(-85, -65, data_shape),
            ),
        },
    )
    dataset.attrs["source"] = "Test Forecast"

    return dataset


class TestTropicalCycloneEvaluation:
    """Test tropical cyclone specific evaluation functionality."""

    def setup_method(self):
        """Clear registries before each test."""
        tropical_cyclone.clear_ibtracs_registry()

    @patch("extremeweatherbench.evaluate._build_datasets")
    @patch("extremeweatherbench.derived.maybe_derive_variables")
    def test_ibtracs_registration_during_evaluation(
        self,
        mock_derive_vars,
        mock_build_datasets,
        sample_tc_case_operator,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test that IBTrACS data is registered during evaluation."""
        # Setup mocks
        mock_build_datasets.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        def mock_derive_side_effect(ds, variables, **kwargs):
            # Create a copy of the dataset and add the derived variable
            ds_copy = ds.copy()
            # Add TrackSeaLevelPressure as a derived variable
            ds_copy["TrackSeaLevelPressure"] = xr.DataArray(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                dims=["time", "prediction_timedelta"],
                coords={
                    "time": ds.time,
                    "prediction_timedelta": ds.prediction_timedelta,
                },
            )
            return ds_copy

        mock_derive_vars.side_effect = mock_derive_side_effect

        # Mock the target's maybe_align_forecast_to_target method
        sample_tc_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        # Mock the metric computation
        mock_metric_instance = MagicMock()
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        sample_tc_case_operator.metric[0].return_value = mock_metric_instance

        # Run the evaluation
        evaluate.compute_case_operator(sample_tc_case_operator)

        # Check that IBTrACS data was registered
        case_id = str(sample_tc_case_operator.case_metadata.case_id_number)
        registered_data = tropical_cyclone.get_ibtracs_data(case_id)

        assert registered_data is not None
        xr.testing.assert_equal(registered_data, sample_ibtracs_dataset)

    @patch("extremeweatherbench.evaluate._build_datasets")
    @patch("extremeweatherbench.derived.maybe_derive_variables")
    def test_non_ibtracs_target_no_registration(
        self,
        mock_derive_vars,
        mock_build_datasets,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test that non-IBTrACS targets don't trigger registration."""
        # Create case operator with non-IBTrACS target
        from extremeweatherbench.regions import CenteredRegion

        case_metadata = cases.IndividualCase(
            case_id_number=157,
            title="Test Non-TC Case",
            start_date=pd.Timestamp("2023-09-01"),
            end_date=pd.Timestamp("2023-09-05"),
            location=CenteredRegion(25.0, -75.0, 5.0),
            event_type="heat_wave",
        )

        mock_target = MagicMock(spec=inputs.ERA5)
        mock_target.__class__.__name__ = "ERA5"
        mock_target.variables = ["surface_air_temperature"]

        mock_forecast = MagicMock(spec=inputs.KerchunkForecast)
        mock_forecast.variables = ["surface_air_temperature"]

        mock_metric = MagicMock(spec=metrics.BaseMetric)

        case_operator = cases.CaseOperator(
            case_metadata=case_metadata,
            metric=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        # Setup mocks
        mock_build_datasets.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )
        mock_derive_vars.side_effect = lambda ds, variables, **kwargs: ds

        case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        mock_metric_instance = MagicMock()
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        case_operator.metric[0].return_value = mock_metric_instance

        # Run the evaluation
        evaluate.compute_case_operator(case_operator)

        # Check that IBTrACS data was NOT registered
        case_id = str(case_operator.case_metadata.case_id_number)
        registered_data = tropical_cyclone.get_ibtracs_data(case_id)

        assert registered_data is None

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_case_id_passed_to_derive_variables(
        self,
        mock_build_datasets,
        sample_tc_case_operator,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test that case_id is passed to maybe_derive_variables."""
        # Setup mocks
        mock_build_datasets.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        sample_tc_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        mock_metric_instance = MagicMock()
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        sample_tc_case_operator.metric[0].return_value = mock_metric_instance

        with patch("extremeweatherbench.derived.maybe_derive_variables") as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

            # Run the evaluation
            evaluate.compute_case_operator(sample_tc_case_operator)

            # Check that maybe_derive_variables was called with case_id
            assert mock_derive.call_count == 2  # Called for both forecast and target

            # Check that case_id was passed in kwargs
            for call in mock_derive.call_args_list:
                kwargs = call[1]
                assert "case_id" in kwargs
                assert kwargs["case_id"] == str(
                    sample_tc_case_operator.case_metadata.case_id_number
                )


class TestDerivedVariableIntegration:
    """Test integration of derived variables in evaluation."""

    def setup_method(self):
        """Clear caches before each test."""
        tropical_cyclone.clear_ibtracs_registry()
        derived.TropicalCycloneTrackVariable.clear_cache()

    @patch(
        "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
    )
    @patch("extremeweatherbench.events.tropical_cyclone.generate_tc_variables")
    def test_maybe_derive_variables_with_tc_tracks(
        self, mock_generate_tc_vars, mock_create_tracks, sample_tc_forecast_dataset
    ):
        """Test maybe_derive_variables with TC track variables."""
        # Create a dataset that needs TC track derivation
        base_dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 10, 10)),
                ),
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 10, 10)),
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 10, 10)),
                ),
                "geopotential": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(5000, 1000, (2, 10, 10)) * 9.80665,
                ),
            },
            coords={
                "time": pd.date_range("2023-09-01", periods=2, freq="6h"),
                "latitude": np.linspace(20, 30, 10),
                "longitude": np.linspace(-80, -70, 10),
            },
        )

        # Mock the track computation
        mock_tracks = xr.Dataset(
            {
                "tc_slp": (
                    ["time", "prediction_timedelta"],
                    [[101000, 101010], [101020, 101030]],
                ),
                "tc_latitude": (
                    ["time", "prediction_timedelta"],
                    [[25.0, 25.5], [26.0, 26.5]],
                ),
                "tc_longitude": (
                    ["time", "prediction_timedelta"],
                    [[-75.0, -74.5], [-74.0, -73.5]],
                ),
                "tc_vmax": (
                    ["time", "prediction_timedelta"],
                    [[25.0, 26.0], [27.0, 28.0]],
                ),
            }
        )

        mock_generate_tc_vars.return_value = base_dataset
        mock_create_tracks.return_value = mock_tracks

        # Register IBTrACS data for the case
        ibtracs_data = xr.Dataset(
            {
                "latitude": (["time"], [25.0, 26.0]),
                "longitude": (["time"], [-75.0, -74.0]),
            },
            coords={"time": pd.date_range("2023-09-01", periods=2, freq="6h")},
        )

        tropical_cyclone.register_ibtracs_data("test_case", ibtracs_data)

        # Test derivation
        variables = [derived.TrackSeaLevelPressure]
        result = derived.maybe_derive_variables(
            base_dataset, variables, case_id="test_case"
        )

        # Should have the derived variable
        assert isinstance(result, xr.Dataset)
        assert "TrackSeaLevelPressure" in result.data_vars

    def test_maybe_pull_required_variables_from_derived_input_tc(self):
        """Test pulling required variables from TC derived variables."""
        incoming_variables = [
            "existing_variable",
            derived.TrackSeaLevelPressure,
            derived.TrackSurfaceWindSpeed,
        ]

        result = derived.maybe_pull_required_variables_from_derived_input(
            incoming_variables
        )

        # Should include original string variable and all required variables from derived variables
        expected_vars = [
            "existing_variable",
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

        for var in expected_vars:
            assert var in result

        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_maybe_pull_required_variables_with_instances_and_classes(self):
        """Test with both instances and classes of derived variables."""
        track_slp_instance = derived.TrackSeaLevelPressure()

        incoming_variables = [
            "base_variable",
            track_slp_instance,  # Instance
            derived.TrackSurfaceWindSpeed,  # Class
        ]

        result = derived.maybe_pull_required_variables_from_derived_input(
            incoming_variables
        )

        # Should handle both instances and classes
        expected_vars = [
            "base_variable",
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

        for var in expected_vars:
            assert var in result


@pytest.mark.integration
class TestFullTCEvaluationWorkflow:
    """Integration tests for the full TC evaluation workflow."""

    def setup_method(self):
        """Clear all caches and registries."""
        tropical_cyclone.clear_ibtracs_registry()
        derived.TropicalCycloneTrackVariable.clear_cache()

    @patch("extremeweatherbench.evaluate._build_datasets")
    @patch(
        "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
    )
    @patch("extremeweatherbench.events.tropical_cyclone.generate_tc_variables")
    def test_full_workflow_mock(
        self,
        mock_generate_tc_vars,
        mock_create_tracks,
        mock_build_datasets,
        sample_tc_case_operator,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test the full TC evaluation workflow with mocked computations."""
        # Setup the forecast dataset to have required variables
        forecast_with_tc_vars = sample_tc_forecast_dataset.copy()
        forecast_with_tc_vars.update(
            {
                "surface_eastward_wind": (
                    ["time", "prediction_timedelta"],
                    np.random.normal(
                        0,
                        10,
                        sample_tc_forecast_dataset.air_pressure_at_mean_sea_level.shape,
                    ),
                ),
                "surface_northward_wind": (
                    ["time", "prediction_timedelta"],
                    np.random.normal(
                        0,
                        10,
                        sample_tc_forecast_dataset.air_pressure_at_mean_sea_level.shape,
                    ),
                ),
                "geopotential": (
                    ["time", "prediction_timedelta"],
                    np.random.normal(
                        5000,
                        1000,
                        sample_tc_forecast_dataset.air_pressure_at_mean_sea_level.shape,
                    )
                    * 9.80665,
                ),
            }
        )

        # Mock the builds
        mock_build_datasets.return_value = (
            forecast_with_tc_vars,
            sample_ibtracs_dataset,
        )
        mock_generate_tc_vars.return_value = forecast_with_tc_vars

        # Mock track computation result
        mock_tracks = xr.Dataset(
            {
                "tc_slp": sample_tc_forecast_dataset.air_pressure_at_mean_sea_level,
                "tc_latitude": sample_tc_forecast_dataset.latitude,
                "tc_longitude": sample_tc_forecast_dataset.longitude,
                "tc_vmax": (
                    ["time", "prediction_timedelta"],
                    np.random.uniform(
                        20,
                        50,
                        sample_tc_forecast_dataset.air_pressure_at_mean_sea_level.shape,
                    ),
                ),
            }
        )
        mock_create_tracks.return_value = mock_tracks

        # Setup case operator mocks
        sample_tc_case_operator.target.maybe_align_forecast_to_target.return_value = (
            forecast_with_tc_vars,
            sample_ibtracs_dataset,
        )

        # Mock metric computation
        mock_metric_instance = MagicMock()
        mock_metric_instance.name = "TestTCMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray(
            [0.5],
            dims=["case"],
            coords={"case": [sample_tc_case_operator.case_metadata.case_id_number]},
        )
        sample_tc_case_operator.metric[0].return_value = mock_metric_instance

        # Run the evaluation
        result = evaluate.compute_case_operator(sample_tc_case_operator)

        # Should return a DataFrame with results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check that IBTrACS data was registered
        case_id = str(sample_tc_case_operator.case_metadata.case_id_number)
        registered_data = tropical_cyclone.get_ibtracs_data(case_id)
        assert registered_data is not None

        # Check that the metric was computed
        mock_metric_instance.compute_metric.assert_called_once()

        # Check that proper functions were called for TC processing
        mock_generate_tc_vars.assert_called()
        mock_create_tracks.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
