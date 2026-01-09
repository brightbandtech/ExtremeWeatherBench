"""Tests for evaluate_tools module.

Tests are designed to minimize mocking by using synthetic datasets and real
metric classes wherever possible.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, metrics
from extremeweatherbench import evaluate_tools as et

# =============================================================================
# Fixtures - Synthetic Data
# =============================================================================


@pytest.fixture
def synthetic_forecast_ds():
    """Create a synthetic forecast dataset."""
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(3, 10, 10),
                dims=["lead_time", "latitude", "longitude"],
                coords={
                    "lead_time": pd.timedelta_range("1D", periods=3, freq="1D"),
                    "latitude": np.linspace(30, 40, 10),
                    "longitude": np.linspace(-100, -90, 10),
                },
            ),
            "precipitation": xr.DataArray(
                np.random.rand(3, 10, 10),
                dims=["lead_time", "latitude", "longitude"],
                coords={
                    "lead_time": pd.timedelta_range("1D", periods=3, freq="1D"),
                    "latitude": np.linspace(30, 40, 10),
                    "longitude": np.linspace(-100, -90, 10),
                },
            ),
        }
    )


@pytest.fixture
def synthetic_target_ds():
    """Create a synthetic target dataset."""
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.random.randn(3, 10, 10),
                dims=["lead_time", "latitude", "longitude"],
                coords={
                    "lead_time": pd.timedelta_range("1D", periods=3, freq="1D"),
                    "latitude": np.linspace(30, 40, 10),
                    "longitude": np.linspace(-100, -90, 10),
                },
            ),
            "precipitation": xr.DataArray(
                np.random.rand(3, 10, 10),
                dims=["lead_time", "latitude", "longitude"],
                coords={
                    "lead_time": pd.timedelta_range("1D", periods=3, freq="1D"),
                    "latitude": np.linspace(30, 40, 10),
                    "longitude": np.linspace(-100, -90, 10),
                },
            ),
        }
    )


@pytest.fixture
def prepared_datasets(synthetic_forecast_ds, synthetic_target_ds):
    """Create PreparedDatasets from synthetic data."""
    return et.PreparedDatasets(
        forecast=synthetic_forecast_ds,
        target=synthetic_target_ds,
    )


@pytest.fixture
def simple_metric():
    """Create a simple metric instance."""
    return metrics.MeanAbsoluteError()


@pytest.fixture
def metric_with_variables():
    """Create a metric with explicit variables."""
    return metrics.MeanAbsoluteError(
        forecast_variable="temperature",
        target_variable="temperature",
    )


@pytest.fixture
def mock_case_metadata():
    """Create mock case metadata."""
    return mock.Mock(
        case_id_number=1,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-01-05"),
    )


@pytest.fixture
def mock_input_base():
    """Create a mock InputBase with variables."""
    input_base = mock.Mock()
    input_base.name = "test_forecast"
    input_base.variables = ["temperature", "precipitation"]
    return input_base


@pytest.fixture
def mock_target_base():
    """Create a mock target InputBase."""
    target_base = mock.Mock()
    target_base.name = "test_target"
    target_base.variables = ["temperature", "precipitation"]
    target_base.maybe_align_forecast_to_target = lambda f, t: (f, t)
    return target_base


@pytest.fixture
def mock_case_operator(
    mock_case_metadata, mock_input_base, mock_target_base, simple_metric
):
    """Create a mock case operator with one metric."""
    case_op = mock.Mock(spec=cases.CaseOperator)
    case_op.case_metadata = mock_case_metadata
    case_op.forecast = mock_input_base
    case_op.target = mock_target_base
    case_op.metric_list = [simple_metric]
    return case_op


# =============================================================================
# Test: dataset_cache context manager
# =============================================================================


class TestDatasetCache:
    """Tests for the dataset_cache context manager."""

    def test_creates_temp_directory_when_none_provided(self):
        """Test that a temp directory is created when cache_dir is None."""
        with et.dataset_cache() as memory:
            assert memory is not None
            assert hasattr(memory, "cache")
            # Memory location should be in a temp dir
            assert "ewb_cache_" in memory.location

    def test_uses_provided_directory(self, tmp_path):
        """Test that the provided directory is used."""
        cache_dir = tmp_path / "my_cache"
        with et.dataset_cache(cache_dir) as memory:
            assert memory is not None
            assert str(cache_dir) in memory.location

    def test_memory_can_cache_function(self):
        """Test that the memory instance can cache functions."""
        call_count = 0

        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        with et.dataset_cache() as memory:
            cached_func = memory.cache(expensive_func)
            result1 = cached_func(5)
            result2 = cached_func(5)

            assert result1 == 10
            assert result2 == 10
            assert call_count == 1  # Only called once due to caching


# =============================================================================
# Test: make_cache_key
# =============================================================================


class TestMakeCacheKey:
    """Tests for the make_cache_key function."""

    def test_returns_string(self, mock_case_operator):
        """Test that make_cache_key returns a string."""
        key = et.make_cache_key(mock_case_operator)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_same_input_same_key(self, mock_case_operator):
        """Test that the same input produces the same key."""
        key1 = et.make_cache_key(mock_case_operator)
        key2 = et.make_cache_key(mock_case_operator)
        assert key1 == key2

    def test_different_case_id_different_key(
        self, mock_case_operator, mock_case_metadata
    ):
        """Test that different case IDs produce different keys."""
        key1 = et.make_cache_key(mock_case_operator)

        mock_case_metadata.case_id_number = 999
        key2 = et.make_cache_key(mock_case_operator)

        assert key1 != key2


# =============================================================================
# Test: validate_case_operator_metrics
# =============================================================================


class TestValidateCaseOperatorMetrics:
    """Tests for validate_case_operator_metrics function."""

    def test_instantiates_metric_class(self, mock_case_operator):
        """Test that metric classes are instantiated."""
        mock_case_operator.metric_list = [metrics.MeanAbsoluteError]

        # Create a proper dataclass mock
        case_op = mock.Mock(spec=cases.CaseOperator)
        case_op.case_metadata = mock_case_operator.case_metadata
        case_op.forecast = mock_case_operator.forecast
        case_op.target = mock_case_operator.target
        case_op.metric_list = [metrics.MeanAbsoluteError]

        with mock.patch("dataclasses.replace") as mock_replace:
            mock_replace.return_value = case_op
            result = et.validate_case_operator_metrics(case_op)

            # Should have called replace to update metric_list
            mock_replace.assert_called_once()
            call_kwargs = mock_replace.call_args[1]
            # Verify the metric was instantiated
            assert isinstance(call_kwargs["metric_list"][0], metrics.MeanAbsoluteError)

    def test_raises_on_invalid_metric(self, mock_case_operator):
        """Test that invalid metrics raise TypeError."""
        mock_case_operator.metric_list = ["not_a_metric"]

        with pytest.raises(TypeError, match="BaseMetric instance"):
            et.validate_case_operator_metrics(mock_case_operator)


# =============================================================================
# Test: collect_claimed_variables
# =============================================================================


class TestCollectClaimedVariables:
    """Tests for collect_claimed_variables function - no mocks needed."""

    def test_returns_empty_sets_for_no_explicit_vars(self):
        """Test that empty sets are returned when no explicit vars."""
        metric_list = [
            metrics.MeanAbsoluteError(),
            metrics.RootMeanSquaredError(),
        ]
        forecast_vars, target_vars = et.collect_claimed_variables(metric_list)

        assert forecast_vars == set()
        assert target_vars == set()

    def test_collects_explicit_variables(self):
        """Test that explicit variables are collected."""
        metric_list = [
            metrics.MeanAbsoluteError(
                forecast_variable="temperature",
                target_variable="temperature",
            ),
            metrics.RootMeanSquaredError(
                forecast_variable="precipitation",
                target_variable="precipitation",
            ),
        ]
        forecast_vars, target_vars = et.collect_claimed_variables(metric_list)

        assert forecast_vars == {"temperature", "precipitation"}
        assert target_vars == {"temperature", "precipitation"}

    def test_handles_mixed_metrics(self):
        """Test with mix of explicit and non-explicit metrics."""
        metric_list = [
            metrics.MeanAbsoluteError(
                forecast_variable="temperature",
                target_variable="temperature",
            ),
            metrics.RootMeanSquaredError(),  # No explicit vars
        ]
        forecast_vars, target_vars = et.collect_claimed_variables(metric_list)

        assert forecast_vars == {"temperature"}
        assert target_vars == {"temperature"}


# =============================================================================
# Test: get_variable_pairs_for_metric
# =============================================================================


class TestGetVariablePairsForMetric:
    """Tests for get_variable_pairs_for_metric function."""

    def test_uses_explicit_variables(self, mock_case_operator):
        """Test that explicit variables are used when provided."""
        metric = metrics.MeanAbsoluteError(
            forecast_variable="temperature",
            target_variable="temperature",
        )
        pairs = et.get_variable_pairs_for_metric(
            metric, mock_case_operator, set(), set()
        )

        assert pairs == [("temperature", "temperature")]

    def test_uses_all_available_when_no_explicit(self, mock_case_operator):
        """Test that all available variables are used when no explicit."""
        metric = metrics.MeanAbsoluteError()
        pairs = et.get_variable_pairs_for_metric(
            metric, mock_case_operator, set(), set()
        )

        assert len(pairs) == 2
        assert ("temperature", "temperature") in pairs
        assert ("precipitation", "precipitation") in pairs

    def test_excludes_claimed_variables(self, mock_case_operator):
        """Test that claimed variables are excluded."""
        metric = metrics.MeanAbsoluteError()
        claimed = {"temperature"}
        pairs = et.get_variable_pairs_for_metric(
            metric, mock_case_operator, claimed, claimed
        )

        assert pairs == [("precipitation", "precipitation")]


# =============================================================================
# Test: create_jobs_for_case
# =============================================================================


class TestCreateJobsForCase:
    """Tests for create_jobs_for_case function."""

    def test_creates_job_for_single_metric(
        self, mock_case_operator, prepared_datasets
    ):
        """Test that a job is created for a single metric."""
        jobs = et.create_jobs_for_case(mock_case_operator, prepared_datasets)

        # Should create 2 jobs (one per variable pair)
        assert len(jobs) == 2
        assert all(isinstance(job, et.MetricJob) for job in jobs)

    def test_job_has_correct_attributes(
        self, mock_case_operator, prepared_datasets
    ):
        """Test that jobs have correct attributes."""
        mock_case_operator.metric_list = [
            metrics.MeanAbsoluteError(
                forecast_variable="temperature",
                target_variable="temperature",
            )
        ]
        jobs = et.create_jobs_for_case(mock_case_operator, prepared_datasets)

        assert len(jobs) == 1
        job = jobs[0]
        assert job.case_operator == mock_case_operator
        assert isinstance(job.metric, metrics.MeanAbsoluteError)
        assert job.forecast_var == "temperature"
        assert job.target_var == "temperature"

    def test_creates_jobs_for_multiple_metrics(
        self, mock_case_operator, prepared_datasets
    ):
        """Test that jobs are created for multiple metrics."""
        mock_case_operator.metric_list = [
            metrics.MeanAbsoluteError(
                forecast_variable="temperature",
                target_variable="temperature",
            ),
            metrics.RootMeanSquaredError(
                forecast_variable="precipitation",
                target_variable="precipitation",
            ),
        ]
        jobs = et.create_jobs_for_case(mock_case_operator, prepared_datasets)

        assert len(jobs) == 2

    def test_creates_multiple_jobs_for_multiple_variable_pairs(
        self, mock_case_operator, prepared_datasets
    ):
        """Test jobs are created for each variable pair."""
        # Metric without explicit vars should get all variable pairs
        mock_case_operator.metric_list = [metrics.MeanAbsoluteError()]
        jobs = et.create_jobs_for_case(mock_case_operator, prepared_datasets)

        # Should have 2 jobs (temperature, precipitation)
        assert len(jobs) == 2
        vars_in_jobs = {job.forecast_var for job in jobs}
        assert vars_in_jobs == {"temperature", "precipitation"}


# =============================================================================
# Test: PreparedDatasets dataclass
# =============================================================================


class TestPreparedDatasets:
    """Tests for PreparedDatasets dataclass."""

    def test_stores_datasets(self, synthetic_forecast_ds, synthetic_target_ds):
        """Test that datasets are stored correctly."""
        datasets = et.PreparedDatasets(
            forecast=synthetic_forecast_ds,
            target=synthetic_target_ds,
        )

        assert datasets.forecast is synthetic_forecast_ds
        assert datasets.target is synthetic_target_ds


# =============================================================================
# Test: MetricJob dataclass
# =============================================================================


class TestMetricJob:
    """Tests for MetricJob dataclass."""

    def test_stores_all_attributes(self, mock_case_operator, simple_metric):
        """Test that all attributes are stored correctly."""
        job = et.MetricJob(
            case_operator=mock_case_operator,
            metric=simple_metric,
            forecast_var="temperature",
            target_var="temperature",
            metric_kwargs={"key": "value"},
        )

        assert job.case_operator == mock_case_operator
        assert job.metric == simple_metric
        assert job.forecast_var == "temperature"
        assert job.target_var == "temperature"
        assert job.metric_kwargs == {"key": "value"}


# =============================================================================
# Test: build_metric_jobs (integration-ish)
# =============================================================================


class TestBuildMetricJobs:
    """Tests for build_metric_jobs function."""

    def test_returns_empty_list_for_empty_input(self):
        """Test that empty input returns empty list."""
        with et.dataset_cache() as memory:
            jobs = et.build_metric_jobs([], memory)
            assert jobs == []

    def test_serial_path_with_mocked_datasets(self, mock_case_operator):
        """Test serial job building with mocked dataset preparation."""
        with et.dataset_cache() as memory:
            with mock.patch.object(
                et, "process_case_operator"
            ) as mock_process:
                mock_job = et.MetricJob(
                    case_operator=mock_case_operator,
                    metric=metrics.MeanAbsoluteError(),
                    forecast_var="temp",
                    target_var="temp",
                    metric_kwargs={},
                )
                mock_process.return_value = [mock_job]

                jobs = et.build_metric_jobs(
                    [mock_case_operator], memory, parallel_config=None
                )

                assert len(jobs) == 1
                mock_process.assert_called_once()

    def test_parallel_path_called_when_config_provided(self, mock_case_operator):
        """Test that parallel path is used when parallel_config is provided."""
        with et.dataset_cache() as memory:
            with mock.patch.object(
                et, "_build_metric_jobs_parallel"
            ) as mock_parallel:
                mock_parallel.return_value = []

                et.build_metric_jobs(
                    [mock_case_operator],
                    memory,
                    parallel_config={"n_jobs": 2},
                )

                mock_parallel.assert_called_once()


# =============================================================================
# Test: process_case_operator (integration)
# =============================================================================


class TestProcessCaseOperator:
    """Tests for process_case_operator function."""

    def test_returns_empty_list_when_datasets_empty(self, mock_case_operator):
        """Test that empty list is returned when datasets are empty."""
        with et.dataset_cache() as memory:
            with mock.patch.object(
                et, "validate_case_operator_metrics"
            ) as mock_validate:
                mock_validate.return_value = mock_case_operator
                with mock.patch.object(
                    et, "prepare_aligned_datasets"
                ) as mock_prepare:
                    mock_prepare.return_value = None

                    jobs = et.process_case_operator(mock_case_operator, memory)

                    assert jobs == []

    def test_creates_jobs_when_datasets_valid(
        self, mock_case_operator, prepared_datasets
    ):
        """Test that jobs are created when datasets are valid."""
        with et.dataset_cache() as memory:
            with mock.patch.object(
                et, "validate_case_operator_metrics"
            ) as mock_validate:
                mock_validate.return_value = mock_case_operator
                with mock.patch.object(
                    et, "prepare_aligned_datasets"
                ) as mock_prepare:
                    mock_prepare.return_value = prepared_datasets

                    jobs = et.process_case_operator(mock_case_operator, memory)

                    # Should have 2 jobs (one per variable)
                    assert len(jobs) == 2
                    assert all(isinstance(j, et.MetricJob) for j in jobs)


# =============================================================================
# Integration test with real data flow
# =============================================================================


class TestIntegrationWithSyntheticData:
    """Integration tests using synthetic data and minimal mocking."""

    def test_end_to_end_job_creation(
        self, mock_case_operator, synthetic_forecast_ds, synthetic_target_ds
    ):
        """Test complete flow from case operator to jobs."""
        with et.dataset_cache() as memory:
            # Mock validation and data loading
            with mock.patch.object(
                et, "validate_case_operator_metrics"
            ) as mock_validate:
                mock_validate.return_value = mock_case_operator
                with mock.patch.object(
                    et, "build_datasets"
                ) as mock_build:
                    mock_build.return_value = (
                        synthetic_forecast_ds,
                        synthetic_target_ds,
                    )

                    jobs = et.process_case_operator(mock_case_operator, memory)

                    assert len(jobs) == 2
                    # Verify job structure
                    for job in jobs:
                        assert job.forecast_var in ["temperature", "precipitation"]
                        assert job.target_var in ["temperature", "precipitation"]
                        assert isinstance(job.metric, metrics.MeanAbsoluteError)
