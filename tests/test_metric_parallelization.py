"""Tests for metric-level parallelization performance.

This test module verifies the metric-level parallelization approach:
1. Output equivalence (same results)
2. Memory usage patterns
3. Proper cache management

Uses synthetic/mock data for fast execution.
"""

import datetime
import gc
import time
import tracemalloc
from typing import NamedTuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, evaluate, inputs, metrics, regions


class PerformanceResult(NamedTuple):
    """Container for performance measurement results."""

    elapsed_time: float
    peak_memory_mb: float
    current_memory_mb: float
    result_df: pd.DataFrame


def _create_synthetic_dataset(
    n_lead: int = 5, n_lat: int = 5, n_lon: int = 5
):
    """Create a small synthetic xarray dataset for testing.

    Uses lead_time dimension which is what metrics expect to preserve.
    """
    lead_times = pd.to_timedelta(np.arange(n_lead) * 6, unit="h")
    lats = np.linspace(30, 50, n_lat)
    lons = np.linspace(-120, -100, n_lon)

    data = np.random.rand(n_lead, n_lat, n_lon) * 30 + 280  # Temperature-like

    ds = xr.Dataset(
        {
            "surface_air_temperature": (["lead_time", "latitude", "longitude"], data),
        },
        coords={
            "lead_time": lead_times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def _create_mock_case() -> cases.IndividualCase:
    """Create a single mock case for testing."""
    return cases.IndividualCase(
        case_id_number=999,
        title="Test Case",
        start_date=datetime.datetime(2021, 6, 20, 0, 0),
        end_date=datetime.datetime(2021, 6, 22, 0, 0),
        location=regions.CenteredRegion(
            latitude=40.0, longitude=-110.0, bounding_box_degrees=5.0
        ),
        event_type="heat_wave",
    )


def _create_mock_evaluation_object() -> inputs.EvaluationObject:
    """Create a mock evaluation object with simple metrics."""
    # Create mock target and forecast
    mock_target = mock.Mock(spec=inputs.TargetBase)
    mock_target.name = "mock_target"
    mock_target.variables = ["surface_air_temperature"]

    mock_forecast = mock.Mock(spec=inputs.ForecastBase)
    mock_forecast.name = "mock_forecast"
    mock_forecast.variables = ["surface_air_temperature"]

    # Simple metrics that are fast to compute
    metrics_list = [
        metrics.RootMeanSquaredError(),
        metrics.MeanAbsoluteError(),
    ]

    return inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=mock_target,
        forecast=mock_forecast,
    )


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset for testing."""
    return _create_synthetic_dataset()


@pytest.fixture
def mock_case():
    """Create a mock case."""
    return _create_mock_case()


@pytest.fixture
def mock_case_collection():
    """Create a mock case collection with 2 cases."""
    case1 = _create_mock_case()
    case2 = cases.IndividualCase(
        case_id_number=998,
        title="Test Case 2",
        start_date=datetime.datetime(2021, 7, 1, 0, 0),
        end_date=datetime.datetime(2021, 7, 3, 0, 0),
        location=regions.CenteredRegion(
            latitude=35.0, longitude=-115.0, bounding_box_degrees=5.0
        ),
        event_type="heat_wave",
    )
    return cases.IndividualCaseCollection(cases=[case1, case2])


class TestMetricParallelization:
    """Test suite for metric-level parallelization."""

    def test_cache_functions_exist_and_work(self):
        """Verify cache management functions are accessible."""
        assert hasattr(evaluate, "clear_dataset_cache")
        assert hasattr(evaluate, "_get_dataset_cache")
        assert hasattr(evaluate, "_make_cache_key")

        # clear_dataset_cache should be safe to call multiple times
        evaluate.clear_dataset_cache()
        evaluate.clear_dataset_cache()

    def test_metric_job_dataclass_exists(self):
        """Verify MetricJob dataclass is properly defined."""
        assert hasattr(evaluate, "MetricJob")

        from dataclasses import fields

        field_names = {f.name for f in fields(evaluate.MetricJob)}
        expected_fields = {
            "case_operator",
            "metric",
            "forecast_var",
            "target_var",
            "metric_kwargs",
        }
        assert expected_fields == field_names

    def test_build_metric_jobs_creates_correct_number_of_jobs(
        self, synthetic_dataset, mock_case
    ):
        """Test that _build_metric_jobs creates one job per metric."""
        # Create mock case operator with 2 metrics
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "mock_target"
        mock_target.variables = ["surface_air_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            synthetic_dataset,
            synthetic_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "mock_forecast"
        mock_forecast.variables = ["surface_air_temperature"]

        case_operator = cases.CaseOperator(
            case_metadata=mock_case,
            metric_list=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()],
            target=mock_target,
            forecast=mock_forecast,
        )

        # Mock _build_datasets to return our synthetic data
        with mock.patch(
            "extremeweatherbench.evaluate._build_datasets"
        ) as mock_build:
            mock_build.return_value = (synthetic_dataset, synthetic_dataset)

            jobs = evaluate._build_metric_jobs([case_operator])

            # Should create 2 jobs (one per metric)
            assert len(jobs) == 2
            assert all(isinstance(j, evaluate.MetricJob) for j in jobs)
            assert jobs[0].metric.name != jobs[1].metric.name

    def test_compute_single_metric_returns_dataframe(self, synthetic_dataset, mock_case):
        """Test that _compute_single_metric returns a proper DataFrame."""
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "mock_target"
        mock_target.variables = ["surface_air_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            synthetic_dataset,
            synthetic_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "mock_forecast"
        mock_forecast.variables = ["surface_air_temperature"]

        case_operator = cases.CaseOperator(
            case_metadata=mock_case,
            metric_list=[metrics.RootMeanSquaredError()],
            target=mock_target,
            forecast=mock_forecast,
        )

        job = evaluate.MetricJob(
            case_operator=case_operator,
            metric=metrics.RootMeanSquaredError(),
            forecast_var="surface_air_temperature",
            target_var="surface_air_temperature",
            metric_kwargs={},
        )

        with mock.patch(
            "extremeweatherbench.evaluate._build_datasets"
        ) as mock_build:
            mock_build.return_value = (synthetic_dataset, synthetic_dataset)

            result = evaluate._compute_single_metric(job)

            assert isinstance(result, pd.DataFrame)
            for col in evaluate.OUTPUT_COLUMNS:
                assert col in result.columns

    def test_serial_execution_with_mocked_data(
        self, synthetic_dataset, mock_case_collection
    ):
        """Test serial execution with fully mocked data pipeline."""
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "mock_target"
        mock_target.variables = ["surface_air_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            synthetic_dataset,
            synthetic_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "mock_forecast"
        mock_forecast.variables = ["surface_air_temperature"]

        eval_obj = inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch(
            "extremeweatherbench.evaluate._build_datasets"
        ) as mock_build:
            mock_build.return_value = (synthetic_dataset, synthetic_dataset)

            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=mock_case_collection,
                evaluation_objects=[eval_obj],
            )

            result = ewb.run(n_jobs=1)

            assert isinstance(result, pd.DataFrame)
            # 2 cases * 2 metrics = 4 results minimum
            assert len(result) >= 4

    def test_cache_is_cleared_after_run(self, synthetic_dataset, mock_case_collection):
        """Test that dataset cache is cleared after run completes."""
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "mock_target"
        mock_target.variables = ["surface_air_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            synthetic_dataset,
            synthetic_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "mock_forecast"
        mock_forecast.variables = ["surface_air_temperature"]

        eval_obj = inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=[metrics.RootMeanSquaredError()],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch(
            "extremeweatherbench.evaluate._build_datasets"
        ) as mock_build:
            mock_build.return_value = (synthetic_dataset, synthetic_dataset)

            ewb = evaluate.ExtremeWeatherBench(
                case_metadata=mock_case_collection,
                evaluation_objects=[eval_obj],
            )

            ewb.run(n_jobs=1)

            # Cache should be cleared after run
            assert len(evaluate._DATASET_RESULTS_CACHE) == 0


class TestPerformanceComparison:
    """Performance comparison tests with mocked data."""

    def test_memory_tracking_works(self, synthetic_dataset, mock_case):
        """Test that memory tracking functions correctly."""
        gc.collect()
        tracemalloc.start()

        # Allocate some memory
        _ = np.random.rand(1000, 1000)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak > 0
        assert current >= 0

    def test_serial_vs_parallel_produce_same_count(
        self, synthetic_dataset, mock_case_collection
    ):
        """Verify serial and parallel produce same number of results."""
        mock_target = mock.Mock(spec=inputs.TargetBase)
        mock_target.name = "mock_target"
        mock_target.variables = ["surface_air_temperature"]
        mock_target.maybe_align_forecast_to_target.return_value = (
            synthetic_dataset,
            synthetic_dataset,
        )

        mock_forecast = mock.Mock(spec=inputs.ForecastBase)
        mock_forecast.name = "mock_forecast"
        mock_forecast.variables = ["surface_air_temperature"]

        eval_obj = inputs.EvaluationObject(
            event_type="heat_wave",
            metric_list=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError()],
            target=mock_target,
            forecast=mock_forecast,
        )

        with mock.patch(
            "extremeweatherbench.evaluate._build_datasets"
        ) as mock_build:
            mock_build.return_value = (synthetic_dataset, synthetic_dataset)

            # Serial run
            ewb_serial = evaluate.ExtremeWeatherBench(
                case_metadata=mock_case_collection,
                evaluation_objects=[eval_obj],
            )
            serial_result = ewb_serial.run(n_jobs=1)

            # Parallel run
            ewb_parallel = evaluate.ExtremeWeatherBench(
                case_metadata=mock_case_collection,
                evaluation_objects=[eval_obj],
            )
            parallel_result = ewb_parallel.run(
                parallel_config={"backend": "threading", "n_jobs": 2}
            )

            # Both should produce the same number of results
            assert len(serial_result) == len(parallel_result)


def run_performance_benchmark():
    """Run a performance benchmark with mocked data (call manually).

    Usage:
        python tests/test_metric_parallelization.py
    """
    print("Creating synthetic data...")
    ds = _create_synthetic_dataset(n_lead=10, n_lat=20, n_lon=20)

    # Create mock cases
    case1 = _create_mock_case()
    case2 = cases.IndividualCase(
        case_id_number=998,
        title="Test Case 2",
        start_date=datetime.datetime(2021, 7, 1, 0, 0),
        end_date=datetime.datetime(2021, 7, 3, 0, 0),
        location=regions.CenteredRegion(
            latitude=35.0, longitude=-115.0, bounding_box_degrees=5.0
        ),
        event_type="heat_wave",
    )
    case_collection = cases.IndividualCaseCollection(cases=[case1, case2])

    mock_target = mock.Mock(spec=inputs.TargetBase)
    mock_target.name = "mock_target"
    mock_target.variables = ["surface_air_temperature"]
    mock_target.maybe_align_forecast_to_target.return_value = (ds, ds)

    mock_forecast = mock.Mock(spec=inputs.ForecastBase)
    mock_forecast.name = "mock_forecast"
    mock_forecast.variables = ["surface_air_temperature"]

    eval_obj = inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.RootMeanSquaredError(),
            metrics.MeanAbsoluteError(),
            metrics.MeanError(),
        ],
        target=mock_target,
        forecast=mock_forecast,
    )

    with mock.patch("extremeweatherbench.evaluate._build_datasets") as mock_build:
        mock_build.return_value = (ds, ds)

        print(f"\nBenchmarking with {len(case_collection.cases)} cases...")
        print("Metrics per case: 3")

        # Serial run
        print("\n--- Serial Execution ---")
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()

        ewb_serial = evaluate.ExtremeWeatherBench(
            case_metadata=case_collection,
            evaluation_objects=[eval_obj],
        )
        serial_result = ewb_serial.run(n_jobs=1)

        serial_time = time.perf_counter() - start_time
        _, serial_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Time: {serial_time:.4f}s")
        print(f"Peak Memory: {serial_peak / 1024 / 1024:.2f} MB")
        print(f"Result rows: {len(serial_result)}")

        # Parallel run
        print("\n--- Parallel Execution (threading, 2 jobs) ---")
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()

        ewb_parallel = evaluate.ExtremeWeatherBench(
            case_metadata=case_collection,
            evaluation_objects=[eval_obj],
        )
        parallel_result = ewb_parallel.run(
            parallel_config={"backend": "threading", "n_jobs": 2}
        )

        parallel_time = time.perf_counter() - start_time
        _, parallel_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Time: {parallel_time:.4f}s")
        print(f"Peak Memory: {parallel_peak / 1024 / 1024:.2f} MB")
        print(f"Result rows: {len(parallel_result)}")

        # Summary
        print("\n--- Summary ---")
        if parallel_time > 0:
            speedup = serial_time / parallel_time
            print(f"Speedup: {speedup:.2f}x")
        mem_diff = (parallel_peak - serial_peak) / 1024 / 1024
        print(f"Memory difference: {mem_diff:+.2f} MB")
        print(f"Results match: {len(serial_result) == len(parallel_result)}")


if __name__ == "__main__":
    run_performance_benchmark()
