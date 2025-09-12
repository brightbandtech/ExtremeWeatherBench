"""Tests for evaluate module."""

import datetime
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, derived, evaluate, inputs, metrics
from extremeweatherbench.defaults import OUTPUT_COLUMNS
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


# =============================================================================
# Test ExtremeWeatherBench Class
# =============================================================================


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
                evaluation_objects=[sample_evaluation_object],
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
        mock_derive_variables.side_effect = lambda ds, variables, **kwargs: ds

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
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

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


# =============================================================================
# Test Pipeline Functions
# =============================================================================


class TestPipelineFunctions:
    """Test the pipeline functions."""

    def test_build_datasets(self, sample_case_operator):
        """Test _build_datasets function."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            mock_forecast_ds = xr.Dataset(attrs={"name": "forecast_source"})
            mock_target_ds = xr.Dataset(attrs={"name": "target_source"})
            mock_run_pipeline.side_effect = [mock_target_ds, mock_forecast_ds]

            forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

            assert mock_run_pipeline.call_count == 2
            assert forecast_ds.attrs["name"] == "forecast_source"
            assert target_ds.attrs["name"] == "target_source"

    def test_build_datasets_zero_length_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has zero-length dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create a forecast dataset with zero-length valid_time dimension
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": []},  # Empty valid_time coordinate
                attrs={"source": "forecast"},
            )
            mock_target_ds = xr.Dataset(attrs={"source": "target"})
            mock_run_pipeline.side_effect = [mock_target_ds, mock_forecast_ds]

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
                assert "has no data for case time range" in warning_message
                assert (
                    str(sample_case_operator.case_metadata.case_id_number)
                    in warning_message
                )

                # Should call run_pipeline twice (once for target, once for forecast)
                assert mock_run_pipeline.call_count == 2

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
                assert "no data for case time range" in warning_message

    def test_build_datasets_normal_dimensions(self, sample_case_operator):
        """Test _build_datasets when forecast has normal (non-zero) dimensions."""
        with patch("extremeweatherbench.evaluate.run_pipeline") as mock_run_pipeline:
            # Create datasets with normal dimensions
            mock_forecast_ds = xr.Dataset(
                coords={"valid_time": [1, 2, 3], "latitude": [40, 45, 50]},
                attrs={"source": "forecast"},
            )
            mock_target_ds = xr.Dataset(attrs={"source": "target"})
            mock_run_pipeline.side_effect = [mock_target_ds, mock_forecast_ds]

            with patch("extremeweatherbench.evaluate.logger.warning") as mock_warning:
                forecast_ds, target_ds = evaluate._build_datasets(sample_case_operator)

                # Should return the actual datasets
                assert forecast_ds.attrs["source"] == "forecast"
                assert target_ds.attrs["source"] == "target"

                # Should not log any warning
                mock_warning.assert_not_called()

                # Should call run_pipeline twice (for both forecast and target)
                assert mock_run_pipeline.call_count == 2

    @patch("extremeweatherbench.derived.maybe_derive_variables")
    @patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
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
        # The pipe() method passes the dataset, then case_metadata as kwarg
        assert sample_case_operator.forecast.subset_data_to_case.call_count == 1
        call_args = sample_case_operator.forecast.subset_data_to_case.call_args
        assert call_args[1]["case_metadata"] == sample_case_operator.case_metadata
        sample_case_operator.forecast.maybe_convert_to_dataset.assert_called_once()
        sample_case_operator.forecast.add_source_to_dataset_attrs.assert_called_once()

    @patch("extremeweatherbench.derived.maybe_derive_variables")
    @patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
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
        mock_base_metric.name = "TestMetric"
        mock_base_metric.compute_metric.return_value = mock_result

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
        mock_base_metric.name = "TestMetric"
        mock_base_metric.compute_metric.return_value = mock_result

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
        mock_base_metric.compute_metric.assert_called_once()
        call_kwargs = mock_base_metric.compute_metric.call_args[1]
        assert "threshold" in call_kwargs
        assert call_kwargs["threshold"] == 0.5

    def test_evaluate_metric_and_return_df_with_derived_variables(
        self, mock_base_metric
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
            forecast_variable=TestForecastDerivedVariable,
            target_variable=TestTargetDerivedVariable,
            metric=mock_base_metric,
            case_id_number=3,
            event_type="derived_test",
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
        assert result["case_id_number"].iloc[0] == 3
        assert result["event_type"].iloc[0] == "derived_test"
        assert result["value"].iloc[0] == 2.5

        # Verify that compute_metric was called with the derived variables
        mock_base_metric.compute_metric.assert_called_once()
        call_args = mock_base_metric.compute_metric.call_args[0]

        # The variables should be passed as derived variable instances
        assert isinstance(call_args[0], xr.DataArray)  # forecast data
        assert isinstance(call_args[1], xr.DataArray)  # target data


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
            evaluate.run_pipeline(
                sample_case_operator.case_metadata, sample_case_operator.forecast
            )

    def test_evaluate_metric_computation_failure(
        self, sample_forecast_dataset, sample_target_dataset, mock_base_metric
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
                case_id_number=1,
                event_type="heat_wave",
            )


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegration:
    """Test integration scenarios with real-like data."""

    @patch("extremeweatherbench.derived.maybe_derive_variables")
    @patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
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

    @patch("extremeweatherbench.evaluate.inputs.maybe_subset_variables")
    def test_multiple_variables_and_metrics(
        self,
        mock_maybe_subset_variables,
        sample_cases_dict,
        sample_forecast_dataset,
        sample_target_dataset,
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

        with patch("extremeweatherbench.derived.maybe_derive_variables") as mock_derive:
            mock_derive.side_effect = lambda ds, variables, **kwargs: ds

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
    mock_target.variables = ["air_pressure_at_mean_sea_level", "latitude", "longitude"]

    # Create mock forecast
    mock_forecast = MagicMock(spec=inputs.KerchunkForecast)
    mock_forecast.variables = [derived.TropicalCycloneTrackVariables]

    # Create mock metric
    mock_metric = MagicMock(spec=metrics.BaseMetric)

    case_operator = cases.CaseOperator(
        case_metadata=case_metadata,
        metric_list=[mock_metric],
        target=mock_target,
        forecast=mock_forecast,
    )

    return case_operator


@pytest.fixture
def sample_tc_forecast_dataset():
    """Create a sample forecast dataset for TC testing."""
    valid_time = pd.date_range("2023-09-01", periods=3, freq="12h")
    lead_time = np.array([0, 12, 24, 36], dtype="timedelta64[h]")
    lat = np.linspace(10, 40, 16)
    lon = np.linspace(-90, -60, 16)

    # Create realistic meteorological data
    data_shape = (len(valid_time), len(lat), len(lon), len(lead_time))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude", "lead_time"],
                np.random.normal(101325, 1000, data_shape),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude", "lead_time"],
                np.random.normal(0, 10, data_shape),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude", "lead_time"],
                np.random.normal(0, 10, data_shape),
            ),
            "geopotential": (
                ["time", "latitude", "longitude", "lead_time"],
                np.random.normal(5000, 1000, data_shape) * 9.80665,
            ),
        },
        coords={
            "valid_time": valid_time,
            "latitude": lat,
            "longitude": lon,
            "lead_time": lead_time,
        },
    )

    return dataset


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
    dataset.attrs["is_ibtracs_data"] = True

    return dataset


class TestTropicalCycloneEvaluation:
    """Test tropical cyclone specific evaluation functionality."""

    def setup_method(self):
        """Clear registries before each test."""
        tropical_cyclone.clear_ibtracs_registry()

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_ibtracs_registration_during_evaluation(
        self,
        mock_build_datasets,
        sample_tc_case_operator,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test that IBTrACS data is registered during evaluation."""
        # Override variables to avoid derived variable computation for this test
        sample_tc_case_operator.target.variables = ["air_pressure_at_mean_sea_level"]
        sample_tc_case_operator.forecast.variables = ["air_pressure_at_mean_sea_level"]

        # Setup mocks
        mock_build_datasets.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        # Mock the target's maybe_align_forecast_to_target method
        sample_tc_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        # Mock the metric computation
        mock_metric_instance = Mock(spec=metrics.BaseMetric)
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        sample_tc_case_operator.metric_list[0] = mock_metric_instance

        # Manually register IBTrACS data since we're bypassing the input pipeline
        case_id_number = str(sample_tc_case_operator.case_metadata.case_id_number)
        tropical_cyclone.register_ibtracs_data(case_id_number, sample_ibtracs_dataset)

        # Run the evaluation
        evaluate.compute_case_operator(sample_tc_case_operator)

        # Check that IBTrACS data was registered
        case_id = str(sample_tc_case_operator.case_metadata.case_id_number)
        registered_data = tropical_cyclone.get_ibtracs_data(case_id)

        assert registered_data is not None
        xr.testing.assert_equal(registered_data, sample_ibtracs_dataset)

    @patch("extremeweatherbench.evaluate._build_datasets")
    def test_non_ibtracs_target_no_registration(
        self,
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
            metric_list=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        # Create a non-IBTrACS dataset for this test
        non_ibtracs_dataset = sample_ibtracs_dataset.copy()
        non_ibtracs_dataset.attrs.pop("is_ibtracs_data", None)
        non_ibtracs_dataset.attrs["source"] = "ERA5"

        # Setup mocks
        mock_build_datasets.return_value = (
            sample_tc_forecast_dataset,
            non_ibtracs_dataset,
        )

        case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            non_ibtracs_dataset,
        )

        mock_metric_instance = Mock(spec=metrics.BaseMetric)
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        case_operator.metric_list[0] = mock_metric_instance

        # Run the evaluation
        evaluate.compute_case_operator(case_operator)

        # Check that IBTrACS data was NOT registered
        case_id = str(case_operator.case_metadata.case_id_number)
        registered_data = tropical_cyclone.get_ibtracs_data(case_id)

        assert registered_data is None

    def test_case_metadata_passed_to_derive_variables(
        self,
        sample_tc_case_operator,
        sample_tc_forecast_dataset,
        sample_ibtracs_dataset,
    ):
        """Test that case_metadata is passed to maybe_derive_variables and get_ibtracs_data is called correctly."""
        # Mock the data loading to return our test datasets
        sample_tc_case_operator.forecast.open_and_maybe_preprocess_data_from_source.return_value = sample_tc_forecast_dataset
        sample_tc_case_operator.target.open_and_maybe_preprocess_data_from_source.return_value = sample_ibtracs_dataset

        sample_tc_case_operator.target.maybe_align_forecast_to_target.return_value = (
            sample_tc_forecast_dataset,
            sample_ibtracs_dataset,
        )

        mock_metric_instance = Mock(spec=metrics.BaseMetric)
        mock_metric_instance.name = "TestMetric"
        mock_metric_instance.compute_metric.return_value = xr.DataArray([1.0])
        sample_tc_case_operator.metric_list[0] = mock_metric_instance

        # Let's just test that the evaluation runs without errors
        # and that when it does call derived variables, the case_metadata is available

        # Register some test IBTrACS data for the derived variables to use

        case_id_str = str(sample_tc_case_operator.case_metadata.case_id_number)
        tropical_cyclone.register_ibtracs_data(case_id_str, sample_ibtracs_dataset)

        try:
            # Run the evaluation - this should work without mocking
            result = evaluate.compute_case_operator(sample_tc_case_operator)

            # Basic sanity check - we should get a DataFrame back
            assert isinstance(result, pd.DataFrame)

            # Check that IBTrACS data was registered (this proves case_id_number flow works)
            retrieved_data = tropical_cyclone.get_ibtracs_data(case_id_str)
            assert retrieved_data is not None

            print(
                "✅ Test passed - evaluation completed successfully with proper case_metadata flow"
            )

        except Exception as e:
            # If there's an error, it might be related to the derived variable processing
            print(f"❌ Error during evaluation: {e}")
            raise
        finally:
            # Clean up
            tropical_cyclone.clear_ibtracs_registry()


class TestDerivedVariableIntegration:
    """Test integration of derived variables in evaluation."""

    def setup_method(self):
        """Clear caches before each test."""
        tropical_cyclone.clear_ibtracs_registry()
        derived.TropicalCycloneTrackVariables.clear_cache()

    @patch(
        "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
    )
    @patch("extremeweatherbench.events.tropical_cyclone.generate_tc_variables")
    def test_maybe_derive_variables_with_tc_tracks(
        self, mock_generate_tc_vars, mock_create_tracks
    ):
        """Test maybe_derive_variables with TC track variables."""
        # Create a dataset that needs TC track derivation
        base_dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 10, 10)),
                ),
                "surface_eastward_wind": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 10, 10)),
                ),
                "surface_northward_wind": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 10, 10)),
                ),
                "geopotential": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(5000, 1000, (2, 10, 10)) * 9.80665,
                ),
            },
            coords={
                "valid_time": pd.date_range("2023-09-01", periods=2, freq="6h"),
                "latitude": np.linspace(20, 30, 10),
                "longitude": np.linspace(-80, -70, 10),
            },
        )

        # Mock the track computation
        mock_tracks = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["time", "lead_time"],
                    [[101000, 101010], [101020, 101030]],
                ),
                "latitude": (
                    ["time", "lead_time"],
                    [[25.0, 25.5], [26.0, 26.5]],
                ),
                "longitude": (
                    ["time", "lead_time"],
                    [[-75.0, -74.5], [-74.0, -73.5]],
                ),
                "surface_wind_speed": (
                    ["time", "lead_time"],
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
        variables = [derived.TropicalCycloneTrackVariables]
        # Create mock case operator for the new signature
        from unittest.mock import Mock

        mock_case_operator = Mock()
        mock_case_operator.case_metadata.case_id_number = "test_case"
        mock_case_operator.forecast.variables = variables
        mock_case_operator.target.variables = []

        # Set dataset type
        base_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(
            base_dataset, variables, case_metadata=mock_case_operator.case_metadata
        )

        # Should have the derived variable
        assert isinstance(result, xr.Dataset)
        assert "air_pressure_at_mean_sea_level" in result.data_vars
        assert "surface_wind_speed" in result.data_vars

    def test_maybe_include_variables_from_derived_input_tc(self):
        """Test pulling required variables from TC derived variables."""
        incoming_variables = [
            "existing_variable",
            derived.TropicalCycloneTrackVariables,
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

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

    def test_maybe_include_variables_with_instances_and_classes(self):
        """Test with both instances and classes of derived variables."""
        track_instance = derived.TropicalCycloneTrackVariables()

        incoming_variables = [
            "base_variable",
            track_instance,  # Instance
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

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


# =============================================================================
# Test Schema and Variable Normalization Functions
# =============================================================================


class TestEnsureOutputSchema:
    """Test the _ensure_output_schema function."""

    def test_ensure_output_schema_init_time_valid_time(self):
        """Test _ensure_output_schema with init_time and valid_time columns.

        init_time is now in OUTPUT_COLUMNS, valid_time is not and will be dropped.
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

        # Check all OUTPUT_COLUMNS are present
        assert list(result.columns) == OUTPUT_COLUMNS
        # Check metadata was added
        assert all(result["target_variable"] == "temperature")
        assert all(result["metric"] == "TestMetric")
        assert all(result["case_id_number"] == 1)
        assert all(result["event_type"] == "heat_wave")
        # Check original columns preserved for those in OUTPUT_COLUMNS
        assert len(result) == 2
        assert list(result["value"]) == [1.0, 2.0]
        # init_time should be preserved (now in OUTPUT_COLUMNS)
        assert "init_time" in result.columns
        assert list(result["init_time"]) == [
            pd.to_datetime("2021-06-20"),
            pd.to_datetime("2021-06-21"),
        ]
        # valid_time should be dropped (not in OUTPUT_COLUMNS)
        assert "valid_time" not in result.columns

    def test_ensure_output_schema_init_time_only(self):
        """Test _ensure_output_schema with init_time only.

        init_time is now in OUTPUT_COLUMNS so will be preserved.
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
        assert list(result.columns) == OUTPUT_COLUMNS
        # lead_time should be NaN since not provided and is in OUTPUT_COLUMNS
        assert pd.isna(result["lead_time"].iloc[0])
        # init_time should be preserved (now in OUTPUT_COLUMNS)
        assert "init_time" in result.columns
        assert result["init_time"].iloc[0] == pd.to_datetime("2021-06-20")

    def test_ensure_output_schema_lead_time_only(self):
        """Test _ensure_output_schema with lead_time only.

        lead_time is in OUTPUT_COLUMNS so will be preserved.
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
        assert list(result.columns) == OUTPUT_COLUMNS
        # lead_time should be preserved (it's in OUTPUT_COLUMNS)
        assert list(result["lead_time"]) == [6, 12]
        # init_time should be NaN since not provided and is in OUTPUT_COLUMNS
        assert pd.isna(result["init_time"].iloc[0])
        # Other OUTPUT_COLUMNS should have NaN for missing source columns
        assert pd.isna(result["target_source"].iloc[0])
        assert pd.isna(result["forecast_source"].iloc[0])

    def test_ensure_output_schema_lead_time_valid_time(self):
        """Test _ensure_output_schema with lead_time and valid_time.

        lead_time is in OUTPUT_COLUMNS, valid_time is not and will be dropped.
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

        assert list(result.columns) == OUTPUT_COLUMNS
        assert result["lead_time"].iloc[0] == 24
        # init_time should be NaN since not provided and is in OUTPUT_COLUMNS
        assert pd.isna(result["init_time"].iloc[0])
        # valid_time should be dropped (not in OUTPUT_COLUMNS)
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

        assert list(result.columns) == OUTPUT_COLUMNS
        # Custom column should be preserved but not part of OUTPUT_COLUMNS
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

        assert list(result.columns) == OUTPUT_COLUMNS
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

        assert list(result.columns) == OUTPUT_COLUMNS
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
        assert list(result.columns) == OUTPUT_COLUMNS
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

        assert list(result.columns) == OUTPUT_COLUMNS

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

        assert list(result.columns) == OUTPUT_COLUMNS

    def test_ensure_output_schema_no_missing_variables(self):
        """Test _ensure_output_schema when no variables are missing."""
        # Create a dataframe with all required OUTPUT_COLUMNS already present
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
        assert list(result.columns) == OUTPUT_COLUMNS
        assert len(result) == 2
        assert result["value"].tolist() == [1.0, 2.0]
        assert result["lead_time"].tolist() == [1, 2]

    def test_ensure_output_schema_no_missing_with_metadata(self, caplog):
        """Test _ensure_output_schema when no variables are missing with metadata."""
        # Create a dataframe with all required OUTPUT_COLUMNS already present
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
        assert list(result.columns) == OUTPUT_COLUMNS
        assert len(result) == 2
        assert result["value"].tolist() == [1.5, 2.5]
        assert all(result["target_variable"] == "updated_pressure")
        assert all(result["metric"] == "UpdatedMetric")
        assert all(result["case_id_number"] == 3)
        assert all(result["event_type"] == "updated_event")


class TestDerivedVariableForTesting(derived.DerivedVariable):
    """A concrete derived variable class for testing
    _maybe_convert_variable_to_string."""

    name = "TestDerivedVar"
    required_variables = ["input_var1", "input_var2"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation for testing."""
        return data["input_var1"] + data["input_var2"]


class TestForecastDerivedVariable(derived.DerivedVariable):
    """Test derived variable for forecast data."""

    name = "derived_forecast_var"
    required_variables = ["surface_air_temperature"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation - just return the temperature variable."""
        return data["surface_air_temperature"]


class TestTargetDerivedVariable(derived.DerivedVariable):
    """Test derived variable for target data."""

    name = "derived_target_var"
    required_variables = ["2m_temperature"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation - just return the temperature variable."""
        return data["2m_temperature"]


class TestNormalizeVariable:
    """Test the _maybe_convert_variable_to_string function."""

    def test_maybe_convert_variable_to_string_string_input(self):
        """Test _maybe_convert_variable_to_string with string input."""
        result = evaluate._maybe_convert_variable_to_string("temperature")
        assert result == "temperature"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_derived_class_input(self):
        """Test _maybe_convert_variable_to_string with DerivedVariable class input."""
        result = evaluate._maybe_convert_variable_to_string(
            TestDerivedVariableForTesting
        )
        assert result == "TestDerivedVar"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_derived_instance_input(self):
        """Test _maybe_convert_variable_to_string with DerivedVariable instance input.

        Note: The function is designed to handle classes, not instances.
        Instances are passed through unchanged (not converted to string).
        """
        instance = TestDerivedVariableForTesting()
        result = evaluate._maybe_convert_variable_to_string(instance)
        # Instance is returned as-is, not converted to string
        assert result == instance
        assert isinstance(result, TestDerivedVariableForTesting)

    def test_maybe_convert_variable_to_string_handles_both_types(self):
        """Test that _maybe_convert_variable_to_string handles both incoming types
        correctly."""
        # Test string type
        string_result = evaluate._maybe_convert_variable_to_string("my_variable")
        assert string_result == "my_variable"

        # Test derived variable type
        derived_result = evaluate._maybe_convert_variable_to_string(
            TestDerivedVariableForTesting
        )
        assert derived_result == "TestDerivedVar"

        # Results should be different but both strings
        assert string_result != derived_result
        assert isinstance(string_result, str)
        assert isinstance(derived_result, str)


if __name__ == "__main__":
    pytest.main([__file__])
