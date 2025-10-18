"""Integration tests.

This test suite validates that the evaluation system correctly pairs forecast and
target variables using the zip() pairing logic, covering various scenarios:
1. Single variable pairing
2. Multiple variable pairwise pairing
3. Duplicate variable handling
4. Mismatched variable count handling
"""

import datetime
from typing import List
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench import cases, evaluate, inputs, metrics, regions


class MockMetric(metrics.BaseMetric):
    """A simple mock metric for testing."""

    name = "MockMetric"

    @classmethod
    def _compute_metric(cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs):
        """Return a simple mean absolute difference."""
        diff = abs(forecast - target)
        # Reduce to a scalar but return as DataArray for EWB compatibility
        result_value = float(diff.mean().values)
        # Return as DataArray with lead_time dimension for EWB compatibility
        result = xr.DataArray(
            [result_value], dims=["lead_time"], coords={"lead_time": [0]}
        )
        return result


@pytest.fixture
def mock_metric():
    """Create a mock metric instance."""
    return MockMetric


@pytest.fixture
def sample_case():
    """Create a sample IndividualCase for testing."""
    return cases.IndividualCase(
        case_id_number=1,
        title="Variable Pairing Test",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 6, 25),
        location=regions.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=5.0
        ),
        event_type="test_event",
    )


@pytest.fixture
def base_forecast_dataset():
    """Create a base forecast dataset with multiple variables."""
    time = pd.date_range("2021-06-20", periods=5)
    latitude = np.array([44.0, 45.0, 46.0])
    longitude = np.array([-121.0, -120.0, -119.0])

    # Create distinct data for each variable to verify correct pairing
    temp_data = np.random.random((len(time), len(latitude), len(longitude))) + 273.15
    pressure_data = (
        np.random.random((len(time), len(latitude), len(longitude))) + 1013.25
    )
    humidity_data = np.random.random((len(time), len(latitude), len(longitude))) * 100

    return xr.Dataset(
        {
            "var_a": (["valid_time", "latitude", "longitude"], temp_data),
            "var_b": (["valid_time", "latitude", "longitude"], pressure_data),
            "var_c": (["valid_time", "latitude", "longitude"], humidity_data),
        },
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={"source": "test_forecast"},
    )


@pytest.fixture
def base_target_dataset():
    """Create a base target dataset with multiple variables."""
    time = pd.date_range("2021-06-20", periods=5)
    latitude = np.array([44.0, 45.0, 46.0])
    longitude = np.array([-121.0, -120.0, -119.0])

    # Create distinct data for each variable
    # Offset for difference
    temp_data = (
        np.random.random((len(time), len(latitude), len(longitude))) + 273.15 + 1.0
    )
    pressure_data = (
        np.random.random((len(time), len(latitude), len(longitude))) + 1013.25 + 2.0
    )
    humidity_data = (
        np.random.random((len(time), len(latitude), len(longitude))) * 100 + 3.0
    )
    wind_data = np.random.random((len(time), len(latitude), len(longitude))) * 10 + 4.0

    return xr.Dataset(
        {
            "var_x": (["valid_time", "latitude", "longitude"], temp_data),
            "var_y": (["valid_time", "latitude", "longitude"], pressure_data),
            "var_z": (["valid_time", "latitude", "longitude"], humidity_data),
            "var_w": (["valid_time", "latitude", "longitude"], wind_data),
        },
        coords={
            "valid_time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={"source": "test_target"},
    )


def create_mock_input(variables: List[str], dataset: xr.Dataset, input_type: str):
    """Helper to create mock forecast or target inputs."""
    mock_input = mock.Mock()
    mock_input.name = f"Mock{input_type.title()}"
    mock_input.variables = variables

    # Mock all pipeline methods to return the dataset
    mock_input.open_and_maybe_preprocess_data_from_source.return_value = dataset
    mock_input.maybe_map_variable_names.return_value = dataset
    mock_input.maybe_subset_variables.return_value = dataset
    mock_input.subset_data_to_case.return_value = dataset
    mock_input.maybe_convert_to_dataset.return_value = dataset
    mock_input.add_source_to_dataset_attrs.return_value = dataset

    if input_type == "target":
        # This should return (aligned_forecast, aligned_target)
        # We'll update this in create_case_operator
        mock_input.maybe_align_forecast_to_target = mock.Mock()

    return mock_input


def create_case_operator(
    sample_case,
    forecast_vars: List[str],
    target_vars: List[str],
    forecast_dataset: xr.Dataset,
    target_dataset: xr.Dataset,
    mock_metric_list,
):
    """Helper to create a CaseOperator with specified variables."""
    # Create datasets that contain the requested variables
    forecast_data = {}
    for i, var in enumerate(forecast_vars):
        # Use existing variables from forecast dataset or create new ones
        if f"var_{chr(ord('a') + i)}" in forecast_dataset.data_vars:
            forecast_data[var] = forecast_dataset[f"var_{chr(ord('a') + i)}"]
        else:
            # Fallback to first available variable
            available_vars = list(forecast_dataset.data_vars.keys())
            if available_vars:
                forecast_data[var] = forecast_dataset[available_vars[0]]

    target_data = {}
    for i, var in enumerate(target_vars):
        # Use existing variables from target dataset or create new ones
        if f"var_{chr(ord('x') + i)}" in target_dataset.data_vars:
            target_data[var] = target_dataset[f"var_{chr(ord('x') + i)}"]
        else:
            # Fallback to first available variable
            available_vars = list(target_dataset.data_vars.keys())
            if available_vars:
                target_data[var] = target_dataset[available_vars[0]]

    # Create new datasets with the correct variable names
    custom_forecast_dataset = xr.Dataset(
        forecast_data, coords=forecast_dataset.coords, attrs=forecast_dataset.attrs
    )

    custom_target_dataset = xr.Dataset(
        target_data, coords=target_dataset.coords, attrs=target_dataset.attrs
    )

    mock_forecast = create_mock_input(
        forecast_vars, custom_forecast_dataset, "forecast"
    )
    mock_target = create_mock_input(target_vars, custom_target_dataset, "target")

    # Set up the alignment function to return the correct datasets
    mock_target.maybe_align_forecast_to_target.return_value = (
        custom_forecast_dataset,
        custom_target_dataset,
    )

    return cases.CaseOperator(
        case_metadata=sample_case,
        metric_list=[mock_metric_list],
        target=mock_target,
        forecast=mock_forecast,
    )


@pytest.mark.integration
class TestInputsIntegration:
    """Integration tests for inputs module."""

    # zarr throws a consolidated metadata warning that
    # is inconsequential (as of now)
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_era5_full_workflow_with_zarr(self, temp_zarr_file):
        """Test complete ERA5 workflow with zarr file."""
        era5 = inputs.ERA5(
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test opening data
        data = era5._open_data_from_source()
        assert isinstance(data, xr.Dataset)
        assert "2m_temperature" in data.data_vars

        # Test conversion
        dataset = era5.maybe_convert_to_dataset(data)
        assert isinstance(dataset, xr.Dataset)

    def test_ghcn_full_workflow_with_parquet(self, temp_parquet_file):
        """Test complete GHCN workflow with parquet file."""
        ghcn = inputs.GHCN(
            source=temp_parquet_file,
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test opening data
        data = ghcn._open_data_from_source()
        assert isinstance(data, pl.LazyFrame)

        # Test conversion
        dataset = ghcn._custom_convert_to_dataset(data)
        assert isinstance(dataset, xr.Dataset)

    def test_era5_alignment_comprehensive(
        self, sample_era5_dataset, sample_forecast_with_valid_time
    ):
        """Test comprehensive ERA5 alignment scenarios."""
        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test with matching spatial grids but different time ranges
        target_subset = sample_era5_dataset.sel(time=slice("2021-06-20", "2021-06-21"))
        forecast_subset = sample_forecast_with_valid_time.sel(
            valid_time=slice("2021-06-20 12:00", "2021-06-21 12:00")
        )

        aligned_forecast, aligned_target = era5.maybe_align_forecast_to_target(
            forecast_subset, target_subset
        )

        # Should find overlapping times
        # Note: dimensions keep their original names after alignment
        assert len(aligned_target.time) > 0  # Target uses 'time'
        assert len(aligned_forecast.valid_time) > 0  # Forecast uses 'valid_time'

        # Should have overlapping time periods - but lengths may differ due to
        # different time ranges. This is expected when target and forecast
        # have different time coverage


class TestVariablePairingIntegration:
    """Integration tests for variable pairing behavior."""

    def test_single_variable_pairing(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test 1: Single forecast variable vs single target variable."""
        # Create case operator with single variables
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a"],
            target_vars=["var_x"],
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1, "Should have exactly one evaluation result"

        # Check that the correct variables were paired
        assert result["target_variable"].iloc[0] == "var_x"
        # Note: forecast variable name is not stored in the results by design
        # The important thing is that the metric was computed correctly

        # Verify metric was computed
        assert result["metric"].iloc[0] == "MockMetric"
        assert result["case_id_number"].iloc[0] == 1
        assert result["event_type"].iloc[0] == "test_event"

    def test_two_variable_pairwise_pairing(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test 2: Two forecast variables vs two target variables (pairwise)."""
        # Create case operator with two variables each
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a", "var_b"],
            target_vars=["var_x", "var_y"],
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2, "Should have exactly two evaluation results"

        # Sort by target variable for consistent testing
        result = result.sort_values("target_variable").reset_index(drop=True)

        # Check pairings: var_a <-> var_x, var_b <-> var_y
        assert result["target_variable"].iloc[0] == "var_x"
        assert result["target_variable"].iloc[1] == "var_y"

        # Both should use the same metric and case info
        assert all(result["metric"] == "MockMetric")
        assert all(result["case_id_number"] == 1)
        assert all(result["event_type"] == "test_event")

    def test_duplicate_forecast_variables(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test 3: Duplicate forecast variables [a,a] vs [b,c]."""
        # Create case operator with duplicate forecast variable
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a", "var_a"],  # Duplicate forecast variable
            target_vars=["var_x", "var_y"],  # Different target variables
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2, "Should have exactly two evaluation results"

        # Sort by target variable for consistent testing
        result = result.sort_values("target_variable").reset_index(drop=True)

        # Check pairings: var_a <-> var_x, var_a <-> var_y
        assert result["target_variable"].iloc[0] == "var_x"
        assert result["target_variable"].iloc[1] == "var_y"

        # Both evaluations should use the same forecast variable (var_a)
        # but different target variables
        assert all(result["metric"] == "MockMetric")
        assert all(result["case_id_number"] == 1)

    def test_mismatched_variable_counts(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test 4: Mismatched counts [a] vs [b,c] - should only compare a,b."""
        # Create case operator with mismatched variable counts
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a"],  # Single forecast variable
            target_vars=["var_x", "var_y"],  # Two target variables
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert (
            len(result) == 1
        ), "Should have exactly one evaluation result (only first pairing)"

        # Check that only the first pairing was created: var_a <-> var_x
        assert result["target_variable"].iloc[0] == "var_x"
        # var_y should not appear since there's no corresponding forecast variable

        assert result["metric"].iloc[0] == "MockMetric"
        assert result["case_id_number"].iloc[0] == 1

    def test_reverse_mismatched_variable_counts(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test 4b: Reverse mismatch [a,b] vs [c] - should only compare a,c."""
        # Create case operator with reverse mismatched variable counts
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a", "var_b"],  # Two forecast variables
            target_vars=["var_x"],  # Single target variable
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert (
            len(result) == 1
        ), "Should have exactly one evaluation result (only first pairing)"

        # Check that only the first pairing was created: var_a <-> var_x
        assert result["target_variable"].iloc[0] == "var_x"
        # var_b should not be evaluated since there's no corresponding target variable

        assert result["metric"].iloc[0] == "MockMetric"
        assert result["case_id_number"].iloc[0] == 1

    def test_complex_variable_pairing(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test complex scenario with three variables to verify exact pairing logic."""
        # Create case operator with three variables each
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a", "var_b", "var_c"],
            target_vars=["var_x", "var_y", "var_z"],
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3, "Should have exactly three evaluation results"

        # Sort by target variable for consistent testing
        result = result.sort_values("target_variable").reset_index(drop=True)

        # Check all three pairings: var_a <-> var_x, var_b <-> var_y, var_c <-> var_z
        expected_targets = ["var_x", "var_y", "var_z"]
        actual_targets = result["target_variable"].tolist()
        assert actual_targets == expected_targets

        # All should use the same metric and case info
        assert all(result["metric"] == "MockMetric")
        assert all(result["case_id_number"] == 1)
        assert all(result["event_type"] == "test_event")

    def test_empty_variable_lists(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test edge case with empty variable lists."""
        # Create case operator with empty variable lists
        case_op = create_case_operator(
            sample_case,
            forecast_vars=[],
            target_vars=[],
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        msg = "Should have no evaluation results for empty variable lists"
        assert len(result) == 0, msg

        # Should still have the expected columns structure
        expected_columns = [
            "value",
            "target_variable",
            "metric",
            "case_id_number",
            "event_type",
        ]
        assert all(col in result.columns for col in expected_columns)

    def test_metric_values_different_for_different_pairings(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Verify that different variable pairings produce different metric values."""
        # Create case operator with two different variable pairings
        case_op = create_case_operator(
            sample_case,
            forecast_vars=["var_a", "var_b"],
            target_vars=["var_x", "var_y"],
            forecast_dataset=base_forecast_dataset,
            target_dataset=base_target_dataset,
            mock_metric_list=mock_metric,
        )

        # Execute evaluation
        result = evaluate.compute_case_operator(case_op)

        # Verify that we get different metric values for different pairings
        # (since we created datasets with different base values for each variable)
        assert len(result) == 2
        values = result["value"].tolist()

        # The values should be different because we're comparing different variables
        # with different underlying data patterns
        msg = "Different variable pairings should produce different metric values"
        assert len(set(values)) == 2, msg


class TestExtremeWeatherBenchVariablePairing:
    """Test the full ExtremeWeatherBench workflow with variable pairing scenarios."""

    def test_full_workflow_single_variable(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test full ExtremeWeatherBench workflow with single variable pairing."""
        # Create datasets with the right variable names
        forecast_data = {"var_a": base_forecast_dataset["var_a"]}
        target_data = {"var_x": base_target_dataset["var_x"]}

        custom_forecast_dataset = xr.Dataset(
            forecast_data,
            coords=base_forecast_dataset.coords,
            attrs=base_forecast_dataset.attrs,
        )
        custom_target_dataset = xr.Dataset(
            target_data,
            coords=base_target_dataset.coords,
            attrs=base_target_dataset.attrs,
        )

        # Create evaluation object
        mock_forecast = create_mock_input(
            ["var_a"], custom_forecast_dataset, "forecast"
        )
        mock_target = create_mock_input(["var_x"], custom_target_dataset, "target")

        # Set up the alignment function properly
        mock_target.maybe_align_forecast_to_target.return_value = (
            custom_forecast_dataset,
            custom_target_dataset,
        )

        evaluation_obj = inputs.EvaluationObject(
            event_type="test_event",
            metric_list=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        # Create cases dict
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Test Case",
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
                    "event_type": "test_event",
                }
            ]
        }

        # Create and run ExtremeWeatherBench
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=cases_dict,
            evaluation_objects=[evaluation_obj],
        )

        result = ewb.run()

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["target_variable"].iloc[0] == "var_x"
        assert result["metric"].iloc[0] == "MockMetric"

    def test_full_workflow_multiple_variables(
        self, sample_case, base_forecast_dataset, base_target_dataset, mock_metric
    ):
        """Test full ExtremeWeatherBench workflow with multiple variable pairing."""
        # Create datasets with the right variable names
        forecast_data = {
            "var_a": base_forecast_dataset["var_a"],
            "var_b": base_forecast_dataset["var_b"],
        }
        target_data = {
            "var_x": base_target_dataset["var_x"],
            "var_y": base_target_dataset["var_y"],
        }

        custom_forecast_dataset = xr.Dataset(
            forecast_data,
            coords=base_forecast_dataset.coords,
            attrs=base_forecast_dataset.attrs,
        )
        custom_target_dataset = xr.Dataset(
            target_data,
            coords=base_target_dataset.coords,
            attrs=base_target_dataset.attrs,
        )

        # Create evaluation object with multiple variables
        mock_forecast = create_mock_input(
            ["var_a", "var_b"], custom_forecast_dataset, "forecast"
        )
        mock_target = create_mock_input(
            ["var_x", "var_y"], custom_target_dataset, "target"
        )

        # Set up the alignment function properly
        mock_target.maybe_align_forecast_to_target.return_value = (
            custom_forecast_dataset,
            custom_target_dataset,
        )

        evaluation_obj = inputs.EvaluationObject(
            event_type="test_event",
            metric_list=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        # Create cases dict
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Test Case",
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
                    "event_type": "test_event",
                }
            ]
        }

        # Create and run ExtremeWeatherBench
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=cases_dict,
            evaluation_objects=[evaluation_obj],
        )

        result = ewb.run()

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two variable pairings

        # Check that both pairings are present
        target_vars = set(result["target_variable"])
        assert target_vars == {"var_x", "var_y"}
        assert all(result["metric"] == "MockMetric")


if __name__ == "__main__":
    pytest.main([__file__])
