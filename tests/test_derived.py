"""Comprehensive unit tests for the extremeweatherbench.derived module.

This test suite covers:
- Abstract base class DerivedVariable
- All concrete derived variable implementations
- Utility functions for variable derivation
- Edge cases and error conditions
- Mock implementations for testing
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import derived
from extremeweatherbench.events import tropical_cyclone

# flake8: noqa: E501


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    time = pd.date_range("2021-06-20", freq="6h", periods=8)
    latitudes = np.linspace(30, 50, 11)
    longitudes = np.linspace(250, 270, 21)
    level = [1000, 850, 700, 500, 300, 200]

    # Create realistic sample data
    np.random.seed(42)
    base_data = np.random.normal(
        20, 5, size=(len(time), len(latitudes), len(longitudes))
    )
    level_data = np.random.normal(
        0, 10, size=(len(time), len(level), len(latitudes), len(longitudes))
    )

    dataset = xr.Dataset(
        {
            # Basic surface variables
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    101325, 1000, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    5, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    2, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_wind_speed": (
                ["time", "latitude", "longitude"],
                np.random.uniform(
                    0, 15, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            # 3D atmospheric variables
            "eastward_wind": (
                ["time", "level", "latitude", "longitude"],
                level_data + np.random.normal(10, 5, size=level_data.shape),
            ),
            "northward_wind": (
                ["time", "level", "latitude", "longitude"],
                level_data + np.random.normal(3, 5, size=level_data.shape),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                np.random.exponential(0.008, size=level_data.shape),
            ),
            "geopotential": (
                ["time", "level", "latitude", "longitude"],
                level_data * 100 + np.random.normal(50000, 5000, size=level_data.shape),
            ),
            # Test variables
            "test_variable_1": (["time", "latitude", "longitude"], base_data),
            "test_variable_2": (["time", "latitude", "longitude"], base_data + 5),
            "single_variable": (["time", "latitude", "longitude"], base_data * 2),
        },
        coords={
            "time": time,
            "latitude": latitudes,
            "longitude": longitudes,
            "level": level,
        },
    )

    return dataset


class TestValidDerivedVariable(derived.DerivedVariable):
    """A valid test implementation of DerivedVariable for testing purposes."""

    required_variables = ["test_variable_1", "test_variable_2"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that sums two variables."""
        return data[cls.required_variables[0]] + data[cls.required_variables[1]]


class TestMinimalDerivedVariable(derived.DerivedVariable):
    """A minimal test implementation with one required variable."""

    required_variables = ["single_variable"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that returns the variable unchanged."""
        return data[cls.required_variables[0]]


class TestDerivedVariableWithoutName(derived.DerivedVariable):
    """A test implementation that returns a DataArray without a name."""

    required_variables = ["single_variable"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that returns DataArray without name."""
        result = data[cls.required_variables[0]]
        result.name = None
        return result


class TestDerivedVariableAbstractClass:
    """Test the abstract base class DerivedVariable."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that DerivedVariable cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            derived.DerivedVariable()

    def test_name_property_default(self):
        """Test that the name property defaults to class name."""
        assert TestValidDerivedVariable().name == "TestValidDerivedVariable"

    def test_compute_method_calls_derive_variable(self, sample_dataset):
        """Test that compute method calls derive_variable and validates inputs."""
        result = TestValidDerivedVariable.compute(sample_dataset)

        assert isinstance(result, xr.DataArray)
        # Should be sum of test_variable_1 and test_variable_2
        expected = sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        xr.testing.assert_equal(result, expected)

    def test_compute_raises_error_missing_variables(self, sample_dataset):
        """Test that compute raises error when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_dataset.drop_vars("test_variable_2")

        with pytest.raises(
            ValueError, match="Input variable test_variable_2 not found in data"
        ):
            TestValidDerivedVariable.compute(incomplete_dataset)

    def test_required_variables_class_attribute(self):
        """Test that required_variables is properly defined as class attribute."""
        assert hasattr(TestValidDerivedVariable, "required_variables")
        assert TestValidDerivedVariable.required_variables == [
            "test_variable_1",
            "test_variable_2",
        ]


class TestMaybeDeriveVariablesFunction:
    """Comprehensive tests for the maybe_derive_variables function."""

    def test_only_string_variables(self, sample_dataset):
        """Test function with only string variables - should return unchanged."""
        variables = ["air_pressure_at_mean_sea_level", "surface_eastward_wind"]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the exact same dataset when no derived variables present
        xr.testing.assert_equal(result, sample_dataset)
        assert id(result) != id(sample_dataset)  # Should be a copy, not same object

    def test_empty_variable_list(self, sample_dataset):
        """Test function with empty variable list."""
        result = derived.maybe_derive_variables(sample_dataset, [])

        # Should return original dataset unchanged
        xr.testing.assert_equal(result, sample_dataset)

    def test_single_derived_variable_dataarray(self, sample_dataset):
        """Test with single derived variable that returns DataArray."""
        variables = [TestValidDerivedVariable()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Original variables should be preserved
        for var in sample_dataset.data_vars:
            assert var in result.data_vars
        # New derived variable should be added
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_multiple_derived_variables(self, sample_dataset):
        """Test with multiple derived variables."""
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Both derived variables should be added
        assert "TestValidDerivedVariable" in result.data_vars
        assert "TestMinimalDerivedVariable" in result.data_vars
        # Original dataset variables should be preserved
        for var in sample_dataset.data_vars:
            assert var in result.data_vars

    def test_mixed_string_and_derived_variables(self, sample_dataset):
        """Test with mix of string and derived variables."""
        variables = [
            "air_pressure_at_mean_sea_level",  # String variable (should be unchanged)
            TestValidDerivedVariable(),  # Derived variable instance
            TestMinimalDerivedVariable(),  # Another derived variable instance
        ]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Original variables should still be there
        assert "air_pressure_at_mean_sea_level" in result.data_vars
        # New derived variables should be added
        assert "TestValidDerivedVariable" in result.data_vars
        assert "TestMinimalDerivedVariable" in result.data_vars

    def test_dataarray_without_name_gets_assigned_name(self, sample_dataset):
        """Test DataArray without name gets assigned class name with warning."""
        variables = [TestDerivedVariableWithoutName()]

        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "TestDerivedVariableWithoutName" in result.data_vars
        # Verify the DataArray got the correct name assigned
        derived_var = result["TestDerivedVariableWithoutName"]
        assert derived_var.name == "TestDerivedVariableWithoutName"

    def test_kwargs_passed_to_compute(self, sample_dataset):
        """Test that kwargs are passed to derived variable compute methods."""

        class TestDerivedVariableWithKwargs(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                multiplier = kwargs.get("multiplier", 1)
                return data[cls.required_variables[0]] * multiplier

            @classmethod
            def compute(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                # Override to accept kwargs
                for v in cls.required_variables:
                    if v not in data.data_vars:
                        raise ValueError(f"Input variable {v} not found in data")
                return cls.derive_variable(data, **kwargs)

        variables = [TestDerivedVariableWithKwargs()]
        test_multiplier = 5.0

        result = derived.maybe_derive_variables(
            sample_dataset, variables, multiplier=test_multiplier
        )

        assert "TestDerivedVariableWithKwargs" in result.data_vars
        expected = sample_dataset["test_variable_1"] * test_multiplier
        xr.testing.assert_equal(result["TestDerivedVariableWithKwargs"], expected)

    def test_derived_variable_returns_dataset_different_dims(self, sample_dataset):
        """Test derived variable that returns Dataset with different dimensions."""

        class TestDatasetReturnVariable(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.Dataset:
                # Return a dataset with different dimensions
                new_coords = {"new_dim": [1, 2, 3], "other_dim": [10, 20]}
                return xr.Dataset(
                    {
                        "new_variable": xr.DataArray(
                            np.ones((3, 2)),
                            dims=["new_dim", "other_dim"],
                            coords=new_coords,
                        )
                    }
                )

        variables = [TestDatasetReturnVariable()]

        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the new dataset, not merge with original
        assert isinstance(result, xr.Dataset)
        assert "new_variable" in result.data_vars
        # Original dataset variables should NOT be present
        assert "test_variable_1" not in result.data_vars
        assert set(result.dims) == {"new_dim", "other_dim"}

    def test_derived_variable_missing_required_vars(self, sample_dataset):
        """Test derived variable with missing required variables."""

        class TestMissingVarDerived(derived.DerivedVariable):
            required_variables = ["nonexistent_variable"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data[cls.required_variables[0]]

        variables = [TestMissingVarDerived()]

        with pytest.raises(ValueError, match="Input variable nonexistent_variable"):
            derived.maybe_derive_variables(sample_dataset, variables)

    def test_no_derived_variables_in_list(self, sample_dataset):
        """Test when no derived variables are in the variable list."""
        variables = ["var1", "var2", "var3"]  # All strings

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return original dataset since no derived variables to process
        xr.testing.assert_equal(result, sample_dataset)

    def test_derived_data_dict_handling(self, sample_dataset):
        """Test internal derived_data dictionary logic."""
        # Test that derived_data dict is properly built and merged
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Both variables should exist
        assert "TestValidDerivedVariable" in result.data_vars
        assert "TestMinimalDerivedVariable" in result.data_vars

        # Check that the merge preserved all original variables
        original_vars = set(sample_dataset.data_vars.keys())
        result_vars = set(result.data_vars.keys())
        assert original_vars.issubset(result_vars)

    def test_derived_variable_compute_exception_propagates(self, sample_dataset):
        """Test that exceptions from derived variable compute methods propagate."""

        class TestExceptionDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                raise RuntimeError("Test exception from derive_variable")

        variables = [TestExceptionDerived()]

        with pytest.raises(RuntimeError, match="Test exception from derive_variable"):
            derived.maybe_derive_variables(sample_dataset, variables)

    def test_duplicate_derived_variable_names(self, sample_dataset):
        """Test behavior with multiple derived variables with same name."""

        class TestDuplicateName1(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data[cls.required_variables[0]] * 2

        class TestDuplicateName2(derived.DerivedVariable):
            required_variables = ["test_variable_2"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data[cls.required_variables[0]] * 3

        # Rename both to have same name by overriding name property
        TestDuplicateName1.__name__ = "SameName"
        TestDuplicateName2.__name__ = "SameName"

        variables = [TestDuplicateName1(), TestDuplicateName2()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Second variable should overwrite the first due to dict behavior
        assert "SameName" in result.data_vars
        # Should contain the result from the second variable (test_variable_2 * 3)
        expected = sample_dataset["test_variable_2"] * 3
        xr.testing.assert_equal(result["SameName"], expected)

    def test_early_return_from_dataset_with_different_dims(self, sample_dataset):
        """Test early return when first derived var returns dataset with diff dims."""

        class TestEarlyReturnDataset(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.Dataset:
                # Return dataset with different dims - should trigger early return
                return xr.Dataset(
                    {"special_var": xr.DataArray([1, 2, 3], dims=["special_dim"])}
                )

        class TestNeverExecuted(derived.DerivedVariable):
            required_variables = ["test_variable_2"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                # This should never be called due to early return
                return data[cls.required_variables[0]]

        variables = [TestEarlyReturnDataset(), TestNeverExecuted()]

        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the special dataset, not merged
        assert isinstance(result, xr.Dataset)
        assert "special_var" in result.data_vars
        assert "TestNeverExecuted" not in result.data_vars
        assert list(result.dims) == ["special_dim"]

    def test_derived_variable_returns_dataset_matching_dims(self, sample_dataset):
        """Test derived variable that returns Dataset with matching dimensions."""

        class TestDatasetMatchingDims(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.Dataset:
                # Return a dataset with same dimensions as input but multiple variables
                return xr.Dataset(
                    {
                        "derived_var_1": data["test_variable_1"] * 2,
                        "derived_var_2": data["test_variable_1"] + 10,
                        "derived_var_3": data["test_variable_1"] ** 2,
                    }
                )

        variables = [TestDatasetMatchingDims()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should merge the dataset variables, not return early
        assert isinstance(result, xr.Dataset)
        # All three derived variables should be present
        assert "derived_var_1" in result.data_vars
        assert "derived_var_2" in result.data_vars
        assert "derived_var_3" in result.data_vars
        # Original variables should still be present
        for var in sample_dataset.data_vars:
            assert var in result.data_vars
        # Verify the computed values are correct
        expected_var_1 = sample_dataset["test_variable_1"] * 2
        expected_var_2 = sample_dataset["test_variable_1"] + 10
        expected_var_3 = sample_dataset["test_variable_1"] ** 2
        xr.testing.assert_equal(result["derived_var_1"], expected_var_1)
        xr.testing.assert_equal(result["derived_var_2"], expected_var_2)
        xr.testing.assert_equal(result["derived_var_3"], expected_var_3)

    def test_mixed_dataarray_and_dataset_outputs(self, sample_dataset):
        """Test mix of derived variables returning DataArrays and matching Dataset."""

        class TestDataArrayOutput(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data["test_variable_1"] * 3

        class TestDatasetOutput(derived.DerivedVariable):
            required_variables = ["test_variable_2"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.Dataset:
                # Return dataset with matching dimensions
                return xr.Dataset(
                    {
                        "multi_var_1": data["test_variable_2"] / 2,
                        "multi_var_2": data["test_variable_2"] + 5,
                    }
                )

        variables = [TestDataArrayOutput(), TestDatasetOutput()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should merge all variables from both derived variables
        assert isinstance(result, xr.Dataset)
        # DataArray output should be present
        assert "TestDataArrayOutput" in result.data_vars
        # Dataset outputs should be present
        assert "multi_var_1" in result.data_vars
        assert "multi_var_2" in result.data_vars
        # Original variables should still be present
        for var in sample_dataset.data_vars:
            assert var in result.data_vars
        # Verify computed values
        expected_dataarray = sample_dataset["test_variable_1"] * 3
        expected_multi_1 = sample_dataset["test_variable_2"] / 2
        expected_multi_2 = sample_dataset["test_variable_2"] + 5
        xr.testing.assert_equal(result["TestDataArrayOutput"], expected_dataarray)
        xr.testing.assert_equal(result["multi_var_1"], expected_multi_1)
        xr.testing.assert_equal(result["multi_var_2"], expected_multi_2)


class TestUtilityFunctions:
    """Test utility functions in the derived module."""

    def test_maybe_pull_required_variables_from_derived_input_with_instances(self):
        """Test maybe_pull_required_variables_from_derived_input with instances."""
        incoming_variables = [
            "existing_variable",
            TestValidDerivedVariable(),
            TestMinimalDerivedVariable(),
            "another_existing_variable",
        ]

        result = derived.maybe_pull_required_variables_from_derived_input(
            incoming_variables
        )

        expected = [
            "existing_variable",
            "another_existing_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_pull_required_variables_from_derived_input_with_classes(self):
        """Test maybe_pull_required_variables_from_derived_input with classes."""
        incoming_variables = [
            "existing_variable",
            TestValidDerivedVariable,  # Class, not instance
            TestMinimalDerivedVariable,  # Class, not instance
        ]

        result = derived.maybe_pull_required_variables_from_derived_input(
            incoming_variables
        )

        expected = [
            "existing_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_pull_required_variables_only_strings(self):
        """Test maybe_pull_required_variables_from_derived_input with only strings."""
        incoming_variables = ["var1", "var2", "var3"]

        result = derived.maybe_pull_required_variables_from_derived_input(
            incoming_variables
        )

        assert result == incoming_variables


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions across the module."""

    def test_derived_variable_with_empty_dataset(self):
        """Test behavior with empty datasets."""
        empty_dataset = xr.Dataset()

        with pytest.raises(ValueError, match="Input variable .* not found in data"):
            TestValidDerivedVariable.compute(empty_dataset)

    def test_derived_variable_with_wrong_dimensions(self, sample_dataset):
        """Test behavior when variables have unexpected dimensions."""
        # Create dataset with wrong dimensions for test variables
        wrong_dim_dataset = sample_dataset.copy()
        wrong_dim_dataset["test_variable_1"] = xr.DataArray(
            np.ones((5,)),  # Wrong shape
            dims=["wrong_dim"],
            coords={"wrong_dim": range(5)},
        )

        # This should still work because xarray handles broadcasting
        result = TestValidDerivedVariable.compute(wrong_dim_dataset)
        assert isinstance(result, xr.DataArray)

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create a larger dataset to test performance characteristics
        time = pd.date_range("2021-01-01", freq="6h", periods=100)
        latitudes = np.linspace(-90, 90, 181)
        longitudes = np.linspace(0, 359, 360)

        large_dataset = xr.Dataset(
            {
                "test_variable_1": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(
                        0, 1, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
                "test_variable_2": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(
                        5, 2, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
            },
            coords={"time": time, "latitude": latitudes, "longitude": longitudes},
        )

        # This tests that the computation doesn't crash with larger data
        result = TestValidDerivedVariable.compute(large_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(time), len(latitudes), len(longitudes))


class TestIntegrationWithRealData:
    """Integration tests that simulate real-world usage patterns."""

    def test_pipeline_integration(self, sample_dataset):
        """Test integration of multiple derived variables in a pipeline."""
        # Simulate a pipeline that uses multiple derived variables
        variables_to_derive = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        # Step 1: Pull required variables
        required_vars = derived.maybe_pull_required_variables_from_derived_input(
            ["surface_wind_speed"] + variables_to_derive
        )

        # Step 2: Subset dataset to required variables
        available_vars = [
            var for var in required_vars if var in sample_dataset.data_vars
        ]
        subset_dataset = sample_dataset[available_vars]

        # Step 3: Derive variables
        final_dataset = derived.maybe_derive_variables(
            subset_dataset, variables_to_derive
        )

        # Verify results
        assert "TestValidDerivedVariable" in final_dataset.data_vars
        assert "TestMinimalDerivedVariable" in final_dataset.data_vars
        assert len(final_dataset.data_vars) >= 2

    @pytest.mark.parametrize(
        "variable_combination",
        [
            [TestValidDerivedVariable()],
            [TestMinimalDerivedVariable()],
            [TestValidDerivedVariable(), TestMinimalDerivedVariable()],
            [],
        ],
    )
    def test_parametrized_variable_combinations(
        self, sample_dataset, variable_combination
    ):
        """Test different combinations of derived variables."""
        result = derived.maybe_derive_variables(sample_dataset, variable_combination)

        assert isinstance(result, xr.Dataset)
        # Should have at least the original variables
        for var in sample_dataset.data_vars:
            assert var in result.data_vars


@pytest.fixture
def sample_tc_forecast_dataset():
    """Create a sample forecast dataset for TC testing."""
    time = pd.date_range("2023-09-01", periods=3, freq="12h")
    prediction_timedelta = np.array([0, 12, 24, 36], dtype="timedelta64[h]")
    lat = np.linspace(10, 40, 16)
    lon = np.linspace(-90, -60, 16)

    # Create realistic meteorological data
    data_shape = (len(time), len(lat), len(lon), len(prediction_timedelta))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(101325, 1000, data_shape),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(0, 10, data_shape),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(0, 10, data_shape),
            ),
            "geopotential": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(5000, 1000, data_shape) * 9.80665,
            ),
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "prediction_timedelta": prediction_timedelta,
        },
    )

    return dataset


@pytest.fixture
def sample_tc_tracks_dataset():
    """Create a sample TC tracks dataset."""
    time = pd.date_range("2023-09-01", periods=3, freq="12h")
    prediction_timedelta = np.array([0, 12, 24, 36], dtype="timedelta64[h]")

    data_shape = (len(time), len(prediction_timedelta))

    dataset = xr.Dataset(
        {
            "tc_slp": (
                ["time", "prediction_timedelta"],
                np.random.normal(101000, 1000, data_shape),
            ),
            "tc_latitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(15, 35, data_shape),
            ),
            "tc_longitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(-85, -65, data_shape),
            ),
            "tc_vmax": (
                ["time", "prediction_timedelta"],
                np.random.uniform(20, 60, data_shape),
            ),
            "track_id": (
                ["time", "prediction_timedelta"],
                np.random.randint(1, 5, data_shape),
            ),
        },
        coords={
            "time": time,
            "prediction_timedelta": prediction_timedelta,
        },
    )

    return dataset


class TestTropicalCycloneTrackVariables:
    """Test the base TropicalCycloneTrackVariables class."""

    def setup_method(self):
        """Clear cache before each test."""
        derived.TropicalCycloneTrackVariables.clear_cache()
        tropical_cyclone.clear_ibtracs_registry()

    def test_required_variables(self):
        """Test that required variables are properly defined."""
        expected_vars = [
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

        assert derived.TropicalCycloneTrackVariables.required_variables == expected_vars

    @patch(
        "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
    )
    @patch("extremeweatherbench.events.tropical_cyclone.generate_tc_variables")
    def test_get_or_compute_tracks_no_cache(
        self,
        mock_generate_tc_vars,
        mock_create_tracks,
        sample_tc_forecast_dataset,
        sample_tc_tracks_dataset,
    ):
        """Test track computation when not cached."""
        # Setup mocks
        mock_generate_tc_vars.return_value = sample_tc_forecast_dataset
        mock_create_tracks.return_value = sample_tc_tracks_dataset

        # Register IBTrACS data
        ibtracs_data = xr.Dataset(
            {
                "latitude": (["time"], [25.0, 26.0]),
                "longitude": (["time"], [-75.0, -74.0]),
            },
            coords={"time": pd.date_range("2023-09-01", periods=2, freq="6h")},
        )

        result = derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
            sample_tc_forecast_dataset, ibtracs_data=ibtracs_data
        )

        # Should call the necessary functions
        mock_generate_tc_vars.assert_called_once()
        mock_create_tracks.assert_called_once()

        # Should return the tracks dataset
        assert result is sample_tc_tracks_dataset

    @patch(
        "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
    )
    @patch("extremeweatherbench.events.tropical_cyclone.generate_tc_variables")
    def test_get_or_compute_tracks_with_cache(
        self,
        mock_generate_tc_vars,
        mock_create_tracks,
        sample_tc_forecast_dataset,
        sample_tc_tracks_dataset,
    ):
        """Test that cache is used when available."""
        # Manually add to cache
        cache_key = tropical_cyclone._generate_cache_key(sample_tc_forecast_dataset)
        tropical_cyclone._TC_TRACK_CACHE[cache_key] = sample_tc_tracks_dataset

        result = derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
            sample_tc_forecast_dataset,
            ibtracs_data=xr.Dataset(),  # Empty dataset
        )

        # Should not call computation functions
        mock_generate_tc_vars.assert_not_called()
        mock_create_tracks.assert_not_called()

        # Should return cached dataset
        assert result is sample_tc_tracks_dataset

    def test_get_or_compute_tracks_no_ibtracs_error(self, sample_tc_forecast_dataset):
        """Test error when no IBTrACS data is provided."""
        with pytest.raises(ValueError, match="No IBTrACS data provided"):
            derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
                sample_tc_forecast_dataset
            )

    @patch("extremeweatherbench.events.tropical_cyclone.get_ibtracs_data")
    def test_get_or_compute_tracks_from_registry(
        self, mock_get_ibtracs, sample_tc_forecast_dataset
    ):
        """Test getting IBTrACS data from registry using case_id."""
        # Mock registry return
        ibtracs_data = xr.Dataset(
            {
                "latitude": (["time"], [25.0]),
                "longitude": (["time"], [-75.0]),
            },
            coords={"time": [pd.Timestamp("2023-09-01")]},
        )

        mock_get_ibtracs.return_value = ibtracs_data

        with patch(
            "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
        ) as mock_create:
            mock_create.return_value = xr.Dataset()

            derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
                sample_tc_forecast_dataset, case_id="test_case_123"
            )

            # Should have tried to get from registry
            mock_get_ibtracs.assert_called_once_with("test_case_123")

    def test_clear_cache(self, sample_tc_tracks_dataset):
        """Test cache clearing functionality."""
        # Add something to cache
        tropical_cyclone._TC_TRACK_CACHE["test_key"] = sample_tc_tracks_dataset

        # Verify it's there
        assert len(tropical_cyclone._TC_TRACK_CACHE) == 1

        # Clear cache
        derived.TropicalCycloneTrackVariables.clear_cache()

        # Should be empty
        assert len(tropical_cyclone._TC_TRACK_CACHE) == 0

    @patch.object(derived.TropicalCycloneTrackVariables, "_get_or_compute_tracks")
    def test_derive_variable_base_class(
        self, mock_get_tracks, sample_tc_forecast_dataset, sample_tc_tracks_dataset
    ):
        """Test the base derive_variable method."""
        mock_get_tracks.return_value = sample_tc_tracks_dataset

        result = derived.TropicalCycloneTrackVariables.derive_variable(
            sample_tc_forecast_dataset
        )

        # Should call _get_or_compute_tracks
        mock_get_tracks.assert_called_once()

        # Should return a DataArray (converted from Dataset)
        assert isinstance(result, xr.Dataset)


class TestWindDataPreparation:
    """Test wind data preparation in TC track computation."""

    def test_prepare_wind_data_with_components_only(self):
        """Test preparation when only wind components are available."""
        # Create dataset with only wind components
        time = pd.date_range("2023-09-01", periods=2, freq="6h")
        lat = np.linspace(20, 30, 5)
        lon = np.linspace(-80, -70, 5)

        u_wind = np.random.normal(0, 10, (2, 5, 5))
        v_wind = np.random.normal(0, 10, (2, 5, 5))

        dataset = xr.Dataset(
            {
                "surface_eastward_wind": (["time", "latitude", "longitude"], u_wind),
                "surface_northward_wind": (["time", "latitude", "longitude"], v_wind),
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 5, 5)),
                ),
                "geopotential": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(5000, 1000, (2, 5, 5)) * 9.80665,
                ),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        # Mock the track computation to test preparation
        with patch(
            "extremeweatherbench.events.tropical_cyclone.generate_tc_variables"
        ) as mock_gen_vars:
            with patch(
                "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
            ) as mock_create:
                mock_gen_vars.return_value = dataset
                mock_create.return_value = xr.Dataset()

                derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
                    dataset,
                    ibtracs_data=xr.Dataset(
                        {
                            "latitude": (["time"], [25.0]),
                            "longitude": (["time"], [-75.0]),
                        },
                        coords={"time": [pd.Timestamp("2023-09-01")]},
                    ),
                )

                # Check that generate_tc_variables was called with data that has wind speed
                called_dataset = mock_gen_vars.call_args[0][0]
                assert "surface_wind_speed" in called_dataset.data_vars

                # Verify wind speed calculation
                expected_wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                calculated_wind_speed = called_dataset["surface_wind_speed"].values
                np.testing.assert_allclose(calculated_wind_speed, expected_wind_speed)

    def test_prepare_wind_data_with_existing_wind_speed(self):
        """Test preparation when wind speed already exists."""
        # Create dataset with existing wind speed
        time = pd.date_range("2023-09-01", periods=2, freq="6h")
        lat = np.linspace(20, 30, 5)
        lon = np.linspace(-80, -70, 5)

        wind_speed = np.random.uniform(0, 30, (2, 5, 5))

        dataset = xr.Dataset(
            {
                "surface_wind_speed": (["time", "latitude", "longitude"], wind_speed),
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 5, 5)),
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 5, 5)),
                ),
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 5, 5)),
                ),
                "geopotential": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(5000, 1000, (2, 5, 5)) * 9.80665,
                ),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        # Mock the track computation
        with patch(
            "extremeweatherbench.events.tropical_cyclone.generate_tc_variables"
        ) as mock_gen_vars:
            with patch(
                "extremeweatherbench.events.tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter"
            ) as mock_create:
                mock_gen_vars.return_value = dataset
                mock_create.return_value = xr.Dataset()

                derived.TropicalCycloneTrackVariables._get_or_compute_tracks(
                    dataset,
                    ibtracs_data=xr.Dataset(
                        {
                            "latitude": (["time"], [25.0]),
                            "longitude": (["time"], [-75.0]),
                        },
                        coords={"time": [pd.Timestamp("2023-09-01")]},
                    ),
                )

                # Check that the original wind speed was preserved
                called_dataset = mock_gen_vars.call_args[0][0]
                np.testing.assert_array_equal(
                    called_dataset["surface_wind_speed"].values, wind_speed
                )
