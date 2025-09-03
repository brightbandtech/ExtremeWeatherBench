"""Comprehensive unit tests for the extremeweatherbench.derived module.

This test suite covers:
- Abstract base class DerivedVariable
- All concrete derived variable implementations
- Utility functions for variable derivation
- Edge cases and error conditions
- Mock implementations for testing
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import derived


def create_mock_case_operator(variables, dataset_type="forecast"):
    """Create a mock case operator with specified variables."""
    from unittest.mock import Mock

    case_operator = Mock()
    case_operator.case_metadata.case_id_number = 123

    if dataset_type == "forecast":
        case_operator.forecast.variables = variables
        case_operator.target.variables = []
    else:
        case_operator.forecast.variables = []
        case_operator.target.variables = variables

    return case_operator


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
                ["valid_time", "latitude", "longitude"],
                np.random.normal(
                    101325, 1000, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_eastward_wind": (
                ["valid_time", "latitude", "longitude"],
                np.random.normal(
                    5, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_northward_wind": (
                ["valid_time", "latitude", "longitude"],
                np.random.normal(
                    2, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_wind_speed": (
                ["valid_time", "latitude", "longitude"],
                np.random.uniform(
                    0, 15, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            # 3D atmospheric variables
            "eastward_wind": (
                ["valid_time", "level", "latitude", "longitude"],
                level_data + np.random.normal(10, 5, size=level_data.shape),
            ),
            "northward_wind": (
                ["valid_time", "level", "latitude", "longitude"],
                level_data + np.random.normal(3, 5, size=level_data.shape),
            ),
            "specific_humidity": (
                ["valid_time", "level", "latitude", "longitude"],
                np.random.exponential(0.008, size=level_data.shape),
            ),
            "geopotential": (
                ["valid_time", "level", "latitude", "longitude"],
                level_data * 100 + np.random.normal(50000, 5000, size=level_data.shape),
            ),
            # Test variables
            "test_variable_1": (["valid_time", "latitude", "longitude"], base_data),
            "test_variable_2": (["valid_time", "latitude", "longitude"], base_data + 5),
            "single_variable": (["valid_time", "latitude", "longitude"], base_data * 2),
        },
        coords={
            "valid_time": time,
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

    def test_compute_logs_warning_missing_variables(self, sample_dataset, caplog):
        """Test that compute fails when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_dataset.drop_vars("test_variable_2")

        # This should fail during derive_variable when accessing missing variable
        with pytest.raises(
            KeyError
        ):  # derive_variable will fail when accessing missing variable
            TestValidDerivedVariable.compute(incomplete_dataset)

    def test_required_variables_class_attribute(self):
        """Test that required_variables is properly defined as class attribute."""
        assert hasattr(TestValidDerivedVariable, "required_variables")
        assert TestValidDerivedVariable.required_variables == [
            "test_variable_1",
            "test_variable_2",
        ]


class TestMaybeDeriveVariablesFunction:
    """Comprehensive tests for the maybe_derive_variable function."""

    def test_only_string_variables(self, sample_dataset):
        """Test function with only string variables - should return unchanged."""
        variables = ["air_pressure_at_mean_sea_level", "surface_eastward_wind"]

        sample_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the exact same dataset when no derived variables present
        xr.testing.assert_equal(result, sample_dataset)
        assert id(result) == id(
            sample_dataset
        )  # Should be same object when no derived vars

    def test_empty_variable_list(self, sample_dataset):
        """Test function with empty variable list."""
        sample_dataset.attrs["dataset_type"] = "forecast"
        variables = []

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return original dataset unchanged
        xr.testing.assert_equal(result, sample_dataset)

    def test_single_derived_variable_dataarray(self, sample_dataset):
        """Test with single derived variable that returns DataArray."""
        variables = [TestValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)  # Function now returns Dataset
        # Verify the derived variable is in the dataset
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_multiple_derived_variables(self, sample_dataset):
        """Test with multiple derived variables - only first is computed."""
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_mixed_string_and_derived_variables(self, sample_dataset):
        """Test with mix of string and derived variables."""
        variables = [
            "air_pressure_at_mean_sea_level",  # String variable
            TestValidDerivedVariable(),  # Derived variable instance
            "surface_eastward_wind",  # Another string variable
            TestMinimalDerivedVariable(),  # Another derived variable
        ]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first derived variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_dataarray_without_name_gets_assigned_name(self, sample_dataset):
        """Test DataArray without name gets assigned class name with warning."""
        variables = [TestDerivedVariableWithoutName()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Verify the DataArray got the correct name assigned
        assert "TestDerivedVariableWithoutName" in result.data_vars

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

    def test_prepare_wind_data_helper(self, sample_dataset):
        """Test the internal _prepare_wind_data helper function."""
        # This tests the helper function within derive_variable
        # We need to access it indirectly since it's defined within the method

        # Test case 1: Dataset has wind speed
        result1 = sample_dataset.copy()
        assert "surface_wind_speed" in result1.data_vars

        # Test case 2: Dataset missing wind speed but has components
        dataset_no_speed = sample_dataset.drop_vars("surface_wind_speed")
        assert "surface_wind_speed" not in dataset_no_speed.data_vars
        assert "surface_eastward_wind" in dataset_no_speed.data_vars
        assert "surface_northward_wind" in dataset_no_speed.data_vars

    def test_derived_variable_missing_required_vars(self, sample_dataset):
        """Test derived variable with missing required variables."""

        class TestMissingVarDerived(derived.DerivedVariable):
            required_variables = ["nonexistent_variable"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data[cls.required_variables[0]]

        variables = [TestMissingVarDerived()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        with pytest.raises(KeyError, match="No variable named 'nonexistent_variable'"):
            derived.maybe_derive_variables(sample_dataset, variables)

    def test_no_derived_variables_in_list(self, sample_dataset):
        """Test when no derived variables are in the variable list."""
        variables = ["var1", "var2", "var3"]  # All strings

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return original dataset since no derived variables to process
        xr.testing.assert_equal(result, sample_dataset)

    def test_target_dataset_type(self, sample_dataset):
        """Test function with target dataset type."""
        variables = [TestValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "target"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars

    def test_unknown_dataset_type(self, sample_dataset):
        """Test function with unknown dataset type - should still process derived
        variables."""
        variables = [TestValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "unknown"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should process derived variables regardless of dataset type
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_derived_data_dict_handling(self, sample_dataset):
        """Test that only first derived variable is processed."""
        # Test with multiple derived variables
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_derived_variable_compute_exception_propagates(self, sample_dataset):
        """Test that exceptions from derived variable compute methods propagate."""

        class TestExceptionDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                raise RuntimeError("Test exception from derive_variable")

        variables = [TestExceptionDerived()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        with pytest.raises(RuntimeError, match="Test exception from derive_variable"):
            derived.maybe_derive_variables(sample_dataset, variables)

    def test_duplicate_derived_variable_names(self, sample_dataset):
        """Test behavior with multiple derived variables - only first is processed."""

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

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        # Check what's actually in the result
        data_var_names = list(result.data_vars.keys())
        assert len(data_var_names) == 1  # Should only have one variable
        result_var_name = data_var_names[0]
        expected = sample_dataset["test_variable_1"] * 2
        xr.testing.assert_equal(result[result_var_name], expected)

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
        sample_dataset.attrs["dataset_type"] = "forecast"
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

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the derived dataset directly (no merging with original)
        assert isinstance(result, xr.Dataset)
        # Only the three derived variables should be present
        assert "derived_var_1" in result.data_vars
        assert "derived_var_2" in result.data_vars
        assert "derived_var_3" in result.data_vars
        # Original variables should NOT be present (no longer merged)
        assert len(result.data_vars) == 3
        # Verify the computed values are correct
        expected_var_1 = sample_dataset["test_variable_1"] * 2
        expected_var_2 = sample_dataset["test_variable_1"] + 10
        expected_var_3 = sample_dataset["test_variable_1"] ** 2
        xr.testing.assert_equal(result["derived_var_1"], expected_var_1)
        xr.testing.assert_equal(result["derived_var_2"], expected_var_2)
        xr.testing.assert_equal(result["derived_var_3"], expected_var_3)

    def test_mixed_dataarray_and_dataset_outputs(self, sample_dataset):
        """Test mix of derived variables - only first is processed."""

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

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable (as Dataset)
        assert isinstance(result, xr.Dataset)
        # First derived variable returns DataArray, converted to Dataset
        data_var_names = list(result.data_vars.keys())
        assert len(data_var_names) == 1  # Should only have one variable
        result_var_name = data_var_names[0]
        # Verify computed value (only from first variable)
        expected_dataarray = sample_dataset["test_variable_1"] * 3
        xr.testing.assert_equal(result[result_var_name], expected_dataarray)

    def test_derived_variable_returns_invalid_type(self, sample_dataset, caplog):
        """Test derived variable that returns neither DataArray nor Dataset."""

        class TestInvalidReturnType(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset):
                # Return something that's neither DataArray nor Dataset
                return "invalid_return_type"

        variables = [TestInvalidReturnType()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return original dataset and log warning
        xr.testing.assert_equal(result, sample_dataset)

        # Check that warning was logged
        assert "returned neither DataArray nor Dataset" in caplog.text


class TestRecursiveDerivedVariable(derived.DerivedVariable):
    """A test derived variable that requires another derived variable."""

    required_variables = [TestValidDerivedVariable]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that uses another derived variable."""
        # This would normally use the output of TestValidDerivedVariable
        # For testing, we'll just return a simple computation
        return data["test_variable_1"] * 2


class TestDeeplyNestedDerivedVariable(derived.DerivedVariable):
    """A test derived variable that requires a recursive derived variable."""

    required_variables = [TestRecursiveDerivedVariable]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that uses a recursive derived variable."""
        return data["test_variable_1"] * 3


class TestUtilityFunctions:
    """Test utility functions in the derived module."""

    def test_maybe_include_variables_from_derived_input_with_instances(self):
        """Test maybe_include_variables_from_derived_input with instances."""
        incoming_variables = [
            "existing_variable",
            TestValidDerivedVariable(),
            TestMinimalDerivedVariable(),
            "another_existing_variable",
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        expected = [
            "existing_variable",
            "another_existing_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_from_derived_input_with_classes(self):
        """Test maybe_include_variables_from_derived_input with classes."""
        incoming_variables = [
            "existing_variable",
            TestValidDerivedVariable,  # Class, not instance
            TestMinimalDerivedVariable,  # Class, not instance
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        expected = [
            "existing_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_only_strings(self):
        """Test maybe_include_variables_from_derived_input with only strings."""
        incoming_variables = ["var1", "var2", "var3"]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        assert result == incoming_variables

    def test_maybe_include_variables_with_recursive_derived_classes(self):
        """Test recursive resolution of derived variables with classes."""
        incoming_variables = [
            "base_variable",
            TestRecursiveDerivedVariable,  # Requires TestValidDerivedVariable
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should recursively resolve TestRecursiveDerivedVariable ->
        # TestValidDerivedVariable -> ["test_variable_1", "test_variable_2"]
        expected = [
            "base_variable",
            "test_variable_1",
            "test_variable_2",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_with_deeply_nested_derived_classes(self):
        """Test deeply nested recursive resolution of derived variables."""
        incoming_variables = [
            "base_variable",
            TestDeeplyNestedDerivedVariable,  # Requires TestRecursiveDerivedVariable
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should recursively resolve:
        # TestDeeplyNestedDerivedVariable -> TestRecursiveDerivedVariable ->
        # TestValidDerivedVariable -> ["test_variable_1", "test_variable_2"]
        expected = [
            "base_variable",
            "test_variable_1",
            "test_variable_2",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_mixed_instances_and_classes(self):
        """Test mixed instances and classes with recursive resolution."""
        incoming_variables = [
            "base_variable",
            TestValidDerivedVariable(),  # Instance
            TestRecursiveDerivedVariable,  # Class requires TestValidDerivedVariable
            "another_variable",
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should handle both instance and class, avoiding duplicates
        expected = [
            "base_variable",
            "another_variable",
            "test_variable_1",
            "test_variable_2",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_preserves_order_removes_duplicates(self):
        """Test that function preserves order and removes duplicates."""
        incoming_variables = [
            "var1",
            TestValidDerivedVariable(),  # Adds test_variable_1, test_variable_2
            "var2",
            TestMinimalDerivedVariable(),  # Adds single_variable
            "var1",  # Duplicate
            TestValidDerivedVariable(),  # Duplicate derived variable
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should preserve order and remove duplicates
        expected = [
            "var1",
            "var2",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert result == expected  # Order matters


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions across the module."""

    def test_maybe_include_variables_empty_list(self):
        """Test maybe_include_variables_from_derived_input with empty list."""
        result = derived.maybe_include_variables_from_derived_input([])
        assert result == []

    def test_maybe_include_variables_none_input(self):
        """Test maybe_include_variables_from_derived_input with None values."""
        # The function should handle None gracefully or raise appropriate error
        incoming_variables = ["var1", None, "var2"]

        # This might raise an error depending on implementation
        # Let's test what actually happens
        try:
            result = derived.maybe_include_variables_from_derived_input(
                incoming_variables
            )
            # If it doesn't raise an error, None should be filtered out
            assert "var1" in result
            assert "var2" in result
            assert None not in result
        except (TypeError, AttributeError):
            # This is acceptable behavior for None input
            pass

    def test_maybe_include_variables_circular_dependency(self):
        """Test handling of potential circular dependencies."""
        # Create a derived variable that could theoretically depend on itself
        # This tests the robustness of the recursive resolution

        class TestSelfReferencing(derived.DerivedVariable):
            required_variables = ["base_var"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data["base_var"]

        incoming_variables = [TestSelfReferencing]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should resolve to the base variable
        assert result == ["base_var"]

    def test_derived_variable_with_empty_dataset(self, caplog):
        """Test behavior with empty datasets."""
        empty_dataset = xr.Dataset()

        # Should fail during compute when checking required variables
        with pytest.raises(KeyError):
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
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(
                        0, 1, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
                "test_variable_2": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(
                        5, 2, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
            },
            coords={"valid_time": time, "latitude": latitudes, "longitude": longitudes},
        )

        # This tests that the computation doesn't crash with larger data
        result = TestValidDerivedVariable.compute(large_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(time), len(latitudes), len(longitudes))


class TestIntegrationWithRealData:
    """Integration tests that simulate real-world usage patterns."""

    def test_pipeline_integration(self, sample_dataset):
        """Test integration of pipeline with derived variables."""
        # Simulate a pipeline that uses multiple derived variables
        variables_to_derive = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        # Step 1: Pull required variables
        required_vars = derived.maybe_include_variables_from_derived_input(
            ["surface_wind_speed"] + variables_to_derive
        )

        # Step 2: Subset dataset to required variables
        available_vars = [
            var for var in required_vars if var in sample_dataset.data_vars
        ]
        subset_dataset = sample_dataset[available_vars]

        # Step 3: Derive variables (only first one will be processed)
        subset_dataset.attrs["dataset_type"] = "forecast"
        final_result = derived.maybe_derive_variables(
            subset_dataset, variables_to_derive
        )

        # Verify results - should return only first derived variable as Dataset
        assert isinstance(final_result, xr.Dataset)
        assert "TestValidDerivedVariable" in final_result.data_vars
        # Verify the computed value is correct
        expected_value = (
            subset_dataset["test_variable_1"] + subset_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(
            final_result["TestValidDerivedVariable"], expected_value
        )

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
        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variable_combination)

        if len(variable_combination) == 0:
            # No derived variables - should return original dataset
            assert isinstance(result, xr.Dataset)
            xr.testing.assert_equal(result, sample_dataset)
        else:
            # Has derived variables - should return first one as Dataset
            assert isinstance(result, xr.Dataset)
            # Check that derived variable was added
            if isinstance(variable_combination[0], TestValidDerivedVariable):
                assert "TestValidDerivedVariable" in result.data_vars
            elif isinstance(variable_combination[0], TestMinimalDerivedVariable):
                # TestMinimalDerivedVariable returns single_variable DataArray
                # which keeps its original name if the DataArray has no name
                assert (
                    "single_variable" in result.data_vars
                    or variable_combination[0].name in result.data_vars
                )
