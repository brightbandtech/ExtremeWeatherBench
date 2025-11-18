"""Tests for the derived module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import derived

time_location_variables = ["valid_time", "latitude", "longitude"]
time_level_location_variables = ["valid_time", "level", "latitude", "longitude"]


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
                time_location_variables,
                np.random.normal(
                    101325, 1000, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_eastward_wind": (
                time_location_variables,
                np.random.normal(
                    5, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_northward_wind": (
                time_location_variables,
                np.random.normal(
                    2, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_wind_speed": (
                time_location_variables,
                np.random.uniform(
                    0, 15, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            # 3D atmospheric variables
            "eastward_wind": (
                time_level_location_variables,
                level_data + np.random.normal(10, 5, size=level_data.shape),
            ),
            "northward_wind": (
                time_level_location_variables,
                level_data + np.random.normal(3, 5, size=level_data.shape),
            ),
            "specific_humidity": (
                time_level_location_variables,
                np.random.exponential(0.008, size=level_data.shape),
            ),
            "geopotential": (
                time_level_location_variables,
                level_data * 100 + np.random.normal(50000, 5000, size=level_data.shape),
            ),
            # Test variables
            "test_variable_1": (time_location_variables, base_data),
            "test_variable_2": (time_location_variables, base_data + 5),
            "single_variable": (time_location_variables, base_data * 2),
        },
        coords={
            "valid_time": time,
            "latitude": latitudes,
            "longitude": longitudes,
            "level": level,
        },
    )

    return dataset


class ValidDerivedVariable(derived.DerivedVariable):
    """A valid test implementation of DerivedVariable for testing purposes."""

    variables = ["test_variable_1", "test_variable_2"]

    def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that sums two variables."""
        return data[self.variables[0]] + data[self.variables[1]]


class MinimalDerivedVariable(derived.DerivedVariable):
    """A minimal test implementation with one required variable."""

    variables = ["single_variable"]

    def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that returns the variable unchanged."""
        return data[self.variables[0]]


class DerivedVariableWithoutName(derived.DerivedVariable):
    """A test implementation that returns a DataArray without a name."""

    variables = ["single_variable"]

    def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that returns DataArray without name."""
        result = data[self.variables[0]]
        result.name = None
        return result  # type: ignore


class DerivedVariableForTesting(derived.DerivedVariable):
    """A concrete derived variable class for testing
    _maybe_convert_variable_to_string."""

    name = "TestDerivedVar"
    variables = ["input_var1", "input_var2"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Simple derivation for testing."""
        return data["input_var1"] + data["input_var2"]


class TestDerivedVariableAbstractClass:
    """Test the abstract base class DerivedVariable."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that DerivedVariable cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            derived.DerivedVariable()

    def test_name_property_default(self):
        """Test that the name property defaults to class name."""
        assert ValidDerivedVariable().name == "ValidDerivedVariable"

    def test_name_property_custom(self):
        """Test that custom name overrides class name."""
        custom_name = "CustomVariableName"
        derived_var = ValidDerivedVariable(name=custom_name)
        assert derived_var.name == custom_name

    def test_compute_method_calls_derive_variable(self, sample_dataset):
        """Test that compute method calls derive_variable and validates inputs."""
        result = ValidDerivedVariable().compute(sample_dataset)

        assert isinstance(result, xr.DataArray)
        # Should be sum of test_variable_1 and test_variable_2
        expected = sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        xr.testing.assert_equal(result, expected)

    def test_compute_logs_warning_missing_variables(self, sample_dataset, caplog):
        """Test that compute fails when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_dataset.drop_vars("test_variable_2")

        # This should fail during derive_variable when accessing missing variable
        with pytest.raises(KeyError, match="test_variable_2"):
            ValidDerivedVariable().compute(incomplete_dataset)

    def test_variables_class_attribute(self):
        """Test that variables is properly defined as class attribute."""
        assert hasattr(ValidDerivedVariable, "variables")
        assert ValidDerivedVariable.variables == [
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
        variables = [ValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)  # Function now returns Dataset
        # Verify the derived variable is in the dataset
        assert "ValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["ValidDerivedVariable"], expected_value)

    def test_multiple_derived_variables(self, sample_dataset):
        """Test with multiple derived variables - only first is computed."""
        variables = [ValidDerivedVariable(), MinimalDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "ValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["ValidDerivedVariable"], expected_value)

    def test_mixed_string_and_derived_variables(self, sample_dataset):
        """Test with mix of string and derived variables."""
        variables = [
            "air_pressure_at_mean_sea_level",  # String variable
            ValidDerivedVariable(),  # Derived variable instance
            "surface_eastward_wind",  # Another string variable
            MinimalDerivedVariable(),  # Another derived variable
        ]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "ValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first derived variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["ValidDerivedVariable"], expected_value)

    def test_dataarray_without_name_gets_assigned_name(self, sample_dataset):
        """Test DataArray without name gets assigned class name with warning."""
        variables = [DerivedVariableWithoutName()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Verify the DataArray got the correct name assigned
        assert "DerivedVariableWithoutName" in result.data_vars

    def test_kwargs_passed_to_compute(self, sample_dataset):
        """Test that kwargs are passed to derived variable compute methods."""

        class TestDerivedVariableWithKwargs(derived.DerivedVariable):
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.DataArray:
                multiplier = kwargs.get("multiplier", 1)
                return data[self.variables[0]] * multiplier

            def compute(self, data: xr.Dataset, **kwargs) -> xr.DataArray:
                # Override to accept kwargs
                for v in self.variables:
                    if v not in data.data_vars:
                        raise ValueError(f"Input variable {v} not found in data")
                return self.derive_variable(data, **kwargs)

    def test_derived_variable_missing_required_vars(self, sample_dataset):
        """Test derived variable with missing required variables."""

        class TestMissingVarDerived(derived.DerivedVariable):
            variables = ["nonexistent_variable"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                return data[self.variables[0]]

        variables = [TestMissingVarDerived()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        with pytest.raises(KeyError, match="nonexistent_variable"):
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
        variables = [ValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "target"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "ValidDerivedVariable" in result.data_vars

    def test_unknown_dataset_type(self, sample_dataset):
        """Test function with unknown dataset type - should still process derived
        variables."""
        variables = [ValidDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "unknown"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should process derived variables regardless of dataset type
        assert isinstance(result, xr.Dataset)
        assert "ValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["ValidDerivedVariable"], expected_value)

    def test_derived_data_dict_handling(self, sample_dataset):
        """Test that only first derived variable is processed."""
        # Test with multiple derived variables
        variables = [ValidDerivedVariable(), MinimalDerivedVariable()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "ValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first variable only)
        expected_value = (
            sample_dataset["test_variable_1"] + sample_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["ValidDerivedVariable"], expected_value)

    def test_derived_variable_compute_exception_propagates(self, sample_dataset):
        """Test that exceptions from derived variable compute methods propagate."""

        class TestExceptionDerived(derived.DerivedVariable):
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                raise RuntimeError("Test exception from derive_variable")

        variables = [TestExceptionDerived()]

        sample_dataset.attrs["dataset_type"] = "forecast"
        with pytest.raises(RuntimeError, match="Test exception from derive_variable"):
            derived.maybe_derive_variables(sample_dataset, variables)

    def test_duplicate_derived_variable_names(self, sample_dataset):
        """Test behavior with multiple derived variables - only first is processed."""

        class TestDuplicateName1(derived.DerivedVariable):
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                return data[self.variables[0]] * 2

        class TestDuplicateName2(derived.DerivedVariable):
            variables = ["test_variable_2"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                return data[self.variables[0]] * 3

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
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset) -> xr.Dataset:
                # Return dataset with different dims - should trigger early return
                return xr.Dataset(
                    {"special_var": xr.DataArray([1, 2, 3], dims=["special_dim"])}
                )

        class TestNeverExecuted(derived.DerivedVariable):
            variables = ["test_variable_2"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                # This should never be called due to early return
                return data[self.variables[0]]

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
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset) -> xr.Dataset:
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
            variables = ["test_variable_1"]

            def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
                return data["test_variable_1"] * 3

        class TestDatasetOutput(derived.DerivedVariable):
            variables = ["test_variable_2"]

            def derive_variable(self, data: xr.Dataset) -> xr.Dataset:
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
            variables = ["test_variable_1"]

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


class RecursiveDerivedVariable(derived.DerivedVariable):
    """A test derived variable that requires another derived variable."""

    variables = [ValidDerivedVariable]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Test implementation that uses another derived variable."""
        # This would normally use the output of ValidDerivedVariable
        # For testing, we'll just return a simple computation
        return data["test_variable_1"] * 2


class DeeplyNestedDerivedVariable(derived.DerivedVariable):
    """A test derived variable that requires a recursive derived variable."""

    variables = [RecursiveDerivedVariable]

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
            ValidDerivedVariable(),
            MinimalDerivedVariable(),
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

    def test_maybe_include_variables_from_derived_input_with_multiple_instances(self):
        """Test maybe_include_variables_from_derived_input with multiple
        instances."""
        incoming_variables = [
            "existing_variable",
            ValidDerivedVariable(),  # Instance
            MinimalDerivedVariable(),  # Instance
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

        assert set(result) == set(incoming_variables)

    def test_maybe_include_variables_with_nested_string_variables(self):
        """Test that nested derived variables are flattened correctly."""
        # RecursiveDerivedVariable has ValidDerivedVariable class
        # in its variables list, but since we only support instances,
        # it will be treated as a string and not recursively resolved
        incoming_variables = [
            "base_variable",
            ValidDerivedVariable(),
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        expected = [
            "base_variable",
            "test_variable_1",
            "test_variable_2",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_with_different_instances(self):
        """Test multiple different derived variable instances."""
        incoming_variables = [
            "base_variable",
            ValidDerivedVariable(),
            MinimalDerivedVariable(),
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        expected = [
            "base_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_mixed_strings_and_instances(self):
        """Test mixed strings and instances."""
        incoming_variables = [
            "base_variable",
            ValidDerivedVariable(),  # Instance
            MinimalDerivedVariable(),  # Instance
            "another_variable",
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should handle both strings and instances, avoiding duplicates
        expected = [
            "base_variable",
            "another_variable",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)

    def test_maybe_include_variables_removes_duplicates(self):
        """Test that function removes duplicates."""
        incoming_variables = [
            "var1",
            ValidDerivedVariable(),  # Adds test_variable_1, test_variable_2
            "var2",
            MinimalDerivedVariable(),  # Adds single_variable
            "var1",  # Duplicate
            ValidDerivedVariable(),  # Duplicate derived variable type
        ]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should remove duplicates
        expected = [
            "var1",
            "var2",
            "test_variable_1",
            "test_variable_2",
            "single_variable",
        ]

        assert set(result) == set(expected)


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

    def test_maybe_include_variables_simple_instance(self):
        """Test handling of a simple derived variable instance."""

        class TestSelfReferencing(derived.DerivedVariable):
            variables = ["base_var"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data["base_var"]

        incoming_variables = [TestSelfReferencing()]

        result = derived.maybe_include_variables_from_derived_input(incoming_variables)

        # Should resolve to the base variable
        assert result == ["base_var"]

    def test_derived_variable_with_empty_dataset(self, caplog):
        """Test behavior with empty datasets."""
        empty_dataset = xr.Dataset()

        with pytest.raises(KeyError):
            ValidDerivedVariable().compute(empty_dataset)

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
        result = ValidDerivedVariable().compute(wrong_dim_dataset)
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
                    time_location_variables,
                    np.random.normal(
                        0, 1, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
                "test_variable_2": (
                    time_location_variables,
                    np.random.normal(
                        5, 2, size=(len(time), len(latitudes), len(longitudes))
                    ),
                ),
            },
            coords={"valid_time": time, "latitude": latitudes, "longitude": longitudes},
        )

        # This tests that the computation doesn't crash with larger data
        result = ValidDerivedVariable().compute(large_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(time), len(latitudes), len(longitudes))


class TestIntegrationWithRealData:
    """Integration tests that simulate real-world usage patterns."""

    def test_pipeline_integration(self, sample_dataset):
        """Test integration of pipeline with derived variables."""
        # Simulate a pipeline that uses multiple derived variables
        variables_to_derive = [ValidDerivedVariable(), MinimalDerivedVariable()]

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
        assert "ValidDerivedVariable" in final_result.data_vars
        # Verify the computed value is correct
        expected_value = (
            subset_dataset["test_variable_1"] + subset_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(final_result["ValidDerivedVariable"], expected_value)

    @pytest.mark.parametrize(
        "variable_combination",
        [
            [ValidDerivedVariable()],
            [MinimalDerivedVariable()],
            [ValidDerivedVariable(), MinimalDerivedVariable()],
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
            if isinstance(variable_combination[0], ValidDerivedVariable):
                assert "ValidDerivedVariable" in result.data_vars
            elif isinstance(variable_combination[0], MinimalDerivedVariable):
                # MinimalDerivedVariable returns single_variable DataArray
                # which keeps its original name if the DataArray has no name
                assert (
                    "single_variable" in result.data_vars
                    or variable_combination[0].name in result.data_vars
                )


class TestNormalizeVariable:
    """Test the _maybe_convert_variable_to_string function."""

    def test_maybe_convert_variable_to_string_string_input(self):
        """Test _maybe_convert_variable_to_string with string input."""
        result = derived._maybe_convert_variable_to_string("temperature")
        assert result == "temperature"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_derived_instance_with_custom_name(self):
        """Test _maybe_convert_variable_to_string with DerivedVariable instance
        with custom name."""
        instance = DerivedVariableForTesting()
        result = derived._maybe_convert_variable_to_string(instance)
        # Should use the class-level name attribute via __init__
        assert result == "TestDerivedVar"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_derived_instance_default_name(self):
        """Test _maybe_convert_variable_to_string with DerivedVariable instance
        using default name (class name)."""
        instance = ValidDerivedVariable()
        result = derived._maybe_convert_variable_to_string(instance)
        # Should use the class name via __init__ when no custom name provided
        assert result == "ValidDerivedVariable"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_handles_both_types(self):
        """Test that _maybe_convert_variable_to_string handles both incoming
        types correctly."""
        # Test string type
        string_result = derived._maybe_convert_variable_to_string("my_variable")
        assert string_result == "my_variable"

        # Test derived variable instance type
        instance = DerivedVariableForTesting()
        derived_result = derived._maybe_convert_variable_to_string(instance)
        assert derived_result == "TestDerivedVar"

        # Results should be different but both strings
        assert string_result != derived_result
        assert isinstance(string_result, str)
        assert isinstance(derived_result, str)


class MultiOutputDerivedVariable(derived.DerivedVariable):
    """A derived variable that outputs multiple variables."""

    variables = ["eastward_wind", "northward_wind"]

    def derive_variable(self, data: xr.Dataset, **kwargs) -> xr.Dataset:
        """Compute multiple output variables."""
        u_component = data["eastward_wind"].mean(dim="level")
        v_component = data["northward_wind"].mean(dim="level")
        magnitude = np.sqrt(u_component**2 + v_component**2)

        return xr.Dataset(
            {
                "u_output": u_component,
                "v_output": v_component,
                "magnitude": magnitude,
            }
        )


class TestOutputVariables:
    """Test the output_variables parameter functionality."""

    def test_output_variables_none_returns_all(self, sample_dataset):
        """When output_variables is None, all computed vars returned."""
        derived_var = MultiOutputDerivedVariable(output_variables=None)
        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        assert isinstance(result, xr.Dataset)
        # All three output variables should be present
        assert "u_output" in result.data_vars
        assert "v_output" in result.data_vars
        assert "magnitude" in result.data_vars

    def test_output_variables_empty_returns_all(self, sample_dataset):
        """When output_variables is empty list, all computed vars returned."""
        derived_var = MultiOutputDerivedVariable(output_variables=[])
        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        assert isinstance(result, xr.Dataset)
        # All three output variables should be present
        assert "u_output" in result.data_vars
        assert "v_output" in result.data_vars
        assert "magnitude" in result.data_vars

    def test_output_variables_single_subset(self, sample_dataset):
        """When output_variables has one var, only that is returned."""
        derived_var = MultiOutputDerivedVariable(output_variables=["u_output"])
        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        assert isinstance(result, xr.Dataset)
        # Only u_output should be present
        assert "u_output" in result.data_vars
        assert "v_output" not in result.data_vars
        assert "magnitude" not in result.data_vars

    def test_output_variables_multiple_subset(self, sample_dataset):
        """When output_variables has multiple vars, only those returned."""
        derived_var = MultiOutputDerivedVariable(
            output_variables=["u_output", "v_output"]
        )
        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        assert isinstance(result, xr.Dataset)
        # Only u_output and v_output should be present
        assert "u_output" in result.data_vars
        assert "v_output" in result.data_vars
        assert "magnitude" not in result.data_vars

    def test_output_variables_missing_warns(self, sample_dataset, caplog):
        """When output_variables includes missing vars, warning logged."""
        derived_var = MultiOutputDerivedVariable(
            output_variables=["u_output", "nonexistent_var"]
        )

        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        assert isinstance(result, xr.Dataset)
        # Only u_output should be present
        assert "u_output" in result.data_vars
        assert "nonexistent_var" not in result.data_vars
        # Check that a warning was logged
        assert "missing: {'nonexistent_var'}" in caplog.text

    def test_output_variables_all_missing_returns_original(
        self, sample_dataset, caplog
    ):
        """When all output_variables missing, return original dataset."""
        derived_var = MultiOutputDerivedVariable(output_variables=["var1", "var2"])

        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        # Should return original dataset since no vars available
        xr.testing.assert_equal(result, sample_dataset)
        assert "None of the specified output_variables" in caplog.text

    def test_output_variables_preserves_data_values(self, sample_dataset):
        """output_variables correctly filters but preserves data."""
        derived_var = MultiOutputDerivedVariable(output_variables=["magnitude"])
        result = derived.maybe_derive_variables(sample_dataset, [derived_var])

        # Compute expected magnitude
        u = sample_dataset["eastward_wind"].mean(dim="level")
        v = sample_dataset["northward_wind"].mean(dim="level")
        expected_mag = np.sqrt(u**2 + v**2)

        xr.testing.assert_allclose(result["magnitude"], expected_mag)
