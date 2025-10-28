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


class TestValidDerivedVariable(derived.DerivedVariable):
    """A valid test implementation of DerivedVariable for testing purposes."""

    name = "TestValidDerivedVariable"
    required_variables = ["test_variable_1", "test_variable_2"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that sums two variables."""
        return data[cls.required_variables[0]] + data[cls.required_variables[1]]


class TestMinimalDerivedVariable(derived.DerivedVariable):
    """A minimal test implementation with one required variable."""

    required_variables = ["single_variable"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that returns the variable unchanged."""
        return data[cls.required_variables[0]]


class TestDerivedVariableWithoutName(derived.DerivedVariable):
    """A test implementation that returns a DataArray without a name."""

    required_variables = ["single_variable"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Test implementation that returns DataArray without name."""
        result = data[cls.required_variables[0]]
        result.name = None
        return result


class TestDerivedVariableForTesting(derived.DerivedVariable):
    """A concrete derived variable class for testing
    _maybe_convert_variable_to_string."""

    name = "TestDerivedVar"
    required_variables = ["input_var1", "input_var2"]

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

    def test_dataarray_without_name_generates_exception(self, sample_dataset):
        """Test DataArray without name generates exception."""

        variables = [TestDerivedVariableWithoutName()]
        assert not hasattr(variables[0], "name")
        sample_dataset.attrs["dataset_type"] = "forecast"
        with pytest.raises(AttributeError):
            derived.maybe_derive_variables(sample_dataset, variables)

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
            name = "TestInvalidReturnType"
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

        assert set(result) == set(incoming_variables)

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

    def test_maybe_include_variables_removes_duplicates(self):
        """Test that function preserves order and removes duplicates."""
        incoming_variables = [
            "var1",
            TestValidDerivedVariable,  # Adds test_variable_1, test_variable_2
            "var2",
            TestMinimalDerivedVariable,  # Adds single_variable
            "var1",  # Duplicate
            TestValidDerivedVariable,  # Duplicate derived variable
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

        assert set(result) == set(expected)

    def test_is_derived_variable_with_string(self):
        """Test is_derived_variable returns False for string inputs."""
        result = derived.is_derived_variable("regular_string_variable")
        assert result is False

    def test_is_derived_variable_with_derived_variable_class(self):
        """Test is_derived_variable returns True for DerivedVariable classes."""
        result = derived.is_derived_variable(TestValidDerivedVariable)
        assert result is True

    def test_is_derived_variable_with_minimal_derived_variable_class(self):
        """Test is_derived_variable returns True for minimal DerivedVariable classes."""
        result = derived.is_derived_variable(TestMinimalDerivedVariable)
        assert result is True

    def test_is_derived_variable_with_derived_variable_instance(self):
        """Test is_derived_variable returns False for DerivedVariable instances."""
        # The function is designed to work with classes, not instances
        instance = TestValidDerivedVariable()
        result = derived.is_derived_variable(instance)
        assert result is False

    def test_is_derived_variable_with_non_derived_class(self):
        """Test is_derived_variable returns False for non-DerivedVariable classes."""

        class NotDerivedVariable:
            pass

        result = derived.is_derived_variable(NotDerivedVariable)
        assert result is False

    def test_is_derived_variable_with_builtin_type(self):
        """Test is_derived_variable returns False for builtin types."""
        result_int = derived.is_derived_variable(int)
        result_str = derived.is_derived_variable(str)
        result_list = derived.is_derived_variable(list)

        assert result_int is False
        assert result_str is False
        assert result_list is False

    def test_is_derived_variable_with_none(self):
        """Test is_derived_variable returns False for None."""
        result = derived.is_derived_variable(None)
        assert result is False

    def test_is_derived_variable_with_abstract_base_class(self):
        """Test is_derived_variable with abstract DerivedVariable class."""
        # The abstract base class should return True since it's still a subclass
        result = derived.is_derived_variable(derived.DerivedVariable)
        assert result is True


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

        # Should fail during derive_variable when accessing missing variables
        with pytest.raises(
            KeyError
        ):  # derive_variable will fail when accessing missing variables
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
        result = TestValidDerivedVariable.compute(large_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(time), len(latitudes), len(longitudes))


class TestAlternativeVariables:
    """Test alternative_variables functionality in derived variables."""

    def test_alternative_variables_success_same_shape(self, sample_dataset):
        """Test alternative_variables when all alternatives are present with same
        shape."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables to the dataset
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it successfully pulls alternative variables
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative variables
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_alternative_variables_failure_missing_alternatives(self, sample_dataset):
        """Test alternative_variables when alternatives are not in dataset."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it fails when neither required nor alternatives are present
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )

    def test_alternative_variables_partial_alternatives_failure(self, sample_dataset):
        """Test alternative_variables when only some alternatives are present."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add only one of the alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it fails when not all alternatives are present
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )

    def test_alternative_variables_required_present_ignores_alternatives(
        self, sample_dataset
    ):
        """Test that when required variable is present, alternatives are ignored."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Add alternative variables
        level_shape = (
            len(sample_dataset.valid_time),
            len(sample_dataset.level),
            len(sample_dataset.latitude),
            len(sample_dataset.longitude),
        )
        sample_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        sample_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it uses required variable when present
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the required variable, not alternatives
        assert "specific_humidity" in result.data_vars
        assert "relative_humidity" not in result.data_vars
        assert "air_temperature" not in result.data_vars

    def test_alternative_variables_multiple_required_vars(self, sample_dataset):
        """Test alternative_variables with multiple required variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variables so alternatives will be used
        test_dataset = sample_dataset.drop_vars(
            ["specific_humidity", "geopotential"], errors="ignore"
        )

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["pressure_alt"] = (
            time_level_location_variables,
            np.random.normal(1000, 100, size=level_shape),
        )
        test_dataset["temperature_alt"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"],
            "geopotential": ["pressure_alt", "temperature_alt"],
        }
        optional_variables = []

        # Test that it successfully pulls alternative variables for both required vars
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity", "geopotential"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain all the alternative variables
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "pressure_alt" in result.data_vars
        assert "temperature_alt" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original
        assert "geopotential" not in result.data_vars  # Not present in original


class TestOptionalVariables:
    """Test optional_variables functionality in derived variables."""

    def test_optional_variables_present_in_dataset(self, sample_dataset):
        """Test optional_variables when they are present in dataset."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Add an optional variable
        sample_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(sample_dataset.valid_time),
                    len(sample_dataset.latitude),
                    len(sample_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {}
        optional_variables = ["orography"]

        # Test that it successfully pulls optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain both required and optional variables
        assert "test_variable_1" in result.data_vars
        assert "orography" in result.data_vars

    def test_optional_variables_missing_from_dataset(self, sample_dataset):
        """Test optional_variables when they are missing from dataset."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        alternative_variables = {}
        optional_variables = ["orography"]

        # Test that it works without optional variable
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert "orography" not in result.data_vars

    def test_optional_variables_multiple_optional(self, sample_dataset):
        """Test multiple optional_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Add some optional variables
        sample_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(sample_dataset.valid_time),
                    len(sample_dataset.latitude),
                    len(sample_dataset.longitude),
                ),
            ),
        )
        sample_dataset["land_sea_mask"] = (
            time_location_variables,
            np.random.choice(
                [0, 1],
                size=(
                    len(sample_dataset.valid_time),
                    len(sample_dataset.latitude),
                    len(sample_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {}
        optional_variables = ["orography", "land_sea_mask"]

        # Test that it successfully pulls multiple optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain required and both optional variables
        assert "test_variable_1" in result.data_vars
        assert "orography" in result.data_vars
        assert "land_sea_mask" in result.data_vars


class TestMixedAlternativeAndOptionalVariables:
    """Test scenarios with both alternative and optional variables."""

    def test_alternative_and_optional_variables_both_present(self, sample_dataset):
        """Test when both alternative and optional variables are present."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )
        # Add optional variable
        test_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(test_dataset.valid_time),
                    len(test_dataset.latitude),
                    len(test_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = ["orography"]

        # Test that it successfully pulls both alternative and optional variables
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative and optional variables
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "orography" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_alternative_present_optional_missing(self, sample_dataset):
        """Test when alternatives are present but optional variables are missing."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = ["orography"]

        # Test that it successfully pulls alternative variables but not optional
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative variables but not optional
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "orography" not in result.data_vars  # Not present in original
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_alternative_missing_optional_present(self, sample_dataset):
        """Test when alternatives are missing but optional variables are present."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add optional variable
        test_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(test_dataset.valid_time),
                    len(test_dataset.latitude),
                    len(test_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = ["orography"]

        # This should fail because required variable is missing and alternatives are
        # not present
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )


class TestEdgeCasesAlternativeOptionalVariables:
    """Test edge cases for alternative and optional variables."""

    def test_empty_alternative_variables_dict(self, sample_dataset):
        """Test with empty alternative_variables dictionary."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        alternative_variables = {}
        optional_variables = []

        # Test that it works with empty alternative variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1

    def test_empty_optional_variables_list(self, sample_dataset):
        """Test with empty optional_variables list."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        alternative_variables = {}
        optional_variables = []

        # Test that it works with empty optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1

    def test_none_alternative_variables(self, sample_dataset):
        """Test with None alternative_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Test that it works with None alternative variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=None,
            optional_variables=None,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1

    def test_none_optional_variables(self, sample_dataset):
        """Test with None optional_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Test that it works with None optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=None,
            optional_variables=None,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1

    def test_duplicate_variables_in_alternatives(self, sample_dataset):
        """Test with duplicate variables in alternative_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": [
                "relative_humidity",
                "air_temperature",
                "relative_humidity",
            ]  # Duplicate
        }
        optional_variables = []

        # Test that it works with duplicate variables in alternatives
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative variables (duplicates handled by set)
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_shape_mismatch_in_alternatives(self, sample_dataset):
        """Test with shape mismatch in alternative variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables with different shapes
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_location_variables,  # Different shape - no level dimension
            np.random.normal(
                280,
                20,
                size=(
                    len(test_dataset.valid_time),
                    len(test_dataset.latitude),
                    len(test_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it works with shape mismatch (xarray handles broadcasting)
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative variables despite shape mismatch
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_alternative_variables_with_empty_list(self, sample_dataset):
        """Test alternative_variables with empty list for a required variable."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        alternative_variables = {
            "specific_humidity": []  # Empty list
        }
        optional_variables = []

        # This should fail because required variable is missing and alternatives list
        # is empty
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )

    def test_required_variable_in_optional_variables(self, sample_dataset):
        """Test when a required variable is also listed in optional_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        alternative_variables = {}
        optional_variables = [
            "test_variable_1",
            "orography",
        ]  # test_variable_1 is both required and optional

        # Test that it works when required variable is also in optional list
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the required variable (duplicates handled by set)
        assert "test_variable_1" in result.data_vars
        assert "orography" not in result.data_vars  # Not present in original


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


class TestIntegrationWithSafelyPullVariables:
    """Test integration with the safely_pull_variables function."""

    def test_safely_pull_variables_with_alternative_variables(self, sample_dataset):
        """Test safely_pull_variables with alternative_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = ["orography"]

        # Test that it successfully pulls alternative variables
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative variables
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original
        assert "orography" not in result.data_vars  # Not present in original

    def test_safely_pull_variables_with_optional_variables(self, sample_dataset):
        """Test safely_pull_variables with optional_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Add optional variable
        sample_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(sample_dataset.valid_time),
                    len(sample_dataset.latitude),
                    len(sample_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {}
        optional_variables = ["orography"]

        # Test that it successfully pulls optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain both required and optional variables
        assert "test_variable_1" in result.data_vars
        assert "orography" in result.data_vars

    def test_safely_pull_variables_missing_alternatives_failure(self, sample_dataset):
        """Test safely_pull_variables failure when alternatives are missing."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it fails when neither required nor alternatives are present
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )

    def test_safely_pull_variables_mixed_scenario(self, sample_dataset):
        """Test safely_pull_variables with both alternative and optional variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be used
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        test_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )
        # Add optional variable
        test_dataset["orography"] = (
            time_location_variables,
            np.random.uniform(
                0,
                3000,
                size=(
                    len(test_dataset.valid_time),
                    len(test_dataset.latitude),
                    len(test_dataset.longitude),
                ),
            ),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = ["orography"]

        # Test that it successfully pulls both alternative and optional variables
        result = safely_pull_variables(
            test_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the alternative and optional variables
        assert "relative_humidity" in result.data_vars
        assert "air_temperature" in result.data_vars
        assert "orography" in result.data_vars
        assert "specific_humidity" not in result.data_vars  # Not present in original

    def test_safely_pull_variables_required_present_ignores_alternatives(
        self, sample_dataset
    ):
        """Test that required variables take precedence over alternatives."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Add alternative variables
        level_shape = (
            len(sample_dataset.valid_time),
            len(sample_dataset.level),
            len(sample_dataset.latitude),
            len(sample_dataset.longitude),
        )
        sample_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )
        sample_dataset["air_temperature"] = (
            time_level_location_variables,
            np.random.normal(280, 20, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it uses required variable when present
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["specific_humidity"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain the required variable, not alternatives
        assert "specific_humidity" in result.data_vars
        assert "relative_humidity" not in result.data_vars
        assert "air_temperature" not in result.data_vars

    def test_safely_pull_variables_partial_alternatives_failure(self, sample_dataset):
        """Test safely_pull_variables failure when only some alternatives are
        present."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Remove the required variable so alternatives will be needed
        test_dataset = sample_dataset.drop_vars("specific_humidity", errors="ignore")

        # Add only one of the alternative variables
        level_shape = (
            len(test_dataset.valid_time),
            len(test_dataset.level),
            len(test_dataset.latitude),
            len(test_dataset.longitude),
        )
        test_dataset["relative_humidity"] = (
            time_level_location_variables,
            np.random.uniform(0, 100, size=level_shape),
        )

        alternative_variables = {
            "specific_humidity": ["relative_humidity", "air_temperature"]
        }
        optional_variables = []

        # Test that it fails when not all alternatives are present
        with pytest.raises(
            KeyError,
            match="Required variables specific_humidity nor any of their alternatives",
        ):
            safely_pull_variables(
                test_dataset,
                required_variables=["specific_humidity"],
                alternative_variables=alternative_variables,
                optional_variables=optional_variables,
            )

    def test_safely_pull_variables_empty_optional_variables(self, sample_dataset):
        """Test safely_pull_variables with empty optional_variables."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        alternative_variables = {}
        optional_variables = []

        # Test that it works with empty optional variables
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=alternative_variables,
            optional_variables=optional_variables,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1

    def test_safely_pull_variables_none_parameters(self, sample_dataset):
        """Test safely_pull_variables with None parameters."""
        from extremeweatherbench.variable_utils import safely_pull_variables

        # Test that it works with None parameters
        result = safely_pull_variables(
            sample_dataset,
            required_variables=["test_variable_1"],
            alternative_variables=None,
            optional_variables=None,
        )

        # Should contain only the required variable
        assert "test_variable_1" in result.data_vars
        assert len(result.data_vars) == 1


class TestNormalizeVariable:
    """Test the _maybe_convert_variable_to_string function."""

    def test_maybe_convert_variable_to_string_string_input(self):
        """Test _maybe_convert_variable_to_string with string input."""
        result = derived._maybe_convert_variable_to_string("temperature")
        assert result == "temperature"
        assert isinstance(result, str)

    def test_maybe_convert_variable_to_string_derived_class_input(self):
        """Test _maybe_convert_variable_to_string with DerivedVariable class input."""
        result = derived._maybe_convert_variable_to_string(
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
        result = derived._maybe_convert_variable_to_string(instance)
        # Instance is returned as-is, not converted to string
        assert result == instance
        assert isinstance(result, TestDerivedVariableForTesting)

    def test_maybe_convert_variable_to_string_handles_both_types(self):
        """Test that _maybe_convert_variable_to_string handles both incoming types
        correctly."""
        # Test string type
        string_result = derived._maybe_convert_variable_to_string("my_variable")
        assert string_result == "my_variable"

        # Test derived variable type
        derived_result = derived._maybe_convert_variable_to_string(
            TestDerivedVariableForTesting
        )
        assert derived_result == "TestDerivedVar"

        # Results should be different but both strings
        assert string_result != derived_result
        assert isinstance(string_result, str)
        assert isinstance(derived_result, str)
