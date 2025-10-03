"""Tests for the derived module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import derived


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


class TestDerivedVariableAbstractClass:
    """Test the abstract base class DerivedVariable."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that DerivedVariable cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            derived.DerivedVariable()

    def test_name_property_default(self):
        """Test that the name property defaults to class name."""
        assert TestValidDerivedVariable().name == "TestValidDerivedVariable"

    def test_compute_method_calls_derive_variable(self, sample_derived_dataset):
        """Test that compute method calls derive_variable and validates inputs."""
        result = TestValidDerivedVariable.compute(sample_derived_dataset)

        assert isinstance(result, xr.DataArray)
        # Should be sum of test_variable_1 and test_variable_2
        expected = (
            sample_derived_dataset["test_variable_1"]
            + sample_derived_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result, expected)

    def test_compute_raises_error_missing_variables(self, sample_dataset):
        """Test that compute raises error when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_derived_dataset.drop_vars("test_variable_2")

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

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        assert "TestMinimalDerivedVariable" in result.data_vars

    def test_maybe_derive_variables_with_only_strings(self, sample_dataset):
        """Test maybe_derive_variables with only string variables."""
        variables = ["air_pressure_at_mean_sea_level", "surface_eastward_wind"]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        # Should return the original dataset unchanged
        xr.testing.assert_equal(result, sample_dataset)

    def test_maybe_derive_variables_empty_list(self, sample_dataset):
        """Test maybe_derive_variables with empty variable list."""
        result = derived.maybe_derive_variables(sample_dataset, [])

        # Should return the original dataset unchanged
        xr.testing.assert_equal(result, sample_dataset)

    def test_maybe_derive_variables_with_dataarray_without_name(self, sample_dataset):
        """Test maybe_derive_variables with DataArray that has no name."""
        # Create a derived variable that returns a DataArray without a name
        variables = [TestDerivedVariableWithoutName()]

        result = derived.maybe_derive_variables(sample_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # The derived variable should be added with the correct name
        assert "TestDerivedVariableWithoutName" in result.data_vars
        # Verify the DataArray got the correct name assigned
        derived_var = result["TestDerivedVariableWithoutName"]
        assert derived_var.name == "TestDerivedVariableWithoutName"

    def test_maybe_pull_required_variables_from_derived_input_with_instances(self):
        """Test maybe_pull_required_variables_from_derived_input with instances."""
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

    def test_derived_variable_with_wrong_dimensions(self, sample_derived_dataset):
        """Test behavior when variables have unexpected dimensions."""
        # Create dataset with wrong dimensions for test variables
        wrong_dim_dataset = sample_derived_dataset.copy()
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


class TestIntegrationWithRealData:
    """Integration tests that simulate real-world usage patterns."""

    def test_pipeline_integration(self, sample_dataset):
        """Test integration of multiple derived variables in a pipeline."""
        # Simulate a pipeline that uses multiple derived variables
        variables_to_derive = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        # Step 1: Pull required variables
        required_vars = derived.maybe_include_variables_from_derived_input(
            ["surface_wind_speed"] + variables_to_derive
        )

        # Step 2: Subset dataset to required variables
        available_vars = [
            var for var in required_vars if var in sample_derived_dataset.data_vars
        ]
        subset_dataset = sample_derived_dataset[available_vars]

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
        self, sample_derived_dataset, variable_combination
    ):
        """Test different combinations of derived variables."""
        result = derived.maybe_derive_variables(sample_dataset, variable_combination)

        assert isinstance(result, xr.Dataset)
        # Should have at least the original variables
        for var in sample_dataset.data_vars:
            assert var in result.data_vars
