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


# flake8: noqa: E501


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

    def test_compute_raises_error_missing_variables(self, sample_derived_dataset):
        """Test that compute raises error when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_derived_dataset.drop_vars("test_variable_2")

        # This should fail during derive_variable when trying to access missing variable
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
    """Comprehensive tests for the maybe_derive_variables function."""

    def test_only_string_variables(self, sample_derived_dataset):
        """Test function with only string variables - should return unchanged."""
        variables = ["air_pressure_at_mean_sea_level", "surface_eastward_wind"]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return the exact same dataset when no derived variables present
        xr.testing.assert_equal(result, sample_derived_dataset)
        assert id(result) == id(
            sample_derived_dataset
        )  # Should be same object when no derived vars

    def test_empty_variable_list(self, sample_derived_dataset):
        """Test function with empty variable list."""
        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        variables = []

        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return original dataset unchanged
        xr.testing.assert_equal(result, sample_derived_dataset)

    def test_single_derived_variable_dataarray(self, sample_derived_dataset):
        """Test with single derived variable that returns DataArray."""
        variables = [TestValidDerivedVariable()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        assert isinstance(result, xr.Dataset)  # Function now returns Dataset
        # Verify the derived variable is in the dataset
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct
        expected_value = (
            sample_derived_dataset["test_variable_1"]
            + sample_derived_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_multiple_derived_variables(self, sample_derived_dataset):
        """Test with multiple derived variables - only first is computed."""
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first variable only)
        expected_value = (
            sample_derived_dataset["test_variable_1"]
            + sample_derived_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_mixed_string_and_derived_variables(self, sample_derived_dataset):
        """Test with mix of string and derived variables."""
        variables = [
            "air_pressure_at_mean_sea_level",  # String variable
            TestValidDerivedVariable(),  # Derived variable instance
            "surface_eastward_wind",  # Another string variable
            TestMinimalDerivedVariable(),  # Another derived variable
        ]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars
        # Verify the computed value is correct (from first derived variable only)
        expected_value = (
            sample_derived_dataset["test_variable_1"]
            + sample_derived_dataset["test_variable_2"]
        )
        xr.testing.assert_equal(result["TestValidDerivedVariable"], expected_value)

    def test_dataarray_without_name_gets_assigned_name(self, sample_derived_dataset):
        """Test DataArray without name gets assigned class name with warning."""
        variables = [TestDerivedVariableWithoutName()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        # Logger warnings are not pytest warnings, so just check functionality
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        assert isinstance(result, xr.Dataset)
        # Verify the DataArray got the correct name assigned
        assert "TestDerivedVariableWithoutName" in result.data_vars

    def test_kwargs_passed_to_compute(self, sample_derived_dataset):
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

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(
            sample_derived_dataset, variables, multiplier=test_multiplier
        )

        # Custom compute method keeps original variable name
        assert "test_variable_1" in result.data_vars
        expected = sample_derived_dataset["test_variable_1"] * test_multiplier
        xr.testing.assert_equal(result["test_variable_1"], expected)

    def test_multiple_derived_variables_second_implementation(
        self, sample_derived_dataset
    ):
        """Test with multiple derived variables - second implementation."""
        variables = [TestValidDerivedVariable(), TestMinimalDerivedVariable()]

        # Logger warnings are not pytest warnings, so just check functionality
        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return only the first derived variable as Dataset
        assert isinstance(result, xr.Dataset)
        assert "TestValidDerivedVariable" in result.data_vars

    def test_derived_variable_missing_required_vars(self, sample_derived_dataset):
        """Test derived variable with missing required variables."""

        class TestMissingVarDerived(derived.DerivedVariable):
            required_variables = ["nonexistent_variable"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
                return data[cls.required_variables[0]]

        variables = [TestMissingVarDerived()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        with pytest.raises(KeyError, match="nonexistent_variable"):
            derived.maybe_derive_variables(sample_derived_dataset, variables)

    def test_no_derived_variables_in_list(self, sample_derived_dataset):
        """Test when no derived variables are in the variable list."""
        variables = ["var1", "var2", "var3"]  # All strings

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return original dataset since no derived variables to process
        xr.testing.assert_equal(result, sample_derived_dataset)

    def test_derived_data_dict_handling(self, sample_derived_dataset):
        """Test internal derived_data dictionary logic."""
        # This tests the internal logic of how derived variables are processed
        # We'll test it indirectly through the public interface

        class TestDerivedVariableA(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                return data[cls.required_variables[0]] * 2

        class TestDerivedVariableB(derived.DerivedVariable):
            required_variables = ["test_variable_2"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                return data[cls.required_variables[0]] * 3

        # Test with multiple derived variables
        variables = [TestDerivedVariableA(), TestDerivedVariableB()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"
        result = derived.maybe_derive_variables(sample_derived_dataset, variables)

        # Should return only the first derived variable (current behavior)
        assert isinstance(result, xr.Dataset)
        assert "TestDerivedVariableA" in result.data_vars
        # Verify the computed value is correct
        expected_value = sample_derived_dataset["test_variable_1"] * 2
        xr.testing.assert_equal(result["TestDerivedVariableA"], expected_value)

    def test_error_handling_invalid_derived_variable(self, sample_derived_dataset):
        """Test error handling for invalid derived variable objects."""
        # Test with a non-DerivedVariable object
        invalid_variable = "not_a_derived_variable"

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        with pytest.raises(AttributeError):
            derived.maybe_derive_variables(sample_derived_dataset, [invalid_variable])

    def test_error_handling_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when derived variable requires missing variables."""

        # Create a derived variable that requires a variable not in the dataset
        class TestMissingVarDerived(derived.DerivedVariable):
            required_variables = ["missing_variable"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                return data[cls.required_variables[0]]

        variables = [TestMissingVarDerived()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        with pytest.raises(KeyError, match="missing_variable"):
            derived.maybe_derive_variables(sample_derived_dataset, variables)

    def test_error_handling_derive_variable_returns_none(self, sample_derived_dataset):
        """Test error handling when derive_variable returns None."""

        class TestNoneReturnDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                return None

        variables = [TestNoneReturnDerived()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        with pytest.raises(TypeError, match="Expected DataArray or Dataset"):
            derived.maybe_derive_variables(sample_derived_dataset, variables)

    def test_error_handling_derive_variable_returns_invalid_type(
        self, sample_derived_dataset
    ):
        """Test error handling when derive_variable returns invalid type."""

        class TestInvalidReturnDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> str:
                return "invalid_return_type"

        variables = [TestInvalidReturnDerived()]

        sample_derived_dataset.attrs["dataset_type"] = "forecast"

        with pytest.raises(TypeError, match="Expected DataArray or Dataset"):
            derived.maybe_derive_variables(sample_derived_dataset, variables)

    def test_performance_with_large_datasets(self, sample_derived_dataset):
        """Test performance with larger datasets."""
        # Create a larger dataset for performance testing
        large_dataset = sample_derived_dataset.copy()
        # Expand dimensions to make it larger
        large_dataset = large_dataset.expand_dims({"batch": 10, "ensemble": 5})

        class TestPerformanceDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                # Simple operation that should be fast
                return data[cls.required_variables[0]] * 1.5

        variables = [TestPerformanceDerived()]

        large_dataset.attrs["dataset_type"] = "forecast"

        # This should complete without performance issues
        result = derived.maybe_derive_variables(large_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "TestPerformanceDerived" in result.data_vars

    def test_edge_case_empty_dataset(self):
        """Test edge case with completely empty dataset."""
        # Create an empty dataset
        empty_dataset = xr.Dataset()

        class TestEmptyDatasetDerived(derived.DerivedVariable):
            required_variables = []

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                # Return a simple DataArray
                return xr.DataArray(
                    np.ones((5, 5)),
                    dims=["x", "y"],
                    coords={"x": range(5), "y": range(5)},
                )

        variables = [TestEmptyDatasetDerived()]

        empty_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(empty_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "TestEmptyDatasetDerived" in result.data_vars

    def test_edge_case_single_point_dataset(self):
        """Test edge case with single point dataset."""
        # Create a single point dataset
        single_point_dataset = xr.Dataset(
            {
                "test_variable_1": xr.DataArray(
                    [42.0],
                    dims=["time"],
                    coords={"time": [pd.Timestamp("2021-01-01")]},
                )
            }
        )

        class TestSinglePointDerived(derived.DerivedVariable):
            required_variables = ["test_variable_1"]

            @classmethod
            def derive_variable(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
                return data[cls.required_variables[0]] * 2

        variables = [TestSinglePointDerived()]

        single_point_dataset.attrs["dataset_type"] = "forecast"

        result = derived.maybe_derive_variables(single_point_dataset, variables)

        assert isinstance(result, xr.Dataset)
        assert "TestSinglePointDerived" in result.data_vars
        expected_value = single_point_dataset["test_variable_1"] * 2
        xr.testing.assert_equal(result["TestSinglePointDerived"], expected_value)


class TestAtmosphericRiverMask:
    """Test the AtmosphericRiverMask derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.AtmosphericRiverMask.required_variables == [
            "eastward_integrated_vapor_transport",
            "northward_integrated_vapor_transport",
        ]

    def test_derive_variable_returns_dataset(self, sample_derived_dataset):
        """Test that derive_variable returns a Dataset."""
        result = derived.AtmosphericRiverMask.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.Dataset)
        assert "atmospheric_river_mask" in result.data_vars

    def test_atmospheric_river_mask_values(self, sample_derived_dataset):
        """Test that atmospheric river mask has correct values."""
        result = derived.AtmosphericRiverMask.derive_variable(sample_derived_dataset)

        # Check that mask values are binary (0 or 1)
        mask_values = result["atmospheric_river_mask"].values
        assert np.all(np.isin(mask_values, [0, 1]))

        # Check that mask has the same dimensions as input
        expected_dims = sample_derived_dataset[
            "eastward_integrated_vapor_transport"
        ].dims
        assert result["atmospheric_river_mask"].dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.AtmosphericRiverMask.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "atmospheric_river_mask"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars(
            "northward_integrated_vapor_transport"
        )

        with pytest.raises(KeyError):
            derived.AtmosphericRiverMask.derive_variable(incomplete_dataset)


class TestCravenBrooksSignificantSevere:
    """Test the CravenBrooksSignificantSevere derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.CravenBrooksSignificantSevere.required_variables == [
            "convective_available_potential_energy",
            "convective_inhibition",
            "lifted_index",
            "surface_based_lifted_index",
            "surface_based_cape",
            "surface_based_cin",
            "surface_based_lcl",
            "surface_based_lfc",
            "surface_based_el",
            "significant_severe_parameter",
            "low_level_wind_shear",
            "surface_air_temperature",
            "surface_dew_point_temperature",
            "surface_air_pressure",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

    def test_optional_variables(self):
        """Test that optional variables are correctly specified."""
        assert derived.CravenBrooksSignificantSevere.optional_variables == [
            "surface_air_pressure",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

    def test_derive_variable_returns_dataset(self, sample_derived_dataset):
        """Test that derive_variable returns a Dataset."""
        result = derived.CravenBrooksSignificantSevere.derive_variable(
            sample_derived_dataset
        )
        assert isinstance(result, xr.Dataset)
        assert "craven_brooks_significant_severe" in result.data_vars

    def test_craven_brooks_values(self, sample_derived_dataset):
        """Test that Craven-Brooks index has correct values."""
        result = derived.CravenBrooksSignificantSevere.derive_variable(
            sample_derived_dataset
        )

        # Check that index values are reasonable
        index_values = result["craven_brooks_significant_severe"].values
        assert not np.any(np.isnan(index_values))

        # Check that index has the same dimensions as input
        expected_dims = sample_derived_dataset[
            "convective_available_potential_energy"
        ].dims
        assert result["craven_brooks_significant_severe"].dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.CravenBrooksSignificantSevere.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "craven_brooks_significant_severe"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars(
            "convective_available_potential_energy"
        )

        with pytest.raises(KeyError):
            derived.CravenBrooksSignificantSevere.derive_variable(incomplete_dataset)


class TestWindSpeed:
    """Test the WindSpeed derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.WindSpeed.required_variables == [
            "eastward_wind",
            "northward_wind",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.WindSpeed.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "wind_speed"

    def test_wind_speed_calculation(self, sample_derived_dataset):
        """Test that wind speed is calculated correctly."""
        result = derived.WindSpeed.derive_variable(sample_derived_dataset)

        # Check that wind speed values are non-negative
        wind_speed_values = result.values
        assert np.all(wind_speed_values >= 0)

        # Check that wind speed has the same dimensions as input
        expected_dims = sample_derived_dataset["eastward_wind"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.WindSpeed.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "wind_speed"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("northward_wind")

        with pytest.raises(KeyError):
            derived.WindSpeed.derive_variable(incomplete_dataset)


class TestGreatCircleMask:
    """Test the GreatCircleMask derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.GreatCircleMask.required_variables == [
            "latitude",
            "longitude",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.GreatCircleMask.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "great_circle_mask"

    def test_great_circle_mask_values(self, sample_derived_dataset):
        """Test that great circle mask has correct values."""
        result = derived.GreatCircleMask.derive_variable(sample_derived_dataset)

        # Check that mask values are binary (0 or 1)
        mask_values = result.values
        assert np.all(np.isin(mask_values, [0, 1]))

        # Check that mask has the same dimensions as input
        expected_dims = sample_derived_dataset["latitude"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.GreatCircleMask.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "great_circle_mask"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("longitude")

        with pytest.raises(KeyError):
            derived.GreatCircleMask.derive_variable(incomplete_dataset)


class TestOrography:
    """Test the Orography derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.Orography.required_variables == [
            "orography",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.Orography.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "orography"

    def test_orography_values(self, sample_derived_dataset):
        """Test that orography values are preserved."""
        result = derived.Orography.derive_variable(sample_derived_dataset)

        # Check that orography values are unchanged
        original_values = sample_derived_dataset["orography"].values
        result_values = result.values
        np.testing.assert_array_equal(original_values, result_values)

        # Check that orography has the same dimensions as input
        expected_dims = sample_derived_dataset["orography"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.Orography.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "orography"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("orography")

        with pytest.raises(KeyError):
            derived.Orography.derive_variable(incomplete_dataset)


class TestSurfacePressure:
    """Test the SurfacePressure derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.SurfacePressure.required_variables == [
            "air_pressure_at_mean_sea_level",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.SurfacePressure.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "surface_pressure"

    def test_surface_pressure_values(self, sample_derived_dataset):
        """Test that surface pressure values are preserved."""
        result = derived.SurfacePressure.derive_variable(sample_derived_dataset)

        # Check that surface pressure values are unchanged
        original_values = sample_derived_dataset[
            "air_pressure_at_mean_sea_level"
        ].values
        result_values = result.values
        np.testing.assert_array_equal(original_values, result_values)

        # Check that surface pressure has the same dimensions as input
        expected_dims = sample_derived_dataset["air_pressure_at_mean_sea_level"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.SurfacePressure.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "surface_pressure"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars(
            "air_pressure_at_mean_sea_level"
        )

        with pytest.raises(KeyError):
            derived.SurfacePressure.derive_variable(incomplete_dataset)


class TestGeopotentialThickness:
    """Test the GeopotentialThickness derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.GeopotentialThickness.required_variables == [
            "geopotential",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.GeopotentialThickness.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "geopotential_thickness"

    def test_geopotential_thickness_values(self, sample_derived_dataset):
        """Test that geopotential thickness values are preserved."""
        result = derived.GeopotentialThickness.derive_variable(sample_derived_dataset)

        # Check that geopotential thickness values are unchanged
        original_values = sample_derived_dataset["geopotential"].values
        result_values = result.values
        np.testing.assert_array_equal(original_values, result_values)

        # Check that geopotential thickness has the same dimensions as input
        expected_dims = sample_derived_dataset["geopotential"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.GeopotentialThickness.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "geopotential_thickness"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("geopotential")

        with pytest.raises(KeyError):
            derived.GeopotentialThickness.derive_variable(incomplete_dataset)


class TestTropicalCycloneWindSpeed:
    """Test the TropicalCycloneWindSpeed derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.TropicalCycloneWindSpeed.required_variables == [
            "eastward_wind",
            "northward_wind",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.TropicalCycloneWindSpeed.derive_variable(
            sample_derived_dataset
        )
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropical_cyclone_wind_speed"

    def test_tropical_cyclone_wind_speed_values(self, sample_derived_dataset):
        """Test that tropical cyclone wind speed values are correct."""
        result = derived.TropicalCycloneWindSpeed.derive_variable(
            sample_derived_dataset
        )

        # Check that wind speed values are non-negative
        wind_speed_values = result.values
        assert np.all(wind_speed_values >= 0)

        # Check that wind speed has the same dimensions as input
        expected_dims = sample_derived_dataset["eastward_wind"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.TropicalCycloneWindSpeed.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropical_cyclone_wind_speed"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("northward_wind")

        with pytest.raises(KeyError):
            derived.TropicalCycloneWindSpeed.derive_variable(incomplete_dataset)


class TestTropicalCyclonePressure:
    """Test the TropicalCyclonePressure derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.TropicalCyclonePressure.required_variables == [
            "air_pressure_at_mean_sea_level",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.TropicalCyclonePressure.derive_variable(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropical_cyclone_pressure"

    def test_tropical_cyclone_pressure_values(self, sample_derived_dataset):
        """Test that tropical cyclone pressure values are preserved."""
        result = derived.TropicalCyclonePressure.derive_variable(sample_derived_dataset)

        # Check that pressure values are unchanged
        original_values = sample_derived_dataset[
            "air_pressure_at_mean_sea_level"
        ].values
        result_values = result.values
        np.testing.assert_array_equal(original_values, result_values)

        # Check that pressure has the same dimensions as input
        expected_dims = sample_derived_dataset["air_pressure_at_mean_sea_level"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.TropicalCyclonePressure.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropical_cyclone_pressure"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars(
            "air_pressure_at_mean_sea_level"
        )

        with pytest.raises(KeyError):
            derived.TropicalCyclonePressure.derive_variable(incomplete_dataset)


class TestTropicalCycloneGeopotential:
    """Test the TropicalCycloneGeopotential derived variable."""

    def test_required_variables(self):
        """Test that required variables are correctly specified."""
        assert derived.TropicalCycloneGeopotential.required_variables == [
            "geopotential",
        ]

    def test_derive_variable_returns_dataarray(self, sample_derived_dataset):
        """Test that derive_variable returns a DataArray."""
        result = derived.TropicalCycloneGeopotential.derive_variable(
            sample_derived_dataset
        )
        assert isinstance(result, xr.DataArray)

        # Check that geopotential values are unchanged
        original_values = sample_derived_dataset["geopotential"].values
        result_values = result.values
        np.testing.assert_array_equal(original_values, result_values)

        # Check that geopotential has the same dimensions as input
        expected_dims = sample_derived_dataset["geopotential"].dims
        assert result.dims == expected_dims

    def test_compute_method_integration(self, sample_derived_dataset):
        """Test the compute method integration."""
        result = derived.TropicalCycloneGeopotential.compute(sample_derived_dataset)
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropical_cyclone_geopotential"

    def test_missing_required_variables(self, sample_derived_dataset):
        """Test error handling when required variables are missing."""
        incomplete_dataset = sample_derived_dataset.drop_vars("geopotential")

        with pytest.raises(KeyError):
            derived.TropicalCycloneGeopotential.derive_variable(incomplete_dataset)
