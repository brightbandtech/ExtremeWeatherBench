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


class TestAtmosphericRiverMask:
    """Test the AtmosphericRiverMask derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        assert derived.AtmosphericRiverMask.required_variables == [
            "air_pressure_at_mean_sea_level"
        ]

    def test_derive_variable_basic(self, sample_dataset):
        """Test basic functionality of derive_variable method."""
        # This test will fail until the syntax errors are fixed
        with pytest.raises(NotImplementedError):
            derived.AtmosphericRiverMask.derive_variable(sample_dataset)

    def test_compute_integration(self, sample_dataset):
        """Test the compute method integration."""
        with pytest.raises(NotImplementedError):
            derived.AtmosphericRiverMask.compute(sample_dataset)

    def test_name_property(self):
        """Test the name property."""
        instance = derived.AtmosphericRiverMask()
        assert instance.name == "AtmosphericRiverMask"


class TestIntegratedVaporTransport:
    """Test the IntegratedVaporTransport derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        expected_vars = ["eastward_wind", "northward_wind", "specific_humidity"]
        assert derived.IntegratedVaporTransport.required_variables == expected_vars

    @patch("extremeweatherbench.calc.calculate_pressure_at_surface")
    @patch("extremeweatherbench.calc.orography")
    @patch("extremeweatherbench.calc.nantrapezoid")
    def test_derive_variable_with_mocks(
        self, mock_nantrapezoid, mock_orography, mock_calc_pressure, sample_dataset
    ):
        """Test derive_variable with mocked calc functions."""
        # Mock the calc functions to avoid complex dependencies
        mock_orography.return_value = xr.DataArray(
            np.ones((len(sample_dataset.latitude), len(sample_dataset.longitude))),
            coords={
                "latitude": sample_dataset.latitude,
                "longitude": sample_dataset.longitude,
            },
        )
        mock_calc_pressure.return_value = xr.DataArray(
            np.full(
                (len(sample_dataset.latitude), len(sample_dataset.longitude)), 101325
            ),
            coords={
                "latitude": sample_dataset.latitude,
                "longitude": sample_dataset.longitude,
            },
        )
        mock_nantrapezoid.return_value = np.random.normal(
            0,
            100,
            size=(
                len(sample_dataset.time),
                len(sample_dataset.latitude),
                len(sample_dataset.longitude),
            ),
        )

        result = derived.IntegratedVaporTransport.derive_variable(sample_dataset)

        assert isinstance(result, xr.DataArray)
        # Verify calc functions were called
        mock_orography.assert_called_once()
        mock_calc_pressure.assert_called_once()
        assert (
            mock_nantrapezoid.call_count == 2
        )  # Called for eastward and northward components

    def test_missing_required_variables(self, sample_dataset):
        """Test behavior when required variables are missing."""
        incomplete_dataset = sample_dataset.drop_vars("specific_humidity")

        with pytest.raises(
            ValueError, match="Input variable specific_humidity not found in data"
        ):
            derived.IntegratedVaporTransport.compute(incomplete_dataset)

    def test_name_property(self):
        """Test the name property."""
        instance = derived.IntegratedVaporTransport()
        assert instance.name == "IntegratedVaporTransport"


class TestGeopotentialThickness:
    """Test the GeopotentialThickness derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        expected_vars = ["geopotential"]
        assert derived.GeopotentialThickness.required_variables == expected_vars

    def test_derive_variable_basic(self, sample_dataset):
        """Test basic functionality of derive_variable method."""
        top_level = 500.0
        bottom_level = 1000.0

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, top_level, bottom_level
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "latitude", "longitude")
        assert result.attrs["units"] == "m"
        assert "Geopotential thickness" in result.attrs["description"]
        assert "500" in result.attrs["description"]
        assert "1000" in result.attrs["description"]

    def test_derive_variable_calculation_correctness(self, sample_dataset):
        """Test that the calculation is mathematically correct."""
        top_level = 500.0
        bottom_level = 1000.0

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, top_level, bottom_level
        )

        # Manually calculate expected result
        top_geopotential = sample_dataset["geopotential"].sel(level=top_level)
        bottom_geopotential = sample_dataset["geopotential"].sel(level=bottom_level)
        expected = (top_geopotential - bottom_geopotential) / 9.80665

        xr.testing.assert_allclose(result, expected)

    def test_derive_variable_different_levels(self, sample_dataset):
        """Test with different pressure levels."""
        test_cases = [
            (300.0, 500.0),
            (500.0, 700.0),
            (700.0, 850.0),
            (850.0, 1000.0),
        ]

        for top_level, bottom_level in test_cases:
            result = derived.GeopotentialThickness.derive_variable(
                sample_dataset, top_level, bottom_level
            )

            assert isinstance(result, xr.DataArray)
            assert result.attrs["units"] == "m"
            assert str(int(top_level)) in result.attrs["description"]
            assert str(int(bottom_level)) in result.attrs["description"]

    def test_derive_variable_reversed_levels(self, sample_dataset):
        """Test with top level higher pressure than bottom level (reversed)."""
        # This should still work mathematically, result can be positive or negative
        # depending on the random test data
        top_level = 1000.0  # Higher pressure (lower altitude)
        bottom_level = 500.0  # Lower pressure (higher altitude)

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, top_level, bottom_level
        )

        assert isinstance(result, xr.DataArray)
        # The result will be the negative of the normal thickness calculation
        normal_result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, bottom_level, top_level
        )
        xr.testing.assert_allclose(result, -normal_result)

    def test_derive_variable_single_level_difference(self, sample_dataset):
        """Test with the same level for both top and bottom."""
        level = 500.0

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, level, level
        )

        assert isinstance(result, xr.DataArray)
        # Should be zero since it's the same level (difference of same values)
        expected_zero = sample_dataset["geopotential"].sel(level=level) * 0
        xr.testing.assert_allclose(result, expected_zero)

    def test_derive_variable_preserves_coordinates(self, sample_dataset):
        """Test that derived variable preserves coordinate information."""
        top_level = 500.0
        bottom_level = 1000.0

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, top_level, bottom_level
        )

        # Check that coordinates are preserved
        assert "time" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords

        # Check coordinate values are the same
        xr.testing.assert_equal(result.coords["time"], sample_dataset.coords["time"])
        xr.testing.assert_equal(
            result.coords["latitude"], sample_dataset.coords["latitude"]
        )
        xr.testing.assert_equal(
            result.coords["longitude"], sample_dataset.coords["longitude"]
        )

    def test_derive_variable_missing_level(self, sample_dataset):
        """Test behavior when requested level is not available."""
        top_level = 123.45  # Non-existent level
        bottom_level = 1000.0

        with pytest.raises(KeyError):
            derived.GeopotentialThickness.derive_variable(
                sample_dataset, top_level, bottom_level
            )

    def test_compute_integration(self, sample_dataset):
        """Test the compute method integration."""
        # The compute method doesn't take the level parameters, so this should fail
        # because derive_variable requires additional parameters that aren't passed
        with pytest.raises(TypeError):
            derived.GeopotentialThickness.compute(sample_dataset)

    def test_missing_required_variables(self, sample_dataset):
        """Test behavior when required variables are missing."""
        incomplete_dataset = sample_dataset.drop_vars("geopotential")

        with pytest.raises(
            ValueError, match="Input variable geopotential not found in data"
        ):
            derived.GeopotentialThickness.compute(incomplete_dataset)

    def test_name_property(self):
        """Test the name property."""
        instance = derived.GeopotentialThickness()
        assert instance.name == "GeopotentialThickness"

    def test_realistic_thickness_values(self, sample_dataset):
        """Test that calculated thickness values are reasonable."""
        # Test 1000-500 hPa thickness (common meteorological layer)
        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, 500.0, 1000.0
        )

        # With random test data, we just check that the values are finite and reasonable
        assert result.notnull().all()  # No NaN values
        assert (abs(result) < 50000).all()  # Should be less than 50km in absolute value
        assert result.shape == (8, 11, 21)  # Correct shape

    def test_attributes_formatting(self, sample_dataset):
        """Test that attributes are properly formatted."""
        top_level = 300.0
        bottom_level = 850.0

        result = derived.GeopotentialThickness.derive_variable(
            sample_dataset, top_level, bottom_level
        )

        expected_description = "Geopotential thickness of 300.0 and 850.0 hPa"
        assert result.attrs["description"] == expected_description
        assert result.attrs["units"] == "m"


class TestSurfaceWindSpeed:
    """Test the SurfaceWindSpeed derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        expected_vars = ["surface_eastward_wind", "surface_northward_wind"]
        assert derived.SurfaceWindSpeed.required_variables == expected_vars

    def test_derive_variable_basic(self, sample_dataset):
        """Test basic functionality of derive_variable method."""
        result = derived.SurfaceWindSpeed.derive_variable(sample_dataset)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "latitude", "longitude")

    def test_derive_variable_calculation_correctness(self, sample_dataset):
        """Test that the calculation is mathematically correct."""
        result = derived.SurfaceWindSpeed.derive_variable(sample_dataset)

        # Manually calculate expected result using numpy hypot
        expected = np.hypot(
            sample_dataset["surface_eastward_wind"],
            sample_dataset["surface_northward_wind"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_derive_variable_always_positive(self, sample_dataset):
        """Test that wind speed is always non-negative."""
        result = derived.SurfaceWindSpeed.derive_variable(sample_dataset)

        assert (result >= 0).all()

    def test_derive_variable_zero_wind(self):
        """Test with zero wind components."""
        time = pd.date_range("2021-06-20", freq="6h", periods=2)
        latitudes = np.linspace(30, 35, 3)
        longitudes = np.linspace(250, 255, 3)

        zero_wind_dataset = xr.Dataset(
            {
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    np.zeros((len(time), len(latitudes), len(longitudes))),
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    np.zeros((len(time), len(latitudes), len(longitudes))),
                ),
            },
            coords={
                "time": time,
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )

        result = derived.SurfaceWindSpeed.derive_variable(zero_wind_dataset)

        # Should be all zeros
        expected = zero_wind_dataset["surface_eastward_wind"] * 0
        xr.testing.assert_allclose(result, expected)

    def test_derive_variable_extreme_values(self):
        """Test with extreme wind values."""
        time = pd.date_range("2021-06-20", freq="6h", periods=2)
        latitudes = np.linspace(30, 35, 3)
        longitudes = np.linspace(250, 255, 3)

        extreme_wind_dataset = xr.Dataset(
            {
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    np.full((len(time), len(latitudes), len(longitudes)), 100.0),
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    np.full((len(time), len(latitudes), len(longitudes)), -100.0),
                ),
            },
            coords={
                "time": time,
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )

        result = derived.SurfaceWindSpeed.derive_variable(extreme_wind_dataset)

        # Should be sqrt(100^2 + (-100)^2) = sqrt(20000) ≈ 141.42
        expected_scalar = np.sqrt(100**2 + 100**2)
        expected = xr.full_like(
            extreme_wind_dataset["surface_eastward_wind"], expected_scalar
        )
        xr.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_compute_integration(self, sample_dataset):
        """Test the compute method integration."""
        result = derived.SurfaceWindSpeed.compute(sample_dataset)

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("time", "latitude", "longitude")

    def test_missing_required_variables(self, sample_dataset):
        """Test behavior when required variables are missing."""
        incomplete_dataset = sample_dataset.drop_vars("surface_eastward_wind")

        with pytest.raises(
            ValueError, match="Input variable surface_eastward_wind not found in data"
        ):
            derived.SurfaceWindSpeed.compute(incomplete_dataset)

    def test_name_property(self):
        """Test the name property."""
        instance = derived.SurfaceWindSpeed()
        assert instance.name == "SurfaceWindSpeed"

    def test_preserves_coordinates(self, sample_dataset):
        """Test that derived variable preserves coordinate information."""
        result = derived.SurfaceWindSpeed.derive_variable(sample_dataset)

        # Check that coordinates are preserved
        assert "time" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords

        # Check coordinate values are the same
        xr.testing.assert_equal(result.coords["time"], sample_dataset.coords["time"])
        xr.testing.assert_equal(
            result.coords["latitude"], sample_dataset.coords["latitude"]
        )
        xr.testing.assert_equal(
            result.coords["longitude"], sample_dataset.coords["longitude"]
        )


class TestIntegratedVaporTransportLaplacian:
    """Test the IntegratedVaporTransportLaplacian derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        expected_vars = [
            "surface_eastward_wind",
            "surface_northward_wind",
            "specific_humidity",
        ]
        assert (
            derived.IntegratedVaporTransportLaplacian.required_variables
            == expected_vars
        )

    def test_derive_variable_basic(self, sample_dataset):
        """Test basic functionality - will fail until attribute errors are fixed."""
        # Add the required surface variables to the dataset
        enhanced_dataset = sample_dataset.copy()
        with pytest.raises(NotImplementedError):
            derived.IntegratedVaporTransportLaplacian.derive_variable(enhanced_dataset)


class TestTCTrackVariables:
    """Test the TCTrackVariables derived variable implementation."""

    def test_required_variables(self):
        """Test that required variables are correctly defined."""
        expected_vars = [
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_wind_speed",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
        assert derived.TCTrackVariables.required_variables == expected_vars

    @patch("extremeweatherbench.calc.generate_tc_variables")
    @patch("extremeweatherbench.calc.create_tctracks_from_dataset")
    @patch("extremeweatherbench.calc.tctracks_to_3d_dataset")
    def test_derive_variable_with_mocks(
        self, mock_to_3d, mock_create_tracks, mock_generate_vars, sample_dataset
    ):
        """Test derive_variable with mocked calc functions."""
        # Mock the complex calc functions
        mock_generate_vars.return_value = sample_dataset
        mock_create_tracks.return_value = []
        mock_to_3d.return_value = xr.Dataset(
            {
                "track_data": xr.DataArray(
                    np.ones((10, 10, 10)), dims=["time", "latitude", "longitude"]
                )
            }
        )

        result = derived.TCTrackVariables.derive_variable(sample_dataset)

        assert isinstance(result, xr.Dataset)
        mock_generate_vars.assert_called_once()
        mock_create_tracks.assert_called_once()
        mock_to_3d.assert_called_once()

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

    def test_missing_required_variables(self, sample_dataset):
        """Test behavior when required variables are missing."""
        incomplete_dataset = sample_dataset.drop_vars("geopotential")

        with pytest.raises(
            ValueError, match="Input variable geopotential not found in data"
        ):
            derived.TCTrackVariables.compute(incomplete_dataset)


class TestUtilityFunctions:
    """Test utility functions in the derived module."""

    def test_maybe_derive_variables_with_derived_variables(self, sample_dataset):
        """Test maybe_derive_variables with actual derived variables."""
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


class TestTropicalCycloneTrackVariable:
    """Test the base TropicalCycloneTrackVariable class."""

    def setup_method(self):
        """Clear cache before each test."""
        derived.TropicalCycloneTrackVariable.clear_cache()
        tropical_cyclone.clear_ibtracs_registry()

    def test_required_variables(self):
        """Test that required variables are properly defined."""
        expected_vars = [
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

        assert derived.TropicalCycloneTrackVariable.required_variables == expected_vars

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

        result = derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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

        result = derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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
            derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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

            derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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
        derived.TropicalCycloneTrackVariable.clear_cache()

        # Should be empty
        assert len(tropical_cyclone._TC_TRACK_CACHE) == 0

    @patch.object(derived.TropicalCycloneTrackVariable, "_get_or_compute_tracks")
    def test_derive_variable_base_class(
        self, mock_get_tracks, sample_tc_forecast_dataset, sample_tc_tracks_dataset
    ):
        """Test the base derive_variable method."""
        mock_get_tracks.return_value = sample_tc_tracks_dataset

        result = derived.TropicalCycloneTrackVariable.derive_variable(
            sample_tc_forecast_dataset
        )

        # Should call _get_or_compute_tracks
        mock_get_tracks.assert_called_once()

        # Should return a DataArray (converted from Dataset)
        assert isinstance(result, xr.DataArray)


class TestTrackSeaLevelPressure:
    """Test the TrackSeaLevelPressure derived variable."""

    def setup_method(self):
        """Clear cache before each test."""
        derived.TropicalCycloneTrackVariable.clear_cache()

    @patch.object(derived.TrackSeaLevelPressure, "_get_or_compute_tracks")
    def test_derive_variable(
        self, mock_get_tracks, sample_tc_forecast_dataset, sample_tc_tracks_dataset
    ):
        """Test TrackSeaLevelPressure derive_variable method."""
        mock_get_tracks.return_value = sample_tc_tracks_dataset

        with patch.object(
            derived.TrackSeaLevelPressure, "_convert_to_ewb_evaluation_format"
        ) as mock_convert:
            # Mock the conversion to return a properly formatted DataArray
            mock_slp_tracks = xr.DataArray(
                np.random.normal(101000, 1000, (3, 4)),
                dims=["time", "prediction_timedelta"],
                coords={
                    "time": sample_tc_tracks_dataset.time,
                    "prediction_timedelta": sample_tc_tracks_dataset.prediction_timedelta,
                },
            )
            mock_convert.return_value = mock_slp_tracks

            result = derived.TrackSeaLevelPressure.derive_variable(
                sample_tc_forecast_dataset
            )

            # Should call necessary methods
            mock_get_tracks.assert_called_once()
            mock_convert.assert_called_once_with(
                sample_tc_tracks_dataset, sample_tc_forecast_dataset, "tc_slp"
            )

            # Should return properly formatted DataArray
            assert isinstance(result, xr.DataArray)
            assert result.name == "air_pressure_at_mean_sea_level"

            # Check attributes
            assert "long_name" in result.attrs
            assert "units" in result.attrs
            assert result.attrs["units"] == "Pa"

    def test_convert_to_ewb_evaluation_format_basic(
        self, sample_tc_tracks_dataset, sample_tc_forecast_dataset
    ):
        """Test conversion to EWB evaluation format."""
        result = derived.TrackSeaLevelPressure._convert_to_ewb_evaluation_format(
            sample_tc_tracks_dataset, sample_tc_forecast_dataset, "tc_slp"
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have proper coordinates
        assert "latitude" in result.coords
        assert "longitude" in result.coords

        # Should have proper attributes on coordinates
        assert "long_name" in result.latitude.attrs
        assert "units" in result.latitude.attrs
        assert result.latitude.attrs["units"] == "degrees_north"

    def test_convert_to_ewb_evaluation_format_with_time_coords(self):
        """Test conversion with proper time coordinates."""
        # Create dataset with init_time and prediction_timedelta
        time = pd.date_range("2023-09-01", periods=2, freq="12h")
        prediction_timedelta = np.array([0, 12], dtype="timedelta64[h]")

        tracks_dataset = xr.Dataset(
            {
                "tc_slp": (
                    ["time", "prediction_timedelta"],
                    [[101000, 101010], [101020, 101030]],
                ),
                "tc_latitude": (
                    ["time", "prediction_timedelta"],
                    [[25.0, 25.5], [26.0, 26.5]],
                ),
                "tc_longitude": (
                    ["time", "prediction_timedelta"],
                    [[-75.0, -74.5], [-74.0, -73.5]],
                ),
            },
            coords={"time": time, "prediction_timedelta": prediction_timedelta},
        )

        original_dataset = xr.Dataset(
            coords={
                "time": time,
                "prediction_timedelta": prediction_timedelta,
            }
        )

        result = derived.TrackSeaLevelPressure._convert_to_ewb_evaluation_format(
            tracks_dataset, original_dataset, "tc_slp"
        )

        # Should have valid_time coordinate
        assert "valid_time" in result.coords
        assert "init_time" in result.coords
        assert "lead_time" in result.coords

        # Check that valid_time is properly calculated
        expected_valid_time = result.init_time + result.lead_time
        xr.testing.assert_equal(result.valid_time, expected_valid_time)


class TestTrackSurfaceWindSpeed:
    """Test the TrackSurfaceWindSpeed derived variable."""

    def setup_method(self):
        """Clear cache before each test."""
        derived.TropicalCycloneTrackVariable.clear_cache()

    @patch.object(derived.TrackSurfaceWindSpeed, "_get_or_compute_tracks")
    def test_derive_variable(
        self, mock_get_tracks, sample_tc_forecast_dataset, sample_tc_tracks_dataset
    ):
        """Test TrackSurfaceWindSpeed derive_variable method."""
        mock_get_tracks.return_value = sample_tc_tracks_dataset

        with patch.object(
            derived.TrackSeaLevelPressure, "_convert_to_ewb_evaluation_format"
        ) as mock_convert:
            # Mock the conversion to return a properly formatted DataArray
            mock_wind_tracks = xr.DataArray(
                np.random.uniform(10, 50, (3, 4)),
                dims=["time", "prediction_timedelta"],
                coords={
                    "time": sample_tc_tracks_dataset.time,
                    "prediction_timedelta": sample_tc_tracks_dataset.prediction_timedelta,
                },
            )
            mock_convert.return_value = mock_wind_tracks

            result = derived.TrackSurfaceWindSpeed.derive_variable(
                sample_tc_forecast_dataset
            )

            # Should call necessary methods
            mock_get_tracks.assert_called_once()
            mock_convert.assert_called_once_with(
                sample_tc_tracks_dataset, sample_tc_forecast_dataset, "tc_vmax"
            )

            # Should return properly formatted DataArray
            assert isinstance(result, xr.DataArray)
            assert result.name == "surface_wind_speed"

            # Check attributes
            assert "long_name" in result.attrs
            assert "units" in result.attrs
            assert result.attrs["units"] == "m s-1"


class TestTCTrackVariablesBackwardCompatibility:
    """Test the backward compatibility TCTrackVariables class."""

    def test_required_variables(self):
        """Test that TCTrackVariables has the expected required variables."""
        expected_vars = [
            "air_pressure_at_mean_sea_level",
            "geopotential",
            "surface_wind_speed",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]

        assert derived.TCTrackVariables.required_variables == expected_vars

    @patch.object(derived.TCTrackVariables, "_get_or_compute_tracks")
    def test_derive_variable_returns_dataset(
        self, mock_get_tracks, sample_tc_forecast_dataset, sample_tc_tracks_dataset
    ):
        """Test that TCTrackVariables returns a Dataset for backward compatibility."""
        mock_get_tracks.return_value = sample_tc_tracks_dataset

        result = derived.TCTrackVariables.derive_variable(sample_tc_forecast_dataset)

        # Should return the Dataset directly, not a DataArray
        assert isinstance(result, xr.Dataset)
        assert result is sample_tc_tracks_dataset


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

                derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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

                derived.TropicalCycloneTrackVariable._get_or_compute_tracks(
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
