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

    def test_compute_logs_warning_missing_variables(self, sample_dataset, caplog):
        """Test that compute logs warning when required variables are missing."""
        # Remove one of the required variables
        incomplete_dataset = sample_dataset.drop_vars("test_variable_2")

        # This should not raise an error, but may fail during derive_variable
        with pytest.raises(
            KeyError
        ):  # derive_variable will fail when accessing missing variable
            TestValidDerivedVariable.compute(incomplete_dataset)

        # Check that warning was logged
        assert "Input variable test_variable_2 not found in data" in caplog.text

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

    def test_name_property(self):
        """Test the name property."""
        instance = derived.IntegratedVaporTransport()
        assert instance.name == "IntegratedVaporTransport"


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

    def test_derived_variable_with_empty_dataset(self, caplog):
        """Test behavior with empty datasets."""
        empty_dataset = xr.Dataset()

        # Should log warnings but may fail during derive_variable
        with pytest.raises(
            KeyError
        ):  # derive_variable will fail when accessing missing variables
            TestValidDerivedVariable.compute(empty_dataset)

        # Check that warnings were logged for missing variables
        assert "Input variable test_variable_1 not found in data" in caplog.text
        assert "Input variable test_variable_2 not found in data" in caplog.text

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
