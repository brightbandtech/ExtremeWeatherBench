"""
Comprehensive unit tests for the extremeweatherbench.events.severe_convection module.

This test suite covers:
- All atmospheric physics calculation functions
- Input validation and error handling
- Physical consistency checks
- Edge cases and boundary conditions
- Performance with realistic atmospheric data

Test data is designed to represent realistic atmospheric conditions
while being deterministic for reliable testing.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench.events import severe_convection as sc


@pytest.fixture
def sample_pressure_levels():
    """Standard atmospheric pressure levels in hPa (descending order)."""
    return np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100])


@pytest.fixture
def sample_temperature_profile():
    """Realistic temperature profile in Celsius for testing."""
    # Typical temperature profile: warm at surface, cooling with height
    return np.array(
        [25.0, 20.0, 15.0, 5.0, -15.0, -25.0, -35.0, -45.0, -55.0, -65.0, -75.0]
    )


@pytest.fixture
def sample_dewpoint_profile():
    """Realistic dewpoint profile in Celsius for testing."""
    # Dewpoint decreases with height, always <= temperature
    return np.array(
        [20.0, 15.0, 10.0, 0.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0]
    )


@pytest.fixture
def sample_wind_profile():
    """Realistic wind profiles in m/s for testing."""
    return {
        "u": np.array([5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]),
        "v": np.array([2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]),
    }


@pytest.fixture
def sample_atmospheric_dataset(
    sample_pressure_levels,
    sample_temperature_profile,
    sample_dewpoint_profile,
    sample_wind_profile,
):
    """Create a realistic atmospheric dataset for testing convection calculations."""
    time = pd.date_range("2021-06-20", freq="6h", periods=4)
    latitudes = np.linspace(35, 45, 6)  # Mid-latitude region
    longitudes = np.linspace(-100, -90, 11)  # Central US

    # Create 4D arrays (time, lat, lon, level)
    nt, nlat, nlon, nlev = (
        len(time),
        len(latitudes),
        len(longitudes),
        len(sample_pressure_levels),
    )

    # Add some spatial and temporal variability
    np.random.seed(42)  # For reproducible tests

    # Temperature with realistic lapse rate and variability
    temp_base = np.tile(sample_temperature_profile, (nt, nlat, nlon, 1))
    temp_noise = np.random.normal(0, 2, (nt, nlat, nlon, nlev))
    temperature = temp_base + temp_noise

    # Dewpoint always less than temperature
    dew_base = np.tile(sample_dewpoint_profile, (nt, nlat, nlon, 1))
    dew_noise = np.random.normal(0, 1.5, (nt, nlat, nlon, nlev))
    dewpoint = np.minimum(dew_base + dew_noise, temperature - 0.5)

    # Wind profiles with some variability
    u_base = np.tile(sample_wind_profile["u"], (nt, nlat, nlon, 1))
    v_base = np.tile(sample_wind_profile["v"], (nt, nlat, nlon, 1))
    u_noise = np.random.normal(0, 3, (nt, nlat, nlon, nlev))
    v_noise = np.random.normal(0, 3, (nt, nlat, nlon, nlev))
    u_wind = u_base + u_noise
    v_wind = v_base + v_noise

    # Pressure levels broadcasted to match data dimensions
    pressure = np.broadcast_to(sample_pressure_levels, (nt, nlat, nlon, nlev))

    # Surface variables (first level)
    surface_u = u_wind[..., 0]
    surface_v = v_wind[..., 0]

    dataset = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude", "level"], temperature),
            "dewpoint": (["time", "latitude", "longitude", "level"], dewpoint),
            "pressure": (["time", "latitude", "longitude", "level"], pressure),
            "eastward_wind": (["time", "latitude", "longitude", "level"], u_wind),
            "northward_wind": (["time", "latitude", "longitude", "level"], v_wind),
            "surface_eastward_wind": (["time", "latitude", "longitude"], surface_u),
            "surface_northward_wind": (["time", "latitude", "longitude"], surface_v),
        },
        coords={
            "time": time,
            "latitude": latitudes,
            "longitude": longitudes,
            "level": sample_pressure_levels,
        },
    )

    return dataset


class TestPhysicalConstants:
    """Test the physical constants defined in the module."""

    def test_constants_exist(self):
        """Test that all required physical constants are defined."""
        assert hasattr(sc, "Rd")
        assert hasattr(sc, "Cp_d")
        assert hasattr(sc, "g")
        assert hasattr(sc, "epsilon")
        assert hasattr(sc, "kappa")

    def test_constant_values(self):
        """Test that constants have expected physical values."""
        # Dry air gas constant
        assert 287.0 < sc.Rd < 288.0
        # Specific heat of dry air
        assert 1004.0 < sc.Cp_d < 1005.0
        # Gravitational acceleration
        assert 9.8 < sc.g < 9.82
        # Water vapor / dry air molecular weight ratio
        assert 0.621 < sc.epsilon < 0.623
        # Poisson constant (should equal Rd/Cp_d)
        assert abs(sc.kappa - sc.Rd / sc.Cp_d) < 1e-10


class TestBasicDatasetChecks:
    """Test the _basic_ds_checks function."""

    def test_basic_ds_checks_pressure_sorting(self, sample_atmospheric_dataset):
        """Test that pressure levels are sorted in descending order."""
        # Create dataset with ascending pressure levels
        ds = sample_atmospheric_dataset.copy()
        ds = ds.sortby("level", ascending=True)  # Wrong order

        # Function should correct the order
        result = sc._basic_ds_checks(ds)

        # Check that pressure is now descending
        assert result["level"][0] > result["level"][-1]
        np.testing.assert_array_equal(result["level"], np.sort(ds["level"])[::-1])

    def test_basic_ds_checks_level_dimension_position(self, sample_atmospheric_dataset):
        """Test that level dimension is moved to the last position."""
        # Transpose to put level dimension first
        ds = sample_atmospheric_dataset.transpose(
            "level", "time", "latitude", "longitude"
        )

        result = sc._basic_ds_checks(ds)

        # Level should now be the last dimension
        assert list(result.dims)[-1] == "level"

    def test_basic_ds_checks_stratosphere_removal(self, sample_atmospheric_dataset):
        """Test that levels above 50 hPa are removed."""
        # Add some high altitude levels
        ds = sample_atmospheric_dataset.copy()
        high_levels = np.array([30, 20, 10])
        ds = ds.sel(level=slice(None)).interp(
            level=np.concatenate([ds.level.values, high_levels])
        )

        result = sc._basic_ds_checks(ds)

        # All resulting levels should be >= 50 hPa
        assert np.all(result["level"] >= 50)

    def test_basic_ds_checks_returns_dataset(self, sample_atmospheric_dataset):
        """Test that function returns an xarray Dataset."""
        result = sc._basic_ds_checks(sample_atmospheric_dataset)
        assert isinstance(result, xr.Dataset)


class TestThermodynamicFunctions:
    """Test basic thermodynamic calculation functions."""

    def test_mixing_ratio_basic(self):
        """Test mixing ratio calculation with known values."""
        # Test with simple values
        partial_p = 10.0  # hPa
        total_p = 1000.0  # hPa

        result = sc.mixing_ratio(partial_p, total_p)

        # Expected: epsilon * e / (p - e)
        expected = sc.epsilon * partial_p / (total_p - partial_p)
        assert abs(result - expected) < 1e-10

    def test_mixing_ratio_arrays(self):
        """Test mixing ratio with array inputs."""
        partial_p = np.array([5.0, 10.0, 15.0])
        total_p = np.array([1000.0, 1000.0, 1000.0])

        result = sc.mixing_ratio(partial_p, total_p)

        assert result.shape == partial_p.shape
        assert np.all(result > 0)
        assert np.all(result < 0.1)  # Reasonable physical range

    def test_mixing_ratio_edge_cases(self):
        """Test mixing ratio edge cases."""
        # Very small partial pressure
        result = sc.mixing_ratio(0.1, 1000.0)
        assert result > 0
        assert result < 1e-3

        # Partial pressure approaching total pressure
        result = sc.mixing_ratio(999.0, 1000.0)
        assert result > 0.1  # Should be large but finite

    def test_saturation_vapor_pressure_known_values(self):
        """Test saturation vapor pressure at known temperatures."""
        # At 0°C, should be ~6.11 hPa
        result = sc.saturation_vapor_pressure(0.0)
        assert 6.0 < result < 6.3

        # At 20°C, should be ~23.4 hPa
        result = sc.saturation_vapor_pressure(20.0)
        assert 23.0 < result < 24.0

        # At 30°C, should be ~42.4 hPa
        result = sc.saturation_vapor_pressure(30.0)
        assert 42.0 < result < 43.0

    def test_saturation_vapor_pressure_arrays(self):
        """Test saturation vapor pressure with array inputs."""
        temps = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
        result = sc.saturation_vapor_pressure(temps)

        assert result.shape == temps.shape
        # Should increase with temperature
        assert np.all(np.diff(result) > 0)
        # All values should be positive
        assert np.all(result > 0)

    def test_exner_function(self):
        """Test Exner function calculation."""
        # At 1000 hPa (reference pressure)
        result = sc.exner_function(1000.0)
        assert abs(result - 1.0) < 1e-10

        # At 500 hPa
        result = sc.exner_function(500.0)
        expected = (500.0 / 1000.0) ** sc.kappa
        assert abs(result - expected) < 1e-10

    def test_potential_temperature(self):
        """Test potential temperature calculation."""
        # At reference pressure, theta should equal temperature
        temp = 20.0  # Celsius
        pressure = 1000.0  # hPa

        result = sc.potential_temperature(temp, pressure, units="C")
        expected = temp + 273.15  # Convert to Kelvin
        assert abs(result - expected) < 1e-10

        # Test Kelvin input
        result_k = sc.potential_temperature(temp + 273.15, pressure, units="K")
        assert abs(result_k - expected) < 1e-10

    def test_potential_temperature_arrays(self):
        """Test potential temperature with array inputs."""
        temps = np.array([20.0, 15.0, 10.0])
        pressures = np.array([1000.0, 850.0, 700.0])

        result = sc.potential_temperature(temps, pressures)

        assert result.shape == temps.shape
        # All should be > 0 K
        assert np.all(result > 273.0)
        # Higher altitude (lower pressure) should have higher potential temperature
        assert result[2] > result[0]  # 700 hPa vs 1000 hPa

    def test_dewpoint_from_vapor_pressure(self):
        """Test dewpoint calculation from vapor pressure."""
        # Known relationships
        vp = 6.11  # hPa (approximately at 0°C)
        result = sc.dewpoint_from_vapor_pressure(vp)
        assert abs(result - 0.0) < 0.5  # Should be close to 0°C

        # Test with array
        vps = np.array([3.0, 6.11, 12.0, 23.4])
        result = sc.dewpoint_from_vapor_pressure(vps)
        assert result.shape == vps.shape
        # Should increase with vapor pressure
        assert np.all(np.diff(result) > 0)


class TestVirtualTemperature:
    """Test virtual temperature calculations."""

    def test_virtual_temperature_basic(self):
        """Test virtual temperature calculation."""
        temp = 273.15 + 20.0  # 20°C in Kelvin
        mixing_ratio = 0.010  # 10 g/kg

        result = sc.virtual_temperature(temp, mixing_ratio)

        # Virtual temperature should be higher than actual temperature for moist air
        assert result > temp
        # Typical correction is 1-3 K for moist air
        assert result - temp < 5.0

    def test_virtual_temperature_dry_air(self):
        """Test virtual temperature with very dry air."""
        temp = 273.15 + 20.0
        mixing_ratio = 0.0  # Completely dry

        result = sc.virtual_temperature(temp, mixing_ratio)

        # With no moisture, virtual temp should equal actual temp
        assert abs(result - temp) < 1e-10

    def test_virtual_temperature_from_dewpoint(self):
        """Test virtual temperature calculation from dewpoint."""
        pressure = 1000.0  # hPa
        temperature = 20.0  # Celsius
        dewpoint = 15.0  # Celsius

        result = sc.virtual_temperature_from_dewpoint(pressure, temperature, dewpoint)

        # Should be higher than original temperature
        assert result > temperature + 273.15
        # But not too much higher (reasonable moist air correction)
        assert result < temperature + 273.15 + 5.0


class TestLiftingCondensationLevel:
    """Test LCL calculation functions."""

    def test_lifting_condensation_level_basic(self):
        """Test LCL calculation with realistic values."""
        pressure = np.array([1000.0])
        temperature = np.array([25.0])
        dewpoint = np.array([20.0])

        lcl_p, lcl_td = sc.lifting_condensation_level(pressure, temperature, dewpoint)

        # LCL pressure should be less than surface pressure
        assert lcl_p[0] < pressure[0]
        # LCL pressure should be reasonable (typically 800-950 hPa for surface parcels)
        assert 700 < lcl_p[0] < 1000
        # LCL temperature should equal dewpoint temperature
        assert abs(lcl_td[0] - dewpoint[0]) < 1.0

    def test_lifting_condensation_level_dry_air(self):
        """Test LCL calculation with very dry air."""
        pressure = np.array([1000.0])
        temperature = np.array([30.0])
        dewpoint = np.array([0.0])  # Very dry

        lcl_p, lcl_td = sc.lifting_condensation_level(pressure, temperature, dewpoint)

        # Very dry air should have low LCL pressure
        assert lcl_p[0] < 800

    def test_new_lcl_function(self):
        """Test the new LCL calculation function."""
        pressure = 1000.0
        temperature = 25.0 + 273.15  # Kelvin
        dewpoint = 20.0 + 273.15  # Kelvin

        lcl_p, lcl_t = sc.new_lcl(pressure, temperature, dewpoint)

        # Should return reasonable values
        assert 700 < lcl_p < 1000
        assert 250 < lcl_t < 300  # Kelvin
        # LCL temperature should be less than original temperature
        assert lcl_t < temperature


class TestMixedLayerFunctions:
    """Test mixed layer calculations."""

    def test_mixed_parcel_basic(self, sample_atmospheric_dataset):
        """Test mixed parcel calculation."""
        ds = sample_atmospheric_dataset.isel(time=0, latitude=0, longitude=0)

        # Extract 1D profiles for testing
        ds_1d = xr.Dataset(
            {
                "temperature": (["level"], ds.temperature.values),
                "dewpoint": (["level"], ds.dewpoint.values),
                "pressure": (["level"], ds.pressure.values),
            },
            coords={"level": ds.level},
        )

        start_p, temp, dewpoint = sc.mixed_parcel(
            ds_1d,
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
            depth=100.0,
            temperature_units="C",
        )

        # Start pressure should be surface pressure
        assert abs(start_p - ds.level[0].item()) < 1e-6
        # Temperature should be reasonable
        assert 250 < temp < 320  # Kelvin
        # Dewpoint should be less than temperature
        assert dewpoint < temp

    def test_interp_integrate_function(self):
        """Test the interpolation and integration helper function."""
        pressure = np.array([1000, 900, 800])
        pressure_interp = np.array([1000, 950, 900, 850, 800])
        layer_depth = 200  # hPa

        # Test with simple linear data
        vars_data = np.array([10, 8, 6])  # Decreases linearly with height

        result = sc._interp_integrate(pressure, pressure_interp, layer_depth, vars_data)

        # Should return a scalar
        assert np.isscalar(result)
        # Should be reasonable average
        assert 6 < result < 10


class TestConvectionCalculations:
    """Test CAPE and CIN calculations."""

    @pytest.mark.slow
    def test_mixed_layer_cape_cin_basic(self, sample_atmospheric_dataset):
        """Test mixed layer CAPE and CIN calculation."""
        # Use a small subset for faster testing
        ds = sample_atmospheric_dataset.isel(
            time=0, latitude=slice(0, 2), longitude=slice(0, 2)
        )

        cape, cin = sc.mixed_layer_cape_cin(
            ds,
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
            depth=100.0,
        )

        # Check shapes
        expected_shape = (2, 2)  # lat x lon
        assert cape.shape == expected_shape
        assert cin.shape == expected_shape

        # CAPE should be non-negative
        assert np.all(cape >= 0)
        # CIN should be non-positive
        assert np.all(cin <= 0)

        # Values should be in reasonable ranges
        assert np.all(cape < 10000)  # Extreme CAPE would be > 5000 J/kg
        assert np.all(cin > -5000)  # Extreme CIN would be < -500 J/kg

    def test_find_intersections_basic(self):
        """Test the find_intersections function."""
        x = np.array([1000, 900, 800, 700, 600])
        y1 = np.array([10, 8, 6, 4, 2])  # Decreasing
        y2 = np.array([2, 4, 6, 8, 10])  # Increasing

        # These should intersect at 800 hPa where both equal 6
        x_int, y_int = sc.find_intersections(x, y1, y2, direction="all")

        # Should find one intersection
        assert len(x_int) >= 1
        # Intersection should be near x=800, y=6
        if len(x_int) > 0:
            assert 750 < x_int[0] < 850
            assert 5 < y_int[0] < 7

    def test_level_free_convection_basic(self):
        """Test LFC calculation."""
        pressure = np.array([1000, 900, 800, 700, 600])
        temperature = np.array([25, 20, 15, 10, 5])
        dewpoint = np.array([20, 15, 10, 5, 0])
        parcel_profile = np.array([25, 22, 18, 14, 10])  # Warmer than environment

        lfc_p, lfc_t = sc.level_free_convection(
            pressure, temperature, dewpoint, parcel_profile
        )

        # Should find a reasonable LFC
        if not np.isnan(lfc_p):
            assert 600 < lfc_p < 1000
            assert 0 < lfc_t < 30

    def test_equilibrium_level_basic(self):
        """Test equilibrium level calculation."""
        pressure = np.array([1000, 900, 800, 700, 600, 500, 400])
        temperature = np.array([25, 20, 15, 10, 5, 0, -10])
        dewpoint = np.array([20, 15, 10, 5, 0, -10, -20])
        parcel_profile = np.array([25, 22, 18, 14, 8, -5, -15])  # Crosses environment

        el_p, el_t = sc.equilibrium_level(
            pressure, temperature, dewpoint, parcel_profile
        )

        # Should find a reasonable EL or return NaN
        if not np.isnan(el_p):
            assert 300 < el_p < 800
            assert -20 < el_t < 20


class TestSevereWeatherIndices:
    """Test severe weather parameter calculations."""

    def test_low_level_shear_basic(self, sample_atmospheric_dataset):
        """Test low-level shear calculation."""
        ds = sample_atmospheric_dataset.isel(time=0)

        shear = sc.low_level_shear(
            ds,
            eastward_wind_var="eastward_wind",
            northward_wind_var="northward_wind",
            surface_eastward_wind_var="surface_eastward_wind",
            surface_northward_wind_var="surface_northward_wind",
        )

        # Check shape matches spatial dimensions
        expected_shape = ds.surface_eastward_wind.shape
        assert shear.shape == expected_shape

        # Shear should be non-negative (magnitude)
        assert np.all(shear >= 0)

        # Should be reasonable values (0-50 m/s typical)
        assert np.all(shear < 100)

    @pytest.mark.slow
    def test_craven_brooks_significant_severe(self, sample_atmospheric_dataset):
        """Test Craven-Brooks significant severe parameter."""
        # Use small subset for faster testing
        ds = sample_atmospheric_dataset.isel(
            time=0, latitude=slice(0, 2), longitude=slice(0, 2)
        )

        cbss = sc.craven_brooks_significant_severe(
            ds,
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
            eastward_wind_var="eastward_wind",
            northward_wind_var="northward_wind",
            surface_eastward_wind_var="surface_eastward_wind",
            surface_northward_wind_var="surface_northward_wind",
            depth=100.0,
        )

        # Check output type and shape
        assert isinstance(cbss, xr.DataArray)
        expected_shape = (2, 2)
        assert cbss.shape == expected_shape

        # CBSS should be non-negative (product of CAPE and shear magnitude)
        assert np.all(cbss >= 0)

        # Should be in reasonable range for the parameter
        assert np.all(cbss < 200000)  # Very high values would be > 100,000 m³/s³


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_dewpoint_from_specific_humidity(self):
        """Test dewpoint calculation from specific humidity."""
        specific_humidity = xr.DataArray([0.005, 0.010, 0.015])  # kg/kg
        pressure = xr.DataArray([1000.0, 1000.0, 1000.0])  # hPa

        result = sc.dewpoint_from_specific_humidity(specific_humidity, pressure)

        # Should return reasonable dewpoint values
        assert isinstance(result, xr.DataArray)
        assert result.shape == specific_humidity.shape
        # Higher specific humidity should give higher dewpoint
        assert np.all(np.diff(result) > 0)
        # Should be reasonable values (-50 to 30°C range)
        assert np.all(result > -50)
        assert np.all(result < 50)

    def test_log_interpolate_function(self):
        """Test logarithmic interpolation function."""
        x = np.array([500, 600, 700, 800, 900])  # Target pressures
        xp = np.array([1000, 900, 800, 700, 600, 500])  # Source pressures
        var = np.array([25, 20, 15, 10, 5, 0])  # Temperature data

        result = sc.log_interpolate(x, xp, var)

        # Should return interpolated values
        assert result.shape == (len(x),)
        # Values should be reasonable (between min and max of input)
        assert np.all(result >= np.min(var))
        assert np.all(result <= np.max(var))

    def test_insert_lcl_level_fast(self):
        """Test fast LCL level insertion."""
        # Simple test case
        pressure = np.array([[[1000, 900, 800, 700]]])
        temperature = np.array([[[25, 20, 15, 10]]])
        lcl_pressure = np.array([[[850]]])

        result = sc.insert_lcl_level_fast(pressure, temperature, lcl_pressure)

        # Should have one more level than input
        assert result.shape[-1] == pressure.shape[-1] + 1
        # LCL pressure should be included
        assert 850 in result.flatten()


class TestPhysicalConsistency:
    """Test physical consistency of calculations."""

    def test_temperature_dewpoint_relationship(self, sample_atmospheric_dataset):
        """Test that dewpoint is always <= temperature."""
        ds = sample_atmospheric_dataset

        # Check all grid points and levels
        assert np.all(
            ds.dewpoint <= ds.temperature + 0.1
        )  # Small tolerance for numerical errors

    def test_pressure_monotonicity(self, sample_atmospheric_dataset):
        """Test that pressure decreases with level index (after sorting)."""
        ds = sc._basic_ds_checks(sample_atmospheric_dataset)

        # Pressure should be in descending order
        assert np.all(np.diff(ds.level) < 0)

    def test_cape_cin_energy_conservation(self, sample_atmospheric_dataset):
        """Test basic energy conservation principles in CAPE/CIN calculations."""
        # Use minimal dataset for faster testing
        ds = sample_atmospheric_dataset.isel(time=0, latitude=0, longitude=0)

        # Create more realistic unstable environment for CAPE calculation
        ds = ds.copy()
        # Make surface warmer and moister
        ds["temperature"][0] = 30.0  # °C
        ds["dewpoint"][0] = 25.0  # °C

        cape, cin = sc.mixed_layer_cape_cin(
            ds.expand_dims(["time", "latitude", "longitude"], axis=[0, 1, 2]),
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
            depth=50.0,
        )

        # Physical constraints
        assert cape[0, 0] >= 0  # CAPE cannot be negative
        assert cin[0, 0] <= 0  # CIN cannot be positive

        # For unstable environment, should have some CAPE
        assert cape[0, 0] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_pressure_units(self):
        """Test behavior with invalid pressure values."""
        # Negative pressure should be handled gracefully
        with pytest.warns(RuntimeWarning, match="invalid value"):
            result = sc.exner_function(-100.0)
            assert np.isnan(result)

    def test_invalid_temperature_units(self):
        """Test behavior with invalid temperature values."""
        # Test potential temperature with wrong units
        with pytest.raises(ValueError):
            sc.potential_temperature(20.0, 1000.0, units="invalid")

    def test_missing_data_handling(self, sample_atmospheric_dataset):
        """Test handling of NaN values in input data."""
        ds = sample_atmospheric_dataset.copy()

        # Introduce some NaN values
        ds["temperature"][0, 0, 0, :] = np.nan

        # Should handle NaN gracefully without crashing
        try:
            cape, cin = sc.mixed_layer_cape_cin(
                ds.isel(time=0, latitude=slice(0, 1), longitude=slice(0, 1)),
                pressure_var="pressure",
                temperature_var="temperature",
                temperature_dewpoint_var="dewpoint",
            )
            # Results should contain NaN where input was NaN
            assert np.isnan(cape[0, 0])
            assert np.isnan(cin[0, 0])
        except Exception as e:
            pytest.fail(f"Function should handle NaN input gracefully, but raised: {e}")

    def test_extreme_atmospheric_conditions(self):
        """Test with extreme but physically possible atmospheric conditions."""
        # Very cold conditions
        cold_temp = -80.0
        pressure = 200.0  # High altitude

        # Should not crash with extreme values
        es = sc.saturation_vapor_pressure(cold_temp)
        assert es > 0  # Should still be positive

        mr = sc.saturation_mixing_ratio(pressure, cold_temp)
        assert mr > 0  # Should be positive but very small
        assert mr < 0.001  # Very dry air


class TestPerformance:
    """Test performance characteristics with larger datasets."""

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with a larger, more realistic dataset."""
        # Create larger dataset similar to operational weather models
        time = pd.date_range("2021-06-20", freq="6h", periods=8)
        latitudes = np.linspace(25, 50, 26)  # ~1 degree resolution
        longitudes = np.linspace(-125, -75, 51)  # US domain
        levels = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200])

        # Create realistic atmospheric profiles
        nt, nlat, nlon, nlev = (len(time), len(latitudes), len(longitudes), len(levels))

        np.random.seed(42)

        # Temperature decreasing with height
        temp_profile = np.array([25, 18, 12, 2, -18, -30, -45, -52, -62])
        temperature = np.tile(temp_profile, (nt, nlat, nlon, 1))
        temperature += np.random.normal(0, 3, (nt, nlat, nlon, nlev))

        # Dewpoint always less than temperature
        dew_profile = np.array([20, 12, 6, -5, -25, -40, -55, -62, -72])
        dewpoint = np.tile(dew_profile, (nt, nlat, nlon, 1))
        dewpoint += np.random.normal(0, 2, (nt, nlat, nlon, nlev))
        dewpoint = np.minimum(dewpoint, temperature - 1.0)

        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "latitude", "longitude", "level"],
                    temperature,
                ),
                "dewpoint": (["time", "latitude", "longitude", "level"], dewpoint),
                "pressure": (
                    ["time", "latitude", "longitude", "level"],
                    np.broadcast_to(levels, (nt, nlat, nlon, nlev)),
                ),
            },
            coords={
                "time": time,
                "latitude": latitudes,
                "longitude": longitudes,
                "level": levels,
            },
        )

        # Test with subset for reasonable test time
        ds_subset = ds.isel(
            time=slice(0, 2), latitude=slice(0, 5), longitude=slice(0, 10)
        )

        import time as timer

        start_time = timer.time()

        # This should complete in reasonable time (< 30 seconds)
        cape, cin = sc.mixed_layer_cape_cin(
            ds_subset,
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
        )

        elapsed_time = timer.time() - start_time

        # Performance check - should complete in reasonable time
        assert elapsed_time < 60  # Should complete within 1 minute

        # Verify results are reasonable
        assert cape.shape == (2, 5, 10)  # time x lat x lon
        assert cin.shape == (2, 5, 10)
        assert np.all(cape >= 0)
        assert np.all(cin <= 0)


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("depth", [50, 100, 150])
def test_mixed_layer_depth_variations(sample_atmospheric_dataset, depth):
    """Test mixed layer calculations with different depths."""
    ds = sample_atmospheric_dataset.isel(time=0, latitude=0, longitude=0)

    try:
        start_p, temp, dewpoint = sc.mixed_parcel(
            ds.expand_dims(["time", "latitude", "longitude"], axis=[0, 1, 2]),
            pressure_var="pressure",
            temperature_var="temperature",
            temperature_dewpoint_var="dewpoint",
            depth=depth,
        )

        # Should return reasonable values regardless of depth
        assert 250 < temp < 320  # Kelvin
        assert dewpoint < temp

    except Exception as e:
        pytest.fail(f"Mixed parcel calculation failed for depth {depth}: {e}")


@pytest.mark.parametrize(
    "temp,expected_range",
    [
        (-40, (0.1, 1.0)),  # Very cold
        (0, (6.0, 6.5)),  # Freezing point
        (20, (23.0, 24.0)),  # Room temperature
        (40, (73.0, 75.0)),  # Hot day
    ],
)
def test_saturation_vapor_pressure_ranges(temp, expected_range):
    """Test saturation vapor pressure across temperature range."""
    result = sc.saturation_vapor_pressure(temp)
    assert expected_range[0] <= result <= expected_range[1]


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestPhysicalConstants::test_constant_values", "-v"])
