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

    # Temperature with realistic lapse rate and variability (convert to Kelvin)
    temp_base = np.tile(sample_temperature_profile, (nt, nlat, nlon, 1))
    temp_noise = np.random.normal(0, 2, (nt, nlat, nlon, nlev))
    temperature = temp_base + temp_noise + 273.15  # Convert to Kelvin

    # Dewpoint always less than temperature (convert to Kelvin)
    dew_base = np.tile(sample_dewpoint_profile, (nt, nlat, nlon, 1))
    dew_noise = np.random.normal(0, 1.5, (nt, nlat, nlon, nlev))
    dewpoint = np.minimum(dew_base + dew_noise, temperature - 273.15 - 0.5) + 273.15

    # Wind profiles with some variability
    u_base = np.tile(sample_wind_profile["u"], (nt, nlat, nlon, 1))
    v_base = np.tile(sample_wind_profile["v"], (nt, nlat, nlon, 1))
    u_noise = np.random.normal(0, 3, (nt, nlat, nlon, nlev))
    v_noise = np.random.normal(0, 3, (nt, nlat, nlon, nlev))
    u_wind = u_base + u_noise
    v_wind = v_base + v_noise

    # Pressure levels broadcasted to match data dimensions
    pressure = np.broadcast_to(sample_pressure_levels, (nt, nlat, nlon, nlev))

    # Calculate geopotential using hypsometric equation
    # Constants
    g = 9.80665  # gravitational acceleration (m/s²)
    Rd = 287.04  # dry air gas constant (J/kg/K)

    # Calculate geopotential height using standard atmosphere
    # Start from surface and integrate upward
    # Temperature is already in Kelvin
    geopotential = np.zeros_like(pressure)

    for i in range(nlev):
        if i == 0:
            # Surface level - assume sea level
            geopotential[..., i] = 0.0
        else:
            # Use hypsometric equation between levels
            p1 = pressure[..., i - 1]
            p2 = pressure[..., i]
            T_mean = (temperature[..., i - 1] + temperature[..., i]) / 2

            # Change in geopotential height
            dz = (Rd * T_mean / g) * np.log(p1 / p2)
            geopotential[..., i] = geopotential[..., i - 1] + g * dz

    # Surface variables (first level)
    surface_u = u_wind[..., 0]
    surface_v = v_wind[..., 0]

    dataset = xr.Dataset(
        {
            "air_temperature": (
                ["time", "latitude", "longitude", "level"],
                temperature,
            ),
            "dewpoint_temperature": (
                ["time", "latitude", "longitude", "level"],
                dewpoint,
            ),
            "pressure": (["time", "latitude", "longitude", "level"], pressure),
            "geopotential": (
                ["time", "latitude", "longitude", "level"],
                geopotential,
            ),
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


class TestSevereWeatherIndices:
    """Test severe weather parameter calculations."""

    def test_low_level_shear_basic(self, sample_atmospheric_dataset):
        """Test low-level shear calculation."""
        ds = sample_atmospheric_dataset.isel(time=0)

        shear = sc.low_level_shear(
            ds["eastward_wind"],
            ds["northward_wind"],
            ds["surface_eastward_wind"],
            ds["surface_northward_wind"],
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
            ds["air_temperature"],
            ds["dewpoint_temperature"],
            ds["geopotential"],
            ds["level"],
            ds["eastward_wind"],
            ds["northward_wind"],
            ds["surface_eastward_wind"],
            ds["surface_northward_wind"],
            layer_depth=100.0,
        )

        # Check output type and shape
        assert isinstance(cbss, xr.DataArray)
        expected_shape = (2, 2)
        assert cbss.shape == expected_shape

        # CBSS should be non-negative (product of CAPE and shear magnitude)
        # Handle NaN values from numerical issues
        valid_cbss = cbss.values[~np.isnan(cbss.values)]
        if len(valid_cbss) > 0:
            assert np.all(valid_cbss >= 0)

        # Should be in reasonable range for the parameter (handle NaN values)
        if len(valid_cbss) > 0:
            assert np.all(
                valid_cbss < 200000
            )  # Very high values would be > 100,000 m³/s³


class TestCapeRegression:
    """
    Regression tests to ensure CAPE calculations remain consistent.

    These tests verify that the severe_convection module produces identical results
    to the working derived.py implementation, preventing future regressions.
    """

    @pytest.fixture
    def regression_profile_data(self):
        """Real atmospheric profile data that previously caused CAPE calculation
        issues."""
        temperature_data = np.array(
            [
                [
                    [
                        [
                            301.28543,
                            299.1248,
                            297.09048,
                            295.43634,
                            294.92474,
                            295.21307,
                            293.94577,
                            292.04193,
                            290.3143,
                            288.84793,
                            287.33905,
                            283.8867,
                            279.625,
                            275.22556,
                            270.86078,
                            267.3587,
                            262.63983,
                            257.73883,
                            251.98715,
                            243.52557,
                            233.61082,
                            227.47162,
                            221.12878,
                            214.0264,
                            205.74919,
                            197.13243,
                            191.42583,
                            199.72433,
                            207.74922,
                            216.8675,
                            222.79562,
                        ]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        dewpoint_data = np.array(
            [
                [
                    [
                        [
                            297.61893116,
                            297.13275546,
                            296.44573097,
                            294.80113066,
                            291.13749249,
                            283.34013614,
                            282.99282511,
                            283.63682326,
                            284.33227243,
                            283.24106933,
                            280.77318722,
                            275.38862408,
                            272.4862496,
                            265.95849926,
                            259.03940721,
                            250.30099479,
                            251.54912813,
                            247.51921695,
                            233.09329238,
                            230.85397818,
                            227.70883628,
                            220.04485818,
                            212.19363724,
                            204.95689596,
                            198.84596866,
                            193.71371149,
                            187.67939167,
                            186.69142712,
                            183.83190612,
                            181.16366895,
                            178.95581394,
                        ]
                    ]
                ]
            ]
        )

        pressure_data = np.array(
            [
                [
                    [
                        [
                            1000.0,
                            975.0,
                            950.0,
                            925.0,
                            900.0,
                            875.0,
                            850.0,
                            825.0,
                            800.0,
                            775.0,
                            750.0,
                            700.0,
                            650.0,
                            600.0,
                            550.0,
                            500.0,
                            450.0,
                            400.0,
                            350.0,
                            300.0,
                            250.0,
                            225.0,
                            200.0,
                            175.0,
                            150.0,
                            125.0,
                            100.0,
                            70.0,
                            50.0,
                            30.0,
                            20.0,
                        ]
                    ]
                ]
            ]
        )

        wind_data = np.zeros_like(temperature_data)
        pressure_levels = np.array(
            [
                1000.0,
                975.0,
                950.0,
                925.0,
                900.0,
                875.0,
                850.0,
                825.0,
                800.0,
                775.0,
                750.0,
                700.0,
                650.0,
                600.0,
                550.0,
                500.0,
                450.0,
                400.0,
                350.0,
                300.0,
                250.0,
                225.0,
                200.0,
                175.0,
                150.0,
                125.0,
                100.0,
                70.0,
                50.0,
                30.0,
                20.0,
            ]
        )

        # Calculate geopotential using hypsometric equation
        g = 9.80665  # gravitational acceleration (m/s²)
        Rd = 287.04  # dry air gas constant (J/kg/K)

        geopotential_data = np.zeros_like(temperature_data)
        nlev = len(pressure_levels)

        for i in range(nlev):
            if i == 0:
                geopotential_data[..., i] = 0.0
            else:
                p1 = pressure_data[..., i - 1]
                p2 = pressure_data[..., i]
                T_mean = (temperature_data[..., i - 1] + temperature_data[..., i]) / 2

                # Change in geopotential
                dz = (Rd * T_mean / g) * np.log(p1 / p2)
                geopotential_data[..., i] = geopotential_data[..., i - 1] + g * dz

        return xr.Dataset(
            {
                "air_temperature": (
                    ["time", "latitude", "longitude", "level"],
                    temperature_data,
                ),
                "dewpoint_temperature": (
                    ["time", "latitude", "longitude", "level"],
                    dewpoint_data,
                ),
                "pressure": (["time", "latitude", "longitude", "level"], pressure_data),
                "geopotential": (
                    ["time", "latitude", "longitude", "level"],
                    geopotential_data,
                ),
                "eastward_wind": (
                    ["time", "latitude", "longitude", "level"],
                    wind_data,
                ),
                "northward_wind": (
                    ["time", "latitude", "longitude", "level"],
                    wind_data,
                ),
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    wind_data[:, :, :, 0],
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    wind_data[:, :, :, 0],
                ),
            },
            coords={
                "time": [0],
                "latitude": [0],
                "longitude": [0],
                "level": pressure_levels,
            },
        )

    def test_craven_brooks_with_realistic_shear(self, regression_profile_data):
        """Test Craven-Brooks significant severe parameter with realistic wind shear."""
        # Add realistic wind shear to the dataset
        shear_u = np.linspace(5, 25, 31)  # 5 to 25 m/s eastward wind
        shear_v = np.linspace(2, 10, 31)  # 2 to 10 m/s northward wind

        regression_profile_data["eastward_wind"] = (
            ["time", "latitude", "longitude", "level"],
            shear_u.reshape(1, 1, 1, -1),
        )
        regression_profile_data["northward_wind"] = (
            ["time", "latitude", "longitude", "level"],
            shear_v.reshape(1, 1, 1, -1),
        )
        regression_profile_data["surface_eastward_wind"] = (
            ["time", "latitude", "longitude"],
            np.array([[[5.0]]]),
        )
        regression_profile_data["surface_northward_wind"] = (
            ["time", "latitude", "longitude"],
            np.array([[[2.0]]]),
        )

        cbss = sc.craven_brooks_significant_severe(
            regression_profile_data["air_temperature"],
            regression_profile_data["dewpoint_temperature"],
            regression_profile_data["geopotential"],
            regression_profile_data["level"],
            regression_profile_data["eastward_wind"],
            regression_profile_data["northward_wind"],
            regression_profile_data["surface_eastward_wind"],
            regression_profile_data["surface_northward_wind"],
        )

        # Expected: CAPE (~2595) * Shear (~10.8 m/s) ≈ 28,026 m³/s³
        # (using sc.compute_mixed_layer_cape)
        expected_cbss = 28026
        assert np.isclose(cbss.values[0, 0, 0], expected_cbss, rtol=0.15), (
            f"CBSS mismatch: got {cbss.values[0, 0, 0]:.0f}, expected ~{expected_cbss}"
        )

        # Should be in a reasonable range for severe weather potential (> 10,000 m³/s³)
        assert cbss.values[0, 0, 0] > 10000, (
            f"CBSS should indicate severe weather potential (> 10,000), "
            f"got {cbss.values[0, 0, 0]:.0f}"
        )


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestPhysicalConstants::test_constant_values", "-v"])


class TestComputeCapeCin:
    """Test the compute_cape_cin wrapper function."""

    def test_single_profile(self):
        """Test integration with DataArrays containing a single profile."""

        # Splice in a pseudo-realistic atmosphere profile with our fixture dataset
        pressures = np.array([1000, 850, 700, 500, 300, 200])
        temp_sfc = 290.0  # K
        lapse_rate = 6.5 / 1000.0  # K/km
        geopotential = 29.3 * temp_sfc * np.log(1013.25 / pressures)
        temp_profile = temp_sfc - lapse_rate * geopotential
        dewpoint_profile = temp_profile - 5.0

        # Create xarray DataArrays
        pressures = xr.DataArray(pressures, dims=["level"], coords={"level": pressures})
        temp_profile = xr.DataArray(
            temp_profile, dims=["level"], coords={"level": pressures}
        )
        dewpoint_profile = xr.DataArray(
            dewpoint_profile, dims=["level"], coords={"level": pressures}
        )
        geopotential = xr.DataArray(
            geopotential, dims=["level"], coords={"level": pressures}
        )

        cape = sc.compute_mixed_layer_cape(
            pressures,
            temp_profile,
            dewpoint_profile,
            geopotential,
        )
        print(cape.values)
        # Check that the output has the correct shape (scalars)
        assert cape.values.shape == ()
        assert "level" not in cape.dims

        # Check that we don't have any NaNs and that all data are physically reasonable
        assert np.isfinite(cape)
        assert cape >= 0.0, "CAPE should be non-negative"

    def test_reverse_pressure_order_profile(self):
        """Test integration with DataArrays containing a single profile."""

        # Splice in a pseudo-realistic atmosphere profile with our fixture dataset
        pressures = np.array([200, 300, 500, 700, 850, 1000])
        temp_sfc = 290.0  # K
        lapse_rate = 6.5 / 1000.0  # K/km
        geopotential = 29.3 * temp_sfc * np.log(1013.25 / pressures)
        temp_profile = temp_sfc - lapse_rate * geopotential
        dewpoint_profile = temp_profile - 5.0

        # Create xarray DataArrays
        pressures = xr.DataArray(pressures, dims=["level"], coords={"level": pressures})
        temp_profile = xr.DataArray(
            temp_profile[::-1], dims=["level"], coords={"level": pressures}
        )
        dewpoint_profile = xr.DataArray(
            dewpoint_profile[::-1], dims=["level"], coords={"level": pressures}
        )
        geopotential = xr.DataArray(
            geopotential[::-1], dims=["level"], coords={"level": pressures}
        )

        cape = sc.compute_mixed_layer_cape(
            pressures,
            temp_profile,
            dewpoint_profile,
            geopotential,
        )
        print(cape.values)
        # Check that the output has the correct shape (scalars)
        assert cape.values.shape == ()
        assert "level" not in cape.dims

        # Check that we don't have any NaNs and that all data are physically reasonable
        assert np.isfinite(cape)
        assert cape == 0.0, "CAPE should be 0.0 when pressure is in reverse order"

    def test_grid_profiles(self):
        """Test integration with typical gridded NWP-like data."""

        n_times, n_lats, n_lons = 5, 10, 15

        # Splice in a pseudo-realistic atmosphere profile with our fixture dataset
        pressures = np.array([1000, 850, 700, 500, 300, 200])

        temp_sfc = 290.0 + np.random.normal(0, 1, (n_times, n_lats, n_lons, 1))  # K
        lapse_rate = 6.5 / 1000.0  # K/km
        geopotential = (
            29.3 * temp_sfc * np.log(1013.25 / pressures)
        )  # (n_times, n_lats, n_lons, n_levels)
        temp_profile = temp_sfc - lapse_rate * geopotential
        dewpoint_profile = temp_profile - 5.0

        # Create an xarray Dataset with these profiles
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "latitude", "longitude", "level"],
                    temp_profile,
                ),
                "dewpoint": (
                    ["time", "latitude", "longitude", "level"],
                    dewpoint_profile,
                ),
                "geopotential": (
                    ["time", "latitude", "longitude", "level"],
                    geopotential,
                ),
                "pressure": (["level"], pressures),
            },
            coords={
                "time": range(n_times),
                "latitude": range(n_lats),
                "longitude": range(n_lons),
                "level": pressures,
            },
        )

        # Compute CAPE and CIN for each profile
        cape = sc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
        )

        # Check that the output has the correct shape (scalars)
        assert cape.values.shape == (n_times, n_lats, n_lons)
        assert "level" not in cape.dims

        # Check that we don't have any NaNs and that all data are physically reasonable
        assert np.all(np.isfinite(cape))
        assert np.all(cape >= 0.0), "CAPE should be non-negative"

    def test_grid_profiles_with_dask(self):
        """Test integration with typical gridded NWP-like data, but backed by dask
        arrays."""

        n_times, n_lats, n_lons = 5, 4, 4

        # Splice in a pseudo-realistic atmosphere profile with our fixture dataset
        pressures = np.array([1000, 850, 700, 500, 300, 200])
        temp_sfc = 290.0 + np.random.normal(0, 1, (n_times, n_lats, n_lons, 1))  # K
        lapse_rate = 6.5 / 1000.0  # K/km
        geopotential = (
            29.3 * temp_sfc * np.log(1013.25 / pressures)
        )  # (n_times, n_lats, n_lons, n_levels)
        temp_profile = temp_sfc - lapse_rate * geopotential
        dewpoint_profile = temp_profile - 5.0

        # Create an xarray Dataset with these profiles
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "latitude", "longitude", "level"],
                    temp_profile,
                ),
                "dewpoint": (
                    ["time", "latitude", "longitude", "level"],
                    dewpoint_profile,
                ),
                "geopotential": (
                    ["time", "latitude", "longitude", "level"],
                    geopotential,
                ),
                "pressure": (["level"], pressures),
            },
            coords={
                "time": range(n_times),
                "latitude": range(n_lats),
                "longitude": range(n_lons),
                "level": pressures,
            },
        )
        ds = ds.chunk({"time": 1, "level": -1})

        # Compute CAPE for each profile
        # NOTE: Use parallel=False with Dask to avoid Numba threading conflicts.
        # Dask already provides parallelism at the chunk level, so Numba parallel
        # features would conflict with Dask's threading. In general it should be
        # safe to use parallel=True when Dask is distributing to multiple proceses;
        # here, we're keeping things very simple and creating Dask arrays in-place
        # within the pytest process.
        cape = sc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
            parallel=False,
        )

        # Check that the output has the correct shape (scalars)
        assert cape.values.shape == (n_times, n_lats, n_lons)
        assert "level" not in cape.dims

        # Check that we don't have any NaNs and that all data are physically reasonable
        assert np.all(np.isfinite(cape))
        assert np.all(cape >= 0.0), "CAPE should be non-negative"

    def test_parallel_serial_equivalence(self):
        """Test that parallel and serial CAPE calculations produce equivalent
        results."""

        n_times, n_lats, n_lons = 5, 4, 4

        # Splice in a pseudo-realistic atmosphere profile with our fixture dataset
        pressures = np.array([1000, 850, 700, 500, 300, 200])
        temp_sfc = 290.0 + np.random.normal(0, 1, (n_times, n_lats, n_lons, 1))  # K
        lapse_rate = 6.5 / 1000.0  # K/km
        geopotential = (
            29.3 * temp_sfc * np.log(1013.25 / pressures)
        )  # (n_times, n_lats, n_lons, n_levels)
        temp_profile = temp_sfc - lapse_rate * geopotential
        dewpoint_profile = temp_profile - 5.0

        # Create an xarray Dataset with these profiles
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "latitude", "longitude", "level"],
                    temp_profile,
                ),
                "dewpoint": (
                    ["time", "latitude", "longitude", "level"],
                    dewpoint_profile,
                ),
                "geopotential": (
                    ["time", "latitude", "longitude", "level"],
                    geopotential,
                ),
                "pressure": (["level"], pressures),
            },
            coords={
                "time": range(n_times),
                "latitude": range(n_lats),
                "longitude": range(n_lons),
                "level": pressures,
            },
        )

        cape_parallel = sc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
            parallel=True,
        )
        cape_serial = sc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
            parallel=False,
        )

        assert np.allclose(cape_parallel, cape_serial)
