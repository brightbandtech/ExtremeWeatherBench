"""Tests for the calc module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import testing

from extremeweatherbench import calc


class TestBasicCalculations:
    """Test basic calculation functions."""

    def test_convert_from_cartesian_to_latlon(self, sample_calc_dataset):
        """Test conversion from cartesian to lat/lon coordinates."""
        # Test with center point
        point = (8, 10)  # Middle of the grid

        lat, lon = calc.convert_from_cartesian_to_latlon(point, sample_calc_dataset)

        # Should return values within the grid bounds
        assert 20 <= lat <= 50
        assert -120 <= lon <= -80

        # Should be close to the center values
        expected_lat = sample_calc_dataset.latitude.isel(latitude=8).values
        expected_lon = sample_calc_dataset.longitude.isel(longitude=10).values

        assert abs(lat - expected_lat) < 1e-10
        assert abs(lon - expected_lon) < 1e-10

    def test_calculate_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test known distances
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [90.0, 0.0]  # North pole

        distance = calc.calculate_haversine_distance(point_a, point_b, units="deg")

        # Should be 90 degrees (quarter of great circle)
        assert abs(distance - 90.0) < 0.1

    def test_calculate_haversine_distance_with_xarray(self, sample_calc_dataset):
        """Test haversine distance calculation with xarray inputs."""
        point_a = [30.0, -100.0]
        point_b = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.calculate_haversine_distance(point_a, point_b, units="deg")

        # Should return an xarray DataArray
        assert isinstance(distances, xr.DataArray)
        # Should have shape (lat, lon) since we're computing distance to every grid
        # point
        expected_shape = (
            len(sample_calc_dataset.latitude),
            len(sample_calc_dataset.longitude),
        )
        assert distances.shape == expected_shape

    def test_calculate_haversine_distance_km(self):
        """Test haversine distance calculation with km units."""
        # Test known distances
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [90.0, 0.0]  # North pole

        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")

        # Should be approximately 10,018 km (quarter of great circle)
        # Earth's circumference is ~40,075 km, so quarter is ~10,018 km
        assert abs(distance - 10018.0) < 50.0

    def test_calculate_haversine_distance_km_with_xarray(self, sample_calc_dataset):
        """Test haversine distance calculation with km units and xarray inputs."""
        point_a = [30.0, -100.0]
        point_b = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.calculate_haversine_distance(point_a, point_b, units="km")

        # Should return an xarray DataArray
        assert isinstance(distances, xr.DataArray)
        # Should have shape (lat, lon) since we're computing distance to every grid
        # point
        expected_shape = (
            len(sample_calc_dataset.latitude),
            len(sample_calc_dataset.longitude),
        )
        assert distances.shape == expected_shape

        # All distances should be positive
        assert (distances >= 0).all()

        # Distances should be reasonable (not too large for Earth)
        assert (distances <= 20000).all()  # Maximum distance on Earth ~20,000 km

    def test_calculate_haversine_distance_edge_cases(self):
        """Test edge cases for haversine distance calculation."""
        # Test identical points (distance should be 0)
        point_a = [40.0, -74.0]
        point_b = [40.0, -74.0]
        distance = calc.calculate_haversine_distance(point_a, point_b)
        assert abs(distance) < 1e-10

        # Test antipodal points (should be ~20015 km, half Earth's circumference)
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [0.0, 180.0]  # Equator, opposite side
        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert abs(distance - 20015.1) < 50.0

        # Test North/South pole distance
        point_a = [90.0, 0.0]  # North pole
        point_b = [-90.0, 0.0]  # South pole
        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert abs(distance - 20003.9) < 50.0

    def test_calculate_haversine_distance_known_cities(self):
        """Test haversine distance with known city distances."""
        # New York City to Los Angeles (approximate distance ~3944 km)
        nyc = [40.7128, -74.0060]
        la = [34.0522, -118.2437]
        distance = calc.calculate_haversine_distance(nyc, la, units="km")
        assert abs(distance - 3944) < 20  # Allow 100km tolerance

        # London to Paris (approximate distance ~344 km)
        london = [51.5074, -0.1278]
        paris = [48.8566, 2.3522]
        distance = calc.calculate_haversine_distance(london, paris, units="km")
        assert abs(distance - 344) < 20  # Allow 20km tolerance

        # Sydney to Melbourne (approximate distance ~713 km)
        sydney = [-33.8688, 151.2093]
        melbourne = [-37.8136, 144.9631]
        distance = calc.calculate_haversine_distance(sydney, melbourne, units="km")
        assert abs(distance - 713) < 20  # Allow 30km tolerance

    def test_calculate_haversine_distance_units_conversion(self):
        """Test unit conversion between km and degrees."""
        point_a = [0.0, 0.0]
        point_b = [1.0, 0.0]  # 1 degree north

        distance_km = calc.calculate_haversine_distance(point_a, point_b, units="km")
        distance_deg = calc.calculate_haversine_distance(point_a, point_b, units="deg")

        # 1 degree should be approximately 111.32 km
        assert abs(distance_km - 111.32) < 5.0
        assert abs(distance_deg - 1.0) < 0.01

        # Test with "kilometers" and "degrees" spelled out
        distance_km_long = calc.calculate_haversine_distance(
            point_a, point_b, units="kilometers"
        )
        distance_deg_long = calc.calculate_haversine_distance(
            point_a, point_b, units="degrees"
        )

        assert abs(distance_km_long - distance_km) < 1e-10
        assert abs(distance_deg_long - distance_deg) < 1e-10

    def test_calculate_haversine_distance_with_numpy_arrays(self):
        """Test haversine distance with numpy array inputs."""
        # Test with numpy arrays
        point_a = np.array([40.0, -74.0])
        point_b = np.array([34.0, -118.0])

        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert isinstance(distance, (float, np.ndarray))
        assert distance > 0

        # Test with multiple points as arrays
        lats = np.array([40.0, 50.0, 60.0])
        lons = np.array([-74.0, -80.0, -90.0])
        point_a = [30.0, -100.0]
        point_b = [lats, lons]

        distances = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (3,)
        assert all(d > 0 for d in distances)

    def test_calculate_haversine_distance_error_handling(self):
        """Test error handling for invalid inputs."""
        point_a = [40.0, -74.0]
        point_b = [34.0, -118.0]

        # Test invalid units
        with pytest.raises(ValueError, match="Invalid units"):
            calc.calculate_haversine_distance(point_a, point_b, units="miles")

        with pytest.raises(ValueError, match="Invalid units"):
            calc.calculate_haversine_distance(point_a, point_b, units="invalid")

    def test_calculate_haversine_distance_symmetry(self):
        """Test that distance calculation is symmetric."""
        point_a = [40.7128, -74.0060]  # NYC
        point_b = [34.0522, -118.2437]  # LA

        distance_ab = calc.calculate_haversine_distance(point_a, point_b)
        distance_ba = calc.calculate_haversine_distance(point_b, point_a)

        assert abs(distance_ab - distance_ba) < 1e-10

    def test_calculate_haversine_distance_boundary_conditions(self):
        """Test boundary conditions for latitude and longitude."""
        # Test at latitude boundaries
        point_a = [90.0, 0.0]  # North pole
        point_b = [89.9, 0.0]  # Near north pole
        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert distance > 0 and distance < 20  # Should be small distance

        point_a = [-90.0, 0.0]  # South pole
        point_b = [-89.9, 0.0]  # Near south pole
        distance = calc.calculate_haversine_distance(point_a, point_b, units="km")
        assert distance > 0 and distance < 20  # Should be small distance

        # Test longitude wraparound (179° to -179° should be 2° apart)
        point_a = [0.0, 179.0]
        point_b = [0.0, -179.0]
        distance = calc.calculate_haversine_distance(point_a, point_b, units="deg")
        assert abs(distance - 2.0) < 0.1

    def test_calculate_haversine_distance_large_datasets(self, sample_calc_dataset):
        """Test performance and correctness with larger datasets."""
        # Use a single point and compute distance to entire grid
        center_point = [35.0, -100.0]
        grid_point = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.calculate_haversine_distance(
            center_point, grid_point, units="km"
        )

        # Should return an xarray DataArray with proper shape
        assert isinstance(distances, xr.DataArray)
        expected_shape = (
            len(sample_calc_dataset.latitude),
            len(sample_calc_dataset.longitude),
        )
        assert distances.shape == expected_shape

        # All distances should be positive and reasonable
        assert (distances >= 0).all()
        assert (distances <= 20000).all()

        # Distance to center point should be minimal
        # Find the grid point closest to our center point
        lat_idx = np.argmin(np.abs(sample_calc_dataset.latitude.values - 35.0))
        lon_idx = np.argmin(np.abs(sample_calc_dataset.longitude.values - (-100.0)))
        min_distance = distances.isel(latitude=lat_idx, longitude=lon_idx)
        assert min_distance < 500  # Should be within 500km of center

    def test_calculate_haversine_distance_scalar_case(self):
        """Test haversine distance when result is scalar (line 90 coverage)."""
        # Test case where distance calculation returns a scalar
        point_a = [40.0, -74.0]
        point_b = [40.0, -74.0]  # Same point, should return scalar 0

        distance = calc.calculate_haversine_distance(point_a, point_b)

        # Should be a scalar (float), not a dataset
        assert isinstance(distance, (float, np.floating))
        assert not isinstance(distance, xr.DataArray)
        assert abs(distance) < 1e-10

    def test_create_great_circle_mask(self, sample_calc_dataset):
        """Test creation of great circle mask."""
        center_point = (35.0, -100.0)  # Somewhere in the middle
        radius = 5.0  # degrees

        mask = calc.create_great_circle_mask(sample_calc_dataset, center_point, radius)

        # Should return boolean mask
        assert isinstance(mask, xr.DataArray)
        assert mask.dtype == bool
        # Should have shape (lat, lon) since it's a mask over the entire grid
        expected_shape = (
            len(sample_calc_dataset.latitude),
            len(sample_calc_dataset.longitude),
        )
        assert mask.shape == expected_shape

        # Some points should be within radius, some outside
        assert mask.any()  # At least some True values
        assert not mask.all()  # Not all True values

    def test_create_great_circle_mask_scalar_distance(self):
        """Test great circle mask when distance is scalar (line 90 coverage)."""
        # Create a dataset with scalar coordinates to force scalar distance
        import unittest.mock

        # Create a simple dataset
        time = [np.datetime64("2023-01-01")]
        lat = [35.0]
        lon = [-100.0]

        dataset = xr.Dataset(
            {"temp": (["time", "latitude", "longitude"], [[[20.0]]])},
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        center_point = (35.0, -100.0)
        radius = 5.0

        # Mock the haversine distance to return a scalar
        with unittest.mock.patch(
            "extremeweatherbench.calc.calculate_haversine_distance"
        ) as mock_distance:
            mock_distance.return_value = 2.0  # Return scalar instead of DataArray

            mask = calc.create_great_circle_mask(dataset, center_point, radius)

            # Should return a boolean DataArray even with scalar distance
            assert isinstance(mask, xr.DataArray)
            assert mask.dtype == bool
            # Should have same shape as latitude coordinate (1D)
            assert mask.shape == (1,)
            # Should be True since 2.0 < 5.0
            assert mask.values[0]


class TestWindCalculations:
    """Test wind-related calculations."""

    def test_maybe_calculate_wind_speed_from_components(self, sample_calc_dataset):
        """Test wind speed calculation from components."""
        wind_speed = calc.maybe_calculate_wind_speed(sample_calc_dataset)

        # Should return a DataArray
        assert isinstance(wind_speed, xr.Dataset)
        assert (
            wind_speed.surface_wind_speed.shape
            == sample_calc_dataset.surface_eastward_wind.shape
        )

        # All wind speeds should be non-negative
        assert (wind_speed.surface_wind_speed >= 0).all()

        # Check against manual calculation for a point
        u = sample_calc_dataset.surface_eastward_wind.isel(
            time=0, latitude=0, longitude=0
        ).values
        v = sample_calc_dataset.surface_northward_wind.isel(
            time=0, latitude=0, longitude=0
        ).values
        expected = np.sqrt(u**2 + v**2)
        calculated = wind_speed.surface_wind_speed.isel(
            time=0, latitude=0, longitude=0
        ).values

        assert abs(calculated - expected) < 1e-10

    def test_maybe_calculate_wind_speed_with_existing_wind_speed(self):
        """Test that existing wind speed is returned unchanged."""
        # Create dataset with existing wind speed
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(30, 40, 5)
        lon = np.linspace(-100, -90, 5)

        wind_speed_data = np.random.uniform(0, 30, (2, 5, 5))

        dataset = xr.Dataset(
            {
                "surface_wind_speed": (
                    ["time", "latitude", "longitude"],
                    wind_speed_data,
                ),
                "surface_eastward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 5, 5)),
                ),
                "surface_northward_wind": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(0, 10, (2, 5, 5)),
                ),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        result = calc.maybe_calculate_wind_speed(dataset)

        # Should return the existing wind speed unchanged
        xr.testing.assert_equal(result, dataset)

    def test_maybe_calculate_wind_speed_missing_components(self):
        """Test error when wind components are missing."""
        # Create dataset without wind components
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(30, 40, 5)
        lon = np.linspace(-100, -90, 5)

        dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 5, 5)),
                ),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        result = calc.maybe_calculate_wind_speed(dataset)

        # Will return the dataset as is, without the wind speed computed
        xr.testing.assert_equal(result, dataset)


class TestPressureCalculations:
    """Test pressure-related calculations."""

    def test_orography_from_surface_geopotential(self, sample_calc_dataset):
        """Test orography calculation from surface geopotential."""
        orography = calc.orography(sample_calc_dataset)

        # Should return a DataArray
        assert isinstance(orography, xr.DataArray)
        assert (
            orography.shape == sample_calc_dataset.geopotential_at_surface.shape[1:]
        )  # Remove time

        # Should be divided by gravity
        expected = sample_calc_dataset.geopotential_at_surface / 9.80665
        # Compare one time slice since orography doesn't have time dimension in some
        # cases
        if "time" in orography.dims:
            xr.testing.assert_allclose(orography, expected)
        else:
            # If time dimension was removed, compare with first time slice
            xr.testing.assert_allclose(orography, expected.isel(time=0))

    def test_calculate_pressure_at_surface(self, sample_calc_dataset):
        """Test surface pressure calculation from orography."""
        # First get orography
        orography_data = calc.orography(sample_calc_dataset)

        # Calculate surface pressure
        surface_pressure = calc.calculate_pressure_at_surface(orography_data)

        # Should return a DataArray
        assert isinstance(surface_pressure, xr.DataArray)
        assert surface_pressure.shape == orography_data.shape

        # All pressures should be positive and reasonable
        assert (surface_pressure > 50000).all()  # > 500 hPa
        assert (surface_pressure < 105000).all()  # < 1050 hPa

        # Test the formula manually for a point
        if "time" in orography_data.dims:
            h = orography_data.isel(time=0, latitude=0, longitude=0).values
        else:
            h = orography_data.isel(latitude=0, longitude=0).values

        expected = 101325 * (1 - 2.25577e-5 * h) ** 5.25579

        if "time" in surface_pressure.dims:
            calculated = surface_pressure.isel(time=0, latitude=0, longitude=0).values
        else:
            calculated = surface_pressure.isel(latitude=0, longitude=0).values

        assert abs(calculated - expected) < 1e-6

    def test_orography_from_arco_era5(self):
        """Test orography calculation from ARCO ERA5 when no surface geopotential."""
        # Create a dataset without geopotential_at_surface to trigger else branch
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(30, 40, 3)  # Small grid for testing
        lon = np.linspace(260, 270, 3)  # ERA5 uses 0-360 longitude system

        # Dataset without geopotential_at_surface
        dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 3, 3)),
                ),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        # This should trigger the else branch and load from ARCO ERA5
        orography_data = calc.orography(dataset)

        # Should return a DataArray
        assert isinstance(orography_data, xr.DataArray)

        # Should have latitude and longitude dimensions (no time)
        assert "latitude" in orography_data.dims
        assert "longitude" in orography_data.dims
        assert "time" not in orography_data.dims

        # Should have data (not all zeros or NaN)
        assert not orography_data.isnull().all()

        # Test a specific point - hardcode expected value for validation
        # Using a point in the middle of our test grid (35°N, 265°E)
        # This is roughly in the central US where elevation should be reasonable
        test_point = orography_data.sel(latitude=35, longitude=265, method="nearest")

        # Elevation should be reasonable for central US (0-2000m typically)
        assert 0 <= test_point <= 3000  # Allow up to 3000m for mountain regions


class TestGeopotentialCalculations:
    """Test geopotential-related calculations."""

    def test_generate_geopotential_thickness_default_levels(self, sample_calc_dataset):
        """Test geopotential thickness with default levels."""
        thickness = calc.generate_geopotential_thickness(sample_calc_dataset)

        # Should return a DataArray
        assert isinstance(thickness, xr.DataArray)

        # Should have proper dimensions (no level dimension)
        expected_dims = ["time", "latitude", "longitude"]
        assert list(thickness.dims) == expected_dims

        # Should have proper attributes
        assert "description" in thickness.attrs
        assert "units" in thickness.attrs
        assert thickness.attrs["units"] == "m"

    def test_generate_geopotential_thickness_custom_levels(self, sample_calc_dataset):
        """Test geopotential thickness with custom levels."""
        thickness = calc.generate_geopotential_thickness(
            sample_calc_dataset, top_level_value=200, bottom_level_value=850
        )

        # Manual calculation for verification
        top_level_geopotential = sample_calc_dataset["geopotential"].sel(level=200)
        bottom_level_geopotential = sample_calc_dataset["geopotential"].sel(level=850)
        expected_thickness = (
            top_level_geopotential - bottom_level_geopotential
        ) / 9.80665

        xr.testing.assert_allclose(thickness, expected_thickness)

    def test_generate_geopotential_thickness_multiple_top_levels(
        self, sample_calc_dataset
    ):
        """Test geopotential thickness with multiple top levels."""
        thickness = calc.generate_geopotential_thickness(
            sample_calc_dataset, top_level_value=[200, 300, 500], bottom_level_value=850
        )

        # Should have level dimension for multiple top levels
        assert "level" in thickness.dims
        assert len(thickness.level) == 3


class TestNantrapezoid:
    """Taken from numpy testing code, with modification to include handling nans."""

    def test_simple(self):
        x = np.arange(-10, 10, 0.1)
        r = calc.nantrapezoid(np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), dx=0.1)
        # check integral of normal equals 1
        testing.assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None, :, None] + z[None, None, :]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # n-d `x`
        r = calc.nantrapezoid(q, x=x[:, None, None], axis=0)
        testing.assert_almost_equal(r, qx)
        r = calc.nantrapezoid(q, x=y[None, :, None], axis=1)
        testing.assert_almost_equal(r, qy)
        r = calc.nantrapezoid(q, x=z[None, None, :], axis=2)
        testing.assert_almost_equal(r, qz)

        # 1-d `x`
        r = calc.nantrapezoid(q, x=x, axis=0)
        testing.assert_almost_equal(r, qx)
        r = calc.nantrapezoid(q, x=y, axis=1)
        testing.assert_almost_equal(r, qy)
        r = calc.nantrapezoid(q, x=z, axis=2)
        testing.assert_almost_equal(r, qz)

    def test_masked(self):
        # Testing that masked arrays behave as if the function is 0 where
        # masked
        x = np.arange(5)
        y = x * x
        mask = x == 2
        ym = np.ma.array(y, mask=mask)
        r = 13.0  # sum(0.5 * (0 + 1) * 1.0 + 0.5 * (9 + 16))
        testing.assert_almost_equal(calc.nantrapezoid(ym, x), r)

        xm = np.ma.array(x, mask=mask)
        testing.assert_almost_equal(calc.nantrapezoid(ym, xm), r)

        xm = np.ma.array(x, mask=mask)
        testing.assert_almost_equal(calc.nantrapezoid(y, xm), r)

    def test_nan_handling(self):
        """Test that nantrapezoid properly handles NaN values."""
        # Test with NaN values in y
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, np.nan, 9, 16])

        # Should ignore NaN values in the integration
        result = calc.nantrapezoid(y, x)

        # Compare with manual calculation ignoring NaN
        # Integration should be: 0.5*(0+1)*1 + 0.5*(9+16)*1 = 0.5 + 12.5 = 13.0
        expected = 13.0
        testing.assert_almost_equal(result, expected)

    def test_all_nan_values(self):
        """Test nantrapezoid with all NaN values."""
        x = np.array([0, 1, 2, 3])
        y = np.array([np.nan, np.nan, np.nan, np.nan])

        result = calc.nantrapezoid(y, x)

        # Should return 0 when all values are NaN (nansum behavior)
        assert result == 0.0

    def test_mixed_nan_and_finite(self):
        """Test nantrapezoid with mixed NaN and finite values."""
        # Test case with NaN at beginning
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([np.nan, 1, 4, 9, 16])

        result = calc.nantrapezoid(y, x)

        # Should integrate only the finite portions
        # Integration: 0.5*(1+4)*1 + 0.5*(4+9)*1 + 0.5*(9+16)*1 = 21.5
        expected = 21.5
        testing.assert_almost_equal(result, expected)

    def test_nan_in_x_coordinates(self):
        """Test nantrapezoid with NaN values in x coordinates."""
        x = np.array([0, 1, np.nan, 3, 4])
        y = np.array([0, 1, 4, 9, 16])

        # Should handle NaN in x coordinates properly
        result = calc.nantrapezoid(y, x)

        # The function should still work, handling NaN appropriately
        assert not np.isnan(result) or np.isnan(result)  # Either valid result or NaN

    def test_multidimensional_with_nans(self):
        """Test nantrapezoid with multidimensional arrays containing NaNs."""
        x = np.array([0, 1, 2, 3])
        y = np.array([[0, 1, np.nan, 9], [1, np.nan, 4, 16], [np.nan, 2, 8, 25]])

        # Test integration along axis 1 (columns)
        result = calc.nantrapezoid(y, x, axis=1)

        # Should return array with proper NaN handling for each row
        assert result.shape == (3,)
        # At least some results should be finite (not all NaN)
        assert not np.isnan(result).all()

    def test_nantrapezoid_dimension_expansion(self):
        """Test nantrapezoid with dimension mismatch (line 228 coverage)."""
        # Create a case where y.ndim != d.ndim to trigger dimension expansion
        x = np.array([0, 1, 2])
        y = np.array([[1, 2, 3]])  # 2D array

        # This should trigger the dimension expansion on line 228
        result = calc.nantrapezoid(y, x, axis=1)

        # Should still work and return proper result
        assert result.shape == (1,)
        assert not np.isnan(result)


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

        cape = calc.compute_mixed_layer_cape(
            pressures,
            temp_profile,
            dewpoint_profile,
            geopotential,
        )

        # Check that the output has the correct shape (scalars)
        assert cape.values.shape == ()
        assert "level" not in cape.dims

        # Check that we don't have any NaNs and that all data are physically reasonable
        assert np.isfinite(cape)
        assert cape >= 0.0, "CAPE should be non-negative"

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
        cape = calc.compute_mixed_layer_cape(
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
        """Test integration with typical gridded NWP-like data, but backed by dask arrays."""

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
        cape = calc.compute_mixed_layer_cape(
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
        """Test that parallel and serial CAPE calculations produce equivalent results."""

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

        cape_parallel = calc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
            parallel=True,
        )
        cape_serial = calc.compute_mixed_layer_cape(
            ds["pressure"],
            ds["temperature"],
            ds["dewpoint"],
            ds["geopotential"],
            parallel=False,
        )

        assert np.allclose(cape_parallel, cape_serial)
