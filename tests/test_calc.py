import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc


class TestBasicCalculations:
    """Test basic calculation functions."""

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
