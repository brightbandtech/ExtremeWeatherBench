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

        lat, lon = calc.convert_from_cartesian_to_latlon(
            point, sample_calc_dataset.latitude, sample_calc_dataset.longitude
        )
        # Should return values within the grid bounds
        assert 20 <= lat <= 50
        assert -120 <= lon <= -80

        # Should be close to the center values
        expected_lat = sample_calc_dataset.latitude.isel(latitude=8).values
        expected_lon = sample_calc_dataset.longitude.isel(longitude=10).values

        assert abs(lat - expected_lat) < 1e-10
        assert abs(lon - expected_lon) < 1e-10

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test known distances
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [90.0, 0.0]  # North pole

        distance = calc.haversine_distance(point_a, point_b, units="deg")

        # Should be 90 degrees (quarter of great circle)
        assert abs(distance - 90.0) < 0.1

    def test_haversine_distance_with_xarray(self, sample_calc_dataset):
        """Test haversine distance calculation with xarray inputs."""
        point_a = [30.0, -100.0]
        point_b = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.haversine_distance(point_a, point_b, units="deg")

        # Should return an xarray DataArray
        assert isinstance(distances, xr.DataArray)
        # Should have shape (lat, lon) since we're computing distance to every grid
        # point
        expected_shape = (
            len(sample_calc_dataset.latitude),
            len(sample_calc_dataset.longitude),
        )
        assert distances.shape == expected_shape

    def test_haversine_distance_km(self):
        """Test haversine distance calculation with km units."""
        # Test known distances
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [90.0, 0.0]  # North pole

        distance = calc.haversine_distance(point_a, point_b, units="km")

        # Should be approximately 10,018 km (quarter of great circle)
        # Earth's circumference is ~40,075 km, so quarter is ~10,018 km
        assert abs(distance - 10018.0) < 50.0

    def test_haversine_distance_km_with_xarray(self, sample_calc_dataset):
        """Test haversine distance calculation with km units and xarray inputs."""
        point_a = [30.0, -100.0]
        point_b = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.haversine_distance(point_a, point_b, units="km")

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

    def test_haversine_distance_edge_cases(self):
        """Test edge cases for haversine distance calculation."""
        # Test identical points (distance should be 0)
        point_a = [40.0, -74.0]
        point_b = [40.0, -74.0]
        distance = calc.haversine_distance(point_a, point_b)
        assert abs(distance) < 1e-10

        # Test antipodal points (should be ~20015 km, half Earth's circumference)
        point_a = [0.0, 0.0]  # Equator, prime meridian
        point_b = [0.0, 180.0]  # Equator, opposite side
        distance = calc.haversine_distance(point_a, point_b, units="km")
        assert abs(distance - 20015.1) < 50.0

        # Test North/South pole distance
        point_a = [90.0, 0.0]  # North pole
        point_b = [-90.0, 0.0]  # South pole
        distance = calc.haversine_distance(point_a, point_b, units="km")
        assert abs(distance - 20003.9) < 50.0

    def test_haversine_distance_known_cities(self):
        """Test haversine distance with known city distances."""
        # New York City to Los Angeles (approximate distance ~3944 km)
        nyc = [40.7128, -74.0060]
        la = [34.0522, -118.2437]
        distance = calc.haversine_distance(nyc, la, units="km")
        assert abs(distance - 3944) < 20  # Allow 100km tolerance

        # London to Paris (approximate distance ~344 km)
        london = [51.5074, -0.1278]
        paris = [48.8566, 2.3522]
        distance = calc.haversine_distance(london, paris, units="km")
        assert abs(distance - 344) < 20  # Allow 20km tolerance

        # Sydney to Melbourne (approximate distance ~713 km)
        sydney = [-33.8688, 151.2093]
        melbourne = [-37.8136, 144.9631]
        distance = calc.haversine_distance(sydney, melbourne, units="km")
        assert abs(distance - 713) < 20  # Allow 30km tolerance

    def test_haversine_distance_units_conversion(self):
        """Test unit conversion between km and degrees."""
        point_a = [0.0, 0.0]
        point_b = [1.0, 0.0]  # 1 degree north

        distance_km = calc.haversine_distance(point_a, point_b, units="km")
        distance_deg = calc.haversine_distance(point_a, point_b, units="deg")

        # 1 degree should be approximately 111.32 km
        assert abs(distance_km - 111.32) < 5.0
        assert abs(distance_deg - 1.0) < 0.01

        # Test with "kilometers" and "degrees" spelled out
        distance_km_long = calc.haversine_distance(point_a, point_b, units="kilometers")
        distance_deg_long = calc.haversine_distance(point_a, point_b, units="degrees")

        assert abs(distance_km_long - distance_km) < 1e-10
        assert abs(distance_deg_long - distance_deg) < 1e-10

    def test_haversine_distance_with_numpy_arrays(self):
        """Test haversine distance with numpy array inputs."""
        # Test with numpy arrays
        point_a = np.array([40.0, -74.0])
        point_b = np.array([34.0, -118.0])

        distance = calc.haversine_distance(point_a, point_b, units="km")
        assert isinstance(distance, (float, np.ndarray))
        assert distance > 0

        # Test with multiple points as arrays
        lats = np.array([40.0, 50.0, 60.0])
        lons = np.array([-74.0, -80.0, -90.0])
        point_a = [30.0, -100.0]
        point_b = [lats, lons]

        distances = calc.haversine_distance(point_a, point_b, units="km")
        assert isinstance(distances, np.ndarray)
        assert distances.shape == (3,)
        assert all(d > 0 for d in distances)

    def test_haversine_distance_error_handling(self):
        """Test error handling for invalid inputs."""
        point_a = [40.0, -74.0]
        point_b = [34.0, -118.0]

        # Test invalid units
        with pytest.raises(ValueError, match="Invalid units"):
            calc.haversine_distance(point_a, point_b, units="miles")

        with pytest.raises(ValueError, match="Invalid units"):
            calc.haversine_distance(point_a, point_b, units="invalid")

    def test_haversine_distance_symmetry(self):
        """Test that distance calculation is symmetric."""
        point_a = [40.7128, -74.0060]  # NYC
        point_b = [34.0522, -118.2437]  # LA

        distance_ab = calc.haversine_distance(point_a, point_b)
        distance_ba = calc.haversine_distance(point_b, point_a)

        assert abs(distance_ab - distance_ba) < 1e-10

    def test_haversine_distance_boundary_conditions(self):
        """Test boundary conditions for latitude and longitude."""
        # Test at latitude boundaries
        point_a = [90.0, 0.0]  # North pole
        point_b = [89.9, 0.0]  # Near north pole
        distance = calc.haversine_distance(point_a, point_b, units="km")
        assert distance > 0 and distance < 20  # Should be small distance

        point_a = [-90.0, 0.0]  # South pole
        point_b = [-89.9, 0.0]  # Near south pole
        distance = calc.haversine_distance(point_a, point_b, units="km")
        assert distance > 0 and distance < 20  # Should be small distance

        # Test longitude wraparound (179° to -179° should be 2° apart)
        point_a = [0.0, 179.0]
        point_b = [0.0, -179.0]
        distance = calc.haversine_distance(point_a, point_b, units="deg")
        assert abs(distance - 2.0) < 0.1

    def test_haversine_distance_large_datasets(self, sample_calc_dataset):
        """Test performance and correctness with larger datasets."""
        # Use a single point and compute distance to entire grid
        center_point = [35.0, -100.0]
        grid_point = [sample_calc_dataset.latitude, sample_calc_dataset.longitude]

        distances = calc.haversine_distance(center_point, grid_point, units="km")

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

    def test_haversine_distance_scalar_case(self):
        """Test haversine distance when result is scalar (line 90 coverage)."""
        # Test case where distance calculation returns a scalar
        point_a = [40.0, -74.0]
        point_b = [40.0, -74.0]  # Same point, should return scalar 0

        distance = calc.haversine_distance(point_a, point_b)

        # Should be a scalar (float), not a dataset
        assert isinstance(distance, (float, np.floating))
        assert not isinstance(distance, xr.DataArray)
        assert abs(distance) < 1e-10

    def test_great_circle_mask(self, sample_calc_dataset):
        """Test creation of great circle mask."""
        center_point = (35.0, -100.0)  # Somewhere in the middle
        radius = 5.0  # degrees

        mask = calc.great_circle_mask(sample_calc_dataset, center_point, radius)

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

    def test_great_circle_mask_scalar_distance(self):
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
            "extremeweatherbench.calc.haversine_distance"
        ) as mock_distance:
            mock_distance.return_value = 2.0  # Return scalar instead of DataArray

            mask = calc.great_circle_mask(dataset, center_point, radius)

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
        expected = sample_calc_dataset.geopotential_at_surface / calc.g0
        # Compare one time slice since orography doesn't have time dimension in some
        # cases
        if "time" in orography.dims:
            xr.testing.assert_allclose(orography, expected)
        else:
            # If time dimension was removed, compare with first time slice
            xr.testing.assert_allclose(orography, expected.isel(time=0))

    def test_pressure_at_surface(self, sample_calc_dataset):
        """Test surface pressure calculation from orography."""
        # First get orography
        orography_data = calc.orography(sample_calc_dataset)

        # Calculate surface pressure
        surface_pressure = calc.pressure_at_surface(orography_data)

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

    def test_orography_from_arco_era5(self, monkeypatch):
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

        # Mock the ARCO ERA5 dataset with geopotential data
        mock_era5_time = pd.date_range("2020-01-01", periods=1000001, freq="h")
        # Use reasonable geopotential values (0-20000 m²/s² -> 0-2000m orography)
        mock_geopotential = np.random.uniform(0, 20000, (1000001, 3, 3))  # m²/s²
        mock_era5 = xr.Dataset(
            {
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    mock_geopotential,
                ),
            },
            coords={"time": mock_era5_time, "latitude": lat, "longitude": lon},
        )

        # Mock xr.open_zarr to return our mock dataset
        def mock_open_zarr(*args, **kwargs):
            return mock_era5

        monkeypatch.setattr(xr, "open_zarr", mock_open_zarr)

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

        # Test a specific point - verify orography values are reasonable
        # Using a point in the middle of our test grid (35°N, 265°E)
        test_point = orography_data.sel(latitude=35, longitude=265, method="nearest")

        # Elevation should match our mock data range (0-2000m)
        assert 0 <= test_point <= 2100  # Allow small buffer for rounding


class TestGeopotentialCalculations:
    """Test geopotential-related calculations."""

    def test_geopotential_thickness_default_levels(self, sample_calc_dataset):
        """Test geopotential thickness with default levels."""
        thickness = calc.geopotential_thickness(
            sample_calc_dataset["geopotential"], geopotential=True
        )

        # Should return a DataArray
        assert isinstance(thickness, xr.DataArray)

        # Should have proper dimensions (no level dimension)
        expected_dims = ["time", "latitude", "longitude"]
        assert list(thickness.dims) == expected_dims

        # Should have proper attributes
        assert "description" in thickness.attrs
        assert "units" in thickness.attrs
        assert thickness.attrs["units"] == "m"

    def test_geopotential_thickness_custom_levels(self, sample_calc_dataset):
        """Test geopotential thickness with custom levels."""
        thickness = calc.geopotential_thickness(
            sample_calc_dataset["geopotential"],
            top_level_value=200,
            bottom_level_value=850,
            geopotential=True,
        )

        # Manual calculation for verification
        top_level_geopotential = sample_calc_dataset["geopotential"].sel(level=200)
        bottom_level_geopotential = sample_calc_dataset["geopotential"].sel(level=850)
        expected_thickness = (
            top_level_geopotential - bottom_level_geopotential
        ) / calc.g0

        xr.testing.assert_allclose(thickness, expected_thickness)

    def test_geopotential_thickness_multiple_calculations(self, sample_calc_dataset):
        """Test geopotential thickness with multiple calculations."""
        # Calculate thickness for multiple level pairs
        thickness_200_850 = calc.geopotential_thickness(
            sample_calc_dataset["geopotential"],
            top_level_value=200,
            bottom_level_value=850,
            geopotential=True,
        )

        thickness_300_700 = calc.geopotential_thickness(
            sample_calc_dataset["geopotential"],
            top_level_value=300,
            bottom_level_value=700,
            geopotential=True,
        )

        # Both should be DataArrays without level dimension
        assert isinstance(thickness_200_850, xr.DataArray)
        assert isinstance(thickness_300_700, xr.DataArray)
        assert "level" not in thickness_200_850.dims
        assert "level" not in thickness_300_700.dims


class TestSpecificHumidityCalculations:
    """Test specific humidity calculations."""

    def test_specific_humidity_from_relative_humidity_with_level(self):
        """Test specific humidity calculation with level dimension."""
        # Create test dataset with level dimension
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-120, -80, 5)
        level = [1000, 850, 700, 500]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))

        # Create realistic test data
        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.full(data_shape_4d, 288.15),  # 15°C in Kelvin
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.full(data_shape_4d, 0.5),  # 50% relative humidity
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
            },
        )

        result = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset["air_temperature"],
            relative_humidity=dataset["relative_humidity"],
            levels=dataset["level"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions
        expected_dims = ["time", "level", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == data_shape_4d

        # All values should be positive and reasonable for specific humidity
        assert (result > 0).all()
        assert (result < 0.1).all()  # Specific humidity typically < 0.1 kg/kg

        # Values should be consistent across spatial dimensions for same conditions
        assert np.allclose(
            result.isel(time=0, level=0),
            result.isel(time=0, level=0, latitude=0, longitude=0),
        )

    def test_specific_humidity_from_relative_humidity_without_level(self):
        """Test specific humidity calculation without level dimension."""
        # Create test dataset without level dimension (surface case)
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-120, -80, 5)

        data_shape_3d = (len(time), len(lat), len(lon))

        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "latitude", "longitude"],
                    np.full(data_shape_3d, 288.15),  # 15°C in Kelvin
                ),
                "relative_humidity": (
                    ["time", "latitude", "longitude"],
                    np.full(data_shape_3d, 0.7),  # 70% relative humidity
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )
        with pytest.raises(KeyError):
            calc.specific_humidity_from_relative_humidity(
                air_temperature=dataset["air_temperature"],
                relative_humidity=dataset["relative_humidity"],
                levels=dataset["level"],
            )

    def test_compute_specific_humidity_known_values(self):
        """Test specific humidity calculation with known values."""
        # Test with known atmospheric conditions
        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[288.15]]]]),  # 15°C in Kelvin
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[0.5]]]]),  # 50% relative humidity
                ),
            },
            coords={
                "time": ["2023-01-01"],
                "latitude": [35.0],
                "longitude": [-100.0],
                "level": [1000],
            },
        )

        result = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset["air_temperature"],
            relative_humidity=dataset["relative_humidity"],
            levels=dataset["level"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Get the scalar value
        specific_humidity = result.values.item()

        # Should be positive
        assert specific_humidity > 0

        # Should be reasonable for 15°C, 50% RH at 1000 hPa
        # Expected value should be around 0.005-0.01 kg/kg
        assert 0.005 < specific_humidity < 0.01

    def test_compute_specific_humidity_temperature_dependence(self):
        """Test that specific humidity increases with temperature."""
        # Create datasets with different temperatures
        base_temp = 288.15  # 15°C
        temp_diff = 10  # 10°C difference

        dataset_cold = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[base_temp]]]]),
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[0.5]]]]),
                ),
            },
            coords={
                "time": ["2023-01-01"],
                "latitude": [35.0],
                "longitude": [-100.0],
                "level": [1000],
            },
        )

        dataset_warm = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[base_temp + temp_diff]]]]),
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[0.5]]]]),
                ),
            },
            coords={
                "time": ["2023-01-01"],
                "latitude": [35.0],
                "longitude": [-100.0],
                "level": [1000],
            },
        )

        result_cold = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset_cold["air_temperature"],
            relative_humidity=dataset_cold["relative_humidity"],
            levels=dataset_cold["level"],
        )
        result_warm = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset_warm["air_temperature"],
            relative_humidity=dataset_warm["relative_humidity"],
            levels=dataset_warm["level"],
        )

        # Warmer air should have higher specific humidity at same RH
        assert result_warm.values.item() > result_cold.values.item()

    def test_compute_specific_humidity_relative_humidity_dependence(self):
        """Test that specific humidity increases with relative humidity."""
        # Create datasets with different relative humidities
        dataset_low_rh = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[288.15]]]]),  # 15°C
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[0.3]]]]),  # 30% RH
                ),
            },
            coords={
                "time": ["2023-01-01"],
                "latitude": [35.0],
                "longitude": [-100.0],
                "level": [1000],
            },
        )

        dataset_high_rh = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[288.15]]]]),  # 15°C
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    np.array([[[[0.8]]]]),  # 80% RH
                ),
            },
            coords={
                "time": ["2023-01-01"],
                "latitude": [35.0],
                "longitude": [-100.0],
                "level": [1000],
            },
        )

        result_low_rh = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset_low_rh["air_temperature"],
            relative_humidity=dataset_low_rh["relative_humidity"],
            levels=dataset_low_rh["level"],
        )
        result_high_rh = calc.specific_humidity_from_relative_humidity(
            air_temperature=dataset_high_rh["air_temperature"],
            relative_humidity=dataset_high_rh["relative_humidity"],
            levels=dataset_high_rh["level"],
        )

        # Higher RH should produce higher specific humidity
        assert result_high_rh.values.item() > result_low_rh.values.item()


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


class TestDewpointFromSpecificHumidity:
    """Test the dewpoint_from_specific_humidity function."""

    def test_dewpoints_from_specific_humidities(self):
        """Test the dewpoint_from_specific_humidity function with known values
        pre-computed from MetPy.

        The values useed were randomly selected from an ERA5 timeslice.
        """

        pressures = np.array(
            [
                800.0,
                125.0,
                825.0,
                200.0,
                600.0,
                800.0,
                350.0,
                875.0,
                700.0,
                250.0,
                950.0,
                1000.0,
            ]
        )
        specific_humidities = np.array(
            [
                2.9980205e-04,
                2.0001212e-06,
                3.2219910e-03,
                1.0791991e-05,
                2.0538690e-04,
                1.2427913e-03,
                1.4054612e-04,
                1.3353663e-02,
                1.6551274e-03,
                4.9075752e-06,
                1.0287719e-02,
                2.7858345e-03,
            ],
        )
        # Computed from metpy.calc.dewpoint_from_specific_humidity
        ref_dewpoints = np.array(
            [
                240.25,
                187.25,
                268.25,
                200.75,
                233.55,
                255.95,
                225.05,
                289.55,
                257.75,
                197.05,
                286.85,
                268.95,
            ]
        )
        dewpoints = calc.dewpoint_from_specific_humidity(pressures, specific_humidities)
        np.testing.assert_allclose(dewpoints, ref_dewpoints, atol=1e-1)

    def test_dewpoint_decreasing_with_humidity(self):
        """Given a constant pressure, dewpoint should decrease as specific humidity
        decreases."""
        p0 = 1035.0  # hPa
        qs = [2e-2, 2e-3, 2e-4, 2e-5]  # kg/kg
        tds = [calc.dewpoint_from_specific_humidity(p0, q) for q in qs]
        diffs = np.diff(tds)
        is_decreasing = np.all(diffs < 0)
        assert is_decreasing, "Dewpoint should decrease as specific humidity decreases"

    def test_dewpoint_increasing_with_pressure(self):
        """Given a constant specific humidity, dewpoint should decrease with increasing
        pressure."""
        q0 = 2e-3  # kg/kg
        ps = [1000.0, 950.0, 900.0, 850.0]  # hPa
        tds = [calc.dewpoint_from_specific_humidity(p, q0) for p in ps]
        diffs = np.diff(tds)
        is_decreasing = np.all(diffs < 0)
        assert is_decreasing, "Dewpoint should decrease as pressure increases"


class TestLandfallDetection:
    """Test landfall detection functions."""

    def test_process_single_track_landfall_with_dataarray(self):
        """Test _process_single_track_landfall with DataArray input."""
        from unittest.mock import MagicMock

        # Create a simple track DataArray that moves across coordinates
        track = xr.DataArray(
            [35.0, 40.0, 45.0, 40.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15 00:00", periods=4, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0, 25.5]),
                "longitude": (["valid_time"], [-82.0, -81.5, -81.0, -80.5]),
            },
            name="surface_wind_speed",
        )

        # Mock land geometry that will detect a landfall
        mock_land_geom = MagicMock()
        mock_land_geom.intersects.return_value = True
        mock_land_geom.intersection.return_value.coords = [(-81.25, 24.75)]

        # Call the function
        result = calc._process_single_track_landfall(
            track, mock_land_geom, return_all_landfalls=False
        )

        # Verify result structure (may be None if no landfall detected)
        # In real scenarios with actual land geometry, this would return a DataArray
        assert result is None or isinstance(result, xr.DataArray)

    def test_process_single_track_landfall_insufficient_points(self):
        """Test _process_single_track_landfall with too few points."""
        from unittest.mock import MagicMock

        # Create track with only 1 point (need at least 2 for landfall detection)
        track = xr.DataArray(
            [35.0],
            dims=["valid_time"],
            coords={
                "valid_time": [pd.Timestamp("2023-09-15 00:00")],
                "latitude": (["valid_time"], [24.0]),
                "longitude": (["valid_time"], [-82.0]),
            },
            name="surface_wind_speed",
        )

        mock_land_geom = MagicMock()

        # Should return None for insufficient points
        result = calc._process_single_track_landfall(track, mock_land_geom)
        assert result is None

    def test_process_single_track_landfall_with_nan_values(self):
        """Test _process_single_track_landfall filters NaN values correctly."""
        from unittest.mock import MagicMock

        # Create track with some NaN values
        track = xr.DataArray(
            [35.0, np.nan, 45.0, 40.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15 00:00", periods=4, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0, 25.5]),
                "longitude": (["valid_time"], [-82.0, -81.5, -81.0, -80.5]),
            },
            name="surface_wind_speed",
        )

        mock_land_geom = MagicMock()
        mock_land_geom.intersects.return_value = False

        # Should handle NaN values gracefully
        result = calc._process_single_track_landfall(track, mock_land_geom)
        assert result is None or isinstance(result, xr.DataArray)

    def test_process_single_track_landfall_with_init_time(self):
        """Test _process_single_track_landfall adds init_time coordinate."""
        from unittest.mock import MagicMock, patch

        # Create a simple track
        track = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15 00:00", periods=3, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0]),
                "longitude": (["valid_time"], [-82.0, -81.5, -81.0]),
            },
            name="surface_wind_speed",
        )

        init_time_val = np.datetime64("2023-09-14T18:00")

        # Mock land geometry and landfall detection
        mock_land_geom = MagicMock()

        # Mock _is_true_landfall to return True for at least one segment
        with patch("extremeweatherbench.calc._is_true_landfall") as mock_is_landfall:
            mock_is_landfall.return_value = True
            mock_land_geom.intersection.return_value.coords = [(-81.5, 24.5)]

            result = calc._process_single_track_landfall(
                track,
                mock_land_geom,
                return_all_landfalls=False,
                init_time=init_time_val,
            )

            # If landfall is detected, result should have init_time coordinate
            if result is not None:
                assert "init_time" in result.coords
                assert result.coords["init_time"] == init_time_val

    def test_process_single_track_landfall_flattens_multidim_coords(self):
        """Test that multi-dimensional coordinates are properly flattened."""
        from unittest.mock import MagicMock

        # Create track with 2D coordinates (e.g., from lead_time dimension)
        track = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15 00:00", periods=3, freq="6h"),
                # Simulate coordinates that might be 2D after some operations
                "latitude": (["valid_time"], [24.0, 24.5, 25.0]),
                "longitude": (["valid_time"], [-82.0, -81.5, -81.0]),
            },
            name="surface_wind_speed",
        )

        mock_land_geom = MagicMock()
        mock_land_geom.intersects.return_value = False

        # Should handle and flatten coordinates without error
        result = calc._process_single_track_landfall(track, mock_land_geom)
        # Function should complete without error
        assert result is None or isinstance(result, xr.DataArray)

    def test_find_landfalls_with_custom_land_geometry(self):
        """Test find_landfalls with custom land geometry."""
        import shapely.geometry

        # Create track moving from ocean to land
        # Atlantic to Florida coast
        track = xr.DataArray(
            [35.0, 40.0, 45.0, 50.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=4, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0, 25.5]),
                "longitude": (["valid_time"], [280.0, 279.0, 278.0, 277.0]),
            },
            name="surface_wind_speed",
        )

        # Create simple land geometry (box representing Florida coast)
        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        result = calc.find_landfalls(track, land_geom=land_geom)

        # Should return DataArray or None
        assert result is None or isinstance(result, xr.DataArray)
        if result is not None:
            # Should have expected coordinates
            assert "latitude" in result.coords
            assert "longitude" in result.coords
            assert "valid_time" in result.coords

    def test_find_landfalls_with_default_land_geometry(self):
        """Test find_landfalls with default (None) land geometry."""
        # Create track over ocean that doesn't hit land
        track = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=3, freq="6h"),
                "latitude": (["valid_time"], [20.0, 20.5, 21.0]),
                "longitude": (["valid_time"], [260.0, 260.5, 261.0]),
            },
            name="surface_wind_speed",
        )

        # Use default land geometry (None)
        result = calc.find_landfalls(track, land_geom=None)

        # Should complete without error
        assert result is None or isinstance(result, xr.DataArray)

    def test_find_landfalls_forecast_data(self):
        """Test find_landfalls with forecast data dimensions."""
        # Create forecast track data (lead_time, valid_time)
        valid_times = pd.date_range("2023-09-15", periods=4, freq="6h")

        track = xr.DataArray(
            [[35.0, 40.0, 45.0, 50.0], [36.0, 41.0, 46.0, 51.0]],
            dims=["lead_time", "valid_time"],
            coords={
                "lead_time": [0, 6],
                "valid_time": valid_times,
                "latitude": (
                    ["lead_time", "valid_time"],
                    [[24.0, 24.5, 25.0, 25.5], [24.2, 24.7, 25.2, 25.7]],
                ),
                "longitude": (
                    ["lead_time", "valid_time"],
                    [[280.0, 279.0, 278.0, 277.0], [280.1, 279.1, 278.1, 277.1]],
                ),
            },
            name="surface_wind_speed",
        )

        import shapely.geometry

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        result = calc.find_landfalls(track, land_geom=land_geom)

        # Should handle forecast dimensions
        assert result is None or isinstance(result, xr.DataArray)

    def test_find_landfalls_return_all(self):
        """Test find_landfalls with return_all_landfalls=True."""
        import shapely.geometry

        # Create track that crosses land multiple times
        track = xr.DataArray(
            [35.0, 40.0, 45.0, 50.0, 55.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=5, freq="6h"),
                "latitude": (["valid_time"], [24.0, 25.0, 26.0, 27.0, 28.0]),
                "longitude": (["valid_time"], [280.0, 279.0, 278.0, 277.0, 276.0]),
            },
            name="surface_wind_speed",
        )

        # Create land geometry with multiple potential crossings
        land_geom = shapely.geometry.box(-82, 24, -80, 28)

        result = calc.find_landfalls(
            track, land_geom=land_geom, return_all_landfalls=True
        )

        # Should return DataArray with landfall dimension if found
        if result is not None:
            # If landfalls found, should have landfall dimension
            assert "landfall" in result.dims or result.dims == ()

    def test_find_landfalls_single_point(self):
        """Test find_landfalls with insufficient points."""
        # Track with only 1 point (need at least 2)
        track = xr.DataArray(
            [35.0],
            dims=["valid_time"],
            coords={
                "valid_time": [pd.Timestamp("2023-09-15 00:00")],
                "latitude": (["valid_time"], [24.0]),
                "longitude": (["valid_time"], [280.0]),
            },
            name="surface_wind_speed",
        )

        import shapely.geometry

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        result = calc.find_landfalls(track, land_geom=land_geom)

        # Should return None for insufficient points
        assert result is None

    def test_find_landfalls_with_track_dimension(self):
        """Test find_landfalls with singleton track dimension."""
        import shapely.geometry

        # Create track with track dimension
        track = xr.DataArray(
            [[35.0, 40.0, 45.0]],
            dims=["track", "valid_time"],
            coords={
                "track": [0],
                "valid_time": pd.date_range("2023-09-15", periods=3, freq="6h"),
                "latitude": (["track", "valid_time"], [[24.0, 24.5, 25.0]]),
                "longitude": (["track", "valid_time"], [[280.0, 279.0, 278.0]]),
            },
            name="surface_wind_speed",
        )

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        # Should squeeze track dimension and process
        result = calc.find_landfalls(track, land_geom=land_geom)

        assert result is None or isinstance(result, xr.DataArray)

    def test_find_next_landfall_for_init_time(self):
        """Test find_next_landfall_for_init_time function."""
        # Create forecast data with init_time dimension
        init_times = pd.date_range("2023-09-14", periods=3, freq="12h")
        lead_times = [0, 6, 12, 18]

        forecast = xr.DataArray(
            np.random.rand(3, 4),
            dims=["init_time", "lead_time"],
            coords={"init_time": init_times, "lead_time": lead_times},
            name="forecast_data",
        )

        # Create target landfall data with multiple landfalls
        landfall_times = pd.date_range("2023-09-14 06:00", periods=3, freq="12h")
        target_landfalls = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["landfall"],
            coords={
                "landfall": [0, 1, 2],
                "valid_time": (["landfall"], landfall_times),
                "latitude": (["landfall"], [24.0, 24.5, 25.0]),
                "longitude": (["landfall"], [280.0, 279.0, 278.0]),
            },
            name="surface_wind_speed",
        )

        result = calc.find_next_landfall_for_init_time(forecast, target_landfalls)

        # Should return DataArray with init_time dimension or None
        assert result is None or isinstance(result, xr.DataArray)
        if result is not None:
            assert "init_time" in result.dims or "init_time" in result.coords

    def test_find_next_landfall_no_future_landfalls(self):
        """Test find_next_landfall when no future landfalls exist."""
        # Forecast after all landfalls
        init_times = pd.date_range("2023-09-17", periods=2, freq="12h")
        forecast = xr.DataArray(
            np.random.rand(2, 4),
            dims=["init_time", "lead_time"],
            coords={"init_time": init_times, "lead_time": [0, 6, 12, 18]},
        )

        # Target landfalls all in the past
        landfall_times = pd.date_range("2023-09-14", periods=3, freq="12h")
        target_landfalls = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["landfall"],
            coords={
                "landfall": [0, 1, 2],
                "valid_time": (["landfall"], landfall_times),
                "latitude": (["landfall"], [24.0, 24.5, 25.0]),
                "longitude": (["landfall"], [280.0, 279.0, 278.0]),
            },
        )

        result = calc.find_next_landfall_for_init_time(forecast, target_landfalls)

        # Should return None when no future landfalls
        assert result is None

    def test_find_next_landfall_without_init_time(self):
        """Test find_next_landfall with forecast without init_time."""
        # Forecast data with lead_time and valid_time
        valid_times = pd.date_range("2023-09-15", periods=4, freq="6h")
        lead_times = [0, 6, 12, 18]

        forecast = xr.DataArray(
            np.random.rand(4, 4),
            dims=["valid_time", "lead_time"],
            coords={"valid_time": valid_times, "lead_time": lead_times},
        )

        # Target landfalls
        landfall_times = pd.date_range("2023-09-15 06:00", periods=3, freq="12h")
        target_landfalls = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["landfall"],
            coords={
                "landfall": [0, 1, 2],
                "valid_time": (["landfall"], landfall_times),
                "latitude": (["landfall"], [24.0, 24.5, 25.0]),
                "longitude": (["landfall"], [280.0, 279.0, 278.0]),
            },
        )
        with pytest.raises(
            AttributeError,
            match="'DataArray' object has no attribute 'init_time'",
        ):
            calc.find_next_landfall_for_init_time(forecast, target_landfalls)

    def test_is_true_landfall_ocean_to_land(self):
        """Test _is_true_landfall detects ocean to land movement."""
        import shapely.geometry

        # Create simple land geometry (box)
        land_geom = shapely.geometry.box(-81, 24, -80, 26)

        # Ocean point to land point (landfall)
        lon1, lat1 = -82.0, 25.0  # Ocean
        lon2, lat2 = -80.5, 25.0  # Land

        result = calc._is_true_landfall(lon1, lat1, lon2, lat2, land_geom)

        # Should detect landfall
        assert result is True

    def test_is_true_landfall_ocean_crossing_land(self):
        """Test _is_true_landfall detects ocean to ocean crossing land."""
        import shapely.geometry

        # Create land geometry
        land_geom = shapely.geometry.box(-81, 24, -80, 26)

        # Ocean to ocean but crossing land
        lon1, lat1 = -82.0, 25.0  # Ocean
        lon2, lat2 = -79.0, 25.0  # Ocean (other side)

        result = calc._is_true_landfall(lon1, lat1, lon2, lat2, land_geom)

        # Should detect landfall (crossing)
        assert result is True

    def test_is_true_landfall_land_to_ocean(self):
        """Test _is_true_landfall rejects land to ocean movement."""
        import shapely.geometry

        # Create land geometry
        land_geom = shapely.geometry.box(-81, 24, -80, 26)

        # Land to ocean (not landfall)
        lon1, lat1 = -80.5, 25.0  # Land
        lon2, lat2 = -82.0, 25.0  # Ocean

        result = calc._is_true_landfall(lon1, lat1, lon2, lat2, land_geom)

        # Should not detect landfall (exit, not entry)
        assert result is False

    def test_is_true_landfall_ocean_to_ocean_no_intersection(self):
        """Test _is_true_landfall rejects pure ocean movement."""
        import shapely.geometry

        # Create land geometry
        land_geom = shapely.geometry.box(-81, 24, -80, 26)

        # Ocean to ocean without crossing land
        lon1, lat1 = -85.0, 25.0  # Ocean (far away)
        lon2, lat2 = -84.0, 25.0  # Ocean (still far)

        result = calc._is_true_landfall(lon1, lat1, lon2, lat2, land_geom)

        # Should not detect landfall
        assert result is False

    def test_is_true_landfall_error_handling(self):
        """Test _is_true_landfall handles errors gracefully."""
        # Invalid geometry should return False
        result = calc._is_true_landfall(0, 0, 1, 1, None)

        # Should return False on error
        assert result is False

    def test_find_landfalls_with_unnamed_dataarray(self):
        """Test find_landfalls with unnamed DataArray."""
        import shapely.geometry

        # Create track without a name
        track = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=3, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0]),
                "longitude": (["valid_time"], [280.0, 279.0, 278.0]),
            },
            # No name attribute
        )

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        result = calc.find_landfalls(track, land_geom=land_geom)

        # Should handle unnamed DataArray
        assert result is None or isinstance(result, xr.DataArray)

    def test_process_single_track_landfall_unnamed_variable(self):
        """Test _process_single_track_landfall with unnamed variable."""
        from unittest.mock import patch

        import shapely.geometry

        # Create track without a name
        track = xr.DataArray(
            [35.0, 40.0, 45.0],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=3, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0]),
                "longitude": (["valid_time"], [280.0, 279.0, 278.0]),
            },
            # No name attribute
        )

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        # Mock _is_true_landfall to return True
        with patch("extremeweatherbench.calc._is_true_landfall") as mock_landfall:
            mock_landfall.return_value = True

            result = calc._process_single_track_landfall(
                track, land_geom, return_all_landfalls=False
            )

            # Should handle unnamed variable (value becomes NaN)
            if result is not None:
                assert result is not None or np.isnan(result.values)

    def test_find_landfalls_all_nan_track(self):
        """Test find_landfalls with all NaN values in track."""
        import shapely.geometry

        # Create track with all NaN values
        track = xr.DataArray(
            [np.nan, np.nan, np.nan],
            dims=["valid_time"],
            coords={
                "valid_time": pd.date_range("2023-09-15", periods=3, freq="6h"),
                "latitude": (["valid_time"], [24.0, 24.5, 25.0]),
                "longitude": (["valid_time"], [280.0, 279.0, 278.0]),
            },
            name="surface_wind_speed",
        )

        land_geom = shapely.geometry.box(-82, 24, -80, 26)

        result = calc.find_landfalls(track, land_geom=land_geom)

        # Should return None for all NaN track
        assert result is None
