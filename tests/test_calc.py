# tests/test_calc.py
"""Comprehensive unit tests for the calc module.

This test suite covers:
- Basic calculation functions
- Wind speed calculations
- Pressure calculations
- Geopotential thickness calculations
- TC variable generation
- Numerical integration functions
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc


@pytest.fixture
def sample_calc_dataset():
    """Create a sample dataset for calc testing."""
    time = pd.date_range("2023-01-01", periods=5, freq="6h")
    lat = np.linspace(20, 50, 16)
    lon = np.linspace(-120, -80, 21)
    level = [1000, 850, 700, 500, 300, 200]

    # Create realistic meteorological data
    data_shape_3d = (len(time), len(lat), len(lon))
    data_shape_4d = (len(time), len(level), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                np.random.normal(101325, 1000, data_shape_3d),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape_3d),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape_3d),
            ),
            "geopotential": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(5000, 1000, data_shape_4d) * 9.80665,
            ),
            "geopotential_at_surface": (
                ["time", "latitude", "longitude"],
                np.random.normal(500, 200, data_shape_3d) * 9.80665,
            ),
            "eastward_wind": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(0, 15, data_shape_4d),
            ),
            "northward_wind": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(0, 15, data_shape_4d),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                np.random.uniform(0.001, 0.02, data_shape_4d),
            ),
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "level": level,
        },
    )

    return dataset


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


class TestWindCalculations:
    """Test wind-related calculations."""

    def test_calculate_wind_speed_from_components(self, sample_calc_dataset):
        """Test wind speed calculation from components."""
        wind_speed = calc.calculate_wind_speed(sample_calc_dataset)

        # Should return a DataArray
        assert isinstance(wind_speed, xr.DataArray)
        assert wind_speed.shape == sample_calc_dataset.surface_eastward_wind.shape

        # All wind speeds should be non-negative
        assert (wind_speed >= 0).all()

        # Check against manual calculation for a point
        u = sample_calc_dataset.surface_eastward_wind.isel(
            time=0, latitude=0, longitude=0
        ).values
        v = sample_calc_dataset.surface_northward_wind.isel(
            time=0, latitude=0, longitude=0
        ).values
        expected = np.sqrt(u**2 + v**2)
        calculated = wind_speed.isel(time=0, latitude=0, longitude=0).values

        assert abs(calculated - expected) < 1e-10

    def test_calculate_wind_speed_with_existing_wind_speed(self):
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

        result = calc.calculate_wind_speed(dataset)

        # Should return the existing wind speed unchanged
        xr.testing.assert_equal(result, dataset["surface_wind_speed"])

    def test_calculate_wind_speed_missing_components(self):
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

        with pytest.raises(ValueError, match="No suitable wind speed variables found"):
            calc.calculate_wind_speed(dataset)


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
