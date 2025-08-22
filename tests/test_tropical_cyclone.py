# tests/test_tropical_cyclone.py
"""Comprehensive unit tests for the tropical cyclone module.

This test suite covers:
- Registry functions for IBTrACS data
- Cache key generation
- TC track detection and filtering
- Helper functions for contour detection
- Dataset conversion utilities
"""

# flake8: noqa: E501
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench.events import tropical_cyclone


@pytest.fixture
def sample_tc_dataset():
    """Create a sample dataset for TC detection testing."""
    time = pd.date_range("2023-09-01", periods=5, freq="6h")
    lead_time = np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")
    lat = np.linspace(10, 40, 31)
    lon = np.linspace(-80, -50, 31)

    # Create realistic pressure field with a low-pressure system
    data_shape = (len(time), len(lat), len(lon), len(lead_time))

    # Create base pressure field
    base_pressure = np.full(data_shape, 101325.0)

    # Add a low pressure center
    center_lat, center_lon = 15, 2  # indices for lat ~25N, lon ~65W
    for t in range(len(time)):
        for lt in range(len(lead_time)):
            # Create a low pressure center that moves
            offset_lat = center_lat + t
            offset_lon = center_lon + t

            for i in range(len(lat)):
                for j in range(len(lon)):
                    dist = np.sqrt((i - offset_lat) ** 2 + (j - offset_lon) ** 2)
                    if dist < 5:
                        pressure_drop = 2000 * np.exp(
                            -dist / 2
                        )  # ~20 hPa drop at center
                        base_pressure[t, i, j, lt] -= pressure_drop

    # Create wind field
    wind_u = np.random.normal(0, 5, data_shape)
    wind_v = np.random.normal(0, 5, data_shape)

    # Create geopotential field at 500hPa
    geopotential = np.random.normal(5500, 100, data_shape) * 9.80665

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                base_pressure,
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                wind_u,
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                wind_v,
            ),
            "geopotential": (
                ["time", "latitude", "longitude", "prediction_timedelta"],
                geopotential,
            ),
        },
        coords={
            "valid_time": time,
            "latitude": lat,
            "longitude": lon,
            "lead_time": lead_time,
        },
    )

    return dataset


@pytest.fixture
def sample_ibtracs_dataset():
    """Create a sample IBTrACS dataset."""
    valid_time = pd.date_range("2023-09-01", periods=10, freq="6h")

    # Create a realistic storm track
    n_points = len(valid_time)
    lats = np.linspace(12, 30, n_points)  # Storm moving northward
    lons = np.linspace(-75, -65, n_points)  # Storm moving westward

    dataset = xr.Dataset(
        {
            "latitude": (["valid_time"], lats),
            "longitude": (["valid_time"], lons),
            "max_sustained_wind": (["valid_time"], np.random.uniform(30, 80, n_points)),
            "min_pressure": (["valid_time"], np.random.uniform(950, 1010, n_points)),
        },
        coords={"valid_time": valid_time},
    )

    return dataset


class TestIBTrACSRegistry:
    """Test the IBTrACS data registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        tropical_cyclone.clear_ibtracs_registry()

    def test_register_and_get_ibtracs_data(self, sample_ibtracs_dataset):
        """Test registering and retrieving IBTrACS data."""
        case_id = "test_case_123"

        # Initially should return None
        assert tropical_cyclone.get_ibtracs_data(case_id) is None

        # Register data
        tropical_cyclone.register_ibtracs_data(case_id, sample_ibtracs_dataset)

        # Should now return the dataset
        retrieved_data = tropical_cyclone.get_ibtracs_data(case_id)
        assert retrieved_data is not None
        xr.testing.assert_equal(retrieved_data, sample_ibtracs_dataset)

    def test_clear_ibtracs_registry(self, sample_ibtracs_dataset):
        """Test clearing the IBTrACS registry."""
        case_id = "test_case_456"

        # Register data
        tropical_cyclone.register_ibtracs_data(case_id, sample_ibtracs_dataset)
        assert tropical_cyclone.get_ibtracs_data(case_id) is not None

        # Clear registry
        tropical_cyclone.clear_ibtracs_registry()
        assert tropical_cyclone.get_ibtracs_data(case_id) is None

    def test_multiple_case_ids(self, sample_ibtracs_dataset):
        """Test handling multiple case IDs."""
        case_id1 = "case_1"
        case_id2 = "case_2"

        # Modify dataset for second case
        dataset2 = sample_ibtracs_dataset.copy()
        dataset2["latitude"] = dataset2["latitude"] + 5

        # Register both
        tropical_cyclone.register_ibtracs_data(case_id1, sample_ibtracs_dataset)
        tropical_cyclone.register_ibtracs_data(case_id2, dataset2)

        # Verify both can be retrieved correctly
        retrieved1 = tropical_cyclone.get_ibtracs_data(case_id1)
        retrieved2 = tropical_cyclone.get_ibtracs_data(case_id2)

        xr.testing.assert_equal(retrieved1, sample_ibtracs_dataset)
        xr.testing.assert_equal(retrieved2, dataset2)


class TestCacheKeyGeneration:
    """Test the cache key generation function."""

    def test_generate_cache_key_basic(self, sample_tc_dataset):
        """Test basic cache key generation."""
        cache_key = tropical_cyclone._generate_cache_key(sample_tc_dataset)

        # Should be a valid MD5 hash
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length

        # Should be hexadecimal
        int(cache_key, 16)  # Should not raise an error

    def test_cache_key_consistency(self, sample_tc_dataset):
        """Test that cache key is consistent for same data."""
        key1 = tropical_cyclone._generate_cache_key(sample_tc_dataset)
        key2 = tropical_cyclone._generate_cache_key(sample_tc_dataset)

        assert key1 == key2

    def test_cache_key_different_for_different_data(self, sample_tc_dataset):
        """Test that different data produces different cache keys."""
        key1 = tropical_cyclone._generate_cache_key(sample_tc_dataset)

        # Modify the dataset
        modified_dataset = sample_tc_dataset.copy()
        modified_dataset["air_pressure_at_mean_sea_level"] += 100

        key2 = tropical_cyclone._generate_cache_key(modified_dataset)

        assert key1 != key2

    def test_cache_key_with_missing_variables(self):
        """Test cache key generation with missing expected variables."""
        # Create dataset with only some expected variables
        time = pd.date_range("2023-09-01", periods=2, freq="6h")
        lat = np.linspace(10, 20, 5)
        lon = np.linspace(-70, -60, 5)

        dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (2, 5, 5)),
                ),
                # Missing other expected variables
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

        # Should still generate a key without error
        cache_key = tropical_cyclone._generate_cache_key(dataset)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32


class TestTropicalCycloneDetection:
    """Test tropical cyclone detection functions."""

    def test_find_furthest_contour_from_point(self):
        """Test finding furthest point in contour."""
        # Simple square contour
        contour = [(0, 0), (0, 1), (1, 1), (1, 0)]
        point = (0.5, 0.5)  # Center point

        furthest, _ = tropical_cyclone.find_furthest_contour_from_point(contour, point)

        # All corners are equidistant, so any corner is valid so long as its a tuple
        assert tuple(furthest) in contour

    def test_find_furthest_contour_with_numpy_arrays(self):
        """Test with numpy arrays instead of lists."""
        contour = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([1, 1])

        furthest, _ = tropical_cyclone.find_furthest_contour_from_point(contour, point)

        # Should return a point from the contour
        assert len(furthest) == 2

    @patch("extremeweatherbench.events.tropical_cyclone.measure")
    def test_find_contours_from_point_specified_field(self, mock_measure):
        """Test contour finding from field."""
        # Create simple field
        lat = np.linspace(10, 20, 11)
        lon = np.linspace(-70, -60, 11)
        field_data = np.random.normal(0, 1, (11, 11))

        field = xr.DataArray(
            field_data,
            coords={"latitude": lat, "longitude": lon},
            dims=["latitude", "longitude"],
        )

        # Mock the measure.find_contours function
        mock_measure.find_contours.return_value = [
            np.array([[5, 5], [6, 5], [6, 6], [5, 6]])  # Simple square contour
        ]

        point = (5, 5)  # Middle of the domain (array indices)
        level = 0.0

        contours = tropical_cyclone.find_contours_from_point_specified_field(
            field, point, level
        )

        assert isinstance(contours, list)
        mock_measure.find_contours.assert_called_once()

    def test_create_tctracks_from_dataset_with_ibtracs_filter(
        self, sample_tc_dataset, sample_ibtracs_dataset
    ):
        """Test TC track creation with IBTrACS filtering."""
        # This is a complex integration test
        with patch(
            "extremeweatherbench.events.tropical_cyclone._create_tctracks_optimized_with_ibtracs"
        ) as mock_create:
            # Mock the return value
            mock_result = xr.Dataset(
                {
                    "tc_slp": (
                        ["time", "prediction_timedelta"],
                        np.random.normal(101000, 1000, (5, 5)),
                    ),
                    "tc_latitude": (
                        ["time", "prediction_timedelta"],
                        np.random.uniform(10, 30, (5, 5)),
                    ),
                    "tc_longitude": (
                        ["time", "prediction_timedelta"],
                        np.random.uniform(-80, -50, (5, 5)),
                    ),
                    "tc_vmax": (
                        ["time", "prediction_timedelta"],
                        np.random.uniform(20, 50, (5, 5)),
                    ),
                }
            )
            mock_create.return_value = mock_result

            result = tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter(
                sample_tc_dataset, sample_ibtracs_dataset
            )

            assert isinstance(result, xr.Dataset)
            mock_create.assert_called_once()


class TestDistanceCalculations:
    """Test distance calculation functions."""

    def test_calculate_great_circle_distance_vectorized(self):
        """Test vectorized great circle distance calculation."""
        # Test known distance: equator to North Pole = 90 degrees = pi/2 radians
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 90.0, 0.0

        distance = tropical_cyclone._calculate_great_circle_distance_vectorized(
            lat1, lon1, lat2, lon2
        )

        # Should be approximately 90 degrees
        assert abs(distance - 90.0) < 0.1

    def test_calculate_great_circle_distance_scalar(self):
        """Test scalar great circle distance calculation."""
        # Test same known distance
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 90.0, 0.0

        distance = tropical_cyclone._calculate_great_circle_distance(
            lat1, lon1, lat2, lon2
        )

        assert abs(distance - 90.0) < 0.1

    def test_distance_calculation_consistency(self):
        """Test that vectorized and scalar versions give same results."""
        lat1, lon1 = 25.0, -80.0
        lat2, lon2 = 30.0, -75.0

        dist_vectorized = tropical_cyclone._calculate_great_circle_distance_vectorized(
            lat1, lon1, lat2, lon2
        )
        dist_scalar = tropical_cyclone._calculate_great_circle_distance(
            lat1, lon1, lat2, lon2
        )

        assert abs(dist_vectorized - dist_scalar) < 1e-10


class TestUtilityFunctions:
    """Test utility functions in the tropical cyclone module."""

    def test_safe_extract_value_scalar(self):
        """Test safe extraction of scalar values."""
        # Test with scalar
        value = tropical_cyclone._safe_extract_value(42.0)
        assert value == 42.0

        # Test with numpy scalar
        np_scalar = np.float64(42.0)
        value = tropical_cyclone._safe_extract_value(np_scalar)
        assert value == 42.0

        # Test with 0-d array
        array_0d = np.array(42.0)
        value = tropical_cyclone._safe_extract_value(array_0d)
        assert value == 42.0

    def test_safe_extract_value_array(self):
        """Test safe extraction from arrays."""
        # Test with 1-d array - should return first element
        array_1d = np.array([42.0, 43.0, 44.0])
        value = tropical_cyclone._safe_extract_value(array_1d)
        assert value == 42.0

    def test_create_spatial_mask_vectorized(self):
        """Test vectorized spatial mask creation."""
        lat_coords = np.array([20.0, 25.0, 30.0])
        lon_coords = np.array([-80.0, -75.0, -70.0])

        # Create mock nearby IBTrACS data
        nearby_ibtracs = pd.DataFrame(
            {"latitude": [22.0, 27.0], "longitude": [-78.0, -73.0]}
        )

        max_distance = 5.0

        mask = tropical_cyclone._create_spatial_mask_vectorized(
            lat_coords, lon_coords, nearby_ibtracs, max_distance
        )

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3, 3)  # lat x lon
        assert mask.dtype == bool


@pytest.mark.integration
class TestTCIntegration:
    """Integration tests for TC functionality."""

    def test_full_tc_pipeline_mock(self, sample_tc_dataset, sample_ibtracs_dataset):
        """Test the full TC detection pipeline with mocked components."""
        # This tests the integration without actually running expensive computations

        with patch(
            "extremeweatherbench.events.tropical_cyclone._process_entire_dataset_compact"
        ) as mock_process:
            # Mock the processing function to return some fake detections
            mock_process.return_value = (
                np.array([5]),  # n_detections
                np.array([0]),  # lt_indices
                np.array([0]),  # vt_indices
                np.array([1]),  # track_ids
                np.array([25.0]),  # lats
                np.array([-75.0]),  # lons
                np.array([101000.0]),  # slp_vals
                np.array([25.0]),  # wind_vals
            )

            result = tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter(
                sample_tc_dataset, sample_ibtracs_dataset
            )

            assert isinstance(result, xr.Dataset)
            # Check that expected variables are present
            expected_vars = ["tc_slp", "tc_latitude", "tc_longitude", "tc_vmax"]
            for var in expected_vars:
                assert var in result.data_vars
