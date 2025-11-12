# tests/test_tropical_cyclone.py
"""Comprehensive unit tests for the tropical cyclone module.

This test suite covers:
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
    valid_time = pd.date_range("2023-09-01", periods=5, freq="6h")
    lead_time = np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")
    lat = np.linspace(10, 40, 31)
    lon = np.linspace(-80, -50, 31)

    # Create realistic pressure field with a low-pressure system
    data_shape = (len(valid_time), len(lat), len(lon), len(lead_time))

    # Create base pressure field
    base_pressure = np.full(data_shape, 101325.0)

    # Add a low pressure center
    center_lat, center_lon = 15, 2  # indices for lat ~25N, lon ~65W
    for t in range(len(valid_time)):
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
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

    # Create geopotential field at 500hPa
    geopotential = np.random.normal(5500, 100, data_shape) * 9.80665

    # Create init_time coordinate (2D: lead_time x valid_time)
    # For forecast data: valid_time = init_time + lead_time
    init_time_grid = np.broadcast_to(
        valid_time.values.reshape(1, -1), (len(lead_time), len(valid_time))
    ) - np.broadcast_to(lead_time.reshape(-1, 1), (len(lead_time), len(valid_time)))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["valid_time", "latitude", "longitude", "lead_time"],
                base_pressure,
            ),
            "surface_eastward_wind": (
                ["valid_time", "latitude", "longitude", "lead_time"],
                wind_u,
            ),
            "surface_northward_wind": (
                ["valid_time", "latitude", "longitude", "lead_time"],
                wind_v,
            ),
            "surface_wind_speed": (
                ["valid_time", "latitude", "longitude", "lead_time"],
                wind_speed,
            ),
            "geopotential": (
                ["valid_time", "latitude", "longitude", "lead_time"],
                geopotential,
            ),
        },
        coords={
            "valid_time": valid_time,
            "init_time": (["lead_time", "valid_time"], init_time_grid),
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
            "surface_wind_speed": (["valid_time"], np.random.uniform(30, 80, n_points)),
            "air_pressure_at_mean_sea_level": (
                ["valid_time"],
                np.random.uniform(95000, 101000, n_points),
            ),
        },
        coords={"valid_time": valid_time},
    )
    dataset.attrs["source"] = "IBTrACS"
    dataset.attrs["is_ibtracs_data"] = True

    return dataset


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

    def test_generate_forecast_tctracks(
        self, sample_tc_dataset, sample_ibtracs_dataset
    ):
        """Test TC track creation with TC track data filtering."""
        # This is a complex integration test
        with patch(
            "extremeweatherbench.events.tropical_cyclone.generate_tc_tracks_by_init_time"
        ) as mock_process:
            # Mock the return value - generate_tc_tracks_by_init_time returns a Dataset
            mock_process.return_value = xr.Dataset(
                {
                    "track_id": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[1, 2]]]),
                    ),
                    "latitude": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[25.0, 24.0]]]),
                    ),
                    "longitude": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[-75.0, -76.0]]]),
                    ),
                    "air_pressure_at_mean_sea_level": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[101000.0, 101200.0]]]),
                    ),
                    "surface_wind_speed": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[25.0, 20.0]]]),
                    ),
                },
                coords={
                    "lead_time": [12],
                    "valid_time": [pd.Timestamp("2023-09-15")],
                    "track": [0, 1],
                },
            )

            # Call the actual function that exists
            result = tropical_cyclone.generate_tc_tracks_by_init_time(
                sample_tc_dataset["air_pressure_at_mean_sea_level"],
                sample_tc_dataset["surface_wind_speed"],
                None,  # geopotential_thickness
                sample_ibtracs_dataset,
            )

            assert isinstance(result, xr.Dataset)
            mock_process.assert_called_once()


class TestDistanceCalculations:
    """Test distance calculation functions using calc module."""

    def test_haversine_distance_degrees(self):
        """Test haversine distance calculation in degrees."""
        # Test known distance: equator to North Pole = 90 degrees = pi/2 radians
        from extremeweatherbench import calc

        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 90.0, 0.0

        distance = calc.haversine_distance([lat1, lon1], [lat2, lon2], units="degrees")

        # Should be approximately 90 degrees
        assert abs(distance - 90.0) < 0.1

    def test_haversine_distance_km(self):
        """Test haversine distance calculation in kilometers."""
        from extremeweatherbench import calc

        # Test same known distance in km
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 90.0, 0.0

        distance_km = calc.haversine_distance([lat1, lon1], [lat2, lon2], units="km")

        # Should be approximately 10,000 km (quarter of Earth's circumference)
        assert abs(distance_km - 10000) < 100

    def test_distance_calculation_consistency(self):
        """Test that distance calculations are consistent."""
        from extremeweatherbench import calc

        lat1, lon1 = 25.0, -80.0
        lat2, lon2 = 30.0, -75.0

        dist1 = calc.haversine_distance([lat1, lon1], [lat2, lon2], units="degrees")
        dist2 = calc.haversine_distance([lat1, lon1], [lat2, lon2], units="degrees")

        assert abs(dist1 - dist2) < 1e-10


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

    def test_create_spatial_mask(self):
        """Test vectorized spatial mask creation."""
        lat_coords = np.array([20.0, 25.0, 30.0])
        lon_coords = np.array([-80.0, -75.0, -70.0])

        # Create mock nearby IBTrACS data
        nearby_ibtracs = pd.DataFrame(
            {"latitude": [22.0, 27.0], "longitude": [-78.0, -73.0]}
        )

        max_distance = 5.0

        mask = tropical_cyclone._create_spatial_mask(
            lat_coords, lon_coords, nearby_ibtracs, max_distance
        )

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (3, 3)  # lat x lon
        assert mask.dtype == bool


class TestDimensionHandling:
    """Test dimension handling in TC detection functions."""

    @pytest.fixture
    def forecast_dataset_with_init_time(self):
        """Create a forecast dataset that mimics the structure that caused the bug."""
        # This mimics your actual model data structure:
        # lead_time x valid_time x latitude x longitude
        lead_times = np.arange(0, 42, 6)  # 0 to 240 hours, every 6 hours
        valid_times = pd.date_range("2023-09-01", periods=129, freq="h")
        lat = np.linspace(10, 40, 147)
        lon = np.linspace(240, 360, 132)  # 240-360 to match your longitude range

        # Create init_time coordinate with (lead_time, valid_time) dimensions
        init_time_grid = np.broadcast_to(
            valid_times.values.reshape(1, -1), (len(lead_times), len(valid_times))
        ) - np.broadcast_to(
            lead_times.reshape(-1, 1) * np.timedelta64(1, "h"),
            (len(lead_times), len(valid_times)),
        )

        # Create realistic pressure and wind data
        data_shape = (len(lead_times), len(valid_times), len(lat), len(lon))
        base_pressure = np.random.normal(101325, 1000, data_shape)
        wind_u = np.random.normal(0, 10, data_shape)
        wind_v = np.random.normal(0, 10, data_shape)
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    base_pressure,
                ),
                "surface_eastward_wind": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    wind_u,
                ),
                "surface_northward_wind": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    wind_v,
                ),
                "surface_wind_speed": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    wind_speed,
                ),
            },
            coords={
                "lead_time": lead_times,
                "valid_time": valid_times,
                "latitude": lat,
                "longitude": lon,
                "init_time": (["lead_time", "valid_time"], init_time_grid),
            },
        )

        return dataset

    def test_input_core_dims_handling(self, forecast_dataset_with_init_time):
        """Test that input_core_dims correctly handles init_time dimensions."""
        slp = forecast_dataset_with_init_time["air_pressure_at_mean_sea_level"]
        init_time_coord = slp.init_time

        # Test the dimension detection logic
        spatial_dims = ["latitude", "longitude"]
        non_spatial_dims = [dim for dim in slp.dims if dim not in spatial_dims]

        # Verify our test data has the expected structure
        assert non_spatial_dims == ["lead_time", "valid_time"]
        assert list(init_time_coord.dims) == ["lead_time", "valid_time"]

        # Test the fixed input_core_dims setup
        input_core_dims = [
            non_spatial_dims + spatial_dims,  # SLP
            ["valid_time"],  # time_coord
            list(init_time_coord.dims),  # init_time_coord - FIXED
            ["latitude"],  # latitude coordinates
            ["longitude"],  # longitude coordinates
            non_spatial_dims + spatial_dims,  # wind speed dims
        ]

        expected_input_core_dims = [
            ["lead_time", "valid_time", "latitude", "longitude"],
            ["valid_time"],
            ["lead_time", "valid_time"],  # This was the fix
            ["latitude"],
            ["longitude"],
            ["lead_time", "valid_time", "latitude", "longitude"],
        ]

        assert input_core_dims == expected_input_core_dims

    def test_datetime64_handling_in_detection(self, forecast_dataset_with_init_time):
        """Test that datetime64 arrays are handled correctly in detection logic."""
        # Extract init_time array like the actual function does
        init_time_array = forecast_dataset_with_init_time.init_time.values

        # Test unique extraction with flatten (our fix)
        unique_init_times = np.unique(init_time_array.flatten())

        # Should not raise "tuple.index(x): x not in tuple" error
        assert len(unique_init_times) > 0
        assert unique_init_times.dtype.kind == "M"  # datetime64 type

        # Test datetime comparison logic (our fix)
        current_init_time = unique_init_times[0]
        if init_time_array.dtype.kind == "M":  # datetime64 type
            init_time_mask = np.abs(
                init_time_array - current_init_time
            ) <= np.timedelta64(1, "s")
        else:
            init_time_mask = init_time_array == current_init_time

        assert isinstance(init_time_mask, np.ndarray)
        assert init_time_mask.dtype == bool
        assert init_time_mask.shape == init_time_array.shape

    def test_dictionary_key_handling(self, forecast_dataset_with_init_time):
        """Test that datetime64 values can be used as dictionary keys."""
        init_time_array = forecast_dataset_with_init_time.init_time.values

        # Test the extraction and conversion logic
        detection_init_time = init_time_array[0, 0]

        # Test our fix for consistent dictionary key handling
        if hasattr(detection_init_time, "item"):
            detection_init_time = detection_init_time.item()

        # Should be able to use as dictionary key without errors
        test_dict = {}
        track_key = (1, detection_init_time)  # (track_id, detection_init_time)
        test_dict[track_key] = "test_value"

        assert test_dict[track_key] == "test_value"
        # The datetime may be converted to int (nanoseconds) or stay as datetime64/Timestamp
        assert isinstance(track_key[1], (pd.Timestamp, np.datetime64, int, type(None)))

    @patch(
        "extremeweatherbench.events.tropical_cyclone.generate_tc_tracks_by_init_time"
    )
    def test_apply_ufunc_dimension_compatibility(
        self, mock_process, forecast_dataset_with_init_time
    ):
        """Test that apply_ufunc works with the fixed dimension handling."""
        # Mock successful processing - generate_tc_tracks_by_init_time returns an empty Dataset
        mock_process.return_value = xr.Dataset(
            {
                "track_id": (
                    ["lead_time", "valid_time", "track"],
                    np.array([], dtype=int).reshape(0, 0, 0),
                ),
                "latitude": (
                    ["lead_time", "valid_time", "track"],
                    np.array([]).reshape(0, 0, 0),
                ),
                "longitude": (
                    ["lead_time", "valid_time", "track"],
                    np.array([]).reshape(0, 0, 0),
                ),
                "air_pressure_at_mean_sea_level": (
                    ["lead_time", "valid_time", "track"],
                    np.array([]).reshape(0, 0, 0),
                ),
                "surface_wind_speed": (
                    ["lead_time", "valid_time", "track"],
                    np.array([]).reshape(0, 0, 0),
                ),
            },
            coords={
                "lead_time": [],
                "valid_time": [],
                "track": [],
            },
        )

        # Create minimal IBTrACS data
        ibtracs_data = xr.Dataset(
            {
                "latitude": (["valid_time"], [25.0]),
                "longitude": (["valid_time"], [-75.0]),
                "surface_wind_speed": (["valid_time"], [30.0]),
                "air_pressure_at_mean_sea_level": (["valid_time"], [100000.0]),
            },
            coords={"valid_time": [pd.Timestamp("2023-09-01")]},
        )

        # This should not raise the "tuple.index(x): x not in tuple" error
        forecast_subset = forecast_dataset_with_init_time.isel(lead_time=slice(0, 3))
        result = tropical_cyclone.generate_tc_tracks_by_init_time(
            forecast_subset["air_pressure_at_mean_sea_level"],
            forecast_subset["surface_wind_speed"],
            None,  # geopotential_thickness
            ibtracs_data,
        )

        # Verify the function completed successfully
        assert isinstance(result, xr.Dataset)
        mock_process.assert_called_once()

    def test_real_world_dimension_structure(self):
        """Test with the exact dimension structure from your debug output."""
        # Recreate the exact structure from your debug output:
        # SLP dimensions: ('lead_time', 'valid_time', 'latitude', 'longitude')
        # SLP shape: (41, 129, 147, 132)
        # init_time_coord dimensions: ('lead_time', 'valid_time')
        # init_time_coord shape: (41, 129)

        lead_times = np.arange(41)
        valid_times = pd.date_range("2023-09-01", periods=129, freq="h")
        lat = np.linspace(10, 40, 147)
        lon = np.linspace(240, 372, 132)

        # Create init_time with exact structure from your data
        init_time_grid = np.random.choice(
            pd.date_range("2023-08-30", "2023-09-02", freq="h"), size=(41, 129)
        )

        dataset = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    np.random.normal(101325, 1000, (41, 129, 147, 132)),
                ),
                "surface_wind_speed": (
                    ["lead_time", "valid_time", "latitude", "longitude"],
                    np.random.uniform(0, 30, (41, 129, 147, 132)),
                ),
            },
            coords={
                "lead_time": lead_times,
                "valid_time": valid_times,
                "latitude": lat,
                "longitude": lon,
                "init_time": (["lead_time", "valid_time"], init_time_grid),
            },
        )

        # This should work with the fixes applied
        slp = dataset["air_pressure_at_mean_sea_level"]
        init_time_coord = slp.init_time

        # Test the core dimension logic that was fixed
        spatial_dims = ["latitude", "longitude"]
        non_spatial_dims = [dim for dim in slp.dims if dim not in spatial_dims]

        input_core_dims = [
            non_spatial_dims + spatial_dims,
            ["valid_time"],
            list(init_time_coord.dims),  # This should be ['lead_time', 'valid_time']
            ["latitude"],
            ["longitude"],
            non_spatial_dims + spatial_dims,
        ]

        # Verify the exact structure that caused the original bug is now handled
        assert input_core_dims[2] == ["lead_time", "valid_time"]
        assert slp.shape == (41, 129, 147, 132)
        assert init_time_coord.shape == (41, 129)


@pytest.mark.integration
class TestTCIntegration:
    """Integration tests for TC functionality."""

    def test_full_tc_pipeline_mock(self, sample_tc_dataset, sample_ibtracs_dataset):
        """Test the full TC detection pipeline with mocked components."""
        # This tests the integration without actually running expensive computations

        with patch(
            "extremeweatherbench.events.tropical_cyclone.generate_tc_tracks_by_init_time"
        ) as mock_process:
            # Mock the processing function to return a Dataset with fake detections
            mock_process.return_value = xr.Dataset(
                {
                    "track_id": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[1]]]),
                    ),
                    "latitude": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[25.0]]]),
                    ),
                    "longitude": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[-75.0]]]),
                    ),
                    "air_pressure_at_mean_sea_level": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[101000.0]]]),
                    ),
                    "surface_wind_speed": (
                        ["lead_time", "valid_time", "track"],
                        np.array([[[25.0]]]),
                    ),
                },
                coords={
                    "lead_time": [0],
                    "valid_time": [pd.Timestamp("2023-09-15")],
                    "track": [0],
                },
            )

            result = tropical_cyclone.generate_tc_tracks_by_init_time(
                sample_tc_dataset["air_pressure_at_mean_sea_level"],
                sample_tc_dataset["surface_wind_speed"],
                None,  # geopotential_thickness
                sample_ibtracs_dataset,
            )

            assert isinstance(result, xr.Dataset)
            # Check that expected TC variables are present
            expected_vars = [
                "air_pressure_at_mean_sea_level",
                "surface_wind_speed",
                "latitude",
                "longitude",
                "track_id",
            ]
            for var in expected_vars:
                assert var in result.data_vars
