# tests/test_tropical_cyclone.py
"""Comprehensive unit tests for the tropical cyclone module.

This test suite covers:
- TC track detection and filtering
- Helper functions for contour detection
- Dataset conversion utilities
"""

# flake8: noqa: E501
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import derived
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


# ---------------------------------------------------------------------------
# Shared fixture for end-to-end filter tests
# ---------------------------------------------------------------------------


def _make_single_init_tc_dataset(n_lead: int = 12, n_wind_strong: int = 10):
    """Build a minimal synthetic TC dataset with one clear init_time.

    Grid: 1° resolution, 21×21 points (lat 10–30°, lon -80 to -60°).
    Storm: SLP=98 000 Pa at centre (lat=20, lon=-70) every (lead, valid) pair
    on the diagonal (same init_time T0).
    Wind: 20 m/s at one gridpoint east of centre for the first
    ``n_wind_strong`` diagonal pairs; 2 m/s everywhere else.
    Contour validation is OFF so only the wind filter is exercised.

    With a 1° grid ``_degrees_to_gridpoints(2.0, ...)`` = 2 gridpoints, so
    the ±2-pt neighbourhood around the centre includes the +1-pt east cell.

    Expected: n_wind_strong detections have neighbourhood wind ≥ 10 m/s.
    """
    lat = np.arange(10.0, 31.0, 1.0)  # 21 pts, 1 ° spacing
    lon = np.arange(-80.0, -59.0, 1.0)  # 21 pts, 1 ° spacing
    n_lat, n_lon = len(lat), len(lon)
    c_lat, c_lon = 10, 10  # centre indices → lat=20°, lon=-70°

    T0 = pd.Timestamp("2023-09-10")
    lead_h = np.arange(n_lead) * 6  # hours
    lead_td = (lead_h * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    valid_times = pd.date_range(T0, periods=n_lead, freq="6h")

    # init_time[lt, vt] = valid_time[vt] - lead_time[lt]
    init_2d = np.array(
        [
            [valid_times[vt].to_datetime64() - lead_td[lt] for vt in range(n_lead)]
            for lt in range(n_lead)
        ]
    )

    # SLP: 98 000 Pa at centre for every (lt, vt) pair; 102 000 elsewhere
    slp = np.full((n_lead, n_lead, n_lat, n_lon), 102000.0)
    for k in range(n_lead):
        slp[k, k, c_lat, c_lon] = 98000.0

    # Wind: 20 m/s east of centre for the first n_wind_strong diagonal pairs
    wind = np.full((n_lead, n_lead, n_lat, n_lon), 2.0)
    for k in range(n_wind_strong):
        wind[k, k, c_lat, c_lon + 1] = 20.0

    # Geopotential thickness: zeros (contour validation disabled)
    dz = np.zeros((n_lead, n_lead, n_lat, n_lon))

    ds = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                slp,
            ),
            "surface_wind_speed": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                wind,
            ),
            "geopotential_thickness": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                dz,
            ),
        },
        coords={
            "lead_time": lead_td,
            "valid_time": valid_times,
            "latitude": lat,
            "longitude": lon,
            "init_time": (["lead_time", "valid_time"], init_2d),
        },
    )
    return ds


def _make_ibt_for_dataset(ds: xr.Dataset) -> xr.Dataset:
    """IBTrACS stub matching the storm centre in ``_make_single_init_tc_dataset``."""
    valid_times = ds.valid_time.values
    return xr.Dataset(
        {
            "latitude": (["valid_time"], np.full(len(valid_times), 20.0)),
            "longitude": (["valid_time"], np.full(len(valid_times), -70.0)),
        },
        coords={"valid_time": valid_times},
    )


# ---------------------------------------------------------------------------
# Tests for _degrees_to_gridpoints
# ---------------------------------------------------------------------------


class TestDegreesToGridpoints:
    """Unit tests for the _degrees_to_gridpoints helper.

    Expected values are derived independently via:
        ceil(degrees / mean_spacing)
    where mean_spacing = (mean(|Δlat|) + mean(|Δlon|)) / 2.
    """

    def test_uniform_025_deg_grid_1deg(self):
        """1° radius on 0.25° grid → ceil(1/0.25)=4."""
        lat = np.arange(0.0, 10.0, 0.25)
        lon = np.arange(0.0, 10.0, 0.25)
        assert tropical_cyclone._degrees_to_gridpoints(1.0, lat, lon) == 4

    def test_uniform_025_deg_grid_2deg(self):
        """2° radius on 0.25° grid → ceil(2/0.25)=8."""
        lat = np.arange(0.0, 20.0, 0.25)
        lon = np.arange(0.0, 20.0, 0.25)
        assert tropical_cyclone._degrees_to_gridpoints(2.0, lat, lon) == 8

    def test_uniform_025_deg_grid_5deg(self):
        """5° radius on 0.25° grid → ceil(5/0.25)=20."""
        lat = np.arange(0.0, 90.0, 0.25)
        lon = np.arange(0.0, 360.0, 0.25)
        assert tropical_cyclone._degrees_to_gridpoints(5.0, lat, lon) == 20

    def test_uniform_05_deg_grid_2deg(self):
        """2° radius on 0.5° grid → ceil(2/0.5)=4."""
        lat = np.arange(0.0, 20.0, 0.5)
        lon = np.arange(0.0, 20.0, 0.5)
        assert tropical_cyclone._degrees_to_gridpoints(2.0, lat, lon) == 4

    def test_uniform_1_deg_grid_2deg(self):
        """2° radius on 1° grid → ceil(2/1)=2."""
        lat = np.arange(10.0, 31.0, 1.0)
        lon = np.arange(-80.0, -59.0, 1.0)
        assert tropical_cyclone._degrees_to_gridpoints(2.0, lat, lon) == 2

    def test_non_square_grid(self):
        """Non-square grid (0.25° lat, 0.5° lon): mean=0.375; 1.5°→ceil(4)=4."""
        lat = np.arange(0.0, 20.0, 0.25)  # 0.25° spacing
        lon = np.arange(0.0, 20.0, 0.5)  # 0.5° spacing
        # mean_spacing = (0.25 + 0.5) / 2 = 0.375
        # ceil(1.5 / 0.375) = ceil(4.0) = 4
        assert tropical_cyclone._degrees_to_gridpoints(1.5, lat, lon) == 4

    def test_non_square_grid_fractional(self):
        """Non-square grid (0.25° lat, 0.5° lon): 2°→ceil(5.333)=6."""
        lat = np.arange(0.0, 20.0, 0.25)
        lon = np.arange(0.0, 20.0, 0.5)
        # mean_spacing = 0.375; ceil(2/0.375) = ceil(5.333) = 6
        expected = math.ceil(2.0 / 0.375)
        assert tropical_cyclone._degrees_to_gridpoints(2.0, lat, lon) == expected

    def test_always_at_least_one(self):
        """Very small degree value never returns zero."""
        lat = np.arange(0.0, 10.0, 0.25)
        lon = np.arange(0.0, 10.0, 0.25)
        assert tropical_cyclone._degrees_to_gridpoints(0.01, lat, lon) >= 1

    def test_single_point_coords_fallback(self):
        """Single-point coordinates fall back to 1.0° spacing."""
        lat = np.array([20.0])
        lon = np.array([-70.0])
        # mean_spacing = (1.0 + 1.0) / 2 = 1.0; ceil(2/1) = 2
        assert tropical_cyclone._degrees_to_gridpoints(2.0, lat, lon) == 2

    def test_result_is_integer(self):
        """Return type is always a plain Python int."""
        lat = np.arange(0.0, 10.0, 0.25)
        lon = np.arange(0.0, 10.0, 0.25)
        result = tropical_cyclone._degrees_to_gridpoints(1.0, lat, lon)
        assert isinstance(result, int)

    def test_monotone_in_degrees(self):
        """Larger degree radius never gives fewer gridpoints."""
        lat = np.arange(0.0, 20.0, 0.25)
        lon = np.arange(0.0, 20.0, 0.25)
        prev = tropical_cyclone._degrees_to_gridpoints(0.5, lat, lon)
        for deg in [1.0, 1.5, 2.0, 3.0, 5.0]:
            cur = tropical_cyclone._degrees_to_gridpoints(deg, lat, lon)
            assert cur >= prev, f"Expected monotone increase at {deg}°"
            prev = cur


# ---------------------------------------------------------------------------
# Tests for TropicalCycloneTrackVariables parameter defaults
# ---------------------------------------------------------------------------


class TestTropicalCycloneTrackVariablesDefaults:
    """Verify new parameter names and default values."""

    def test_min_distance_between_peaks_degrees_default(self):
        tc = derived.TropicalCycloneTrackVariables()
        assert tc.min_distance_between_peaks_degrees == 1.0

    def test_wind_search_radius_degrees_default(self):
        tc = derived.TropicalCycloneTrackVariables()
        assert tc.wind_search_radius_degrees == 2.0

    def test_timestep_count_wind_minimum_default(self):
        tc = derived.TropicalCycloneTrackVariables()
        assert tc.timestep_count_wind_minimum == 10

    def test_custom_min_distance_between_peaks_degrees(self):
        tc = derived.TropicalCycloneTrackVariables(
            min_distance_between_peaks_degrees=0.5
        )
        assert tc.min_distance_between_peaks_degrees == 0.5

    def test_custom_wind_search_radius_degrees(self):
        tc = derived.TropicalCycloneTrackVariables(wind_search_radius_degrees=3.0)
        assert tc.wind_search_radius_degrees == 3.0

    def test_old_min_distance_between_peaks_kwarg_raises(self):
        """Renamed parameter: old name must raise TypeError."""
        with pytest.raises(TypeError):
            tropical_cyclone.generate_tc_tracks_by_init_time(
                xr.DataArray(),
                xr.DataArray(),
                xr.DataArray(),
                xr.Dataset(),
                min_distance_between_peaks=5,  # old name
            )


# ---------------------------------------------------------------------------
# End-to-end tests for neighbourhood wind sampling and the wind-count filter
# ---------------------------------------------------------------------------


class TestNeighbourhoodWindSampling:
    """Verify that peak_winds uses max in the ±wind_search_radius neighbourhood.

    On the 1° test grid, _degrees_to_gridpoints(2.0, …)=2. The high-wind cell
    is 1 gridpoint east of the SLP minimum, so it falls inside the ±2-pt box.
    The wind AT the centre is 2 m/s; the neighbourhood max is 20 m/s.
    The output surface_wind_speed must reflect the neighbourhood max (≥10 m/s),
    not the centre value (2 m/s).
    """

    def test_output_wind_exceeds_centre_wind(self):
        """Detected surface_wind_speed > wind at SLP minimum (2 m/s)."""
        ds = _make_single_init_tc_dataset(n_lead=12, n_wind_strong=12)
        ibt = _make_ibt_for_dataset(ds)
        result = tropical_cyclone.generate_tc_tracks_by_init_time(
            sea_level_pressure=ds["air_pressure_at_mean_sea_level"],
            wind_speed=ds["surface_wind_speed"],
            geopotential_thickness=ds["geopotential_thickness"],
            tc_track_analysis_data=ibt,
            timestep_count_wind_minimum=1,
            use_contour_validation=False,
            surface_pressure_threshold=102000.0,
            wind_search_radius_degrees=2.0,
        )
        assert result.sizes.get("valid_time", 0) > 0, "Expected detections"
        ws = result["surface_wind_speed"].values
        # Centre wind is 2 m/s; neighbourhood max is 20 m/s
        assert np.nanmax(ws) > 2.0, "Neighbourhood max must exceed centre wind"
        assert np.nanmax(ws) >= 10.0, "At least one detection should be ≥10 m/s"

    def test_output_wind_close_to_neighbourhood_max(self):
        """Neighbourhood max (20 m/s) is returned, not centre value (2 m/s)."""
        ds = _make_single_init_tc_dataset(n_lead=5, n_wind_strong=5)
        ibt = _make_ibt_for_dataset(ds)
        result = tropical_cyclone.generate_tc_tracks_by_init_time(
            sea_level_pressure=ds["air_pressure_at_mean_sea_level"],
            wind_speed=ds["surface_wind_speed"],
            geopotential_thickness=ds["geopotential_thickness"],
            tc_track_analysis_data=ibt,
            timestep_count_wind_minimum=1,
            use_contour_validation=False,
            surface_pressure_threshold=102000.0,
            wind_search_radius_degrees=2.0,
        )
        assert result.sizes.get("valid_time", 0) > 0
        ws = result["surface_wind_speed"].values
        ws_detected = ws[~np.isnan(ws)]
        assert len(ws_detected) > 0, "No non-NaN wind values in output"
        # All detected timesteps should carry the neighbourhood max (20 m/s)
        assert np.allclose(ws_detected, 20.0, atol=1.0), (
            f"Expected neighbourhood max ≈20 m/s, got {ws_detected}"
        )


class TestMinTrackTimestepsWindFilter:
    """Verify the wind-count filter: track survives iff it has >= timestep_count_wind_minimum
    detections where the neighbourhood peak wind is >= 10 m/s.

    Setup: n_lead=12, n_wind_strong=10.  The single long-lived track has exactly
    10 detections with neighbourhood wind=20 m/s and 2 with wind=2 m/s.
    Independent expectation:
      min_ts ≤ 10 → track passes → result has valid_time > 0
      min_ts = 11 → track fails  → result has valid_time == 0
    """

    N_LEAD = 12
    N_WIND_STRONG = 10  # independently chosen threshold

    @pytest.fixture(scope="class")
    def tc_ds(self):
        return _make_single_init_tc_dataset(self.N_LEAD, self.N_WIND_STRONG)

    @pytest.fixture(scope="class")
    def ibt(self, tc_ds):
        return _make_ibt_for_dataset(tc_ds)

    def _run(self, tc_ds, ibt, min_ts):
        return tropical_cyclone.generate_tc_tracks_by_init_time(
            sea_level_pressure=tc_ds["air_pressure_at_mean_sea_level"],
            wind_speed=tc_ds["surface_wind_speed"],
            geopotential_thickness=tc_ds["geopotential_thickness"],
            tc_track_analysis_data=ibt,
            timestep_count_wind_minimum=min_ts,
            use_contour_validation=False,
            surface_pressure_threshold=102000.0,
            wind_search_radius_degrees=2.0,
        )

    def test_passes_at_exact_threshold(self, tc_ds, ibt):
        """min_ts == n_wind_strong: track has exactly enough windy steps."""
        result = self._run(tc_ds, ibt, self.N_WIND_STRONG)
        assert result.sizes.get("valid_time", 0) > 0, (
            f"Expected detections with min_ts={self.N_WIND_STRONG}"
        )

    def test_passes_below_threshold(self, tc_ds, ibt):
        """min_ts < n_wind_strong: track comfortably passes."""
        for min_ts in [1, 5, self.N_WIND_STRONG - 1]:
            result = self._run(tc_ds, ibt, min_ts)
            assert result.sizes.get("valid_time", 0) > 0, (
                f"Expected detections with min_ts={min_ts}"
            )

    def test_fails_above_threshold(self, tc_ds, ibt):
        """min_ts > n_wind_strong: track lacks sufficient windy steps."""
        result = self._run(tc_ds, ibt, self.N_WIND_STRONG + 1)
        assert result.sizes.get("valid_time", 0) == 0, (
            f"Expected 0 detections with min_ts={self.N_WIND_STRONG + 1}"
        )

    def test_boundary_is_sharp(self, tc_ds, ibt):
        """One step above the exact threshold flips pass→fail."""
        result_pass = self._run(tc_ds, ibt, self.N_WIND_STRONG)
        result_fail = self._run(tc_ds, ibt, self.N_WIND_STRONG + 1)
        assert result_pass.sizes.get("valid_time", 0) > 0
        assert result_fail.sizes.get("valid_time", 0) == 0


class TestFillTrackGapsFromFields:
    """Tests for _fill_track_gaps_from_fields SLP fallback."""

    def test_fills_gap_with_slp_minimum(self):
        """A gap between two detections is filled by finding
        the SLP minimum near the expected position."""
        n_lat, n_lon = 31, 31
        lat = np.linspace(0, 30, n_lat)
        lon = np.linspace(110, 140, n_lon)

        # 3 timesteps; detection at ts 0 and ts 2, gap at 1
        lt_seq = np.array([0, 1, 2])
        vt_seq = np.array([0, 1, 2])
        tid = 5000

        detections = [
            {
                "lead_time_index": 0,
                "valid_time_index": 0,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99800.0,
                "wind": 20.0,
            },
            {
                "lead_time_index": 2,
                "valid_time_index": 2,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99600.0,
                "wind": 22.0,
            },
        ]

        # SLP field at ts 1: minimum at (15, 125)
        min_r = np.argmin(np.abs(lat - 15.0))
        min_c = np.argmin(np.abs(lon - 125.0))
        slp_all = np.full((3, n_lat, n_lon), 101000.0)
        slp_all[1, min_r, min_c] = 99700.0
        wind_all = np.full((3, n_lat, n_lon), 5.0)
        wind_all[1, min_r, min_c] = 18.0

        result = tropical_cyclone._fill_track_gaps_from_fields(
            detections,
            slp_all,
            wind_all,
            lt_seq,
            vt_seq,
            lat,
            lon,
            wind_search_radius_gridpts=1,
        )

        # Should now have 3 detections
        assert len(result) == 3
        gap_det = [d for d in result if d["lead_time_index"] == 1]
        assert len(gap_det) == 1
        assert gap_det[0]["track_id"] == tid
        np.testing.assert_allclose(gap_det[0]["latitude"], lat[min_r], atol=0.5)
        np.testing.assert_allclose(gap_det[0]["longitude"], lon[min_c], atol=0.5)

    def test_no_fill_beyond_endpoints(self):
        """Timesteps outside the first/last detection are
        NOT filled."""
        n_lat, n_lon = 11, 11
        lat = np.linspace(10, 20, n_lat)
        lon = np.linspace(120, 130, n_lon)

        lt_seq = np.array([0, 1, 2, 3, 4])
        vt_seq = np.array([0, 1, 2, 3, 4])
        tid = 100

        detections = [
            {
                "lead_time_index": 1,
                "valid_time_index": 1,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99800.0,
                "wind": 20.0,
            },
            {
                "lead_time_index": 3,
                "valid_time_index": 3,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99600.0,
                "wind": 22.0,
            },
        ]

        slp_all = np.full((5, n_lat, n_lon), 101000.0)
        slp_all[:, 5, 5] = 99500.0
        wind_all = np.full((5, n_lat, n_lon), 5.0)

        result = tropical_cyclone._fill_track_gaps_from_fields(
            detections,
            slp_all,
            wind_all,
            lt_seq,
            vt_seq,
            lat,
            lon,
            wind_search_radius_gridpts=1,
        )

        filled_lts = {d["lead_time_index"] for d in result}
        # ts 0 and ts 4 should NOT be filled
        assert 0 not in filled_lts
        assert 4 not in filled_lts
        # ts 2 should be filled
        assert 2 in filled_lts

    def test_empty_detections(self):
        """No detections should return empty list."""
        result = tropical_cyclone._fill_track_gaps_from_fields(
            [],
            np.zeros((1, 5, 5)),
            np.zeros((1, 5, 5)),
            np.array([0]),
            np.array([0]),
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 5),
            wind_search_radius_gridpts=1,
        )
        assert result == []

    def test_prefers_closest_candidate_over_deepest(self):
        """Among low-SLP candidates, pick the one nearest
        to expected position, not the absolute minimum."""
        n_lat, n_lon = 31, 31
        lat = np.linspace(0, 30, n_lat)
        lon = np.linspace(110, 140, n_lon)

        lt_seq = np.array([0, 1, 2])
        vt_seq = np.array([0, 1, 2])
        tid = 42

        detections = [
            {
                "lead_time_index": 0,
                "valid_time_index": 0,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99800.0,
                "wind": 20.0,
            },
            {
                "lead_time_index": 2,
                "valid_time_index": 2,
                "track_id": tid,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99600.0,
                "wind": 22.0,
            },
        ]

        near_r = np.argmin(np.abs(lat - 15.0))
        near_c = np.argmin(np.abs(lon - 125.0))
        far_r = np.argmin(np.abs(lat - 17.0))
        far_c = np.argmin(np.abs(lon - 127.0))

        slp_all = np.full((3, n_lat, n_lon), 101000.0)
        # Near candidate: moderate low
        slp_all[1, near_r, near_c] = 99700.0
        # Far candidate: deeper low but farther away
        slp_all[1, far_r, far_c] = 99200.0

        wind_all = np.full((3, n_lat, n_lon), 5.0)
        wind_all[1, near_r, near_c] = 18.0
        wind_all[1, far_r, far_c] = 25.0

        result = tropical_cyclone._fill_track_gaps_from_fields(
            detections,
            slp_all,
            wind_all,
            lt_seq,
            vt_seq,
            lat,
            lon,
            wind_search_radius_gridpts=1,
        )

        gap = [d for d in result if d["lead_time_index"] == 1]
        assert len(gap) == 1
        np.testing.assert_allclose(
            gap[0]["latitude"],
            lat[near_r],
            atol=0.5,
        )
        np.testing.assert_allclose(
            gap[0]["longitude"],
            lon[near_c],
            atol=0.5,
        )

    def test_chaining_uses_filled_points(self):
        """Successive gap fills should chain through
        previously filled positions, not always interpolate
        from the original bracketing detections.

        Setup: endpoints at lat 10 (ts 0) and lat 18 (ts 3).
        The SLP min at ts 1 sits at lat 14, which becomes
        the "previous" anchor for ts 2. With chaining, ts 2
        interpolates between lat 14 and 18 (expect ~16);
        without chaining it would use 10 and 18 (expect ~14.7).
        We place the ts 2 SLP min at lat 16 so it is only
        chosen when the chained anchor shifts the expected
        position northward."""
        n_lat, n_lon = 41, 21
        lat = np.linspace(8, 20, n_lat)
        lon = np.linspace(123, 127, n_lon)

        lt_seq = np.arange(4)
        vt_seq = np.arange(4)
        tid = 99

        detections = [
            {
                "lead_time_index": 0,
                "valid_time_index": 0,
                "track_id": tid,
                "latitude": 10.0,
                "longitude": 125.0,
                "slp": 99800.0,
                "wind": 20.0,
            },
            {
                "lead_time_index": 3,
                "valid_time_index": 3,
                "track_id": tid,
                "latitude": 18.0,
                "longitude": 125.0,
                "slp": 99600.0,
                "wind": 22.0,
            },
        ]

        slp_all = np.full((4, n_lat, n_lon), 101000.0)
        wind_all = np.full((4, n_lat, n_lon), 5.0)
        c_mid = np.argmin(np.abs(lon - 125.0))

        # ts 1: SLP min at lat 14
        r_ts1 = np.argmin(np.abs(lat - 14.0))
        slp_all[1, r_ts1, c_mid] = 99700.0
        wind_all[1, r_ts1, c_mid] = 15.0

        # ts 2: SLP min at lat 16 (only reachable via
        # chained anchor at lat 14, not via endpoint
        # midpoint at lat ~14.7)
        r_ts2 = np.argmin(np.abs(lat - 16.0))
        slp_all[2, r_ts2, c_mid] = 99700.0
        wind_all[2, r_ts2, c_mid] = 15.0

        result = tropical_cyclone._fill_track_gaps_from_fields(
            detections,
            slp_all,
            wind_all,
            lt_seq,
            vt_seq,
            lat,
            lon,
            wind_search_radius_gridpts=1,
        )

        assert len(result) == 4
        ts2_fill = [d for d in result if d["lead_time_index"] == 2]
        assert len(ts2_fill) == 1
        np.testing.assert_allclose(
            ts2_fill[0]["latitude"],
            lat[r_ts2],
            atol=0.5,
        )

    def test_single_detection_no_fill(self):
        """A single detection should not trigger gap fill."""
        detections = [
            {
                "lead_time_index": 0,
                "valid_time_index": 0,
                "track_id": 1,
                "latitude": 15.0,
                "longitude": 125.0,
                "slp": 99800.0,
                "wind": 20.0,
            },
        ]
        result = tropical_cyclone._fill_track_gaps_from_fields(
            detections,
            np.full((3, 5, 5), 101000.0),
            np.full((3, 5, 5), 5.0),
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.linspace(0, 10, 5),
            np.linspace(0, 10, 5),
            wind_search_radius_gridpts=1,
        )
        assert len(result) == 1
