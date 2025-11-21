"""Tests for the utils module."""

import datetime
import operator

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from extremeweatherbench import utils


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (180, 180), (360, 0), (-179, 181), (-360, 0), (540, 180), (359, 359)],
)
def test_convert_longitude_to_360_param(input, expected):
    assert utils.convert_longitude_to_360(input) == expected


def test_convert_longitude_to_180():
    """Test converting longitude from [0, 360) to [-180, 180) range."""
    # Test scalar values
    assert utils.convert_longitude_to_180(0) == 0
    assert utils.convert_longitude_to_180(180) == -180
    assert utils.convert_longitude_to_180(270) == -90
    assert utils.convert_longitude_to_180(359) == -1

    # Test with xarray Dataset (note: result is sorted by longitude)
    ds = xr.Dataset(coords={"longitude": [0, 90, 180, 270, 359]})
    converted_ds = utils.convert_longitude_to_180(ds)
    # After conversion and sorting: -180, -90, -1, 0, 90
    expected_lons_sorted = [-180, -90, -1, 0, 90]
    np.testing.assert_allclose(converted_ds.longitude.values, expected_lons_sorted)

    # Test with custom longitude name
    ds_custom = xr.Dataset(coords={"lon": [0, 90, 180, 270]})
    converted_custom = utils.convert_longitude_to_180(ds_custom, longitude_name="lon")
    # After conversion and sorting: -180, -90, 0, 90
    expected_custom_sorted = [-180, -90, 0, 90]
    np.testing.assert_allclose(converted_custom.lon.values, expected_custom_sorted)


def test_remove_ocean_gridpoints():
    """Test removing ocean gridpoints from dataset."""
    # Create a simple test dataset with known land/ocean points
    ds = xr.Dataset(
        data_vars={"temperature": (["latitude", "longitude"], [[1, 2], [3, 4]])},
        coords={
            "latitude": [40.0, 41.0],  # Land coordinates (roughly US)
            "longitude": [260.0, 261.0],  # Convert from -100, -99
        },
    )

    result = utils.remove_ocean_gridpoints(ds)

    # Should return a dataset (may have NaNs for ocean points)
    assert isinstance(result, xr.Dataset)
    assert "temperature" in result.data_vars
    assert result.sizes == ds.sizes


def test_load_events_yaml():
    """Test loading events yaml file."""
    result = utils.load_events_yaml()

    # Should return a dictionary
    assert isinstance(result, dict)
    # Should contain 'cases' key (based on the existing yaml structure)
    assert "cases" in result


def test_read_event_yaml(tmp_path):
    """Test reading events yaml from file."""
    # Create a temporary yaml file
    yaml_content = {
        "cases": {"test_case": {"start_date": "2020-01-01", "end_date": "2020-01-02"}}
    }

    yaml_file = tmp_path / "test_events.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)

    result = utils.read_event_yaml(yaml_file)

    assert isinstance(result, dict)
    assert "cases" in result
    assert "test_case" in result["cases"]
    assert result["cases"]["test_case"]["start_date"] == "2020-01-01"


def test_derive_indices_from_init_time_and_lead_time():
    """Test deriving indices from init_time and lead_time coordinates."""
    # Create test dataset
    ds = xr.Dataset(
        coords={
            "init_time": pd.date_range("2020-01-01", "2020-01-03", freq="D"),
            "lead_time": [0, 24, 48],  # hours
        }
    )

    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2020, 1, 4)

    indices = utils.derive_indices_from_init_time_and_lead_time(
        ds, start_date, end_date
    )

    # Should return tuple of arrays (init_time_indices, lead_time_indices)
    assert isinstance(indices, tuple)
    assert len(indices) == 2
    assert isinstance(indices[0], np.ndarray)
    assert isinstance(indices[1], np.ndarray)


def test_filter_kwargs_for_callable():
    """Test filtering kwargs to match callable signature."""

    # Define test function
    def test_func(a, b, c=None):
        return a + b

    # Test with matching kwargs
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}  # d should be filtered out
    filtered = utils.filter_kwargs_for_callable(kwargs, test_func)

    expected = {"a": 1, "b": 2, "c": 3}
    assert filtered == expected

    # Test with method (bound method)
    class TestClass:
        def method(self, x, y):
            return x + y

    obj = TestClass()
    kwargs_method = {"x": 1, "y": 2, "z": 3}  # z should be filtered out
    filtered_method = utils.filter_kwargs_for_callable(kwargs_method, obj.method)

    # Bound methods already have 'self' excluded from signature
    expected_method = {"x": 1, "y": 2}
    assert filtered_method == expected_method


def test_min_if_all_timesteps_present():
    """Test returning minimum if all timesteps are present."""
    # Test with complete timesteps
    da_complete = xr.DataArray([1, 2, 3, 4], dims=["time"])
    result_complete = utils.min_if_all_timesteps_present(da_complete, 6)

    # Should return minimum value
    assert result_complete.values == 1

    # Test with incomplete timesteps
    da_incomplete = xr.DataArray([1, 2, 3], dims=["time"])
    result_incomplete = utils.min_if_all_timesteps_present(da_incomplete, 6)

    # Should return NaN
    assert np.isnan(result_incomplete.values)


def test_min_if_all_timesteps_present_forecast():
    """Test returning minimum for forecast with valid_time dimension."""
    # Test with complete timesteps
    da_complete = xr.DataArray(
        [[1, 2, 3], [4, 5, 6]],
        dims=["lead_time", "valid_time"],
        coords={"lead_time": [0, 6], "valid_time": [0, 1, 2]},
    )
    result_complete = utils.min_if_all_timesteps_present_forecast(da_complete, 8)

    # Should return minimum along valid_time dimension
    expected = xr.DataArray([1, 4], dims=["lead_time"], coords={"lead_time": [0, 6]})
    xr.testing.assert_equal(result_complete, expected)

    # Test with incomplete timesteps
    da_incomplete = xr.DataArray(
        [[1, 2], [4, 5]],
        dims=["lead_time", "valid_time"],
        coords={"lead_time": [0, 6], "valid_time": [0, 1]},
    )
    result_incomplete = utils.min_if_all_timesteps_present_forecast(da_incomplete, 8)

    # Should return NaN array with same lead_time dimension
    assert len(result_incomplete.lead_time) == 2
    assert np.all(np.isnan(result_incomplete.values))


def test_determine_temporal_resolution():
    """Test determining time resolution in hours."""
    # Create dataset with 6-hourly resolution
    times = pd.date_range("2020-01-01", "2020-01-02", freq="6h")[:-1]  # 4 timesteps/day
    ds = xr.Dataset(
        data_vars={"temp": (["valid_time"], [1, 2, 3, 4])}, coords={"valid_time": times}
    )

    result = utils.determine_temporal_resolution(ds)
    assert result == 6  # 6-hour resolution

    # Test with hourly resolution
    times_hourly = pd.date_range("2020-01-01", "2020-01-02", freq="1h")[:-1]
    ds_hourly = xr.Dataset(
        data_vars={"temp": (["valid_time"], range(24))},
        coords={"valid_time": times_hourly},
    )

    result_hourly = utils.determine_temporal_resolution(ds_hourly)
    assert result_hourly == 1  # 1-hour resolution


def test_determine_temporal_resolution_multiple_resolutions():
    """Test that max resolution is returned when multiple resolutions exist."""
    # Create dataset with mixed time resolutions - some 1h gaps, some 6h gaps
    # This simulates missing data where some timesteps are present at 1h
    # resolution but others have 6h gaps
    times_mixed = pd.to_datetime(
        [
            "2020-01-01 00:00",  # Start
            "2020-01-01 01:00",  # 1h gap
            "2020-01-01 02:00",  # 1h gap
            "2020-01-01 08:00",  # 6h gap (missing 03:00-07:00)
            "2020-01-01 09:00",  # 1h gap
            "2020-01-01 15:00",  # 6h gap (missing 10:00-14:00)
        ]
    )

    ds_mixed = xr.Dataset(
        data_vars={"temp": (["valid_time"], [1, 2, 3, 4, 5, 6])},
        coords={"valid_time": times_mixed, "init_time": [pd.Timestamp("2020-01-01")]},
    )

    result = utils.determine_temporal_resolution(ds_mixed)

    # Should return 1 (minimum time gap in hours) when multiple resolutions
    # are present, even though some gaps are 6 hours
    # This confirms the function takes the minimum gap, not the maximum
    assert result == 1


def test_convert_init_time_to_valid_time():
    """Test converting init_time coordinate to valid_time."""
    # Create test dataset
    ds = xr.Dataset(
        data_vars={"temp": (["init_time", "lead_time"], [[1, 2], [3, 4]])},
        coords={
            "init_time": pd.date_range("2020-01-01", periods=2, freq="D"),
            "lead_time": [0, 24],  # hours
        },
    )

    result = utils.convert_init_time_to_valid_time(ds)

    # Should have valid_time coordinate
    assert "valid_time" in result.coords
    # Should maintain lead_time dimension
    assert "lead_time" in result.dims
    # Should have swapped init_time for valid_time in the primary dimension
    assert "init_time" not in result.dims or "valid_time" in result.dims


def test_maybe_get_closest_timestamp_to_center_of_valid_times_single_output():
    """Test passthrough behavior with single output time."""
    # Create valid_time values (center will be middle value)
    valid_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=5, freq="6h"), dims=["valid_time"]
    )

    # Single output time
    output_time = xr.DataArray([pd.Timestamp("2021-01-01 06:00")], dims=["time"])

    result = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
        output_time, valid_times
    )

    # Should return the same single output time (passthrough)
    xr.testing.assert_equal(result, output_time)


def test_maybe_get_closest_timestamp_to_center_of_valid_times_two_outputs():
    """Test closest selection with two output times."""
    # Create valid_time values (center: 2021-01-01 12:00)
    valid_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=5, freq="6h"), dims=["valid_time"]
    )

    # Two output times - one closer to center than the other
    output_times = xr.DataArray(
        [
            pd.Timestamp("2021-01-01 06:00"),  # 6 hours from center
            pd.Timestamp("2021-01-01 15:00"),  # 3 hours from center
        ],
        dims=["time"],
    )

    result = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
        output_times, valid_times
    )

    # Should return the closer one (15:00)
    expected = xr.DataArray(pd.Timestamp("2021-01-01 15:00"))
    xr.testing.assert_equal(result, expected)


def test_maybe_get_closest_timestamp_to_center_of_valid_times_three_outputs():
    """Test closest selection with three output times."""
    # Create valid_time values (center: 2021-01-01 12:00)
    valid_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=5, freq="6h"), dims=["valid_time"]
    )

    # Three output times with varying distances from center
    output_times = xr.DataArray(
        [
            pd.Timestamp("2021-01-01 03:00"),  # 9 hours from center
            pd.Timestamp("2021-01-01 11:00"),  # 1 hour from center (closest)
            pd.Timestamp("2021-01-01 18:00"),  # 6 hours from center
        ],
        dims=["time"],
    )

    result = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
        output_times, valid_times
    )

    # Should return the closest one (11:00)
    expected = xr.DataArray(pd.Timestamp("2021-01-01 11:00"))
    xr.testing.assert_equal(result, expected)


def test_maybe_get_closest_timestamp_to_center_of_valid_times_many_outputs():
    """Test closest selection with many (20) output times."""
    # Create valid_time values (center: 2021-01-02 12:00)
    valid_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=9, freq="6h"), dims=["valid_time"]
    )

    # 20 output times spread over several days
    output_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=20, freq="3h"), dims=["time"]
    )

    result = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
        output_times, valid_times
    )

    # Center time should be valid_times[4] = 2021-01-02 12:00
    center_time = valid_times.values[4]  # 2021-01-02 12:00

    # Find which output time is actually closest
    time_diffs = np.abs(output_times - center_time)
    closest_idx = np.argmin(time_diffs.data)
    expected = output_times[closest_idx]

    xr.testing.assert_equal(result, expected)


def test_maybe_get_closest_timestamp_to_center_of_valid_times_even_valid_times():
    """Test with even number of valid_times (center calculation)."""
    # Create even number of valid_time values (center: index 2, 2021-01-01 12:00)
    valid_times = xr.DataArray(
        pd.date_range("2021-01-01", periods=4, freq="6h"), dims=["valid_time"]
    )

    # Multiple output times
    output_times = xr.DataArray(
        [
            pd.Timestamp("2021-01-01 09:00"),  # 3 hours from center
            pd.Timestamp("2021-01-01 13:00"),  # 1 hour from center (closest)
            pd.Timestamp("2021-01-01 21:00"),  # 9 hours from center
        ],
        dims=["time"],
    )

    result = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
        output_times, valid_times
    )

    # Should return the closest one (13:00)
    expected = xr.DataArray(pd.Timestamp("2021-01-01 13:00"))
    xr.testing.assert_equal(result, expected)


class TestStackSparseDataFromDims:
    """Test the stack_dataarray_from_dims function."""

    def test_basic_functionality(self):
        """Test basic functionality with valid sparse data."""
        import sparse

        # Create a simple sparse array with known coordinates
        coords = ([0, 1, 2], [0, 1, 0])  # (lat_indices, lon_indices)
        data = [1.0, 2.0, 3.0]  # values at those coordinates
        shape = (3, 2)  # (lat, lon)

        sparse_array = sparse.COO(coords, data, shape=shape)

        # Create xarray DataArray with sparse data
        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        # Test stacking both dimensions
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        # Should return a DataArray with stacked dimension
        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        # Should be densified (no longer sparse)
        assert not isinstance(result.data, sparse.COO)

    def test_empty_sparse_data(self):
        """Test edge case with empty sparse data."""
        import sparse

        # Create empty sparse array
        coords = ([], [])  # No coordinates
        data = []  # No data
        shape = (3, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        # Test with empty data
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        # Should handle empty data gracefully
        assert isinstance(result, xr.DataArray)
        assert result.size == 0

    def test_single_dimension_stack(self):
        """Test with single dimension to stack."""
        import sparse

        # Create sparse array with data in one dimension
        coords = ([0, 2], [0, 0])  # Only latitude varies
        data = [1.0, 2.0]
        shape = (3, 1)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0]},
        )

        # Test stacking only latitude
        result = utils.stack_dataarray_from_dims(da, stack_dims=["latitude"])

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        # Should preserve the longitude dimension
        assert "longitude" in result.dims

    def test_multiple_dimensions_stack(self):
        """Test with multiple dimensions to stack."""
        import sparse

        # Create 3D sparse array
        coords = ([0, 1, 2], [0, 1, 0], [0, 0, 1])  # (time, lat, lon)
        data = [1.0, 2.0, 3.0]
        shape = (3, 2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0],
            },
        )

        # Test stacking lat and lon, keeping time
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        assert "time" in result.dims

    def test_max_size_parameter(self):
        """Test max_size parameter behavior."""
        import sparse

        # Create sparse array
        coords = ([0, 1], [0, 1])
        data = [1.0, 2.0]
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        # Test with different max_size values
        result_small = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"], max_size=1
        )
        result_large = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"], max_size=1000000
        )

        # Both should work but may have different internal representations
        assert isinstance(result_small, xr.DataArray)
        assert isinstance(result_large, xr.DataArray)

    def test_invalid_dimensions(self):
        """Test error handling with invalid dimensions."""
        import sparse

        coords = ([0, 1], [0, 1])
        data = [1.0, 2.0]
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        # Test with non-existent dimension
        with pytest.raises((ValueError, KeyError)):
            utils.stack_dataarray_from_dims(da, stack_dims=["nonexistent"])

    def test_no_sparse_coordinates(self):
        """Test with sparse data that has minimal coordinates."""
        import sparse

        # Create sparse array with single point
        coords = ([0], [0])
        data = [5.0]
        shape = (1, 1)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0], "longitude": [100.0]},
        )

        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims

    def test_duplicate_coordinate_values(self):
        """Test with duplicate coordinate values in sparse data."""
        import sparse

        # Create sparse array where multiple sparse indices map to same coords
        coords = ([0, 0, 1], [0, 1, 0])  # Two points at lat[0]
        data = [1.0, 2.0, 3.0]
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        # Should handle duplicate coordinate scenarios
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims

    def test_zero_size_array(self):
        """Test with zero-size sparse array."""
        import sparse

        # Create zero-size sparse array
        sparse_array = sparse.COO([], [], shape=(0, 0))

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [], "longitude": []},
        )

        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        # With empty data, function returns original DataArray with densified data
        assert isinstance(result, xr.DataArray)
        assert result.size == 0
        # Should preserve original dimensions for empty data
        assert "latitude" in result.dims
        assert "longitude" in result.dims
        # Data should be densified (no longer sparse)
        assert not isinstance(result.data, sparse.COO)

    def test_large_sparse_array_with_small_max_size(self):
        """Test behavior with large sparse array and small max_size."""
        import sparse

        # Create a larger sparse array
        n_points = 1000
        coords = (
            np.random.randint(0, 100, n_points),
            np.random.randint(0, 100, n_points),
        )
        data = np.random.random(n_points)
        shape = (100, 100)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(0, 90, 100),
                "longitude": np.linspace(-180, 180, 100),
            },
        )

        # Test with very small max_size
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"], max_size=10
        )

        assert isinstance(result, xr.DataArray)
        # Should still work even with small max_size

    def test_all_dimensions_stacked(self):
        """Test stacking all dimensions of the array."""
        import sparse

        # Create 3D sparse array
        coords = ([0, 1, 2], [0, 1, 0], [1, 0, 1])
        data = [1.0, 2.0, 3.0]
        shape = (3, 2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0],
            },
        )

        # Stack all dimensions
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["time", "latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        # Should only have the stacked dimension
        assert len(result.dims) == 1

    def test_non_sparse_data_input(self):
        """Test that function handles non-sparse data appropriately."""
        # Create regular (dense) DataArray
        da = xr.DataArray(
            np.random.random((3, 2)),
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        # Should handle non-sparse data by using regular stacking
        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        # Verify it creates a stacked dimension
        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        # Should have 3 * 2 = 6 stacked coordinates
        assert len(result.stacked) == 6
        # Original dimensions should be gone
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims

    def test_empty_stack_dims_list(self):
        """Test with empty stack_dims list."""
        import sparse

        coords = ([0, 1], [0, 1])
        data = [1.0, 2.0]
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        # Test with empty stack_dims - should raise error
        with pytest.raises((ValueError, IndexError)):
            utils.stack_dataarray_from_dims(da, stack_dims=[])

    def test_coordinate_value_extraction(self):
        """Test that coordinate values are correctly extracted."""
        import sparse

        # Create sparse array with specific pattern
        coords = ([0, 2], [1, 0])  # Specific lat/lon indices
        data = [10.0, 20.0]
        shape = (3, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [1.0, 2.0, 3.0],  # lat[0]=1.0, lat[2]=3.0
                "longitude": [100.0, 200.0],  # lon[1]=200.0, lon[0]=100.0
            },
        )

        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims
        # Check that we have the expected coordinate combinations
        # Should have coordinates for (lat[0], lon[1]) and (lat[2], lon[0])
        # which are (1.0, 200.0) and (3.0, 100.0)
        assert len(result.stacked) == 2

    def test_mixed_data_types_in_sparse_array(self):
        """Test with different data types in sparse array."""
        import sparse

        # Create sparse array with integer data
        coords = ([0, 1], [0, 1])
        data = [1, 2]  # integers instead of floats
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        result = utils.stack_dataarray_from_dims(
            da, stack_dims=["latitude", "longitude"]
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims

    def test_very_large_max_size(self):
        """Test with extremely large max_size parameter."""
        import sparse

        coords = ([0, 1], [0, 1])
        data = [1.0, 2.0]
        shape = (2, 2)

        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        )

        # Test with very large max_size
        result = utils.stack_dataarray_from_dims(
            da,
            stack_dims=["latitude", "longitude"],
            max_size=10**10,  # Very large number
        )

        result = utils.reduce_dataarray(da, np.sum, ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        assert not isinstance(result.data, sparse.COO)

    def test_sparse_data_stacked_dimension(self):
        """Test sparse data creates stacked dimension."""
        import sparse

        coords = ([0, 1, 2], [0, 1, 0])
        data = [10.0, 20.0, 30.0]
        shape = (3, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [100.0, 110.0],
            },
        )

        # For sparse data, it should stack and reduce
        result = utils.reduce_dataarray(da, "mean", ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        # Result should be scalar after reducing all dims
        assert result.ndim == 0


class TestConvertDayYearofDayToTime:
    """Tests for convert_day_yearofday_to_time utility function."""

    def test_basic_conversion(self):
        """Test basic conversion of dayofyear and hour to time coordinate."""
        # Create a simple dataset with dayofyear and hour coords
        ds = xr.Dataset(
            {"temperature": (["dayofyear", "hour"], np.random.randn(3, 2))},
            coords={
                "dayofyear": [1, 2, 3],
                "hour": [0, 6],
            },
        )

        result = utils.convert_day_yearofday_to_time(ds, year=2023)

        # Check that valid_time coordinate was created
        assert "valid_time" in result.coords
        # Check that dayofyear and hour were removed
        assert "dayofyear" not in result.coords
        assert "hour" not in result.coords
        # Check that the time dimension was stacked
        assert "valid_time" in result.dims
        # Should have 3 * 2 = 6 timesteps
        assert len(result.valid_time) == 6

    def test_year_parameter(self):
        """Test that different years produce different dates."""
        ds = xr.Dataset(
            {"temperature": (["dayofyear", "hour"], np.random.randn(2, 2))},
            coords={
                "dayofyear": [1, 2],
                "hour": [0, 6],
            },
        )

        result_2023 = utils.convert_day_yearofday_to_time(ds, year=2023)
        result_2024 = utils.convert_day_yearofday_to_time(ds, year=2024)

        # First timestamp should be different years
        assert result_2023.valid_time[0].dt.year.item() == 2023
        assert result_2024.valid_time[0].dt.year.item() == 2024

    def test_correct_time_sequence(self):
        """Test that times are created in correct 6-hour intervals."""
        ds = xr.Dataset(
            {"temperature": (["dayofyear", "hour"], np.random.randn(2, 4))},
            coords={
                "dayofyear": [1, 2],
                "hour": [0, 6, 12, 18],
            },
        )

        result = utils.convert_day_yearofday_to_time(ds, year=2023)

        # Check the time coordinate is correctly created
        expected_times = pd.date_range(start="2023-01-01", periods=8, freq="6h")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result.valid_time.values),
            expected_times,
        )

    def test_preserves_data_values(self):
        """Test that data values are preserved after transformation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ds = xr.Dataset(
            {"temperature": (["dayofyear", "hour"], data)},
            coords={
                "dayofyear": [1, 2, 3],
                "hour": [0, 6],
            },
        )

        result = utils.convert_day_yearofday_to_time(ds, year=2023)

        # Data should be flattened in the stacked dimension
        assert result["temperature"].shape == (6,)
        # Values should match the flattened original data
        np.testing.assert_array_equal(result["temperature"].values, data.flatten())

    def test_with_additional_dimensions(self):
        """Test conversion with additional dimensions like latitude."""
        ds = xr.Dataset(
            {
                "temperature": (
                    ["dayofyear", "hour", "latitude"],
                    np.random.randn(2, 2, 3),
                )
            },
            coords={
                "dayofyear": [1, 2],
                "hour": [0, 6],
                "latitude": [10.0, 20.0, 30.0],
            },
        )

        result = utils.convert_day_yearofday_to_time(ds, year=2023)

        # Should preserve latitude dimension
        assert "latitude" in result.dims
        assert len(result.latitude) == 3
        # Should have stacked time dimension
        assert "valid_time" in result.dims
        assert len(result.valid_time) == 4

    def test_with_dataarray(self):
        """Test that conversion works with DataArray as well as Dataset."""
        # Create a DataArray with dayofyear and hour coords
        da = xr.DataArray(
            np.random.randn(3, 2),
            dims=["dayofyear", "hour"],
            coords={
                "dayofyear": [1, 2, 3],
                "hour": [0, 6],
            },
            name="temperature",
        )

        result = utils.convert_day_yearofday_to_time(da, year=2023)

        # Check that valid_time coordinate was created
        assert "valid_time" in result.coords
        # Check that dayofyear and hour were removed
        assert "dayofyear" not in result.coords
        assert "hour" not in result.coords
        # Check that the time dimension was stacked
        assert "valid_time" in result.dims
        # Should have 3 * 2 = 6 timesteps
        assert len(result.valid_time) == 6
        # Should still be a DataArray
        assert isinstance(result, xr.DataArray)

    def test_dataarray_preserves_data_values(self):
        """Test that DataArray data values are preserved."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        da = xr.DataArray(
            data,
            dims=["dayofyear", "hour"],
            coords={
                "dayofyear": [1, 2, 3],
                "hour": [0, 6],
            },
        )

        result = utils.convert_day_yearofday_to_time(da, year=2023)

        # Data should be flattened in the stacked dimension
        assert result.shape == (6,)
        # Values should match the flattened original data
        np.testing.assert_array_equal(result.values, data.flatten())

    def test_dataarray_with_additional_dimensions(self):
        """Test DataArray conversion with additional dimensions."""
        da = xr.DataArray(
            np.random.randn(2, 2, 3),
            dims=["dayofyear", "hour", "latitude"],
            coords={
                "dayofyear": [1, 2],
                "hour": [0, 6],
                "latitude": [10.0, 20.0, 30.0],
            },
        )

        result = utils.convert_day_yearofday_to_time(da, year=2023)

        # Should preserve latitude dimension
        assert "latitude" in result.dims
        assert len(result.latitude) == 3
        # Should have stacked time dimension
        assert "valid_time" in result.dims
        assert len(result.valid_time) == 4
        # Should still be a DataArray
        assert isinstance(result, xr.DataArray)


class TestInterpClimatologyToTarget:
    """Tests for interp_climatology_to_target utility function."""

    def test_with_dense_3d_target(self):
        """Test interpolation with dense 3D target data."""
        # Create climatology
        climatology = xr.DataArray(
            np.random.randn(5, 5),
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(0, 40, 5),
                "longitude": np.linspace(100, 140, 5),
            },
        )
        # Create dense target with 3+ dimensions
        target = xr.DataArray(
            np.random.randn(10, 3, 3),
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=10, freq="1D"),
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [110.0, 120.0, 130.0],
            },
        )

        result = utils.interp_climatology_to_target(target, climatology)

        # Should interpolate to target's lat/lon grid
        assert result.dims == ("latitude", "longitude")
        assert len(result.latitude) == 3
        assert len(result.longitude) == 3
        np.testing.assert_array_equal(result.latitude.values, target.latitude.values)
        np.testing.assert_array_equal(result.longitude.values, target.longitude.values)

    def test_ndim_less_than_3(self):
        """Test that ndim < 3 triggers the stacked interpolation path."""
        # Create climatology
        climatology = xr.DataArray(
            [[10.0, 20.0], [30.0, 40.0]],
            dims=["latitude", "longitude"],
            coords={
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0],
            },
        )
        # Create a 1D target (ndim < 3)
        target = xr.DataArray(
            np.random.randn(5),
            dims=["time"],
            coords={"time": range(5)},
        )

        # The function checks ndim < 3, which is True here
        # It will try to access target["stacked"]["latitude/longitude"]
        # This will fail, but that's expected for this edge case
        # The test verifies the branch logic exists
        with pytest.raises((KeyError, TypeError)):
            utils.interp_climatology_to_target(target, climatology)

    def test_with_different_grid(self):
        """Test that interpolation works when grids don't match."""
        # Create climatology on one grid
        climatology = xr.DataArray(
            np.arange(9).reshape(3, 3),
            dims=["latitude", "longitude"],
            coords={
                "latitude": [0.0, 10.0, 20.0],
                "longitude": [100.0, 110.0, 120.0],
            },
        )
        # Create target on a different grid
        target = xr.DataArray(
            np.random.randn(5, 2, 2),
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=5, freq="1D"),
                "latitude": [5.0, 15.0],  # Different from climatology
                "longitude": [105.0, 115.0],  # Different from climatology
            },
        )

        result = utils.interp_climatology_to_target(target, climatology)

        # Should interpolate to target's grid
        assert result.dims == ("latitude", "longitude")
        np.testing.assert_array_equal(result.latitude.values, [5.0, 15.0])
        np.testing.assert_array_equal(result.longitude.values, [105.0, 115.0])

    def test_nearest_neighbor_interpolation(self):
        """Test that nearest neighbor method is used."""
        # Create climatology with known values
        climatology = xr.DataArray(
            [[100.0, 200.0], [300.0, 400.0]],
            dims=["latitude", "longitude"],
            coords={
                "latitude": [0.0, 20.0],
                "longitude": [100.0, 120.0],
            },
        )
        # Create target at exact climatology points
        target = xr.DataArray(
            np.random.randn(5, 2, 2),
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=5, freq="1D"),
                "latitude": [0.0, 20.0],
                "longitude": [100.0, 120.0],
            },
        )

        result = utils.interp_climatology_to_target(target, climatology)

        # Should have exact values since points match
        np.testing.assert_array_almost_equal(result.values, climatology.values)


class TestReduceXarrayMethod:
    """Test the reduce_dataarray function."""

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample DataArray for testing."""
        data = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0, 120.0],
            },
        )

    @pytest.fixture
    def dataarray_with_nans(self):
        """Create DataArray with NaN values for testing skipna."""
        data = np.array(
            [
                [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]],
                [[7.0, 8.0, 9.0], [np.nan, 11.0, 12.0]],
            ]
        )
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0, 120.0],
            },
        )

    # Tests for numpy reduction functions (callables)
    def test_numpy_mean(self, sample_dataarray):
        """Test reduction using np.mean."""
        result = utils.reduce_dataarray(sample_dataarray, np.mean, ["time"])
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_numpy_sum(self, sample_dataarray):
        """Test reduction using np.sum."""
        result = utils.reduce_dataarray(sample_dataarray, np.sum, ["time"])
        expected = sample_dataarray.sum(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_numpy_nanmean(self, dataarray_with_nans):
        """Test reduction using np.nanmean."""
        result = utils.reduce_dataarray(dataarray_with_nans, np.nanmean, ["time"])
        # np.nanmean should ignore NaN values
        assert not np.isnan(result.values).all()

    def test_numpy_nansum(self, dataarray_with_nans):
        """Test reduction using np.nansum."""
        result = utils.reduce_dataarray(
            dataarray_with_nans, np.nansum, ["latitude", "longitude"]
        )
        # Result should have only time dimension
        assert result.dims == ("time",)
        assert not np.isnan(result.values).all()

    def test_custom_callable(self, sample_dataarray):
        """Test reduction using custom callable."""

        def custom_func(x, axis):
            return np.max(x, axis=axis)

        result = utils.reduce_dataarray(sample_dataarray, custom_func, ["time"])
        expected = sample_dataarray.max(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_callable_ignores_method_kwargs(self, sample_dataarray):
        """Test that method_kwargs are ignored for callables."""
        # This should work and ignore skipna parameter
        result = utils.reduce_dataarray(
            sample_dataarray, np.mean, ["time"], skipna=True
        )
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    # Tests for xarray built-in methods (strings)
    def test_xarray_mean_string(self, sample_dataarray):
        """Test reduction using 'mean' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "mean", ["time"])
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_sum_string(self, sample_dataarray):
        """Test reduction using 'sum' string method."""
        result = utils.reduce_dataarray(
            sample_dataarray, "sum", ["latitude", "longitude"]
        )
        expected = sample_dataarray.sum(dim=["latitude", "longitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_min_string(self, sample_dataarray):
        """Test reduction using 'min' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "min", ["time"])
        expected = sample_dataarray.min(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_max_string(self, sample_dataarray):
        """Test reduction using 'max' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "max", ["longitude"])
        expected = sample_dataarray.max(dim=["longitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_std_string(self, sample_dataarray):
        """Test reduction using 'std' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "std", ["time"])
        expected = sample_dataarray.std(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_median_string(self, sample_dataarray):
        """Test reduction using 'median' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "median", ["latitude"])
        expected = sample_dataarray.median(dim=["latitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_var_string(self, sample_dataarray):
        """Test reduction using 'var' string method."""
        result = utils.reduce_dataarray(sample_dataarray, "var", ["time"])
        expected = sample_dataarray.var(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    # Tests for dimension handling
    def test_single_dimension_reduction(self, sample_dataarray):
        """Test reducing a single dimension."""
        result = utils.reduce_dataarray(sample_dataarray, "mean", ["time"])
        assert "time" not in result.dims
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_multiple_dimension_reduction(self, sample_dataarray):
        """Test reducing multiple dimensions."""
        result = utils.reduce_dataarray(
            sample_dataarray, "sum", ["latitude", "longitude"]
        )
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims
        assert "time" in result.dims

    def test_all_dimensions_reduction(self, sample_dataarray):
        """Test reducing all dimensions."""
        result = utils.reduce_dataarray(
            sample_dataarray, "mean", ["time", "latitude", "longitude"]
        )
        # Result should be scalar (0 dimensions)
        assert len(result.dims) == 0
        assert isinstance(result.values.item(), (float, np.floating))

    def test_nonexistent_dimension_raises_error(self, sample_dataarray):
        """Test that non-existent dimension raises error."""
        with pytest.raises((ValueError, KeyError)):
            utils.reduce_dataarray(sample_dataarray, "mean", ["nonexistent_dim"])

    # Tests for error handling
    def test_invalid_method_type_raises_typeerror(self, sample_dataarray):
        """Test that invalid method type raises TypeError."""
        with pytest.raises(TypeError, match="method must be str or callable"):
            utils.reduce_dataarray(sample_dataarray, 123, ["time"])

    def test_nonexistent_xarray_method_raises_valueerror(self, sample_dataarray):
        """Test that non-existent xarray method raises ValueError."""
        with pytest.raises(ValueError, match="DataArray has no method"):
            utils.reduce_dataarray(sample_dataarray, "nonexistent_method", ["time"])

    def test_xarray_attribute_without_method(self, sample_dataarray):
        """Test that xarray attribute that isn't a method raises some kind of error."""
        with pytest.raises(TypeError, match="not callable"):
            utils.reduce_dataarray(sample_dataarray, "shape", ["time"])

    def test_empty_reduce_dims(self, sample_dataarray):
        """Test with empty reduce_dims list."""
        # This is edge case - xarray handles it gracefully
        result = utils.reduce_dataarray(sample_dataarray, "mean", [])
        # Should return the same array
        xr.testing.assert_equal(result, sample_dataarray)

    # Tests for method_kwargs
    def test_skipna_true(self, dataarray_with_nans):
        """Test skipna=True skips NaN values."""
        result = utils.reduce_dataarray(
            dataarray_with_nans, "mean", ["time"], skipna=True
        )
        # With skipna=True, result should not have NaNs
        assert not np.isnan(result.values).all()

    def test_skipna_false(self, dataarray_with_nans):
        """Test skipna=False propagates NaN values."""
        result = utils.reduce_dataarray(
            dataarray_with_nans, "mean", ["time"], skipna=False
        )
        # With skipna=False, NaNs should propagate
        assert np.isnan(result.values).any()

    def test_keep_attrs(self, sample_dataarray):
        """Test keep_attrs parameter."""
        sample_dataarray.attrs["test_attr"] = "test_value"
        result = utils.reduce_dataarray(
            sample_dataarray, "mean", ["time"], keep_attrs=True
        )
        assert result.attrs.get("test_attr") == "test_value"

    def test_multiple_method_kwargs(self, dataarray_with_nans):
        """Test multiple method_kwargs together."""
        dataarray_with_nans.attrs["units"] = "K"
        result = utils.reduce_dataarray(
            dataarray_with_nans,
            "sum",
            ["latitude"],
            skipna=True,
            keep_attrs=True,
        )
        assert result.attrs.get("units") == "K"
        assert not np.isnan(result.values).all()

    # Tests for sparse data handling
    def test_sparse_data_handling(self):
        """Test that sparse data is handled correctly."""
        import sparse

        # Create sparse array
        coords = ([0, 1, 2], [0, 1, 0])
        data = [1.0, 2.0, 3.0]
        shape = (3, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [100.0, 110.0],
            },
        )

        # Test that sparse data is handled
        result = utils.reduce_dataarray(da, "sum", ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        # Result should be densified scalar
        assert not isinstance(result.data, sparse.COO)

    def test_sparse_data_with_callable(self):
        """Test sparse data with callable method."""
        import sparse

        coords = ([0, 1, 0], [0, 1, 1], [0, 1, 0])
        data = [1.0, 2.0, 3.0]
        shape = (2, 2, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0],
            },
        )

        result = utils.reduce_dataarray(da, np.sum, ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        assert not isinstance(result.data, sparse.COO)

    def test_sparse_data_stacked_dimension(self):
        """Test sparse data creates stacked dimension."""
        import sparse

        coords = ([0, 1, 2], [0, 1, 0])
        data = [10.0, 20.0, 30.0]
        shape = (3, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [100.0, 110.0],
            },
        )

        # For sparse data, it should stack and reduce
        result = utils.reduce_dataarray(da, "mean", ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        # Result should be scalar after reducing all dims
        assert result.ndim == 0


class TestMaybeComputeAndMaybeCache:
    """Test the maybe_compute_and_maybe_cache function."""

    @pytest.fixture
    def sample_case(self):
        """Create a sample IndividualCase for testing."""
        from extremeweatherbench import cases, regions

        return cases.IndividualCase(
            case_id_number=42,
            title="Test Case",
            start_date=datetime.datetime(2023, 1, 1),
            end_date=datetime.datetime(2023, 1, 5),
            location=regions.CenteredRegion(
                latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
            ),
            event_type="test_event",
        )

    def test_no_compute_no_cache(self, sample_case):
        """Test with pre_compute=False and no cache - stays lazy."""
        # Create lazy datasets
        ds1 = xr.Dataset(
            {"temp": (["time", "lat"], [[1, 2], [3, 4]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds1.attrs["name"] = "forecast"

        ds2 = xr.Dataset(
            {"temp": (["time", "lat"], [[5, 6], [7, 8]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds2.attrs["name"] = "target"

        # Call with pre_compute=False and no cache_dir
        result = utils.maybe_compute_and_maybe_cache(
            ds1, ds2, pre_compute=False, cache_dir=None, case_metadata=sample_case
        )

        # Check results
        assert len(result) == 2
        assert all(isinstance(ds, xr.Dataset) for ds in result)
        # Should still be lazy
        assert hasattr(result[0].temp.data, "chunks")
        assert hasattr(result[1].temp.data, "chunks")

    def test_compute_without_caching(self, sample_case):
        """Test with pre_compute=True without cache directory."""
        # Create lazy datasets with names
        ds1 = xr.Dataset(
            {"temp": (["time", "lat"], [[1, 2], [3, 4]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds1.attrs["name"] = "forecast"

        ds2 = xr.Dataset(
            {"temp": (["time", "lat"], [[5, 6], [7, 8]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds2.attrs["name"] = "target"

        # Call with pre_compute=True but no cache_dir
        result = utils.maybe_compute_and_maybe_cache(
            ds1, ds2, pre_compute=True, cache_dir=None, case_metadata=sample_case
        )

        # Check results
        assert len(result) == 2
        assert all(isinstance(ds, xr.Dataset) for ds in result)
        # Should be computed (not lazy)
        assert not hasattr(result[0].temp.data, "chunks")
        assert not hasattr(result[1].temp.data, "chunks")

    def test_cache_without_precompute(self, sample_case, tmp_path):
        """Test caching (pre_compute=False) still computes for caching."""
        # Create lazy datasets with names
        ds1 = xr.Dataset(
            {"temp": (["time", "lat"], [[1, 2], [3, 4]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds1.attrs["name"] = "forecast"

        ds2 = xr.Dataset(
            {"temp": (["time", "lat"], [[5, 6], [7, 8]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds2.attrs["name"] = "target"

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Call with cache_dir but pre_compute=False (still computes for cache)
        result = utils.maybe_compute_and_maybe_cache(
            ds1, ds2, pre_compute=False, cache_dir=cache_dir, case_metadata=sample_case
        )

        # Check results
        assert len(result) == 2
        # Should be computed because cache_dir is set
        assert not hasattr(result[0].temp.data, "chunks")
        assert not hasattr(result[1].temp.data, "chunks")

        # Verify cache files were created
        expected_files = [
            cache_dir / f"case_id_number_{sample_case.case_id_number}_forecast.nc",
            cache_dir / f"case_id_number_{sample_case.case_id_number}_target.nc",
        ]
        for expected_file in expected_files:
            assert expected_file.exists()

        # Verify cached files can be loaded
        cached_ds1 = xr.open_dataset(expected_files[0])
        cached_ds2 = xr.open_dataset(expected_files[1])
        xr.testing.assert_equal(result[0], cached_ds1)
        xr.testing.assert_equal(result[1], cached_ds2)

    def test_compute_with_caching(self, sample_case, tmp_path):
        """Test with both pre_compute=True and cache_dir."""
        # Create datasets with names
        ds1 = xr.Dataset(
            {"temp": (["time", "lat"], [[1, 2], [3, 4]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds1.attrs["name"] = "forecast"

        ds2 = xr.Dataset(
            {"temp": (["time", "lat"], [[5, 6], [7, 8]])},
            coords={"time": [0, 1], "lat": [10, 20]},
        ).chunk()
        ds2.attrs["name"] = "target"

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Call with both pre_compute=True and cache_dir
        result = utils.maybe_compute_and_maybe_cache(
            ds1, ds2, pre_compute=True, cache_dir=cache_dir, case_metadata=sample_case
        )

        # Check results
        assert len(result) == 2
        # Should be computed
        assert not hasattr(result[0].temp.data, "chunks")
        assert not hasattr(result[1].temp.data, "chunks")

        # Verify cache files were created
        expected_files = [
            cache_dir / f"case_id_number_{sample_case.case_id_number}_forecast.nc",
            cache_dir / f"case_id_number_{sample_case.case_id_number}_target.nc",
        ]
        for expected_file in expected_files:
            assert expected_file.exists()

    def test_compute_multiple_datasets(self, sample_case):
        """Test with more than 2 datasets and pre_compute=True."""
        # Create 4 datasets
        datasets = []
        for i in range(4):
            ds = xr.Dataset({"var": (["x"], [i, i + 1])}, coords={"x": [0, 1]}).chunk()
            ds.attrs["name"] = f"dataset_{i}"
            datasets.append(ds)

        # Call function with pre_compute=True
        result = utils.maybe_compute_and_maybe_cache(
            *datasets, pre_compute=True, cache_dir=None, case_metadata=sample_case
        )

        # Check results
        assert len(result) == 4
        assert all(isinstance(ds, xr.Dataset) for ds in result)
        # All should be computed
        for ds in result:
            assert not hasattr(ds["var"].data, "chunks")

    def test_single_dataset_stays_lazy(self, sample_case):
        """Test single dataset with no compute or cache stays lazy."""
        ds = xr.Dataset(
            {"temp": (["time"], [1, 2, 3])}, coords={"time": [0, 1, 2]}
        ).chunk()
        ds.attrs["name"] = "single"

        result = utils.maybe_compute_and_maybe_cache(
            ds, pre_compute=False, cache_dir=None, case_metadata=sample_case
        )

        assert len(result) == 1
        assert isinstance(result[0], xr.Dataset)
        # Should still be lazy
        assert hasattr(result[0].temp.data, "chunks")

    def test_dataset_names_preserved(self, sample_case):
        """Test that dataset names from attrs are preserved."""
        ds1 = xr.Dataset({"var1": (["x"], [1, 2])}, coords={"x": [0, 1]})
        ds1.attrs["name"] = "my_forecast"

        ds2 = xr.Dataset({"var2": (["x"], [3, 4])}, coords={"x": [0, 1]})
        ds2.attrs["name"] = "my_target"

        result = utils.maybe_compute_and_maybe_cache(
            ds1, ds2, pre_compute=True, cache_dir=None, case_metadata=sample_case
        )

        # Names should be preserved in attrs if set
        assert result[0].attrs.get("name") == "my_forecast"
        assert result[1].attrs.get("name") == "my_target"

    def test_cache_dir_as_string(self, sample_case, tmp_path):
        """Test that cache_dir works as string path."""
        ds = xr.Dataset({"temp": (["x"], [1, 2])}, coords={"x": [0, 1]}).chunk()
        ds.attrs["name"] = "test"

        cache_dir = tmp_path / "cache_str"
        cache_dir.mkdir()

        # Pass as string instead of Path
        result = utils.maybe_compute_and_maybe_cache(
            ds, pre_compute=False, cache_dir=str(cache_dir), case_metadata=sample_case
        )

        assert len(result) == 1
        # Should be computed due to cache_dir
        assert not hasattr(result[0].temp.data, "chunks")
        # Verify cache file was created
        expected_file = (
            cache_dir / f"case_id_number_{sample_case.case_id_number}_test.nc"
        )
        assert expected_file.exists()

    def test_with_lazy_dask_arrays_precompute(self, sample_case):
        """Test lazy dask arrays with pre_compute=True."""
        import dask.array as da

        # Create dataset with dask arrays
        ds = xr.Dataset(
            {"temp": (["time", "lat"], da.ones((5, 10), chunks=(2, 5)))},
            coords={"time": range(5), "lat": range(10)},
        )
        ds.attrs["name"] = "lazy"

        result = utils.maybe_compute_and_maybe_cache(
            ds, pre_compute=True, cache_dir=None, case_metadata=sample_case
        )

        # Should be computed
        assert len(result) == 1
        assert not hasattr(result[0].temp.data, "chunks")
        assert isinstance(result[0].temp.data, np.ndarray)

    def test_dataset_without_name_attribute(self, sample_case, tmp_path):
        """Test caching works even if dataset has no name attr."""
        ds = xr.Dataset({"temp": (["x"], [1, 2])}, coords={"x": [0, 1]})
        # Don't set name attribute

        cache_dir = tmp_path / "cache_noname"
        cache_dir.mkdir()

        # This should work using default name
        result = utils.maybe_compute_and_maybe_cache(
            ds, pre_compute=False, cache_dir=cache_dir, case_metadata=sample_case
        )

        assert len(result) == 1


class TestNoCircularImports:
    """Test that there are no circular imports."""

    def test_import_utils_then_evaluate(self):
        """Test importing utils then evaluate doesn't cause circular imports."""
        import sys

        # Remove modules if already imported
        modules_to_remove = [
            k for k in sys.modules.keys() if k.startswith("extremeweatherbench")
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Import utils first
        from extremeweatherbench import utils as utils_module

        # Then import evaluate
        from extremeweatherbench import evaluate as evaluate_module

        # Verify both imported successfully
        assert utils_module is not None
        assert evaluate_module is not None
        assert hasattr(utils_module, "maybe_compute_and_maybe_cache")

    def test_import_evaluate_then_utils(self):
        """Test importing evaluate then utils works fine."""
        import sys

        # Remove modules if already imported
        modules_to_remove = [
            k for k in sys.modules.keys() if k.startswith("extremeweatherbench")
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Import evaluate first
        from extremeweatherbench import evaluate as evaluate_module

        # Then import utils
        from extremeweatherbench import utils as utils_module

        # Verify both imported successfully
        assert evaluate_module is not None
        assert utils_module is not None

    def test_evaluate_uses_utils_function(self):
        """Test that evaluate module can access the function."""
        from extremeweatherbench import utils

        # Verify evaluate can call the function from utils
        assert hasattr(utils, "maybe_compute_and_maybe_cache")
        assert callable(utils.maybe_compute_and_maybe_cache)

    def test_function_accessible_from_both_modules(self):
        """Test that function is accessible correctly from both modules."""
        from extremeweatherbench import cases, regions, utils

        # Create a sample case
        sample_case = cases.IndividualCase(
            case_id_number=1,
            title="Test",
            start_date=datetime.datetime(2023, 1, 1),
            end_date=datetime.datetime(2023, 1, 2),
            location=regions.CenteredRegion(
                latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
            ),
            event_type="test",
        )

        # Create a simple dataset
        ds = xr.Dataset({"temp": (["x"], [1, 2, 3])}, coords={"x": [0, 1, 2]})
        ds.attrs["name"] = "test_ds"

        # Call the function through utils module with pre_compute
        result = utils.maybe_compute_and_maybe_cache(
            ds, pre_compute=True, cache_dir=None, case_metadata=sample_case
        )

        assert len(result) == 1
        assert isinstance(result[0], xr.Dataset)


class TestMaybeGetOperator:
    """Test the maybe_get_operator function."""

    def test_greater_than_operator(self):
        """Test '>' operator string returns operator.gt."""
        op = utils.maybe_get_operator(">")
        assert op is operator.gt
        assert op(5, 3) is True
        assert op(3, 5) is False

    def test_greater_than_or_equal_operator(self):
        """Test '>=' operator string returns operator.ge."""
        op = utils.maybe_get_operator(">=")
        assert op is operator.ge
        assert op(5, 3) is True
        assert op(5, 5) is True
        assert op(3, 5) is False

    def test_less_than_operator(self):
        """Test '<' operator string returns operator.lt."""
        op = utils.maybe_get_operator("<")
        assert op is operator.lt
        assert op(3, 5) is True
        assert op(5, 3) is False

    def test_less_than_or_equal_operator(self):
        """Test '<=' operator string returns operator.le."""
        op = utils.maybe_get_operator("<=")
        assert op is operator.le
        assert op(3, 5) is True
        assert op(5, 5) is True
        assert op(5, 3) is False

    def test_equal_operator(self):
        """Test '==' operator string returns operator.eq."""
        op = utils.maybe_get_operator("==")
        assert op is operator.eq
        assert op(5, 5) is True
        assert op(5, 3) is False

    def test_not_equal_operator(self):
        """Test '!=' operator string returns operator.ne."""
        op = utils.maybe_get_operator("!=")
        assert op is operator.ne
        assert op(5, 3) is True
        assert op(5, 5) is False

    def test_callable_passthrough(self):
        """Test passing a callable returns it unchanged."""

        def custom_op(a, b):
            return a + b > 10

        result = utils.maybe_get_operator(custom_op)
        assert result is custom_op
        assert result(5, 6) is True
        assert result(3, 4) is False

    def test_lambda_passthrough(self):
        """Test passing a lambda returns it unchanged."""
        # Create lambda inline to avoid linter warning
        result = utils.maybe_get_operator(lambda a, b: a * b > 20)
        # Test the returned callable works correctly
        assert callable(result)
        assert result(5, 5) is True
        assert result(2, 3) is False

    def test_builtin_function_passthrough(self):
        """Test passing a builtin function works."""
        result = utils.maybe_get_operator(max)
        assert result is max
        assert result(5, 3) == 5

    def test_operator_module_function_passthrough(self):
        """Test passing an operator module function works."""
        result = utils.maybe_get_operator(operator.add)
        assert result is operator.add
        assert result(5, 3) == 8

    def test_invalid_string_raises_keyerror(self):
        """Test invalid operator string raises KeyError."""
        with pytest.raises(KeyError):
            utils.maybe_get_operator("invalid")

    def test_similar_but_invalid_strings(self):
        """Test strings that look like operators raise KeyError."""
        invalid_ops = ["=>", "=<", "===", ">==", "<<", ">>"]
        for invalid_op in invalid_ops:
            with pytest.raises(KeyError):
                utils.maybe_get_operator(invalid_op)

    def test_empty_string_raises_keyerror(self):
        """Test empty string raises KeyError."""
        with pytest.raises(KeyError):
            utils.maybe_get_operator("")

    def test_none_returns_none(self):
        """Test None input returns None (treated as callable)."""
        # None is technically callable-like in isinstance check
        # isinstance(None, str) == False
        result = utils.maybe_get_operator(None)
        # Should return None since it's not a string
        assert result is None

    def test_integer_returns_integer(self):
        """Test integer input returns integer (not a string)."""
        # Edge case: non-string, non-callable input
        result = utils.maybe_get_operator(123)
        assert result == 123

    def test_operators_with_arrays(self):
        """Test operator functions work with numpy arrays."""
        op_gt = utils.maybe_get_operator(">")
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([2, 2, 2, 2])
        result = op_gt(arr1, arr2)
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(result, expected)

    def test_operators_with_xarray(self):
        """Test operator functions work with xarray objects."""
        op_lt = utils.maybe_get_operator("<")
        da1 = xr.DataArray([1, 2, 3])
        da2 = xr.DataArray([2, 2, 2])
        result = op_lt(da1, da2)
        expected = xr.DataArray([True, False, False])
        xr.testing.assert_equal(result, expected)

    def test_all_operators_in_dict(self):
        """Test all operators in module dict are accessible."""
        # Verify the operators dict at module level
        expected_ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }

        for op_str, op_func in expected_ops.items():
            result = utils.maybe_get_operator(op_str)
            assert result is op_func

    def test_operator_return_types(self):
        """Test operators return correct boolean types."""
        ops = [">", ">=", "<", "<=", "==", "!="]
        for op_str in ops:
            op = utils.maybe_get_operator(op_str)
            result = op(5, 3)
            assert isinstance(result, bool)

    def test_case_sensitivity(self):
        """Test operator strings are case-sensitive."""
        # Uppercase versions should raise KeyError
        with pytest.raises(KeyError):
            utils.maybe_get_operator("GT")
        with pytest.raises(KeyError):
            utils.maybe_get_operator("LT")

    def test_whitespace_in_string(self):
        """Test whitespace in operator string raises KeyError."""
        with pytest.raises(KeyError):
            utils.maybe_get_operator(" > ")
        with pytest.raises(KeyError):
            utils.maybe_get_operator("> ")


class TestMaybeDensifyDataArray:
    """Test the maybe_densify_dataarray function."""

    def test_densify_sparse_data(self):
        """Test densifying sparse data with default max_size."""
        import sparse

        # Create sparse array
        coords = ([0, 1, 2], [0, 1, 0])
        data = [1.0, 2.0, 3.0]
        shape = (3, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        # Verify input is sparse
        assert isinstance(da.data, sparse.COO)

        # Densify
        result = utils.maybe_densify_dataarray(da)

        # Should be densified
        assert not isinstance(result.data, sparse.COO)
        assert isinstance(result.data, np.ndarray)

        # Values should be preserved
        assert result.values[0, 0] == 1.0
        assert result.values[1, 1] == 2.0
        assert result.values[2, 0] == 3.0

    def test_dense_data_unchanged(self):
        """Test that dense data remains unchanged."""
        # Create dense array
        da = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        # Verify input is dense
        assert isinstance(da.data, np.ndarray)

        # Apply function
        result = utils.maybe_densify_dataarray(da)

        # Should remain dense
        assert isinstance(result.data, np.ndarray)

        # Values should be unchanged
        xr.testing.assert_equal(result, da)

    def test_custom_max_size_allows_densification(self):
        """Test custom max_size for small arrays allows densification."""
        import sparse

        # Create small sparse array
        coords = ([0, 1], [0, 1])
        data = [10.0, 20.0]
        shape = (2, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [0, 1]},
        )

        # With large max_size, should densify
        result = utils.maybe_densify_dataarray(da, max_size=1000)

        assert not isinstance(result.data, sparse.COO)
        assert isinstance(result.data, np.ndarray)

    def test_small_max_size_raises_error(self):
        """Test very small max_size raises error for large sparse arrays."""
        import sparse

        # Create large sparse array with shape > max_size
        # Shape is 100x100 = 10,000 elements but only 3 non-zero
        coords = ([0, 50, 99], [0, 50, 99])
        data = [1.0, 2.0, 3.0]
        shape = (100, 100)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["x", "y"],
            coords={"x": range(100), "y": range(100)},
        )

        # max_size smaller than array size and low density raises error
        with pytest.raises(ValueError, match="large sparse array"):
            utils.maybe_densify_dataarray(da, max_size=100)

    def test_empty_sparse_array(self):
        """Test empty sparse array can be densified."""
        import sparse

        # Create empty sparse array
        coords = ([], [])
        data = []
        shape = (3, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["latitude", "longitude"],
            coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
        )

        result = utils.maybe_densify_dataarray(da)

        # Should be densified
        assert not isinstance(result.data, sparse.COO)
        assert isinstance(result.data, np.ndarray)

        # All values should be zero (default fill value)
        assert (result.values == 0).all()

    def test_multidimensional_sparse_array(self):
        """Test densifying multidimensional sparse arrays."""
        import sparse

        # Create 3D sparse array
        coords = ([0, 1, 2], [0, 1, 0], [0, 0, 1])
        data = [1.0, 2.0, 3.0]
        shape = (3, 2, 2)
        sparse_array = sparse.COO(coords, data, shape=shape)

        da = xr.DataArray(
            sparse_array,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "latitude": [10.0, 20.0],
                "longitude": [100.0, 110.0],
            },
        )

        result = utils.maybe_densify_dataarray(da)

        # Should be densified
        assert not isinstance(result.data, sparse.COO)
        assert isinstance(result.data, np.ndarray)

        # Check dimensions preserved
        assert result.shape == (3, 2, 2)
        assert list(result.dims) == ["time", "latitude", "longitude"]
