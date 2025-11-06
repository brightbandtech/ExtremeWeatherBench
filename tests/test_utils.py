"""Tests for the utils module."""

import datetime

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
    """Test the stack_sparse_data_from_dims function."""

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
        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(da, stack_dims=["latitude"])

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
        result = utils.stack_sparse_data_from_dims(
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
        result_small = utils.stack_sparse_data_from_dims(
            da, stack_dims=["latitude", "longitude"], max_size=1
        )
        result_large = utils.stack_sparse_data_from_dims(
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
            utils.stack_sparse_data_from_dims(da, stack_dims=["nonexistent"])

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

        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(
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

        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(
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

        # This should raise an error or handle gracefully since it expects
        # sparse data
        with pytest.raises(AttributeError):
            utils.stack_sparse_data_from_dims(da, stack_dims=["latitude", "longitude"])

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
            utils.stack_sparse_data_from_dims(da, stack_dims=[])

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

        result = utils.stack_sparse_data_from_dims(
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

        result = utils.stack_sparse_data_from_dims(
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
        result = utils.stack_sparse_data_from_dims(
            da,
            stack_dims=["latitude", "longitude"],
            max_size=10**10,  # Very large number
        )

        assert isinstance(result, xr.DataArray)
        assert "stacked" in result.dims


class TestReduceXarrayMethod:
    """Test the reduce_xarray_method function."""

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
        result = utils.reduce_xarray_method(sample_dataarray, np.mean, ["time"])
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_numpy_sum(self, sample_dataarray):
        """Test reduction using np.sum."""
        result = utils.reduce_xarray_method(sample_dataarray, np.sum, ["time"])
        expected = sample_dataarray.sum(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_numpy_nanmean(self, dataarray_with_nans):
        """Test reduction using np.nanmean."""
        result = utils.reduce_xarray_method(dataarray_with_nans, np.nanmean, ["time"])
        # np.nanmean should ignore NaN values
        assert not np.isnan(result.values).all()

    def test_numpy_nansum(self, dataarray_with_nans):
        """Test reduction using np.nansum."""
        result = utils.reduce_xarray_method(
            dataarray_with_nans, np.nansum, ["latitude", "longitude"]
        )
        # Result should have only time dimension
        assert result.dims == ("time",)
        assert not np.isnan(result.values).all()

    def test_custom_callable(self, sample_dataarray):
        """Test reduction using custom callable."""

        def custom_func(x, axis):
            return np.max(x, axis=axis)

        result = utils.reduce_xarray_method(sample_dataarray, custom_func, ["time"])
        expected = sample_dataarray.max(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_callable_ignores_method_kwargs(self, sample_dataarray):
        """Test that method_kwargs are ignored for callables."""
        # This should work and ignore skipna parameter
        result = utils.reduce_xarray_method(
            sample_dataarray, np.mean, ["time"], skipna=True
        )
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    # Tests for xarray built-in methods (strings)
    def test_xarray_mean_string(self, sample_dataarray):
        """Test reduction using 'mean' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "mean", ["time"])
        expected = sample_dataarray.mean(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_sum_string(self, sample_dataarray):
        """Test reduction using 'sum' string method."""
        result = utils.reduce_xarray_method(
            sample_dataarray, "sum", ["latitude", "longitude"]
        )
        expected = sample_dataarray.sum(dim=["latitude", "longitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_min_string(self, sample_dataarray):
        """Test reduction using 'min' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "min", ["time"])
        expected = sample_dataarray.min(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_max_string(self, sample_dataarray):
        """Test reduction using 'max' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "max", ["longitude"])
        expected = sample_dataarray.max(dim=["longitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_std_string(self, sample_dataarray):
        """Test reduction using 'std' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "std", ["time"])
        expected = sample_dataarray.std(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_median_string(self, sample_dataarray):
        """Test reduction using 'median' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "median", ["latitude"])
        expected = sample_dataarray.median(dim=["latitude"])
        xr.testing.assert_allclose(result, expected)

    def test_xarray_var_string(self, sample_dataarray):
        """Test reduction using 'var' string method."""
        result = utils.reduce_xarray_method(sample_dataarray, "var", ["time"])
        expected = sample_dataarray.var(dim=["time"])
        xr.testing.assert_allclose(result, expected)

    # Tests for dimension handling
    def test_single_dimension_reduction(self, sample_dataarray):
        """Test reducing a single dimension."""
        result = utils.reduce_xarray_method(sample_dataarray, "mean", ["time"])
        assert "time" not in result.dims
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_multiple_dimension_reduction(self, sample_dataarray):
        """Test reducing multiple dimensions."""
        result = utils.reduce_xarray_method(
            sample_dataarray, "sum", ["latitude", "longitude"]
        )
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims
        assert "time" in result.dims

    def test_all_dimensions_reduction(self, sample_dataarray):
        """Test reducing all dimensions."""
        result = utils.reduce_xarray_method(
            sample_dataarray, "mean", ["time", "latitude", "longitude"]
        )
        # Result should be scalar (0 dimensions)
        assert len(result.dims) == 0
        assert isinstance(result.values.item(), (float, np.floating))

    def test_nonexistent_dimension_raises_error(self, sample_dataarray):
        """Test that non-existent dimension raises error."""
        with pytest.raises((ValueError, KeyError)):
            utils.reduce_xarray_method(sample_dataarray, "mean", ["nonexistent_dim"])

    # Tests for error handling
    def test_invalid_method_type_raises_typeerror(self, sample_dataarray):
        """Test that invalid method type raises TypeError."""
        with pytest.raises(TypeError, match="method must be str or callable"):
            utils.reduce_xarray_method(sample_dataarray, 123, ["time"])

    def test_nonexistent_xarray_method_raises_valueerror(self, sample_dataarray):
        """Test that non-existent xarray method raises ValueError."""
        with pytest.raises(ValueError, match="DataArray has no method"):
            utils.reduce_xarray_method(sample_dataarray, "nonexistent_method", ["time"])

    def test_empty_reduce_dims(self, sample_dataarray):
        """Test with empty reduce_dims list."""
        # This is edge case - xarray handles it gracefully
        result = utils.reduce_xarray_method(sample_dataarray, "mean", [])
        # Should return the same array
        xr.testing.assert_equal(result, sample_dataarray)

    # Tests for method_kwargs
    def test_skipna_true(self, dataarray_with_nans):
        """Test skipna=True skips NaN values."""
        result = utils.reduce_xarray_method(
            dataarray_with_nans, "mean", ["time"], skipna=True
        )
        # With skipna=True, result should not have NaNs
        assert not np.isnan(result.values).all()

    def test_skipna_false(self, dataarray_with_nans):
        """Test skipna=False propagates NaN values."""
        result = utils.reduce_xarray_method(
            dataarray_with_nans, "mean", ["time"], skipna=False
        )
        # With skipna=False, NaNs should propagate
        assert np.isnan(result.values).any()

    def test_keep_attrs(self, sample_dataarray):
        """Test keep_attrs parameter."""
        sample_dataarray.attrs["test_attr"] = "test_value"
        result = utils.reduce_xarray_method(
            sample_dataarray, "mean", ["time"], keep_attrs=True
        )
        assert result.attrs.get("test_attr") == "test_value"

    def test_multiple_method_kwargs(self, dataarray_with_nans):
        """Test multiple method_kwargs together."""
        dataarray_with_nans.attrs["units"] = "K"
        result = utils.reduce_xarray_method(
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
        result = utils.reduce_xarray_method(da, "sum", ["latitude", "longitude"])
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

        result = utils.reduce_xarray_method(da, np.sum, ["latitude", "longitude"])
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
        result = utils.reduce_xarray_method(da, "mean", ["latitude", "longitude"])
        assert isinstance(result, xr.DataArray)
        # Result should be scalar after reducing all dims
        assert result.ndim == 0
