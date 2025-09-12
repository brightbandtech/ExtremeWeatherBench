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


def test_default_preprocess():
    """Test default preprocess function."""
    # Import the function from inputs module since it was moved there
    from extremeweatherbench.inputs import _default_preprocess

    # Test with xarray Dataset
    ds = xr.Dataset({"temp": (["x"], [1, 2, 3])})
    result = _default_preprocess(ds)
    assert result is ds  # Should return the same object unchanged

    # Test with pandas DataFrame
    df = pd.DataFrame({"a": [1, 2, 3]})
    result_df = _default_preprocess(df)
    assert result_df is df


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
