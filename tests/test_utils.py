import pytest
from extremeweatherbench import utils
import pandas as pd
import numpy as np
import xarray as xr


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (180, 180), (360, 0), (-179, 181), (-360, 0), (540, 180), (359, 359)],
)
def test_convert_longitude_to_360_param(input, expected):
    assert utils.convert_longitude_to_360(input) == expected


def test_center_forecast_on_time(
    sample_subset_forecast_dataarray,
    sample_subset_gridded_obs_dataarray,
):
    """Test that the center_forecast_on_time function properly centers the forecast on the given
    timestamp, including at edges of the forecast which produce truncated outputs."""
    hours = 48
    test_aligned_da = utils.temporal_align_dataarrays(
        sample_subset_forecast_dataarray,
        sample_subset_gridded_obs_dataarray,
        pd.Timestamp(sample_subset_forecast_dataarray.init_time[0].values),
    )[0]

    test_aligned_da_timestamps = test_aligned_da.time.values
    test_timestamps = [
        test_aligned_da_timestamps[0],
        test_aligned_da_timestamps[2],
        test_aligned_da_timestamps[len(test_aligned_da_timestamps) // 2],
        test_aligned_da_timestamps[-3],
        test_aligned_da_timestamps[-1],
    ]
    aligned_forecast_timedelta = pd.to_timedelta(
        test_aligned_da.time[-1].values - test_aligned_da.time[0].values
    )
    assert aligned_forecast_timedelta > pd.to_timedelta(hours * 2, unit="h")
    for timestamp in test_timestamps:
        centered_forecast_da = utils.center_forecast_on_time(
            test_aligned_da, timestamp, hours=hours
        )
        centered_forecast_timedelta = pd.to_timedelta(
            centered_forecast_da.time[-1].values - centered_forecast_da.time[0].values
        )
        index = np.where(test_aligned_da.time.values == timestamp)[0][0]
        if index == 0 and centered_forecast_timedelta == pd.to_timedelta(
            hours, unit="h"
        ):
            # test if centering on the 0th index produced a forecast with the correct time range
            assert centered_forecast_da.time[0] == timestamp
            assert centered_forecast_da.time[-1] == timestamp + pd.to_timedelta(
                hours, unit="h"
            )
        elif index == len(
            test_aligned_da.time.values
        ) - 1 and centered_forecast_timedelta == pd.to_timedelta(hours, unit="h"):
            # test if centering on the last index produced a forecast with the correct time range
            assert centered_forecast_da.time[0] == timestamp - pd.to_timedelta(
                hours, unit="h"
            )
            assert centered_forecast_da.time[-1] == timestamp
        elif index == len(
            test_aligned_da.time.values
        ) // 2 and centered_forecast_timedelta == pd.to_timedelta(hours * 2, unit="h"):
            # test if centering on the middle index produced a forecast with the correct time range,
            # with the added condition that the timedelta is twice the size of the hours
            # in case future modifications to fixtures changes the mock forecast time range
            assert centered_forecast_da.time[0] == timestamp - pd.to_timedelta(
                hours, unit="h"
            )
            assert centered_forecast_da.time[-1] == timestamp + pd.to_timedelta(
                hours, unit="h"
            )
            # TODO: Add test for the case where the centered forecast is close to the edges,
            # e.g. at index 2 or -3


def test_temporal_align_dataarrays(
    sample_forecast_dataarray, sample_gridded_obs_dataarray
):
    """Test that the conversion from init time to valid time (named as time) produces an aligned
    dataarray to ensure metrics are applied properly."""
    init_time_datetime = pd.Timestamp(
        sample_forecast_dataarray.init_time[0].values
    ).to_pydatetime()
    aligned_forecast, aligned_obs = utils.temporal_align_dataarrays(
        sample_forecast_dataarray, sample_gridded_obs_dataarray, init_time_datetime
    )

    # Check aligned datasets have same time coordinates
    assert (aligned_forecast.time == aligned_obs.time).all()

    # Check forecast was properly subset by init time
    assert aligned_forecast.init_time.size == 1
    assert pd.Timestamp(aligned_forecast.init_time.values) == pd.Timestamp(
        sample_forecast_dataarray.init_time[0].values
    )


def test_process_dataarray_for_output(sample_results_dataarray_list):
    """Test to ensure the inputs to process_dataarray_for_output always result in a DataArray
    regardless of length."""
    test_output = utils.process_dataarray_for_output(sample_results_dataarray_list)
    assert isinstance(test_output, xr.DataArray)

    test_len_0_output = utils.process_dataarray_for_output([])
    assert isinstance(test_len_0_output, xr.DataArray)


def test_obs_finer_temporal_resolution(
    sample_forecast_dataarray, sample_gridded_obs_dataarray
):
    """Test when observation has finer temporal resolution."""
    aligned_obs = utils.align_observations_temporal_resolution(
        sample_forecast_dataarray, sample_gridded_obs_dataarray
    )
    # Check that observation was modified
    assert len(np.unique(np.diff(aligned_obs.time))) == 1
    assert np.unique(np.diff(aligned_obs.time)).astype("timedelta64[h]").astype(
        int
    ) == np.unique(np.diff(sample_forecast_dataarray.lead_time))


def test_obs_finer_temporal_resolution_data(
    sample_forecast_dataarray, sample_gridded_obs_dataarray
):
    """Test when observation has finer temporal resolution than forecast,
    that the outputs are the correct values at the hour."""
    aligned_obs = utils.align_observations_temporal_resolution(
        sample_forecast_dataarray, sample_gridded_obs_dataarray
    )

    test_truncated_obs, test_aligned_obs = xr.align(
        sample_gridded_obs_dataarray, aligned_obs, join="inner"
    )
    # Check that observation was modified
    assert (test_aligned_obs == test_truncated_obs).all()


def test_obs_coarser_temporal_resolution(
    sample_forecast_dataarray, sample_gridded_obs_dataarray
):
    """Test when observation has coarser temporal resolution than forecast,
    that observations are unmodified."""
    sample_forecast_dataarray = sample_forecast_dataarray.isel(init_time=0)
    aligned_obs = utils.align_observations_temporal_resolution(
        sample_forecast_dataarray,
        sample_gridded_obs_dataarray.resample(time="12h").first(),
    )
    # Check that observation was modified
    assert (aligned_obs == sample_gridded_obs_dataarray).all()


def test_clip_dataset_to_bounding_box_degrees():
    # Create a sample dataset
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(0, 359, 360)
    data = np.random.rand(181, 360)
    ds = xr.Dataset(
        {"data": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )

    # Test case 1: Single value for box_degrees, latitude ascending
    location_center = utils.Location(latitude=40, longitude=100)
    box_degrees = 10
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )
    assert np.all(clipped_ds.latitude >= 35)
    assert np.all(clipped_ds.latitude <= 45)
    assert np.all(clipped_ds.longitude >= 95)
    assert np.all(clipped_ds.longitude <= 105)

    # Test case 2: Tuple for box_degrees, latitude ascending
    location_center = utils.Location(latitude=40, longitude=100)
    box_degrees = (5, 10)
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )
    assert np.all(clipped_ds.latitude >= 37.5)
    assert np.all(clipped_ds.latitude <= 42.5)
    assert np.all(clipped_ds.longitude >= 95)
    assert np.all(clipped_ds.longitude <= 105)

    # Test case 3: Negative longitude, latitude ascending
    location_center = utils.Location(latitude=40, longitude=-100)
    box_degrees = 10
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )
    assert np.all(clipped_ds.latitude >= 35)
    assert np.all(clipped_ds.latitude <= 45)
    assert np.all(clipped_ds.longitude >= 255)  # -100 + 360 - 5 = 255
    assert np.all(clipped_ds.longitude <= 265)  # -100 + 360 + 5 = 265

    # Test case 4: Latitude descending
    ds_desc = ds.reindex(latitude=ds.latitude[::-1])
    location_center = utils.Location(latitude=40, longitude=100)
    box_degrees = 10
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds_desc, location_center, box_degrees
    )
    assert np.all(clipped_ds.latitude >= 35)
    assert np.all(clipped_ds.latitude <= 45)
    assert np.all(clipped_ds.longitude >= 95)
    assert np.all(clipped_ds.longitude <= 105)

    # Test wrapping around prime meridian
    location_center = utils.Location(latitude=0, longitude=0)
    box_degrees = 10

    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )

    assert np.any(clipped_ds.longitude < 10)
    assert np.any(clipped_ds.longitude > 350)

    # Test case 5: Edge cases, ensuring no errors when clipping at boundaries
    location_center = utils.Location(latitude=90, longitude=100)
    box_degrees = 10
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )

    location_center = utils.Location(latitude=-90, longitude=100)
    box_degrees = 10
    clipped_ds = utils.clip_dataset_to_bounding_box_degrees(
        ds, location_center, box_degrees
    )


def test_swap_coords_basic():
    # Create one-dimensional DataArrays
    coord_from = xr.DataArray(np.array([1, 2, 3, 4]))
    coord_to = xr.DataArray(np.array([10, 20, 30, 40]))
    coord_match = xr.DataArray(np.array([1, 0, 3, 0]))
    # Expected: where coord_from equals coord_match use coord_to value, else use coord_match value.
    # Index 0: 1 == 1 -> 10
    # Index 1: 2 != 0 -> 0
    # Index 2: 3 == 3 -> 30
    # Index 3: 4 != 0 -> 0
    expected = np.array([10, 0, 30, 0])
    result = utils.swap_coords(coord_from, coord_to, coord_match)
    assert np.array_equal(result.values, expected)


def test_swap_coords_multidimensional():
    # Create two-dimensional DataArrays
    coord_from = xr.DataArray(np.array([[1, 2], [3, 4]]))
    coord_to = xr.DataArray(np.array([[100, 200], [300, 400]]))
    coord_match = xr.DataArray(np.array([[1, 0], [4, 4]]))
    # Expected:
    # (0,0): 1 == 1 -> 100
    # (0,1): 2 != 0 -> 0
    # (1,0): 3 != 4 -> 4
    # (1,1): 4 == 4 -> 400
    expected = np.array([[100, 0], [4, 400]])
    result = utils.swap_coords(coord_from, coord_to, coord_match)
    assert np.array_equal(result.values, expected)


def test_derive_indices_valid_range():
    # Create a simple dataset with two init_time values and five lead_time values
    init_times = pd.to_datetime(["2023-01-01 00:00", "2023-01-02 00:00"])
    lead_times = np.array([0, 6, 12, 18, 24])
    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
        }
    )

    # Define a time window that should capture specific valid times.
    # For the first init_time row, valid forecast times are:
    # 2023-01-01 00:00, 06:00, 12:00, 18:00, 2023-01-02 00:00
    # For the second init_time row, valid forecast times are:
    # 2023-01-02 00:00, 06:00, 12:00, 18:00, 2023-01-03 00:00
    # Set start and end so that only a subset falls in the window:
    # window: (2023-01-01 03:00, 2023-01-02 15:00)
    start_date = pd.Timestamp("2023-01-01 03:00")
    end_date = pd.Timestamp("2023-01-02 15:00")

    # Call function under test
    indices = utils.derive_indices_from_init_time_and_lead_time(
        ds, start_date, end_date
    )

    # indices is a tuple (row_indices, col_indices) from np.where:
    # Expected valid times:
    # For first row (init_time=2023-01-01 00:00): 06:00, 12:00, 18:00, 2023-01-02 00:00 => 4 values
    # For second row (init_time=2023-01-02 00:00): 2023-01-02 00:00, 06:00, 12:00 => 3 values
    expected_count = 4 + 3
    assert indices[0].size == expected_count

    # Check that the computed times fall within the expected range.
    # Build the valid times grid for testing:
    forecast_times = np.empty(
        (len(init_times), len(lead_times)), dtype="datetime64[ns]"
    )
    for i, init in enumerate(init_times):
        forecast_times[i, :] = init + pd.to_timedelta(lead_times, unit="h")
    valid_times = forecast_times[indices]
    assert all(valid_times > start_date)
    assert all(valid_times < end_date)


def test_derive_indices_no_matches():
    # Create a dataset with one init_time and a few lead_time values
    init_times = pd.to_datetime(["2023-01-01 00:00"])
    lead_times = np.array([0, 6, 12])
    ds = xr.Dataset(
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
        }
    )

    # Define a time window that excludes all forecast times.
    # Forecast times for this dataset are: 2023-01-01 00:00, 06:00, and 12:00.
    # Use a window that does not cover any of these.
    start_date = pd.Timestamp("2023-01-01 13:00")
    end_date = pd.Timestamp("2023-01-01 14:00")

    # Call function under test
    indices = utils.derive_indices_from_init_time_and_lead_time(
        ds, start_date, end_date
    )

    # Expect empty indices
    assert indices[0].size == 0
    assert indices[1].size == 0


def test_reshape_dataset_to_include_latlon():
    # Create a sample dataset with dims: init_time, lead_time, and a dummy dimension to be reshaped.

    init_times = pd.date_range("2023-01-01", periods=2)
    lead_times = [0, 6]
    # Dummy index with two unique values; each dummy value has associated latitude and longitude.
    dummy_vals = ["a", "b"]
    lat_vals = [10.0, 20.0]
    lon_vals = [100.0, 110.0]
    # Create a data variable with shape (init_time, lead_time, dummy)
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ]
    )
    ds = xr.Dataset(
        {"var": (("init_time", "lead_time", "dummy"), data)},
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "dummy": dummy_vals,
            "latitude": ("dummy", lat_vals),
            "longitude": ("dummy", lon_vals),
        },
    )

    reshaped_ds = utils.reshape_dataset_to_include_latlon(ds, "dummy")

    # Check that the resulting dataset has the expected dimensions.
    expected_dims = {"init_time", "lead_time", "latitude", "longitude"}
    assert set(reshaped_ds.dims) == expected_dims

    # Check that the latitude and longitude coordinates are as expected.
    np.testing.assert_allclose(reshaped_ds.latitude.data, np.array(lat_vals))
    np.testing.assert_allclose(reshaped_ds.longitude.data, np.array(lon_vals))

    # Check that the variable 'var' has been reshaped correctly.
    # The expected shape is (2, 2, 2, 2): for each combination of init_time and lead_time,
    # there is a scalar value for each (latitude, longitude) pair.
    expected_shape = (2, 2, 2, 2)
    assert reshaped_ds["var"].shape == expected_shape

    # Verify that values are correctly mapped.
    # For dummy 'a' (lat 10, lon 100), the value should equal the corresponding value from the original dataset.
    # For init_time index 0 and lead_time index 0, original value is 1.
    val_a = ds["var"].sel(dummy="a").isel(init_time=0, lead_time=0).item()
    val_from_reshaped_a = (
        reshaped_ds["var"]
        .sel(latitude=10.0, longitude=100.0)
        .isel(init_time=0, lead_time=0)
        .item()
    )
    assert val_a == val_from_reshaped_a

    # For dummy 'b' (lat 20, lon 110), for example init_time index 1 and lead_time index 1, original value is 8.
    val_b = ds["var"].sel(dummy="b").isel(init_time=1, lead_time=1).item()
    val_from_reshaped_b = (
        reshaped_ds["var"]
        .sel(latitude=20.0, longitude=110.0)
        .isel(init_time=1, lead_time=1)
        .item()
    )
    assert val_b == val_from_reshaped_b
