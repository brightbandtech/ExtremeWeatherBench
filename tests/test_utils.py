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


def test_align_point_obs_from_gridded(sample_forecast_dataarray, sample_point_obs_df):
    point_obs_metadata_vars = utils.POINT_OBS_METADATA_VARS

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataarray, sample_point_obs_df, point_obs_metadata_vars
    )

    # Test basic properties
    assert isinstance(forecast, xr.DataArray)
    assert isinstance(obs, xr.DataArray)

    # Test dimensions
    assert "station" in forecast.dims
    assert "station" in obs.dims
    assert len(forecast.station) == len(obs.station)

    # Test coordinates
    assert "init_time" in forecast.coords
    assert "lead_time" in forecast.coords
    assert "time" in forecast.coords

    # Test metadata preservation
    assert "elevation" in obs.coords
    assert "network" in obs.coords


def test_align_point_obs_from_gridded_empty_intersection(sample_forecast_da):
    # Create observations with no matching times
    df = pd.DataFrame(
        {
            "station": ["A"],
            "latitude": [35.5],
            "longitude": [-99.5],
            "temperature": [20.0],
            "elevation": [1000],
            "network": ["METAR"],
        }
    )
    df = pd.concat([df], keys=[pd.Timestamp("2022-01-01")], names=["time"])

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_da, df, ["elevation", "network"]
    )

    # Should return empty DataArrays
    assert len(forecast.station) == 0
    assert len(obs.station) == 0


def test_align_point_obs_from_gridded_input_validation():
    # Test with invalid inputs
    with pytest.raises((AttributeError, TypeError)):
        utils.align_point_obs_from_gridded(
            None,  # invalid forecast
            pd.DataFrame(),  # empty dataframe
            [],  # empty metadata vars
        )


def test_align_point_obs_coordinate_values(sample_forecast_da, sample_point_obs_df):
    point_obs_metadata_vars = ["elevation", "network"]

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_da, sample_point_obs_df, point_obs_metadata_vars
    )

    # Test that coordinates match between forecast and obs
    np.testing.assert_array_equal(forecast.station.values, obs.station.values)
    np.testing.assert_array_equal(forecast.init_time.values, obs.init_time.values)
    np.testing.assert_array_equal(forecast.lead_time.values, obs.lead_time.values)

    # Test that time coordinates are properly calculated
    for init_time, lead_time in zip(forecast.init_time, forecast.lead_time):
        expected_time = pd.Timestamp(init_time.values) + pd.Timedelta(
            hours=int(lead_time.values)
        )
        mask = (forecast.init_time == init_time) & (forecast.lead_time == lead_time)
        actual_time = forecast.time.where(mask).dropna("station").values[0]
        assert pd.Timestamp(actual_time) == expected_time
