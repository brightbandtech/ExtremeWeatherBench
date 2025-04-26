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


def test_align_point_obs_from_gridded_basic(
    sample_forecast_dataset, sample_point_obs_df_with_attrs
):
    """Test basic functionality of align_point_obs_from_gridded."""  # Adjust point obs to match forecast time
    valid_time = pd.Timestamp(
        sample_forecast_dataset.init_time[0].values
    ) + pd.Timedelta(hours=6)
    sample_point_obs_df_with_attrs.iloc[
        0, sample_point_obs_df_with_attrs.columns.get_loc("time")
    ] = valid_time
    sample_point_obs_df_with_attrs.iloc[
        1, sample_point_obs_df_with_attrs.columns.get_loc("time")
    ] = valid_time
    # Convert longitude to 0-360 range to match forecast
    sample_point_obs_df_with_attrs["longitude"] = sample_point_obs_df_with_attrs[
        "longitude"
    ].apply(lambda x: x + 360 if x < 0 else x)

    data_var = ["surface_air_temperature"]
    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataset, sample_point_obs_df_with_attrs, data_var
    )
    # Check basic properties
    assert isinstance(forecast, xr.Dataset)
    assert isinstance(obs, xr.Dataset)
    assert "station_id" in forecast.dims
    assert "station_id" in obs.dims
    assert len(forecast.station_id) == len(sample_point_obs_df_with_attrs)
    assert len(obs.station_id) == len(sample_point_obs_df_with_attrs)
    assert all(
        station in forecast.station_id.values
        for station in sample_point_obs_df_with_attrs["station_id"]
    )


def test_align_point_obs_from_gridded_multiple_times(
    sample_forecast_dataarray, sample_point_obs_df_with_attrs
):
    """Test aligning point observations with multiple valid times."""
    # Create a larger dataframe with multiple times
    dfs = []
    for i in range(3):
        df_copy = sample_point_obs_df_with_attrs.copy()
        valid_time = pd.Timestamp(
            sample_forecast_dataarray.init_time[0].values
        ) + pd.Timedelta(hours=i * 6)
        df_copy["time"] = valid_time
        dfs.append(df_copy)

    multi_time_df = pd.concat(dfs, ignore_index=True)

    # Convert longitude to 0-360 range
    multi_time_df["longitude"] = multi_time_df["longitude"].apply(
        lambda x: x + 360 if x < 0 else x
    )

    # Set latitude and longitude values that fall within the forecast grid
    multi_time_df["latitude"] = np.tile(np.array([40.5, 41.8]), 3)
    multi_time_df["longitude"] = np.tile(
        np.array([260.5, 260.2]), 3
    )  # ~-99.5, ~-99.8 in 0-360 space

    data_var = ["surface_air_temperature"]

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataarray, multi_time_df, data_var
    )

    # Check there are multiple stations and times
    assert len(forecast.station_id) == len(multi_time_df)
    assert len(np.unique(forecast.time.values)) == 3


def test_align_point_obs_from_gridded_missing_times(
    sample_forecast_dataarray, sample_point_obs_df_with_attrs
):
    """Test behavior when some observation times don't match forecast times."""
    # Create a dataframe with some matching times and some that don't match
    valid_time1 = pd.Timestamp(
        sample_forecast_dataarray.init_time[0].values
    ) + pd.Timedelta(hours=6)
    valid_time2 = pd.Timestamp(
        sample_forecast_dataarray.init_time[0].values
    ) + pd.Timedelta(hours=7)  # Not in forecast

    df = sample_point_obs_df_with_attrs.copy()
    df.iloc[0, df.columns.get_loc("time")] = valid_time1
    df.iloc[1, df.columns.get_loc("time")] = valid_time2

    # Convert longitude to 0-360 range
    df["longitude"] = df["longitude"].apply(lambda x: x + 360 if x < 0 else x)

    # Set latitude and longitude values that fall within the forecast grid
    df["latitude"] = np.array([40.5, 41.8])
    df["longitude"] = np.array([260.5, 260.2])  # ~-99.5, ~-99.8 in 0-360 space

    data_var = ["surface_air_temperature"]

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataarray, df, data_var
    )

    # Should only include the valid time that matches a forecast time
    assert len(forecast.station_id) == 1


def test_align_point_obs_from_gridded_out_of_bounds(
    sample_forecast_dataarray, sample_point_obs_df_with_attrs
):
    """Test behavior when point observations are outside the forecast domain."""
    # Set timestamp to match forecast
    valid_time = pd.Timestamp(
        sample_forecast_dataarray.init_time[0].values
    ) + pd.Timedelta(hours=6)
    sample_point_obs_df_with_attrs["time"] = valid_time

    # Put one point within forecast domain and one outside
    sample_point_obs_df_with_attrs["latitude"] = np.array(
        [40.5, 95.0]
    )  # 95 is outside valid range
    sample_point_obs_df_with_attrs["longitude"] = np.array([260.5, 260.2])

    data_var = ["surface_air_temperature"]

    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataarray, sample_point_obs_df_with_attrs, data_var
    )

    # Should only include the point within the domain
    assert len(forecast.station_id) == 1
    assert (
        forecast.station_id.values[0] == sample_point_obs_df_with_attrs["station_id"][0]
    )


def test_align_point_obs_from_gridded_all_empty():
    # Test with empty inputs
    forecast, obs = utils.align_point_obs_from_gridded(
        xr.Dataset(),  # invalid forecast
        pd.DataFrame(),  # empty dataframe
        [],  # empty data vars
    )
    assert isinstance(forecast, xr.Dataset)
    assert isinstance(obs, xr.Dataset)
    assert len(forecast.data_vars) == 0
    assert len(obs.data_vars) == 0


def test_align_point_obs_from_gridded_empty_point_obs(
    sample_forecast_dataarray, sample_point_obs_df_with_attrs
):
    """Test behavior when point observations are empty."""
    # Create an empty DataFrame with the same columns as the sample
    empty_df = pd.DataFrame(
        columns=[
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "elevation",
            "surface_air_temperature",
            "surface_air_pressure",
            "air_pressure_at_mean_sea_level",
            "accumulated_1_hour_precipitation",
            "surface_wind_speed",
            "surface_wind_from_direction",
            "surface_dew_point_temperature",
            "surface_relative_humidity",
            "hour_dist",
            "time",
            "case_id",
        ]
    )
    # Add metadata attributes to match the sample
    empty_df.attrs = sample_point_obs_df_with_attrs.attrs

    data_var = ["surface_air_temperature"]

    # Test with empty observations
    forecast, obs = utils.align_point_obs_from_gridded(
        sample_forecast_dataarray, empty_df, data_var
    )

    # Should return empty datasets
    assert isinstance(forecast, xr.Dataset)
    assert isinstance(obs, xr.Dataset)
    assert len(forecast.data_vars) == 0
    assert len(obs.data_vars) == 0

    # Set timestamp to match forecast


def test_align_point_obs_from_gridded_vars_empty_forecast(
    sample_point_obs_df_with_attrs,
):
    """Test behavior when point observations have variables not in the forecast."""
    # Create a dataframe with variables not in the forecast
    data_var = [
        "surface_air_temperature",
        "surface_air_pressure",
        "air_pressure_at_mean_sea_level",
        "accumulated_1_hour_precipitation",
        "surface_wind_speed",
        "surface_wind_from_direction",
        "surface_dew_point_temperature",
    ]

    empty_forecast = xr.Dataset()
    # If the forecast is empty, the function should raise a KeyError
    # there should have been an exception raised well before this point in this case
    with pytest.raises(KeyError):
        forecast, obs = utils.align_point_obs_from_gridded(
            empty_forecast, sample_point_obs_df_with_attrs, data_var
        )


def test_location_subset_point_obs():
    # Create sample data
    df = pd.DataFrame(
        {
            "latitude": [0, 1, 2, 3, 4],
            "longitude": [0, 1, 2, 3, 4],
            "value": ["a", "b", "c", "d", "e"],
        }
    )

    # Test bounds
    result = utils.location_subset_point_obs(
        df, min_lat=1, max_lat=3, min_lon=1, max_lon=3
    )
    assert len(result) == 3
    assert all(result["value"] == ["b", "c", "d"])
    assert all((result["latitude"] >= 1) & (result["latitude"] <= 3))
    assert all((result["longitude"] >= 1) & (result["longitude"] <= 3))

    # Test empty result
    result = utils.location_subset_point_obs(
        df, min_lat=10, max_lat=20, min_lon=10, max_lon=20
    )
    assert len(result) == 0
