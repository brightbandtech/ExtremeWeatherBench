import pytest
from extremeweatherbench import utils
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (180, 180), (360, 0), (-179, 181), (-360, 0), (540, 180), (359, 359)],
)
def test_convert_longitude_to_360_param(input, expected):
    assert utils.convert_longitude_to_360(input) == expected


def test_center_forecast_on_time(
    mock_single_init_time_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
):
    """Test that the center_forecast_on_time function properly centers the forecast on the given
    timestamp, including at edges of the forecast which produce truncated outputs."""
    hours = 48
    test_aligned_da = utils.temporal_align_dataarrays(
        mock_single_init_time_subset_forecast_dataarray,
        mock_subset_gridded_obs_dataarray,
        pd.Timestamp(
            mock_single_init_time_subset_forecast_dataarray.init_time[0].values
        ),
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


def test_temporal_align_dataarrays(mock_forecast_dataarray, mock_gridded_obs_dataarray):
    """Test that the conversion from init time to valid time (named as time) produces an aligned
    dataarray to ensure metrics are applied properly."""
    init_time_datetime = pd.Timestamp(
        mock_forecast_dataarray.init_time[0].values
    ).to_pydatetime()
    aligned_forecast, aligned_obs = utils.temporal_align_dataarrays(
        mock_forecast_dataarray, mock_gridded_obs_dataarray, init_time_datetime
    )

    # Check aligned datasets have same time coordinates
    assert (aligned_forecast.time == aligned_obs.time).all()

    # Check forecast was properly subset by init time
    assert aligned_forecast.init_time.size == 1
    assert pd.Timestamp(aligned_forecast.init_time.values) == pd.Timestamp(
        mock_forecast_dataarray.init_time[0].values
    )
