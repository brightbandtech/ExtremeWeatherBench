import pytest
import xarray as xr
import pandas as pd
import numpy as np


@pytest.fixture
def mock_forecast_dataset():
    init_time = pd.date_range("2020-01-01", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.rand(5, 180, 360, len(lead_time))
    latitudes = np.linspace(-90, 90, 180)
    longitudes = np.linspace(-180, 179, 360)
    dataset = xr.Dataset(
        {
            "air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "eastward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "northward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
        },
        coords={
            "init_time": init_time,
            "latitude": latitudes,
            "longitude": longitudes,
            "lead_time": lead_time,
        },
    )
    lead_time_grid, init_time_grid = np.meshgrid(dataset.lead_time, dataset.init_time)
    # Step 2: Flatten the meshgrid and convert lead_time to timedelta
    valid_time = init_time_grid.flatten() + pd.to_timedelta(
        lead_time_grid.flatten(), unit="h"
    )
    dataset.coords["time"] = valid_time
    return dataset
