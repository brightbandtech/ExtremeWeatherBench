"""Forecast-shaped inputs and the result arrays an evaluation returns."""

import numpy as np
import pandas as pd
import xarray as xr


def make_sample_forecast_dataset():
    init_time = pd.date_range("2021-06-20", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.RandomState(21897820).standard_normal(
        size=(len(init_time), 181, 360, len(lead_time)),
    )
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                20 + 5 * data,
            ),
            "surface_eastward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "surface_northward_wind": (
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
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-21 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 24
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-20 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 23
    return dataset


def make_sample_forecast_with_valid_time():
    """Create a forecast dataset with valid_time dimension instead of
    init_time/lead_time."""
    valid_time = pd.date_range("2021-06-20", freq="6h", periods=40)
    data = np.random.RandomState(54321).standard_normal(size=(len(valid_time), 91, 180))
    latitudes = np.linspace(-90, 90, 91)
    longitudes = np.linspace(0, 359, 180)

    dataset = xr.Dataset(
        {
            "surface_air_temperature": (
                ["valid_time", "latitude", "longitude"],
                273.15 + 10 * data,
            ),
            "surface_pressure": (
                ["valid_time", "latitude", "longitude"],
                101325 + 1000 * data,
            ),
        },
        coords={
            "valid_time": valid_time,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    return dataset


def make_init_lead_forecast_dataset(n_init, n_lead, chunk=True, lead_unit=None):
    """Forecast-shaped dataset with init_time and lead_time dimensions."""
    lead = np.arange(0, n_lead * 6, 6)
    if lead_unit is not None:
        lead = pd.to_timedelta(lead, unit=lead_unit)
    ds = xr.Dataset(
        {
            "t": (
                ["init_time", "lead_time", "latitude"],
                np.arange(float(n_init * n_lead * 4)).reshape(n_init, n_lead, 4),
            )
        },
        coords={
            "init_time": pd.date_range("2020-01-01", periods=n_init, freq="D"),
            "lead_time": lead,
            "latitude": np.linspace(0, 3, 4),
        },
    )
    return ds.chunk({"init_time": 1}) if chunk else ds


def make_pattern_stack_dataarray(
    n_lead=3, n_valid=4, n_lat=9, n_lon=11, chunk=False, seed=0
):
    """Stack of 2-D non-negative fields, the shape SpatialDisplacement sees."""
    rng = np.random.default_rng(seed)
    values = rng.random((n_lead, n_valid, n_lat, n_lon))
    values[values < 0.4] = 0.0
    da = xr.DataArray(
        values,
        dims=["lead_time", "valid_time", "latitude", "longitude"],
        coords={
            "lead_time": np.arange(0, n_lead * 6, 6),
            "valid_time": pd.date_range("2021-02-01", periods=n_valid, freq="6h"),
            "latitude": np.linspace(30.0, 46.0, n_lat),
            "longitude": np.linspace(-130.0, -110.0, n_lon),
        },
    )
    return da.chunk({"lead_time": 1, "valid_time": 2}) if chunk else da


def make_sample_results_dataarray_list():
    results_da_list = [
        xr.DataArray(
            data=[5],
            dims=["lead_time"],
            coords={"lead_time": [0]},
        ),
        xr.DataArray(
            data=[6],
            dims=["lead_time"],
            coords={"lead_time": [6]},
        ),
    ]
    return results_da_list


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]
