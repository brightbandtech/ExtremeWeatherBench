"""Deterministic repros for defects found by test_hypothesis_fixtures.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, metrics, regions, utils
from extremeweatherbench.sources import xarray_dataset


def test_convert_init_time_to_valid_time_coerces_int_lead_time_to_hours():
    """Int-hours lead_time is added to init_time as hours, not nanoseconds."""
    init_time = pd.date_range("2021-06-20", periods=2, freq="6h")
    lead_time = np.array([0, 6, 12])
    ds = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "lead_time"],
                np.arange(6.0).reshape(2, 3),
            )
        },
        coords={"init_time": init_time, "lead_time": lead_time},
    )
    result = utils.convert_init_time_to_valid_time(ds)
    lead_hours = pd.to_timedelta(lead_time, unit="h").to_numpy()
    # Overlapping init_time/lead_time combinations share a valid_time slot,
    # so the result axis is the unique set of combined times, not all pairs.
    combos = (init_time.values[:, None] + lead_hours[None, :]).ravel()
    expected = np.sort(np.unique(combos))
    actual = np.sort(result.valid_time.values)
    np.testing.assert_array_equal(actual, expected)


def test_check_for_spatial_data_handles_antimeridian_seam():
    longitude = np.array([170.0, 175.0, -180.0, -175.0, -170.0])
    latitude = np.array([20.0, 30.0, 40.0])
    ds = xr.Dataset(
        {"v": (["latitude", "longitude"], np.zeros((3, 5)))},
        coords={"latitude": latitude, "longitude": longitude},
    )
    region = regions.BoundingBoxRegion(
        latitude_min=20.0,
        latitude_max=40.0,
        longitude_min=150.0,
        longitude_max=-100.0,
    )
    assert xarray_dataset.check_for_spatial_data(ds, region)


def _global_grid(step: float = 0.25, lon_min: float = 0.0) -> xr.Dataset:
    longitude = np.arange(lon_min, lon_min + 360.0, step)
    latitude = np.arange(40.0, 76.0, step)
    data = np.broadcast_to(
        np.sin(np.deg2rad(longitude)), (latitude.size, longitude.size)
    ).copy()
    return xr.Dataset(
        {"v": (["latitude", "longitude"], data)},
        coords={"latitude": latitude, "longitude": longitude},
    )


@pytest.mark.parametrize(
    ("region_spec", "lon_min", "span"),
    [
        (125, 0.0, 32.4),
        (143, 0.0, 19.8),
        ("antimeridian", -180.0, 10.0),
    ],
)
def test_wrapping_region_mask_is_contiguous(region_spec, lon_min, span):
    """A region that wraps a longitude seam masks to one unbroken block."""
    step = 0.25
    if region_spec == "antimeridian":
        region = regions.BoundingBoxRegion.create_region(
            latitude_min=45.0,
            latitude_max=70.0,
            longitude_min=175.0,
            longitude_max=-175.0,
        )
    else:
        region = next(
            case.location
            for case in cases.load_cases()
            if case.case_id_number == region_spec
        )
    grid = _global_grid(step, lon_min)
    longitude = region.mask(grid).longitude.values
    assert longitude.size < grid.sizes["longitude"]
    np.testing.assert_allclose(np.diff(longitude), step)
    assert longitude.max() - longitude.min() == pytest.approx(span, abs=2 * step)


def _build_forecast_target(target_values: np.ndarray) -> tuple:
    valid_time = pd.date_range("2021-06-20", periods=3, freq="6h")
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])
    forecast = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    target = xr.DataArray(
        target_values,
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    return forecast, target


def test_maximum_mean_absolute_error_returns_nan_on_all_nan_target():
    forecast, target = _build_forecast_target(np.full((3, 2, 2), np.nan))
    metric = metrics.MaximumMeanAbsoluteError()
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())


def test_minimum_mean_absolute_error_returns_nan_on_all_nan_target():
    forecast, target = _build_forecast_target(np.full((3, 2, 2), np.nan))
    metric = metrics.MinimumMeanAbsoluteError()
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())


def test_duration_mean_error_returns_nan_on_empty_valid_time_axis():
    valid_time = pd.DatetimeIndex([], dtype="datetime64[ns]")
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])
    forecast = xr.DataArray(
        np.empty((0, 2, 2)),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    target = xr.DataArray(
        np.empty((0, 2, 2)),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    metric = metrics.DurationMeanError(threshold_criteria=273.0)
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())
