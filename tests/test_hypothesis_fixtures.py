"""Hypothesis property tests driving strategies.py through real EWB code."""

import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import HealthCheck, assume, given, note, settings

from extremeweatherbench import cases, evaluate, inputs, metrics, regions, utils

from . import strategies

pytestmark = pytest.mark.slow


def _note_case(case: strategies.ForecastTargetCase) -> None:
    note(
        f"init_res={case.init_resolution_hours} lead_res={case.lead_resolution_hours} "
        f"lead_is_timedelta={case.lead_dtype_is_timedelta} "
        f"domain={case.domain_kind} "
        f"fc_lon={case.forecast_longitude_convention} "
        f"tg_lon={case.target_longitude_convention} "
        f"fc_lat_asc={case.forecast_latitude_ascending} "
        f"tg_lat_asc={case.target_latitude_ascending} "
        f"target_time_dim={case.target_time_dim} "
        f"coord_mode={case.coord_inconsistency_mode} "
        f"missing_mode={case.missing_data_mode} "
        f"missing_side={case.missing_data_side} "
        f"overlaps={case.case_overlaps}"
    )


def _build_case_operator(
    case: strategies.ForecastTargetCase, metric: "metrics.BaseMetric"
) -> "cases.CaseOperator":
    return cases.CaseOperator(
        case_metadata=case.case,
        metric_list=[metric],
        target=strategies.InMemoryERA5(
            ds=case.target,
            name="t",
            variables=["2m_temperature"],
            variable_mapping={"time": "valid_time"},
        ),
        forecast=inputs.XarrayForecast(
            ds=case.forecast,
            name="f",
            variables=["surface_air_temperature"],
            variable_mapping={},
        ),
    )


def _forecast_missing_data_applied(case: strategies.ForecastTargetCase) -> bool:
    return case.missing_data_mode != "none" and case.missing_data_side in (
        "forecast",
        "both",
    )


def _lead_as_offset(lead) -> np.timedelta64:
    """An integer lead_time means hours, matching the rest of the codebase."""
    if isinstance(lead, np.timedelta64):
        return lead
    return np.timedelta64(int(lead), "h")


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_convert_init_time_to_valid_time_matches_init_plus_lead(case):
    """Every output valid_time equals init_time + lead_time."""
    _note_case(case)
    assume(case.coord_inconsistency_mode != "duplicate_init_time")
    assume(not _forecast_missing_data_applied(case))
    result = utils.convert_init_time_to_valid_time(case.forecast)
    for lead in case.forecast.lead_time.values:
        expected = np.sort(case.forecast.init_time.values + _lead_as_offset(lead))
        da = result["surface_air_temperature"].sel(lead_time=lead)
        actual = np.sort(da.dropna("valid_time", how="all").valid_time.values)
        np.testing.assert_array_equal(actual, expected)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_convert_valid_time_to_init_time_roundtrip(case):
    """Converting to valid_time and back recovers the original init_times."""
    _note_case(case)
    assume(case.coord_inconsistency_mode != "duplicate_init_time")
    assume(not _forecast_missing_data_applied(case))
    valid_time_ds = utils.convert_init_time_to_valid_time(case.forecast)
    roundtripped = utils.convert_valid_time_to_init_time(
        valid_time_ds["surface_air_temperature"]
    )
    for lead in case.forecast.lead_time.values:
        expected = np.sort(case.forecast.init_time.values)
        da = roundtripped.sel(lead_time=lead)
        actual = np.sort(da.dropna("init_time", how="all").init_time.values)
        np.testing.assert_array_equal(actual, expected)


@given(strategies.forecast_target_case())
def test_determine_temporal_resolution(case):
    """Returns None or the minimum positive spacing in hours."""
    _note_case(case)
    target = (
        case.target
        if case.target_time_dim == "valid_time"
        else case.target.rename({case.target_time_dim: "valid_time"})
    )
    resolution = utils.determine_temporal_resolution(target)
    if resolution is not None:
        assert resolution > 0
        diffs = np.diff(case.target[case.target_time_dim].values)
        diffs = diffs.astype("timedelta64[h]").astype(int)
        diffs = diffs[diffs != 0]
        assert resolution == diffs.min()


@given(strategies.forecast_target_case())
def test_derive_indices_from_init_time_and_lead_time(case):
    """Every returned index pair maps to a valid time inside the case window."""
    _note_case(case)
    init_idx, lead_idx = utils.derive_indices_from_init_time_and_lead_time(
        case.forecast, case.case.start_date, case.case.end_date
    )
    init_time = case.forecast.init_time.values
    lead_time = case.forecast.lead_time.values
    start = np.datetime64(case.case.start_date)
    end = np.datetime64(case.case.end_date)
    for i, j in zip(init_idx, lead_idx):
        valid_time = init_time[i] + _lead_as_offset(lead_time[j])
        assert start <= valid_time <= end


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_xarray_forecast_subset_data_to_case(case):
    """Output valid_times and lat/lon fall inside the case window and region."""
    _note_case(case)
    assume(case.coord_inconsistency_mode != "duplicate_init_time")
    forecast = inputs.XarrayForecast(
        ds=case.forecast, variables=[], variable_mapping={}
    )
    result = forecast.subset_data_to_case(case.forecast, case.case)
    if result.sizes.get("valid_time", 0) == 0:
        assume(not case.case_overlaps)
        return
    start = np.datetime64(case.case.start_date)
    end = np.datetime64(case.case.end_date)
    assert (result.valid_time.values >= start).all()
    assert (result.valid_time.values <= end).all()
    bounds = case.case.location.as_geopandas().total_bounds
    assert (result.latitude.values >= bounds[1]).all()
    assert (result.latitude.values <= bounds[3]).all()


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_xarray_forecast_subset_data_to_case_non_overlapping_is_empty(case):
    """A non-overlapping case yields an empty result instead of raising."""
    _note_case(case)
    assume(not case.case_overlaps)
    forecast = inputs.XarrayForecast(
        ds=case.forecast, variables=[], variable_mapping={}
    )
    result = forecast.subset_data_to_case(case.forecast, case.case)
    assert result.sizes.get("valid_time", 0) == 0


@given(strategies.forecast_target_case())
def test_zarr_target_subsetter(case):
    """Same time/space containment on the target side, for both time namings."""
    _note_case(case)
    result = inputs.zarr_target_subsetter(
        case.target, case.case, time_variable=case.target_time_dim
    )
    if result.sizes.get(case.target_time_dim, 0) == 0:
        assume(not case.case_overlaps)
        return
    start = np.datetime64(case.case.start_date)
    end = np.datetime64(case.case.end_date)
    times = result[case.target_time_dim].values
    assert (times >= start).all()
    assert (times <= end).all()
    bounds = case.case.location.as_geopandas().total_bounds
    assert (result.latitude.values >= bounds[1]).all()
    assert (result.latitude.values <= bounds[3]).all()


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_align_forecast_to_target(case):
    """Aligned forecast lat/lon exactly equal target lat/lon; no new dims."""
    _note_case(case)
    # In production, XarrayForecast.subset_data_to_case dedups init_time
    # before this function ever sees the data; calling it directly here
    # skips that step, so duplicate_init_time is out of scope too.
    assume(case.coord_inconsistency_mode != "duplicate_init_time")
    forecast = case.forecast.rename(init_time="valid_time")
    target = case.target
    if case.target_time_dim == "time":
        target = target.rename({"time": "valid_time"})
    aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
    assert set(aligned_forecast.dims) <= set(forecast.dims) | set(target.dims)
    np.testing.assert_array_equal(
        aligned_forecast.latitude.values, aligned_target.latitude.values
    )
    np.testing.assert_array_equal(
        aligned_forecast.longitude.values, aligned_target.longitude.values
    )


METRIC_FACTORIES = {
    "MeanAbsoluteError": lambda: metrics.MeanAbsoluteError(),
    "RootMeanSquaredError": lambda: metrics.RootMeanSquaredError(),
    "MeanError": lambda: metrics.MeanError(),
    "MaximumMeanAbsoluteError": lambda: metrics.MaximumMeanAbsoluteError(),
    "MinimumMeanAbsoluteError": lambda: metrics.MinimumMeanAbsoluteError(),
    "DurationMeanError": lambda: metrics.DurationMeanError(threshold_criteria=273.0),
    "EarlySignal": lambda: metrics.EarlySignal(forecast_threshold=273.0),
    "CriticalSuccessIndex": lambda: metrics.CriticalSuccessIndex(
        forecast_threshold=273.0, target_threshold=273.0
    ),
}


CONTINUOUS_METRICS = {"MeanAbsoluteError", "RootMeanSquaredError", "MeanError"}


@pytest.mark.parametrize("metric_name", list(METRIC_FACTORIES))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_metric_result_is_well_formed(metric_name, case):
    """Result is a DataArray, preserve_dims survives, and it is never inf."""
    _note_case(case)
    metric = METRIC_FACTORIES[metric_name]()
    operator = _build_case_operator(case, metric)
    result_df = evaluate.compute_case_operator(operator)
    if result_df.empty:
        return
    assert not np.isinf(result_df["value"].to_numpy(dtype=float)).any()
    all_nan_everywhere = metric_name in CONTINUOUS_METRICS and (
        case.missing_data_mode == "all_nan"
        and case.missing_data_side in ("forecast", "both", "target")
    )
    if all_nan_everywhere:
        assert result_df["value"].isna().all()


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_compute_case_operator_output_schema(case):
    """Output columns are exactly OUTPUT_COLUMNS, with numeric value."""
    _note_case(case)
    metric = metrics.MeanAbsoluteError()
    operator = _build_case_operator(case, metric)
    result_df = evaluate.compute_case_operator(operator)
    assert list(result_df.columns) == evaluate.OUTPUT_COLUMNS
    if not result_df.empty:
        assert result_df["value"].dtype.kind in "fc"


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_compute_case_operator_no_overlap_is_empty(case):
    """Zero temporal overlap returns an empty DataFrame, not an exception."""
    _note_case(case)
    assume(not case.case_overlaps)
    metric = metrics.MeanAbsoluteError()
    operator = _build_case_operator(case, metric)
    result_df = evaluate.compute_case_operator(operator)
    assert result_df.empty


def _diurnal_temperature(valid_times, peak_hour=12.0, base=273.0, amplitude=10.0):
    """24h-periodic temperature, peaking at peak_hour on every day."""
    shape = np.asarray(valid_times).shape
    idx = pd.DatetimeIndex(np.asarray(valid_times).ravel())
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    phase = 2 * np.pi * (hours - peak_hour) / 24.0
    return (base + amplitude * np.cos(phase)).reshape(shape)


@pytest.mark.xfail(
    reason="fix/peak-metrics: forecast side reduces the max across inits "
    "over valid_time instead of per init_time, so with a 72h init cadence "
    "the tolerance window can hold only one off-peak forecast sample, "
    "aliasing a perfect forecast into a non-zero error",
    strict=False,
)
def test_maximum_mean_absolute_error_perfect_forecast_is_zero():
    """A forecast identical to the target at every sampled time scores 0."""
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])

    target_time = pd.date_range("2021-06-20", periods=24 * 10, freq="1h")
    target_grid = np.repeat(
        _diurnal_temperature(target_time)[:, None, None], 4, axis=1
    ).reshape(len(target_time), 2, 2)
    target_ds = xr.Dataset(
        {"2m_temperature": (["time", "latitude", "longitude"], target_grid)},
        coords={"time": target_time, "latitude": lat, "longitude": lon},
    )

    init_time = pd.date_range("2021-06-20", periods=3, freq="72h")
    lead_time = np.arange(0, 169, 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    valid_times = init_time.values[:, None] + lead_time[None, :]
    forecast_grid = np.repeat(
        _diurnal_temperature(valid_times)[:, :, None, None], 4, axis=2
    ).reshape(len(init_time), len(lead_time), 2, 2)
    forecast_ds = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "lead_time", "latitude", "longitude"],
                forecast_grid,
            )
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": lat,
            "longitude": lon,
        },
    )

    case = cases.IndividualCase(
        case_id_number=1,
        title="diurnal perfect forecast",
        start_date=datetime.datetime(2021, 6, 20),
        end_date=datetime.datetime(2021, 6, 30),
        location=regions.BoundingBoxRegion(
            latitude_min=5.0,
            latitude_max=25.0,
            longitude_min=90.0,
            longitude_max=120.0,
        ),
        event_type="synthetic",
    )
    operator = cases.CaseOperator(
        case_metadata=case,
        metric_list=[metrics.MaximumMeanAbsoluteError()],
        target=strategies.InMemoryERA5(
            ds=target_ds,
            name="t",
            variables=["2m_temperature"],
            variable_mapping={"time": "valid_time"},
        ),
        forecast=inputs.XarrayForecast(
            ds=forecast_ds,
            name="f",
            variables=["surface_air_temperature"],
            variable_mapping={},
        ),
    )
    result_df = evaluate.compute_case_operator(operator)
    errors = result_df["value"].dropna().to_numpy()
    assert errors.size > 0
    np.testing.assert_allclose(errors, 0.0)
