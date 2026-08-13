"""Hypothesis property tests driving strategies.py through real EWB code."""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, note, settings

from extremeweatherbench import cases, evaluate, inputs, metrics, utils

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


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_convert_init_time_to_valid_time_matches_init_plus_lead(case):
    """Every output valid_time equals init_time + lead_time."""
    _note_case(case)
    assume(case.lead_dtype_is_timedelta)
    assume(case.coord_inconsistency_mode != "duplicate_init_time")
    assume(not _forecast_missing_data_applied(case))
    result = utils.convert_init_time_to_valid_time(case.forecast)
    for lead in case.forecast.lead_time.values:
        expected = np.sort(case.forecast.init_time.values + lead)
        da = result["surface_air_temperature"].sel(lead_time=lead)
        actual = np.sort(da.dropna("valid_time", how="all").valid_time.values)
        np.testing.assert_array_equal(actual, expected)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_convert_valid_time_to_init_time_roundtrip(case):
    """Converting to valid_time and back recovers the original init_times."""
    _note_case(case)
    assume(case.lead_dtype_is_timedelta)
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
    assume(case.lead_dtype_is_timedelta)
    init_idx, lead_idx = utils.derive_indices_from_init_time_and_lead_time(
        case.forecast, case.case.start_date, case.case.end_date
    )
    init_time = case.forecast.init_time.values
    lead_time = case.forecast.lead_time.values
    start = np.datetime64(case.case.start_date)
    end = np.datetime64(case.case.end_date)
    for i, j in zip(init_idx, lead_idx):
        valid_time = init_time[i] + lead_time[j]
        assert start <= valid_time <= end


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_xarray_forecast_subset_data_to_case(case):
    """Output valid_times and lat/lon fall inside the case window and region."""
    _note_case(case)
    assume(case.lead_dtype_is_timedelta)
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
    assume(case.lead_dtype_is_timedelta)
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
# Confirmed defect: idxmax/idxmin crash on a degenerate (all-NaN, or emptied
# by alignment) target valid_time axis instead of returning NaN. Pinned in
# tests/test_hypothesis_regressions.py.
IDXMINMAX_METRICS = {"MaximumMeanAbsoluteError", "MinimumMeanAbsoluteError"}


def _assume_pipeline_supported(case: strategies.ForecastTargetCase) -> None:
    assume(case.lead_dtype_is_timedelta)
    assume(
        case.domain_kind != "antimeridian"
        or case.forecast_longitude_convention == "0-360"
    )
    assume(
        case.domain_kind != "antimeridian"
        or case.target_longitude_convention == "0-360"
    )


@pytest.mark.parametrize("metric_name", list(METRIC_FACTORIES))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(strategies.forecast_target_case())
def test_metric_result_is_well_formed(metric_name, case):
    """Result is a DataArray, preserve_dims survives, and it is never inf."""
    _note_case(case)
    _assume_pipeline_supported(case)
    target_is_all_nan = case.missing_data_mode == "all_nan" and (
        case.missing_data_side in ("target", "both")
    )
    assume(metric_name not in IDXMINMAX_METRICS or not target_is_all_nan)
    # Confirmed defect: dropping an init_time can leave the aligned data with
    # a zero-length valid_time axis, which crashes DurationMeanError. Pinned
    # in tests/test_hypothesis_regressions.py.
    assume(
        metric_name != "DurationMeanError"
        or case.coord_inconsistency_mode != "drop_init_times"
    )
    metric = METRIC_FACTORIES[metric_name]()
    operator = _build_case_operator(case, metric)
    if metric_name in IDXMINMAX_METRICS:
        try:
            result_df = evaluate.compute_case_operator(operator)
        except (KeyError, ValueError):
            return
    else:
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
    _assume_pipeline_supported(case)
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
