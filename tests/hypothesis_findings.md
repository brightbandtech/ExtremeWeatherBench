# Hypothesis fixture findings

The Hypothesis suite in `tests/test_hypothesis_fixtures.py` found four genuine
defects, all fixed across two stacked branches: defects 2, 3 and 4 on
`fix/spatial-check-and-metric-edge-cases`, defect 1 here. Each fix flipped its
deterministic repro in `tests/test_hypothesis_regressions.py` from `xfail` to a
real assertion, and removed the `assume()` narrowing that had been hiding it,
so the property tests now cover the axis that exposed it. No `xfail` remains in
that module.

## Resolved defects

### 1. Integer `lead_time` silently misread as nanoseconds

- **Axis combination**: `lead_dtype_is_timedelta=False` (any other axes).
- **Symptom**: `utils.convert_init_time_to_valid_time` added `lead_time` to
  `init_time` with no dtype coercion, so an integer array of hours was read as
  nanoseconds and `valid_time` collapsed onto `init_time` (lead `6` became
  `6 ns`, not 6 hours). No exception was raised.
  `utils.convert_valid_time_to_init_time` had the same uncoerced subtraction.
- **Root cause**: `utils.py`, `xr.DataArray(ds.init_time) + xr.DataArray(
  ds.lead_time)`. Integer-means-hours is the convention everywhere else in the
  codebase (`utils.py` `pd.to_timedelta(..., unit="h")`, `calc.py` and
  `metrics.py` `np.timedelta64(int(...), "h")`), so these two were the
  outliers.
- **Resolution**: `utils._lead_time_as_timedelta`, applied in both directions.
  It returns the input untouched when it is already `timedelta64`, which is the
  production path, so real runs pay nothing. The int-hour `lead_time` fixtures
  in `tests/conftest.py` and `tests/test_evaluate.py` were migrated to
  `timedelta64[ns]` to match what the CIRA preprocess hooks in `defaults.py`
  actually produce; the `lead_dtype_is_timedelta=False` strategy axis keeps the
  coercion path covered.

  Removing the six `assume(case.lead_dtype_is_timedelta)` narrowings exposed two
  further places that assumed `timedelta64` lead times. Both are fixed here:

  - `metrics._calculate_event_duration` derived its lead-time step with
    `np.diff(mask.lead_time.values)`. With integer hours, `expected_gap /
    lt_step[0]` yields a `datetime.timedelta` rather than a ratio and the
    surrounding `int()` raises `TypeError`. It now coerces through the same
    helper, which makes both dtypes behave identically.
  - `test_convert_init_time_to_valid_time_matches_init_plus_lead` and
    `test_derive_indices_from_init_time_and_lead_time` computed their *expected*
    values with raw `init_time + lead` arithmetic, reproducing the very bug
    under test. The first failed outright once the narrowing was gone; the
    second passed vacuously, because a nanosecond-collapsed valid_time still
    falls inside the case window. Both now use a local `_lead_as_offset` helper
    that spells out the integer-means-hours convention independently of the
    production implementation, so the assertion is not tautological.

### 2. Antimeridian-crossing `-180/180` grids crash the pre-flight spatial check

- **Axis combination**: `domain_kind="antimeridian"` with either
  `forecast_longitude_convention` or `target_longitude_convention` set to
  `"-180-180"`.
- **Symptom**: `KeyError: 180.0` from `.sel(longitude=slice(...))` on a
  seam-crossing longitude axis, raised inside
  `sources/xarray_dataset.py::check_for_spatial_data` before any metric ran.
- **Root cause**: `.sel()` with a `slice` requires a monotonic index, which a
  seam-crossing axis is not in the region's frame.
- **Resolution**: the slice-based block is replaced with boolean comparisons on
  the 1-D coordinate arrays, which are monotonicity- and order-agnostic. That
  made the reversed-latitude retry dead, so it is gone. The check now reads only
  coordinates and never indexes a data variable, so it stays lazy-safe.

  Two details are worth recording, because both were non-obvious:

  - `Region.as_geopandas().total_bounds` **cannot** express an antimeridian
    region. The geometry is a `MultiPolygon` split at the seam, so its combined
    bounds always collapse to the full globe, `[-180, lat_min, 180, lat_max]`.
    The fix therefore uses `geometry.explode(...).bounds` to recover the two
    real lobes and reports a hit if any lobe matches. Using the collapsed bounds
    instead would make the longitude test vacuous for every antimeridian region.
  - The `% 360` normalisation inverts any lobe straddling longitude 0, so a
    plain region like 30W-30E normalises to `(330.0, 30.0)`. Longitude matching
    keeps two branches for this: `AND` when the normalised bounds are ordered,
    `OR` when they wrap. Both branches are reachable, and the wrap branch is
    about the prime meridian, not the antimeridian. `tests/test_sources.py`
    covers it, since no `domain_kind` in `strategies.py` generates that shape.

  The new check is stricter than the old one in one respect: the old code
  returned `sum(data.sizes.values()) > 0`, which summed the latitude and
  longitude sizes and so reported data whenever *latitude* matched. It is now a
  true `AND` over both axes. No existing test depended on the loose behavior.

### 3. `MaximumMeanAbsoluteError` / `MinimumMeanAbsoluteError` crash on a
   degenerate target time axis instead of returning NaN

- **Axis combination**: `missing_data_mode="all_nan"` with `missing_data_side`
  in `("target", "both")`, or any `coord_inconsistency_mode` that leaves the
  aligned target with a zero-length `valid_time` axis.
- **Symptom**: with an all-NaN target, `idxmax`/`idxmin` return `NaT` and the
  following `.sel(valid_time=NaT)` raises `KeyError` instead of the metric
  yielding NaN, which is what every other metric does for all-NaN input. With a
  zero-length axis, `idxmax`/`idxmin` raise `ValueError` from
  `nanargmax`/`nanargmin`.
- **Resolution**: an early return in both `_compute_metric` methods, matching
  the idiom the landfall metrics already use, returning
  `utils._create_nan_dataarray(self.preserve_dims)`. The size test is ordered
  before `.isnull().all()` so the reduction, which forces a compute on dask
  input, is only reached when the axis is non-empty. The guard also short
  circuits ahead of the forecast-side reduction, the more expensive half.

### 4. `DurationMeanError` crashes on a zero-length `valid_time` axis

- **Axis combination**: `coord_inconsistency_mode="drop_init_times"`. Dropping
  an init_time can leave the aligned data with a zero-length `valid_time` axis
  even when `case_overlaps=True` at the unaligned level.
- **Symptom**: `metrics._calculate_event_duration` built its gap mask with
  `np.concatenate([[False], ...])`, which yields a length-1 array from the
  leading `[False]` even for empty input, against a length-0 `valid_time`
  coordinate. `xr.DataArray` then raised `CoordinateValidationError`.
- **Resolution**: the mask is preallocated with `np.zeros(vt.size, dtype=bool)`,
  correct at sizes 0 and 1 and cheaper than concatenating. An early return for a
  zero-length axis was added on top, because a second failure surfaces once the
  gap construction is fixed: the downstream `groupby(preserve_dims)` raises
  `KeyError('init_time')` on empty input. Empty input now yields NaN.

## Strategy and coverage extensions

- `INIT_RESOLUTIONS_HOURS` gained `72`, so the strategies can generate the
  72-hour 00Z-only production cadence documented in `HEAT_METRIC_QC.md`. The
  rejection rate is unchanged at 300 examples (47.36% to 47.59%), because the
  case window is derived from the generated valid-time range and so scales with
  whatever cadence is drawn.
- `test_maximum_mean_absolute_error_perfect_forecast_is_zero` is the suite's
  first value-level oracle. Everything before it asserted only absence-of-crash
  and output well-formedness, which is structurally blind to a wrong-but-finite
  answer. See the peak-metric finding below for what it pins.

## Resolved on `fix/peak-metrics`

### The peak metrics aliased the diurnal cycle at coarse init cadences

`MaximumMeanAbsoluteError` took a true maximum on the target side, but on the
forecast side reduced over a `tolerance_range_hours` window pooled across
initializations rather than per initialization. At a 72-hour cadence that window
can hold a single forecast sample, so the "maximum" it compared against was
whichever lone snapshot happened to land in the window. Because every init is
00Z, that snapshot's time of day is fixed by `lead_time mod 24 h`, which aliased
the diurnal cycle straight into the reported error.

The consequence was that a forecast *identical to the target* did not score
zero. With a target peaking at 283 K at 12Z and a perfect forecast sampled every
6 hours from 72-hourly 00Z inits, the per-lead errors were 20 K at lead 0, 10 K
at lead 6, 0 K at lead 12 where the sampling happens to align, then 10 K and
20 K again. `MinimumMeanAbsoluteError` had the same construction.

`fix/peak-metrics` reduces over `lead_time` at fixed `init_time`, so each
initialization contributes its own trajectory through the window and the
reported lead is `peak_time - init_time`.
`test_maximum_mean_absolute_error_perfect_forecast_is_zero` was written here as
an `xfail` naming that branch as the resolving change; it now passes and the
marker is gone.

## Open findings, not fixed here

### `Region.mask` does not filter longitude for antimeridian regions

`Region.get_adjusted_bounds` derives its bounds from
`as_geopandas().total_bounds`, which for a seam-crossing `MultiPolygon`
collapses to `[-180, lat_min, 180, lat_max]` (see defect 2). `Region.mask` then
detects the `MultiPolygon` and takes its wrap branch, `lon >= -180 OR
lon <= 180`, which every longitude satisfies. So an antimeridian region masks on
latitude only and keeps the entire longitude axis.

Found by cross-checking the rewritten `check_for_spatial_data` against
`Region.mask` over 400 random region/grid pairs. The five disagreements were all
of this form, and in every one the new check was right and `mask` was wrong.
Left alone because `regions.py` was outside the scope of these fixes, but it is
a real defect and it is now the looser of the two code paths.

### `DurationMeanError` requires an `init_time` coordinate to exist

`_calculate_event_duration` raises `ValueError: You are requesting to preserve a
dimension which does not appear in your data` whenever forecast and target lack
an `init_time` coordinate entirely, regardless of `valid_time` length. This is
distinct from defect 4 and predates it; the empty-axis early return only hides
it for the empty case. Not filed as part of this work.

## Strategy artifacts, not defects

- **Forecast-side `all_nan` modes combined with a structural "which valid_times
  survived `convert_init_time_to_valid_time`" check**: distinguishing "this
  valid_time was never produced by this lead" (an artifact of the outer-join
  reindex) from "this valid_time was produced but is NaN because the strategy
  blanked it" is ambiguous from values alone. Narrowed with
  `assume(not _forecast_missing_data_applied(case))` in the two conversion
  property tests.
- **`coord_inconsistency_mode="duplicate_init_time"` fed directly to
  `align_forecast_to_target`**: in production,
  `ForecastBase.subset_data_to_case` deduplicates `init_time` before this
  function runs. Calling it directly on undeduplicated data, as the unit test
  does, produces a duplicate `valid_time` index that `xr.align` rejects.
  Narrowed with `assume(coord_inconsistency_mode != "duplicate_init_time")`.
- **`DurationMeanError` / `CriticalSuccessIndex` returning `0` rather than `NaN`
  on all-NaN input**: both are threshold-based and mask NaN to "no event" /
  `False` instead of propagating it. The masking is explicit in the
  implementation, so it is deliberate, but it is documented in neither
  docstring. The "all-NaN yields NaN" property is therefore asserted only for
  the plain continuous metrics. Still worth a maintainer's confirmation: a
  duration error of exactly `0` is indistinguishable from "no data at all".

## Pre-existing infrastructure notes

- `pyproject.toml` declares `[tool.pytest]`, not `[tool.pytest.ini_options]`.
  Pytest only reads the latter, so the `addopts`/`markers` configured there,
  including the `slow` marker this module uses, are silently inert. Recorded,
  not fixed.
- `tests/conftest.py::make_sample_forecast_dataset` used plain `int` hours,
  exactly the shape that triggers defect 1, while production preprocessing
  produces `timedelta64`. That fixture had been silently exercising the buggy
  path. Migrated here.

## Verification

Full suite `1137 passed, 1 xfailed`
(`uv run pytest -q --ignore=tests/test_golden.py`), against a base of
`1130 passed, 5 xfailed`. All five original xfails became real assertions, two
`check_for_spatial_data` tests were added for the prime-meridian shape, and the
single remaining xfail is the new peak-metric oracle, which is waiting on
`fix/peak-metrics`.

Applying the coercion alone, before any fixture change, left the suite at
`1130 passed, 5 xfailed`, unchanged. Nothing in the suite had encoded the
nanosecond misreading as an expectation, so the fix had no blast radius of its
own; every failure that appeared came later, from removing the narrowings.

`HYPOTHESIS_PROFILE=ewb-sweep uv run pytest tests/test_hypothesis_fixtures.py`:
`18 passed, 1 xfailed` at 300 examples each. `pre-commit run --all-files` clean.

The numbers above are as of this branch. On `fix/peak-metrics`, which landed
after it, the peak-metric oracle passes and its `xfail` is removed, leaving no
`xfail` anywhere in the Hypothesis modules.

Coverage widened as intended. `--hypothesis-show-statistics` reports 22 distinct
rejection categories before the narrowings were removed and 8 after. The nine
`_assume_pipeline_supported` categories, which had been discarding 7-29% of
examples on every metric and schema test, are gone entirely, as is
`test_derive_indices_from_init_time_and_lead_time`'s only narrowing. What
remains is the deliberate strategy artifacts below plus the case-overlap
filters, which are inherent to the no-overlap tests.
