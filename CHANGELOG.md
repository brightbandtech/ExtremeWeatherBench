# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is the source of truth for what each tagged version shipped.
GitHub's auto-generated release notes can list PRs that were already
squash-merged into `main`, because `develop` still contains those original
commits. Prefer this changelog when writing GitHub release notes.

## [1.1.0] - 2026-08-13

### Added

- Unified evaluation progress bar in `progress.py`: one 0–100% bar with
  ETA in both serial and parallel mode, plus `--no-progress` and
  `EWB_DISABLE_PROGRESS` ([#381](https://github.com/brightbandtech/ExtremeWeatherBench/pull/381)).
- Shared-target precompute uses a tqdm bar (case/target/phase postfix
  and nested dask tasks) instead of one INFO line per target.
  `get_climatology` now logs before the remote zarr open.
- Python 3.14 support
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).
- Hypothesis property tests for forecast/target input variation
  ([#388](https://github.com/brightbandtech/ExtremeWeatherBench/pull/388)).
- CAPE accuracy note in the docs
  ([#384](https://github.com/brightbandtech/ExtremeWeatherBench/pull/384)).

### Changed

- Peak metrics (`MaximumMeanAbsoluteError`, `MinimumMeanAbsoluteError`,
  `MaximumLowestMeanAbsoluteError`) reduce each forecast initialization
  over its own `lead_time` window, not across different model runs at a
  fixed lead time. Completeness guards count present values rather than
  coordinate slots
  ([#393](https://github.com/brightbandtech/ExtremeWeatherBench/pull/393)).
- CAPE now marches the moist adiabat and uses a single epsilon constant.
  Numeric CAPE output changes (about 2.6% of the mean signal on the ERA5
  reference profiles). Stored CAPE-derived scores will not match 1.0.2
  ([#384](https://github.com/brightbandtech/ExtremeWeatherBench/pull/384)).
- Package imports are lazy via `lazy-loader`. Legacy aliases such as
  `ewb.forecasts.ZarrForecast` and `ewb.targets.ERA5` no longer work;
  use `ewb.inputs.ZarrForecast` and `ewb.inputs.ERA5`
  ([#352](https://github.com/brightbandtech/ExtremeWeatherBench/pull/352)).
- Requires Python `>=3.12,<3.15`. Python 3.11 is no longer supported
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).
- `cartopy` moved out of core dependencies into the `data-prep` extra.
  Natural Earth land masks now use pooch and geopandas
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).
- Dependency refresh: `icechunk>=2.1.2`, `kerchunk>=0.2.10` from PyPI
  (git override removed), `pandas>=2.2.3,<3`, `numba>=0.66`, and
  `virtualizarr>=2.7.3` in `data-prep`
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).
- `MaximumMeanAbsoluteError` honors the instance `reduce_spatial_dims`
  instead of hardcoding latitude/longitude
  ([#383](https://github.com/brightbandtech/ExtremeWeatherBench/pull/383)).

### Removed

- Python 3.11 support
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).
- Unused core pins (`eccodes`, `frozenlist`, `pyogrio`) and `cartopy`
  from the default install
  ([#386](https://github.com/brightbandtech/ExtremeWeatherBench/pull/386)).

### Fixed

- `KerchunkForecast` no longer clobbers anonymous S3 access, so public
  CIRA/NODD kerchunk references open without credentials
  ([#382](https://github.com/brightbandtech/ExtremeWeatherBench/pull/382)).
- `open_kerchunk_reference` no longer mutates the caller's
  `storage_options` dict
  ([#382](https://github.com/brightbandtech/ExtremeWeatherBench/pull/382)).
- Documented example config uses `case_list` (the name the CLI expects)
  ([#382](https://github.com/brightbandtech/ExtremeWeatherBench/pull/382)).
- `check_for_spatial_data` no longer crashes on antimeridian-crossing
  `-180/180` grids, and now requires a match on both latitude and
  longitude
  ([#387](https://github.com/brightbandtech/ExtremeWeatherBench/pull/387)).
- Peak metrics and `DurationMeanError` return NaN on degenerate time
  axes instead of raising
  ([#387](https://github.com/brightbandtech/ExtremeWeatherBench/pull/387)).
- Integer `lead_time` values are treated as hours, not nanoseconds, in
  `convert_init_time_to_valid_time`
  ([#387](https://github.com/brightbandtech/ExtremeWeatherBench/pull/387)).
- `maybe_densify_dataarray` copies instead of mutating the parent
  dataset
  ([#383](https://github.com/brightbandtech/ExtremeWeatherBench/pull/383)).
- Landfall, timestep-completeness, and temporal-resolution helpers no
  longer materialize whole arrays to answer emptiness/shape questions
  ([#383](https://github.com/brightbandtech/ExtremeWeatherBench/pull/383)).

## [1.0.2.post1] - 2026-04-30

Documentation-only post-release so PyPI includes the docs refresh.

### Changed

- Rebuilt case-study docs, filled in cookbook recipes, and switched the
  docs toolchain to zensical
  ([#370](https://github.com/brightbandtech/ExtremeWeatherBench/pull/370),
  [#369](https://github.com/brightbandtech/ExtremeWeatherBench/pull/369)).

## [1.0.2] - 2026-04-30

### Added

- Rank-oriented copula skill score (ROCSS)
  ([#300](https://github.com/brightbandtech/ExtremeWeatherBench/pull/300)).
- Temperature event finder and climatology scripts, including updated
  2 m temperature quantile climatology
  ([#345](https://github.com/brightbandtech/ExtremeWeatherBench/pull/345),
  [#349](https://github.com/brightbandtech/ExtremeWeatherBench/pull/349),
  [#354](https://github.com/brightbandtech/ExtremeWeatherBench/pull/354)).
- Marginal severe and temperature event cases
  ([#351](https://github.com/brightbandtech/ExtremeWeatherBench/pull/351)).
- `overlap_target_threshold` on `EarlySignal`
  ([#350](https://github.com/brightbandtech/ExtremeWeatherBench/pull/350)).
- Heatwave preprocess that removes ocean gridpoints
  ([#348](https://github.com/brightbandtech/ExtremeWeatherBench/pull/348)).
- Icechunk as a core dependency, with example scripts and the CIRA
  store updated accordingly
  ([#335](https://github.com/brightbandtech/ExtremeWeatherBench/pull/335),
  [#333](https://github.com/brightbandtech/ExtremeWeatherBench/pull/333)).

### Changed

- Dask and distributed moved to core dependencies
  ([#336](https://github.com/brightbandtech/ExtremeWeatherBench/pull/336)).
- Atmospheric river detection: pressure ceiling, cleaner derived
  variable, latitude filter above 15°, and forecast parallelization
  ([#344](https://github.com/brightbandtech/ExtremeWeatherBench/pull/344),
  [#365](https://github.com/brightbandtech/ExtremeWeatherBench/pull/365),
  [#334](https://github.com/brightbandtech/ExtremeWeatherBench/pull/334)).
- Tropical cyclone tracking, landfall handling, and case bounds
  (including removal of non-landfalling TCs)
  ([#339](https://github.com/brightbandtech/ExtremeWeatherBench/pull/339),
  [#338](https://github.com/brightbandtech/ExtremeWeatherBench/pull/338),
  [#342](https://github.com/brightbandtech/ExtremeWeatherBench/pull/342),
  [#355](https://github.com/brightbandtech/ExtremeWeatherBench/pull/355),
  [#361](https://github.com/brightbandtech/ExtremeWeatherBench/pull/361),
  [#362](https://github.com/brightbandtech/ExtremeWeatherBench/pull/362)).
- Temperature event bounds, duration logic, and GHCNh filtering
  ([#353](https://github.com/brightbandtech/ExtremeWeatherBench/pull/353),
  [#357](https://github.com/brightbandtech/ExtremeWeatherBench/pull/357),
  [#359](https://github.com/brightbandtech/ExtremeWeatherBench/pull/359),
  [#346](https://github.com/brightbandtech/ExtremeWeatherBench/pull/346)).

### Fixed

- Mixed-layer CAPE/CIN: three CIN computation bugs
  ([#366](https://github.com/brightbandtech/ExtremeWeatherBench/pull/366)).
- `EarlySignal` no longer fails on dask-backed arrays
  ([#347](https://github.com/brightbandtech/ExtremeWeatherBench/pull/347),
  [#340](https://github.com/brightbandtech/ExtremeWeatherBench/pull/340)).
- Numba thread safety when used with dask
  ([#337](https://github.com/brightbandtech/ExtremeWeatherBench/pull/337)).
- Level chunking for `nantrapezoid_pressure_levels`
  ([#343](https://github.com/brightbandtech/ExtremeWeatherBench/pull/343)).
- Flaky pressure test from an unseeded fixture
  ([#367](https://github.com/brightbandtech/ExtremeWeatherBench/pull/367)).

## [1.0.1] - 2026-03-20

### Changed

- Installation instructions in the README
  ([#327](https://github.com/brightbandtech/ExtremeWeatherBench/pull/327)).

## [1.0.0] - 2026-01-26

First stable release, published to PyPI.

### Added

- Public import style `import extremeweatherbench as ewb`
  ([#321](https://github.com/brightbandtech/ExtremeWeatherBench/pull/321),
  [#325](https://github.com/brightbandtech/ExtremeWeatherBench/pull/325)).
- Golden tests for guarding significant version changes
  ([#323](https://github.com/brightbandtech/ExtremeWeatherBench/pull/323)).

### Changed

- PyPI packaging and install documentation
  ([#315](https://github.com/brightbandtech/ExtremeWeatherBench/pull/315)).

## [0.3.0] - 2026-01-26

Automated git comparison with 0.2.0 is unreliable after a
`git-filter-repo` cleanup. Notable changes from
[#322](https://github.com/brightbandtech/ExtremeWeatherBench/pull/322):

### Added

- Forecast wrapper for custom xarray datasets.
- CIRA icechunk store.

### Changed

- Case YAML is a list of dicts (the `cases` key is no longer required).
- Geopotential calculations now convert to geopotential height
  correctly.

### Removed

- `IndividualCaseCollection`.

### Fixed

- `DurationMeanError` and IBTrACS memory issues.

## [0.2.0] - 2025-12-02

### Changed

- Dependency and lockfile updates ahead of the 1.0 packaging work.

## [0.1.0] - 2025-01-14

Initial tagged preview.

[unreleased]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v1.0.2.post1...v1.1.0
[1.0.2.post1]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v1.0.2...v1.0.2.post1
[1.0.2]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/brightbandtech/ExtremeWeatherBench/releases/tag/v0.3.0
[0.2.0]: https://github.com/brightbandtech/ExtremeWeatherBench/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brightbandtech/ExtremeWeatherBench/releases/tag/v0.1.0
