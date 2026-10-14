"""Lightweight quality control for GHCNh station observations.

The GHCNh parquet used as the default GHCN target contains values that are not
real observations: missing-value sentinels such as -999 (e.g. one Hungarian
station reports -999 C for every hour of 2020), physically impossible values
such as 928 C air temperature or a relative humidity of 13.8 million percent,
and isolated spikes. Some stations also report values that are wrong for long
stretches while staying in range and smooth (see the excluded stations below).
Scoring forecasts against these inflates errors badly.

The checks here run on a polars LazyFrame in the raw GHCNh units (C, hPa, m/s,
mm, %, degrees), before any unit conversion. A value that fails a check is set
to null; the rest of its row is kept, since one bad variable does not make the
other variables at that station and time bad. The exception is a known-bad
station, whose rows are dropped entirely. The checks are applied in order:

0. Excluded stations: rows from stations whose name matches an excluded
   pattern are dropped.
1. Sentinel values: values equal to a known missing-value code are nulled.
2. Range: values outside a physically plausible range for the variable are
   nulled.
3. Spike: a value that jumps away from both its previous and next observation
   at the same station by more than a threshold, in opposite directions, and
   with both neighbors close in time, is nulled.

Public functions:
    exclude_stations: Drop rows from stations whose name matches a pattern.
    mask_sentinel_values: Null values equal to a missing-value sentinel.
    mask_out_of_range: Null values outside per-variable plausible ranges.
    mask_spikes: Null isolated spikes in each station's time series.
    apply_ghcnh_quality_control: Apply all of the checks.
"""

import logging
from collections.abc import Iterable, Mapping

import polars as pl

logger = logging.getLogger(__name__)

# Stations dropped entirely, matched as substrings of the GHCNh station name.
# "MOVING ICE STATION" marks the automatic weather stations on the Greenland ice
# sheet (39 stations in ghcnh_all_2020_2024). Several report temperatures far too
# warm for their site for weeks at a time, which no sentinel, range, or spike
# check catches: e.g. NEM (GLU00000037, about 2450 m) averages +4.3 C over
# Jan-Mar 2023, and scored forecasts at 26-28 K RMSE on EWB marginal heat case
# 70 while ERA5 gave about 4-6 K. They are all excluded until a check that
# compares stations against climatology or their neighbors is in place.
GHCNH_EXCLUDED_STATION_NAME_PATTERNS: tuple[str, ...] = ("MOVING ICE STATION",)

# Missing-value codes seen in GHCNh, applied to every checked variable. 999.9 is
# not included because it is a valid pressure in hPa; codes that are only
# invalid for some variables (e.g. 999 for wind direction, -99.9 for
# temperature) are caught by the range check instead.
GHCNH_SENTINEL_VALUES: tuple[float, ...] = (-9999.0, -999.9, -999.0, 9999.0)

# Physically plausible ranges (inclusive) in raw GHCNh units. Bounds sit just
# outside observed records so that only impossible values are removed.
GHCNH_VALID_RANGES: dict[str, tuple[float, float]] = {
    # C; world records -89.2 C and 56.7 C
    "surface_air_temperature": (-90.0, 60.0),
    # C; highest recorded dew point is about 35 C
    "surface_dew_point": (-90.0, 40.0),
    # hPa; station pressure, down to stations above 5 km elevation
    "surface_air_pressure": (300.0, 1100.0),
    # hPa; records about 870 hPa and 1084 hPa
    "air_pressure_at_mean_sea_level": (850.0, 1090.0),
    # m/s; strongest measured gust is about 113 m/s
    "surface_wind_speed": (0.0, 115.0),
    # degrees; 999 marks variable or missing direction
    "surface_wind_from_direction": (0.0, 360.0),
    # %; small tolerance above 100 for sensor and rounding error
    "surface_relative_humidity": (0.0, 105.0),
    # mm; world record 1-hour rainfall is about 305 mm
    "accumulated_1_hour_precipitation": (0.0, 400.0),
}

# Minimum jump (raw units) away from both neighbors for a value to count as a
# spike. Only smooth, continuous variables are checked; wind, humidity, and
# precipitation legitimately change abruptly.
GHCNH_SPIKE_THRESHOLDS: dict[str, float] = {
    "surface_air_temperature": 10.0,  # C
    "surface_dew_point": 10.0,  # C
    "surface_air_pressure": 10.0,  # hPa
    "air_pressure_at_mean_sea_level": 10.0,  # hPa
}

# Neighbors further apart in time than this are not used for the spike check.
GHCNH_SPIKE_MAX_GAP_HOURS: float = 3.0


def _present(lf: pl.LazyFrame, columns: Iterable[str]) -> list[str]:
    """Return the columns of ``columns`` that exist in ``lf``, in order."""
    names = set(lf.collect_schema().names())
    return [c for c in columns if c in names]


def exclude_stations(
    lf: pl.LazyFrame,
    name_patterns: Iterable[str] = GHCNH_EXCLUDED_STATION_NAME_PATTERNS,
    name_column: str = "name",
) -> pl.LazyFrame:
    """Drop rows from stations whose name contains any of ``name_patterns``.

    Args:
        lf: Station observations.
        name_patterns: Substrings to match (literally) against station names.
        name_column: Column holding the station name.

    Returns:
        ``lf`` without the matching stations' rows. If ``name_column`` is
        missing or there are no patterns, ``lf`` is returned unchanged.
    """
    name_patterns = list(name_patterns)
    if not name_patterns or not _present(lf, [name_column]):
        return lf
    matches = pl.any_horizontal(
        pl.col(name_column).str.contains(p, literal=True).fill_null(False)
        for p in name_patterns
    )
    return lf.filter(~matches)


def mask_sentinel_values(
    lf: pl.LazyFrame,
    columns: Iterable[str],
    sentinels: Iterable[float] = GHCNH_SENTINEL_VALUES,
) -> pl.LazyFrame:
    """Null values equal to a missing-value sentinel.

    Args:
        lf: Station observations.
        columns: Columns to check; columns not in ``lf`` are ignored.
        sentinels: Missing-value codes to null.

    Returns:
        ``lf`` with sentinel values in ``columns`` replaced by null.
    """
    sentinels = list(sentinels)
    return lf.with_columns(
        pl.when(pl.col(c).is_in(sentinels)).then(None).otherwise(pl.col(c)).alias(c)
        for c in _present(lf, columns)
    )


def mask_out_of_range(
    lf: pl.LazyFrame,
    valid_ranges: Mapping[str, tuple[float, float]] = GHCNH_VALID_RANGES,
) -> pl.LazyFrame:
    """Null values outside a per-variable plausible range.

    Args:
        lf: Station observations.
        valid_ranges: Mapping of column name to inclusive (min, max); columns
            not in ``lf`` are ignored.

    Returns:
        ``lf`` with out-of-range values replaced by null.
    """
    return lf.with_columns(
        pl.when(pl.col(c).is_between(*valid_ranges[c]))
        .then(pl.col(c))
        .otherwise(None)
        .alias(c)
        for c in _present(lf, valid_ranges)
    )


def mask_spikes(
    lf: pl.LazyFrame,
    thresholds: Mapping[str, float] = GHCNH_SPIKE_THRESHOLDS,
    max_gap_hours: float = GHCNH_SPIKE_MAX_GAP_HOURS,
    station_column: str = "station",
    time_column: str = "valid_time",
) -> pl.LazyFrame:
    """Null isolated spikes in each station's time series.

    A value is a spike when, compared with the previous and next non-null
    observation at the same station, it differs from both by more than the
    variable's threshold, in opposite directions (up then down, or down then
    up), and both neighbors are within ``max_gap_hours``. A sustained change,
    such as a front passing, moves in one direction and is kept.

    Args:
        lf: Station observations.
        thresholds: Mapping of column name to the minimum jump that counts as
            a spike; columns not in ``lf`` are ignored.
        max_gap_hours: Neighbors further away in time than this are not used.
        station_column: Column identifying the station.
        time_column: Observation time column.

    Returns:
        ``lf`` with spikes replaced by null and rows sorted by station and
        time. If ``station_column`` or ``time_column`` is missing, ``lf`` is
        returned unchanged.
    """
    columns = _present(lf, thresholds)
    if not columns or len(_present(lf, [station_column, time_column])) < 2:
        return lf
    lf = lf.sort(station_column, time_column)
    max_gap = pl.duration(seconds=int(max_gap_hours * 3600))
    masked = []
    for c in columns:
        # neighbors are taken among this variable's non-null values only, so a
        # missing reading does not hide a spike next to it
        valid = pl.col(c).is_not_null()
        value = pl.col(c)
        time = pl.when(valid).then(pl.col(time_column))
        prev_value = value.forward_fill().shift(1).over(station_column)
        next_value = value.backward_fill().shift(-1).over(station_column)
        prev_time = time.forward_fill().shift(1).over(station_column)
        next_time = time.backward_fill().shift(-1).over(station_column)
        rise = value - prev_value
        fall = next_value - value
        is_spike = (
            valid
            & (rise.abs() > thresholds[c])
            & (fall.abs() > thresholds[c])
            & (rise.sign() != fall.sign())
            & ((pl.col(time_column) - prev_time) <= max_gap)
            & ((next_time - pl.col(time_column)) <= max_gap)
        )
        masked.append(pl.when(is_spike).then(None).otherwise(value).alias(c))
    return lf.with_columns(masked)


def apply_ghcnh_quality_control(
    lf: pl.LazyFrame,
    excluded_station_name_patterns: Iterable[str] = (
        GHCNH_EXCLUDED_STATION_NAME_PATTERNS
    ),
    sentinels: Iterable[float] = GHCNH_SENTINEL_VALUES,
    valid_ranges: Mapping[str, tuple[float, float]] = GHCNH_VALID_RANGES,
    spike_thresholds: Mapping[str, float] = GHCNH_SPIKE_THRESHOLDS,
    max_gap_hours: float = GHCNH_SPIKE_MAX_GAP_HOURS,
    station_column: str = "station",
    time_column: str = "valid_time",
) -> pl.LazyFrame:
    """Apply the station, sentinel, range, and spike checks to GHCNh data.

    Values must be in raw GHCNh units (C, hPa, m/s, mm, %, degrees). Rows
    from excluded stations are dropped; otherwise failing values are set to
    null and rows are kept.

    Args:
        lf: Station observations.
        excluded_station_name_patterns: Drop stations whose name contains any
            of these substrings.
        sentinels: Missing-value codes to null in every checked column.
        valid_ranges: Mapping of column name to inclusive (min, max).
        spike_thresholds: Mapping of column name to spike threshold.
        max_gap_hours: Maximum neighbor distance for the spike check.
        station_column: Column identifying the station.
        time_column: Observation time column.

    Returns:
        The quality-controlled observations, sorted by station and time when
        the spike check runs.
    """
    checked = list(dict.fromkeys([*valid_ranges, *spike_thresholds]))
    lf = exclude_stations(lf, excluded_station_name_patterns)
    lf = mask_sentinel_values(lf, checked, sentinels)
    lf = mask_out_of_range(lf, valid_ranges)
    lf = mask_spikes(lf, spike_thresholds, max_gap_hours, station_column, time_column)
    return lf
