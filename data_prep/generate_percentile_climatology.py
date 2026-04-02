#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gcsfs>=2024.12.0",
#     "joblib>=1.3",
#     "numpy>=1.24",
#     "xarray>=2023.1",
#     "zarr>=3.1.0",
#     "flox>=0.9",
# ]
# ///
"""Generate a percentile climatology zarr store from ERA5 ARCO data.

All computation is done in RAM — no intermediate files are written.

Phase 1 – Rolling weighted means:
  For each year in the training period, fetch synoptic-hourly ERA5 data
  from GCS and compute a triangular rolling window weighted mean.  Years
  are processed in parallel via joblib threads so the large DataArrays
  remain in shared memory without pickling overhead.  run_phase1()
  returns a list of per-year DataArrays (~6 GB each).

Phase 2 – Percentile climatology:
  For each day-of-year, each thread slices just its doy from every year
  array (~500 MB peak per worker), groups by hour, and computes the
  quantile.  This avoids ever concatenating all 30 years into one
  ~180 GB DataArray.  Results are combined and written to a zarr store.

When importing from run_climatology_sentinel.py, call run_phase1() once
to get the year_arrays list, then call run_phase2() for each quantile.

Standalone usage:
    uv run generate_percentile_climatology.py \\
        --variable 2m_temperature \\
        --percentile 0.85 \\
        --start-year 1990 \\
        --end-year 2019 \\
        --output /home/taylor/data/climatology_zarr/2m_temperature_p85.zarr
"""

import argparse
import atexit
import logging
import os
import pathlib
import signal
import sys

import joblib
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ERA5_ARCO_URL = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
SYNOPTIC_HOURS = [0, 6, 12, 18]

# Chunk shape matches the existing climatology zarr store used by
# ewb.get_climatology: (dayofyear=46, hour=1, latitude=91, longitude=180).
ZARR_CHUNKS = (46, 1, 91, 180)


# ---------------------------------------------------------------------------
# Process-group cleanup
# ---------------------------------------------------------------------------


def _setup_worker_cleanup() -> None:
    """Kill all loky workers on any exit — see run_climatology_sentinel.py."""
    os.setpgrp()
    pgid = os.getpgrp()

    def _kill_group(signum: int | None = None, frame: object = None) -> None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if signum is not None:
            sys.exit(128 + signum)

    atexit.register(_kill_group)
    signal.signal(signal.SIGINT, _kill_group)
    signal.signal(signal.SIGTERM, _kill_group)


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------


def _rss_gb() -> float:
    """Return current process resident memory in GB via /proc/self/status."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1_000_000  # kB → GB
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_weights(half_window_days: int) -> xr.DataArray:
    """Build a normalised triangular weight DataArray."""
    ramp = np.linspace(0, 1, half_window_days + 1)
    weights = np.concatenate([ramp, ramp[::-1][1:]])
    weights /= weights.sum()
    return xr.DataArray(weights, dims=["window"])


# ---------------------------------------------------------------------------
# Phase 1 – rolling weighted means
# ---------------------------------------------------------------------------


def _process_single_year(
    year: int,
    variable: str,
    half_window_days: int,
) -> xr.DataArray:
    """Fetch and compute the rolling weighted mean for one year.

    Opens its own ERA5 connection so this is safe to call from multiple
    threads concurrently.
    """
    era5 = xr.open_zarr(
        ERA5_ARCO_URL,
        chunks=None,
        storage_options=dict(token="anon"),
    )

    weight_da = _build_weights(half_window_days)
    n_weights = len(weight_da)

    # Buffer on both sides so the rolling window is valid at year edges.
    start_date = f"{year - 1}-12-{31 - half_window_days}"
    end_date = f"{year + 1}-01-{half_window_days}"

    era5_window = era5.sel(time=slice(start_date, end_date))
    era5_window = era5_window.sel(
        time=era5_window.time.dt.hour.isin(SYNOPTIC_HOURS)
    )

    year_data: xr.DataArray = era5_window[variable].compute()
    logger.info("  %d: downloaded, shape=%s  RAM=%.1f GB", year, year_data.shape, _rss_gb())

    hour_arrays = []
    for hour in SYNOPTIC_HOURS:
        hour_data = year_data.sel(time=year_data.time.dt.hour == hour)
        rolled = hour_data.rolling(time=n_weights, center=True)
        weighted = rolled.construct(
            "window",
            sliding_window_view_kwargs={"automatic_rechunk": True},
        ).dot(weight_da)
        weighted = weighted.compute()
        hour_arrays.append(weighted)

    combined = xr.concat(hour_arrays, dim="time").sortby("time")
    combined = combined.sel(time=combined.time.dt.year == year)
    # Restore the variable name lost through rolling/dot operations.
    combined = combined.rename(variable)
    logger.info("  %d: rolling done, shape=%s  RAM=%.1f GB", year, combined.shape, _rss_gb())
    return combined


def run_phase1(
    variable: str,
    start_year: int,
    end_year: int,
    half_window_days: int,
    n_workers: int,
) -> list[xr.DataArray]:
    """Compute rolling weighted means for all years.

    Returns a list of per-year DataArrays (~6 GB each, ~180 GB total).
    Uses joblib threads so arrays stay in shared memory — no pickling.
    Recommended n_workers: 8-16 to avoid GCS rate-limiting.
    """
    years = list(range(start_year, end_year + 1))
    logger.info(
        "Phase 1: fetching %d years with %d threads...  RAM=%.1f GB",
        len(years), n_workers, _rss_gb(),
    )

    year_arrays: list[xr.DataArray] = joblib.Parallel(
        n_jobs=n_workers, backend="threading"
    )(
        joblib.delayed(_process_single_year)(year, variable, half_window_days)
        for year in years
    )

    logger.info(
        "Phase 1 complete: %d year arrays in memory.  RAM=%.1f GB",
        len(year_arrays), _rss_gb(),
    )
    return year_arrays


# ---------------------------------------------------------------------------
# Phase 2 – percentile climatology
# ---------------------------------------------------------------------------


def _compute_doy_quantile(
    doy: int,
    year_arrays: list[xr.DataArray],
    percentile: float,
) -> xr.DataArray | None:
    """Compute the per-hour quantile for one day-of-year.

    Slices only the doy's timesteps from each year array (~500 MB peak
    per call), then groups by hour and computes the quantile.  Safe to
    call from multiple threads — isel returns new views.
    """
    year_slices = []
    for ya in year_arrays:
        mask = (ya.time.dt.dayofyear == doy).values
        if mask.any():
            year_slices.append(ya.isel(time=mask))

    if not year_slices:
        return None

    doy_data = xr.concat(year_slices, dim="time")
    doy_data = doy_data.sel(time=doy_data.time.dt.hour.isin(SYNOPTIC_HOURS))

    return (
        doy_data.groupby("time.hour")
        .quantile(percentile)
        .drop_vars("quantile", errors="ignore")
        .expand_dims({"dayofyear": [doy]})
    )


def run_phase2(
    year_arrays: list[xr.DataArray],
    variable: str,
    percentile: float,
    output_path: pathlib.Path,
    start_year: int,
    end_year: int,
    n_workers: int,
) -> None:
    """Compute the quantile for all doys and write to a zarr store.

    Workers share year_arrays via threading (no memory copies).
    Peak memory per worker: ~500 MB.
    """
    logger.info(
        "Phase 2: computing %.0f%% climatology for %s  RAM=%.1f GB",
        percentile * 100, variable, _rss_gb(),
    )

    doy_results: list[xr.DataArray | None] = joblib.Parallel(
        n_jobs=n_workers, backend="threading"
    )(
        joblib.delayed(_compute_doy_quantile)(doy, year_arrays, percentile)
        for doy in range(1, 367)
    )

    logger.info(
        "Phase 2 doy loop done.  RAM=%.1f GB", _rss_gb()
    )

    valid = [r for r in doy_results if r is not None]
    logger.info("Combining %d doy slices...  RAM=%.1f GB", len(valid), _rss_gb())
    ds_out = xr.combine_by_coords(valid)
    logger.info("Combined shape: %s  RAM=%.1f GB", dict(ds_out.sizes), _rss_gb())
    ds_out.attrs["percentile"] = percentile
    ds_out.attrs["percentile_label"] = f"p{int(round(percentile * 100)):02d}"
    ds_out.attrs["source"] = "ERA5 ARCO"
    ds_out.attrs["start_year"] = start_year
    ds_out.attrs["end_year"] = end_year

    encoding = {variable: {"chunks": list(ZARR_CHUNKS)}}

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing zarr store to %s...  RAM=%.1f GB", output_path, _rss_gb())
    ds_out.to_zarr(output_path, mode="w", zarr_format=2, encoding=encoding)
    logger.info("Zarr store written: %s  RAM=%.1f GB", output_path, _rss_gb())


# ---------------------------------------------------------------------------
# CLI (standalone use)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a percentile climatology zarr store from ERA5. "
            "All computation is done in RAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variable",
        required=True,
        help="ERA5 variable name, e.g. '2m_temperature'.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        required=True,
        help="Percentile as a fraction in (0, 1).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1990,
        help="First year of the training period (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2019,
        help="Last year of the training period (inclusive).",
    )
    parser.add_argument(
        "--half-window-days",
        type=int,
        default=10,
        help="Half-width of the triangular rolling window in days.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output zarr store path.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help=(
            "Number of parallel threads. 8-16 recommended to avoid "
            "GCS rate-limiting during Phase 1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    _setup_worker_cleanup()
    args = parse_args()

    if not 0.0 < args.percentile < 1.0:
        raise ValueError(
            f"--percentile must be in (0, 1), got {args.percentile}"
        )

    logger.info("Variable   : %s", args.variable)
    logger.info("Percentile : %.0f%%", args.percentile * 100)
    logger.info("Period     : %d-%d", args.start_year, args.end_year)
    logger.info("Window     : +/-%d days", args.half_window_days)
    logger.info("Workers    : %d", args.n_workers)
    logger.info("Output     : %s", args.output)

    year_arrays = run_phase1(
        variable=args.variable,
        start_year=args.start_year,
        end_year=args.end_year,
        half_window_days=args.half_window_days,
        n_workers=args.n_workers,
    )

    run_phase2(
        year_arrays=year_arrays,
        variable=args.variable,
        percentile=args.percentile,
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
