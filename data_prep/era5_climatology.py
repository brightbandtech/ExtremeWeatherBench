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
"""ERA5 percentile climatology toolkit.

Three subcommands, each selectable via the first positional argument:

  generate       Compute a single-percentile climatology zarr from ERA5.
  all-quantiles  Compute multiple quantiles in one pass, keeping the
                 per-year rolling means in RAM across all quantiles.
  combine        Merge per-quantile zarr stores (local and/or GCS) into
                 a single zarr with a leading 'quantile' dimension.

All ERA5 computation is done entirely in RAM due to large memory requirements; this
could be optimized in the future.

Variable names are the ARCO ERA5 variable names, available at 
https://github.com/google-research/arco-era5

Usage examples:
--------------
# One percentile:
uv run data_prep/era5_climatology.py generate \\
    --variable 2m_temperature --percentile 0.85 \\
    --output ~/climatology_zarr/2m_temperature_p85.zarr

# Multiple quantiles in one pass:
uv run data_prep/era5_climatology.py all-quantiles  --variable 2m_temperature  --quantiles 0.10 0.25 0.50 0.75 0.90  --output-dir ~/climatology_zarr --n-workers 16 --gcs-quantiles 0.15 0.85

# Combine local stores (+ optionally pull p15/p85 from GCS):
uv run data_prep/era5_climatology.py combine \\
    --input-dir ~/climatology_zarr \\
    --output ~/climatology_zarr/2m_temperature_combined.zarr \\
    --gcs-quantiles 0.15 0.85
"""

import argparse
import gc
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

ERA5_ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GCS_CLIMATOLOGY_URL = (
    "gs://extremeweatherbench/datasets/"
    "surface_air_temperature_1990_2019_climatology.zarr"
)
SYNOPTIC_HOURS = [0, 6, 12, 18]

# Chunk shape matches the existing EWB climatology store:
# (dayofyear=46, hour=1, latitude=91, longitude=180)
ZARR_CHUNKS = (46, 1, 91, 180)

# Combined store: quantile dim is tiny, keep all quantiles in one chunk
COMBINED_ZARR_CHUNKS = {
    "quantile": -1,
    "dayofyear": 46,
    "hour": 1,
    "latitude": 91,
    "longitude": 180,
}


def _setup_worker_cleanup() -> None:
    """Kill the process group on SIGINT/SIGTERM.

    Only registers signal handlers, not atexit, so zarr's async cleanup
    can finish normally on a clean exit.  On KeyboardInterrupt or SIGTERM
    we SIGKILL the whole group to stop all threads immediately.

    The parent shell is NOT in this group (os.setpgrp creates a new one),
    so it is unaffected.
    """
    os.setpgrp()
    pgid = os.getpgrp()

    def _kill_group(signum: int, frame: object = None) -> None:
        logger.info("Killing all worker processes (pgid=%d)...", pgid)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _kill_group)
    signal.signal(signal.SIGTERM, _kill_group)


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


def _build_weights(half_window_days: int) -> xr.DataArray:
    """Build a normalised triangular weight DataArray."""
    ramp = np.linspace(0, 1, half_window_days + 1)
    weights = np.concatenate([ramp, ramp[::-1][1:]])
    weights /= weights.sum()
    return xr.DataArray(weights, dims=["window"])


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
    era5_window = era5_window.sel(time=era5_window.time.dt.hour.isin(SYNOPTIC_HOURS))

    year_data: xr.DataArray = era5_window[variable].compute()
    logger.info(
        "  %d: downloaded, shape=%s  RAM=%.1f GB",
        year,
        year_data.shape,
        _rss_gb(),
    )

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
    del year_data, hour_arrays
    combined = combined.sel(time=combined.time.dt.year == year)
    # Restore the variable name lost through rolling/dot operations.
    combined = combined.rename(variable)
    logger.info(
        "  %d: rolling done, shape=%s  RAM=%.1f GB",
        year,
        combined.shape,
        _rss_gb(),
    )
    return combined


def compute_year_rolling_means(
    variable: str,
    start_year: int,
    end_year: int,
    half_window_days: int,
    n_workers: int,
) -> list[xr.DataArray]:
    """Compute rolling weighted means for all years.

    Returns a list of per-year DataArrays (~6 GB each, ~180 GB total).
    Uses joblib threading so arrays stay in shared memory — no pickling.
    Recommended n_workers: 8-16 to avoid GCS rate-limiting.
    """
    years = list(range(start_year, end_year + 1))
    logger.info(
        "Per-year rolling means: fetching %d years with %d threads...  RAM=%.1f GB",
        len(years),
        n_workers,
        _rss_gb(),
    )

    year_arrays: list[xr.DataArray] = joblib.Parallel(
        n_jobs=n_workers, backend="threading"
    )(
        joblib.delayed(_process_single_year)(year, variable, half_window_days)
        for year in years
    )
    gc.collect()

    logger.info(
        "Per-year rolling means complete: %d year arrays in memory.  RAM=%.1f GB",
        len(year_arrays),
        _rss_gb(),
    )
    return year_arrays


def _compute_dayofyear_quantile(
    dayofyear: int,
    year_arrays: list[xr.DataArray],
    percentile: float,
) -> xr.DataArray | None:
    """Compute the per-hour quantile for one day-of-year.

    Slices only the dayofyear's timesteps from each year array (~500 MB peak
    per call), then groups by hour and computes the quantile.  Safe to
    call from multiple threads — isel returns new views.
    """
    year_slices = []
    for ya in year_arrays:
        mask = (ya.time.dt.dayofyear == dayofyear).values
        if mask.any():
            year_slices.append(ya.isel(time=mask))

    if not year_slices:
        return None

    dayofyear_data = xr.concat(year_slices, dim="time")
    dayofyear_data = dayofyear_data.sel(
        time=dayofyear_data.time.dt.hour.isin(SYNOPTIC_HOURS)
    )

    return (
        dayofyear_data.groupby("time.hour")
        .quantile(percentile)
        .drop_vars("quantile", errors="ignore")
        .expand_dims({"dayofyear": [dayofyear]})
    )


def compute_dayofyear_quantile_climatology(
    year_arrays: list[xr.DataArray],
    variable: str,
    percentile: float,
    output_path: pathlib.Path,
    start_year: int,
    end_year: int,
    n_workers: int,
) -> None:
    """Compute the quantile for all dayofyears and write to a zarr store.

    Workers share year_arrays via threading (no memory copies).
    Peak memory per worker: ~500 MB.
    """
    logger.info(
        "Per-dayofyear quantile: computing %.0f%% climatology for %s  RAM=%.1f GB",
        percentile * 100,
        variable,
        _rss_gb(),
    )

    dayofyear_results: list[xr.DataArray | None] = joblib.Parallel(
        n_jobs=n_workers, backend="threading"
    )(
        joblib.delayed(_compute_dayofyear_quantile)(dayofyear, year_arrays, percentile)
        for dayofyear in range(1, 367)
    )

    logger.info("Per-dayofyear quantile loop done.  RAM=%.1f GB", _rss_gb())

    valid = [r for r in dayofyear_results if r is not None]
    del dayofyear_results
    logger.info("Combining %d dayofyear slices...  RAM=%.1f GB", len(valid), _rss_gb())
    ds_out = xr.combine_by_coords(valid)
    del valid
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
    del ds_out
    gc.collect()
    logger.info("Zarr store written: %s  RAM=%.1f GB", output_path, _rss_gb())


def _load_local_store(zarr_path: pathlib.Path, variable: str) -> xr.DataArray:
    """Open one local zarr and tag it with its quantile coordinate."""
    ds = xr.open_zarr(zarr_path, chunks=None)
    quantile_val = float(ds.attrs["percentile"])
    da = ds[variable].expand_dims({"quantile": [quantile_val]})
    logger.info(
        "  Loaded local q=%.2f from %s  shape=%s",
        quantile_val,
        zarr_path.name,
        dict(da.sizes),
    )
    return da


def _load_gcs_store(quantile: float, variable: str) -> xr.DataArray:
    """Load one quantile from the existing GCS multi-quantile store."""
    da = xr.open_zarr(
        GCS_CLIMATOLOGY_URL,
        storage_options={"anon": True},
        chunks=None,
    )[variable].sel(quantile=quantile)
    da = da.expand_dims({"quantile": [quantile]})
    logger.info("  Loaded GCS q=%.2f  shape=%s", quantile, dict(da.sizes))
    return da


def combine_stores(
    input_dir: pathlib.Path,
    output: pathlib.Path | str,
    variable: str,
    gcs_quantiles: list[float],
    application_credentials: str = "",
) -> None:
    """Concatenate per-quantile stores into one zarr with a quantile dim.

    ``output`` may be a local path or a GCS URL (``gs://bucket/prefix``).
    When writing to GCS, ADC is used unless ``application_credentials``
    points to a service-account JSON file.
    """
    slices: list[xr.DataArray] = []

    local_stores = sorted(input_dir.glob(f"{variable}_p*_climatology.zarr"))
    if not local_stores:
        logger.warning("No local zarr stores found in %s", input_dir)
    for store in local_stores:
        slices.append(_load_local_store(store, variable))

    for q in gcs_quantiles:
        slices.append(_load_gcs_store(q, variable))

    if not slices:
        raise RuntimeError("No data found — check --input-dir and --gcs-quantiles.")

    logger.info(
        "Concatenating %d quantile slices along 'quantile' dim...",
        len(slices),
    )
    combined_da = xr.concat(slices, dim="quantile").sortby("quantile")
    ds_out = combined_da.to_dataset(name=variable)
    ds_out.attrs.update(
        {
            "source": "ERA5 ARCO",
            "quantiles": sorted(combined_da.coords["quantile"].values.tolist()),
        }
    )
    logger.info("Combined shape: %s", dict(ds_out.sizes))

    n_quantiles = ds_out.sizes["quantile"]
    encoding = {
        variable: {
            "chunks": [
                n_quantiles,  # keep all quantiles in one chunk
                COMBINED_ZARR_CHUNKS["dayofyear"],
                COMBINED_ZARR_CHUNKS["hour"],
                COMBINED_ZARR_CHUNKS["latitude"],
                COMBINED_ZARR_CHUNKS["longitude"],
            ]
        }
    }

    output_str = str(output)
    if output_str.startswith("gs://"):
        token = application_credentials if application_credentials else "google_default"
        storage_options = {"token": token}
        logger.info("Writing combined zarr to GCS: %s ...", output_str)
        ds_out.to_zarr(
            output_str,
            mode="w",
            zarr_format=2,
            encoding=encoding,
            storage_options=storage_options,
        )
    else:
        local_out = pathlib.Path(output_str)
        local_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing combined zarr to %s...", local_out)
        ds_out.to_zarr(local_out, mode="w", zarr_format=2, encoding=encoding)

    logger.info("Done: %s", output_str)


def _zarr_store_path(
    output_dir: pathlib.Path, variable: str, quantile: float
) -> pathlib.Path:
    label = f"p{int(round(quantile * 100)):02d}"
    return output_dir / f"{variable.replace(' ', '_')}_{label}_climatology.zarr"


def _zarr_complete(zarr_path: pathlib.Path) -> bool:
    """Return True if the zarr store exists and appears complete."""
    return (zarr_path / "zarr.json").exists() or (zarr_path / ".zmetadata").exists()


def cmd_generate(args: argparse.Namespace) -> None:
    logger.info("Variable   : %s", args.variable)
    logger.info("Percentile : %.0f%%", args.percentile * 100)
    logger.info("Period     : %d-%d", args.start_year, args.end_year)
    logger.info("Window     : +/-%d days", args.half_window_days)
    logger.info("Workers    : %d", args.n_workers)
    logger.info("Output     : %s", args.output)

    year_arrays = compute_year_rolling_means(
        variable=args.variable,
        start_year=args.start_year,
        end_year=args.end_year,
        half_window_days=args.half_window_days,
        n_workers=args.n_workers,
    )
    compute_dayofyear_quantile_climatology(
        year_arrays=year_arrays,
        variable=args.variable,
        percentile=args.percentile,
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        n_workers=args.n_workers,
    )


def cmd_all_quantiles(args: argparse.Namespace) -> None:
    logger.info("Variable   : %s", args.variable)
    logger.info("Quantiles  : %s", args.quantiles)
    logger.info("Period     : %d–%d", args.start_year, args.end_year)
    logger.info("Window     : ±%d days", args.half_window_days)
    logger.info("Workers    : %d", args.n_workers)
    logger.info("Output dir : %s", args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    remaining = [
        q
        for q in args.quantiles
        if not _zarr_complete(_zarr_store_path(args.output_dir, args.variable, q))
    ]
    if not remaining:
        logger.info("All zarr stores already complete. Nothing to do.")
        return

    logger.info(
        "Per-year rolling means: computing for %d-%d"
        " (~180 GB of year arrays stored in RAM)...",
        args.start_year,
        args.end_year,
    )
    year_arrays = compute_year_rolling_means(
        variable=args.variable,
        start_year=args.start_year,
        end_year=args.end_year,
        half_window_days=args.half_window_days,
        n_workers=args.n_workers,
    )

    for i, quantile in enumerate(remaining):
        label = f"p{int(round(quantile * 100)):02d}"
        zarr_out = _zarr_store_path(args.output_dir, args.variable, quantile)
        logger.info("---- Quantile %s (%d/%d) ----", label, i + 1, len(remaining))
        compute_dayofyear_quantile_climatology(
            year_arrays=year_arrays,
            variable=args.variable,
            percentile=quantile,
            output_path=zarr_out,
            start_year=args.start_year,
            end_year=args.end_year,
            n_workers=args.n_workers,
        )
        logger.info("Quantile %s complete: %s", label, zarr_out)

    logger.info("All %d quantiles complete.", len(remaining))


def cmd_combine(args: argparse.Namespace) -> None:
    logger.info("Input dir     : %s", args.input_dir)
    logger.info("Output        : %s", args.output)
    logger.info("Variable      : %s", args.variable)
    if args.gcs_quantiles:
        logger.info("GCS quantiles : %s", args.gcs_quantiles)
    combine_stores(
        input_dir=args.input_dir,
        output=args.output,
        variable=args.variable,
        gcs_quantiles=args.gcs_quantiles,
        application_credentials=getattr(args, "application_credentials", ""),
    )


_DEFAULT_OUTPUT_DIR = pathlib.Path("~/climatology_zarr")
_DEFAULT_COMBINED = (
    "gs://extremeweatherbench/datasets/"
    "surface_air_temperature_1990_2019_climatology.zarr"
)


def _add_era5_args(p: argparse.ArgumentParser) -> None:
    """Attach shared ERA5/period arguments to a subparser."""
    p.add_argument(
        "--variable",
        default="2m_temperature",
        help="ERA5 variable name.",
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=1990,
        help="First year of the training period (inclusive).",
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=2019,
        help="Last year of the training period (inclusive).",
    )
    p.add_argument(
        "--half-window-days",
        type=int,
        default=10,
        help="Half-width of the triangular rolling window in days.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help=(
            "Parallel threads. 8-16 avoids GCS rate-limiting during "
            "per-year download; more threads speeds up per-dayofyear quantile."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="era5_climatology.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser(
        "generate",
        help="Compute a single-percentile climatology zarr from ERA5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_era5_args(p_gen)
    p_gen.add_argument(
        "--percentile",
        type=float,
        required=True,
        help="Percentile as a fraction in (0, 1).",
    )
    p_gen.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output zarr store path.",
    )

    p_all = sub.add_parser(
        "all-quantiles",
        help="Compute multiple quantiles in one pass (year arrays kept in RAM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_era5_args(p_all)
    p_all.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 0.90],
        help="Quantiles to compute, each in (0, 1).",
    )
    p_all.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where per-quantile zarr stores are written.",
    )

    p_comb = sub.add_parser(
        "combine",
        help="Merge per-quantile zarr stores into one zarr with a quantile dim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_comb.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory containing per-quantile zarr stores.",
    )
    p_comb.add_argument(
        "--output",
        default=_DEFAULT_COMBINED,
        help=(
            "Output path for the combined zarr store. "
            "May be a local path or a GCS URL (gs://bucket/prefix)."
        ),
    )
    p_comb.add_argument(
        "--variable",
        default="2m_temperature",
        help="Variable name inside each source zarr store.",
    )
    p_comb.add_argument(
        "--gcs-quantiles",
        nargs="*",
        type=float,
        default=[],
        metavar="Q",
        help=(
            "Additional quantiles to pull from the existing GCS climatology "
            "store (e.g. 0.15 0.85).  Only 0.15 and 0.85 are available."
        ),
    )
    p_comb.add_argument(
        "--application-credentials",
        default="",
        help=(
            "Path to a GCP service-account JSON file for GCS output. "
            "Leave empty to use Application Default Credentials."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command in ("generate", "all-quantiles"):
        _setup_worker_cleanup()

    if args.command == "generate":
        if not 0.0 < args.percentile < 1.0:
            raise ValueError(f"--percentile must be in (0, 1), got {args.percentile}")
        cmd_generate(args)

    elif args.command == "all-quantiles":
        bad = [q for q in args.quantiles if not 0.0 < q < 1.0]
        if bad:
            raise ValueError(f"Quantiles must be in (0, 1); got {bad}")
        cmd_all_quantiles(args)

    elif args.command == "combine":
        cmd_combine(args)


if __name__ == "__main__":
    main()
