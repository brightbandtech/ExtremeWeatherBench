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
"""Sentinel: generate climatology zarr stores for multiple quantiles.

Imports run_phase1 and run_phase2 from generate_percentile_climatology
so that the 30-year rolling-weighted DataArray (~180 GB) is computed
only once and reused for every quantile.  No intermediate files are
written to disk.

Storage at any point:
  ~180 GB  combined in-memory DataArray (all years, Phase 1 output)
  ~12 GB   one zarr store per quantile accumulated over time
  ─────────────────────────────────────────────────────────────────
  Peak: 180 + 5 × 12 = 240 GB  <<  1.35 TB available RAM

Zarr stores that already exist and look complete are skipped so the
script is safe to re-run after an interruption.

Usage:
    uv run data_prep/run_climatology_sentinel.py \\
        --variable 2m_temperature \\
        --quantiles 0.10 0.25 0.50 0.75 0.90 \\
        --output-dir /home/taylor/data/climatology_zarr \\
        --n-workers 16
"""

import argparse
import atexit
import logging
import os
import pathlib
import signal
import sys

# Make data_prep importable regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from generate_percentile_climatology import run_phase1, run_phase2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-group cleanup – kills all loky workers on any exit
# ---------------------------------------------------------------------------


def _setup_worker_cleanup() -> None:
    """Ensure joblib/loky worker processes are killed when this script exits.

    Calling os.setpgrp() makes this process the leader of a new process
    group.  All loky workers inherit that group, so sending SIGKILL to
    the group terminates every worker regardless of how the script exits
    (KeyboardInterrupt, exception, or normal return).

    The parent shell is NOT in this group, so it is unaffected.
    """
    os.setpgrp()
    pgid = os.getpgrp()

    def _kill_group(signum: int | None = None, frame: object = None) -> None:
        logger.info("Killing all worker processes (pgid=%d)...", pgid)
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
# Helpers
# ---------------------------------------------------------------------------


def _zarr_path(output_dir: pathlib.Path, variable: str, quantile: float) -> pathlib.Path:
    label = f"p{int(round(quantile * 100)):02d}"
    safe_var = variable.replace(" ", "_")
    return output_dir / f"{safe_var}_{label}_climatology.zarr"


def _zarr_complete(zarr_path: pathlib.Path) -> bool:
    """Return True if the zarr store exists and appears complete."""
    return (zarr_path / "zarr.json").exists() or (zarr_path / ".zmetadata").exists()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run_sentinel(
    variable: str,
    quantiles: list[float],
    start_year: int,
    end_year: int,
    half_window_days: int,
    output_dir: pathlib.Path,
    n_workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all zarr stores already exist before doing any work.
    remaining = [
        q for q in quantiles
        if not _zarr_complete(_zarr_path(output_dir, variable, q))
    ]
    if not remaining:
        logger.info("All zarr stores already complete. Nothing to do.")
        return

    logger.info(
        "Phase 1: computing rolling weighted means for %d-%d "
        "(stores ~180 GB of year arrays in RAM)...",
        start_year,
        end_year,
    )
    year_arrays = run_phase1(
        variable=variable,
        start_year=start_year,
        end_year=end_year,
        half_window_days=half_window_days,
        n_workers=n_workers,
    )

    for i, quantile in enumerate(remaining):
        label = f"p{int(round(quantile * 100)):02d}"
        zarr_out = _zarr_path(output_dir, variable, quantile)
        logger.info(
            "---- Quantile %s (%d/%d) ----", label, i + 1, len(remaining)
        )
        run_phase2(
            year_arrays=year_arrays,
            variable=variable,
            percentile=quantile,
            output_path=zarr_out,
            start_year=start_year,
            end_year=end_year,
            n_workers=n_workers,
        )
        logger.info("Quantile %s complete: %s", label, zarr_out)

    logger.info("All %d quantiles complete.", len(remaining))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate climatology zarr stores for multiple quantiles, "
            "keeping the 30-year DataArray in RAM across all quantiles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variable",
        default="2m_temperature",
        help="ERA5 variable name.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 0.90],
        help="Quantiles to compute, each in (0, 1).",
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
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/taylor/data/climatology_zarr"),
        help="Directory where per-quantile zarr stores are written.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help=(
            "Parallel threads for both phases. 8–16 avoids GCS rate-limiting "
            "during Phase 1; more threads speeds up Phase 2."
        ),
    )
    return parser.parse_args()


def main() -> None:
    _setup_worker_cleanup()
    args = parse_args()

    bad = [q for q in args.quantiles if not 0.0 < q < 1.0]
    if bad:
        raise ValueError(f"Quantiles must be in (0, 1); got {bad}")

    logger.info("Variable   : %s", args.variable)
    logger.info("Quantiles  : %s", args.quantiles)
    logger.info("Period     : %d–%d", args.start_year, args.end_year)
    logger.info("Window     : ±%d days", args.half_window_days)
    logger.info("Workers    : %d", args.n_workers)
    logger.info("Output dir : %s", args.output_dir)

    run_sentinel(
        variable=args.variable,
        quantiles=args.quantiles,
        start_year=args.start_year,
        end_year=args.end_year,
        half_window_days=args.half_window_days,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
