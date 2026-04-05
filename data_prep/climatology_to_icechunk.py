#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "icechunk>=0.1.0",
#     "xarray>=2023.1",
#     "zarr>=3.1.0",
# ]
# ///
"""Write a percentile climatology netCDF into a GCS-backed icechunk store.

The commit message is generated automatically from the dataset metadata
written by generate_percentile_climatology.py (percentile, variable,
coordinate ranges, etc.), so every snapshot in the store is
self-describing.

If the target store does not yet exist it is created; if it already
exists the climatology is appended as a new snapshot on the requested
branch.

Usage:
    # Authenticate first (write access to the EWB bucket):
    gcloud auth application-default login

    # Write / update the store:
    uv run climatology_to_icechunk.py \\
        --input era5_heatwave_climatology.nc \\
        --bucket extremeweatherbench \\
        --prefix climatology-icechunk \\
        --branch main

    # Dry-run: print the commit message without touching the store:
    uv run climatology_to_icechunk.py \\
        --input era5_heatwave_climatology.nc \\
        --dry-run

To read the store back:
    import icechunk, xarray as xr
    storage = icechunk.gcs_storage(
        bucket="extremeweatherbench",
        prefix="climatology-icechunk",
        anonymous=True,
    )
    repo   = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    ds = xr.open_zarr(session.store)
"""

import argparse
import logging
import pathlib

import icechunk
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commit message
# ---------------------------------------------------------------------------


def build_commit_message(ds: xr.Dataset) -> str:
    """Build a descriptive commit message from dataset metadata.

    Pulls from dataset attributes set by generate_percentile_climatology.py
    and from the coordinate ranges present in the dataset.
    """
    attrs = ds.attrs

    # --- required metadata written by the generation script ---
    percentile_label = attrs.get("percentile_label", "pXX")
    percentile = attrs.get("percentile", "unknown")
    source = attrs.get("source", "unknown source")

    # --- variable name: the first non-coordinate data variable ---
    data_vars = list(ds.data_vars)
    variable = data_vars[0] if data_vars else "unknown_variable"

    # --- coordinate ranges ---
    parts = []

    if "dayofyear" in ds.coords:
        doys = ds.coords["dayofyear"].values
        doy_str = (
            f"doy {int(doys.min())}–{int(doys.max())}"
            if len(doys) > 1
            else f"doy {int(doys[0])}"
        )
        parts.append(doy_str)

    if "hour" in ds.coords:
        hours = sorted(int(h) for h in ds.coords["hour"].values)
        parts.append(f"hours {hours}")

    if "latitude" in ds.coords:
        lats = ds.coords["latitude"].values
        parts.append(f"lat {float(lats.min()):.1f}°–{float(lats.max()):.1f}°")

    if "longitude" in ds.coords:
        lons = ds.coords["longitude"].values
        parts.append(f"lon {float(lons.min()):.1f}°–{float(lons.max()):.1f}°")

    coord_summary = ", ".join(parts) if parts else "full domain"

    # --- training period, if recorded ---
    period_str = ""
    if "start_year" in attrs and "end_year" in attrs:
        period_str = f", period {attrs['start_year']}–{attrs['end_year']}"
    elif "training_period" in attrs:
        period_str = f", period {attrs['training_period']}"

    msg = (
        f"Add {percentile_label} ({float(percentile):.0%}) climatology "
        f"of {variable} from {source}{period_str} "
        f"[{coord_summary}]"
    )
    return msg


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


def _open_or_create_repo(
    storage: icechunk.Storage,
    branch: str,
) -> icechunk.Repository:
    """Open an existing icechunk repo, or create one if it doesn't exist."""
    try:
        repo = icechunk.Repository.open(storage)
        logger.info("Opened existing icechunk repository.")
    except Exception:
        logger.info("No existing repository found – creating a new one.")
        repo = icechunk.Repository.create(storage, icechunk.RepositoryConfig.default())

    if branch != "main" and branch not in repo.list_branches():
        snapshot_id = repo.lookup_branch("main")
        repo.create_branch(branch, snapshot_id=snapshot_id)
        logger.info("Created branch '%s'.", branch)

    return repo


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def write_climatology_to_icechunk(
    input_path: pathlib.Path,
    bucket: str,
    prefix: str,
    branch: str,
    application_credentials: str,
    dry_run: bool,
) -> None:
    """Load a climatology netCDF and commit it to a GCS icechunk store."""
    logger.info("Loading climatology from %s", input_path)
    ds = xr.open_dataset(input_path)

    commit_message = build_commit_message(ds)
    logger.info("Commit message: %s", commit_message)

    if dry_run:
        logger.info("Dry run – skipping store write.")
        return

    # Build the GCS-backed storage. Pass an empty string for
    # application_credentials to let the SDK pick up ADC automatically,
    # or provide an explicit path to a service-account JSON file.
    storage = icechunk.gcs_storage(
        bucket=bucket,
        prefix=prefix,
        application_credentials=application_credentials,
    )

    repo = _open_or_create_repo(storage, branch)
    session = repo.writable_session(branch=branch)

    logger.info(
        "Writing dataset to icechunk store (bucket=%s, prefix=%s, branch=%s)…",
        bucket,
        prefix,
        branch,
    )
    ds.to_zarr(session.store, mode="w")

    snapshot_id = session.commit(commit_message)
    logger.info("Committed snapshot %s on branch '%s'.", snapshot_id, branch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a percentile climatology netCDF into a GCS-backed icechunk store."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to the climatology netCDF produced by "
        "generate_percentile_climatology.py.",
    )
    parser.add_argument(
        "--bucket",
        default="extremeweatherbench",
        help="GCS bucket name.",
    )
    parser.add_argument(
        "--prefix",
        default="climatology-icechunk",
        help="Prefix (subdirectory) within the bucket for the store.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Icechunk branch to commit to. Created if it does not exist.",
    )
    parser.add_argument(
        "--application-credentials",
        default="",
        help=(
            "Path to a GCP service-account JSON file. Leave empty to use "
            "Application Default Credentials (gcloud auth application-default "
            "login)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commit message and exit without writing to the store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    write_climatology_to_icechunk(
        input_path=args.input,
        bucket=args.bucket,
        prefix=args.prefix,
        branch=args.branch,
        application_credentials=args.application_credentials,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
