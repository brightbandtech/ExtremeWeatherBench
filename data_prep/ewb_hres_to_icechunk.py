#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "arraylake>=0.20",
#     "dask>=2024.1",
#     "icechunk>=1.0",
#     "numpy>=1.26",
#     "pandas>=2.0",
#     "xarray>=2025.1",
#     "zarr>=3.1.0",
# ]
# ///
"""Copy the EWB-HRES archive from Arraylake into a standalone GCS Icechunk repo.

The source is the ECMWF IFS HRES archive Brightband processed for the
ExtremeWeatherBench paper (``brightband/ecmwf`` → ``forecast-archive/ewb-hres``).
It is read as-is, never modified, and rewritten so its layout matches the
GraphCast store in ``gs://brightband-public-mlwp-forecast-archive``:

- ERA5/WeatherBench2-style variable names, and ``isobaricInhPa`` → ``level``
- latitude ascending (-90 → 90), level ascending (50 → 1000 hPa, float64)
- ``lead_time`` as timedelta64, plus a 2-D ``valid_time(init_time, lead_time)``
- one 2-D field per chunk: (1, 1, 1, 721, 1440) and (1, 1, 721, 1440), zstd
- the GraphCast store's descriptive attributes on every variable and coordinate

The native time axes are kept: 1462 inits (2023-01-01 to 2024-12-31, every
12 h) and 41 lead times (0-240 h, every 6 h).

The repo is written in batches of init times, one Icechunk commit per batch,
so an interrupted run resumes by skipping batches already committed.

Usage:
    # Dry run: build the template and write one batch of two inits to scratch
    uv run data_prep/ewb_hres_to_icechunk.py \\
        --bucket brightband-scratch --prefix darothen/ifs-hres-dryrun.icechunk \\
        --batch-size 2 --max-batches 1

    # Validate what has been written (schema vs GraphCast, values vs source)
    uv run data_prep/ewb_hres_to_icechunk.py \\
        --bucket brightband-scratch --prefix darothen/ifs-hres-dryrun.icechunk \\
        --validate

    # Full run
    uv run data_prep/ewb_hres_to_icechunk.py \\
        --bucket brightband-public-mlwp-forecast-archive --prefix ifs-hres.hres.icechunk

Requires an Arraylake login with read access to ``brightband/ecmwf``
(``arraylake auth login``) and GCS write access to the target bucket (ADC, or
``--gcloud-token`` to use ``gcloud auth print-access-token`` instead).
"""

import argparse
import logging
import os
import re
import subprocess
import time

import arraylake
import dask
import icechunk
import icechunk.xarray
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_REPO = "brightband/ecmwf"
SOURCE_GROUP = "forecast-archive/ewb-hres"
REFERENCE_BUCKET = "brightband-public-mlwp-forecast-archive"
REFERENCE_PREFIX = "graphcast.hres.icechunk"

RENAME = {
    "isobaricInhPa": "level",
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

# Descriptive attributes copied verbatim from the GraphCast reference store.
# xarray-managed keys ("coordinates", "_FillValue") are left to the encoder.
VAR_ATTRS = {
    "temperature": {
        "long_name": "Temperature",
        "standard_name": "air_temperature",
        "units": "K",
        "short_name": "t",
        "param_id": 130,
    },
    "specific_humidity": {
        "long_name": "Specific humidity",
        "standard_name": "specific_humidity",
        "units": "kg kg-1",
        "short_name": "q",
        "param_id": 133,
    },
    "u_component_of_wind": {
        "long_name": "Eastward wind component",
        "standard_name": "eastward_wind",
        "units": "m s-1",
        "short_name": "u",
        "param_id": 131,
    },
    "v_component_of_wind": {
        "long_name": "Northward wind component",
        "standard_name": "northward_wind",
        "units": "m s-1",
        "short_name": "v",
        "param_id": 132,
    },
    "geopotential": {
        "long_name": "Geopotential",
        "standard_name": "geopotential",
        "units": "m2 s-2",
        "short_name": "z",
        "param_id": 129,
    },
    "2m_temperature": {
        "long_name": "Air temperature at 2-meters",
        "standard_name": "air_temperature",
        "units": "K",
        "short_name": "2t",
        "param_id": 167,
    },
    "mean_sea_level_pressure": {
        "long_name": "Mean sea level pressure",
        "standard_name": "air_pressure_at_mean_sea_level",
        "units": "Pa",
        "short_name": "msl",
        "param_id": 151,
    },
    "10m_u_component_of_wind": {
        "long_name": "Eastward wind component at 10-meters",
        "standard_name": "eastward_wind",
        "units": "m s-1",
        "short_name": "10u",
        "param_id": 165,
    },
    "10m_v_component_of_wind": {
        "long_name": "Northward wind component at 10-meters",
        "standard_name": "northward_wind",
        "units": "m s-1",
        "short_name": "10v",
        "param_id": 166,
    },
}

COORD_ATTRS = {
    "level": {
        "long_name": "Pressure level",
        "standard_name": "air_pressure",
        "units": "hPa",
        "short_name": "p",
        "param_id": -1,
        "positive": "down",
        "stored_direction": "increasing",
    },
    "latitude": {
        "long_name": "Latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
        "short_name": "lat",
        "param_id": -1,
        "stored_direction": "increasing",
    },
    "longitude": {
        "long_name": "Longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
        "short_name": "lon",
        "param_id": -1,
        "stored_direction": "increasing",
    },
    "valid_time": {
        "long_name": "Valid time for forecast",
        "standard_name": "time",
        "short_name": "valid",
        "param_id": -1,
    },
}

ROOT_ATTRS = {
    "GRIB_edition": 1,
    "GRIB_centre": "ecmf",
    "GRIB_centreDescription": "European Centre for Medium-Range Weather Forecasts",
    "GRIB_subCentre": 0,
    "Conventions": "CF-1.7",
    "institution": "European Centre for Medium-Range Weather Forecasts",
    "model": "IFS_HRES",
    "fcst_step_hours": 6,
    "steps": 40,
    "source": "ECMWF IFS HRES operational forecasts (0.25 deg)",
    # TODO(#438): replace with the ECMWF attribution wording cleared for redistribution.
    "attribution": "Source: ECMWF IFS HRES operational forecasts.",
    "processed_by": "Brightband for ExtremeWeatherBench",
}

TIME_UNITS = "hours since 2023-01-01 00:00:00"
ZSTD = [{"name": "zstd", "configuration": {"level": 0, "checksum": False}}]
BATCH_RE = re.compile(r"init_time\[(\d+):(\d+)\]")


def open_source(snapshot_id: str | None = None) -> tuple[xr.Dataset, str]:
    """Open the ewb-hres group on Arraylake, read-only.

    Args:
        snapshot_id: Pin a specific source snapshot; defaults to the tip of main.

    Returns:
        The lazily-loaded source dataset and the snapshot ID it was read from.
    """
    repo = arraylake.Client().get_repo(SOURCE_REPO)
    session = (
        repo.readonly_session(snapshot_id=snapshot_id)
        if snapshot_id
        else repo.readonly_session("main")
    )
    ds = xr.open_zarr(
        session.store,
        group=SOURCE_GROUP,
        consolidated=False,
        chunks={"init_time": 1, "lead_time": 1},
    )
    return ds, session.snapshot_id


TARGET_CHUNKS = {
    "init_time": 1,
    "lead_time": 1,
    "level": 1,
    "latitude": -1,
    "longitude": -1,
}


def coerce(src: xr.Dataset) -> xr.Dataset:
    """Coerce (a slice of) the ewb-hres layout into the GraphCast reference layout.

    Only call this on a small ``init_time`` slice of the source: the dask graph
    it builds covers every chunk of its input.

    Args:
        src: A slice of the dataset returned by ``open_source``.

    Returns:
        A lazily-evaluated dataset with GraphCast names, ordering, attributes and
        chunking, plus a 2-D ``valid_time`` coordinate.

    Raises:
        ValueError: If the source's ``valid_time`` disagrees with init + lead.
    """
    ds = src.rename(RENAME)
    lead = pd.to_timedelta(ds["lead_time"].values, unit="h")
    ds = ds.assign_coords(
        lead_time=("lead_time", lead.values.astype("timedelta64[ns]"))
    )

    valid = (ds["init_time"] + ds["lead_time"]).transpose("init_time", "lead_time")
    if not np.array_equal(valid.values, src["valid_time"].values):
        raise ValueError("Source valid_time does not equal init_time + lead_time")
    ds = ds.drop_vars("valid_time").assign_coords(valid_time=valid)

    ds = ds.sortby("latitude").sortby("level")
    ds = ds.assign_coords(level=ds["level"].astype("float64"))
    ds = ds.transpose("init_time", "lead_time", "level", "latitude", "longitude")

    for name, attrs in VAR_ATTRS.items():
        ds[name].attrs = dict(attrs)
    for name, attrs in COORD_ATTRS.items():
        ds[name].attrs = dict(attrs)
    for name in ("init_time", "lead_time"):
        ds[name].attrs = {}
    ds.attrs = dict(ROOT_ATTRS)

    for var in ds.variables.values():
        var.encoding = {}
    return ds.chunk(TARGET_CHUNKS)


def build_template(src: xr.Dataset) -> xr.Dataset:
    """Build a full-size template whose data are single-chunk placeholders.

    The coordinates (including the full 2-D ``valid_time``) are real values, but
    each data variable is one lazy ``dask.array.empty`` chunk, so writing it with
    ``compute=False`` only creates array metadata. Using the real lazy dataset
    instead would build a dask graph over every chunk in the store and exhaust
    memory.

    Args:
        src: The full dataset returned by ``open_source``.

    Returns:
        A template dataset with the target's full shape, coordinates and attributes.
    """
    import dask.array as da

    proto = coerce(src.isel(init_time=slice(0, 1)))
    init = src["init_time"].values
    lead = proto["lead_time"].values
    valid = init[:, None] + lead[None, :]
    if not np.array_equal(valid, src["valid_time"].values):
        raise ValueError("Source valid_time does not equal init_time + lead_time")

    coords = {
        "init_time": ("init_time", init),
        "lead_time": ("lead_time", lead),
        "valid_time": (
            ("init_time", "lead_time"),
            valid,
            dict(COORD_ATTRS["valid_time"]),
        ),
    }
    for name in ("level", "latitude", "longitude"):
        coords[name] = proto[name].variable
    n = init.size
    data_vars = {}
    for name in VAR_ATTRS:
        var = proto[name]
        shape = (n,) + var.shape[1:]
        data_vars[name] = xr.Variable(
            var.dims, da.empty(shape, chunks=shape, dtype="float32"), attrs=var.attrs
        )
    template = xr.Dataset(data_vars, coords=coords, attrs=dict(ROOT_ATTRS))
    for var in template.variables.values():
        var.encoding = {}
    return template


def encoding_for(ds: xr.Dataset) -> dict:
    """Build the zarr encoding that matches the GraphCast reference store.

    Args:
        ds: The transformed dataset.

    Returns:
        A per-variable encoding mapping for ``to_zarr``.
    """
    enc: dict = {}
    for name in VAR_ATTRS:
        chunks = (1, 1, 1, 721, 1440) if "level" in ds[name].dims else (1, 1, 721, 1440)
        enc[name] = {
            "chunks": chunks,
            "dtype": "float32",
            "compressors": ZSTD,
            "_FillValue": np.float32(np.nan),
        }
    for name in ("init_time", "valid_time"):
        enc[name] = {
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "dtype": "int64",
            "compressors": ZSTD,
        }
    enc["lead_time"] = {"units": "hours", "dtype": "int64", "compressors": ZSTD}
    for name in ("level", "latitude", "longitude"):
        enc[name] = {"dtype": "float64", "compressors": ZSTD}
    return enc


def gcs_storage(bucket: str, prefix: str, token: str | None) -> icechunk.Storage:
    """Build GCS storage, using a bearer token if given, else ADC."""
    if token:
        return icechunk.gcs_storage(bucket=bucket, prefix=prefix, bearer_token=token)
    return icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)


def open_or_create(storage: icechunk.Storage) -> tuple[icechunk.Repository, bool]:
    """Open the target repo, creating it if it doesn't exist.

    Returns:
        The repository and whether it was newly created.
    """
    if icechunk.Repository.exists(storage):
        return icechunk.Repository.open(storage), False
    return icechunk.Repository.create(storage), True


def has_template(repo: icechunk.Repository) -> bool:
    """Whether the template (array metadata + coordinates) has been committed."""
    return any(
        "Initialize ifs-hres template" in snap.message
        for snap in repo.ancestry(branch="main")
    )


def completed_batches(repo: icechunk.Repository) -> set[tuple[int, int]]:
    """Find init_time index ranges already committed, from commit messages."""
    done = set()
    for snap in repo.ancestry(branch="main"):
        m = BATCH_RE.search(snap.message)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def write_template(
    repo: icechunk.Repository, src: xr.Dataset, src_snapshot: str
) -> None:
    """Write array metadata and all coordinates, with no data chunks."""
    ds = build_template(src)
    session = repo.writable_session("main")
    ds.to_zarr(
        session.store,
        mode="w",
        compute=False,
        encoding=encoding_for(ds),
        consolidated=False,
    )
    session.commit(
        f"Initialize ifs-hres template from {SOURCE_REPO}/{SOURCE_GROUP} snapshot {src_snapshot}"
    )
    logger.info(
        "Wrote template (%d inits x %d leads)",
        ds.sizes["init_time"],
        ds.sizes["lead_time"],
    )


def write_batches(
    repo: icechunk.Repository,
    src: xr.Dataset,
    src_snapshot: str,
    batch_size: int,
    max_batches: int | None,
) -> None:
    """Write data in init_time batches, one commit per batch, skipping finished ones.

    Each batch slices the source *before* coercing it, so the dask graph only
    ever covers ``batch_size`` init times.
    """
    n = src.sizes["init_time"]
    bounds = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    done = completed_batches(repo)
    todo = [b for b in bounds if b not in done]
    if max_batches is not None:
        todo = todo[:max_batches]
    logger.info(
        "%d batches total, %d already done, writing %d",
        len(bounds),
        len(done),
        len(todo),
    )

    data_vars = list(VAR_ATTRS)
    for k, (i0, i1) in enumerate(todo, 1):
        sub = coerce(src.isel(init_time=slice(i0, i1)))[data_vars]
        sub = sub.drop_vars(list(sub.coords))  # coords are already in the template
        t0, t1 = (str(x)[:13] for x in src["init_time"].values[[i0, i1 - 1]])
        start = time.monotonic()
        session = repo.writable_session("main")
        icechunk.xarray.to_icechunk(sub, session, region={"init_time": slice(i0, i1)})
        session.commit(
            f"Wrote init_time[{i0}:{i1}] ({t0} - {t1}) from {SOURCE_REPO}/{SOURCE_GROUP} snapshot {src_snapshot}"
        )
        secs = time.monotonic() - start
        gb = sub.nbytes / 1e9
        logger.info(
            "Batch %d/%d init_time[%d:%d] (%s - %s): %.1f GB uncompressed in %.0f s (%.2f GB/s, %.1f s/init)",
            k,
            len(todo),
            i0,
            i1,
            t0,
            t1,
            gb,
            secs,
            gb / secs,
            secs / (i1 - i0),
        )


def validate(
    repo: icechunk.Repository,
    src: xr.Dataset,
    ref_token: str | None,
    samples: int,
    seed: int,
) -> bool:
    """Check the written store's schema against GraphCast and its values against the source.

    Args:
        repo: The target repository.
        src: The (untransformed) source dataset.
        ref_token: Bearer token for the GraphCast reference store, or None for ADC.
        samples: Number of random 2-D fields to compare per variable.
        seed: Random seed for choosing fields.

    Returns:
        True if every check passed.
    """
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        logger.info("%s %s", "PASS" if cond else "FAIL", msg)
        ok &= bool(cond)

    tgt = xr.open_zarr(
        repo.readonly_session("main").store, consolidated=False, chunks=None
    )
    creds = (
        icechunk.gcs_credentials(bearer_token=ref_token)
        if ref_token
        else icechunk.gcs_from_env_credentials()
    )
    ref_repo = icechunk.Repository.open(
        gcs_storage(REFERENCE_BUCKET, REFERENCE_PREFIX, ref_token),
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {f"gs://{REFERENCE_BUCKET}/graphcast/hres/": creds}
        ),
    )
    ref = xr.open_zarr(
        ref_repo.readonly_session("main").store,
        consolidated=False,
        chunks=None,
        decode_times=False,
    ).drop_vars("valid_time")
    ref = xr.decode_cf(ref)

    # Schema vs GraphCast.
    for name in VAR_ATTRS:
        check(tgt[name].dims == ref[name].dims, f"{name} dims {tgt[name].dims}")
        check(tgt[name].dtype == ref[name].dtype, f"{name} dtype {tgt[name].dtype}")
        check(
            tgt[name].encoding["chunks"] == ref[name].encoding["chunks"],
            f"{name} chunks",
        )
        check(tgt[name].attrs == ref[name].attrs, f"{name} attrs match GraphCast")
    for name in ("level", "latitude", "longitude"):
        check(tgt[name].dtype == ref[name].dtype, f"{name} dtype {tgt[name].dtype}")
        check(tgt[name].attrs == ref[name].attrs, f"{name} attrs match GraphCast")
        check(bool((np.diff(tgt[name].values) > 0).all()), f"{name} increasing")
    check(
        np.array_equal(tgt["level"].values, ref["level"].values),
        "same 13 levels as GraphCast",
    )
    check(
        np.array_equal(tgt["latitude"].values, ref["latitude"].values),
        "same latitudes as GraphCast",
    )
    check(
        np.array_equal(tgt["longitude"].values, ref["longitude"].values),
        "same longitudes as GraphCast",
    )
    check(
        tgt["lead_time"].dtype == ref["lead_time"].dtype, "lead_time is timedelta64[ns]"
    )
    check(tgt["valid_time"].dims == ("init_time", "lead_time"), "valid_time is 2-D")
    check(
        np.array_equal(tgt["valid_time"].values, src["valid_time"].values),
        "valid_time equals source valid_time",
    )
    check(
        tgt.sizes["init_time"] == src.sizes["init_time"],
        "init_time axis is native (1462)",
    )
    check(
        tgt.sizes["lead_time"] == src.sizes["lead_time"],
        "lead_time axis is native (41)",
    )

    # Values vs source, bit-for-bit, on random fields from committed batches.
    written = sorted(i for i0, i1 in completed_batches(repo) for i in range(i0, i1))
    check(len(written) > 0, f"{len(written)} init times written")
    if not written:
        return ok
    rng = np.random.default_rng(seed)
    inverse = {v: k for k, v in RENAME.items()}
    for name in VAR_ATTRS:
        for _ in range(samples):
            it = int(rng.choice(written))
            lt = int(rng.integers(src.sizes["lead_time"]))
            sel = {"init_time": it, "lead_time": lt}
            label = f"init={str(src.init_time.values[it])[:13]} lead={lt}"
            if "level" in tgt[name].dims:
                lev = float(rng.choice(tgt["level"].values))
                got = tgt[name].isel(sel).sel(level=lev).values
                want = (
                    src[inverse[name]].isel(sel).sel(isobaricInhPa=lev).values[::-1, :]
                )
                label += f" level={lev:g}"
            else:
                got = tgt[name].isel(sel).values
                want = src[inverse[name]].isel(sel).values[::-1, :]
            check(
                np.array_equal(got, want, equal_nan=True),
                f"{name} bit-identical to source ({label})",
            )
            check(not np.isnan(got).any(), f"{name} has no NaN ({label})")
    return ok


def main() -> None:
    """Parse arguments and run the copy and/or validation."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument(
        "--source-snapshot",
        default=None,
        help="Pin a source snapshot ID (default: tip of main)",
    )
    p.add_argument("--batch-size", type=int, default=4, help="Init times per commit")
    p.add_argument(
        "--max-batches", type=int, default=None, help="Stop after this many batches"
    )
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Dask threads")
    p.add_argument(
        "--gcloud-token",
        action="store_true",
        help="Use `gcloud auth print-access-token`, not ADC",
    )
    p.add_argument(
        "--validate", action="store_true", help="Validate the store instead of writing"
    )
    p.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Fields per variable to compare when validating",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    token = (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode()
        .strip()
        if args.gcloud_token
        else None
    )
    src, src_snapshot = open_source(args.source_snapshot)
    logger.info("Source %s/%s at snapshot %s", SOURCE_REPO, SOURCE_GROUP, src_snapshot)
    repo, created = open_or_create(gcs_storage(args.bucket, args.prefix, token))
    logger.info(
        "%s gs://%s/%s", "Created" if created else "Opened", args.bucket, args.prefix
    )

    if args.validate:
        passed = validate(repo, src, token, args.samples, args.seed)
        logger.info("Validation %s", "PASSED" if passed else "FAILED")
        raise SystemExit(0 if passed else 1)

    with dask.config.set(scheduler="threads", num_workers=args.workers):
        if created or not has_template(repo):
            write_template(repo, src, src_snapshot)
        write_batches(repo, src, src_snapshot, args.batch_size, args.max_batches)


if __name__ == "__main__":
    main()
