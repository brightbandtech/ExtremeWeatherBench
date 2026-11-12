#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "arraylake>=1.8.0",
#     "dask>=2024.1",
#     "icechunk>=2.3.1",
#     "numpy>=1.26",
#     "xarray>=2025.1",
#     "zarr>=3.1.0",
# ]
# ///
"""Add the IFS HRES subset to the Arraylake ``brightband/mlwp-forecast-archive`` repo.

Reads the public GCS Icechunk store built by ``ewb_hres_to_icechunk.py``
(``gs://brightband-public-mlwp-forecast-archive/ifs-hres.hres.icechunk``) and
writes the group ``ifs-hres/hres`` with the same five variables, encoding and
attributes as the existing ``graphcast/hres`` group:

- ``2m_temperature``, ``mean_sea_level_pressure``,
  ``10m_u_component_of_wind``, ``10m_v_component_of_wind``
- ``geopotential_thickness_300-500`` = (z300 - z500) / 9.80665, in m
- one 2-D field per chunk, (1, 1, 721, 1440), zstd level 0, NaN fill
- a 2-D ``valid_time(init_time, lead_time)``

The time axes stay native to HRES: 1462 inits (2023-01-01 to 2024-12-31,
every 12 h) and 41 lead times (0-240 h). The sibling groups have 5840 inits
every 6 h from 2021 and 61 lead times; HRES is not padded to match.

Writes go to an Arraylake branch (created from ``main`` if missing), one
commit per batch of init times, so an interrupted run resumes by skipping
batches already committed. Review the branch, then move ``main`` to it.

Usage:
    # Dry run against the scratch store: template + two batches of two inits
    uv run data_prep/ifs_hres_to_arraylake_mlwp.py \\
        --source-bucket brightband-scratch \\
        --source-prefix darothen/ifs-hres-dryrun.icechunk \\
        --branch ifs-hres-test --batch-size 2 --max-batches 2

    # Validate the branch (schema vs graphcast/hres, values vs the GCS store)
    uv run data_prep/ifs_hres_to_arraylake_mlwp.py --branch ifs-hres-test --validate

    # Full run
    uv run data_prep/ifs_hres_to_arraylake_mlwp.py --branch ifs-hres-build

Requires an Arraylake login with write access to
``brightband/mlwp-forecast-archive`` (``arraylake auth login``) and GCS read
access to the source bucket (ADC, or ``--gcloud-token``).
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
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_REPO = "brightband/mlwp-forecast-archive"
TARGET_GROUP = "ifs-hres/hres"
REFERENCE_GROUP = "graphcast/hres"
SOURCE_BUCKET = "brightband-public-mlwp-forecast-archive"
SOURCE_PREFIX = "ifs-hres.hres.icechunk"

SURFACE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
THICKNESS = "geopotential_thickness_300-500"
DATA_VARS = [*SURFACE_VARS, THICKNESS]
G0 = 9.80665  # m s-2, the standard gravity used for the sibling groups

# Copied from the graphcast/hres group in the target repo. Its stale
# "coordinates: has_data valid_time" is dropped; xarray writes "coordinates".
THICKNESS_ATTRS = {
    "long_name": "Geopotential thickness between 300 and 500 hPa",
    "standard_name": "geopotential_thickness",
    "units": "m",
    "short_name": "zht",
    "param_id": 129,
}

# Group attrs: the source store's ECMWF attribution, licence and disclaimer
# (required; see ewb_hres_to_icechunk.py), plus this group's own modifications.
ATTRIBUTION_KEYS = (
    "attribution",
    "license",
    "license_url",
    "disclaimer",
    "processed_by",
)
MODIFICATIONS = (
    "Subset of gs://{bucket}/{prefix}: 2m_temperature, mean_sea_level_pressure "
    "and 10 m winds unmodified; geopotential_thickness_300-500 derived as "
    "(z300 - z500) / 9.80665 m. See that store's `modifications` for prior changes."
)

ZSTD = [{"name": "zstd", "configuration": {"level": 0, "checksum": False}}]
TIME_UNITS = "hours since 2023-01-01 00:00:00"
TEMPLATE_MSG = f"Initialize {TARGET_GROUP} template"
BATCH_RE = re.compile(re.escape(TARGET_GROUP) + r" init_time\[(\d+):(\d+)\]")


def open_source(bucket: str, prefix: str, token: str | None) -> tuple[xr.Dataset, str]:
    """Open the GCS IFS HRES store lazily, one chunk per 2-D field.

    Returns:
        The dataset and the snapshot ID it was read at.
    """
    storage = (
        icechunk.gcs_storage(bucket=bucket, prefix=prefix, bearer_token=token)
        if token
        else icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    )
    session = icechunk.Repository.open(storage).readonly_session("main")
    # Data variables: one dask chunk per stored 2-D field. valid_time is
    # loaded eagerly, since chunking it per field would create one task per
    # (init, lead) pair, each re-reading the same stored chunk.
    data_chunks = {"init_time": 1, "lead_time": 1, "level": 1}
    ds = xr.open_zarr(session.store, consolidated=False, chunks=None)
    valid_time = ds["valid_time"].variable.load()
    ds = ds.chunk(data_chunks).assign_coords(valid_time=valid_time)
    return ds, session.snapshot_id


def subset(src: xr.Dataset) -> xr.Dataset:
    """Select the five MLWP-archive variables, deriving 300-500 hPa thickness.

    Args:
        src: The full IFS HRES dataset (or an init_time slice of it).

    Returns:
        A dataset with the four surface variables and the thickness, float32.
    """
    z = src["geopotential"]
    thickness = (z.sel(level=300.0) - z.sel(level=500.0)) / G0
    ds = src[SURFACE_VARS].copy()
    ds[THICKNESS] = thickness.drop_vars("level", errors="ignore").astype("float32")
    ds[THICKNESS].attrs = dict(THICKNESS_ATTRS)
    for name in SURFACE_VARS:
        ds[name].attrs = dict(src[name].attrs)
    for var in ds.variables.values():
        var.encoding = {}
    return ds


def group_attrs(src: xr.Dataset, source_url: str) -> dict:
    """Group attributes: the source's ECMWF attribution plus this subset's changes.

    Raises:
        KeyError: If the source store lacks any of the attribution attributes.
    """
    missing = [k for k in ATTRIBUTION_KEYS if k not in src.attrs]
    if missing:
        raise KeyError(f"source store is missing attribution attrs {missing}")
    bucket, prefix = source_url.removeprefix("gs://").split("/", 1)
    attrs = {k: src.attrs[k] for k in ATTRIBUTION_KEYS}
    attrs["modifications"] = MODIFICATIONS.format(bucket=bucket, prefix=prefix)
    return attrs


def build_template(src: xr.Dataset, source_url: str) -> xr.Dataset:
    """Build the target dataset with placeholder (never computed) data.

    Coordinates come from the source in full; each data variable is a single
    ``dask.array.empty`` chunk so the graph stays tiny. Only metadata and
    coordinates are written, via ``to_zarr(compute=False)``.
    """
    import dask.array as da

    coords = {
        name: src[name].variable.load()
        for name in ("init_time", "lead_time", "valid_time", "latitude", "longitude")
    }
    for var in coords.values():
        var.encoding = {}
    shape = tuple(
        src.sizes[d] for d in ("init_time", "lead_time", "latitude", "longitude")
    )
    data_vars = {}
    for name in DATA_VARS:
        attrs = THICKNESS_ATTRS if name == THICKNESS else src[name].attrs
        data_vars[name] = xr.Variable(
            ("init_time", "lead_time", "latitude", "longitude"),
            da.empty(shape, chunks=shape, dtype="float32"),
            attrs=dict(attrs),
        )
    return xr.Dataset(data_vars, coords=coords, attrs=group_attrs(src, source_url))


def encoding_for(ds: xr.Dataset) -> dict:
    """Build the zarr encoding that matches the sibling MLWP groups."""
    enc: dict = {
        name: {
            "chunks": (1, 1, 721, 1440),
            "dtype": "float32",
            "compressors": ZSTD,
            "_FillValue": np.float32(np.nan),
        }
        for name in DATA_VARS
    }
    for name in ("init_time", "valid_time"):
        enc[name] = {
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "dtype": "int64",
            "compressors": ZSTD,
        }
    enc["lead_time"] = {"units": "hours", "dtype": "int64", "compressors": ZSTD}
    for name in ("latitude", "longitude"):
        enc[name] = {"dtype": "float64", "compressors": ZSTD}
    return enc


def ensure_branch(repo: icechunk.Repository, branch: str) -> None:
    """Create ``branch`` from the tip of ``main`` if it doesn't exist yet."""
    if branch in repo.list_branches():
        logger.info("Using existing branch %s", branch)
        return
    tip = repo.lookup_branch("main")
    repo.create_branch(branch, snapshot_id=tip)
    logger.info("Created branch %s from main at %s", branch, tip)


def has_template(repo: icechunk.Repository, branch: str) -> bool:
    """Whether the template for the group has been committed on ``branch``."""
    return any(TEMPLATE_MSG in s.message for s in repo.ancestry(branch=branch))


def completed_batches(repo: icechunk.Repository, branch: str) -> set[tuple[int, int]]:
    """Find init_time index ranges already committed on ``branch``."""
    done = set()
    for snap in repo.ancestry(branch=branch):
        m = BATCH_RE.search(snap.message)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def write_template(
    repo: icechunk.Repository, branch: str, src: xr.Dataset, src_id: str
) -> None:
    """Write the group's array metadata and coordinates, with no data chunks."""
    ds = build_template(src, src_id.split()[0])
    session = repo.writable_session(branch)
    ds.to_zarr(
        session.store,
        group=TARGET_GROUP,
        mode="w-",
        compute=False,
        encoding=encoding_for(ds),
        consolidated=False,
    )
    session.commit(f"{TEMPLATE_MSG} from {src_id}")
    logger.info(
        "Wrote %s template (%d inits x %d leads)",
        TARGET_GROUP,
        ds.sizes["init_time"],
        ds.sizes["lead_time"],
    )


def write_batches(
    repo: icechunk.Repository,
    branch: str,
    src: xr.Dataset,
    src_id: str,
    batch_size: int,
    max_batches: int | None,
) -> None:
    """Write data in init_time batches, one commit per batch, skipping finished ones."""
    n = src.sizes["init_time"]
    bounds = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    done = completed_batches(repo, branch)
    todo = [b for b in bounds if b not in done]
    if max_batches is not None:
        todo = todo[:max_batches]
    logger.info(
        "%d batches total, %d already done, writing %d",
        len(bounds),
        len(done),
        len(todo),
    )
    for k, (i0, i1) in enumerate(todo, 1):
        sub = subset(src.isel(init_time=slice(i0, i1)))
        sub = sub.drop_vars(list(sub.coords))  # coords are already in the template
        t0, t1 = (str(x)[:13] for x in src["init_time"].values[[i0, i1 - 1]])
        start = time.monotonic()
        session = repo.writable_session(branch)
        icechunk.xarray.to_icechunk(
            sub, session, group=TARGET_GROUP, region={"init_time": slice(i0, i1)}
        )
        session.commit(
            f"Wrote {TARGET_GROUP} init_time[{i0}:{i1}] ({t0} - {t1}) from {src_id}"
        )
        secs = time.monotonic() - start
        gb = sub.nbytes / 1e9
        logger.info(
            "Batch %d/%d init_time[%d:%d] (%s - %s): %.1f GB uncompressed in %.0f s (%.1f s/init)",
            k,
            len(todo),
            i0,
            i1,
            t0,
            t1,
            gb,
            secs,
            secs / (i1 - i0),
        )


def validate(
    repo: icechunk.Repository, branch: str, src: xr.Dataset, samples: int, seed: int
) -> bool:
    """Check the group's schema against graphcast/hres and its values against the source.

    Args:
        repo: The target Arraylake repository.
        branch: The branch holding the new group.
        src: The GCS IFS HRES dataset the group was built from.
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

    store = repo.readonly_session(branch).store
    tgt = xr.open_zarr(store, group=TARGET_GROUP, consolidated=False, chunks=None)
    ref = xr.open_zarr(store, group=REFERENCE_GROUP, consolidated=False, chunks=None)

    def strip(attrs: dict) -> dict:
        return {
            k: v for k, v in attrs.items() if k not in ("coordinates", "_FillValue")
        }

    check(
        set(tgt.data_vars) == set(ref.data_vars), f"variables {sorted(tgt.data_vars)}"
    )
    check(
        all(tgt.attrs.get(k) == src.attrs.get(k) for k in ATTRIBUTION_KEYS)
        and "modifications" in tgt.attrs,
        "ECMWF attribution, licence and disclaimer carried from source",
    )
    for name in DATA_VARS:
        t, r = tgt[name], ref[name]
        check(t.dims == r.dims, f"{name} dims {t.dims}")
        check(t.dtype == r.dtype, f"{name} dtype {t.dtype}")
        check(t.encoding["chunks"] == r.encoding["chunks"], f"{name} chunks")
        check(
            t.encoding.get("compressors") == r.encoding.get("compressors"),
            f"{name} compressors",
        )
        check(strip(t.attrs) == strip(r.attrs), f"{name} attrs match {REFERENCE_GROUP}")
    for name in ("latitude", "longitude"):
        check(np.array_equal(tgt[name].values, ref[name].values), f"same {name}s")
    check(
        tgt["lead_time"].dtype == ref["lead_time"].dtype, "lead_time is timedelta64[ns]"
    )
    check(tgt["valid_time"].dims == ("init_time", "lead_time"), "valid_time is 2-D")
    check(
        np.array_equal(tgt["valid_time"].values, src["valid_time"].values),
        "valid_time equals source",
    )
    check(
        np.array_equal(tgt["init_time"].values, src["init_time"].values)
        and np.array_equal(tgt["lead_time"].values, src["lead_time"].values),
        f"native axes ({tgt.sizes['init_time']} inits x {tgt.sizes['lead_time']} leads)",
    )
    main_groups = set(
        xr.open_zarr(
            repo.readonly_session("main").store, group=REFERENCE_GROUP
        ).data_vars
    )
    check(main_groups == set(ref.data_vars), f"{REFERENCE_GROUP} unchanged vs main")

    written = sorted(
        i for i0, i1 in completed_batches(repo, branch) for i in range(i0, i1)
    )
    check(len(written) > 0, f"{len(written)} init times written")
    if not written:
        return ok
    rng = np.random.default_rng(seed)
    for name in DATA_VARS:
        for _ in range(samples):
            it = int(rng.choice(written))
            lt = int(rng.integers(src.sizes["lead_time"]))
            sel = {"init_time": it, "lead_time": lt}
            label = f"init={str(src.init_time.values[it])[:13]} lead={lt}"
            got = tgt[name].isel(sel).values
            want = subset(src.isel(init_time=[it], lead_time=[lt]))[name].values[0, 0]
            check(
                np.array_equal(got, want), f"{name} bit-identical to source ({label})"
            )
            check(not np.isnan(got).any(), f"{name} has no NaN ({label})")
            if name == THICKNESS:
                lo, hi = float(got.min()), float(got.max())
                check(
                    3000 < lo and hi < 4500,
                    f"{name} in [{lo:.0f}, {hi:.0f}] m ({label})",
                )
    return ok


def main() -> None:
    """Parse arguments and run the subset write and/or validation."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-bucket", default=SOURCE_BUCKET)
    p.add_argument("--source-prefix", default=SOURCE_PREFIX)
    p.add_argument("--branch", required=True, help="Arraylake branch to write to")
    p.add_argument("--batch-size", type=int, default=16, help="Init times per commit")
    p.add_argument(
        "--max-batches", type=int, default=None, help="Stop after this many batches"
    )
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="Dask threads")
    p.add_argument(
        "--gcloud-token",
        action="store_true",
        help="Use `gcloud auth print-access-token` to read GCS, not ADC",
    )
    p.add_argument(
        "--validate", action="store_true", help="Validate instead of writing"
    )
    p.add_argument(
        "--samples", type=int, default=3, help="Fields per variable to compare"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.branch == "main" and not args.validate:
        p.error("write to a branch, validate it, then move main to it")

    token = (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode()
        .strip()
        if args.gcloud_token
        else None
    )
    src, src_snapshot = open_source(args.source_bucket, args.source_prefix, token)
    src_id = f"gs://{args.source_bucket}/{args.source_prefix} snapshot {src_snapshot}"
    logger.info("Source %s", src_id)
    repo = arraylake.Client().get_repo(TARGET_REPO)
    logger.info(
        "Target %s (icechunk %s, arraylake %s)",
        TARGET_REPO,
        icechunk.__version__,
        arraylake.__version__,
    )

    if args.validate:
        passed = validate(repo, args.branch, src, args.samples, args.seed)
        logger.info("Validation %s", "PASSED" if passed else "FAILED")
        raise SystemExit(0 if passed else 1)

    ensure_branch(repo, args.branch)
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        if not has_template(repo, args.branch):
            write_template(repo, args.branch, src, src_id)
        write_batches(repo, args.branch, src, src_id, args.batch_size, args.max_batches)


if __name__ == "__main__":
    main()
