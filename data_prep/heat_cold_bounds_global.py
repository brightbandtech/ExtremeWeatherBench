"""Detect heat waves and cold snaps globally from ERA5 reanalysis.

Scans ERA5 2m temperature over an input date range and identifies
heat wave (daily max > 85th percentile for 3+ consecutive days)
and cold snap (daily min < 15th percentile for 3+ consecutive days)
events globally over land.

Bounding boxes represent the maximum spatial extent of each event.
Events terminate when area drops below 50% of their peak area.

Usage:
    python heat_cold_bounds_global.py \\
        --start-date 2023-06-01 --end-date 2023-09-01 \\
        --output heat_cold_global.csv --n-workers 4
"""

import argparse
import logging
import pathlib
import time as time_module
from typing import Dict, List, Optional, Tuple

import numba as nb
import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
from dask.distributed import Client, LocalCluster
from plot_temperature_events import (
    VALID_QUANTILES,
    detect_time_dim,
    max_consecutive_days,
    open_era5_t2m,
    plot_consecutive_map,
    resolve_op,
)

from extremeweatherbench import defaults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3
AREA_DECLINE_FRACTION = 0.5
MIN_GRIDPOINTS = 500


def get_climatology_thresholds(
    q_hw: float = 0.85,
    q_fz: float = 0.15,
    q_hw_upper: Optional[float] = None,
    q_fz_lower: Optional[float] = None,
) -> Tuple[
    xr.DataArray,
    xr.DataArray,
    Optional[xr.DataArray],
    Optional[xr.DataArray],
]:
    """Return percentile climatology DataArrays for heat/freeze detection.

    Args:
        q_hw: Lower-bound quantile for heat wave detection.
            Default is 0.85.
        q_fz: Upper-bound quantile for freeze detection.
            Default is 0.15.
        q_hw_upper: Upper-bound quantile for heat waves. When set,
            only days where temp > q_hw AND temp < q_hw_upper are
            flagged. Default is None.
        q_fz_lower: Lower-bound quantile for freezes. When set, only
            days where temp < q_fz AND temp > q_fz_lower are
            flagged. Default is None.

    Returns:
        A tuple of (clim_hw, clim_fz, clim_hw_upper, clim_fz_lower).
        The last two elements are None when the corresponding optional
        quantile argument is not supplied.
    """
    clim_hw = defaults.get_climatology(q_hw).sortby("latitude")
    clim_fz = defaults.get_climatology(q_fz).sortby("latitude")
    clim_hw_upper = (
        defaults.get_climatology(q_hw_upper).sortby("latitude")
        if q_hw_upper is not None
        else None
    )
    clim_fz_lower = (
        defaults.get_climatology(q_fz_lower).sortby("latitude")
        if q_fz_lower is not None
        else None
    )
    return clim_hw, clim_fz, clim_hw_upper, clim_fz_lower


def build_land_mask(
    lons: xr.DataArray,
    lats: xr.DataArray,
) -> xr.DataArray:
    """Build a boolean land mask (True = land) for the given grid.

    Args:
        lons: 1-D DataArray of longitudes.
        lats: 1-D DataArray of latitudes.

    Returns:
        Boolean DataArray of the same shape as the meshgrid of lons
        and lats, where True indicates a land grid point.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lons, lats)
    return mask == 0


def build_exceedance_masks(
    t2m: xr.DataArray,
    clim_hw: xr.DataArray,
    clim_fz: xr.DataArray,
    land_mask: xr.DataArray,
    op_hw: str = ">",
    op_fz: str = "<",
    clim_hw_upper: Optional[xr.DataArray] = None,
    clim_fz_lower: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Build daily exceedance masks from 6-hourly temperature data.

    Each 6-hourly timestep is compared to its matching
    (dayofyear, hour) climatology. A day passes only if all four
    6-hourly timesteps exceed the threshold.

    When clim_hw_upper is provided, heat-wave days must also satisfy
    temp < upper bound on every 6-hourly step. When clim_fz_lower is
    provided, freeze days must also satisfy temp > lower bound on
    every 6-hourly step.

    Args:
        t2m: 6-hourly 2m temperature DataArray.
        clim_hw: Heat-wave lower-bound climatology indexed by
            (dayofyear, hour).
        clim_fz: Freeze upper-bound climatology indexed by
            (dayofyear, hour).
        land_mask: Boolean DataArray (True = land) matching t2m grid.
        op_hw: Comparison operator string for heat waves.
            Default is ">".
        op_fz: Comparison operator string for freezes.
            Default is "<".
        clim_hw_upper: Optional upper-bound climatology for heat
            waves (exclusive cap). Default is None.
        clim_fz_lower: Optional lower-bound climatology for freezes
            (exclusive floor). Default is None.

    Returns:
        A tuple (hw, fz) of daily boolean DataArrays masked to land,
        where True indicates an exceedance day.
    """
    cmp_hw = resolve_op(op_hw)
    cmp_fz = resolve_op(op_fz)
    tdim = detect_time_dim(t2m)

    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    max_clim_doy = int(clim_hw.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    clim_hw_aligned = clim_hw.sel(
        dayofyear=doy_capped, hour=hour,
    ).reindex_like(t2m, method="nearest")
    clim_fz_aligned = clim_fz.sel(
        dayofyear=doy_capped, hour=hour,
    ).reindex_like(t2m, method="nearest")

    hw_6h = cmp_hw(t2m, clim_hw_aligned)
    fz_6h = cmp_fz(t2m, clim_fz_aligned)

    if clim_hw_upper is not None:
        clim_hw_upper_aligned = clim_hw_upper.sel(
            dayofyear=doy_capped, hour=hour,
        ).reindex_like(t2m, method="nearest")
        hw_6h = hw_6h & (t2m < clim_hw_upper_aligned)

    if clim_fz_lower is not None:
        clim_fz_lower_aligned = clim_fz_lower.sel(
            dayofyear=doy_capped, hour=hour,
        ).reindex_like(t2m, method="nearest")
        fz_6h = fz_6h & (t2m > clim_fz_lower_aligned)

    hw = hw_6h.resample({tdim: "1D"}).min().astype(bool) & land_mask
    fz = fz_6h.resample({tdim: "1D"}).min().astype(bool) & land_mask
    return hw, fz


def apply_consecutive_filter(
    mask: np.ndarray,
    min_days: int = MIN_CONSECUTIVE_DAYS,
    max_grace_days: int = 1,
) -> np.ndarray:
    """Keep runs of ``min_days``+ True days along axis 0.

    After ``min_days`` strict consecutive True days are established,
    gaps of up to ``max_grace_days`` are bridged so the event can
    continue. Runs that never reach ``min_days`` strict consecutive
    True days are discarded.

    Args:
        mask: Boolean array of shape (time, lat, lon).
        min_days: Minimum run length required to qualify as an
            event. Default is 3 (MIN_CONSECUTIVE_DAYS).
        max_grace_days: Maximum gap length to bridge after the
            minimum run is established. Default is 1.

    Returns:
        Boolean array of the same shape with only qualifying runs
        retained.
    """
    struct = np.zeros((min_days, 1, 1), dtype=bool)
    struct[:, 0, 0] = True

    strict = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                mask, structure=struct, border_value=False,
            ),
            structure=struct, border_value=False,
        )
        & mask
    )

    if max_grace_days <= 0:
        return strict

    close_k = np.zeros((2 * max_grace_days + 1, 1, 1), dtype=bool)
    close_k[:, 0, 0] = True
    filled = ndimage.binary_closing(
        mask, structure=close_k, border_value=False,
    )

    filled_runs = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                filled, structure=struct, border_value=False,
            ),
            structure=struct, border_value=False,
        )
        & filled
    )

    lbl_struct = np.zeros((3, 3, 3), dtype=int)
    lbl_struct[0, 1, 1] = 1
    lbl_struct[1, 1, 1] = 1
    lbl_struct[2, 1, 1] = 1
    labels, _ = ndimage.label(filled_runs, structure=lbl_struct)

    valid = np.unique(labels[strict & (labels > 0)])
    return np.isin(labels, valid) & filled_runs


@nb.njit(cache=True)
def _count_overlaps_nb(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_a: int,
    n_b: int,
) -> np.ndarray:
    """Compute a pixel-count overlap matrix between two label grids.

    A single-pass JIT loop avoids the per-blob numpy scan in the
    Python tracker.

    Args:
        labels_a: Integer label array for day t (shape lat x lon).
        labels_b: Integer label array for day t+1 (shape lat x lon).
        n_a: Number of blobs in labels_a.
        n_b: Number of blobs in labels_b.

    Returns:
        Int32 array of shape (n_a, n_b) where element [i, j] is the
        number of pixels where labels_a == i+1 and labels_b == j+1.
    """
    mat = np.zeros((n_a, n_b), dtype=np.int32)
    rows, cols = labels_a.shape
    for r in range(rows):
        for c in range(cols):
            a = labels_a[r, c]
            b = labels_b[r, c]
            if a > 0 and b > 0:
                mat[a - 1, b - 1] += 1
    return mat


def _resolve_event(
    oid: int,
    overlap_mat: Optional[np.ndarray],
    prev_map: Dict[int, int],
    events: Dict[int, Dict],
    cur_map: Dict[int, int],
) -> Optional[int]:
    """Find and merge prior-day events overlapping with blob oid.

    Uses a precomputed overlap matrix column instead of scanning the
    full prev_labels array per blob.

    Args:
        oid: Current-day blob label (1-indexed).
        overlap_mat: Pixel-count overlap matrix from
            _count_overlaps_nb, or None if no previous day.
        prev_map: Mapping from previous-day blob label to event ID.
        events: Mutable dict of all live events keyed by event ID.
        cur_map: Mutable mapping from current-day blob label to event
            ID; updated in-place when events are merged.

    Returns:
        The surviving event ID after merging, or None if no overlap
        with a prior-day event was found.
    """
    if overlap_mat is None:
        return None
    col = overlap_mat[:, oid - 1]
    prev_blob_ids = np.where(col > 0)[0] + 1
    eids = {prev_map[p] for p in prev_blob_ids if p in prev_map}
    alive = {e for e in eids if e in events and not events[e]["done"]}
    if not alive:
        return None

    eid = min(alive)
    for other in alive - {eid}:
        merged = events.pop(other)
        tgt = events[eid]
        tgt["lat_min"] = min(tgt["lat_min"], merged["lat_min"])
        tgt["lat_max"] = max(tgt["lat_max"], merged["lat_max"])
        tgt["lon_min"] = min(tgt["lon_min"], merged["lon_min"])
        tgt["lon_max"] = max(tgt["lon_max"], merged["lon_max"])
        tgt["peak"] = max(tgt["peak"], merged["peak"])
        tgt["start"] = min(tgt["start"], merged["start"])
        for k, v in list(cur_map.items()):
            if v == other:
                cur_map[k] = eid
    return eid


def _terminate_declined_events(
    events: Dict[int, Dict],
    cur_map: Dict[int, int],
) -> None:
    """Mark events as done if absent today or below 50% of peak area.

    Args:
        events: Mutable dict of all live events keyed by event ID.
        cur_map: Mapping from current-day blob label to event ID;
            used to determine which events are still active today.
    """
    active = set(cur_map.values())
    for eid, ev in events.items():
        if ev["done"]:
            continue
        if eid not in active:
            ev["done"] = True
        elif ev["area"] < AREA_DECLINE_FRACTION * ev["peak"]:
            ev["done"] = True


def detect_events(
    filtered_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    event_type: str,
) -> List[Dict]:
    """Track spatiotemporal events from a filtered boolean mask.

    Labels one day at a time (low peak memory) but uses a numba-JIT
    overlap matrix to match blobs across days. A single O(nlat*nlon)
    pass replaces one O(nlat*nlon) scan per blob, which dominates
    when many blobs are active simultaneously.

    An event's bounding box is the union of all its daily extents.
    Events terminate when their active area drops below 50% of peak.

    Args:
        filtered_mask: Boolean array of shape (time, lat, lon) with
            consecutive-day filtering already applied.
        dates: 1-D array of date labels aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array aligned with axis 2.
        event_type: Label string stored in each returned event dict
            (e.g. "heat_wave" or "cold_snap").

    Returns:
        List of event dicts with keys type, start, end, lat_min,
        lat_max, lon_min, lon_max, peak, area, done.
    """
    n_days = filtered_mask.shape[0]
    events: Dict[int, Dict] = {}
    next_id = 1
    prev_labels: Optional[np.ndarray] = None
    prev_n_obj: int = 0
    prev_map: Dict[int, int] = {}

    for di in range(n_days):
        day = filtered_mask[di]

        if not day.any():
            for ev in events.values():
                if not ev["done"]:
                    ev["done"] = True
            prev_labels = None
            prev_n_obj = 0
            prev_map = {}
            continue

        labels, n_obj = ndimage.label(day)
        labels = labels.astype(np.int32)
        cur_map: Dict[int, int] = {}

        overlap_mat: Optional[np.ndarray] = None
        if prev_labels is not None and n_obj > 0 and prev_n_obj > 0:
            overlap_mat = _count_overlaps_nb(
                prev_labels,
                labels,
                prev_n_obj,
                n_obj,
            )

        for oid in range(1, n_obj + 1):
            om = labels == oid
            area = int(om.sum())
            li, lo = np.where(om)

            eid = _resolve_event(
                oid,
                overlap_mat,
                prev_map,
                events,
                cur_map,
            )

            if eid is None:
                eid = next_id
                next_id += 1
                events[eid] = {
                    "type": event_type,
                    "start": dates[di],
                    "end": dates[di],
                    "lat_min": float(lats[li].min()),
                    "lat_max": float(lats[li].max()),
                    "lon_min": float(lons[lo].min()),
                    "lon_max": float(lons[lo].max()),
                    "peak": area,
                    "area": area,
                    "done": False,
                }
            else:
                ev = events[eid]
                ev["end"] = dates[di]
                ev["lat_min"] = min(
                    ev["lat_min"],
                    float(lats[li].min()),
                )
                ev["lat_max"] = max(
                    ev["lat_max"],
                    float(lats[li].max()),
                )
                ev["lon_min"] = min(
                    ev["lon_min"],
                    float(lons[lo].min()),
                )
                ev["lon_max"] = max(
                    ev["lon_max"],
                    float(lons[lo].max()),
                )
                ev["peak"] = max(ev["peak"], area)
                ev["area"] = area

            cur_map[oid] = eid

        _terminate_declined_events(events, cur_map)
        prev_labels = labels
        prev_n_obj = n_obj
        prev_map = cur_map

    for ev in events.values():
        ev["done"] = True

    return list(events.values())


def events_to_dataframe(
    events: List[Dict],
    min_gridpoints: int = MIN_GRIDPOINTS,
) -> pd.DataFrame:
    """Convert event dicts to a labelled DataFrame.

    Args:
        events: Raw event dicts from ``detect_events``.
        min_gridpoints: Drop events whose peak spatial extent (in
            grid points) is below this threshold. Default is 500
            (MIN_GRIDPOINTS).

    Returns:
        DataFrame with columns label, event_type, start_date,
        end_date, latitude_min, latitude_max, longitude_min,
        longitude_max, sorted by start_date. Events below
        min_gridpoints are excluded.
    """
    columns = [
        "label",
        "event_type",
        "start_date",
        "end_date",
        "latitude_min",
        "latitude_max",
        "longitude_min",
        "longitude_max",
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    n_before = len(events)
    events = [e for e in events if e["peak"] >= min_gridpoints]
    logger.info(
        "  Filtered %d events below %d gridpoints; %d remain",
        n_before - len(events),
        min_gridpoints,
        len(events),
    )

    if not events:
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "event_type": e["type"],
            "start_date": str(e["start"]),
            "end_date": str(e["end"]),
            "latitude_min": e["lat_min"],
            "latitude_max": e["lat_max"],
            "longitude_min": e["lon_min"],
            "longitude_max": e["lon_max"],
        }
        for e in events
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


def main():
    parser = argparse.ArgumentParser(
        description=("Detect heat waves and cold snaps globally from ERA5."),
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--output",
        default="heat_cold_global.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers",
    )
    parser.add_argument(
        "--quantile-hw",
        type=float,
        default=0.85,
        help=(
            "Climatology quantile for heat waves "
            f"({VALID_QUANTILES}; default: 0.85)"
        ),
    )
    parser.add_argument(
        "--quantile-fz",
        type=float,
        default=0.15,
        help=(
            "Climatology quantile for freezes "
            f"({VALID_QUANTILES}; default: 0.15)"
        ),
    )
    parser.add_argument(
        "--quantile-hw-upper",
        type=float,
        default=None,
        help=(
            "Upper-bound quantile for heat waves; days must be "
            "> --quantile-hw AND < this value "
            f"({VALID_QUANTILES}; default: None)"
        ),
    )
    parser.add_argument(
        "--quantile-fz-lower",
        type=float,
        default=None,
        help=(
            "Lower-bound quantile for freezes; days must be "
            "< --quantile-fz AND > this value "
            f"({VALID_QUANTILES}; default: None)"
        ),
    )
    parser.add_argument(
        "--operator-hw",
        default=">",
        help="Comparison operator for heat waves (default: >)",
    )
    parser.add_argument(
        "--operator-fz",
        default="<",
        help="Comparison operator for freezes (default: <)",
    )
    args = parser.parse_args()

    wall_start = time_module.time()
    client = Client(
        LocalCluster(n_workers=args.n_workers),
    )
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Pre-warming numba JIT...")
    _tiny = np.zeros((2, 2), dtype=np.int32)
    _tiny[0, 0] = 1
    _tiny[1, 1] = 1
    _count_overlaps_nb(_tiny, _tiny, 1, 1)
    logger.info("  numba ready")

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info("Loading climatology thresholds...")
    clim_hw, clim_fz, clim_hw_upper, clim_fz_lower = (
        get_climatology_thresholds(
            q_hw=args.quantile_hw,
            q_fz=args.quantile_fz,
            q_hw_upper=args.quantile_hw_upper,
            q_fz_lower=args.quantile_fz_lower,
        )
    )

    logger.info("Building land mask...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)

    logger.info("Building exceedance masks (lazy)...")
    hw_lazy, fz_lazy = build_exceedance_masks(
        t2m,
        clim_hw,
        clim_fz,
        land_mask,
        op_hw=args.operator_hw,
        op_fz=args.operator_fz,
        clim_hw_upper=clim_hw_upper,
        clim_fz_lower=clim_fz_lower,
    )

    tdim = detect_time_dim(hw_lazy)

    logger.info("Computing heat wave mask...")
    t0 = time_module.time()
    hw_da = hw_lazy.compute()
    logger.info("  done in %.1f s", time_module.time() - t0)

    logger.info("Computing cold snap mask...")
    t0 = time_module.time()
    fz_da = fz_lazy.compute()
    logger.info("  done in %.1f s", time_module.time() - t0)

    dates = hw_da[tdim].values
    lats = hw_da.latitude.values
    lons = hw_da.longitude.values
    hw_np = hw_da.values.astype(bool)
    fz_np = fz_da.values.astype(bool)
    del hw_da, fz_da

    logger.info(
        "Applying %d-day consecutive filter...",
        MIN_CONSECUTIVE_DAYS,
    )
    hw_filt = apply_consecutive_filter(hw_np)
    fz_filt = apply_consecutive_filter(fz_np)
    logger.info(
        "  HW: %d -> %d True cells",
        hw_np.sum(),
        hw_filt.sum(),
    )
    logger.info(
        "  FZ: %d -> %d True cells",
        fz_np.sum(),
        fz_filt.sum(),
    )
    del hw_np, fz_np

    logger.info("Detecting heat wave events...")
    hw_ev = detect_events(
        hw_filt,
        dates,
        lats,
        lons,
        "heat_wave",
    )
    logger.info("  %d events", len(hw_ev))

    logger.info("Detecting cold snap events...")
    fz_ev = detect_events(
        fz_filt,
        dates,
        lats,
        lons,
        "cold_snap",
    )
    logger.info("  %d events", len(fz_ev))

    logger.info("Computing max-consecutive-days fields for plots...")
    stem = str(pathlib.Path(args.output).with_suffix(""))
    hw_consec = max_consecutive_days(hw_filt)
    plot_consecutive_map(
        hw_consec,
        lats,
        lons,
        "heat_wave",
        title=(f"Consecutive Heatwave Days\n{args.start_date} to {args.end_date}"),
        output_path=f"{stem}_heatwave.png",
    )
    del hw_consec, hw_filt

    fz_consec = max_consecutive_days(fz_filt)
    plot_consecutive_map(
        fz_consec,
        lats,
        lons,
        "cold_snap",
        title=(f"Consecutive Cold Snap Days\n{args.start_date} to {args.end_date}"),
        output_path=f"{stem}_cold_snap.png",
    )
    del fz_consec, fz_filt

    df = events_to_dataframe(hw_ev + fz_ev)
    df.to_csv(args.output, index=False)

    elapsed = time_module.time() - wall_start
    logger.info(
        "Done: %d events -> %s (%.1f s)",
        len(df),
        args.output,
        elapsed,
    )
    client.close()


if __name__ == "__main__":
    main()
