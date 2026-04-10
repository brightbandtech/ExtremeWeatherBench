"""Detect temperature exceedance events globally from ERA5 reanalysis.

Scans ERA5 2m temperature over an input date range and identifies
events where the daily temperature falls within a user-specified
climatology quantile band for 3+ consecutive days, over land.

At least one of --quantile-lower or --quantile-upper must be given.
Both can be combined to define a band (e.g. 50th-85th percentile).

Bounding boxes are first derived from blob tracking, then expanded
using the same edge-validity logic as heat_cold_bounds_case.py:
each edge grows by 1 degree while >= 50% of its land points are
active on the peak-footprint day.  Events terminate when their
active area drops below 50% of peak.

Usage:
    # Anything below the 15th percentile (cold snap)
    python temperature_bounds_global.py \\
        --start-date 2020-01-01 --end-date 2020-12-31 \\
        --quantile-upper 0.15 \\
        --output cold_q15_2020.csv

    # Anything above the 85th percentile (heat wave)
    python temperature_bounds_global.py \\
        --start-date 2023-06-01 --end-date 2023-09-01 \\
        --quantile-lower 0.85 --operator-lower ">=" \\
        --output heat_q85_2023.csv
"""

import argparse
import logging
import pathlib
import time as time_module
from typing import Dict, List, Literal, Optional, Tuple, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
import dask
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
MIN_AREA_KM2 = 200000.0
EXPANSION_DEGREES = 1
MAX_SPATIAL_ITERATIONS = 20
EDGE_VALIDITY_THRESHOLD = 0.5


def compute_grid_cell_area(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return a 2-D (lat, lon) array of grid-cell areas in km².

    Uses the spherical-Earth approximation:
        area = R^2 * dlat_rad * dlon_rad * cos(lat)

    Args:
        lats: 1-D latitude array in degrees.
        lons: 1-D longitude array in degrees; used only for shape.

    Returns:
        Float64 array of shape (len(lats), len(lons)) with each
        cell's surface area in km².
    """
    R_KM = 6371.0
    dlat = float(np.abs(np.diff(lats[:2]))[0]) if len(lats) > 1 else 0.25
    dlon = float(np.abs(np.diff(lons[:2]))[0]) if len(lons) > 1 else 0.25
    cell_km2 = R_KM**2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lats))
    return np.outer(cell_km2, np.ones(len(lons)))


def get_climatology_bounds(
    q_lower: Optional[float] = None,
    q_upper: Optional[float] = None,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Return climatology DataArrays for the lower and/or upper bound.

    Args:
        q_lower: Quantile for the lower bound. None skips it.
        q_upper: Quantile for the upper bound. None skips it.

    Returns:
        Tuple (clim_lower, clim_upper); either may be None.

    Raises:
        ValueError: If both q_lower and q_upper are None.
    """
    if q_lower is None and q_upper is None:
        raise ValueError("At least one of q_lower or q_upper must be set.")
    clim_lower = (
        defaults.get_climatology(q_lower).sortby("latitude")
        if q_lower is not None
        else None
    )
    clim_upper = (
        defaults.get_climatology(q_upper).sortby("latitude")
        if q_upper is not None
        else None
    )
    return clim_lower, clim_upper


def build_land_mask(
    lons: xr.DataArray,
    lats: xr.DataArray,
) -> xr.DataArray:
    """Build a boolean land mask (True = land) for the given grid.

    Args:
        lons: 1-D DataArray of longitudes.
        lats: 1-D DataArray of latitudes.

    Returns:
        Boolean DataArray where True indicates a land grid point.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lons, lats)
    return mask == 0


def _edge_valid_fraction(
    mask_2d: np.ndarray,
    land_2d: np.ndarray,
    edge: str,
    band_pts: int,
) -> float:
    """Return the fraction of land grid points on an edge that are active.

    Args:
        mask_2d: 2-D boolean activity array (lat, lon).
        land_2d: 2-D boolean land mask (lat, lon); True = land.
        edge: One of "north", "south", "east", "west".
        band_pts: Width of the edge strip in grid points.

    Returns:
        Fraction in [0, 1] of land points in the strip that are active,
        or 0.0 if the strip contains no land points.

    Raises:
        ValueError: If edge is not one of the recognised values.
    """
    if edge == "north":
        strip = mask_2d[-band_pts:, :]
        land_strip = land_2d[-band_pts:, :]
    elif edge == "south":
        strip = mask_2d[:band_pts, :]
        land_strip = land_2d[:band_pts, :]
    elif edge == "west":
        strip = mask_2d[:, :band_pts]
        land_strip = land_2d[:, :band_pts]
    elif edge == "east":
        strip = mask_2d[:, -band_pts:]
        land_strip = land_2d[:, -band_pts:]
    else:
        raise ValueError(f"Unknown edge: {edge}")
    n_land = int(land_strip.sum())
    if n_land == 0:
        return 0.0
    return float((strip & land_strip).sum()) / n_land


def expand_event_bounds(
    event: Dict,
    filt_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    land_mask_np: np.ndarray,
) -> Dict:
    """Expand a detected event's bounding box using case-script logic.

    Starting from the blob-tracked bounding box, expands each edge
    outward while >= EDGE_VALIDITY_THRESHOLD of land points on that
    edge are active on the peak-footprint day. Also computes and stores
    max_consecutive_days and mean_consecutive_days within the expanded
    region.

    Args:
        event: Event dict from detect_events.
        filt_mask: Boolean array (time, lat, lon) with consecutive-day
            filtering already applied.
        dates: 1-D datetime64 array of daily timestamps (axis 0).
        lats: 1-D latitude array (axis 1).
        lons: 1-D longitude array (axis 2).
        land_mask_np: 2-D boolean array (lat, lon); True = land.

    Returns:
        The same event dict with updated lat_min/lat_max/lon_min/
        lon_max, max_consecutive_days, and mean_consecutive_days.
    """
    grid_res = float(np.abs(np.diff(lats[:2]))[0]) if len(lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    start_date = np.datetime64(event["start"])
    end_date = np.datetime64(event["end"])
    t_mask = (dates >= start_date) & (dates <= end_date)
    ev_filt = filt_mask[t_mask]

    if ev_filt.shape[0] == 0:
        event["max_consecutive_days"] = 0
        event["mean_consecutive_days"] = 0.0
        return event

    def _lat_idx(val: float) -> int:
        return int(np.argmin(np.abs(lats - val)))

    def _lon_idx(val: float) -> int:
        return int(np.argmin(np.abs(lons - val)))

    idx_s0 = _lat_idx(event["lat_min"])
    idx_n0 = _lat_idx(event["lat_max"])
    idx_w0 = _lon_idx(event["lon_min"])
    idx_e0 = _lon_idx(event["lon_max"])

    daily_counts = ev_filt.sum(axis=(1, 2))
    max_count = daily_counts.max()
    tied_days = np.where(daily_counts == max_count)[0]
    n_tied = len(tied_days)
    if n_tied <= 2:
        peak_day = int(tied_days[-1]) if n_tied == 2 else int(tied_days[0])
    else:
        peak_day = int(tied_days[(n_tied - 1) // 2])
    peak_mask = ev_filt[peak_day]

    idx_s, idx_n, idx_w, idx_e = idx_s0, idx_n0, idx_w0, idx_e0

    init_region_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    edges_active: Dict[str, bool] = {}
    for edge in ("north", "south", "east", "west"):
        if edge == "north":
            strip = init_region_land[-band_pts:, :]
        elif edge == "south":
            strip = init_region_land[:band_pts, :]
        elif edge == "west":
            strip = init_region_land[:, :band_pts]
        else:
            strip = init_region_land[:, -band_pts:]
        land_frac = strip.sum() / max(strip.size, 1)
        edges_active[edge] = bool(land_frac >= 0.25)

    for _ in range(MAX_SPATIAL_ITERATIONS):
        region = peak_mask[idx_s : idx_n + 1, idx_w : idx_e + 1]
        land_region = land_mask_np[idx_s : idx_n + 1, idx_w : idx_e + 1]

        all_done = True
        for edge in list(edges_active.keys()):
            if not edges_active[edge]:
                continue
            frac = _edge_valid_fraction(region, land_region, edge, band_pts)
            if frac < EDGE_VALIDITY_THRESHOLD:
                edges_active[edge] = False
            else:
                all_done = False
                if edge == "north":
                    idx_n = min(len(lats) - 1, idx_n + band_pts)
                elif edge == "south":
                    idx_s = max(0, idx_s - band_pts)
                elif edge == "east":
                    idx_e = min(len(lons) - 1, idx_e + band_pts)
                elif edge == "west":
                    idx_w = max(0, idx_w - band_pts)

        if all_done:
            break

    fin_filt = ev_filt[:, idx_s : idx_n + 1, idx_w : idx_e + 1]
    consec = max_consecutive_days(fin_filt)
    event["lat_min"] = float(lats[idx_s])
    event["lat_max"] = float(lats[idx_n])
    event["lon_min"] = float(lons[idx_w])
    event["lon_max"] = float(lons[idx_e])
    event["max_consecutive_days"] = int(consec.max()) if consec.size > 0 else 0
    active = consec[consec >= MIN_CONSECUTIVE_DAYS]
    event["mean_consecutive_days"] = (
        round(float(active.mean()), 2) if active.size > 0 else 0.0
    )
    return event


def build_exceedance_mask(
    t2m: xr.DataArray,
    clim_lower: Optional[xr.DataArray],
    clim_upper: Optional[xr.DataArray],
    land_mask: xr.DataArray,
    op_lower: str = ">",
    op_upper: str = "<",
) -> xr.DataArray:
    """Build a daily exceedance mask from 6-hourly temperature data.

    Each 6-hourly timestep must satisfy all provided bounds. A day
    passes only when every 6-hourly step satisfies every active bound.
    The result is further masked to land points.

    Args:
        t2m: 6-hourly 2m temperature DataArray.
        clim_lower: Climatology for the lower bound, or None.
        clim_upper: Climatology for the upper bound, or None.
        land_mask: Boolean DataArray (True = land) matching t2m grid.
        op_lower: Comparison operator for the lower bound. Default ">".
        op_upper: Comparison operator for the upper bound. Default "<".

    Returns:
        Daily boolean DataArray masked to land.
    """
    tdim = detect_time_dim(t2m)
    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    ref = clim_lower if clim_lower is not None else clim_upper
    assert ref is not None
    max_clim_doy = int(ref.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    mask_6h = xr.ones_like(t2m, dtype=bool)

    if clim_lower is not None:
        cmp = resolve_op(op_lower)
        aligned = clim_lower.sel(dayofyear=doy_capped, hour=hour).reindex_like(
            t2m, method="nearest"
        )
        mask_6h = mask_6h & cmp(t2m, aligned)

    if clim_upper is not None:
        cmp = resolve_op(op_upper)
        aligned = clim_upper.sel(dayofyear=doy_capped, hour=hour).reindex_like(
            t2m, method="nearest"
        )
        mask_6h = mask_6h & cmp(t2m, aligned)

    return mask_6h.resample({tdim: "1D"}).min().astype(bool) & land_mask


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
        min_days: Minimum run length. Default is 3.
        max_grace_days: Maximum gap length to bridge. Default is 1.

    Returns:
        Boolean array of the same shape with qualifying runs only.
    """
    struct = np.zeros((min_days, 1, 1), dtype=bool)
    struct[:, 0, 0] = True

    strict = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                mask,
                structure=struct,
                border_value=False,
            ),
            structure=struct,
            border_value=False,
        )
        & mask
    )

    if max_grace_days <= 0:
        return strict

    close_k = np.zeros((2 * max_grace_days + 1, 1, 1), dtype=bool)
    close_k[:, 0, 0] = True
    filled = ndimage.binary_closing(
        mask,
        structure=close_k,
        border_value=False,
    )

    filled_runs = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                filled,
                structure=struct,
                border_value=False,
            ),
            structure=struct,
            border_value=False,
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

    Args:
        oid: Current-day blob label (1-indexed).
        overlap_mat: Pixel-count overlap matrix, or None.
        prev_map: Mapping from previous-day blob label to event ID.
        events: Mutable dict of all live events keyed by event ID.
        cur_map: Mutable mapping from current-day blob label to event ID.

    Returns:
        The surviving event ID after merging, or None if no overlap.
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
        tgt["peak_area_km2"] = max(tgt["peak_area_km2"], merged["peak_area_km2"])
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
        cur_map: Mapping from current-day blob label to event ID.
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
    area_grid: Optional[np.ndarray] = None,
    min_seed_area_km2: float = 0.0,
) -> List[Dict]:
    """Track spatiotemporal events from a filtered boolean mask.

    New events are only seeded for blobs whose area meets or exceeds
    ``min_seed_area_km2`` on the current day AND on each of the prior
    ``MIN_CONSECUTIVE_DAYS - 1`` days (strict joint-area check). The
    overlap of the current-day blob footprint with each previous day's
    filtered mask must individually satisfy the threshold, ensuring the
    contiguous region was jointly >= ``min_seed_area_km2`` for all
    MIN_CONSECUTIVE_DAYS days. Blobs that fail but overlap an already-
    established event still extend it.

    Args:
        filtered_mask: Boolean array of shape (time, lat, lon) with
            consecutive-day filtering already applied.
        dates: 1-D array of date labels aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array aligned with axis 2.
        event_type: Label string stored in each returned event dict.
        area_grid: Optional 2-D array (lat, lon) of grid-cell areas
            in km².
        min_seed_area_km2: Minimum contiguous area in km² that the
            blob footprint must satisfy on each of MIN_CONSECUTIVE_DAYS
            days to seed a new event. Default is 0.0 (all blobs seed).

    Returns:
        List of event dicts with keys type, start, end, lat_min,
        lat_max, lon_min, lon_max, peak, peak_area_km2, area, done,
        initial_bbox.
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
            blob_area_km2 = (
                float(area_grid[li, lo].sum()) if area_grid is not None else 0.0
            )

            eid = _resolve_event(
                oid,
                overlap_mat,
                prev_map,
                events,
                cur_map,
            )

            if eid is None:
                if blob_area_km2 < min_seed_area_km2:
                    continue
                # Strict joint-area check: the blob's footprint must
                # also have been >= min_seed_area_km2 on each of the
                # prior MIN_CONSECUTIVE_DAYS-1 days. Skip seeding when
                # insufficient history is available.
                if min_seed_area_km2 > 0:
                    if di < MIN_CONSECUTIVE_DAYS - 1:
                        continue
                    joint_ok = True
                    for k in range(1, MIN_CONSECUTIVE_DAYS):
                        overlap = om & filtered_mask[di - k]
                        prev_area = (
                            float(area_grid[overlap].sum())
                            if area_grid is not None
                            else float(overlap.sum())
                        )
                        if prev_area < min_seed_area_km2:
                            joint_ok = False
                            break
                    if not joint_ok:
                        continue
                eid = next_id
                next_id += 1
                _lat_min = float(lats[li].min())
                _lat_max = float(lats[li].max())
                _lon_min = float(lons[lo].min())
                _lon_max = float(lons[lo].max())
                events[eid] = {
                    "type": event_type,
                    "start": dates[di],
                    "end": dates[di],
                    "lat_min": _lat_min,
                    "lat_max": _lat_max,
                    "lon_min": _lon_min,
                    "lon_max": _lon_max,
                    "peak": area,
                    "peak_area_km2": blob_area_km2,
                    "area": area,
                    "done": False,
                    "initial_bbox": (
                        _lat_min,
                        _lat_max,
                        _lon_min,
                        _lon_max,
                    ),
                }
            else:
                ev = events[eid]
                ev["end"] = dates[di]
                ev["peak_area_km2"] = max(ev["peak_area_km2"], blob_area_km2)
                ev["lat_min"] = min(ev["lat_min"], float(lats[li].min()))
                ev["lat_max"] = max(ev["lat_max"], float(lats[li].max()))
                ev["lon_min"] = min(ev["lon_min"], float(lons[lo].min()))
                ev["lon_max"] = max(ev["lon_max"], float(lons[lo].max()))
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


def enrich_events_with_temps(
    events: List[Dict],
    t2m_daily_np: np.ndarray,
    exc_filt: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> List[Dict]:
    """Add mean and minimum temperature (C) to each event dict.

    Uses a pre-computed global daily-mean temperature array so that
    all events share a single ERA5 fetch rather than making one Dask
    compute call per event. Statistics are computed over exceedant
    (active) grid point-days within each event's expanded bounding
    box and time window.

    Args:
        events: Event dicts with lat_min/max, lon_min/max, start, end.
        t2m_daily_np: Float32 array (days, lat, lon) of daily-mean
            2 m temperature in Celsius, aligned with dates/lats/lons.
        exc_filt: Boolean array (time, lat, lon) from
            apply_consecutive_filter, aligned with dates/lats/lons.
        dates: 1-D datetime64 array aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array aligned with axis 2.

    Returns:
        The same event list with mean_temp_c and min_temp_c added
        (NaN when no active points exist).
    """

    def _idx(arr: np.ndarray, val: float) -> int:
        return int(np.argmin(np.abs(arr - val)))

    for ev in events:
        start_date = np.datetime64(ev["start"])
        end_date = np.datetime64(ev["end"])
        t_indices = np.where((dates >= start_date) & (dates <= end_date))[0]

        idx_s = _idx(lats, ev["lat_min"])
        idx_n = _idx(lats, ev["lat_max"])
        idx_w = _idx(lons, ev["lon_min"])
        idx_e = _idx(lons, ev["lon_max"])

        exc_ev = exc_filt[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]
        t2m_ev = t2m_daily_np[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]

        n_days = min(t2m_ev.shape[0], exc_ev.shape[0])
        active_temps = t2m_ev[:n_days][exc_ev[:n_days]]
        if active_temps.size > 0:
            ev["mean_temp_c"] = round(float(active_temps.mean()), 2)
            ev["min_temp_c"] = round(float(active_temps.min()), 2)
        else:
            ev["mean_temp_c"] = float("nan")
            ev["min_temp_c"] = float("nan")

    return events


def events_to_dataframe(
    events: List[Dict],
    min_gridpoints: int = MIN_GRIDPOINTS,
    min_area_km2: float = MIN_AREA_KM2,
) -> pd.DataFrame:
    """Convert event dicts to a labelled DataFrame.

    Args:
        events: Raw event dicts from detect_events (after enrichment).
        min_gridpoints: Drop events below this peak grid-point count.
        min_area_km2: Drop events below this peak area (km²).

    Returns:
        DataFrame with columns label, event_type, start_date,
        end_date, latitude_min/max, longitude_min/max,
        max_consecutive_days, mean_consecutive_days, mean_temp_c,
        min_temp_c, sorted by start_date. Events with fewer than
        MIN_CONSECUTIVE_DAYS consecutive days are also excluded.
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
        "max_consecutive_days",
        "mean_consecutive_days",
        "mean_temp_c",
        "min_temp_c",
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    n_before = len(events)
    events = [
        e
        for e in events
        if e["peak"] >= min_gridpoints
        and e.get("peak_area_km2", 0.0) >= min_area_km2
        and e.get("max_consecutive_days", 0) >= MIN_CONSECUTIVE_DAYS
    ]
    logger.info(
        "  Filtered %d events (< %d pts, < %.0f km², or < %d consec days); %d remain",
        n_before - len(events),
        min_gridpoints,
        min_area_km2,
        MIN_CONSECUTIVE_DAYS,
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
            "max_consecutive_days": e.get("max_consecutive_days", 0),
            "mean_consecutive_days": e.get("mean_consecutive_days", 0.0),
            "mean_temp_c": e.get("mean_temp_c", float("nan")),
            "min_temp_c": e.get("min_temp_c", float("nan")),
        }
        for e in events
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


def _add_lon_rect(
    ax: plt.Axes,
    lon_min_0360: float,
    lon_max_0360: float,
    lat_min: float,
    lat_max: float,
    color: tuple,
    linewidth: float = 0.8,
    zorder: int = 4,
) -> None:
    """Add a lat/lon rectangle to ax, splitting at the antimeridian if needed.

    Inputs use 0-360 longitude. When the event wraps around the 0°/360°
    boundary the bounding box is split into two rectangles so that
    matplotlib does not draw an inverted or near-zero-width patch.
    """

    def _to_180(lon: float) -> float:
        return lon if lon <= 180 else lon - 360

    lmin = _to_180(lon_min_0360)
    lmax = _to_180(lon_max_0360)

    def _rect(x0: float, x1: float) -> mpatches.Rectangle:
        return mpatches.Rectangle(
            (x0, lat_min),
            x1 - x0,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=(*color[:3], 0.25),
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )

    if lmin <= lmax:
        ax.add_patch(_rect(lmin, lmax))
    else:
        # Event wraps across the antimeridian – draw two segments.
        ax.add_patch(_rect(lmin, 180.0))
        ax.add_patch(_rect(-180.0, lmax))


def plot_events_global(
    df: pd.DataFrame,
    event_type: str,
    title: str,
    output_path: str,
) -> None:
    """Plot detected events as bounding boxes on a global Robinson map.

    Each event is drawn as a rectangle coloured by its start date.

    Args:
        df: DataFrame with latitude_min/max, longitude_min/max,
            start_date columns.
        event_type: 'heat_wave' or 'cold_snap' for colormap.
        title: Figure title.
        output_path: Destination PNG file path.
    """
    cmap_name = "Reds" if event_type == "heat_wave" else "Blues"
    cmap = plt.colormaps.get_cmap(cmap_name)

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=1)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.5, zorder=2)

    if df.empty:
        ax.set_title(f"{title}\n(no events)", fontsize=11, loc="left")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Events plot saved to %s (no events)", output_path)
        return

    start_dates = pd.to_datetime(df["start_date"])
    date_min = start_dates.min()
    date_max = start_dates.max()
    span = max((date_max - date_min).days, 1)
    norm = mcolors.Normalize(vmin=0, vmax=span)

    for _, row in df.iterrows():
        days = (pd.to_datetime(row["start_date"]) - date_min).days
        color = cmap(norm(days))
        _add_lon_rect(
            ax,
            row["longitude_min"],
            row["longitude_max"],
            row["latitude_min"],
            row["latitude_max"],
            color,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    tick_days = np.linspace(0, span, min(7, span + 1))
    cbar.set_ticks(tick_days)
    cbar.set_ticklabels(
        [(date_min + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in tick_days]
    )
    cbar.set_label("Event start date", fontsize=9)
    ax.set_title(f"{title}\n{len(df)} events", fontsize=11, loc="left")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Events plot saved to %s", output_path)


def plot_events_high_consec(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    min_consec: int = 6,
) -> None:
    """Plot events with >= min_consec days, coloured by consecutive days.

    Uses the plasma colormap (neutral, not event-type specific).

    Args:
        df: DataFrame from events_to_dataframe with columns
            latitude_min/max, longitude_min/max, max_consecutive_days.
        title: Figure title.
        output_path: Destination PNG file path.
        min_consec: Minimum max_consecutive_days to include. Default 6.
    """
    sub = df[df["max_consecutive_days"] >= min_consec].copy()

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=1)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.5, zorder=2)

    if sub.empty:
        ax.set_title(
            f"{title}\n(no events with >= {min_consec} consecutive days)",
            fontsize=11,
            loc="left",
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            "High-consec plot saved to %s (no qualifying events)",
            output_path,
        )
        return

    vmin = min_consec
    vmax = int(sub["max_consecutive_days"].max())
    n_levels = vmax - vmin + 1
    cmap = plt.colormaps.get_cmap("plasma_r").resampled(n_levels)
    bin_edges = np.arange(vmin - 0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

    for _, row in sub.iterrows():
        consec_val = int(row["max_consecutive_days"])
        color = cmap(norm(consec_val))
        _add_lon_rect(
            ax,
            row["longitude_min"],
            row["longitude_max"],
            row["latitude_min"],
            row["latitude_max"],
            color,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_ticks(range(vmin, vmax + 1))
    cbar.set_label("Max Consecutive Days", fontsize=9)
    ax.set_title(f"{title}\n{len(sub)} events", fontsize=11, loc="left")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("High-consec plot saved to %s", output_path)


def plot_event_consec_maps(
    df: pd.DataFrame,
    exc_filt: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    stem: str,
    min_consec: int = 6,
) -> None:
    """Plot a per-event consecutive-days map for each qualifying event.

    For each row in df with max_consecutive_days >= min_consec, slices
    exc_filt to the event's bounding box and time window, computes
    per-grid-point max consecutive days, and saves a pcolormesh plot
    using the plasma colormap.

    Args:
        df: DataFrame from events_to_dataframe with columns label,
            start_date, end_date, latitude_min/max, longitude_min/max,
            max_consecutive_days.
        exc_filt: Boolean array (time, lat, lon) from
            apply_consecutive_filter, aligned with dates/lats/lons.
        dates: 1-D datetime64 array aligned with exc_filt axis 0.
        lats: 1-D latitude array aligned with exc_filt axis 1.
        lons: 1-D longitude array (0-360) aligned with exc_filt axis 2.
        stem: Output path stem; each plot saved as
            {stem}_event_{label:04d}_consec.png.
        min_consec: Minimum max_consecutive_days to plot. Default 6.
    """
    qualifying = df[df["max_consecutive_days"] >= min_consec]
    if qualifying.empty:
        logger.info(
            "No events with >= %d consecutive days to plot individually.",
            min_consec,
        )
        return

    logger.info(
        "Plotting %d per-event consecutive-days maps...",
        len(qualifying),
    )

    def _idx(arr: np.ndarray, val: float) -> int:
        return int(np.argmin(np.abs(arr - val)))

    plot_lons = np.where(lons > 180, lons - 360.0, lons)

    for _, row in qualifying.iterrows():
        label = int(row["label"])
        start_date = np.datetime64(row["start_date"])
        end_date = np.datetime64(row["end_date"])
        t_indices = np.where((dates >= start_date) & (dates <= end_date))[0]

        idx_s = _idx(lats, row["latitude_min"])
        idx_n = _idx(lats, row["latitude_max"])
        idx_w = _idx(lons, row["longitude_min"])
        idx_e = _idx(lons, row["longitude_max"])

        ev_filt = exc_filt[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]
        consec = max_consecutive_days(ev_filt)

        ev_lats = lats[idx_s : idx_n + 1]
        ev_lons = plot_lons[idx_w : idx_e + 1]

        plot_data = consec.astype(float)
        plot_data[consec < MIN_CONSECUTIVE_DAYS] = np.nan
        valid = plot_data[~np.isnan(plot_data)]
        if valid.size == 0:
            continue

        vmax = int(valid.max())
        n_levels = vmax - MIN_CONSECUTIVE_DAYS + 1
        cmap = plt.colormaps.get_cmap("plasma_r").resampled(n_levels)
        bin_edges = np.arange(MIN_CONSECUTIVE_DAYS - 0.5, vmax + 1.5, 1)
        norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(10, 8),
        )
        ax.set_extent(
            [
                float(ev_lons.min()) - 1,
                float(ev_lons.max()) + 1,
                float(ev_lats.min()) - 1,
                float(ev_lats.max()) + 1,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        im = ax.pcolormesh(
            ev_lons,
            ev_lats,
            plot_data,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.coastlines(linewidth=0.5, zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="grey")
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="lightgrey")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(MIN_CONSECUTIVE_DAYS, vmax + 1))
        cbar.set_label("Consecutive Days", fontsize=12)

        start_str = str(row["start_date"])[:10]
        end_str = str(row["end_date"])[:10]
        ax.set_title(
            f"Event {label}: Consecutive Exceedance Days\n{start_str} to {end_str}",
            loc="left",
            fontsize=13,
        )
        out_path = f"{stem}_event_{label:04d}_consec.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Event %04d saved to %s", label, out_path)


def main():
    parser = argparse.ArgumentParser(
        description=("Detect temperature exceedance events globally from ERA5."),
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
        default="events_global.csv",
        help="Output CSV path stem (actual file will be {stem}_events.csv)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers",
    )
    parser.add_argument(
        "--dask-batch",
        type=int,
        default=12,
        help=(
            "Number of monthly chunks to submit to Dask at once. "
            "Larger values increase parallelism but use more memory "
            "and create larger task graphs. Default 12 (1 year at a "
            "time)."
        ),
    )
    parser.add_argument(
        "--quantile-lower",
        type=float,
        default=None,
        help=(
            f"Lower-bound climatology quantile ({VALID_QUANTILES}). "
            "At least one of --quantile-lower or --quantile-upper required."
        ),
    )
    parser.add_argument(
        "--operator-lower",
        default=">",
        help="Comparison operator for the lower bound (default: >)",
    )
    parser.add_argument(
        "--quantile-upper",
        type=float,
        default=None,
        help=(
            f"Upper-bound climatology quantile ({VALID_QUANTILES}). "
            "Combine with --quantile-lower to define a band."
        ),
    )
    parser.add_argument(
        "--operator-upper",
        default="<",
        help="Comparison operator for the upper bound (default: <)",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=-90.0,
        help="Minimum latitude to include in detection. Default -90.0",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=90.0,
        help="Maximum latitude to include in detection. Default 90.0",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=1,
        help=(
            "Process the exceedance mask this many months at a time "
            "to stay within Dask worker memory limits. Set to 0 to "
            "compute the full date range at once. Default is 1."
        ),
    )
    parser.add_argument(
        "--min-area-km2",
        type=float,
        default=200_000.0,
        help=(
            "Minimum contiguous blob area in km² required to seed a "
            "new event AND on each of the prior MIN_CONSECUTIVE_DAYS-1 "
            "days (joint-area check). Always applied, including in "
            "sensitivity mode. Default 200 000."
        ),
    )
    parser.add_argument(
        "--sensitivity-thresholds",
        type=float,
        nargs="+",
        default=None,
        metavar="KM2",
        help=(
            "Count surviving events at each supplied area threshold "
            "(km²) and write a summary CSV. Events are seeded at "
            "--min-area-km2 first; thresholds below that value will "
            "equal the seed-filter count. Example: "
            "--sensitivity-thresholds 200000 300000 500000 1000000"
        ),
    )
    args = parser.parse_args()

    if args.quantile_lower is None and args.quantile_upper is None:
        parser.error(
            "At least one of --quantile-lower or --quantile-upper must be provided."
        )

    ql, qu = args.quantile_lower, args.quantile_upper
    if ql is not None and qu is not None:
        event_type = f"q{ql:.2f}-q{qu:.2f}"
    elif ql is not None:
        event_type = f"q{ql:.2f}+"
    else:
        event_type = f"q{qu:.2f}-"

    if ql is not None and ql >= 0.5:
        plot_event_type = "heat_wave"
    elif qu is not None and qu <= 0.5:
        plot_event_type = "cold_snap"
    else:
        plot_event_type = "heat_wave"

    is_sensitivity = args.sensitivity_thresholds is not None
    if is_sensitivity:
        below = [t for t in args.sensitivity_thresholds if t < args.min_area_km2]
        if below:
            logger.warning(
                "Sensitivity thresholds %s are below the seed filter "
                "(%.0f km²); those counts will equal the seed-filter total.",
                below,
                args.min_area_km2,
            )

    wall_start = time_module.time()
    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Pre-warming numba JIT...")
    _tiny = np.zeros((2, 2), dtype=np.int32)
    _tiny[0, 0] = 1
    _tiny[1, 1] = 1
    _count_overlaps_nb(_tiny, _tiny, 1, 1)
    logger.info("  numba ready")

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    if args.lat_min != -90.0 or args.lat_max != 90.0:
        t2m = t2m.sel(latitude=slice(args.lat_min, args.lat_max))
        logger.info(
            "  Latitude filtered to [%.1f, %.1f]",
            args.lat_min,
            args.lat_max,
        )
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info(
        "Loading climatology bounds (lower=%s, upper=%s)...",
        args.quantile_lower,
        args.quantile_upper,
    )
    clim_lower, clim_upper = get_climatology_bounds(
        q_lower=args.quantile_lower,
        q_upper=args.quantile_upper,
    )

    logger.info("Building land mask and grid-cell area array...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)
    land_mask_np = land_mask.values.astype(bool)
    area_grid = compute_grid_cell_area(t2m.latitude.values, t2m.longitude.values)

    tdim = detect_time_dim(t2m)

    # ── Exceedance mask + daily temperature (combined, parallel) ──────
    # Both passes read the same ERA5 data month-by-month. Combining them
    # into one loop and submitting all lazy chunks to dask.compute() at
    # once lets the Dask workers process months in parallel, halving ERA5
    # reads and eliminating sequential chunk overhead.
    t0 = time_module.time()
    if args.chunk_months > 0:
        logger.info(
            "Building lazy graphs for exceedance mask + temperature "
            "(%d-month chunks, parallel)...",
            args.chunk_months,
        )
        month_starts = pd.date_range(args.start_date, args.end_date, freq="MS")
        chunk_labels = []
        exc_lazy_list = []
        t2m_lazy_list = []
        for i in range(0, len(month_starts), args.chunk_months):
            ms = month_starts[i]
            me = month_starts[min(i + args.chunk_months, len(month_starts)) - 1]
            me = me + pd.offsets.MonthEnd(1)
            chunk_labels.append(f"{ms.strftime('%Y-%m')} - {me.strftime('%Y-%m')}")
            t2m_chunk = t2m.sel({tdim: slice(str(ms.date()), str(me.date()))})
            exc_lazy_list.append(
                build_exceedance_mask(
                    t2m_chunk,
                    clim_lower,
                    clim_upper,
                    land_mask,
                    op_lower=args.operator_lower,
                    op_upper=args.operator_upper,
                )
            )
            t2m_lazy_list.append(t2m_chunk.resample({tdim: "1D"}).mean())

        n_chunks = len(exc_lazy_list)
        batch = args.dask_batch
        logger.info(
            "  Computing %d chunks in batches of %d via Dask...",
            n_chunks,
            batch,
        )
        exc_parts: list = []
        t2m_parts: list = []
        for b0 in range(0, n_chunks, batch):
            b1 = min(b0 + batch, n_chunks)
            b_exc = exc_lazy_list[b0:b1]
            b_t2m = t2m_lazy_list[b0:b1]
            b_sz = b1 - b0
            b_results = dask.compute(*b_exc, *b_t2m)
            exc_parts.extend(b_results[:b_sz])
            t2m_parts.extend(b_results[b_sz:])
            for label in chunk_labels[b0:b1]:
                logger.info("  chunk %s done", label)

        exc_da = xr.concat(exc_parts, dim=tdim)
        t2m_daily_da = xr.concat(t2m_parts, dim=tdim)
        del exc_parts, t2m_parts
    else:
        logger.info(
            "Building lazy graphs (all-at-once) for exceedance mask + temperature..."
        )
        exc_lazy = build_exceedance_mask(
            t2m,
            clim_lower,
            clim_upper,
            land_mask,
            op_lower=args.operator_lower,
            op_upper=args.operator_upper,
        )
        t2m_lazy = t2m.resample({tdim: "1D"}).mean()
        exc_da, t2m_daily_da = dask.compute(exc_lazy, t2m_lazy)

    logger.info(
        "  exceedance mask + temperature done in %.1f s",
        time_module.time() - t0,
    )

    dates = exc_da[tdim].values
    lats = exc_da.latitude.values
    lons = exc_da.longitude.values
    exc_np = exc_da.values.astype(bool)
    del exc_da

    t2m_daily_np = (t2m_daily_da.values - 273.15).astype(np.float32)
    del t2m_daily_da

    logger.info(
        "Applying %d-day consecutive filter...",
        MIN_CONSECUTIVE_DAYS,
    )
    exc_filt = apply_consecutive_filter(exc_np)
    logger.info(
        "  %d -> %d True cells",
        exc_np.sum(),
        exc_filt.sum(),
    )
    del exc_np

    logger.info(
        "Detecting events (min_seed_area_km2=%.0f)...",
        args.min_area_km2,
    )
    t0 = time_module.time()
    events = detect_events(
        exc_filt,
        dates,
        lats,
        lons,
        event_type,
        area_grid=area_grid,
        min_seed_area_km2=args.min_area_km2,
    )
    logger.info("  %d raw events (%.1f s)", len(events), time_module.time() - t0)

    logger.info(
        "Expanding event bounds (%d-deg steps)...",
        EXPANSION_DEGREES,
    )
    t0 = time_module.time()
    events = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(expand_event_bounds)(
            ev, exc_filt, dates, lats, lons, land_mask_np
        )
        for ev in events
    )
    logger.info("  done in %.1f s", time_module.time() - t0)

    logger.info("Computing per-event temperature statistics...")
    t0 = time_module.time()
    events = enrich_events_with_temps(events, t2m_daily_np, exc_filt, dates, lats, lons)
    del t2m_daily_np
    logger.info("  done in %.1f s", time_module.time() - t0)

    stem = str(pathlib.Path(args.output).with_suffix(""))

    df = events_to_dataframe(events, min_area_km2=args.min_area_km2)
    events_csv = f"{stem}_events.csv"
    df.to_csv(events_csv, index=False)
    logger.info("%d events written to %s", len(df), events_csv)

    # ── Global consecutive-days map ───────────────────────────────────
    logger.info("Computing max-consecutive-days plot...")
    consec = max_consecutive_days(exc_filt)
    plot_consecutive_map(
        consec,
        lats,
        lons,
        cast(Literal["heat_wave", "cold_snap"], plot_event_type),
        title=(
            f"Consecutive Exceedance Days ({event_type})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_consec.png",
    )
    del consec

    # ── Overview and high-consec bounding-box plots ───────────────────
    plot_events_global(
        df,
        event_type=plot_event_type,
        title=(
            f"Detected events ({event_type.replace('_', ' ')})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_events.png",
    )
    plot_events_high_consec(
        df,
        title=(
            f"Events \u22656 consecutive days ({event_type.replace('_', ' ')})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_high_consec.png",
    )

    # ── Per-event consecutive-days maps (>= 6 consec days) ───────────
    plot_event_consec_maps(df, exc_filt, dates, lats, lons, stem)

    del exc_filt

    # ── Sensitivity table ─────────────────────────────────────────────
    if is_sensitivity:
        thresholds = sorted(args.sensitivity_thresholds)
        logger.info(
            "Sensitivity test: counting events at %d thresholds...",
            len(thresholds),
        )
        rows = []
        for thresh in thresholds:
            df_thresh = events_to_dataframe(events, min_area_km2=thresh)
            rows.append({"area_threshold_km2": thresh, "n_events": len(df_thresh)})
        sens_df = pd.DataFrame(rows)
        print(sens_df.to_string(index=False))
        sens_path = f"{stem}_sensitivity.csv"
        sens_df.to_csv(sens_path, index=False)
        logger.info("Sensitivity table saved to %s", sens_path)

    elapsed = time_module.time() - wall_start
    logger.info("Done in %.1f s", elapsed)
    client.close()


if __name__ == "__main__":
    main()
