"""Detect temperature exceedance events globally from ERA5 reanalysis.

Scans ERA5 2m temperature over an input date range and identifies
events where the daily temperature falls within a user-specified
climatology quantile band for 3+ consecutive days, over land.

At least one of --quantile-lower or --quantile-upper must be given.
Both can be combined to define a band (e.g. 50th–85th percentile).

Bounding boxes are first derived from blob tracking, then expanded
using the same edge-validity logic as heat_cold_bounds_case.py:
each edge grows by 1 degree while >= 50% of its land points are
active on the peak-footprint day.  Events terminate when their
active area drops below 50% of peak.

Usage:
    # Anything above the 85th percentile (heat wave)
    python temperature_bounds_global.py \\
        --start-date 2023-06-01 --end-date 2023-09-01 \\
        --quantile-lower 0.85 --operator-lower ">=" \\
        --event-type heat_wave --output heat_cold_global.csv

    # Band between 50th and 85th (moderate heat)
    python temperature_bounds_global.py \\
        --start-date 2023-06-01 --end-date 2023-09-01 \\
        --quantile-lower 0.50 --quantile-upper 0.85 \\
        --event-type heat_wave --output heat_cold_global.csv
"""

import argparse
import logging
import pathlib
import time as time_module
from typing import Dict, List, Optional, Tuple

import joblib
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
        area = R² × Δlat_rad × Δlon_rad × cos(lat)

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

    At least one of q_lower or q_upper must be provided. Each returned
    DataArray is indexed by (dayofyear, hour) and sorted by latitude.

    Args:
        q_lower: Quantile for the lower bound (e.g. 0.50 means temp
            must exceed the 50th-percentile climatology). None skips
            the lower-bound check.
        q_upper: Quantile for the upper bound (e.g. 0.85 means temp
            must not exceed the 85th-percentile climatology). None
            skips the upper-bound check.

    Returns:
        Tuple (clim_lower, clim_upper); either element may be None
        when the corresponding quantile argument is not supplied.

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
        Boolean DataArray of the same shape as the meshgrid of lons
        and lats, where True indicates a land grid point.
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

    Ocean/masked points are excluded from both numerator and denominator
    so coastal edges are not penalised.

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
    max_consecutive_days within the expanded region.

    The expansion matches the spatial-growth algorithm used in
    heat_cold_bounds_case.py: 1-degree steps, ocean-heavy edges
    disabled upfront, convergence when all active edges drop below 50%.

    Args:
        event: Event dict from detect_events (keys: start, end,
            lat_min, lat_max, lon_min, lon_max).
        filt_mask: Boolean array (time, lat, lon) with consecutive-day
            filtering already applied, aligned with dates/lats/lons.
        dates: 1-D datetime64 array of daily timestamps (axis 0).
        lats: 1-D latitude array (axis 1).
        lons: 1-D longitude array (axis 2).
        land_mask_np: 2-D boolean array (lat, lon); True = land.

    Returns:
        The same event dict with updated lat_min/lat_max/lon_min/
        lon_max and a new max_consecutive_days key.
    """
    grid_res = float(np.abs(np.diff(lats[:2]))[0]) if len(lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    start_date = np.datetime64(event["start"])
    end_date = np.datetime64(event["end"])
    t_mask = (dates >= start_date) & (dates <= end_date)
    ev_filt = filt_mask[t_mask]

    if ev_filt.shape[0] == 0:
        event["max_consecutive_days"] = 0
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

    At least one of clim_lower or clim_upper must be provided.

    Args:
        t2m: 6-hourly 2m temperature DataArray.
        clim_lower: Climatology for the lower bound, indexed by
            (dayofyear, hour). None skips this bound.
        clim_upper: Climatology for the upper bound, indexed by
            (dayofyear, hour). None skips this bound.
        land_mask: Boolean DataArray (True = land) matching t2m grid.
        op_lower: Comparison operator for the lower bound.
            Default is ">".
        op_upper: Comparison operator for the upper bound.
            Default is "<".

    Returns:
        Daily boolean DataArray masked to land where True indicates
        every 6-hourly step satisfied all active bounds.
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
    area_grid: Optional[np.ndarray] = None,
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
        area_grid: Optional 2-D array (lat, lon) of grid-cell areas
            in km². When provided, peak_area_km2 is tracked per event.

    Returns:
        List of event dicts with keys type, start, end, lat_min,
        lat_max, lon_min, lon_max, peak, peak_area_km2, area, done.
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
                    "peak_area_km2": blob_area_km2,
                    "area": area,
                    "done": False,
                }
            else:
                ev = events[eid]
                ev["end"] = dates[di]
                ev["peak_area_km2"] = max(ev["peak_area_km2"], blob_area_km2)
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
    min_area_km2: float = MIN_AREA_KM2,
) -> pd.DataFrame:
    """Convert event dicts to a labelled DataFrame.

    Args:
        events: Raw event dicts from ``detect_events``.
        min_gridpoints: Drop events whose peak grid-point count is
            below this threshold. Default is 500 (MIN_GRIDPOINTS).
        min_area_km2: Drop events whose peak area (km²) is below
            this threshold. Default is 200 000 (MIN_AREA_KM2).

    Returns:
        DataFrame with columns label, event_type, start_date,
        end_date, latitude_min, latitude_max, longitude_min,
        longitude_max, max_consecutive_days, sorted by start_date.
        Events below either threshold are excluded.
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
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    n_before = len(events)
    events = [
        e
        for e in events
        if e["peak"] >= min_gridpoints and e.get("peak_area_km2", 0.0) >= min_area_km2
    ]
    logger.info(
        "  Filtered %d events (< %d pts or < %.0f km²); %d remain",
        n_before - len(events),
        min_gridpoints,
        min_area_km2,
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
        }
        for e in events
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


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
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers",
    )
    parser.add_argument(
        "--quantile-lower",
        type=float,
        default=None,
        help=(
            f"Lower-bound climatology quantile ({VALID_QUANTILES}). "
            "Days where temp op_lower clim_lower are candidates. "
            "At least one of --quantile-lower or --quantile-upper "
            "is required."
        ),
    )
    parser.add_argument(
        "--operator-lower",
        default=">",
        help=("Comparison operator applied to the lower bound (default: >)"),
    )
    parser.add_argument(
        "--quantile-upper",
        type=float,
        default=None,
        help=(
            f"Upper-bound climatology quantile ({VALID_QUANTILES}). "
            "Days where temp op_upper clim_upper are candidates. "
            "Combine with --quantile-lower to define a band."
        ),
    )
    parser.add_argument(
        "--operator-upper",
        default="<",
        help=("Comparison operator applied to the upper bound (default: <)"),
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
    args = parser.parse_args()

    if args.quantile_lower is None and args.quantile_upper is None:
        parser.error(
            "At least one of --quantile-lower or --quantile-upper must be provided."
        )

    # Derive event_type label and plot colour from the quantiles.
    # Format: "q{lower}+" / "q{upper}-" / "q{lower}-q{upper}"
    ql, qu = args.quantile_lower, args.quantile_upper
    if ql is not None and qu is not None:
        event_type = f"q{ql:.2f}-q{qu:.2f}"
    elif ql is not None:
        event_type = f"q{ql:.2f}+"
    else:
        event_type = f"q{qu:.2f}-"
    # Infer warm vs cold for the plot colourmap:
    # warm when the lower bound is above the median (high percentile),
    # cold when the upper bound is below the median (low percentile).
    if ql is not None and ql >= 0.5:
        plot_event_type = "heat_wave"
    elif qu is not None and qu <= 0.5:
        plot_event_type = "cold_snap"
    else:
        plot_event_type = "heat_wave"

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

    logger.info("Building exceedance mask (lazy)...")
    exc_lazy = build_exceedance_mask(
        t2m,
        clim_lower,
        clim_upper,
        land_mask,
        op_lower=args.operator_lower,
        op_upper=args.operator_upper,
    )

    tdim = detect_time_dim(exc_lazy)

    logger.info("Computing exceedance mask...")
    t0 = time_module.time()
    exc_da = exc_lazy.compute()
    logger.info("  done in %.1f s", time_module.time() - t0)

    dates = exc_da[tdim].values
    lats = exc_da.latitude.values
    lons = exc_da.longitude.values
    exc_np = exc_da.values.astype(bool)
    del exc_da

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

    logger.info("Detecting events...")
    t0 = time_module.time()
    events = detect_events(exc_filt, dates, lats, lons, event_type, area_grid=area_grid)
    logger.info("  %d events (%.1f s)", len(events), time_module.time() - t0)

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

    logger.info("Computing max-consecutive-days plot...")
    stem = str(pathlib.Path(args.output).with_suffix(""))
    consec = max_consecutive_days(exc_filt)
    plot_consecutive_map(
        consec,
        lats,
        lons,
        plot_event_type,
        title=(
            f"Consecutive Exceedance Days ({event_type})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_consec.png",
    )
    del consec, exc_filt

    df = events_to_dataframe(events)
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
