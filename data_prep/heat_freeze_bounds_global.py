#!/usr/bin/env python3
"""Detect heat waves and freezes globally from ERA5 reanalysis.

Scans ERA5 2m temperature over an input date range and identifies
heat wave (daily max > 85th percentile for 3+ consecutive days)
and freeze (daily min < 15th percentile for 3+ consecutive days)
events globally over land.

Bounding boxes represent the maximum spatial extent of each event.
Events terminate when area drops below 50% of their peak area.

Usage:
    python heat_freeze_bounds_global.py \\
        --start-date 2023-06-01 --end-date 2023-09-01 \\
        --output heat_freeze_global.csv --n-workers 4
"""

import argparse
import logging
import time as time_module
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
from dask.distributed import Client, LocalCluster

from extremeweatherbench import defaults, inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3
AREA_DECLINE_FRACTION = 0.5


def detect_time_dim(obj: xr.Dataset | xr.DataArray) -> str:
    """Return the name of the time dimension."""
    for name in ("valid_time", "time"):
        if name in obj.dims:
            return name
    raise ValueError(f"No time dimension found. Available: {list(obj.dims)}")


def open_era5_t2m(
    start_date: str,
    end_date: str,
) -> xr.DataArray:
    """Open ERA5 2m temperature lazily for a date range.

    Selects 6-hourly timesteps (0/6/12/18 UTC) to match the
    climatology base and sorts latitude to ascending order.
    """
    ds = xr.open_zarr(
        inputs.ARCO_ERA5_FULL_URI,
        storage_options={"token": "anon"},
        chunks={},
    )
    tdim = detect_time_dim(ds)
    t2m = ds["2m_temperature"].sel({tdim: slice(start_date, end_date)})
    six_hourly = t2m[tdim].dt.hour.isin([0, 6, 12, 18])
    t2m = t2m.sel({tdim: six_hourly})
    return t2m.sortby("latitude")


def get_daily_climatology_thresholds() -> Tuple[xr.DataArray, xr.DataArray]:
    """Derive daily thresholds from 6-hourly percentile climatology.

    Returns the max-over-hours of the 85th percentile (heat wave
    threshold) and min-over-hours of the 15th percentile (freeze
    threshold), each indexed by dayofyear.
    """
    clim_85 = defaults.get_climatology(0.85)
    clim_15 = defaults.get_climatology(0.15)
    return (
        clim_85.max(dim="hour").sortby("latitude"),
        clim_15.min(dim="hour").sortby("latitude"),
    )


def build_land_mask(
    lons: xr.DataArray,
    lats: xr.DataArray,
) -> xr.DataArray:
    """Boolean land mask (True = land) for the given grid."""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lons, lats)
    return mask == 0


def _align_climatology(
    clim: xr.DataArray,
    daily: xr.DataArray,
    tdim: str,
) -> xr.DataArray:
    """Align a dayofyear-indexed climatology to daily data.

    Computes the climatology into memory first (it is small:
    366 x lat x lon) to avoid dask chunk multiplication, then
    indexes by dayofyear via numpy for clean dimension handling.
    """
    clim_vals = clim.compute().values
    clim_doy = clim.dayofyear.values
    doy = daily[tdim].dt.dayofyear.values
    max_doy = int(clim_doy.max())
    doy_idx = np.clip(doy, 1, max_doy) - int(clim_doy.min())

    clim_lats = clim.latitude.values
    clim_lons = clim.longitude.values
    daily_lats = daily.latitude.values
    daily_lons = daily.longitude.values

    if (
        clim_lats.shape == daily_lats.shape
        and np.allclose(clim_lats, daily_lats)
        and clim_lons.shape == daily_lons.shape
        and np.allclose(clim_lons, daily_lons)
    ):
        selected = clim_vals[doy_idx]
    else:
        lat_idx = np.array([np.argmin(np.abs(clim_lats - dl)) for dl in daily_lats])
        lon_idx = np.array([np.argmin(np.abs(clim_lons - dl)) for dl in daily_lons])
        selected = clim_vals[np.ix_(doy_idx, lat_idx, lon_idx)]

    return xr.DataArray(
        selected,
        dims=daily.dims,
        coords=daily.coords,
    )


def build_exceedance_masks(
    t2m: xr.DataArray,
    clim_daily_max: xr.DataArray,
    clim_daily_min: xr.DataArray,
    land_mask: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Build lazy boolean masks for heat wave / freeze exceedance.

    Daily max/min aggregation stays in the dask graph. The
    climatology is loaded into memory once (small) and then
    broadcast against the lazy daily arrays to avoid chunk
    multiplication warnings.
    """
    tdim = detect_time_dim(t2m)

    daily_max = t2m.resample({tdim: "1D"}).max()
    daily_min = t2m.resample({tdim: "1D"}).min()

    clim_max_aligned = _align_climatology(
        clim_daily_max,
        daily_max,
        tdim,
    )
    clim_min_aligned = _align_climatology(
        clim_daily_min,
        daily_min,
        tdim,
    )

    hw = (daily_max > clim_max_aligned) & land_mask
    fz = (daily_min < clim_min_aligned) & land_mask
    return hw, fz


def apply_consecutive_filter(
    mask: np.ndarray,
    min_days: int = MIN_CONSECUTIVE_DAYS,
) -> np.ndarray:
    """Keep only grid points in runs of ``min_days``+ True days.

    Binary erosion removes runs shorter than ``min_days``;
    dilation restores surviving runs to their original extent.
    Operates only along axis 0 (time).
    """
    struct = np.zeros((min_days, 1, 1), dtype=bool)
    struct[:, 0, 0] = True
    eroded = ndimage.binary_erosion(
        mask,
        structure=struct,
        border_value=False,
    )
    dilated = ndimage.binary_dilation(
        eroded,
        structure=struct,
        border_value=False,
    )
    return dilated & mask


def _match_to_previous(
    obj_mask: np.ndarray,
    prev_labels: Optional[np.ndarray],
    prev_map: Dict[int, int],
    events: Dict[int, Dict],
    cur_map: Dict[int, int],
) -> Optional[int]:
    """Find the event id from the previous day that overlaps."""
    if prev_labels is None:
        return None
    overlap = prev_labels[obj_mask]
    prev_ids = set(overlap[overlap > 0])
    eids = {prev_map[p] for p in prev_ids if p in prev_map}
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
    """Terminate events absent today or below 50% of peak."""
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

    Per-day 2-D connected component labelling with overlap-based
    tracking across consecutive days. An event's bounding box is
    the union of all its daily extents. Events terminate when
    their active area drops below 50% of the peak.
    """
    n_days = filtered_mask.shape[0]
    events: Dict[int, Dict] = {}
    next_id = 1
    prev_labels: Optional[np.ndarray] = None
    prev_map: Dict[int, int] = {}

    for di in range(n_days):
        day = filtered_mask[di]

        if not day.any():
            for ev in events.values():
                if not ev["done"]:
                    ev["done"] = True
            prev_labels = None
            prev_map = {}
            continue

        labels, n_obj = ndimage.label(day)
        cur_map: Dict[int, int] = {}

        for oid in range(1, n_obj + 1):
            om = labels == oid
            area = int(om.sum())
            li, lo = np.where(om)

            eid = _match_to_previous(
                om,
                prev_labels,
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
        prev_map = cur_map

    for ev in events.values():
        ev["done"] = True

    return list(events.values())


def events_to_dataframe(events: List[Dict]) -> pd.DataFrame:
    """Convert event dicts to a labelled DataFrame."""
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
        description=("Detect heat waves and freezes globally from ERA5."),
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
        default="heat_freeze_global.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers",
    )
    args = parser.parse_args()

    wall_start = time_module.time()
    client = Client(
        LocalCluster(n_workers=args.n_workers),
    )
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info("Loading climatology thresholds...")
    clim_max, clim_min = get_daily_climatology_thresholds()

    logger.info("Building land mask...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)

    logger.info("Building exceedance masks (lazy)...")
    hw_lazy, fz_lazy = build_exceedance_masks(
        t2m,
        clim_max,
        clim_min,
        land_mask,
    )

    tdim = detect_time_dim(hw_lazy)

    logger.info("Computing heat wave mask...")
    t0 = time_module.time()
    hw_da = hw_lazy.compute()
    logger.info("  done in %.1f s", time_module.time() - t0)

    logger.info("Computing freeze mask...")
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

    logger.info("Detecting freeze events...")
    fz_ev = detect_events(
        fz_filt,
        dates,
        lats,
        lons,
        "freeze",
    )
    logger.info("  %d events", len(fz_ev))
    del hw_filt, fz_filt

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
