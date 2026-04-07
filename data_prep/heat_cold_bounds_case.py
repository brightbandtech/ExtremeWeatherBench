#!/usr/bin/env python3
"""Validate and expand heat wave / cold snap bounding boxes from events.yaml.

For each heat_wave or cold snap event in events.yaml, iteratively
grows the bounding box by 2 degrees in each direction until fewer
than 50% of grid points on an edge exceed the climatological
threshold (or 10 iterations).

Usage:
    python heat_cold_bounds_case.py \\
        --output heat_cold_yaml.csv --n-workers 4
"""

import argparse
import logging
import pathlib
import time as time_module
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
from dask.distributed import Client, LocalCluster
from plot_temperature_events import (
    detect_time_dim,
    max_consecutive_days,
    plot_consecutive_map,
)

from extremeweatherbench import cases, defaults, inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3
EXPANSION_DEGREES = 2.0
MAX_ITERATIONS = 10
EDGE_VALIDITY_THRESHOLD = 0.5
MIN_GRIDPOINTS = 500


def _apply_consecutive_filter(
    mask: np.ndarray,
    min_days: int = MIN_CONSECUTIVE_DAYS,
) -> np.ndarray:
    """Keep only runs of ``min_days``+ True days (axis 0)."""
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


def _edge_valid_fraction(
    filtered: np.ndarray,
    edge: str,
    band_pts: int,
) -> float:
    """Fraction of grid points on *edge* with any valid day."""
    if edge == "north":
        strip = filtered[:, -band_pts:, :]
    elif edge == "south":
        strip = filtered[:, :band_pts, :]
    elif edge == "west":
        strip = filtered[:, :, :band_pts]
    elif edge == "east":
        strip = filtered[:, :, -band_pts:]
    else:
        raise ValueError(f"Unknown edge: {edge}")

    has_event = strip.any(axis=0)
    if has_event.size == 0:
        return 0.0
    return float(has_event.mean())


def process_event(
    single_case: cases.IndividualCase,
) -> Optional[Dict]:
    """Process one event: compute mask and iteratively expand.

    Opens ERA5 and climatology inside the worker to avoid
    serialisation of lazy zarr handles across joblib processes.
    """
    is_heatwave = single_case.event_type == "heat_wave"
    logger.info(
        "Processing case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    bounds = single_case.location.as_geopandas().total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds

    start_date = pd.Timestamp(single_case.start_date) - pd.Timedelta(days=3)
    end_date = pd.Timestamp(single_case.end_date) + pd.Timedelta(days=3)

    pot_lat_min = max(
        -90,
        lat_min - MAX_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lat_max = min(
        90,
        lat_max + MAX_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lon_min = lon_min - MAX_ITERATIONS * EXPANSION_DEGREES
    pot_lon_max = lon_max + MAX_ITERATIONS * EXPANSION_DEGREES

    ds = xr.open_zarr(
        inputs.ARCO_ERA5_FULL_URI,
        storage_options={"token": "anon"},
        chunks={},
    )
    tdim = detect_time_dim(ds)
    t2m = ds["2m_temperature"].sel(
        {tdim: slice(str(start_date), str(end_date))},
    )
    six_hourly = t2m[tdim].dt.hour.isin([0, 6, 12, 18])
    t2m = t2m.sel({tdim: six_hourly}).sortby("latitude")

    # Normalise ERA5 longitudes from 0-360 to -180/180 so that the
    # case bounding boxes (always in -180/180 from geopandas) can be
    # used directly for slicing without wrap-around issues.
    if float(t2m.longitude.max()) > 180:
        t2m = t2m.assign_coords(
            longitude=(t2m.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    t2m = t2m.sel(
        latitude=slice(pot_lat_min, pot_lat_max),
        longitude=slice(pot_lon_min, pot_lon_max),
    )

    if t2m.latitude.size == 0 or t2m.longitude.size == 0:
        logger.warning(
            "  Case %d: empty spatial selection"
            " (lon=[%.2f, %.2f], lat=[%.2f, %.2f]) — skipping",
            single_case.case_id_number,
            pot_lon_min,
            pot_lon_max,
            pot_lat_min,
            pot_lat_max,
        )
        return None

    if is_heatwave:
        daily = t2m.resample({tdim: "1D"}).max()
        clim = (
            defaults.get_climatology(0.85)
            .max(
                dim="hour",
            )
            .sortby("latitude")
        )
    else:
        daily = t2m.resample({tdim: "1D"}).min()
        clim = (
            defaults.get_climatology(0.15)
            .min(
                dim="hour",
            )
            .sortby("latitude")
        )

    # Match climatology longitude convention to the (possibly
    # remapped) ERA5 data so reindex_like aligns correctly.
    if float(clim.longitude.max()) > 180:
        clim = clim.assign_coords(
            longitude=(clim.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    doy = daily[tdim].dt.dayofyear
    max_clim_doy = int(clim.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)
    clim_aligned = clim.sel(
        dayofyear=doy_capped,
    ).reindex_like(daily, method="nearest")

    land_reg = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = (
        land_reg.mask(
            daily.longitude,
            daily.latitude,
        )
        == 0
    )

    if is_heatwave:
        exc_mask = (daily > clim_aligned) & land_mask
    else:
        exc_mask = (daily < clim_aligned) & land_mask

    logger.info("  Computing exceedance mask...")
    mask_np = exc_mask.compute().values.astype(bool)
    filtered = _apply_consecutive_filter(mask_np)

    all_lats = daily.latitude.values
    all_lons = daily.longitude.values
    grid_res = np.abs(np.diff(all_lats[:2]))[0] if len(all_lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    def _lat_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lats - val)))

    def _lon_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lons - val)))

    idx_s = _lat_idx(lat_min)
    idx_n = _lat_idx(lat_max)
    idx_w = _lon_idx(lon_min)
    idx_e = _lon_idx(lon_max)

    edges_active = {
        "north": True,
        "south": True,
        "east": True,
        "west": True,
    }
    n_iter = 0

    for iteration in range(MAX_ITERATIONS):
        n_iter = iteration + 1
        region = filtered[:, idx_s : idx_n + 1, idx_w : idx_e + 1]

        all_done = True
        for edge in list(edges_active.keys()):
            if not edges_active[edge]:
                continue

            frac = _edge_valid_fraction(
                region,
                edge,
                band_pts,
            )
            if frac < EDGE_VALIDITY_THRESHOLD:
                edges_active[edge] = False
            else:
                all_done = False
                if edge == "north":
                    idx_n = min(len(all_lats) - 1, idx_n + band_pts)
                elif edge == "south":
                    idx_s = max(0, idx_s - band_pts)
                elif edge == "east":
                    idx_e = min(len(all_lons) - 1, idx_e + band_pts)
                elif edge == "west":
                    idx_w = max(0, idx_w - band_pts)

        if all_done:
            logger.info(
                "  Converged at iteration %d",
                n_iter,
            )
            break
    else:
        logger.info(
            "  Reached max iterations (%d)",
            MAX_ITERATIONS,
        )

    fin_lats = all_lats[idx_s : idx_n + 1]
    fin_lons = all_lons[idx_w : idx_e + 1]
    fin_filtered = filtered[:, idx_s : idx_n + 1, idx_w : idx_e + 1]
    consec = max_consecutive_days(fin_filtered)
    peak_gridpoints = int(fin_filtered.any(axis=0).sum())

    result = {
        "case_id": single_case.case_id_number,
        "title": single_case.title,
        "event_type": single_case.event_type,
        "start_date": str(single_case.start_date),
        "end_date": str(single_case.end_date),
        "latitude_min": float(all_lats[idx_s]),
        "latitude_max": float(all_lats[idx_n]),
        "longitude_min": float(all_lons[idx_w]),
        "longitude_max": float(all_lons[idx_e]),
        "n_iterations": n_iter,
        "_consec": consec,
        "_lats": fin_lats,
        "_lons": fin_lons,
        "_peak_gridpoints": peak_gridpoints,
    }
    logger.info(
        "  Final bounds: lat [%.2f, %.2f], lon [%.2f, %.2f]",
        result["latitude_min"],
        result["latitude_max"],
        result["longitude_min"],
        result["longitude_max"],
    )
    return result


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert result dicts to a labelled DataFrame."""
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
    if not results:
        return pd.DataFrame(columns=columns)

    valid = [r for r in results if r is not None]
    n_before = len(valid)
    valid = [r for r in valid if r["_peak_gridpoints"] >= MIN_GRIDPOINTS]
    logger.info(
        "  Filtered %d events below %d gridpoints; %d remain",
        n_before - len(valid),
        MIN_GRIDPOINTS,
        len(valid),
    )

    rows = [
        {
            "event_type": r["event_type"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "latitude_min": r["latitude_min"],
            "latitude_max": r["latitude_max"],
            "longitude_min": r["longitude_min"],
            "longitude_max": r["longitude_max"],
        }
        for r in valid
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate / expand heat wave and cold snap bounds from events.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        default="heat_cold_yaml.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (joblib)",
    )
    args = parser.parse_args()

    wall_start = time_module.time()

    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    events_yaml = cases.load_ewb_cases()
    hw_fz = [e for e in events_yaml if e.event_type in ("heat_wave", "cold_snap")]
    logger.info(
        "Found %d heat_wave / cold snap events",
        len(hw_fz),
    )

    results = joblib.Parallel(n_jobs=args.n_workers)(
        joblib.delayed(process_event)(c) for c in hw_fz
    )

    df = results_to_dataframe(results)
    df.to_csv(args.output, index=False)

    out_dir = pathlib.Path(args.output).parent
    for r in results:
        if r is None:
            continue
        consec = r["_consec"]
        lats = r["_lats"]
        lons = r["_lons"]
        case_id = r["case_id"]
        event_type = r["event_type"]
        kind = "heatwave" if event_type == "heat_wave" else "cold_snap"
        start = r["start_date"][:10]
        end = r["end_date"][:10]
        out_png = str(out_dir / f"case_{case_id}_consecutive_{kind}_days.png")
        plot_consecutive_map(
            consec,
            lats,
            lons,
            event_type,
            title=(
                f"Consecutive {kind.capitalize()} Days — {r['title']}\n{start} to {end}"
            ),
            output_path=out_png,
        )

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
