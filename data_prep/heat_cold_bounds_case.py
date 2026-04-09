"""Validate and expand heat wave / freeze bounding boxes.

Reads heat_wave and freeze cases from base_temp_events.yaml (which uses
centered_region),
then:
  1. Uses the centered_region center as the initial box.
  2. Grows the time window forward from start_date-3 one day at a
     time until the single-day exceedance fraction drops below 50 %
     of its peak.
  3. Iteratively grows the spatial box by 2 degrees in each direction
     until fewer than 50 % of grid points on each edge exceed the
     climatological threshold (or 10 iterations).
  4. Writes the final bounds as bounded_region entries into
     events.yaml.

Usage:
    python heat_cold_bounds_case.py \\
        --output heat_cold_yaml.csv --n-workers 4
"""

import argparse
import importlib
import logging
import pathlib
import time as time_module
from typing import Dict, List, Literal, Optional, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
    plot_consecutive_map,
    resolve_op,
)
from ruamel.yaml import YAML

from extremeweatherbench import cases, defaults, inputs, regions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3
EXPANSION_DEGREES = 1
MAX_ITERATIONS = 20
EDGE_VALIDITY_THRESHOLD = 0.5
MIN_GRIDPOINTS = 500
TEMPORAL_LOAD_BUFFER_DAYS = 14
MAX_TEMPORAL_DAYS = 21


def _apply_consecutive_filter(
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

    # Strict 3-day runs (no grace)
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

    # Fill gaps of ≤ max_grace_days
    close_k = np.zeros((2 * max_grace_days + 1, 1, 1), dtype=bool)
    close_k[:, 0, 0] = True
    filled = ndimage.binary_closing(
        mask,
        structure=close_k,
        border_value=False,
    )

    # Runs of min_days+ in the gap-filled mask
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

    # Label each temporal run independently per grid point
    lbl_struct = np.zeros((3, 3, 3), dtype=int)
    lbl_struct[0, 1, 1] = 1
    lbl_struct[1, 1, 1] = 1
    lbl_struct[2, 1, 1] = 1
    labels, _ = ndimage.label(filled_runs, structure=lbl_struct)

    # Keep only runs that contain a strict 3-day block
    valid = np.unique(labels[strict & (labels > 0)])
    return np.isin(labels, valid) & filled_runs


def _edge_valid_fraction(
    mask_2d: np.ndarray,
    land_2d: np.ndarray,
    edge: str,
    band_pts: int,
) -> float:
    """Return the fraction of land grid points on an edge that are active.

    Ocean/masked points are excluded from both numerator and denominator
    so coastal edges aren't penalised.

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


def _to_plot_lon(lon: float) -> float:
    """Convert 0-360 longitude to -180..180 for plotting.

    Args:
        lon: Longitude in 0-360 degrees.

    Returns:
        Longitude in -180..180 degrees.
    """
    return lon - 360 if lon > 180 else lon


def plot_peak_day_with_bounds(
    peak_mask: np.ndarray,
    all_lats: np.ndarray,
    all_lons: np.ndarray,
    initial_box: tuple,
    final_box: tuple,
    event_type: str,
    title: str,
    output_path: str,
) -> None:
    """Plot the peak-footprint day with initial and final bounding boxes.

    Args:
        peak_mask: 2-D boolean array (lat, lon) for the peak day.
        all_lats: 1-D latitude array matching peak_mask rows.
        all_lons: 1-D longitude array (0-360) matching peak_mask cols.
        initial_box: (lat_min, lat_max, lon_min, lon_max) of the
            initial centered region.
        final_box: (lat_min, lat_max, lon_min, lon_max) after spatial
            expansion.
        event_type: ``"heat_wave"`` or ``"freeze"``.
        title: Figure title.
        output_path: Destination PNG file path.
    """
    plot_lons = np.array([_to_plot_lon(x) for x in all_lons])
    lon_min = float(plot_lons.min()) - 1
    lon_max = float(plot_lons.max()) + 1
    lat_min = float(all_lats.min()) - 1
    lat_max = float(all_lats.max()) + 1

    cmap = mcolors.ListedColormap(
        ["white", "firebrick" if event_type == "heat_wave" else "steelblue"]
    )
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(10, 8),
    )
    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)

    ax.pcolormesh(
        plot_lons,
        all_lats,
        peak_mask.astype(float),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    def _add_rect(box, color, label, ls="-"):
        blat_min, blat_max, blon_min, blon_max = box
        blon_min = _to_plot_lon(blon_min)
        blon_max = _to_plot_lon(blon_max)
        rect = mpatches.Rectangle(
            (blon_min, blat_min),
            blon_max - blon_min,
            blat_max - blat_min,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle=ls,
            transform=ccrs.PlateCarree(),
            label=label,
        )
        ax.add_patch(rect)

    _add_rect(initial_box, "blue", "Initial box", ls="--")
    _add_rect(final_box, "green", "Final bounds", ls="-")

    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved peak-day plot: %s", output_path)


def process_event(
    single_case: cases.IndividualCase,
    out_dir: pathlib.Path = pathlib.Path("."),
    quantile: float | None = None,
    op_str: str | None = None,
) -> Optional[Dict]:
    """Process one event: find the time window then expand bounds.

    Args:
        single_case: The individual case to process; must have a
            CenteredRegion location.
        out_dir: Directory in which plots are saved. Default is
            pathlib.Path(".").
        quantile: Climatology quantile. Default is None, which
            resolves to 0.85 for heat_wave or 0.15 for freeze.
        op_str: Comparison operator string (e.g. ">", ">=", "<",
            "<="). Default is None, which resolves to ">" for
            heat_wave or "<" for freeze.

    Returns:
        A dict with keys event_type, start_date, end_date,
        latitude_min/max, longitude_min/max, case_id, title, and
        _peak_gridpoints, or None if the case was skipped.
    """
    is_heatwave = single_case.event_type == "heat_wave"
    logger.info(
        "Processing case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    loc = single_case.location
    if not isinstance(loc, regions.CenteredRegion):
        logger.warning(
            "  Case %d location is %s, not CenteredRegion — skipping",
            single_case.case_id_number,
            type(loc).__name__,
        )
        return None

    center_lat = loc.latitude
    center_lon = loc.longitude
    if isinstance(loc.bounding_box_degrees, tuple):
        half_lat = loc.bounding_box_degrees[0] / 2.0
        half_lon = loc.bounding_box_degrees[1] / 2.0
    else:
        half_lat = loc.bounding_box_degrees / 2.0
        half_lon = loc.bounding_box_degrees / 2.0

    box_lat_min = center_lat - half_lat
    box_lat_max = center_lat + half_lat
    box_lon_min = center_lon - half_lon
    box_lon_max = center_lon + half_lon

    logger.info(
        "  Initial box (%.1f°): center (%.2f, %.2f)"
        "  lat [%.2f, %.2f], lon [%.2f, %.2f]",
        loc.bounding_box_degrees
        if not isinstance(loc.bounding_box_degrees, tuple)
        else loc.bounding_box_degrees[0],
        center_lat,
        center_lon,
        box_lat_min,
        box_lat_max,
        box_lon_min,
        box_lon_max,
    )

    # Time range and spatial pre-fetch extent
    start_date = pd.Timestamp(single_case.start_date) - pd.Timedelta(days=3)
    end_date = pd.Timestamp(single_case.end_date) + pd.Timedelta(
        days=TEMPORAL_LOAD_BUFFER_DAYS
    )

    pot_lat_min = max(
        -90,
        box_lat_min - MAX_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lat_max = min(
        90,
        box_lat_max + MAX_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lon_min = box_lon_min - MAX_ITERATIONS * EXPANSION_DEGREES
    pot_lon_max = box_lon_max + MAX_ITERATIONS * EXPANSION_DEGREES

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

    if float(t2m.longitude.max()) > 180:
        t2m = t2m.assign_coords(
            longitude=(t2m.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    # Convert center lon to match ERA5 convention for slicing
    pot_lon_min_era = pot_lon_min
    pot_lon_max_era = pot_lon_max
    if pot_lon_min_era > 180:
        pot_lon_min_era -= 360
    if pot_lon_max_era > 180:
        pot_lon_max_era -= 360

    t2m = t2m.sel(
        latitude=slice(pot_lat_min, pot_lat_max),
        longitude=slice(
            min(pot_lon_min_era, pot_lon_max_era),
            max(pot_lon_min_era, pot_lon_max_era),
        ),
    )

    if t2m.latitude.size == 0 or t2m.longitude.size == 0:
        logger.warning(
            "  Case %d: empty spatial selection — skipping",
            single_case.case_id_number,
        )
        return None

    # 6-hourly exceedance then daily all-pass
    if quantile is None:
        quantile = 0.85 if is_heatwave else 0.15
    if op_str is None:
        op_str = ">" if is_heatwave else "<"
    cmp = resolve_op(op_str)

    clim = defaults.get_climatology(quantile).sortby("latitude")

    if float(clim.longitude.max()) > 180:
        clim = clim.assign_coords(
            longitude=(clim.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    max_clim_doy = int(clim.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    clim_aligned = clim.sel(
        dayofyear=doy_capped,
        hour=hour,
    ).reindex_like(t2m, method="nearest")

    exc_6h = cmp(t2m, clim_aligned)

    daily_all_pass = exc_6h.resample({tdim: "1D"}).min().astype(bool)

    land_reg = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = (
        land_reg.mask(
            daily_all_pass.longitude,
            daily_all_pass.latitude,
        )
        == 0
    )

    exc_mask = daily_all_pass & land_mask

    logger.info("  Computing exceedance mask...")
    mask_np = exc_mask.compute().values.astype(bool)

    all_lats = daily_all_pass.latitude.values
    all_lons = daily_all_pass.longitude.values
    grid_res = np.abs(np.diff(all_lats[:2]))[0] if len(all_lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    def _lat_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lats - val)))

    def _lon_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lons - val)))

    # Convert box bounds to ERA5 lon convention for index lookup
    box_lon_min_era = box_lon_min
    box_lon_max_era = box_lon_max
    if box_lon_min_era > 180:
        box_lon_min_era -= 360
    if box_lon_max_era > 180:
        box_lon_max_era -= 360

    idx_s0 = _lat_idx(box_lat_min)
    idx_n0 = _lat_idx(box_lat_max)
    idx_w0 = _lon_idx(box_lon_min_era)
    idx_e0 = _lon_idx(box_lon_max_era)

    # Temporal iteration
    # Land mask within the initial box (exclude ocean from denominator)
    land_mask_np = land_mask.compute().values.astype(bool)
    box_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    n_land_pts = int(box_land.sum())
    if n_land_pts == 0:
        logger.warning(
            "  Case %d: no land points in initial box — skipping",
            single_case.case_id_number,
        )
        return None

    n_days_total = min(mask_np.shape[0], MAX_TEMPORAL_DAYS)
    final_t = n_days_total
    established = False

    for t in range(n_days_total):
        if t < 2:
            continue

        # "Currently active": last 3 days all exceed threshold
        box_sl = (
            slice(idx_s0, idx_n0 + 1),
            slice(idx_w0, idx_e0 + 1),
        )
        currently_active = (
            mask_np[t][box_sl] & mask_np[t - 1][box_sl] & mask_np[t - 2][box_sl]
        )
        active_land = int((currently_active & box_land).sum())
        frac = active_land / n_land_pts

        if t == 2 and frac < EDGE_VALIDITY_THRESHOLD:
            logger.warning(
                "  Day 3: only %.1f%% of land points have 3 consecutive days (< 50%%)",
                frac * 100,
            )

        if not established:
            if frac >= EDGE_VALIDITY_THRESHOLD:
                established = True
                logger.info(
                    "  Event established at day %d (%.1f%% of land points active)",
                    t,
                    frac * 100,
                )
        else:
            if frac < EDGE_VALIDITY_THRESHOLD:
                final_t = t + 1
                logger.info(
                    "  Temporal stop at day %d (%.1f%% < 50%% of land points)",
                    t,
                    frac * 100,
                )
                break

        if t == 9:
            logger.warning(
                "  Exceeded 10 days (frac %.1f%%), continuing...",
                frac * 100,
            )

    if not established:
        logger.warning(
            "  Case %d: event never reached 50%% of land points — using all %d days",
            single_case.case_id_number,
            final_t,
        )

    logger.info(
        "  Using %d of %d available days",
        final_t,
        mask_np.shape[0],
    )

    mask_np = mask_np[:final_t]
    filtered = _apply_consecutive_filter(mask_np)

    # Spatial expansion on peak-footprint day
    # Find the timestep(s) with the most active grid points.
    # If tied: 2nd of 2, middle of odd count, first-middle of even.
    daily_counts = filtered.sum(axis=(1, 2))
    max_count = daily_counts.max()
    (tied_days,) = np.where(daily_counts == max_count)
    n_tied = len(tied_days)
    if n_tied <= 2:
        peak_day = int(tied_days[-1]) if n_tied == 2 else int(tied_days[0])
    else:
        peak_day = int(tied_days[(n_tied - 1) // 2])
    peak_mask = filtered[peak_day]
    logger.info(
        "  Peak footprint on day %d (%d active grid points, %d tied days)",
        peak_day,
        int(max_count),
        n_tied,
    )

    idx_s = idx_s0
    idx_n = idx_n0
    idx_w = idx_w0
    idx_e = idx_e0

    # Pre-check: disable edges that are >= 95% ocean
    # in the initial box (prevents expansion through water).
    init_region_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    edges_active = {}
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
        edges_active[edge] = land_frac >= 0.25
        if not edges_active[edge]:
            logger.info(
                "  %s edge disabled (%.1f%% land in initial box)",
                edge.capitalize(),
                land_frac * 100,
            )
    n_iter = 0

    for iteration in range(MAX_ITERATIONS):
        n_iter = iteration + 1
        region = peak_mask[idx_s : idx_n + 1, idx_w : idx_e + 1]
        land_region = land_mask_np[idx_s : idx_n + 1, idx_w : idx_e + 1]

        all_done = True
        for edge in list(edges_active.keys()):
            if not edges_active[edge]:
                continue

            frac = _edge_valid_fraction(
                region,
                land_region,
                edge,
                band_pts,
            )
            if frac < EDGE_VALIDITY_THRESHOLD:
                edges_active[edge] = False
            else:
                all_done = False
                if edge == "north":
                    idx_n = min(
                        len(all_lats) - 1,
                        idx_n + band_pts,
                    )
                elif edge == "south":
                    idx_s = max(0, idx_s - band_pts)
                elif edge == "east":
                    idx_e = min(
                        len(all_lons) - 1,
                        idx_e + band_pts,
                    )
                elif edge == "west":
                    idx_w = max(0, idx_w - band_pts)

        if all_done:
            logger.info(
                "  Spatial expansion converged at iteration %d",
                n_iter,
            )
            break
    else:
        logger.info(
            "  Reached max spatial iterations (%d)",
            MAX_ITERATIONS,
        )

    fin_lats = all_lats[idx_s : idx_n + 1]
    fin_lons = all_lons[idx_w : idx_e + 1]
    fin_filtered = filtered[:, idx_s : idx_n + 1, idx_w : idx_e + 1]
    consec = max_consecutive_days(fin_filtered)
    peak_gridpoints = int((consec >= MIN_CONSECUTIVE_DAYS).sum())

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
        "final_days": final_t,
        "_consec": consec,
        "_lats": fin_lats,
        "_lons": fin_lons,
        "_peak_gridpoints": peak_gridpoints,
    }
    logger.info(
        "  Final bounds: lat [%.2f, %.2f], lon [%.2f, %.2f] (%d days, %d iterations)",
        result["latitude_min"],
        result["latitude_max"],
        result["longitude_min"],
        result["longitude_max"],
        final_t,
        n_iter,
    )

    kind = "heatwave" if is_heatwave else "freeze"
    start = result["start_date"][:10]
    end = result["end_date"][:10]
    out_png = str(out_dir / f"case_{result['case_id']}_consecutive_{kind}_days.png")
    plot_consecutive_map(
        consec,
        fin_lats,
        fin_lons,
        cast(Literal["heat_wave", "cold_snap"], single_case.event_type),
        title=(
            f"Consecutive {kind.capitalize()} Days"
            f" — {result['title']}\n{start} to {end}"
        ),
        output_path=out_png,
    )
    logger.info("  Saved plot: %s", out_png)

    initial_box = (box_lat_min, box_lat_max, box_lon_min, box_lon_max)
    final_box = (
        result["latitude_min"],
        result["latitude_max"],
        result["longitude_min"],
        result["longitude_max"],
    )
    peak_png = str(out_dir / f"case_{result['case_id']}_peak_day_{kind}.png")
    plot_peak_day_with_bounds(
        peak_mask,
        all_lats,
        all_lons,
        initial_box,
        final_box,
        single_case.event_type,
        title=(
            f"Peak Footprint (day {peak_day}) — {result['title']}\n{start} to {end}"
        ),
        output_path=peak_png,
    )

    return result


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert result dicts to a labelled DataFrame.

    Args:
        results: List of dicts returned by process_event (may
            contain None entries which are ignored). Each dict must
            have keys event_type, start_date, end_date,
            latitude_min/max, longitude_min/max, case_id, title, and
            _peak_gridpoints.

    Returns:
        DataFrame with columns label, case_id, title, event_type,
        start_date, end_date, latitude_min, latitude_max,
        longitude_min, longitude_max, sorted by start_date.
        Events below MIN_GRIDPOINTS are excluded.
    """
    columns = [
        "label",
        "case_id",
        "title",
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


def write_bounds_to_yaml(
    results: List[Dict],
    yaml_path: pathlib.Path,
) -> None:
    """Update events.yaml with computed bounded_region bounds.

    Uses ruamel.yaml round-trip mode so indentation, quoting style,
    key order, blank lines, and comments are preserved exactly.
    Only heat_wave / freeze cases with a valid result are modified.

    Args:
        results: List of dicts returned by process_event (None
            entries are ignored).
        yaml_path: Path to the events.yaml file to update in-place.
    """
    result_map = {r["case_id"]: r for r in results if r is not None}
    if not result_map:
        logger.warning("write_bounds_to_yaml: no valid results, skipping.")
        return

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True

    with yaml_path.open("r") as fh:
        data = yaml.load(fh)

    updated = 0
    for entry in data:
        cid = entry.get("case_id_number")
        if cid not in result_map:
            continue
        r = result_map[cid]
        params = entry["location"]["parameters"]
        params["latitude_min"] = round(r["latitude_min"], 2)
        params["latitude_max"] = round(r["latitude_max"], 2)
        params["longitude_min"] = round(r["longitude_min"], 2)
        params["longitude_max"] = round(r["longitude_max"], 2)
        updated += 1

    with yaml_path.open("w") as fh:
        yaml.dump(data, fh)

    logger.info(
        "write_bounds_to_yaml: updated %d cases in %s",
        updated,
        yaml_path,
    )


def _load_base_temp_events() -> list[cases.IndividualCase]:
    """Load all cases from base_temp_events.yaml.

    Returns:
        List of IndividualCase objects parsed from the bundled
        base_temp_events.yaml resource file.
    """
    import extremeweatherbench.data

    old_yaml = importlib.resources.files(
        extremeweatherbench.data,
    ).joinpath("base_temp_events.yaml")
    with importlib.resources.as_file(old_yaml) as f:
        raw = cases.read_incoming_yaml(f)
    return cases.load_individual_cases(raw)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate / expand heat wave and freeze bounds"
            " using base_temp_events.yaml centered_region data."
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
    parser.add_argument(
        "--case-min",
        type=int,
        default=None,
        help="Minimum case_id_number to process (inclusive)",
    )
    parser.add_argument(
        "--case-max",
        type=int,
        default=None,
        help="Maximum case_id_number to process (inclusive)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help=(
            "Climatology quantile "
            f"({VALID_QUANTILES}; "
            "default: 0.85 for heat_wave, 0.15 for freeze)"
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Comparison operator "
            "(>, >=, <, <=, ==; "
            "default: > for heat_wave, < for freeze)"
        ),
    )
    args = parser.parse_args()

    wall_start = time_module.time()

    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    old_cases = _load_base_temp_events()
    hw_fz = [
        e
        for e in old_cases
        if e.event_type in ("heat_wave", "freeze")
        and (args.case_min is None or e.case_id_number >= args.case_min)
        and (args.case_max is None or e.case_id_number <= args.case_max)
    ]
    logger.info(
        "Found %d heat_wave / freeze events in base_temp_events.yaml",
        len(hw_fz),
    )

    out_dir = pathlib.Path(args.output).parent

    results = joblib.Parallel(n_jobs=args.n_workers)(
        joblib.delayed(process_event)(
            c,
            out_dir,
            quantile=args.quantile,
            op_str=args.operator,
        )
        for c in hw_fz
    )

    df = results_to_dataframe(results)
    out_path = pathlib.Path(args.output)
    if out_path.exists():
        df.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df.to_csv(out_path, index=False)

    yaml_path = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "extremeweatherbench"
        / "data"
        / "events.yaml"
    )
    write_bounds_to_yaml(results, yaml_path)

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
