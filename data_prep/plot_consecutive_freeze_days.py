#!/usr/bin/env python3
"""Plot maximum consecutive freeze days for an events.yaml case.

A freeze day is defined as a day where the daily-min 2m
temperature falls below the 15th-percentile climatology. For
each land grid point the longest run of consecutive freeze days
within the case window is plotted. Grid points with fewer than
3 consecutive days are masked.

Usage:
    python plot_consecutive_freeze_days.py --case-id-number 5
    python plot_consecutive_freeze_days.py --case-id-number 5 \\
        --output my_plot.png
"""

import argparse
import logging
import sys
import time as time_module

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import regionmask
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extremeweatherbench import cases, defaults

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from heat_freeze_bounds_global import (
    _align_climatology,
    detect_time_dim,
    open_era5_t2m,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3


def max_consecutive_days(mask_3d: np.ndarray) -> np.ndarray:
    """Max consecutive True days per grid point (axis 0)."""
    nt, nlat, nlon = mask_3d.shape
    flat = mask_3d.reshape(nt, -1).astype(np.int8)
    result = np.zeros(flat.shape[1], dtype=np.int32)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        result[k] = int((ends - starts).max())
    return result.reshape(nlat, nlon)


def load_case(case_id: int) -> cases.IndividualCase:
    """Load a single case from events.yaml by id."""
    all_cases = cases.load_ewb_events_yaml_into_case_list()
    for c in all_cases:
        if c.case_id_number == case_id:
            return c
    raise ValueError(
        f"Case {case_id} not found. Available freeze ids: "
        + ", ".join(
            str(c.case_id_number) for c in all_cases if c.event_type == "freeze"
        )
    )


def compute_consecutive_field(
    single_case: cases.IndividualCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (consecutive_days, lats, lons) for a case."""
    start = str(single_case.start_date.date())
    end = str(single_case.end_date.date())
    bounds = single_case.location.as_geopandas().total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds

    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360

    logger.info("Opening ERA5 for %s to %s ...", start, end)
    t2m = open_era5_t2m(start, end)
    t2m = t2m.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    logger.info("  Subset: %s", dict(t2m.sizes))

    tdim = detect_time_dim(t2m)
    daily_min = t2m.resample({tdim: "1D"}).min()

    logger.info("Loading 15th-percentile climatology...")
    clim_15 = (
        defaults.get_climatology(0.15)
        .min(
            dim="hour",
        )
        .sortby("latitude")
    )
    clim_15 = clim_15.sel(
        latitude=daily_min.latitude,
        longitude=daily_min.longitude,
        method="nearest",
    )

    clim_aligned = _align_climatology(
        clim_15,
        daily_min,
        tdim,
    )

    logger.info("Building land mask...")
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask_raw = land.mask(
        daily_min.longitude,
        daily_min.latitude,
    )
    land_mask = land_mask_raw == 0

    logger.info("Computing exceedance mask...")
    fz_mask = ((daily_min < clim_aligned) & land_mask).compute().values.astype(bool)

    logger.info("Computing max consecutive days...")
    consec = max_consecutive_days(fz_mask)
    lats = daily_min.latitude.values
    lons = daily_min.longitude.values
    return consec, lats, lons


def to_plot_lon(lon: float) -> float:
    """Convert 0-360 longitude to -180..180 for PlateCarree."""
    return lon - 360 if lon > 180 else lon


def plot_consecutive(
    consec: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    single_case: cases.IndividualCase,
    output: str,
) -> None:
    """Create the consecutive freeze days map."""
    plot_data = consec.astype(float)
    plot_data[consec < MIN_CONSECUTIVE_DAYS] = np.nan

    valid = plot_data[~np.isnan(plot_data)]
    vmax = int(valid.max()) if valid.size > 0 else 10

    bounds = single_case.location.as_geopandas().total_bounds
    lon_min = to_plot_lon(bounds[0])
    lon_max = to_plot_lon(bounds[2])
    lat_min, lat_max = bounds[1], bounds[3]
    plot_lons = np.array([to_plot_lon(x) for x in lons])

    n_levels = vmax + 1
    cmap = plt.colormaps.get_cmap("Blues").resampled(n_levels)
    bin_edges = np.arange(-0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(8, 8),
    )
    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="lightblue",
        zorder=0,
    )

    im = ax.pcolormesh(
        plot_lons,
        lats,
        plot_data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    ax.coastlines(linewidth=0.8)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.6,
        edgecolor="grey",
    )
    ax.add_feature(
        cfeature.STATES,
        linewidth=0.4,
        edgecolor="lightgrey",
    )
    ax.add_feature(
        cfeature.LAKES,
        linewidth=0.4,
        edgecolor="blue",
        facecolor="none",
    )
    ax.add_feature(
        cfeature.RIVERS,
        linewidth=0.3,
        edgecolor="blue",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="5%",
        pad=0.1,
        axes_class=plt.Axes,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(range(0, vmax + 1))
    cbar.set_label("Consecutive Days", fontsize=12)

    start_str = str(single_case.start_date.date())
    end_str = str(single_case.end_date.date())
    ax.set_title(
        f"Consecutive Freeze Days\n{start_str} to {end_str}",
        loc="left",
        fontsize=13,
    )

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", output)


def main():
    parser = argparse.ArgumentParser(
        description="Plot consecutive freeze days for a case.",
    )
    parser.add_argument(
        "--case-id-number",
        type=int,
        required=True,
        help="case_id_number from events.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Output PNG path (default: case_N_consecutive_freeze_days.png)"),
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"case_{args.case_id_number}_consecutive_freeze_days.png"

    t0 = time_module.time()
    single_case = load_case(args.case_id_number)
    logger.info(
        "Case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    consec, lats, lons = compute_consecutive_field(single_case)
    plot_consecutive(
        consec,
        lats,
        lons,
        single_case,
        args.output,
    )
    logger.info("Done in %.1f s", time_module.time() - t0)


if __name__ == "__main__":
    main()
