"""Animate daily heat-wave and cold-snap exceedance masks from ERA5.

Produces two animated GIFs — one for heat waves, one for cold snaps —
showing the filtered (3+ consecutive day) exceedance mask day by day on
a global Robinson projection.

Usage:
    python animate_heat_cold_global.py \\
        --start-date 2020-01-01 --end-date 2020-03-01 \\
        --fps 4 --n-workers 4
"""

import argparse
import logging
import time as time_module
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as mpl_anim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster

from heat_cold_bounds_global import (
    apply_consecutive_filter,
    build_exceedance_masks,
    build_land_mask,
    get_climatology_thresholds,
)
from plot_temperature_events import (
    _add_map_features,
    detect_time_dim,
    open_era5_t2m,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_HW_COLOR = "#d73027"
_FZ_COLOR = "#4575b4"


def animate_exceedance(
    filt_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    event_type: Literal["heat_wave", "cold_snap"],
    output_path: str,
    fps: int = 4,
) -> None:
    """Save an animated GIF of the daily filtered exceedance mask.

    Each frame shows one day's exceedance footprint on a global
    Robinson projection. Active grid points are colored red (heat
    wave) or blue (cold snap); inactive land and ocean are rendered
    in whitesmoke and light blue respectively.

    Args:
        filt_mask: Boolean array (time, lat, lon) with the
            consecutive-day filter already applied.
        dates: 1-D datetime64 array aligned with axis 0 of
            filt_mask.
        lats: 1-D latitude array (ascending).
        lons: 1-D longitude array (0-360 degrees).
        event_type: ``"heat_wave"`` or ``"cold_snap"``.
        output_path: Destination ``.gif`` file path.
        fps: Frames per second for the output GIF. Default 4.
    """
    is_hw = event_type == "heat_wave"
    color = _HW_COLOR if is_hw else _FZ_COLOR
    kind = "Heat Wave" if is_hw else "Cold Snap"
    n_days = filt_mask.shape[0]

    rgba = mcolors.to_rgba(color)
    cmap = mcolors.ListedColormap(["none", rgba])

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(14, 7),
    )
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    _add_map_features(ax)

    data0 = filt_mask[0].astype(float)
    mesh = ax.pcolormesh(
        lons,
        lats,
        data0,
        cmap=cmap,
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )

    date_str = str(dates[0])[:10]
    title = ax.set_title(
        f"{kind} Exceedance \u2014 {date_str}",
        loc="left",
        fontsize=13,
    )

    def _update(di: int):
        mesh.set_array(filt_mask[di].astype(float).ravel())
        title.set_text(
            f"{kind} Exceedance \u2014 {str(dates[di])[:10]}"
        )
        if di % 10 == 0:
            logger.info(
                "  Rendering frame %d / %d  (%s)",
                di + 1,
                n_days,
                str(dates[di])[:10],
            )
        return mesh, title

    anim = mpl_anim.FuncAnimation(
        fig,
        _update,
        frames=n_days,
        interval=1000 // fps,
        blit=False,
    )

    writer = mpl_anim.PillowWriter(fps=fps)
    logger.info("Saving %s animation to %s ...", kind, output_path)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close(fig)
    logger.info("  Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Animate daily heat-wave and cold-snap exceedance masks "
            "from ERA5 reanalysis."
        ),
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
        "--output-heat",
        default=None,
        help=(
            "Output GIF path for heat wave animation "
            "(default: heat_exceedance_<start>_<end>.gif)"
        ),
    )
    parser.add_argument(
        "--output-cold",
        default=None,
        help=(
            "Output GIF path for cold snap animation "
            "(default: cold_exceedance_<start>_<end>.gif)"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for the output GIFs (default: 4)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers (default: 4)",
    )
    args = parser.parse_args()

    if args.output_heat is None:
        args.output_heat = (
            f"heat_exceedance_{args.start_date}_{args.end_date}.gif"
        )
    if args.output_cold is None:
        args.output_cold = (
            f"cold_exceedance_{args.start_date}_{args.end_date}.gif"
        )

    wall_start = time_module.time()
    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info("Loading climatology thresholds...")
    clim_hw, clim_fz, _, _ = get_climatology_thresholds()

    logger.info("Building land mask...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)

    logger.info("Building exceedance masks (lazy)...")
    hw_lazy, fz_lazy = build_exceedance_masks(
        t2m, clim_hw, clim_fz, land_mask
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

    logger.info("Applying 3-day consecutive filter...")
    hw_filt = apply_consecutive_filter(hw_np)
    fz_filt = apply_consecutive_filter(fz_np)
    logger.info(
        "  HW active cells: %d -> %d",
        hw_np.sum(),
        hw_filt.sum(),
    )
    logger.info(
        "  FZ active cells: %d -> %d",
        fz_np.sum(),
        fz_filt.sum(),
    )
    del hw_np, fz_np

    animate_exceedance(
        hw_filt,
        dates,
        lats,
        lons,
        "heat_wave",
        args.output_heat,
        fps=args.fps,
    )
    animate_exceedance(
        fz_filt,
        dates,
        lats,
        lons,
        "cold_snap",
        args.output_cold,
        fps=args.fps,
    )

    client.close()
    logger.info("Done in %.1f s", time_module.time() - wall_start)


if __name__ == "__main__":
    main()
