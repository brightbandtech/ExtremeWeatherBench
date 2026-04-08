"""Animate daily heat-wave and cold-snap exceedance masks from ERA5.

Produces two animated GIFs — one for heat waves, one for cold snaps —
showing the filtered (3+ consecutive day) exceedance mask day by day on
a global Robinson projection.

Usage:
    python animate_heat_cold_global.py \\
        --start-date 2020-01-01 --end-date 2020-03-01 \\
        --fps 4 --n-workers 4 --quantile-lower 0.85 --quantile-upper 0.15
        --output events_global.gif
"""

import argparse
import logging
import time as time_module

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as mpl_anim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster
from heat_cold_bounds_global import (
    apply_consecutive_filter,
    build_exceedance_mask,
    build_land_mask,
    get_climatology_bounds,
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

_EVENT_COLOR = "#d73027"


def animate_exceedance(
    filt_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lower_quantile: float | None,
    upper_quantile: float | None,
    output_path: str,
    fps: int = 4,
) -> None:
    """Save an animated GIF of the daily filtered exceedance mask.

    Each frame shows one day's exceedance footprint on a global
    Robinson projection. Active grid points are colored red; inactive land and ocean are rendered
    in whitesmoke and light blue respectively.

    Args:
        filt_mask: Boolean array (time, lat, lon) with the
            consecutive-day filter already applied.
        dates: 1-D datetime64 array aligned with axis 0 of
            filt_mask.
        lats: 1-D latitude array (ascending).
        lons: 1-D longitude array (0-360 degrees).
        lower_quantile: Lower quantile for the event.
        upper_quantile: Upper quantile for the event.
        output_path: Destination ``.gif`` file path.
        fps: Frames per second for the output GIF. Default 4.
    """
    n_days = filt_mask.shape[0]

    rgba = mcolors.to_rgba(_EVENT_COLOR)
    cmap = mcolors.ListedColormap(["none", rgba])

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(12, 5.5),
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
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
    if lower_quantile is not None and upper_quantile is not None:
        kind = f"p{lower_quantile:.2f}-p{upper_quantile:.2f}"
    elif lower_quantile is not None:
        kind = f"p{lower_quantile:.2f}+"
    else:
        kind = f"p{upper_quantile:.2f}-"
    date_str = str(dates[0])[:10]
    title = ax.set_title(
        f"{kind} {date_str}",
        loc="left",
        fontsize=13,
    )

    def _update(di: int):
        mesh.set_array(filt_mask[di].astype(float).ravel())
        title.set_text(f"{kind} {str(dates[di])[:10]}")
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
            "Animate daily exceedance masks "
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
    parser.add_argument(
        "--quantile-lower",
        type=float,
        default=None,
        help="Lower quantile for the event",
    )
    parser.add_argument(
        "--quantile-upper",
        type=float,
        default=None,
        help="Upper quantile for the event",
    )
    parser.add_argument(
        "--output",
        default="events_global.gif",
        help="Output GIF path",
    )
    args = parser.parse_args()

    wall_start = time_module.time()
    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info("Loading climatology thresholds...")
    clim_lower, clim_upper = get_climatology_bounds(q_lower=args.quantile_lower, q_upper=args.quantile_upper)

    logger.info("Building land mask...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)

    logger.info("Building exceedance masks (lazy)...")
    event_lazy = build_exceedance_mask(t2m, clim_lower=clim_lower, clim_upper=clim_upper, land_mask=land_mask)

    tdim = detect_time_dim(event_lazy)

    logger.info("Computing event mask...")
    t0 = time_module.time()
    event_da = event_lazy.compute()
    logger.info("  done in %.1f s", time_module.time() - t0)

    dates = event_da[tdim].values
    lats = event_da.latitude.values
    lons = event_da.longitude.values
    event_np = event_da.values.astype(bool)
    del event_da

    logger.info("Applying 3-day consecutive filter...")
    event_filt = apply_consecutive_filter(event_np)
    logger.info(
        "  HW active cells: %d -> %d",
        event_np.sum(),
        event_filt.sum(),
    )
    del event_np

    animate_exceedance(
        event_filt,
        dates,
        lats,
        lons,
        lower_quantile=args.quantile_lower,
        upper_quantile=args.quantile_upper,
        output_path=args.output,
        fps=args.fps,
    )
    client.close()
    logger.info("Done in %.1f s", time_module.time() - wall_start)


if __name__ == "__main__":
    main()
