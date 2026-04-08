#!/usr/bin/env python3
"""Plotting utilities and CLI for heat wave and cold snap events.

Handles both event types from a single entry point. The event type
is auto-detected from events.yaml; no flag required.

Exported functions used by heat_cold_bounds_global.py and
heat_cold_bounds_case.py:
    max_consecutive_days  -- compute field from boolean mask
    plot_consecutive_map  -- pcolormesh map (Reds/Blues, discrete)

CLI usage:
    python plot_temperature_events.py --case-id-number 2
    python plot_temperature_events.py --case-id-number 30 --output my_plot.png
"""

import argparse
import logging
import operator as op_module
import pathlib
import time as time_module
from typing import Callable, Literal, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extremeweatherbench import cases, defaults, inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3

_OP_MAP: dict[str, Callable] = {
    ">": op_module.gt,
    ">=": op_module.ge,
    "<": op_module.lt,
    "<=": op_module.le,
    "==": op_module.eq,
}

VALID_QUANTILES = [0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90]


def resolve_op(op_str: str) -> Callable:
    """Convert a string operator to a callable.

    Args:
        op_str: One of ">", ">=", "<", "<=", "==".

    Returns:
        The corresponding operator callable.

    Raises:
        ValueError: If op_str is not a recognised operator.
    """
    if op_str not in _OP_MAP:
        raise ValueError(f"Unknown operator {op_str!r}; choose from {list(_OP_MAP)}")
    return _OP_MAP[op_str]


# ── shared ERA5 utilities ─────────────────────────────────────────────


def detect_time_dim(obj: xr.Dataset | xr.DataArray) -> str:
    """Return the name of the time dimension.

    Args:
        obj: An xarray Dataset or DataArray.

    Returns:
        The name of the first matching time dimension.

    Raises:
        ValueError: If no recognised time dimension is found.
    """
    for name in ("valid_time", "time"):
        if name in obj.dims:
            return name
    raise ValueError(f"No time dimension found. Available: {list(obj.dims)}")


def open_era5_t2m(start_date: str, end_date: str) -> xr.DataArray:
    """Open ERA5 2m temperature lazily for a date range.

    Selects 6-hourly timesteps (0/6/12/18 UTC) to match the
    climatology base and sorts latitude to ascending order.

    Args:
        start_date: Inclusive start date string (YYYY-MM-DD).
        end_date: Inclusive end date string (YYYY-MM-DD).

    Returns:
        Lazy DataArray of 2m temperature with ascending latitude.
    """
    ds = xr.open_zarr(
        inputs.ARCO_ERA5_FULL_URI,
        storage_options={"token": "anon"},
        chunks=None,
    )
    tdim = detect_time_dim(ds)
    t2m = ds["2m_temperature"].sel({tdim: slice(start_date, end_date)})
    six_hourly = t2m[tdim].dt.hour.isin([0, 6, 12, 18])
    t2m = t2m.sel({tdim: six_hourly})
    t2m = t2m.chunk({"latitude": -1, "longitude": -1})
    return t2m.sortby("latitude")


def _align_climatology(
    clim: xr.DataArray,
    daily: xr.DataArray,
    tdim: str,
) -> xr.DataArray:
    """Align a dayofyear-indexed climatology to daily ERA5 data.

    Computes the climatology into memory (366 x lat x lon) to avoid
    dask chunk multiplication, then indexes by dayofyear via numpy.

    Args:
        clim: Climatology DataArray indexed by dayofyear.
        daily: Daily ERA5 DataArray to align against.
        tdim: Name of the time dimension in daily.

    Returns:
        DataArray with the same shape as daily, containing the
        climatology value for each timestep's day-of-year.
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

    return xr.DataArray(selected, dims=daily.dims, coords=daily.coords)


# ── constants for bounding-box overview ──────────────────────────────

_HW_COLOR = "#d73027"
_FZ_COLOR = "#4575b4"
_BOX_ALPHA = 0.35
_BOX_EDGE_ALPHA = 0.85


# ── shared helpers ───────────────────────────────────────────────────


def _to_plot_lon(lon: float) -> float:
    """Wrap 0-360 longitude to -180..180 for PlateCarree.

    Args:
        lon: Longitude in 0-360 degrees.

    Returns:
        Longitude in -180..180 degrees.
    """
    return lon - 360.0 if lon > 180.0 else lon


def max_consecutive_days(mask_3d: np.ndarray) -> np.ndarray:
    """Compute max consecutive True days per grid point.

    Args:
        mask_3d: Boolean array of shape (time, lat, lon).

    Returns:
        Int32 array of shape (lat, lon) with the maximum number of
        consecutive True values along axis 0 for each grid point.
    """
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


def _add_map_features(ax) -> None:
    """Add standard cartopy features to an axis.

    Args:
        ax: A cartopy GeoAxes instance.
    """
    ax.coastlines(linewidth=0.5, zorder=10)
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
        linewidth=0.3,
        edgecolor="none",
        facecolor="none",
    )
    ax.add_feature(
        cfeature.RIVERS,
        linewidth=0.2,
        edgecolor="none",
    )


# ── consecutive-days map ─────────────────────────────────────────────


def plot_consecutive_map(
    consec: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    event_type: Literal["heat_wave", "cold_snap"],
    title: str,
    output_path: str,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """Plot a max-consecutive-days field using the standard style.

    Args:
        consec: int32 2-D array (nlat, nlon) of max consecutive days.
        lats: 1-D latitude array matching consec rows.
        lons: 1-D longitude array (0-360) matching consec columns.
        event_type: ``'heat_wave'`` or ``'cold_snap'``.
        title: Two-line figure title (left-aligned).
        output_path: Destination PNG file path.
        extent: (lon_min, lon_max, lat_min, lat_max) in -180..180.
            Derived from lats/lons when None. Default is None.
    """
    plot_data = consec.astype(float)
    plot_data[consec < MIN_CONSECUTIVE_DAYS] = np.nan

    valid = plot_data[~np.isnan(plot_data)]
    vmax = int(valid.max()) if valid.size > 0 else MIN_CONSECUTIVE_DAYS

    cmap_name = "Reds" if event_type == "heat_wave" else "Blues"
    n_levels = vmax - MIN_CONSECUTIVE_DAYS + 1
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(n_levels)
    bin_edges = np.arange(MIN_CONSECUTIVE_DAYS - 0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

    plot_lons = np.array([_to_plot_lon(x) for x in lons])

    if extent is None:
        lon_min, lon_max = float(plot_lons.min()), float(plot_lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
    else:
        lon_min, lon_max, lat_min, lat_max = extent

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(10, 8),
    )
    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)

    im = ax.pcolormesh(
        plot_lons,
        lats,
        plot_data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    _add_map_features(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="5%",
        pad=0.1,
        axes_class=plt.Axes,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(range(MIN_CONSECUTIVE_DAYS, vmax + 1))
    cbar.set_label("Consecutive Days", fontsize=12)

    ax.set_title(title, loc="left", fontsize=13)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved consecutive-days plot to %s", output_path)


# ── bounding-box overview ─────────────────────────────────────────────


def plot_event_bounds(
    df: pd.DataFrame,
    csv_path: str,
    title: str = "Detected Heat Wave and Cold Snap Events",
) -> None:
    """Draw bounding boxes for all events on a global Robinson map.

    Saves a PNG alongside the CSV with the same stem.

    Args:
        df: DataFrame with columns event_type, longitude_min,
            longitude_max, latitude_min, latitude_max.
        csv_path: Path to the output CSV; the PNG is saved with the
            same stem.
        title: Figure title. Default is "Detected Heat Wave and
            Cold Snap Events".
    """
    if df.empty:
        logger.warning("No events to plot -- skipping bounds plot.")
        return

    out_path = str(pathlib.Path(csv_path).with_suffix(".png"))
    hw_count = int((df["event_type"] == "heat_wave").sum())
    fz_count = int((df["event_type"] == "cold_snap").sum())

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(14, 7),
    )
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.3,
        edgecolor="grey",
        zorder=2,
    )

    for _, row in df.iterrows():
        lon0 = _to_plot_lon(float(row["longitude_min"]))
        lon1 = _to_plot_lon(float(row["longitude_max"]))
        lat0, lat1 = float(row["latitude_min"]), float(row["latitude_max"])
        if lon1 < lon0:
            lon1 += 360.0
        color = _HW_COLOR if row["event_type"] == "heat_wave" else _FZ_COLOR
        kw = dict(
            width=lon1 - lon0,
            height=lat1 - lat0,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(
            mpatches.Rectangle(
                xy=(lon0, lat0),
                facecolor=color,
                edgecolor=color,
                alpha=_BOX_ALPHA,
                linewidth=1.0,
                zorder=3,
                **kw,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                xy=(lon0, lat0),
                facecolor="none",
                edgecolor=color,
                alpha=_BOX_EDGE_ALPHA,
                linewidth=1.0,
                zorder=4,
                **kw,
            )
        )

    legend_handles = []
    if hw_count:
        legend_handles.append(
            mpatches.Patch(
                facecolor=_HW_COLOR,
                alpha=0.7,
                label=f"Heat wave ({hw_count})",
            )
        )
    if fz_count:
        legend_handles.append(
            mpatches.Patch(
                facecolor=_FZ_COLOR,
                alpha=0.7,
                label=f"Cold Snap ({fz_count})",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=10,
            framealpha=0.85,
        )

    ax.set_title(title, loc="left", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bounds plot to %s", out_path)


# ── case-level data loading ───────────────────────────────────────────


def load_case(case_id_number: int) -> cases.IndividualCase:
    """Load a single case from events.yaml by case_id_number.

    Args:
        case_id_number: Integer identifier for the case.

    Returns:
        The matching IndividualCase object.

    Raises:
        ValueError: If no case with the given ID exists.
    """
    all_cases = cases.load_ewb_events_yaml_into_case_list()
    for c in all_cases:
        if c.case_id_number == case_id_number:
            return c
    hw_ids = ", ".join(
        str(c.case_id_number) for c in all_cases if c.event_type == "heat_wave"
    )
    cs_ids = ", ".join(
        str(c.case_id_number) for c in all_cases if c.event_type == "cold_snap"
    )
    raise ValueError(
        f"Case {case_id_number} not found.\n"
        f"  heat_wave ids: {hw_ids}\n"
        f"  cold_snap ids:    {cs_ids}"
    )


def compute_consecutive_field(
    single_case: cases.IndividualCase,
    quantile: float | None = None,
    op_str: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (consecutive_days, lats, lons) for a case.

    Args:
        single_case: The case to compute the consecutive field for.
        quantile: Climatology quantile. Default is None, which
            resolves to 0.85 for heat_wave or 0.15 for
            freeze/cold_snap.
        op_str: Comparison operator string (e.g. ">", ">=",
            "<", "<="). Default is None, which resolves to ">"
            for heat_wave or "<" for freeze.

    Returns:
        A tuple of (consec, lats, lons) where consec is an int32
        array of shape (lat, lon) containing max consecutive event
        days, lats is the 1-D latitude array, and lons is the 1-D
        longitude array (0-360).
    """
    is_heatwave = single_case.event_type == "heat_wave"
    if quantile is None:
        quantile = 0.85 if is_heatwave else 0.15
    if op_str is None:
        op_str = ">" if is_heatwave else "<"
    cmp = resolve_op(op_str)

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

    logger.info(
        "Loading q=%.2f climatology (op=%s)...",
        quantile,
        op_str,
    )
    clim = defaults.get_climatology(quantile).sortby("latitude")

    clim = clim.sel(
        latitude=t2m.latitude,
        longitude=t2m.longitude,
        method="nearest",
    )

    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    max_clim_doy = int(clim.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    clim_aligned = clim.sel(
        dayofyear=doy_capped,
        hour=hour,
    )

    logger.info("Computing 6-hourly exceedance...")
    exc_6h = cmp(t2m, clim_aligned)

    daily_all_pass = exc_6h.resample({tdim: "1D"}).min().astype(bool)

    logger.info("Building land mask...")
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask(daily_all_pass.longitude, daily_all_pass.latitude) == 0

    exc = daily_all_pass & land_mask
    mask_np = exc.compute().values.astype(bool)

    logger.info("Computing max consecutive days...")
    consec = max_consecutive_days(mask_np)
    return consec, daily_all_pass.latitude.values, daily_all_pass.longitude.values


# ── CLI ───────────────────────────────────────────────────────────────


def _default_output(case_id_number: int, event_type: str) -> str:
    kind = "heatwave" if event_type == "heat_wave" else "cold_snap"
    return f"case_{case_id_number}_consecutive_{kind}_days.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot consecutive heatwave or cold_snap days for a case from events.yaml."
        ),
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
        help=(
            "Output PNG path "
            "(default: case_N_consecutive_{heatwave|cold_snap}_days.png)"
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help=(
            "Climatology quantile "
            f"(one of {VALID_QUANTILES}; "
            "default: 0.85 for heat_wave, 0.15 for freeze)"
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Comparison operator string "
            "(>, >=, <, <=, ==; "
            "default: > for heat_wave, < for freeze)"
        ),
    )
    args = parser.parse_args()

    t0 = time_module.time()
    single_case = load_case(args.case_id_number)
    logger.info(
        "Case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    if args.output is None:
        args.output = _default_output(
            args.case_id_number,
            single_case.event_type,
        )

    consec, lats, lons = compute_consecutive_field(
        single_case,
        quantile=args.quantile,
        op_str=args.operator,
    )

    kind = "Heat Wave" if single_case.event_type == "heat_wave" else "Cold Snap"
    start_str = str(single_case.start_date.date())
    end_str = str(single_case.end_date.date())

    plot_consecutive_map(
        consec,
        lats,
        lons,
        cast(Literal["heat_wave", "cold_snap"], single_case.event_type),
        title=f"Consecutive {kind} Days\n{start_str} to {end_str}",
        output_path=args.output,
    )
    logger.info("Done in %.1f s", time_module.time() - t0)


if __name__ == "__main__":
    main()
