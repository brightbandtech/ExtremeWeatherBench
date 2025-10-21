"""Plotting functions for severe convection forecasts and analysis.

This module provides standardized plotting functions for severe convection
parameters like Craven-Brooks Significant Severe (CBSS) index, with support
for forecast data, observations, and verification metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def setup_cbss_colormap_and_levels() -> Tuple[
    colors.ListedColormap, colors.BoundaryNorm, np.ndarray
]:
    """Setup colormap and normalization for CBSS plotting.

    Returns:
        Tuple of (colormap, normalization, levels) for CBSS plotting.
        Levels based on thresholds: < 10,000 (Low/transparent),
        10,000-22,500 (Marginal), > 22,500 (Significant).
    """
    levels = np.array([0, 10000, 15000, 22500, 30000, 50000, 75000])
    cmap = plt.cm.YlOrRd
    colors_list = cmap(np.linspace(0, 1, len(levels) - 1))
    colors_list[0] = [1, 1, 1, 0]  # Set first color to transparent white
    cmap_custom = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(levels, cmap_custom.N)
    return cmap_custom, norm, levels


def setup_pph_colormap_and_levels() -> Tuple[
    plt.cm.ScalarMappable, colors.BoundaryNorm, List[float]
]:
    """Setup colormap and normalization for PPH plotting.

    Returns:
        Tuple of (colormap, normalization, levels) for PPH plotting.
    """
    pph_levels = [0.01, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75]
    pph_cmap = plt.cm.viridis
    pph_norm = colors.BoundaryNorm(pph_levels, pph_cmap.N)
    return pph_cmap, pph_norm, pph_levels


def convert_longitude_for_plotting(lon_data: np.ndarray) -> np.ndarray:
    """Convert longitude from 0-360 to -180-180 for plotting.

    Args:
        lon_data: Longitude array in 0-360 format.

    Returns:
        Longitude array in -180-180 format.
    """
    return np.where(lon_data > 180, lon_data - 360, lon_data)


def add_geographic_features(ax, alpha: float = 0.7) -> None:
    """Add standard geographic features to a cartopy axis.

    Args:
        ax: Cartopy axis to add features to.
        alpha: Transparency for some features.
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=alpha)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)


def setup_gridlines(ax, show_left_labels: bool = True) -> None:
    """Setup gridlines with custom formatting.

    Args:
        ax: Cartopy axis to add gridlines to.
        show_left_labels: Whether to show labels on the left side.
    """
    gl = ax.gridlines(
        draw_labels=True,
        alpha=0.5,
        linestyle="--",
        x_inline=False,
        y_inline=False,
        crs=ccrs.PlateCarree(),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(
        dms=False, number_format="02.1f"
    )
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(number_format="02.1f")
    if not show_left_labels:
        gl.left_labels = False


def convert_bbox_longitude(bbox: Dict[str, float]) -> Tuple[float, float]:
    """Convert bounding box longitude from 0-360 to -180-180.

    Args:
        bbox: Bounding box dictionary with longitude_min/max keys.

    Returns:
        Tuple of (lon_min, lon_max) in -180-180 format.
    """
    lon_min = (
        bbox["longitude_min"] - 360
        if bbox["longitude_min"] > 180
        else bbox["longitude_min"]
    )
    lon_max = (
        bbox["longitude_max"] - 360
        if bbox["longitude_max"] > 180
        else bbox["longitude_max"]
    )
    return lon_min, lon_max


def plot_storm_reports(
    ax,
    tornado_reports: Optional[pd.DataFrame] = None,
    hail_reports: Optional[pd.DataFrame] = None,
) -> List[plt.Line2D]:
    """Plot storm reports on the given axis.

    Args:
        ax: Cartopy axis to plot on.
        tornado_reports: DataFrame with tornado report locations.
        hail_reports: DataFrame with hail report locations.

    Returns:
        List of legend elements for the plotted reports.
    """
    legend_elements = []

    if tornado_reports is not None and len(tornado_reports) > 0:
        ax.scatter(
            tornado_reports["longitude"],
            tornado_reports["latitude"],
            c="k",
            s=20,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            marker="^",
            edgecolors="k",
            linewidths=1,
            zorder=10,
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="k",
                markersize=8,
                markeredgecolor="k",
                label="Tornado Reports",
            )
        )

    if hail_reports is not None and len(hail_reports) > 0:
        ax.scatter(
            hail_reports["longitude"],
            hail_reports["latitude"],
            c="green",
            s=20,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            marker="s",
            edgecolors="darkgreen",
            linewidths=1,
            zorder=10,
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="green",
                markersize=8,
                markeredgecolor="darkgreen",
                label="Hail Reports",
            )
        )

    return legend_elements


def plot_pph_contours(
    ax, pph_data: xr.DataArray, pph_cmap, pph_norm, pph_levels: List[float]
) -> List[plt.Line2D]:
    """Plot PPH contours and return legend elements.

    Args:
        ax: Cartopy axis to plot on.
        pph_data: PPH data array.
        pph_cmap: PPH colormap.
        pph_norm: PPH normalization.
        pph_levels: PPH contour levels.

    Returns:
        List of legend elements for PPH contours.
    """
    legend_elements = []

    # Create a mask for values below 0.01
    pph_mask = np.ma.masked_less(pph_data, 0.01)
    pph_cmap_with_alpha = pph_cmap.copy()
    pph_cmap_with_alpha.set_bad("none", alpha=0)

    ax.contour(
        pph_data.longitude,
        pph_data.latitude,
        pph_mask,
        levels=pph_levels,
        transform=ccrs.PlateCarree(),
        cmap=pph_cmap_with_alpha,
        norm=pph_norm,
        extend="both",
        zorder=15,
    )

    # Create legend elements for key PPH levels
    pph_labels = ["PPH 5%", "PPH 15%", "PPH 30%", "PPH 45%", "PPH 60%"]
    for level, label in zip(pph_levels[1:], pph_labels):
        norm_value = pph_norm(level)
        color = pph_cmap(norm_value)
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=1.5, label=label)
        )

    return legend_elements


def plot_cbss_forecast_panel(
    cbss_data: xr.DataArray,
    target_date: pd.Timestamp,
    lead_time_hours: int,
    bbox: Dict[str, float],
    pph_data: Optional[xr.DataArray] = None,
    tornado_reports: Optional[pd.DataFrame] = None,
    hail_reports: Optional[pd.DataFrame] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
) -> Tuple[plt.Figure, plt.Axes, plt.cm.ScalarMappable]:
    """Plot a single CBSS forecast panel.

    Args:
        cbss_data: CBSS data array with lead_time dimension.
        target_date: Forecast initialization time.
        lead_time_hours: Lead time in hours to plot.
        bbox: Bounding box dictionary with lat/lon min/max.
        pph_data: Optional PPH data for contours.
        tornado_reports: Optional tornado report DataFrame.
        hail_reports: Optional hail report DataFrame.
        projection: Cartopy projection to use.

    Returns:
        Tuple of (figure, axis, contour_mappable) for further customization.
    """
    # Setup colormap and levels
    cmap_custom, norm, levels = setup_cbss_colormap_and_levels()

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={"projection": projection})

    # Select data for this lead time
    lead_time_td = pd.Timedelta(hours=lead_time_hours)
    cbss_lt = cbss_data.sel(lead_time=lead_time_td, method="nearest")

    # Convert longitude for plotting
    lon_data = convert_longitude_for_plotting(cbss_lt.longitude.values)

    # Plot CBSS values
    im = ax.contourf(
        lon_data,
        cbss_lt.latitude,
        cbss_lt.values,
        levels=levels,
        cmap=cmap_custom,
        norm=norm,
        extend="max",
        transform=ccrs.PlateCarree(),
    )

    # Add contour lines for key thresholds
    ax.contour(
        lon_data,
        cbss_lt.latitude,
        cbss_lt.values,
        levels=[10000, 22500],
        colors=["black", "darkred"],
        linewidths=[0.5, 0.5],
        linestyles=["-", "-"],
        transform=ccrs.PlateCarree(),
    )

    # Add PPH contours if available
    if pph_data is not None:
        pph_cmap, pph_norm, pph_levels = setup_pph_colormap_and_levels()
        plot_pph_contours(ax, pph_data, pph_cmap, pph_norm, pph_levels)

    # Add geographic features
    add_geographic_features(ax)

    # Plot storm reports
    plot_storm_reports(ax, tornado_reports, hail_reports)

    # Set extent
    lon_min, lon_max = convert_bbox_longitude(bbox)
    ax.set_extent(
        [lon_min, lon_max, bbox["latitude_min"], bbox["latitude_max"]],
        crs=ccrs.PlateCarree(),
    )

    # Add gridlines
    setup_gridlines(ax)

    # Set title
    valid_time = target_date + lead_time_td
    ax.set_title(
        f"CBSS +{lead_time_hours}h\nValid: {valid_time.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=10,
    )

    return fig, ax, im


def plot_cbss_forecast_multipanel(
    cbss_data: xr.DataArray,
    target_date: pd.Timestamp,
    lead_times_hours: List[int],
    bbox: Dict[str, float],
    pph_data: Optional[xr.DataArray] = None,
    tornado_reports: Optional[pd.DataFrame] = None,
    hail_reports: Optional[pd.DataFrame] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    output_filename: Optional[str] = None,
) -> plt.Figure:
    """Create a multi-panel CBSS forecast plot.

    Args:
        cbss_data: CBSS data array with lead_time dimension.
        target_date: Forecast initialization time.
        lead_times_hours: List of lead times in hours to plot.
        bbox: Bounding box dictionary with lat/lon min/max.
        pph_data: Optional PPH data for contours.
        tornado_reports: Optional tornado report DataFrame.
        hail_reports: Optional hail report DataFrame.
        projection: Cartopy projection to use.
        output_filename: Optional filename to save the plot.

    Returns:
        Figure object for further customization.
    """
    # Setup colormaps and levels
    cmap_custom, norm, levels = setup_cbss_colormap_and_levels()

    # Filter available lead times
    available_lead_times = [
        int(pd.Timedelta(lt).total_seconds() / 3600)
        for lt in cbss_data.lead_time.values
    ]
    lead_times_to_plot = [lt for lt in lead_times_hours if lt in available_lead_times]

    n_plots = len(lead_times_to_plot)

    # Create figure with proper layout for colorbar
    fig = plt.figure(figsize=(4 * n_plots, 6.5))

    # Create a grid layout: main plots + legend + colorbar
    gs = fig.add_gridspec(
        3,
        n_plots,
        height_ratios=[1, 0.06, 0.04],
        hspace=0.02,
        top=0.92,
        bottom=0.12,
    )

    # Create subplot axes
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(gs[0, i], projection=projection)
        axes.append(ax)

    # Setup PPH if available
    pph_cmap, pph_norm, pph_levels = setup_pph_colormap_and_levels()

    # Plot each lead time
    im = None  # Will store the last contour plot for colorbar
    all_legend_elements = []

    for i, lead_time_hours in enumerate(lead_times_to_plot):
        lead_time_td = pd.Timedelta(hours=lead_time_hours)
        cbss_lt = cbss_data.sel(lead_time=lead_time_td, method="nearest")
        ax = axes[i]

        # Convert longitude for plotting
        lon_data = convert_longitude_for_plotting(cbss_lt.longitude.values)

        # Plot CBSS values
        im = ax.contourf(
            lon_data,
            cbss_lt.latitude,
            cbss_lt.values,
            levels=levels,
            cmap=cmap_custom,
            norm=norm,
            extend="max",
            transform=ccrs.PlateCarree(),
        )

        # Add contour lines for key thresholds
        ax.contour(
            lon_data,
            cbss_lt.latitude,
            cbss_lt.values,
            levels=[10000, 22500],
            colors=["black", "darkred"],
            linewidths=[0.5, 0.5],
            linestyles=["-", "-"],
            transform=ccrs.PlateCarree(),
        )

        # Add PPH contours if available
        if pph_data is not None:
            pph_legend = plot_pph_contours(ax, pph_data, pph_cmap, pph_norm, pph_levels)
            if i == 0:  # Only add to legend once
                all_legend_elements.extend(pph_legend)

        # Add geographic features
        add_geographic_features(ax)

        # Plot storm reports
        report_legend = plot_storm_reports(ax, tornado_reports, hail_reports)
        if i == 0:  # Only add to legend once
            all_legend_elements.extend(report_legend)

        # Set extent
        lon_min, lon_max = convert_bbox_longitude(bbox)
        ax.set_extent(
            [lon_min, lon_max, bbox["latitude_min"], bbox["latitude_max"]],
            crs=ccrs.PlateCarree(),
        )

        # Add gridlines
        setup_gridlines(ax, show_left_labels=(i == 0))

        # Set title
        valid_time = target_date + lead_time_td
        ax.set_title(
            f"CBSS +{lead_time_hours}h\n"
            f"Valid: {valid_time.strftime('%Y-%m-%d %H:%M')} UTC",
            fontsize=10,
        )

    # Add legend if there are elements
    if all_legend_elements:
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=all_legend_elements,
            loc="center",
            ncol=len(all_legend_elements),
            frameon=False,
            fontsize=10,
        )

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_subplot(gs[2, :])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="max")
        cbar.set_label("Craven-Brooks Significant Severe (m³/s³)", fontsize=12)

    # Add main title
    fig.suptitle(
        f"Craven-Brooks Significant Severe (CBSS) Parameter\n"
        f"HRES | Init: {target_date.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=14,
        y=0.95,
    )

    # Save if filename provided
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved as: {output_filename}")

    return fig


def print_cbss_statistics(cbss_data: xr.DataArray, lead_times_hours: List[int]) -> None:
    """Print CBSS statistics for given lead times.

    Args:
        cbss_data: CBSS data array with lead_time dimension.
        lead_times_hours: List of lead times in hours to analyze.
    """
    logger.info("\n=== CBSS Statistics ===")
    for lead_time_hours in lead_times_hours:
        lead_time_td = pd.Timedelta(hours=lead_time_hours)
        try:
            cbss_lt = cbss_data.sel(lead_time=lead_time_td, method="nearest")
            max_val = float(cbss_lt.max().values)
            mean_val = float(cbss_lt.mean().values)
            logger.info(
                f"Lead time +{lead_time_hours:2d}h: "
                f"Max = {max_val:8.0f}, Mean = {mean_val:6.0f} m³/s³"
            )
        except Exception as e:
            logger.warning(f"Could not compute stats for +{lead_time_hours}h: {e}")
