"""General plotting utilities for weather data visualization.

This module provides common plotting functions that can be reused across
different types of weather plots including atmospheric rivers, severe
convection, tropical cyclones, etc.
"""

import logging
from typing import Dict, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def convert_longitude_for_plotting(lon_data: np.ndarray) -> np.ndarray:
    """Convert longitude from 0-360 to -180-180 for plotting.

    Args:
        lon_data: Longitude array in 0-360 format.

    Returns:
        Longitude array in -180-180 format.
    """
    return np.where(lon_data > 180, lon_data - 360, lon_data)


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


def add_geographic_features(
    ax,
    alpha: float = 0.7,
    coastline_width: float = 0.8,
    border_width: float = 0.5,
    state_width: float = 0.5,
    water_alpha: float = 0.3,
    include_land_ocean: bool = False,
    land_ocean_alpha: float = 0.1,
) -> None:
    """Add standard geographic features to a cartopy axis.

    Args:
        ax: Cartopy axis to add features to.
        alpha: Transparency for state boundaries.
        coastline_width: Line width for coastlines.
        border_width: Line width for country borders.
        state_width: Line width for state boundaries.
        water_alpha: Transparency for lakes and rivers.
        include_land_ocean: Whether to add land/ocean background.
        land_ocean_alpha: Transparency for land/ocean background.
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=coastline_width)
    ax.add_feature(cfeature.BORDERS, linewidth=border_width)
    ax.add_feature(cfeature.STATES, linewidth=state_width, alpha=alpha)
    ax.add_feature(cfeature.LAKES, alpha=water_alpha)
    ax.add_feature(cfeature.RIVERS, alpha=water_alpha)

    if include_land_ocean:
        ax.add_feature(cfeature.LAND, alpha=land_ocean_alpha)
        ax.add_feature(cfeature.OCEAN, alpha=land_ocean_alpha)


def setup_gridlines(
    ax,
    show_left_labels: bool = True,
    show_bottom_labels: bool = True,
    show_top_labels: bool = False,
    show_right_labels: bool = False,
    alpha: float = 0.5,
    linestyle: str = "--",
    number_format: str = "02.1f",
) -> None:
    """Setup gridlines with custom formatting.

    Args:
        ax: Cartopy axis to add gridlines to.
        show_left_labels: Whether to show labels on the left side.
        show_bottom_labels: Whether to show labels on the bottom.
        show_top_labels: Whether to show labels on the top.
        show_right_labels: Whether to show labels on the right.
        alpha: Transparency of gridlines.
        linestyle: Style of gridlines.
        number_format: Format string for coordinate labels.
    """
    gl = ax.gridlines(
        draw_labels=True,
        alpha=alpha,
        linestyle=linestyle,
        x_inline=False,
        y_inline=False,
        crs=ccrs.PlateCarree(),
    )
    gl.top_labels = show_top_labels
    gl.right_labels = show_right_labels
    gl.left_labels = show_left_labels
    gl.bottom_labels = show_bottom_labels
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(
        dms=False, number_format=number_format
    )
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(number_format=number_format)


def create_custom_colormap_with_transparent_low(
    base_cmap_name: str, levels: np.ndarray, transparent_below_index: int = 0
) -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    """Create a custom colormap with transparent values below a threshold.

    Args:
        base_cmap_name: Name of the base matplotlib colormap.
        levels: Array of contour levels.
        transparent_below_index: Index below which values are transparent.

    Returns:
        Tuple of (custom_colormap, boundary_normalization).
    """
    base_cmap = plt.cm.get_cmap(base_cmap_name)
    colors_list = base_cmap(np.linspace(0, 1, len(levels) - 1))

    # Set colors below threshold to transparent
    for i in range(transparent_below_index + 1):
        if i < len(colors_list):
            colors_list[i] = [1, 1, 1, 0]  # Transparent white

    cmap_custom = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(levels, cmap_custom.N)
    return cmap_custom, norm


def setup_figure_with_colorbar_layout(
    n_cols: int,
    figsize_per_panel: Tuple[float, float] = (4, 6),
    include_legend: bool = True,
    include_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.GridSpec]:
    """Setup figure with proper layout for multi-panel plots with colorbar.

    Args:
        n_cols: Number of columns (panels).
        figsize_per_panel: Size per panel (width, height).
        include_legend: Whether to include space for legend.
        include_colorbar: Whether to include space for colorbar.

    Returns:
        Tuple of (figure, gridspec) for creating subplots.
    """
    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1]

    # Calculate height ratios
    height_ratios = [1.0]  # Main plots
    if include_legend:
        height_ratios.append(0.06)  # Legend row
        fig_height += 0.4
    if include_colorbar:
        height_ratios.append(0.04)  # Colorbar row
        fig_height += 0.3

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(
        len(height_ratios),
        n_cols,
        height_ratios=height_ratios,
        hspace=0.02,
        top=0.92,
        bottom=0.12,
    )

    return fig, gs


def add_horizontal_legend(
    fig: plt.Figure,
    gs: plt.GridSpec,
    legend_elements: List[plt.Line2D],
    row_index: int,
    fontsize: int = 10,
) -> None:
    """Add a horizontal legend spanning all columns.

    Args:
        fig: Figure to add legend to.
        gs: GridSpec for layout.
        legend_elements: List of legend elements.
        row_index: Row index in the gridspec for the legend.
        fontsize: Font size for legend text.
    """
    if legend_elements:
        legend_ax = fig.add_subplot(gs[row_index, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_elements,
            loc="center",
            ncol=len(legend_elements),
            frameon=False,
            fontsize=fontsize,
        )


def add_horizontal_colorbar(
    fig: plt.Figure,
    gs: plt.GridSpec,
    mappable,
    row_index: int,
    label: str,
    extend: str = "max",
    fontsize: int = 12,
):
    """Add a horizontal colorbar spanning all columns.

    Args:
        fig: Figure to add colorbar to.
        gs: GridSpec for layout.
        mappable: Mappable object (e.g., contour plot result).
        row_index: Row index in the gridspec for the colorbar.
        label: Label for the colorbar.
        extend: Colorbar extension ('max', 'min', 'both', 'neither').
        fontsize: Font size for colorbar label.

    Returns:
        The created colorbar object.
    """
    cbar_ax = fig.add_subplot(gs[row_index, :])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal", extend=extend)
    cbar.set_label(label, fontsize=fontsize)
    return cbar


def set_axis_extent_from_bbox(
    ax, bbox: Dict[str, float], crs: ccrs.Projection = ccrs.PlateCarree()
) -> None:
    """Set axis extent from bounding box, handling longitude conversion.

    Args:
        ax: Cartopy axis to set extent for.
        bbox: Bounding box with latitude_min/max, longitude_min/max keys.
        crs: Coordinate reference system for the extent.
    """
    lon_min, lon_max = convert_bbox_longitude(bbox)
    ax.set_extent(
        [lon_min, lon_max, bbox["latitude_min"], bbox["latitude_max"]],
        crs=crs,
    )


def generate_extent(
    center_point: Tuple[float, float],
    zoom: float,
    aspect_ratio: Tuple[float, float],
    out_crs: ccrs.Projection = ccrs.Mercator(),
) -> Tuple[float, float, float, float]:
    """Generate extent from central location and zoom level.

    Args:
        center_point: Center of the map as (longitude, latitude).
        zoom: Zoom level [0 to 10].
        aspect_ratio: Aspect ratio (width, height).
        out_crs: Output coordinate reference system for extent values.

    Returns:
        Tuple of (lon_min, lon_max, lat_min, lat_max).
    """
    mercator_crs = ccrs.Mercator()

    # Define zoom scaling
    zoom_coefficient = 2

    # Calculate minimum and maximum longitude
    lon_min, lon_max = (
        center_point[0] - (zoom_coefficient * zoom),
        center_point[0] + (zoom_coefficient * zoom),
    )

    # Transform map center to specified crs (default to Mercator)
    c_mercator = mercator_crs.transform_point(*center_point, src_crs=ccrs.PlateCarree())

    # Transform longitude bounds to specified crs
    lon_min_mercator = mercator_crs.transform_point(
        lon_min, center_point[1], src_crs=ccrs.PlateCarree()
    )[0]
    lon_max_mercator = mercator_crs.transform_point(
        lon_max, center_point[1], src_crs=ccrs.PlateCarree()
    )[0]

    # Calculate latitude bounds using aspect ratio
    lon_distance = lon_max_mercator - lon_min_mercator
    lat_distance = lon_distance * aspect_ratio[1] / aspect_ratio[0]

    lat_max = c_mercator[1] + lat_distance / 2
    lat_min = c_mercator[1] - lat_distance / 2

    # Convert to output coordinate system if needed
    if out_crs != ccrs.Mercator():
        lon_min_out, lat_min_out = out_crs.transform_point(
            lon_min_mercator, lat_min, src_crs=mercator_crs
        )
        lon_max_out, lat_max_out = out_crs.transform_point(
            lon_max_mercator, lat_max, src_crs=mercator_crs
        )
        return lon_min_out, lon_max_out, lat_min_out, lat_max_out

    return lon_min_mercator, lon_max_mercator, lat_min, lat_max
