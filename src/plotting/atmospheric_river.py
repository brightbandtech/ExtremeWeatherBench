import logging

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from extremeweatherbench import utils

from . import plotting

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_ar_mask_animation(
    case_id: int,
    title: str,
    ivt_data: xr.DataArray,
    ar_mask: xr.DataArray,
) -> None:
    """Create an animated plot of AR mask evolution over time.

    Uses the same domain as ax2 in create_case_summary_plot with original
    styling including custom colormap and contour representation.

    Args:
        case_id: Case ID number.
        title: Event title.
        ivt_data: Integrated vapor transport data with time dimension.
        ar_mask: AR mask data with time dimension.
    """
    # Get the time dimension name
    time_dim = "valid_time" if "valid_time" in ivt_data.dims else "time"

    # Create figure and axes matching original styling with tighter layout
    fig = plt.figure(figsize=(16, 9))
    # Adjust subplot parameters to center plot and minimize whitespace
    # Leave space for colorbar on right, but center the main plot area
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # Use general plotting functions for geographic features
    plotting.add_geographic_features(ax, include_land_ocean=True, land_ocean_alpha=0.1)
    # Override borders with custom linestyle
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plotting.setup_gridlines(ax, show_top_labels=False, show_right_labels=False)

    # Set extent to match ax2 domain (same as AR mask extent + 5 degrees)
    first_ar_slice = ar_mask.isel({time_dim: 0})
    center_latitude = (
        first_ar_slice.latitude.min() + first_ar_slice.latitude.max()
    ) / 2
    center_longitude = (
        first_ar_slice.longitude.min() + first_ar_slice.longitude.max()
    ) / 2
    center_point = (
        utils.convert_longitude_to_180(center_longitude.values),
        center_latitude.values,
    )
    lon_min, lon_max, lat_min, lat_max = generate_extent(
        center_point, zoom=8, aspect_ratio=(16, 9), out_crs=ccrs.PlateCarree()
    )

    # Create custom colormap from original code
    cmap_colors = [
        "#ffffff",
        "#bde6fa",
        "#7bbae7",
        "#4892bd",
        "#49ae62",
        "#a7d051",
        "#f9d251",
        "#f7792f",
        "#e43d28",
        "#c11b24",
        "#921318",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cubehelix", cmap_colors)
    bounds = np.arange(0, 1200, 100)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Initialize first frame
    first_time_idx = 0
    ar_slice = ar_mask.isel({time_dim: first_time_idx})
    ivt_slice = ivt_data.isel({time_dim: first_time_idx})
    first_time = ar_mask[time_dim].isel({time_dim: first_time_idx}).values

    # Create initial IVT plot
    im = ax.pcolormesh(
        ivt_slice.longitude,
        ivt_slice.latitude,
        ivt_slice.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )

    # Add AR mask as contour
    _ = ax.contour(
        ar_slice.longitude,
        ar_slice.latitude,
        ar_slice.values,
        levels=[0.5],
        colors="black",
        linewidths=2,
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Integrated Vapor Transport (kgm^-1s^-1)")
    cbar.set_label("Integrated Vapor Transport (kgm^-1s^-1)", size=14)
    cbar.ax.tick_params(labelsize=12)

    # Set initial title matching original format
    _ = ax.set_title(
        f"Case {case_id}: Integrated Vapor Transport and Atmospheric River Mask\n"
        f"{title}\n"
        f"Valid {pd.to_datetime(first_time).strftime('%Y-%m-%d %H:%M')}",
        loc="left",
    )

    def update(frame_idx):
        """Update function for animation."""
        # Clear all previous plots
        ax.clear()

        # Re-add features using general plotting functions
        plotting.add_geographic_features(
            ax, include_land_ocean=True, land_ocean_alpha=0.1
        )
        # Override borders with custom linestyle
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        plotting.setup_gridlines(ax, show_top_labels=False, show_right_labels=False)

        # Reset extent
        ax.set_extent(
            [
                float(first_ar_slice.longitude.min()) - 5,
                float(first_ar_slice.longitude.max()) + 5,
                float(first_ar_slice.latitude.min()) - 5,
                float(first_ar_slice.latitude.max()) + 5,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Get data for this frame
        ar_slice = ar_mask.isel({time_dim: frame_idx})
        ivt_slice = ivt_data.isel({time_dim: frame_idx})
        current_time = ar_mask[time_dim].isel({time_dim: frame_idx}).values

        # Plot IVT background
        im = ax.pcolormesh(
            ivt_slice.longitude,
            ivt_slice.latitude,
            ivt_slice.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
        )

        # Add AR mask as contour
        ax.contour(
            ar_slice.longitude,
            ar_slice.latitude,
            ar_slice.values,
            levels=[0.5],
            colors="black",
            linewidths=2,
            transform=ccrs.PlateCarree(),
        )

        # Update title
        ax.set_title(
            f"Case {case_id}: Integrated Vapor Transport and Atmospheric River Mask\n"
            f"{title}\n"
            f"Valid {pd.to_datetime(current_time).strftime('%Y-%m-%d %H:%M')}",
            loc="left",
        )

        return [im]

    # Create animation
    num_frames = len(ar_mask[time_dim])
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(num_frames),
        interval=200,  # 200ms between frames like original
        blit=False,
        repeat=True,
    )

    # Save animation
    animation_filename = f"case_{case_id:03d}_ar_mask_animation.gif"
    anim.save(animation_filename, writer="pillow", fps=5)
    plt.close()

    logger.info("    Saved AR mask animation: %s", animation_filename)


def generate_extent(center_point, zoom, aspect_ratio, out_crs=ccrs.Mercator()):
    """
    Generate extent from central location and zoom level
    Args:
        center_point (tuple(float, float)): center of the map as (longitude, latitude)
        zoom (float):  Zoom level [0 to 10]
        aspect_ratio (tuple): Aspect ratio x/y
        out_crs (cartopy.crs, optional): Out crs for extent values.
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max)
    """
    mercator_crs = ccrs.Mercator()

    # Define zoom scaling
    zoom_coefficient = 2

    # Calculate minimum longitude (min_lon) and maximum longitude (max_lon)
    lon_min, lon_max = (
        center_point[0] - (zoom_coefficient * zoom),
        center_point[0] + (zoom_coefficient * zoom),
    )

    # Transform map center to specified crs (default to Mercator)
    c_mercator = mercator_crs.transform_point(*center_point, src_crs=ccrs.Mercator())

    # Transform minimum longitude and maximum longitude to specified crs (default to
    # Mercator)
    lon_min = mercator_crs.transform_point(
        lon_min, center_point[1], src_crs=ccrs.Mercator()
    )[0]
    lon_max = mercator_crs.transform_point(
        lon_max, center_point[1], src_crs=ccrs.Mercator()
    )[0]

    # Our goal is to calculate minimum latitude (min_lat) and maximum latitude (max_lat)
    # using center point and distance between min_lon and max_lon
    # To achieve this we will use formula [(lon_distance/lat_distance) =
    # (aspect_ratio[0]/aspect_ratio[1])]
    # Calculate distance between min_lon and max_lon
    lon_distance = (lon_max) - (lon_min)

    # To calculate lat_distance, we will proceed accordingly
    lat_distance = lon_distance * aspect_ratio[1] / aspect_ratio[0]

    # Now calculate max_lat and min_lan by adding/subtracting half of the distance from
    # center latitude
    lat_max = c_mercator[1] + lat_distance / 2
    lat_min = c_mercator[1] - lat_distance / 2

    # We can return our result in any format (eg. in Mercator coordinates or in degrees)
    if out_crs != ccrs.Mercator():
        lon_min, lat_min = out_crs.transform_point(lon_min, lat_min, src_crs=out_crs)
        lon_max, lat_max = out_crs.transform_point(lon_max, lat_max, src_crs=out_crs)

    return lon_min, lon_max, lat_min, lat_max
