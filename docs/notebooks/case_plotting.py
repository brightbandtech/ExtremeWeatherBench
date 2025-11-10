# setup all the imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import xarray as xr
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

from extremeweatherbench import cases, utils


def get_polygon_from_bounding_box(bounding_box):
    """Convert a bounding box tuple to a shapely Polygon."""
    if bounding_box is None:
        return None
    left_lon, right_lon, bot_lat, top_lat = bounding_box
    return Polygon(
        [
            (left_lon, bot_lat),
            (right_lon, bot_lat),
            (right_lon, top_lat),
            (left_lon, top_lat),
            (left_lon, bot_lat),
        ]
    )


def plot_polygon(
    polygon, ax, color="yellow", alpha=0.5, my_zorder=1, linewidth=2, fill=True
):
    """Plot a shapely Polygon on a Cartopy axis."""
    if polygon is None:
        return
    patch = patches.Polygon(
        polygon.exterior.coords,
        closed=True,
        facecolor=color if fill else "none",
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=my_zorder,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(patch)


def plot_all_cases(
    ewb_cases, event_type=None, filename=None, bounding_box=None, fill_boxes=False
):
    """A function to plot all cases
    Args:
        ewb_cases (list): A list of cases to plot.
        event_type (str): The type of event to plot. If None, all
        events will be plotted).
        filename (str): The name of the file to save the plot. If
        None, the plot will not be saved.
        bounding_box (tuple): A tuple of the form (min_lon, min_lat,
        max_lon, max_lat) to set the bounding box for the plot. If
        None, the full world map will be plotted.
    """
    # plot all cases on one giant world map
    _ = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot the full map or a subset if bounding_box is specified
    if bounding_box is None:
        ax.set_global()
    else:
        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())

    # save the bounding box polygon to subset the counts later
    if bounding_box is not None:
        bounding_box_polygon = get_polygon_from_bounding_box(bounding_box)

    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="white")
    ax.add_feature(cfeature.RIVERS, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, edgecolor="black", facecolor="white", zorder=10)

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    sns_palette = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    event_colors = {
        "freeze": sns_palette[0],
        "heat_wave": sns_palette[3],
        "tropical_cyclone": sns_palette[1],
        "severe_convection": sns_palette[5],
        "atmospheric_river": sns_palette[7],
    }

    # Initialize counts for each event type
    counts_by_type = dict(
        {
            "freeze": 0,
            "heat_wave": 0,
            "severe_convection": 0,
            "atmospheric_river": 0,
            "tropical_cyclone": 0,
        }
    )
    zorders = {
        "freeze": 9,
        "heat_wave": 8,
        "atmospheric_river": 2,
        "tropical_cyclone": 10,
        "severe_convection": 0,
    }
    alphas = {
        "freeze": 0.2,
        "heat_wave": 0.2,
        "atmospheric_river": 0.3,
        "tropical_cyclone": 0.07,
        "severe_convection": 0.02,
    }

    # Handle both IndividualCaseCollection and IndividualCase
    if isinstance(ewb_cases, cases.IndividualCaseCollection):
        cases_to_plot = ewb_cases.cases
    elif isinstance(ewb_cases, cases.IndividualCase):
        cases_to_plot = [ewb_cases]
    else:
        raise TypeError(
            f"ewb_cases must be IndividualCase or "
            f"IndividualCaseCollection, got {type(ewb_cases)}"
        )

    # Plot boxes for each case
    for indiv_case in cases_to_plot:
        # Get color based on event type
        indiv_event_type = indiv_case.event_type
        color = event_colors.get(
            indiv_event_type, "gray"
        )  # Default to gray if event type not found

        # check if the case is inside the bounding box
        if bounding_box is not None:
            if not shapely.intersects(
                indiv_case.location.as_geopandas().geometry[0], bounding_box_polygon
            ):
                # print(f"Skipping case {indiv_case.case_id_number} "
                # f"as it is outside the bounding box.")
                continue

        # count the events by type
        counts_by_type[indiv_event_type] += 1

        # Plot the case geopandas info
        if event_type is None or indiv_event_type == event_type:
            # to handle wrapping around the prime meridian, we
            # can't use geopandas plot (and besides it is slow)
            # instead we have multi-polygon patches if it wraps
            # around and we need to plot each polygon separately
            if isinstance(
                indiv_case.location.as_geopandas().geometry.iloc[0],
                shapely.geometry.MultiPolygon,
            ):
                for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                    plot_polygon(
                        poly,
                        ax,
                        color=color,
                        alpha=alphas[indiv_event_type],
                        my_zorder=zorders[indiv_event_type],
                        fill=fill_boxes,
                    )
            else:
                plot_polygon(
                    indiv_case.location.as_geopandas().geometry.iloc[0],
                    ax,
                    color=color,
                    alpha=alphas[indiv_event_type],
                    my_zorder=zorders[indiv_event_type],
                    fill=fill_boxes,
                )

    # Create a custom legend for event types
    if event_type is not None:
        # if we are only plotting one event type, only show that in the legend
        legend_elements = [
            Patch(
                facecolor=event_colors[event_type],
                alpha=0.9,
                label=f"{event_type.replace('_', ' ').title()} (n = %d)"
                % counts_by_type[event_type],
            ),
        ]
    else:
        # otherwise show all event types in the legend
        legend_elements = [
            Patch(
                facecolor=event_colors["heat_wave"],
                alpha=0.9,
                label="Heat Wave (n = %d)" % counts_by_type["heat_wave"],
            ),
            Patch(
                facecolor=event_colors["freeze"],
                alpha=0.9,
                label="Freeze (n = %d)" % counts_by_type["freeze"],
            ),
            Patch(
                facecolor=event_colors["severe_convection"],
                alpha=0.9,
                label="Convection (n = %d)" % counts_by_type["severe_convection"],
            ),
            Patch(
                facecolor=event_colors["atmospheric_river"],
                alpha=0.9,
                label="Atmospheric River (n = %d)"
                % counts_by_type["atmospheric_river"],
            ),
            Patch(
                facecolor=event_colors["tropical_cyclone"],
                alpha=0.9,
                label="Tropical Cyclone (n = %d)" % counts_by_type["tropical_cyclone"],
            ),
        ]

    # Create a larger legend by specifying a larger font size in the prop dictionary
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        framealpha=1,
        frameon=True,
        borderpad=0.5,
        handletextpad=0.8,
        handlelength=2.5,
    )
    legend.set_zorder(10)

    if event_type is None:
        title = "ExtremeWeatherBench Cases (n = %d)" % sum(counts_by_type.values())
    else:
        title = (
            f"ExtremeWeatherBench Cases: "
            f"{event_type.replace('_', ' ').title()} (n = %d)"
            % counts_by_type[event_type]
        )

    ax.set_title(title, loc="left", fontsize=20)

    # save if there is a filename specified (otherwise the user
    # just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches="tight", dpi=300)


# main plotting function for plotting all cases
def plot_all_cases_and_obs(
    ewb_cases,
    event_type=None,
    filename=None,
    bounding_box=None,
    targets=None,
    show_orig_pph=False,
    case_id=None,
):
    """Plot all cases (outlined) and observations (filled) on map.
    Args:
        ewb_cases (list): A list of cases to plot.
        event_type (str): The type of event to plot. If None, all
        events will be plotted).
        filename (str): The name of the file to save the plot. If
        None, the plot will not be saved.
        bounding_box (tuple): A tuple of the form (min_lon, min_lat,
        max_lon, max_lat) to set the bounding box for the plot. If
        None, the full world map will be plotted.
        targets (dict): A dictionary containing observation metadata
        for each case, such as PPH and LSR reports.
    """
    # plot all cases on one giant world map
    _ = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot the full map or a subset if bounding_box is specified
    if bounding_box is None:
        ax.set_global()
    else:
        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())

    # save the bounding box polygon to subset the counts later
    if bounding_box is not None:
        bounding_box_polygon = get_polygon_from_bounding_box(bounding_box)

    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="white")
    ax.add_feature(cfeature.RIVERS, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, edgecolor="black", facecolor="white", zorder=10)

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    sns_palette = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    event_colors = {
        "freeze": sns_palette[0],
        "heat_wave": sns_palette[3],
        "tropical_cyclone": sns_palette[1],
        "severe_convection": sns_palette[5],
        "atmospheric_river": sns_palette[7],
    }

    # Initialize counts for each event type
    counts_by_type = dict(
        {
            "freeze": 0,
            "heat_wave": 0,
            "severe_convection": 0,
            "atmospheric_river": 0,
            "tropical_cyclone": 0,
        }
    )
    zorders = {
        "freeze": 9,
        "heat_wave": 8,
        "atmospheric_river": 2,
        "tropical_cyclone": 10,
        "severe_convection": 0,
    }
    alphas = {
        "freeze": 1,
        "heat_wave": 1,
        "atmospheric_river": 1,
        "tropical_cyclone": 1,
        "severe_convection": 1,
    }

    # Handle both IndividualCaseCollection and IndividualCase
    if isinstance(ewb_cases, cases.IndividualCaseCollection):
        cases_to_plot = ewb_cases.cases
    elif isinstance(ewb_cases, cases.IndividualCase):
        cases_to_plot = [ewb_cases]
    else:
        raise TypeError(
            f"ewb_cases must be IndividualCase or "
            f"IndividualCaseCollection, got {type(ewb_cases)}"
        )

    # Plot boxes for each case
    for indiv_case in cases_to_plot:
        # Get color based on event type
        indiv_event_type = indiv_case.event_type
        color = event_colors.get(
            indiv_event_type, "gray"
        )  # Default to gray if event type not found

        if bounding_box is not None:
            if not shapely.intersects(
                indiv_case.location.as_geopandas().geometry[0], bounding_box_polygon
            ):
                # print(f"Skipping case {indiv_case.case_id_number} "
                # f"as it is outside the bounding box.")
                continue

        # if a specific case id is specified, only plot that case
        if case_id is not None and indiv_case.case_id_number != case_id:
            continue

        # count the events by type
        counts_by_type[indiv_event_type] += 1

        # Plot the case geopandas info
        if indiv_event_type == event_type or event_type is None:
            # print(indiv_case)

            # to handle wrapping around the prime meridian, we
            # can't use geopandas plot (and besides it is slow)
            # instead we have multi-polygon patches if it wraps
            # around and we need to plot each polygon separately
            if isinstance(
                indiv_case.location.as_geopandas().geometry.iloc[0],
                shapely.geometry.MultiPolygon,
            ):
                for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                    plot_polygon(
                        poly,
                        ax,
                        color=color,
                        alpha=alphas[indiv_event_type],
                        my_zorder=zorders[indiv_event_type],
                        linewidth=0.8,
                        fill=False,
                    )
            else:
                plot_polygon(
                    indiv_case.location.as_geopandas().geometry.iloc[0],
                    ax,
                    color=color,
                    alpha=alphas[indiv_event_type],
                    my_zorder=zorders[indiv_event_type],
                    linewidth=0.8,
                    fill=False,
                )

            # grab the target data for this case; targets is a list of tuples of
            # (case_id, target dataset)
            my_target_info = [
                n[1]
                for n in targets
                if n[0] == indiv_case.case_id_number and n[1].attrs["source"] != "ERA5"
            ]
            # print(my_target_info)

            # make a scatter plot of the target points (for hot/cold/tc events)
            if (
                indiv_event_type in ["heat_wave", "freeze", "tropical_cyclone"]
                and len(my_target_info) > 0
            ):
                # Get the data from my_target_info
                data = my_target_info[0]

                # sparse array for GHCN data
                if indiv_event_type in ["heat_wave", "freeze"]:
                    try:
                        data = utils.stack_sparse_data_from_dims(
                            data["surface_air_temperature"], ["latitude", "longitude"]
                        )
                    except Exception as e:
                        print(
                            f"Error stacking sparse data for "
                            f"{indiv_case.case_id_number} from "
                            f"dimensions latitude, longitude: {e}. "
                            f"This is likely because the data is not "
                            f"available for this case."
                        )
                        continue
                try:
                    lat_values = data["latitude"].values
                    lon_values = data["longitude"].values
                except Exception as e:
                    print(
                        f"Error stacking sparse data from dimensions "
                        f"latitude, longitude: {e}"
                    )
                    continue

                # Convert longitude values from 0-360 to -180 to 180 for proper
                # antimeridian handling with Cartopy
                lon_values_180 = utils.convert_longitude_to_180(lon_values)

                # if (np.min(lat_values) <= 0):
                #     print(data)

                ax.scatter(
                    lon_values_180,
                    lat_values,
                    color=color,
                    s=1,
                    alpha=alphas[indiv_event_type],
                    transform=ccrs.Geodetic(),
                    zorder=zorders[indiv_event_type],
                )

            # if it is convective, show the PPH and LSRs
            if indiv_event_type == "severe_convection":
                # Get the data from my_target_info
                data = my_target_info[0]
                # print(data)
                try:
                    data = utils.stack_sparse_data_from_dims(
                        data["report_type"], ["latitude", "longitude"]
                    )
                except Exception as e:
                    print(
                        f"Error stacking sparse data for "
                        f"{indiv_case.case_id_number} from "
                        f"dimensions latitude, longitude: {e}. "
                        f"This is likely because the data is not "
                        f"available for this case."
                    )
                    continue

                for my_data in data:
                    # print(my_data)
                    hail_reports = my_data[my_data == 2]
                    # print(hail_reports)
                    lat_values = hail_reports.latitude.values
                    lon_values = hail_reports.longitude.values
                    ax.scatter(
                        lon_values,
                        lat_values,
                        color="black",
                        alpha=0.9,
                        marker="o",
                        transform=ccrs.Geodetic(),
                        s=6,
                    )

                    tor_reports = my_data[my_data == 3]
                    # print(tor_reports)
                    lat_values = tor_reports.latitude.values
                    lon_values = tor_reports.longitude.values
                    ax.scatter(
                        lon_values,
                        lat_values,
                        color="red",
                        marker="^",
                        transform=ccrs.Geodetic(),
                        s=6,
                    )

    # Create a custom legend for event types
    if event_type is not None:
        # if we are only plotting one event type, only show that in the legend
        legend_elements = [
            Patch(
                facecolor=event_colors[event_type],
                alpha=0.9,
                label=f"{event_type.replace('_', ' ').title()} (n = %d)"
                % counts_by_type[event_type],
            ),
        ]
    else:
        # otherwise show all event types in the legend
        legend_elements = [
            Patch(
                facecolor=event_colors["heat_wave"],
                alpha=0.9,
                label="Heat Wave (n = %d)" % counts_by_type["heat_wave"],
            ),
            Patch(
                facecolor=event_colors["freeze"],
                alpha=0.9,
                label="Freeze (n = %d)" % counts_by_type["freeze"],
            ),
            Patch(
                facecolor=event_colors["severe_convection"],
                alpha=0.9,
                label="Convection (n = %d)" % counts_by_type["severe_convection"],
            ),
            Patch(
                facecolor=event_colors["atmospheric_river"],
                alpha=0.9,
                label="Atmospheric River (n = %d)"
                % counts_by_type["atmospheric_river"],
            ),
            Patch(
                facecolor=event_colors["tropical_cyclone"],
                alpha=0.9,
                label="Tropical Cyclone (n = %d)" % counts_by_type["tropical_cyclone"],
            ),
        ]
    # Create a larger legend by specifying a larger font size in the prop dictionary
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        framealpha=1,
        frameon=True,
        borderpad=0.5,
        handletextpad=0.8,
        handlelength=2.5,
    )
    legend.set_zorder(10)

    if event_type is None:
        title = "ExtremeWeatherBench Cases (n = %d)" % sum(counts_by_type.values())
    else:
        title = (
            f"ExtremeWeatherBench Cases: "
            f"{event_type.replace('_', ' ').title()} (n = %d)"
            % counts_by_type[event_type]
        )

    ax.set_title(title, loc="left", fontsize=20)

    # save if there is a filename specified (otherwise the user
    # just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches="tight", dpi=300)


def plot_boxes(box_list, box_names, title, filename=None):
    # plot all cases on one giant world map
    _ = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="white")
    ax.add_feature(cfeature.RIVERS, edgecolor="black")

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=1, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    _ = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    # Plot boxes for each case
    for box in box_list:
        plot_polygon(box, ax, color="blue", alpha=1, fill=False)

    plt.legend(loc="lower left", fontsize=12)
    ax.set_title(title, loc="left", fontsize=20)

    # save if there is a filename specified (otherwise the user
    # just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches="tight", dpi=300)


def celsius_colormap_and_normalize() -> tuple[mcolors.Colormap, mcolors.Normalize]:
    """Gets the colormap and normalization for 2m temperature.

    Uses a custom colormap for temperature in Celsius.

    Returns:
        A tuple (cmap, norm) for plotting.
    """
    lo_colors = [
        "#E4C7F4",
        "#E53885",
        "#C17CBE",
        "#694396",
        "#CBCCE9",
        "#6361BD",
        "#77FBFE",
    ]
    hi_colors = [
        "#8CE9B0",
        "#479F31",
        "#F0F988",
        "#AD311B",
        "#ECB9F1",
        "#7F266F",
    ]
    colors = lo_colors + hi_colors

    # Calculate the position where we want the 0C jump
    lo = -67.8
    hi = 54.4
    threshold = 0
    threshold_pos = (threshold - lo) / (hi - lo)  # normalize 0°C position to [0,1]

    # Create positions for colors with a small gap around zero_pos
    positions = np.concatenate(
        [
            np.linspace(0, threshold_pos - 0.02, len(lo_colors)),  # Colors up to white
            # [threshold_pos],  # White position
            np.linspace(threshold_pos + 0.02, 1, len(hi_colors)),  # Colors after white
        ]
    )

    return mcolors.LinearSegmentedColormap.from_list(
        "temp_colormap", list(zip(positions, colors))
    ), mcolors.Normalize(vmin=lo, vmax=hi)


def convert_day_yearofday_to_time(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Convert dayofyear and hour coordinates in an xarray Dataset to a new time
    coordinate.

    Args:
        dataset: The input xarray dataset.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f"{year}-01-01",
        periods=len(dataset["dayofyear"]) * len(dataset["hour"]),
        freq="6h",
    )
    dataset = dataset.stack(time=("dayofyear", "hour"))
    # Assign the new time coordinate to the dataset
    dataset = dataset.drop_vars(["time", "dayofyear", "hour"]).assign_coords(
        time=time_dim
    )

    return dataset


def generate_heatwave_dataset(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Calculate times where regional avg temp is above climatology.

    Args:
        era5: ERA5 dataset containing 2m_temperature
        climatology: BB climatology containing
        surface_temperature_85th_percentile
        single_case: cases.IndividualCase object with metadata
    """
    era5_case = era5[["2m_temperature"]].sel(
        time=slice(single_case.start_date, single_case.end_date)
    )
    subset_climatology = convert_day_yearofday_to_time(
        climatology, np.unique(era5_case.time.dt.year.values)[0]
    )
    merged_dataset = xr.merge(
        [
            subset_climatology.rename(
                {"2m_temperature": "surface_temperature_85th_percentile"}
            ),
            era5_case,
        ],
        join="inner",
    )
    if (
        single_case.location.longitude_min < 0
        or single_case.location.longitude_min > 180
    ) and (
        single_case.location.longitude_max > 0
        and single_case.location.longitude_max < 180
    ):
        merged_dataset = utils.convert_longitude_to_180(merged_dataset)
    merged_dataset = merged_dataset.sel(
        latitude=slice(
            single_case.location.latitude_max, single_case.location.latitude_min
        ),
        longitude=slice(
            single_case.location.longitude_min, single_case.location.longitude_max
        ),
    )
    return merged_dataset


def generate_heatwave_plots(
    heatwave_dataset: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Plot max timestep of heatwave event and avg regional temp
    time series on separate plots.

    Args:
        heatwave_dataset: contains 2m_temperature,
        surface_temperature_85th_percentile, time, latitude, longitude
        single_case: cases.IndividualCase object with metadata
    """
    time_based_heatwave_dataset = heatwave_dataset.mean(["latitude", "longitude"])
    # Plot 1: Min timestep of the heatwave event
    fig1, ax1 = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Select the timestep with the maximum spatially averaged temp
    subset_timestep = time_based_heatwave_dataset["time"][
        time_based_heatwave_dataset["2m_temperature"].argmax()
    ]
    # Mask places where temp >= 85th percentile climatology
    temp_data = heatwave_dataset["2m_temperature"] - 273.15
    climatology_data = heatwave_dataset["surface_temperature_85th_percentile"] - 273.15

    # Create mask for values where temp > climatology
    # (heatwave condition)
    mask = temp_data > climatology_data

    # Apply mask to temperature data
    masked_temp = temp_data.where(mask)
    cmap, norm = celsius_colormap_and_normalize()
    im = masked_temp.sel(time=subset_timestep).plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )
    (
        temp_data.sel(time=subset_timestep).plot.contour(
            ax=ax1,
            levels=[0],
            colors="r",
            linewidths=0.75,
            ls=":",
            transform=ccrs.PlateCarree(),
        )
    )
    # Add coastlines and gridlines
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, edgecolor="black")
    ax1.add_feature(cfeature.LAKES, edgecolor="black")
    ax1.add_feature(
        cfeature.RIVERS, edgecolor=[0.59375, 0.71484375, 0.8828125], alpha=0.5
    )
    ax1.add_feature(cfeature.STATES, edgecolor="grey")
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}
    ax1.set_title("")  # clears the default xarray title
    time_str = (
        heatwave_dataset["time"]
        .sel(time=subset_timestep)
        .dt.strftime("%Y-%m-%d %Hz")
        .values
    )
    ax1.set_title(
        f"Temperature Where > 85th Percentile Climatology\n"
        f"{single_case.title}, Case ID {single_case.case_id_number}\n"
        f"{time_str}",
        loc="left",
    )
    # Add the location coordinate as a dot on the map
    ax1.tick_params(axis="y", which="major", labelsize=12)
    # Create a colorbar with the same height as the plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig1.colorbar(im, cax=cax, label="Temp > 85th Percentile (C)")
    cbar.set_label("Temperature (C)", size=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_spatial.png", transparent=True)
    plt.show()

    # Plot 2: Average regional temperature time series
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    lss = ["-.", "-"]
    lc = ["k", "tab:red"]
    lws = [0.75, 1.5]
    for i, variable in enumerate(time_based_heatwave_dataset):
        (time_based_heatwave_dataset[variable] - 273.15).plot(
            ax=ax2, label=variable, lw=lws[i], ls=lss[i], c=lc[i]
        )
    ax2.legend(fontsize=12)
    mask = (
        time_based_heatwave_dataset["2m_temperature"]
        > time_based_heatwave_dataset["surface_temperature_85th_percentile"]
    )
    start = None
    for i, val in enumerate(mask.values):
        if val and start is None:
            start = time_based_heatwave_dataset.time[i].values
        elif not val and start is not None:
            ax2.axvspan(
                start,
                time_based_heatwave_dataset.time[i].values,
                color="red",
                alpha=0.1,
            )
            start = None
    if start is not None:
        ax2.axvspan(
            start, time_based_heatwave_dataset.time[-1].values, color="red", alpha=0.1
        )
    ax2.set_title("")
    ax2.set_title(
        "Spatially Averaged Heatwave Event vs 85th Percentile Climatology",
        fontsize=14,
        loc="left",
    )
    ax2.set_ylabel("Temperature (C)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_tick_params(
        rotation=45,
        labelsize=10,
        pad=0.0001,
    )
    ax2.tick_params(axis="y", labelsize=12)

    # Create legend handles including the axvspan
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="k",
            linestyle="-.",
            linewidth=0.75,
            label="2m Temperature, 85th Percentile",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="-",
            linewidth=1.5,
            label="2m Temperature",
        ),
        Patch(facecolor="red", alpha=0.1, label="Above 85th Percentile"),
    ]
    ax2.legend(handles=legend_elements, fontsize=12)

    ax2.tick_params(axis="y", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_timeseries.png", transparent=True)
    plt.show()


def generate_freeze_dataset(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Calculate times where regional avg temp is below climatology.

    Args:
        era5: ERA5 dataset containing 2m_temperature
        climatology: BB climatology containing
        surface_temperature_15th_percentile
        single_case: cases.IndividualCase object with metadata
    """
    era5_case = era5[["2m_temperature"]].sel(
        time=slice(single_case.start_date, single_case.end_date)
    )
    subset_climatology = convert_day_yearofday_to_time(
        climatology, np.unique(era5_case.time.dt.year.values)[0]
    )
    merged_dataset = xr.merge(
        [
            subset_climatology.rename(
                {"2m_temperature": "surface_temperature_15th_percentile"}
            ),
            era5_case,
        ],
        join="inner",
    )
    if (
        single_case.location.longitude_min < 0
        or single_case.location.longitude_min > 180
    ) and (
        single_case.location.longitude_max > 0
        and single_case.location.longitude_max < 180
    ):
        merged_dataset = utils.convert_longitude_to_180(merged_dataset)
    merged_dataset = merged_dataset.sel(
        latitude=slice(
            single_case.location.latitude_max, single_case.location.latitude_min
        ),
        longitude=slice(
            single_case.location.longitude_min, single_case.location.longitude_max
        ),
    )
    return merged_dataset


def generate_freeze_plots(
    freeze_dataset: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Plot max timestep of freeze event and avg regional temp
    time series on separate plots.

    Args:
        freeze_dataset: contains 2m_temperature,
        surface_temperature_15th_percentile, time, latitude, longitude
        single_case: cases.IndividualCase object with metadata
    """
    time_based_freeze_dataset = freeze_dataset.mean(["latitude", "longitude"])
    # Plot 1: Min timestep of the freeze event
    fig1, ax1 = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Select the timestep with the maximum spatially averaged temp
    subset_timestep = time_based_freeze_dataset["time"][
        time_based_freeze_dataset["2m_temperature"].argmin()
    ]
    # Mask places where temp >= 15th percentile climatology
    temp_data = freeze_dataset["2m_temperature"] - 273.15
    climatology_data = freeze_dataset["surface_temperature_15th_percentile"] - 273.15

    # Create mask for values where temp < climatology
    # (freeze condition)
    mask = temp_data < climatology_data

    # Apply mask to temperature data
    masked_temp = temp_data.where(mask)
    cmap, norm = celsius_colormap_and_normalize()
    im = masked_temp.sel(time=subset_timestep).plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
    )
    (
        temp_data.sel(time=subset_timestep).plot.contour(
            ax=ax1,
            levels=[0],
            colors="r",
            linewidths=0.75,
            ls=":",
            transform=ccrs.PlateCarree(),
        )
    )
    # Add coastlines and gridlines
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, edgecolor="black")
    ax1.add_feature(cfeature.LAKES, edgecolor="black")
    ax1.add_feature(
        cfeature.RIVERS, edgecolor=[0.59375, 0.71484375, 0.8828125], alpha=0.5
    )
    ax1.add_feature(cfeature.STATES, edgecolor="grey")
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}
    ax1.set_title("")  # clears the default xarray title
    time_str = (
        freeze_dataset["time"]
        .sel(time=subset_timestep)
        .dt.strftime("%Y-%m-%d %Hz")
        .values
    )
    ax1.set_title(
        f"Temperature Where < 15th Percentile Climatology\n"
        f"{single_case.title}, Case ID {single_case.case_id_number}\n"
        f"{time_str}",
        loc="left",
    )
    # Add the location coordinate as a dot on the map
    ax1.tick_params(axis="y", which="major", labelsize=12)
    # Create a colorbar with the same height as the plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig1.colorbar(im, cax=cax, label="Temp < 15th Percentile (C)")
    cbar.set_label("Temperature (C)", size=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_spatial.png", transparent=True)
    plt.show()

    # Plot 2: Average regional temperature time series
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    lss = ["-.", "-"]
    lc = ["k", "tab:red"]
    lws = [0.75, 1.5]
    for i, variable in enumerate(time_based_freeze_dataset):
        (time_based_freeze_dataset[variable] - 273.15).plot(
            ax=ax2, label=variable, lw=lws[i], ls=lss[i], c=lc[i]
        )
    ax2.legend(fontsize=12)
    mask = (
        time_based_freeze_dataset["2m_temperature"]
        < time_based_freeze_dataset["surface_temperature_15th_percentile"]
    )
    start = None
    for i, val in enumerate(mask.values):
        if val and start is None:
            start = time_based_freeze_dataset.time[i].values
        elif not val and start is not None:
            ax2.axvspan(
                start,
                time_based_freeze_dataset.time[i].values,
                color="red",
                alpha=0.1,
            )
            start = None
    if start is not None:
        ax2.axvspan(
            start, time_based_freeze_dataset.time[-1].values, color="red", alpha=0.1
        )
    ax2.set_title("")
    ax2.set_title(
        "Spatially Averaged Freeze Event vs 15th Percentile Climatology",
        fontsize=14,
        loc="left",
    )
    ax2.set_ylabel("Temperature (C)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_tick_params(
        rotation=45,
        labelsize=10,
        pad=0.0001,
    )
    ax2.tick_params(axis="y", labelsize=12)

    # Create legend handles including the axvspan
    from matplotlib.patches import Patch

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="k",
            linestyle="-.",
            linewidth=0.75,
            label="2m Temperature, 15th Percentile",
        ),
        plt.Line2D(
            [0],
            [0],
            color="tab:red",
            linestyle="-",
            linewidth=1.5,
            label="2m Temperature",
        ),
        Patch(facecolor="red", alpha=0.1, label="Below 15th Percentile"),
    ]
    ax2.legend(handles=legend_elements, fontsize=12)

    ax2.tick_params(axis="y", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_timeseries.png", transparent=True)
    plt.show()


def plot_results_by_metric(
    data, settings, title, filename=None, show_all_in_legend=False
):
    """
    Plots the results of the ExtremeWeatherBench for the data
    specified.
    parameters:
        data: list of dictionaries containing the data to plot
        settings: list of dictionaries containing the plot settings
        title: string, the title of the plot
        filename: string, filename to save the plot to (None if you
        don't want to save it)
        show_all_in_legend: boolean, if True, then all labels will
        be shown in the legend, if False they will be grouped
    """
    sns.set_theme(style="whitegrid")
    _ = sns.color_palette("tab10")
    _, ax = plt.subplots(figsize=(16, 4))

    legend_elements = []
    legend_labels = list()

    # and add the HRES line
    for my_settings in settings:
        if show_all_in_legend:
            # my_label = f"{my_settings['label_str']} (n={my_n})"
            raise ValueError("need to fix what my_n should be")
        else:
            my_label = my_settings["label_str"]

        if "HRES" in my_label:
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color=my_settings["color"], linewidth=4, label=my_label
                )
            )
            break

    # Add a blank line to your legend_elements list
    legend_elements.append(plt.Line2D([0], [0], color="white", alpha=0, label=" "))

    for i, model in enumerate(data):
        my_mean = model["value"].mean("case_id_number")
        my_n = len(np.unique(model["case_id_number"].values))
        my_settings = settings[i]
        if show_all_in_legend:
            my_label = f"{my_settings['label_str']} (n={my_n})"
        else:
            my_label = my_settings["label_str"]

        plt.plot(
            my_mean,
            color=my_settings["color"],
            linewidth=4,
            label=my_label,
            linestyle=my_settings["linestyle"],
            marker=my_settings["marker"],
            markersize=10,
        )

        # add any unique labels to the legend except for HRES
        # (it gets its own line in the legend)
        if show_all_in_legend or (
            my_label not in legend_labels and "HRES" not in my_label
        ):
            legend_labels.append(my_label)
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color=my_settings["color"], linewidth=4, label=my_label
                )
            )

    # set the xticks in days
    xtick_str = [
        f"{int(my_time / np.timedelta64(1, 'D')):d}"
        for my_time in model["lead_time"].values
    ]
    ax.set_xticks(labels=xtick_str, ticks=np.arange(0, len(model["lead_time"]), 1))

    ax.set_ylabel("Celsius")
    ax.set_xlabel("Lead Time (days)")
    plt.title(title)

    # Add a blank line to your legend_elements list
    legend_elements.append(plt.Line2D([0], [0], color="white", alpha=0, label=" "))

    # now add the unique groups with markers
    my_groups = list()
    for my_settings in settings:
        if my_settings["group"] not in my_groups and my_settings["group"] != "HRES":
            my_groups.append(my_settings["group"])
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="darkgrey",
                    marker=my_settings["marker"],
                    markersize=10,
                    label=my_settings["group"],
                    linestyle=my_settings["linestyle"],
                    linewidth=4,
                )
            )

    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5))

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)


def plot_all_cases_hexbin(
    ewb_cases, event_type=None, filename=None, bounding_box=None, hexbin_size=[100, 100]
):
    """Plot all cases using hexbins to cover the full space
    Args:
        ewb_cases (list): A list of cases to plot.
        event_type (str): The type of event to plot. If None, all
        events will be plotted).
        filename (str): The name of the file to save the plot. If
        None, the plot will not be saved.
        bounding_box (tuple): A tuple of the form (min_lon, min_lat,
        max_lon, max_lat) to set the bounding box for the plot. If
        None, the full world map will be plotted.
    """

    # Create bounding box polygon if provided
    bounding_box_polygon = None
    if bounding_box is not None:
        bounding_box_polygon = get_polygon_from_bounding_box(bounding_box)

    # first make the counts by hexbin
    polygons_by_event = dict()

    # save a list of polygons for each event type
    for indiv_case in ewb_cases.cases:
        # Get color based on event type
        indiv_event_type = indiv_case.event_type

        if indiv_event_type not in polygons_by_event:
            polygons_by_event[indiv_event_type] = []

        # check if the case is inside the bounding box
        if bounding_box is not None:
            if not shapely.intersects(
                indiv_case.location.as_geopandas().geometry[0], bounding_box_polygon
            ):
                # print(f"Skipping case {indiv_case.case_id_number} "
                # f"as it is outside the bounding box.")
                continue

        # grab all points inside polygon and increment all
        # overlapping hexbins
        if event_type is None or indiv_event_type == event_type:
            if isinstance(
                indiv_case.location.as_geopandas().geometry.iloc[0],
                shapely.geometry.MultiPolygon,
            ):
                for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                    polygons_by_event[indiv_event_type].append(poly)
            else:
                polygons_by_event[indiv_event_type].append(
                    indiv_case.location.as_geopandas().geometry.iloc[0]
                )

    # print(polygons_by_event)

    # now make the grid for the hexbins and then find all grid
    # points inside each polygon
    # Create a meshgrid of longitude and latitude
    lon_grid = np.linspace(-180, 180, hexbin_size[0])
    lat_grid = np.linspace(-90, 90, hexbin_size[1])
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid, indexing="ij")

    # Flatten to get all grid point coordinates
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()

    # Create shapely points from the grid coordinates
    # shapely.points takes an array of shape (n, 2) with [lon, lat] pairs
    grid_points_coords = np.column_stack([lon_flat, lat_flat])
    grid_points = shapely.points(grid_points_coords)

    # Store grid points inside polygons by event type
    grid_points_by_event = dict()

    # For each event type, find all grid points inside any of its polygons
    for event_type_name, polygons in polygons_by_event.items():
        # Use a set to avoid duplicates (points can be inside multiple polygons)
        grid_points_set = set()

        # For each polygon in this event type, find points inside it
        for poly in polygons:
            # Check each grid point to see if it's inside the polygon
            # Using polygon.contains() method for point-in-polygon check
            for i, point in enumerate(grid_points):
                if poly.contains(point):
                    grid_point_coord = (lon_flat[i], lat_flat[i])
                    grid_points_set.add(grid_point_coord)

        # Convert set to list and store by event type
        grid_points_by_event[event_type_name] = list(grid_points_set)

    print(grid_points_by_event)

    # Create scatter plot of grid points colored by event type
    _ = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set the map extent
    if bounding_box is None:
        ax.set_global()
    else:
        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())

    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="white")
    ax.add_feature(cfeature.RIVERS, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, edgecolor="black", facecolor="white", zorder=10)

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type (using the same palette as other plots)
    sns_palette = sns.color_palette("tab10")
    event_colors = {
        "freeze": sns_palette[0],
        "heat_wave": sns_palette[3],
        "tropical_cyclone": sns_palette[1],
        "severe_convection": sns_palette[5],
        "atmospheric_river": sns_palette[7],
    }

    # Plot grid points for each event type
    for event_type_name, grid_points_list in grid_points_by_event.items():
        if len(grid_points_list) > 0:
            # Unzip the coordinates
            lons, lats = zip(*grid_points_list)
            color = event_colors.get(event_type_name, "gray")
            ax.scatter(
                lons,
                lats,
                c=color,
                s=10,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                label=event_type_name.replace("_", " ").title(),
                zorder=10,
            )

    # Create legend
    ax.legend(loc="lower left", framealpha=1, frameon=True, fontsize=10)

    # Set title
    total_points = sum(len(points) for points in grid_points_by_event.values())
    ax.set_title(
        f"Grid Points Inside Event Polygons (n = {total_points})",
        loc="left",
        fontsize=20,
    )

    # Save if filename is provided
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches="tight", dpi=300)
