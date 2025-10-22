# setup all the imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from shapely.geometry import Polygon
import shapely
from matplotlib.patches import Patch
from extremeweatherbench import evaluate, utils, cases, defaults
import matplotlib.colors as mcolors
import xarray as xr
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

def plot_polygon(polygon, ax, color='yellow', alpha=0.5, my_zorder=1, linewidth=2):
    """Plot a shapely Polygon on a Cartopy axis."""
    if polygon is None:
        return
    patch = patches.Polygon(
        polygon.exterior.coords,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=my_zorder,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(patch)

def plot_polygon_outline(polygon, ax, color='yellow', alpha=0.5, my_zorder=1, linewidth=2):
    """Plot a shapely Polygon outline on a Cartopy axis."""
    if polygon is None:
        return
    patch = patches.Polygon(
        polygon.exterior.coords,
        closed=True,
        facecolor='none',
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=my_zorder,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(patch)


def plot_all_cases(ewb_cases, event_type=None, filename=None, bounding_box=None, fill_boxes=False):
    """A function to plot all cases
    Args:
        ewb_cases (list): A list of cases to plot.
        event_type (str): The type of event to plot. If None, all events will be plotted).
        filename (str): The name of the file to save the plot. If None, the plot will not be saved.
        bounding_box (tuple): A tuple of the form (min_lon, min_lat, max_lon, max_lat) to set the bounding box for the plot. If None, the full world map will be plotted.
    """
    # plot all cases on one giant world map
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot the full map or a subset if bounding_box is specified
    if (bounding_box is None):
        ax.set_global()
    else:
        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())
    
    # save the bounding box polygon to subset the counts later
    if (bounding_box is not None):
        bounding_box_polygon = get_polygon_from_bounding_box(bounding_box)
        #plot_polygon(bounding_box_polygon, ax, color='yellow', alpha=0.5)
        
    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='white')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='white', zorder=10)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    sns_palette = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    # event_colors = {
    #     'heat_wave': 'firebrick',
    #     'tropical_cyclone': 'darkorange',
    #     'severe_convection': 'orchid',
    #     'atmospheric_river': 'mediumseagreen',
    #     'freeze': 'royalblue',   
    # }
    event_colors = {
            'freeze': sns_palette[0],  
            'heat_wave': sns_palette[3],
            'tropical_cyclone': sns_palette[1],
            'severe_convection': sns_palette[5],
            'atmospheric_river': sns_palette[7],
        }

    # Initialize counts for each event type
    counts_by_type = dict({'freeze': 0, 'heat_wave': 0, 'severe_convection': 0, 'atmospheric_river': 0, 'tropical_cyclone': 0})
    zorders = {'freeze': 9, 'heat_wave': 8, 'atmospheric_river': 2, 'tropical_cyclone': 10, 'severe_convection': 0}
    alphas = {'freeze': 0.2, 'heat_wave': 0.2, 'atmospheric_river': 0.2, 'tropical_cyclone': 0.15, 'severe_convection': 0.02}
    box_alphas = {'freeze': 1, 'heat_wave': 1, 'atmospheric_river': 1, 'tropical_cyclone': 0.5, 'severe_convection': 1}

    # Plot boxes for each case
    for indiv_case in ewb_cases.cases:
        # Get color based on event type
        indiv_event_type = indiv_case.event_type
        color = event_colors.get(indiv_event_type, 'gray')  # Default to gray if event type not found

        # check if the case is inside the bounding box
        if bounding_box is not None:
            if (not shapely.intersects(indiv_case.location.as_geopandas().geometry[0], bounding_box_polygon)):
                #print(f"Skipping case {indiv_case.case_id_number} as it is outside the bounding box.")
                continue

        # count the events by type
        counts_by_type[indiv_event_type] += 1

        # Plot the case geopandas info
        if (event_type is None or indiv_event_type == event_type):
            if (fill_boxes):
                # to handle wrapping around the prime meridian, we can't use geopandas plot (and besides it is slow)
                # instead we have multi-polygon patches if it wraps around and we need to plot each polygon separately
                if isinstance(indiv_case.location.as_geopandas().geometry.iloc[0], shapely.geometry.MultiPolygon):
                    for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                        plot_polygon(poly, ax, color=color, alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
                else:
                    plot_polygon(indiv_case.location.as_geopandas().geometry.iloc[0], ax, color=color, 
                                alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
            else:
                # to handle wrapping around the prime meridian, we can't use geopandas plot (and besides it is slow)
                # instead we have multi-polygon patches if it wraps around and we need to plot each polygon separately
                if isinstance(indiv_case.location.as_geopandas().geometry.iloc[0], shapely.geometry.MultiPolygon):
                    for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                        plot_polygon_outline(poly, ax, color=color, alpha=box_alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
                else:
                    plot_polygon_outline(indiv_case.location.as_geopandas().geometry.iloc[0], ax, color=color, 
                                alpha=box_alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])


        
    # Create a custom legend for event types
    if (event_type is not None):
        # if we are only plotting one event type, only show that in the legend
        legend_elements = [
            Patch(facecolor=event_colors[event_type], alpha=0.9, label=f'{event_type.replace("_", " ").title()} (n = %d)' % counts_by_type[event_type]),
        ]
    else:
        # otherwise show all event types in the legend
        legend_elements = [
            Patch(facecolor=event_colors['heat_wave'], alpha=0.9, label='Heat Wave (n = %d)' % counts_by_type['heat_wave']),
            Patch(facecolor=event_colors['freeze'], alpha=0.9, label='Freeze (n = %d)' % counts_by_type['freeze']),
            Patch(facecolor=event_colors['severe_convection'], alpha=0.9, label='Convection (n = %d)' % counts_by_type['severe_convection']),
            Patch(facecolor=event_colors['atmospheric_river'], alpha=0.9, label='Atmospheric River (n = %d)' % counts_by_type['atmospheric_river']),
            Patch(facecolor=event_colors['tropical_cyclone'], alpha=0.9, label='Tropical Cyclone (n = %d)' % counts_by_type['tropical_cyclone']),
        ]

    # Create a larger legend by specifying a larger font size in the prop dictionary
    legend = ax.legend(handles=legend_elements, loc='lower left', framealpha=1, frameon=True, borderpad=0.5, handletextpad=0.8, handlelength=2.5)
    legend.set_zorder(10)
    
    if (event_type is None):
        title = 'ExtremeWeatherBench Cases (n = %d)' % sum(counts_by_type.values())
    else:
        title = f'ExtremeWeatherBench Cases: {event_type.replace("_", " ").title()} (n = %d)' % counts_by_type[event_type]
    
    ax.set_title(title, loc='left', fontsize=20)
    
    # save if there is a filename specified (otherwise the user just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches='tight', dpi=300)

# main plotting function for plotting all cases
def plot_all_cases_and_obs(ewb_cases, event_type=None, filename=None, bounding_box=None, targets=None, show_orig_pph=False, case_id=None):
    """A function to plot all cases (outlined) and observations (filled) on a world map.
    Args:
        ewb_cases (list): A list of cases to plot.
        event_type (str): The type of event to plot. If None, all events will be plotted).
        filename (str): The name of the file to save the plot. If None, the plot will not be saved.
        bounding_box (tuple): A tuple of the form (min_lon, min_lat, max_lon, max_lat) to set the bounding box for the plot. If None, the full world map will be plotted.
        targets (dict): A dictionary containing observation metadata for each case, such as PPH and LSR reports.
    """
    # plot all cases on one giant world map
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot the full map or a subset if bounding_box is specified
    if (bounding_box is None):
        ax.set_global()
    else:
        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())
    
    # save the bounding box polygon to subset the counts later
    if (bounding_box is not None):
        bounding_box_polygon = get_polygon_from_bounding_box(bounding_box)
        #plot_polygon(bounding_box_polygon, ax, color='yellow', alpha=0.5)
        
    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='white')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='white', zorder=10)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    sns_palette = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    # event_colors = {
    #     'heat_wave': 'firebrick',
    #     'tropical_cyclone': 'darkorange',
    #     'severe_convection': 'orchid',
    #     'atmospheric_river': 'mediumseagreen',
    #     'freeze': 'royalblue',   
    # }
    event_colors = {
            'freeze': sns_palette[0],  
            'heat_wave': sns_palette[3],
            'tropical_cyclone': sns_palette[1],
            'severe_convection': sns_palette[5],
            'atmospheric_river': sns_palette[7],
        }

    # Initialize counts for each event type
    counts_by_type = dict({'freeze': 0, 'heat_wave': 0, 'severe_convection': 0, 'atmospheric_river': 0, 'tropical_cyclone': 0})
    zorders = {'freeze': 9, 'heat_wave': 8, 'atmospheric_river': 2, 'tropical_cyclone': 10, 'severe_convection': 0}
    alphas = {'freeze': 1, 'heat_wave': 1, 'atmospheric_river': 1, 'tropical_cyclone': 1, 'severe_convection': 1}

    # Plot boxes for each case
    for indiv_case in ewb_cases.cases:
        # Get color based on event type
        indiv_event_type = indiv_case.event_type
        color = event_colors.get(indiv_event_type, 'gray')  # Default to gray if event type not found

        # check if the case is inside the bounding box
        if bounding_box is not None:
            if (not shapely.intersects(indiv_case.location.geopandas.geometry[0], bounding_box_polygon)):
                #print(f"Skipping case {indiv_case.case_id_number} as it is outside the bounding box.")
                continue

        # if a specific case id is specified, only plot that case
        if (case_id is not None and indiv_case.case_id_number != case_id):
            continue

        # count the events by type
        counts_by_type[indiv_event_type] += 1

        # Plot the case geopandas info
        if (indiv_event_type == event_type or event_type is None):
            # to handle wrapping around the prime meridian, we can't use geopandas plot (and besides it is slow)
            # instead we have multi-polygon patches if it wraps around and we need to plot each polygon separately
            if isinstance(indiv_case.location.as_geopandas().geometry.iloc[0], shapely.geometry.MultiPolygon):
                for poly in indiv_case.location.as_geopandas().geometry.iloc[0].geoms:
                    plot_polygon_outline(poly, ax, color=color, alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type], linewidth=0.8)
            else:
                plot_polygon_outline(indiv_case.location.as_geopandas().geometry.iloc[0], ax, color=color, 
                                        alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type], linewidth=0.8)
                
            # grab the target data for this case
            my_target_info = [n[1] for n in targets if n[0] == indiv_case.case_id_number and n[1].attrs['source'] != 'ERA5']

            # make a scatter plot of the target points (for hot/cold/tc events)
            if (indiv_event_type in ['heat_wave', 'freeze', 'tropical_cyclone'] and len(my_target_info) > 0):
                # Get the data from my_target_info
                data = my_target_info[0]
                
                # sparse array for GHCN data
                if indiv_event_type in ['heat_wave', 'freeze']:
                    try:
                        data = utils.stack_sparse_data_from_dims(data['surface_air_temperature'],['latitude','longitude'])
                    except Exception as e:
                        print(f"Error stacking sparse data for {indiv_case.case_id_number} from dimensions latitude, longitude: {e}. This is likely because the data is not available for this case.")
                        continue
                try:
                    lat_values = data['latitude'].values
                    lon_values = data['longitude'].values
                except Exception as e:
                    print(f"Error stacking sparse data from dimensions latitude, longitude: {e}")
                    print(indiv_case.case_id_number)
                    print(data)
                    continue
                # Convert longitude values from 0-360 to -180 to 180 for proper 
                # antimeridian handling with Cartopy
                lon_values_180 = utils.convert_longitude_to_180(lon_values)

                # if (np.min(lat_values) <= 0):
                #     print(data)

                ax.scatter(lon_values_180, lat_values, color=color, s=1, alpha=alphas[indiv_event_type], 
                            transform=ccrs.Geodetic(), zorder=zorders[indiv_event_type])

        # # if it is convective, show the PPH and LSRs
        # if (indiv_event_type == 'severe_convection'):
        #     if (case_id is not None and indiv_case.case_id_number != case_id and obs is not None):
        #         continue

        #     # Make sure reports are visible by increasing size and using a distinctive color
        #     # Convert string coordinates to float before plotting
        #     colors = {'tor': 'red','wind': 'blue', 'hail': 'black'}
        #     markers= {'tor': 'o', 'wind': 's', 'hail': '^'}  

        #     # Define zorder values to control plotting order (higher values appear on top)
        #     zorders = {'tor': 10, 'wind': 9, 'hail': 8}
            
        #     # Sort the dataframe by report type to ensure tornadoes are plotted last (on top)
        #     # Create a custom sort order where 'tor' comes last
        #     sort_order = {'hail': 0, 'wind': 1, 'tor': 2}
        #     sorted_df = obs[indiv_case.case_id_number]['lsr_reports'].copy()
        #     sorted_df['sort_key'] = sorted_df['report_type'].map(sort_order)
            
        #     # Group by report type and plot each group with its own color
        #     for report_type, group in sorted_df.sort_values('sort_key').groupby('report_type'):
        #         ax.scatter(group['Longitude'].astype(float), group['Latitude'].astype(float), 
        #                 color=colors.get(report_type, 'gray'), s=20, marker=markers.get(report_type), alpha=0.9,
        #                 transform=ccrs.PlateCarree(), label=f'{report_type.capitalize()} Reports', 
        #                 zorder=zorders.get(report_type, 10))
                
        #     # Plot the PPH outline
        #     pph = obs[indiv_case.case_id_number]['pph']
            
        #     #print (np.max(pph.values), np.min(pph.values))
            
        #     # find the contour outlines
        #     contours = find_contours(pph.values, level=0.01, fully_connected='low')

        #     for contour in contours:
        #         contour_lon_lat_coords = []
        #         for r_idx, c_idx in contour:
        #             # Ensure indices are integers for array lookup
        #             r_idx_int = int(r_idx)
        #             c_idx_int = int(c_idx)

        #             # Get the corresponding longitude and latitude
        #             lon_val = pph.longitude[c_idx_int]
        #             lat_val = pph.latitude[r_idx_int]
        #             contour_lon_lat_coords.append((lon_val, lat_val))

        #         # Convert to a NumPy array for easier handling if needed
        #         contour_lon_lat_coords = np.array(contour_lon_lat_coords)
        #         #print(contour_lon_lat_coords)

        #         # convert from array indices to lat/lon
        #         patch = patches.Polygon(contour_lon_lat_coords, closed=True, facecolor='none', 
        #                                 edgecolor=event_colors['severe_convection'], alpha=1, linewidth=2,
        #                                 transform=ccrs.PlateCarree())
        #         ax.add_patch(patch)
            
        #     if (show_orig_pph):
        #         # Plot the data using contourf
        #         levels = [0.01, .05,.15,.30,.45,.60,.75]  

        #         # Create a custom colormap that sets alpha=0 for values below 0.05
        #         cmap = plt.cm.viridis
        #         norm = mcolors.BoundaryNorm(levels, cmap.N)

        #         # Create the colormap with alpha=0 for values below 0.05
        #         # Create a mask for values below 0.05
        #         mask = np.ma.masked_less(pph, 0.001)
        #         cmap_with_alpha = plt.cm.viridis.copy()
        #         cmap_with_alpha.set_bad('none', alpha=0)  # Set masked values to transparent

        #         contour = ax.contour(pph.longitude, pph.latitude, mask, 
        #                             levels=levels, transform=ccrs.PlateCarree(),
        #                             cmap=cmap_with_alpha, extend='both')
                

        
    # Create a custom legend for event types
    if (event_type is not None):
        # if we are only plotting one event type, only show that in the legend
        legend_elements = [
            Patch(facecolor=event_colors[event_type], alpha=0.9, label=f'{event_type.replace("_", " ").title()} (n = %d)' % counts_by_type[event_type]),
        ]
    else:
        # otherwise show all event types in the legend
        legend_elements = [
            Patch(facecolor=event_colors['heat_wave'], alpha=0.9, label='Heat Wave (n = %d)' % counts_by_type['heat_wave']),
            Patch(facecolor=event_colors['freeze'], alpha=0.9, label='Freeze (n = %d)' % counts_by_type['freeze']),
            Patch(facecolor=event_colors['severe_convection'], alpha=0.9, label='Convection (n = %d)' % counts_by_type['severe_convection']),
            Patch(facecolor=event_colors['atmospheric_river'], alpha=0.9, label='Atmospheric River (n = %d)' % counts_by_type['atmospheric_river']),
            Patch(facecolor=event_colors['tropical_cyclone'], alpha=0.9, label='Tropical Cyclone (n = %d)' % counts_by_type['tropical_cyclone']),
        ]
    # Create a larger legend by specifying a larger font size in the prop dictionary
    legend = ax.legend(handles=legend_elements, loc='lower left', framealpha=1, frameon=True, borderpad=0.5, handletextpad=0.8, handlelength=2.5)
    legend.set_zorder(10)

    if (event_type is None):
        title = 'ExtremeWeatherBench Cases (n = %d)' % sum(counts_by_type.values())
    else:
        title = f'ExtremeWeatherBench Cases: {event_type.replace("_", " ").title()} (n = %d)' % counts_by_type[event_type]
    
    ax.set_title(title, loc='left', fontsize=20)
    
    # save if there is a filename specified (otherwise the user just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches='tight', dpi=300)


def plot_boxes(box_list, box_names, title, filename=None):
    # plot all cases on one giant world map
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
        
    # Add coastlines and gridlines
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='white')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=1, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Define colors for each event type
    # use seaborn color palette for colorblind friendly colors
    sns_palette = sns.color_palette("tab10")
    sns.set_style("whitegrid")

    # Plot boxes for each case
    for box in box_list:
        plot_polygon_outline(box, ax, color='blue', alpha=1)

    plt.legend(loc='lower left', fontsize=12)
    ax.set_title(title, loc='left', fontsize=20)
    
    # save if there is a filename specified (otherwise the user just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches='tight', dpi=300)

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
            np.linspace(
                0, threshold_pos - 0.02, len(lo_colors)
            ),  # Colors up to white
            # [threshold_pos],  # White position
            np.linspace(
                threshold_pos + 0.02, 1, len(hi_colors)
            ),  # Colors after white
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
    dataset = (dataset.drop_vars(['time', 'dayofyear', 'hour'])
               .assign_coords(time=time_dim))

    return dataset

def generate_heatwave_dataset(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Calculate the times where regional average of temperature is above the climatology.
    
    Args:
        era5: ERA5 dataset containing 2m_temperature
        climatology: BB climatology containing surface_temperature_85th_percentile
        single_case: cases.IndividualCase object with associated metadata
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
        single_case.location.longitude_min < 0 or 
        single_case.location.longitude_min > 180 
        ) and (
        single_case.location.longitude_max > 0 and
        single_case.location.longitude_max < 180
            ):
                merged_dataset = utils.convert_longitude_to_180(merged_dataset)
    merged_dataset = merged_dataset.sel(
        latitude=slice(single_case.location.latitude_max, single_case.location.latitude_min),
        longitude=slice(single_case.location.longitude_min, single_case.location.longitude_max),
    )
    return merged_dataset

def generate_heatwave_plots(
    heatwave_dataset: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Plot the max timestep of the heatwave event and the average regional temperature time series
    on separate plots.
    
    Args:
        heatwave_dataset: contains 2m_temperature, surface_temperature_85th_percentile,
        time, latitude, longitude
        single_case: cases.IndividualCase object with associated metadata
    """
    time_based_heatwave_dataset = heatwave_dataset.mean(["latitude", "longitude"])
    # Plot 1: Min timestep of the heatwave event
    fig1, ax1 = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Select the timestep with the maximum spatially averagedtemperature
    subset_timestep = (
        time_based_heatwave_dataset['time'][time_based_heatwave_dataset["2m_temperature"]
        .argmax()]
    )
    # Mask places where temperature >= 85th percentile climatology
    temp_data = heatwave_dataset["2m_temperature"] - 273.15
    climatology_data = heatwave_dataset["surface_temperature_85th_percentile"] - 273.15
    
    # Create mask for values where temp > climatology (heatwave condition)
    mask = temp_data > climatology_data
    
    # Apply mask to temperature data
    masked_temp = temp_data.where(mask)
    cmap, norm = celsius_colormap_and_normalize()
    im = (
        masked_temp
        .sel(time=subset_timestep)
        .plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
        )
    )
    (
        temp_data
        .sel(time=subset_timestep)
        .plot.contour(
            ax=ax1,
            levels=[0],
            colors='r',
            linewidths=0.75,
            ls=':',
            transform=ccrs.PlateCarree(),
        )
    )   
    # Add coastlines and gridlines
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, edgecolor="black")
    ax1.add_feature(cfeature.LAKES, edgecolor="black")
    ax1.add_feature(cfeature.RIVERS, edgecolor=[ 0.59375 , 0.71484375, 0.8828125 ],alpha=0.5)
    ax1.add_feature(cfeature.STATES, edgecolor="grey")
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True,alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}
    ax1.set_title('') # clears the default xarray title
    ax1.set_title(
        f"Temperature Where > 85th Percentile Climatology\n"
        f"{single_case.title}, Case ID {single_case.case_id_number}\n"
        f"{heatwave_dataset['time'].sel(time=subset_timestep).dt.strftime('%Y-%m-%d %Hz').values}", 
        loc='left'
    )
    # Add the location coordinate as a dot on the map
    ax1.tick_params(axis='y', which='major', labelsize=12)
    # Create a colorbar with the same height as the plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig1.colorbar(im, cax=cax, label="Temp > 85th Percentile (C)")
    cbar.set_label("Temperature (C)", size=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_spatial.png",transparent=True)
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
    ax2.set_title('')
    ax2.set_title("Spatially Averaged Heatwave Event vs 85th Percentile Climatology", 
    fontsize=14, loc='left')
    ax2.set_ylabel("Temperature (C)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_tick_params(rotation=45, labelsize=10, pad=0.0001,)
    ax2.tick_params(axis="y", labelsize=12)
    
    # Create legend handles including the axvspan
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='-.', linewidth=0.75, 
                   label='2m Temperature, 85th Percentile'),
        plt.Line2D([0], [0], color='tab:red', linestyle='-', linewidth=1.5, 
                   label='2m Temperature'),
        Patch(facecolor='red', alpha=0.1, label='Above 85th Percentile')
    ]
    ax2.legend(handles=legend_elements, fontsize=12)
    
    ax2.tick_params(axis='y', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_timeseries.png",transparent=True)
    plt.show()


def generate_freeze_dataset(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Calculate the times where regional average of temperature is below the climatology.
    
    Args:
        era5: ERA5 dataset containing 2m_temperature
        climatology: BB climatology containing surface_temperature_15th_percentile
        single_case: cases.IndividualCase object with associated metadata
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
        single_case.location.longitude_min < 0 or 
        single_case.location.longitude_min > 180 
        ) and (
        single_case.location.longitude_max > 0 and
        single_case.location.longitude_max < 180
            ):
                merged_dataset = utils.convert_longitude_to_180(merged_dataset)
    merged_dataset = merged_dataset.sel(
        latitude=slice(single_case.location.latitude_max, single_case.location.latitude_min),
        longitude=slice(single_case.location.longitude_min, single_case.location.longitude_max),
    )
    return merged_dataset

def generate_freeze_plots(
    freeze_dataset: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Plot the max timestep of the freeze event and the average regional temperature time series
    on separate plots.
    
    Args:
        freeze_dataset: contains 2m_temperature, surface_temperature_15th_percentile,
        time, latitude, longitude
        single_case: cases.IndividualCase object with associated metadata
    """
    time_based_freeze_dataset = freeze_dataset.mean(["latitude", "longitude"])
    # Plot 1: Min timestep of the freeze event
    fig1, ax1 = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Select the timestep with the maximum spatially averagedtemperature
    subset_timestep = (
        time_based_freeze_dataset['time'][time_based_freeze_dataset["2m_temperature"]
        .argmin()]
    )
    # Mask places where temperature >= 15th percentile climatology
    temp_data = freeze_dataset["2m_temperature"] - 273.15
    climatology_data = freeze_dataset["surface_temperature_15th_percentile"] - 273.15
    
    # Create mask for values where temp < climatology (freeze condition)
    mask = temp_data < climatology_data
    
    # Apply mask to temperature data
    masked_temp = temp_data.where(mask)
    cmap, norm = celsius_colormap_and_normalize()
    im = (
        masked_temp
        .sel(time=subset_timestep)
        .plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
        )
    )
    (
        temp_data
        .sel(time=subset_timestep)
        .plot.contour(
            ax=ax1,
            levels=[0],
            colors='r',
            linewidths=0.75,
            ls=':',
            transform=ccrs.PlateCarree(),
        )
    )   
    # Add coastlines and gridlines
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, edgecolor="black")
    ax1.add_feature(cfeature.LAKES, edgecolor="black")
    ax1.add_feature(cfeature.RIVERS, edgecolor=[ 0.59375 , 0.71484375, 0.8828125 ],alpha=0.5)
    ax1.add_feature(cfeature.STATES, edgecolor="grey")
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True,alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}
    ax1.set_title('') # clears the default xarray title
    ax1.set_title(
        f"Temperature Where < 15th Percentile Climatology\n"
        f"{single_case.title}, Case ID {single_case.case_id_number}\n"
        f"{freeze_dataset['time'].sel(time=subset_timestep).dt.strftime('%Y-%m-%d %Hz').values}", 
        loc='left'
    )
    # Add the location coordinate as a dot on the map
    ax1.tick_params(axis='y', which='major', labelsize=12)
    # Create a colorbar with the same height as the plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig1.colorbar(im, cax=cax, label="Temp < 15th Percentile (C)")
    cbar.set_label("Temperature (C)", size=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_spatial.png",transparent=True)
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
    ax2.set_title('')
    ax2.set_title("Spatially Averaged Freeze Event vs 15th Percentile Climatology", 
    fontsize=14, loc='left')
    ax2.set_ylabel("Temperature (C)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_tick_params(rotation=45, labelsize=10, pad=0.0001,)
    ax2.tick_params(axis="y", labelsize=12)
    
    # Create legend handles including the axvspan
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='-.', linewidth=0.75, 
                   label='2m Temperature, 15th Percentile'),
        plt.Line2D([0], [0], color='tab:red', linestyle='-', linewidth=1.5, 
                   label='2m Temperature'),
        Patch(facecolor='red', alpha=0.1, label='Below 15th Percentile')
    ]
    ax2.legend(handles=legend_elements, fontsize=12)
    
    ax2.tick_params(axis='y', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"case_{single_case.case_id_number}_timeseries.png",transparent=True)
    plt.show()