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

def plot_polygon(polygon, ax, color='yellow', alpha=0.5, my_zorder=1):
    """Plot a shapely Polygon on a Cartopy axis."""
    if polygon is None:
        return
    patch = patches.Polygon(
        polygon.exterior.coords,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
        zorder=my_zorder,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(patch)

def plot_polygon_outline(polygon, ax, color='yellow', alpha=0.5, my_zorder=1):
    """Plot a shapely Polygon outline on a Cartopy axis."""
    if polygon is None:
        return
    patch = patches.Polygon(
        polygon.exterior.coords,
        closed=True,
        facecolor='none',
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
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
            if (not shapely.intersects(indiv_case.location.geopandas.geometry[0], bounding_box_polygon)):
                #print(f"Skipping case {indiv_case.case_id_number} as it is outside the bounding box.")
                continue

        # count the events by type
        counts_by_type[indiv_event_type] += 1

        # Plot the case geopandas info
        if (event_type is None or indiv_event_type == event_type):
            if (fill_boxes):
                # to handle wrapping around the prime meridian, we can't use geopandas plot (and besides it is slow)
                # instead we have multi-polygon patches if it wraps around and we need to plot each polygon separately
                if isinstance(indiv_case.location.geopandas.geometry.iloc[0], shapely.geometry.MultiPolygon):
                    for poly in indiv_case.location.geopandas.geometry.iloc[0].geoms:
                        plot_polygon(poly, ax, color=color, alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
                else:
                    plot_polygon(indiv_case.location.geopandas.geometry.iloc[0], ax, color=color, 
                                alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
            else:
                # to handle wrapping around the prime meridian, we can't use geopandas plot (and besides it is slow)
                # instead we have multi-polygon patches if it wraps around and we need to plot each polygon separately
                if isinstance(indiv_case.location.geopandas.geometry.iloc[0], shapely.geometry.MultiPolygon):
                    for poly in indiv_case.location.geopandas.geometry.iloc[0].geoms:
                        plot_polygon_outline(poly, ax, color=color, alpha=box_alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
                else:
                    plot_polygon_outline(indiv_case.location.geopandas.geometry.iloc[0], ax, color=color, 
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
    zorders = {'freeze': 10, 'heat_wave': 9, 'atmospheric_river': 2, 'tropical_cyclone': 1, 'severe_convection': 0}
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
            if isinstance(indiv_case.location.geopandas.geometry.iloc[0], shapely.geometry.MultiPolygon):
                for poly in indiv_case.location.geopandas.geometry.iloc[0].geoms:
                    plot_polygon_outline(poly, ax, color=color, alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
            else:
                plot_polygon_outline(indiv_case.location.geopandas.geometry.iloc[0], ax, color=color, 
                                        alpha=alphas[indiv_event_type], my_zorder=zorders[indiv_event_type])
                
            # grab the target data for this case
            my_target_info = [n[1] for n in targets if n[0] == indiv_case.case_id_number and n[1].attrs['source'] != 'ERA5']

            # make a scatter plot of the target points (for hot/cold events)
            if (indiv_event_type in ['heat_wave', 'freeze', 'tropical_cyclone'] and len(my_target_info) > 0):
                # Get the sparse array data
                data = my_target_info[0]
                data = utils.stack_sparse_data_from_dims(data['surface_air_temperature'],['latitude','longitude'])
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
    ax.legend(handles=legend_elements, loc='lower left', framealpha=1, frameon=True, borderpad=0.5, handletextpad=0.8, handlelength=2.5)
    if (event_type is None):
        title = 'ExtremeWeatherBench Cases (n = %d)' % sum(counts_by_type.values())
    else:
        title = f'ExtremeWeatherBench Cases: {event_type.replace("_", " ").title()} (n = %d)' % counts_by_type[event_type]
    
    ax.set_title(title, loc='left', fontsize=20)
    
    # save if there is a filename specified (otherwise the user just wants to see the plot)
    if filename is not None:
        plt.savefig(filename, transparent=False, bbox_inches='tight', dpi=300)