"""Helper functions to identify the date ranges of heat waves and freeze events."""

import xarray as xr
import numpy as np
import pandas as pd
from extremeweatherbench import utils
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter


def subset_mask_and_datasets(
    era5,
    climatology,
    actual_start_date,
    actual_end_date,
    location_center,
    box_length_width_in_km,
):
    # Recalculate the era5_event and mask with the new window
    era5_event = era5[["2m_temperature"]].sel(
        time=slice(actual_start_date, actual_end_date)
    )
    era5_event = era5_event.sel(time=utils.is_6_hourly(era5_event["time.hour"]))
    subset_climatology = utils.convert_day_yearofday_to_time(
        climatology, np.unique(era5_event.time.dt.year.values)[0]
    ).rename_vars({"2m_temperature": "2m_temperature_85th_percentile"})

    merged_dataset = xr.merge([subset_climatology, era5_event], join="inner")
    merged_dataset = utils.convert_longitude_to_180(merged_dataset)
    merged_dataset = utils.clip_dataset_to_bounding_box(
        merged_dataset, location_center, box_length_width_in_km
    )
    merged_dataset = utils.remove_ocean_gridpoints(merged_dataset)
    time_based_merged_dataset = merged_dataset.mean(["latitude", "longitude"])

    mask = (
        time_based_merged_dataset["2m_temperature"]
        > time_based_merged_dataset["2m_temperature_85th_percentile"]
    )
    return mask.compute(), merged_dataset, time_based_merged_dataset


def find_heatwave_events(
    era5,
    climatology,
    start_date,
    end_date,
    location_center,
    box_length_width_in_km,
    threshold=48,
    plot=True,
):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask, merged_dataset, time_based_merged_dataset = subset_mask_and_datasets(
        era5, climatology, start_date, end_date, location_center, box_length_width_in_km
    )
    before = True
    after = True
    while before or after:
        # Check if there are 48 hours before and after the event
        last_true_time = mask.where(mask, drop=True).time[-1].values
        event_end_duration = (mask.time[-1] - last_true_time).values.astype(
            "timedelta64[h]"
        )
        first_true_time = mask.where(mask, drop=True).time[0].values
        event_start_duration = (mask.time[0] - first_true_time).values.astype(
            "timedelta64[h]"
        )
        if abs(event_start_duration) >= np.timedelta64(6, "h"):
            before = False
        else:
            start_date -= pd.DateOffset(hours=6)
            mask, merged_dataset, time_based_merged_dataset = subset_mask_and_datasets(
                era5,
                climatology,
                start_date,
                end_date,
                location_center,
                box_length_width_in_km,
            )
        if abs(event_end_duration) >= np.timedelta64(6, "h"):
            after = False
        else:
            end_date += pd.DateOffset(hours=6)
            mask, merged_dataset, time_based_merged_dataset = subset_mask_and_datasets(
                era5,
                climatology,
                start_date,
                end_date,
                location_center,
                box_length_width_in_km,
            )

    start_date -= pd.DateOffset(hours=42)
    end_date += pd.DateOffset(hours=42)

    mask, merged_dataset, time_based_merged_dataset = subset_mask_and_datasets(
        era5, climatology, start_date, end_date, location_center, box_length_width_in_km
    )
    if plot:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 20), gridspec_kw={"height_ratios": [1, 1]}
        )
        plt.subplots_adjust(hspace=0.3)
        ax1.remove()
        ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
        subset_timestep = (
            merged_dataset["2m_temperature"].mean(["latitude", "longitude"])
            == merged_dataset["2m_temperature"].mean(["latitude", "longitude"]).max()
        )
        im = (
            merged_dataset["2m_temperature"]
            .sel(time=subset_timestep)
            .plot(
                ax=ax1,
                transform=ccrs.PlateCarree(),
                cmap="coolwarm",
                add_colorbar=False,
            )
        )
        # Add coastlines and gridlines
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=":")
        ax1.add_feature(cfeature.LAND, edgecolor="black")
        ax1.add_feature(cfeature.LAKES, edgecolor="black")
        ax1.add_feature(cfeature.RIVERS, edgecolor="black")
        # Add gridlines
        gl = ax1.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        ax1.set_title(
            f"2m Temperature, valid at {merged_dataset['time'].sel(time=subset_timestep).dt.strftime('%Y-%m-%d %Hz').values[0]}",
            fontsize=16,
        )
        # Add the location coordinate as a dot on the map
        ax1.plot(
            location_center["longitude"],
            location_center["latitude"],
            "ko",
            markersize=10,
            transform=ccrs.PlateCarree(),
        )

        # Create a colorbar with the same height as the plot
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, label="2m Temperature (K)")

        for variable in time_based_merged_dataset:
            time_based_merged_dataset[variable].plot(ax=ax2, label=variable)
        ax2.legend(fontsize=12)
        ax2.set_title("Heatwave Event vs 85th Percentile Climatology")
        plt.show()
    return mask, merged_dataset, time_based_merged_dataset
