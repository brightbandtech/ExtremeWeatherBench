"""Helper functions to identify the date ranges of heat waves and freeze events."""

import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from matplotlib import dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import extremeweatherbench.cases as cases
import extremeweatherbench.utils as utils

sns.set_theme(style="whitegrid", context="talk")


def subset_event_and_mask_climatology(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    actual_start_date: datetime.datetime,
    actual_end_date: datetime.datetime,
    single_case: cases.IndividualCase,
):
    """Calculate the times where regional average of temperature exceeds the
    climatology."""
    era5_event = era5[["2m_temperature"]].sel(
        time=slice(actual_start_date, actual_end_date)
    )
    era5_event = era5_event.sel(time=era5_event["time.hour"].isin([0, 6, 12, 18]))
    subset_climatology = convert_day_yearofday_to_time(
        climatology, np.unique(era5_event.time.dt.year.values)[0]
    ).rename_vars({"2m_temperature": "2m_temperature_85th_percentile"})
    location = single_case.location.as_geopandas().total_bounds
    merged_dataset = xr.merge([subset_climatology, era5_event], join="inner")
    merged_dataset = merged_dataset.sel(
        latitude=slice(location[1], location[3]),
        longitude=slice(location[0], location[2]),
    )
    merged_dataset = utils.remove_ocean_gridpoints(merged_dataset)
    time_averaged_merged_dataset = merged_dataset.mean(["latitude", "longitude"])

    mask = (
        time_averaged_merged_dataset["2m_temperature"]
        > time_averaged_merged_dataset["2m_temperature_85th_percentile"]
    )
    return mask.compute(), merged_dataset


def find_heatwave_events(
    era5: xr.Dataset,
    climatology: xr.Dataset,
    single_case: cases.IndividualCase,
    plot: bool = True,
):
    """Find the start and end dates of heatwave events, stepping +- 6 hours until <
    climatology timesteps are located."""
    start_date = pd.to_datetime(single_case.start_date)
    end_date = pd.to_datetime(single_case.end_date)
    location_center = single_case.location
    era5_event = era5[["2m_temperature"]].sel(time=slice(start_date, end_date))
    era5_event = era5_event.sel(time=era5_event["time.hour"].isin([0, 6, 12, 18]))
    subset_climatology = convert_day_yearofday_to_time(
        climatology, np.unique(era5_event.time.dt.year.values)[0]
    ).rename_vars({"2m_temperature": "2m_temperature_85th_percentile"})

    mask, merged_dataset = subset_event_and_mask_climatology(
        era5, subset_climatology, start_date, end_date, single_case
    )
    before = True
    after = True
    while before or after:
        # Check if there are 48 hours before and after the event
        try:
            last_true_time = mask.where(mask, drop=True).time[-1].values
            event_end_duration = (mask.time[-1] - last_true_time).values.astype(
                "timedelta64[h]"
            )
            first_true_time = mask.where(mask, drop=True).time[0].values
            event_start_duration = (mask.time[0] - first_true_time).values.astype(
                "timedelta64[h]"
            )
            if np.datetime64(last_true_time, "D") == np.datetime64("2022-12-31"):
                after = False
            if abs(event_start_duration) >= np.timedelta64(6, "h"):
                before = False
            else:
                start_date -= pd.DateOffset(hours=6)
                mask, merged_dataset, time_based_merged_dataset = (
                    subset_event_and_mask_climatology(
                        era5, climatology, start_date, end_date, single_case
                    )
                )
            if abs(event_end_duration) >= np.timedelta64(6, "h"):
                after = False
            else:
                end_date += pd.DateOffset(hours=6)
                mask, merged_dataset, time_based_merged_dataset = (
                    subset_event_and_mask_climatology(
                        era5, climatology, start_date, end_date, single_case
                    )
                )
        except IndexError:
            print(f"No dates valid for {location_center}, {start_date}, {end_date}")
            before = False
            after = False
    start_date -= pd.DateOffset(hours=42)
    end_date += pd.DateOffset(hours=42)

    mask, merged_dataset, time_based_merged_dataset = subset_event_and_mask_climatology(
        era5, climatology, start_date, end_date, single_case
    )
    if plot:
        case_plot(merged_dataset, time_based_merged_dataset, single_case)
    return (
        mask,
        time_based_merged_dataset.time.min().values,
        time_based_merged_dataset.time.max().values,
    )


def case_plot(
    merged_dataset: xr.Dataset,
    time_based_merged_dataset: xr.Dataset,
    single_case: cases.IndividualCase,
):
    """Plot the max timestep of the heatwave event, the average regional temperature
    time series, and the associated climatology."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 10), gridspec_kw={"height_ratios": [1, 1]}
    )
    plt.subplots_adjust(hspace=0.3)
    ax1.remove()
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    subset_timestep = (
        merged_dataset["2m_temperature"].mean(["latitude", "longitude"])
        == merged_dataset["2m_temperature"].mean(["latitude", "longitude"]).max()
    )
    im = (
        (
            merged_dataset["2m_temperature"]
            - merged_dataset["2m_temperature_85th_percentile"]
        )
        .sel(time=subset_timestep)
        .plot(
            ax=ax1,
            transform=ccrs.PlateCarree(),
            cmap="inferno",
            add_colorbar=False,
            vmin=0,
            vmax=20,
        )
    )
    # Add coastlines and gridlines
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, edgecolor="black")
    ax1.add_feature(cfeature.LAKES, edgecolor="black")
    ax1.add_feature(cfeature.RIVERS, edgecolor="black")
    ax1.add_feature(cfeature.STATES, edgecolor="grey")
    # Add gridlines
    gl = ax1.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}
    time_subset = merged_dataset["time"].sel(time=subset_timestep)
    ax1.set_title(
        f"Event ID {single_case.case_id_number}: 2m Temperature, "
        f"{time_subset.dt.strftime('%Y-%m-%d %Hz').values[0]}",
        fontsize=12,
    )
    # Add the location coordinate as a dot on the map
    # Get center coordinates from the region bounds
    bounds = single_case.location.as_geopandas().total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2  # (lon_min + lon_max) / 2
    center_lat = (bounds[1] + bounds[3]) / 2  # (lat_min + lat_max) / 2

    ax1.plot(
        center_lon,
        center_lat,
        "ko",
        markersize=10,
        transform=ccrs.PlateCarree(),
    )
    # Create a colorbar with the same height as the plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, label="Temp > 85th Percentile (C)")
    cbar.set_label("Temp > 85th Percentile (C)", size=14)
    lss = ["-.", "-"]
    lc = ["k", "tab:red"]
    lws = [0.75, 1.5]
    for i, variable in enumerate(time_based_merged_dataset):
        (time_based_merged_dataset[variable] - 273.15).plot(
            ax=ax2, label=variable, lw=lws[i], ls=lss[i], c=lc[i]
        )
    ax2.legend(fontsize=12)
    mask = (
        time_based_merged_dataset["2m_temperature"]
        > time_based_merged_dataset["2m_temperature_85th_percentile"]
    )
    start = None
    for i, val in enumerate(mask.values):
        if val and start is None:
            start = time_based_merged_dataset.time[i].values
        elif not val and start is not None:
            ax2.axvspan(
                start,
                time_based_merged_dataset.time[i].values,
                color="red",
                alpha=0.1,
            )
            start = None
    if start is not None:
        ax2.axvspan(
            start, time_based_merged_dataset.time[-1].values, color="red", alpha=0.1
        )
    ax2.set_title("Heatwave Event vs 85th Percentile Climatology", fontsize=14)
    ax2.set_ylabel("Temperature (C)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_tick_params(rotation=45, labelsize=10, pad=0.0001)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.legend(["2m Temperature, 85th Percentile", "2m Temperature"], fontsize=12)
    plt.show()


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
    dataset = dataset.stack(time=("dayofyear", "hour")).drop(["dayofyear", "hour"])
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(time=time_dim)

    return dataset
