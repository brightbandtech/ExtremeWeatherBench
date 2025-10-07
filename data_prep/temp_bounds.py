"""Temperature extreme bounds detection and analysis.

This module provides comprehensive functions for detecting temperature extremes
(heatwaves and freezes) with iterative edge expansion, calculating consecutive
extreme days, and creating detailed visualization plots.

The module includes:
- Iterative edge expansion (2 degrees when >=50% of edge gridpoints have extremes)
- Heatwave detection (3 consecutive days of >=85th percentile)
- Freeze detection (3 consecutive days of <=15th percentile)
- Maximum 10 iterations for expansion
- Calculation of total consecutive extreme days
- Two-panel plotting with peak temperature day
- Integration with ARCO ERA5 and climatology datasets
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from tqdm.auto import tqdm

from extremeweatherbench import cases, utils

# Set up logging
logger = logging.getLogger(__name__)

# Data source URIs
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
TEMP_CLIMATOLOGY_URI = "gs://extremeweatherbench/datasets/surface_air_temperature_1990_2019_climatology.zarr"  # noqa: E501


def load_era5_slice_efficiently(
    region_bounds: Dict, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> xr.DataArray:
    """Load ERA5 temperature data efficiently for a specific region.

    Args:
        region_bounds: Dictionary containing latitude and longitude bounds.
        start_date: Start date for data extraction.
        end_date: End date for data extraction.

    Returns:
        ERA5 temperature data for the specified region and time period.
    """
    ds = xr.open_zarr(ARCO_ERA5_FULL_URI, chunks={"time": 48})

    # 1. Variable selection first
    temp_data = ds["2m_temperature"]

    # 2. Time subset
    temp_time = temp_data.sel(time=slice(start_date, end_date))
    temp_time = temp_time.sel(time=temp_time.time.dt.hour.isin([0, 6, 12, 18]))
    if (
        region_bounds["longitude_min"] < region_bounds["longitude_max"]
        and region_bounds["longitude_max"] <= 180
    ):
        temp_time = utils.convert_longitude_to_180(temp_time, "longitude")
    # 3. Spatial subset
    temp_subset = temp_time.sel(
        latitude=slice(region_bounds["latitude_max"], region_bounds["latitude_min"]),
        longitude=slice(region_bounds["longitude_min"], region_bounds["longitude_max"]),
    )

    # Load and rename to match expected format
    temp_loaded = temp_subset.load()
    temp_loaded = temp_loaded.rename({"time": "valid_time"})

    # Check for empty spatial dimensions
    if (
        temp_loaded.sizes.get("latitude", 0) == 0
        or temp_loaded.sizes.get("longitude", 0) == 0
    ):
        lat_size = temp_loaded.sizes.get("latitude", 0)
        lon_size = temp_loaded.sizes.get("longitude", 0)
        print(
            f"  Warning: Empty spatial region loaded - lat: {lat_size}, lon: {lon_size}"
        )
        print(f"    Region bounds: {region_bounds}")

    return temp_loaded


def load_temperature_climatology() -> xr.Dataset:
    """Load the temperature climatology dataset.

    Returns:
        Temperature climatology dataset with quantiles and day-of-year data.
    """
    print("Loading surface air temperature climatology data...")
    temp_climatology = xr.open_zarr(TEMP_CLIMATOLOGY_URI, chunks="auto")
    print(f"Loaded climatology data with shape: {dict(temp_climatology.sizes)}")
    return temp_climatology


def calculate_extent_bounds(
    left_lon: float,
    right_lon: float,
    bottom_lat: float,
    top_lat: float,
    extent_buffer: float = 250,
    extent_units: Literal["degrees", "km"] = "km",
):
    """Calculate extent bounds for a region.

    Args:
        left_lon: Left longitude boundary.
        right_lon: Right longitude boundary.
        bottom_lat: Bottom latitude boundary.
        top_lat: Top latitude boundary.
        extent_buffer: Buffer size for extent calculation.
        extent_units: Units for the buffer (degrees or km).

    Returns:
        Region object with calculated bounds.
    """
    if extent_units == "km":
        # Convert km to degrees (approximate)
        lat_deg_per_km = 1.0 / 111.0
        lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians((bottom_lat + top_lat) / 2)))
        lat_buffer = extent_buffer * lat_deg_per_km
        lon_buffer = extent_buffer * lon_deg_per_km
    else:
        lat_buffer = extent_buffer
        lon_buffer = extent_buffer

    new_bottom_lat = bottom_lat - lat_buffer
    new_top_lat = top_lat + lat_buffer
    new_left_lon = left_lon - lon_buffer
    new_right_lon = right_lon + lon_buffer

    # Ensure bounds are within valid ranges
    new_bottom_lat = max(-90, new_bottom_lat)
    new_top_lat = min(90, new_top_lat)
    new_left_lon = max(-180, new_left_lon)
    new_right_lon = min(180, new_right_lon)

    # Return bounds as dictionary
    return {
        "latitude_min": new_bottom_lat,
        "latitude_max": new_top_lat,
        "longitude_min": new_left_lon,
        "longitude_max": new_right_lon,
    }


def create_temperature_extreme_mask_simple(
    hourly_temp_data: xr.DataArray,
    percentile_threshold: xr.DataArray,
    extreme_type: Literal["heatwave", "freeze"] = "heatwave",
    min_consecutive_days: int = 3,
) -> xr.DataArray:
    """Create mask for temperature extremes using all 6-hourly timesteps.

    Args:
        hourly_temp_data: 6-hourly temperature data with time dimension.
        percentile_threshold: Day-of-year specific percentile threshold for each
            timestep.
        extreme_type: Type of extreme to detect.
        min_consecutive_days: Minimum consecutive calendar days
            (= min_consecutive_days * 4 timesteps). Default is 3 days (12 timesteps).

    Returns:
        Binary mask for all timesteps where consecutive timestep criteria are met.
    """
    # Auto-detect time dimension
    time_dim = "time" if "time" in hourly_temp_data.dims else "valid_time"

    print("Checking each 6-hourly timestep against percentile threshold...")

    # Step 1: Create binary mask for each 6-hourly timestep
    if extreme_type == "heatwave":
        # For heatwaves: 6-hourly temperature > 85th percentile
        timestep_extreme_mask = hourly_temp_data > percentile_threshold
    else:  # freeze
        # For freezes: 6-hourly temperature < 15th percentile
        timestep_extreme_mask = hourly_temp_data < percentile_threshold

    # Step 2: Calculate consecutive timesteps (not days)
    min_timesteps = min_consecutive_days * 4
    print(
        f"    Requiring at least {min_timesteps} consecutive "
        f"6-hourly timesteps ({min_consecutive_days} days)..."
    )

    # Calculate consecutive timesteps using rolling window
    print("    Calculating consecutive timestep counts...")
    consecutive_counts = timestep_extreme_mask.rolling(
        {time_dim: min_timesteps}, center=False
    ).sum()

    # Only keep timesteps that are part of sequences >= min_consecutive_timesteps
    extreme_mask = xr.where(
        consecutive_counts >= min_timesteps,
        consecutive_counts,
        0,
    )

    return extreme_mask


def get_climatology_for_dates(
    climatology_data: xr.DataArray,
    dates: pd.DatetimeIndex,
) -> xr.DataArray:
    """Get climatology data for specific dates.

    Args:
        climatology_data: Climatology data array.
        dates: DatetimeIndex of dates to select.

    Returns:
        Subset of climatology data for the specified dates.
    """
    # Convert dayofyear and hour to time coordinate
    time_dim = pd.date_range(
        start=f"{dates.year[0]}-01-01",
        periods=len(climatology_data["dayofyear"]) * len(climatology_data["hour"]),
        freq="6h",
    )
    clim_subset = climatology_data.stack(time=("dayofyear", "hour")).drop_vars(
        ["dayofyear", "hour"]
    )
    clim_subset = clim_subset.assign_coords(time=time_dim)

    # Select climatology for these days
    clim_subset = clim_subset.sel(time=dates)

    return clim_subset


def process_temperature_extreme_case(
    era5_data: xr.Dataset,
    temp_climatology: xr.Dataset,
    extreme_type: Literal["heatwave", "freeze"],
    case_info: Dict,
    min_consecutive_days: int = 3,
) -> Dict:
    """Process a single case for temperature extreme bounds detection.

    Args:
        era5_data: ERA5 temperature data for the case.
        temp_climatology: Temperature climatology dataset.
        extreme_type: Type of extreme to detect.
        case_info: Case information dictionary.
        min_consecutive_days: Minimum consecutive days for valid event.

    Returns:
        Results dictionary with bounds and metadata.
    """
    # Get temperature data
    temp_data = era5_data["2m_temperature"]

    # Compute daily minimum temperature
    # Get day-of-year specific percentile data
    print(f"  Getting day-of-year specific percentile for {extreme_type}...")
    if extreme_type == "heatwave":
        percentile_value = 0.85
    else:  # freeze
        percentile_value = 0.15

    # Get climatology for the specific dates in our time series
    time_dim = "time" if "time" in era5_data.dims else "valid_time"
    dates = pd.to_datetime(era5_data[time_dim].values)
    percentile_data = (
        get_climatology_for_dates(temp_climatology["2m_temperature"], dates)
        .sel(quantile=percentile_value)
        .rename({"time": "valid_time"})
    )

    # Ensure percentile_data has the same coordinates as temp_data
    percentile_data = percentile_data.reindex_like(temp_data, method="nearest")

    print(f"  Creating {extreme_type} mask...")
    percentile_str = f"{percentile_value * 100:.0f}th"
    print(f"    Criteria: Each 6-hourly timestep must >= {percentile_str} percentile")
    min_timesteps = min_consecutive_days * 4
    print(
        f"    Requirement: At least {min_timesteps} consecutive "
        f"6-hourly timesteps ({min_consecutive_days} days)"
    )
    # Create extreme event mask using ALL timesteps with dual criteria
    extreme_mask = create_temperature_extreme_mask_simple(
        temp_data,
        percentile_data,
        extreme_type,
        min_consecutive_days,
    )

    # Remove ocean gridpoints using existing utils function
    print("  Removing ocean gridpoints from extreme mask...")

    # Check if extreme_mask has any spatial dimensions
    if (
        extreme_mask.sizes.get("latitude", 0) == 0
        or extreme_mask.sizes.get("longitude", 0) == 0
    ):
        lat_size = extreme_mask.sizes.get("latitude", 0)
        lon_size = extreme_mask.sizes.get("longitude", 0)
        print(f"  Warning: Empty extreme mask - lat: {lat_size}, lon: {lon_size}")
        return {
            "bounds": None,
            "mask": xr.DataArray(
                np.array([]).reshape(0, 0),
                dims=["latitude", "longitude"],
                coords={"latitude": [], "longitude": []},
            ),
            "extreme_mask": extreme_mask,
            "temperature_data": era5_data["2m_temperature"],
            "percentile_temperature_data": percentile_data,
            "metadata": {"extreme_type": extreme_type, "total_points": 0},
        }

    # Get maximum consecutive days at each location across all time steps
    # (before ocean masking)
    time_dim = "time" if "time" in extreme_mask.dims else "valid_time"
    mask_total_before = extreme_mask.max(dim=time_dim)

    total_before = mask_total_before.sum().values
    print(f"  Total extreme grid points before ocean masking: {total_before}")

    # Create temporary dataset to use the remove_ocean_gridpoints
    # function
    temp_dataset = xr.Dataset({"extreme_mask": extreme_mask})
    land_only_dataset = utils.remove_ocean_gridpoints(temp_dataset)
    # Keep NaN values for ocean points instead of filling with 0
    # This preserves the ocean/land distinction for edge analysis
    extreme_mask = land_only_dataset["extreme_mask"]

    # Get maximum consecutive days at each location across all time steps
    # (after ocean masking)
    # Take max over ALL time dimensions to get a 2D spatial mask
    time_dims_to_max = [dim for dim in extreme_mask.dims if "time" in dim.lower()]

    mask_total = extreme_mask
    for time_dim_name in time_dims_to_max:
        mask_total = mask_total.max(dim=time_dim_name)
    # Use nansum to handle NaN values properly when counting
    total_after = np.nansum(mask_total.values)
    print(f"  Total extreme grid points after ocean masking: {total_after}")

    # Find bounds
    extreme_locations = mask_total.where(mask_total > 0)

    if np.nansum(extreme_locations.values) == 0:
        print(f"  Warning: No {extreme_type} events detected")
        return {
            "bounds": None,
            "mask": mask_total,
            "extreme_mask": extreme_mask,
            "temperature_data": era5_data[
                "2m_temperature"
            ],  # Include temperature data for plotting
            "percentile_temperature_data": percentile_data,
            "metadata": {"extreme_type": extreme_type, "total_points": 0},
        }

    # Get coordinates where extremes are present
    extreme_coords = extreme_locations.stack(points=["latitude", "longitude"])
    extreme_coords_clean = extreme_coords.dropna("points")

    # Extract bounds
    lats = extreme_coords_clean.latitude.values
    lons = extreme_coords_clean.longitude.values

    # Check if we have any coordinates
    if len(lats) == 0 or len(lons) == 0:
        print("  Warning: No extreme coordinates found after processing")
        return {
            "bounds": None,
            "mask": mask_total,
            "extreme_mask": extreme_mask,
            "temperature_data": era5_data["2m_temperature"],
            "percentile_temperature_data": percentile_data,
            "metadata": {"extreme_type": extreme_type, "total_points": 0},
        }

    left_lon = float(np.min(lons))
    right_lon = float(np.max(lons))
    bottom_lat = float(np.min(lats))
    top_lat = float(np.max(lats))

    # Initial bounds from extreme coordinates
    temperature_bounds = {
        "latitude_min": bottom_lat,
        "latitude_max": top_lat,
        "longitude_min": left_lon,
        "longitude_max": right_lon,
    }

    # Create a buffered region around the detected extremes for plotting
    # This ensures the plot shows some context around the extreme event
    buffered_region = calculate_extent_bounds(
        left_lon=left_lon,
        right_lon=right_lon,
        bottom_lat=bottom_lat,
        top_lat=top_lat,
        extent_buffer=250,  # 250 km buffer
        extent_units="km",
    )

    return {
        "bounds": temperature_bounds,
        "mask": mask_total,
        "extreme_mask": extreme_mask,
        "temperature_data": era5_data["2m_temperature"],
        "percentile_temperature_data": percentile_data,
        "bounds_region": buffered_region,
        "metadata": {
            "extreme_type": extreme_type,
            "total_points": total_after,
            "min_consecutive_days": min_consecutive_days,
        },
    }


def process_temperature_event_optimized(
    event: Dict,
    extreme_type: Literal["heatwave", "freeze"],
    temp_climatology: xr.Dataset,
    expand_degrees: float = 2.0,
    max_days: int = 20,
) -> Optional[Dict]:
    """Process a single temperature extreme event with optimizations.

    Args:
        event: Event dictionary containing case information.
        extreme_type: Type of extreme to detect.
        temp_climatology: Temperature climatology dataset.
        expand_degrees: Degrees to expand region bounds.
        max_days: Maximum days to process.

    Returns:
        Results dictionary if successful, None otherwise.
    """
    print(f"\nProcessing {extreme_type}: {event['title']}")

    try:
        # Create case and get bounds
        case_collection = cases.load_individual_cases({"cases": [event]})
        case = case_collection.cases[0]
        bounds = case.location.get_bounding_coordinates

        # Create expanded region for data loading
        raw_lon_min = bounds.longitude_min - expand_degrees
        raw_lon_max = bounds.longitude_max + expand_degrees

        # Handle longitude expansion carefully to avoid wrapping issues
        # If expansion crosses prime meridian, keep in -180/180 format
        crosses_prime_meridian = (
            raw_lon_min < 0 and raw_lon_max > 0 and bounds.longitude_min < 90
        )
        near_prime_meridian = (
            abs(raw_lon_min - 360) % 360 < 20 and abs(raw_lon_max - 360) % 360 < 20
        )
        if raw_lon_min < -180 or crosses_prime_meridian or near_prime_meridian:
            # If crossing -180/180 or near it, use 0-360 for loading
            region_bounds = {
                "latitude_min": bounds.latitude_min - expand_degrees,
                "latitude_max": bounds.latitude_max + expand_degrees,
                "longitude_min": utils.convert_longitude_to_360(raw_lon_min),
                "longitude_max": utils.convert_longitude_to_360(raw_lon_max),
            }
            print(
                f"  Adjusted longitude bounds for loading: "
                f"{region_bounds['longitude_min']}°-{region_bounds['longitude_max']}°"
            )
        else:
            region_bounds = {
                "latitude_min": bounds.latitude_min - expand_degrees,
                "latitude_max": bounds.latitude_max + expand_degrees,
                "longitude_min": raw_lon_min,
                "longitude_max": raw_lon_max,
            }
            print(
                f"  Initial longitude bounds for loading: "
                f"{region_bounds['longitude_min']}°-{region_bounds['longitude_max']}°"
            )

        # Load ERA5 data
        era5_temp = load_era5_slice_efficiently(
            region_bounds,
            pd.Timestamp(event["start_date"]),
            pd.Timestamp(event["end_date"]),
        )
        era5_data = xr.Dataset({"2m_temperature": era5_temp})

        # Process the case
        result = process_temperature_extreme_case(
            era5_data,
            temp_climatology,
            extreme_type,
            event,
            min_consecutive_days=3,
        )

        if result["bounds"] is not None:
            # Create summary plot
            percentile_value = 0.85 if extreme_type == "heatwave" else 0.15

            # Get time dimension name (could be 'time' or 'valid_time')
            temp_data = result["temperature_data"]
            time_dim = "time" if "time" in temp_data.dims else "valid_time"
            dates = pd.to_datetime(temp_data[time_dim].values)

            percentile_data = get_climatology_for_dates(
                temp_climatology["2m_temperature"], dates
            ).sel(quantile=percentile_value)
            create_comprehensive_summary_plot(
                case_id=event["case_id_number"],
                title=event["title"],
                temperature_data=result["temperature_data"],
                percentile_data=percentile_data,
                extreme_mask=result["mask"],
                extreme_bounds=result["bounds"],
                buffered_bounds=result["bounds_region"],
                extreme_type=extreme_type,
                original_bounds=bounds,
            )

            # Save mask as NetCDF
            save_extreme_mask_netcdf(
                extreme_mask=result["mask"],
                case_id=event["case_id_number"],
                extreme_type=extreme_type,
            )

            return result
        else:
            print(f"  No {extreme_type} events detected for {event['title']}")
            return None

    except Exception as e:
        print(f"  Error processing {event['title']}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def process_all_temperature_events(
    extreme_type: Literal["heatwave", "freeze"],
    output_dir: str = "~/temperature_checks",
    expand_degrees: float = 2.0,
    max_days: int = 20,
    min_consecutive_days: int = 3,
    max_iterations: int = 10,
) -> List[Dict]:
    """Process all temperature extreme events of a given type.

    Args:
        extreme_type: Type of extreme event to process.
        output_dir: Directory to save outputs.
        expand_degrees: Degrees to expand region bounds.
        max_days: Maximum days to process.
        min_consecutive_days: Minimum consecutive days for valid event.
        max_iterations: Maximum expansion iterations.

    Returns:
        List of processed results for each event.
    """
    # Create output directory
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load climatology data
    temp_climatology = load_temperature_climatology()

    # Load events
    events_yaml = utils.load_events_yaml()
    if extreme_type == "heatwave":
        events = [
            event
            for event in events_yaml["cases"]
            if event["event_type"] == "heat_wave"
        ]
    else:
        events = [
            event for event in events_yaml["cases"] if event["event_type"] == "freeze"
        ]

    print(f"Found {len(events)} {extreme_type} events")

    # Process each event
    results = []
    for i, event in enumerate(tqdm(events, desc=f"Processing {extreme_type} events")):
        print(f"\nProcessing {extreme_type} {i + 1}/{len(events)}: {event['title']}")

        try:
            result = process_temperature_event_optimized(
                event,
                extreme_type,
                temp_climatology,
                expand_degrees,
                max_days,
            )

            if result is not None:
                # Add case information to result
                result["case_id"] = event["case_id_number"]
                result["title"] = event["title"]
                result["start_date"] = pd.to_datetime(event["start_date"])
                result["end_date"] = pd.to_datetime(event["end_date"])
                result["original_bounds"] = event.get("location", {})
                result["extreme_type"] = extreme_type
                result["config"] = {
                    "min_consecutive_days": min_consecutive_days,
                    "expand_edges": True,
                    "extent_buffer": expand_degrees,
                    "max_days_processed": max_days,
                }

                results.append(result)
                print(f"  ✅ Successfully processed {event['title']}")
            else:
                print(f"  ❌ No {extreme_type} events detected for {event['title']}")

        except Exception as e:
            print(f"  ❌ Error processing {event['title']}: {str(e)}")
            logger.error(f"Error processing {event['title']}: {str(e)}")
            continue

    # Save results to pickle file
    results_filename = f"{extreme_type}_bounds_results_optimized.pkl"
    results_path = output_path / results_filename

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} {extreme_type} results to {results_path}")

    return results


def create_comprehensive_summary_plot(
    case_id: int,
    title: str,
    temperature_data: xr.DataArray,
    percentile_data: xr.DataArray,
    extreme_mask: xr.DataArray,
    extreme_bounds: Dict,
    buffered_bounds: Dict,
    extreme_type: Literal["heatwave", "freeze"],
    original_bounds: Optional[Dict] = None,
    output_dir: str = "~/temperature_checks",
) -> None:
    """Create a comprehensive two-panel summary plot for temperature extreme analysis.

    Args:
        case_id: Case ID number.
        title: Case title.
        temperature_data: Temperature data array.
        percentile_data: Percentile threshold data.
        extreme_mask: Extreme event mask.
        extreme_bounds: Calculated extreme bounds.
        buffered_bounds: Buffered bounds for visualization.
        extreme_type: Type of extreme event.
        original_bounds: Original case bounds (optional).
        output_dir: Directory to save the plot.
    """
    # Create output directory
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Find peak temperature day
    time_dim = "time" if "time" in temperature_data.dims else "valid_time"
    peak_temp_idx = temperature_data.max(dim=["latitude", "longitude"]).argmax(
        dim=time_dim
    )
    peak_temp_time = temperature_data[time_dim].isel({time_dim: peak_temp_idx})
    peak_temp_data = temperature_data.isel({time_dim: peak_temp_idx})

    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))

    # Left panel: Peak temperature day
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())

    # Plot temperature - ensure data is numeric and loaded
    plot_temp = peak_temp_data.astype(float).load()

    im1 = plot_temp.plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap="Reds" if extreme_type == "heatwave" else "Blues",
        add_colorbar=False,
        vmin=float(plot_temp.min()),
        vmax=float(plot_temp.max()),
    )

    # Add geographic features
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.add_feature(cfeature.OCEAN, alpha=0.3, color="lightblue")
    ax1.add_feature(cfeature.LAND, alpha=0.1, color="lightgray")

    # Set extent
    ax1.set_extent(
        [
            buffered_bounds["longitude_min"],
            buffered_bounds["longitude_max"],
            buffered_bounds["latitude_min"],
            buffered_bounds["latitude_max"],
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add gridlines
    gl1 = ax1.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        alpha=0.5,
        linewidth=0.5,
        linestyle="--",
    )
    gl1.xlabels_top = False
    gl1.ylabels_right = False
    gl1.xformatter = LongitudeFormatter()
    gl1.yformatter = LatitudeFormatter()

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)
    cbar1.set_label("Temperature (K)", rotation=270, labelpad=20, fontsize=12)

    # Add title
    peak_date = pd.to_datetime(peak_temp_time.values).strftime("%Y-%m-%d %H:%M")
    ax1.set_title(
        f"Peak Temperature Day\n{peak_date}", fontsize=14, pad=20, fontweight="bold"
    )

    # Right panel: Consecutive extreme days
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())

    # Plot consecutive days - ensure coordinates are numeric and loaded
    plot_mask = extreme_mask.astype(float).load()
    plot_mask = plot_mask.where(plot_mask > 0, 0)  # Replace NaN with 0

    im2 = plot_mask.plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap="Reds" if extreme_type == "heatwave" else "Blues",
        add_colorbar=False,
        vmin=0,
        vmax=float(plot_mask.max()),
    )

    # Add geographic features
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.add_feature(cfeature.OCEAN, alpha=0.3, color="lightblue")
    ax2.add_feature(cfeature.LAND, alpha=0.1, color="lightgray")

    # Set extent
    ax2.set_extent(
        [
            buffered_bounds["longitude_min"],
            buffered_bounds["longitude_max"],
            buffered_bounds["latitude_min"],
            buffered_bounds["latitude_max"],
        ],
        crs=ccrs.PlateCarree(),
    )

    # Add gridlines
    gl2 = ax2.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        alpha=0.5,
        linewidth=0.5,
        linestyle="--",
    )
    gl2.xlabels_top = False
    gl2.ylabels_right = False
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05)
    cbar2.set_label("Consecutive Days", rotation=270, labelpad=20, fontsize=12)

    # Add title
    ax2.set_title(
        f"Consecutive {extreme_type.capitalize()} Days",
        fontsize=14,
        pad=20,
        fontweight="bold",
    )

    # Add overall title
    fig.suptitle(f"Case {case_id:03d}: {title}", fontsize=18, y=0.95, fontweight="bold")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)

    # Save plot
    plot_filename = f"{extreme_type}_case_{case_id:03d}_summary.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"    Saved summary plot: {plot_path}")

    plt.close()


def save_extreme_mask_netcdf(
    extreme_mask: xr.DataArray,
    case_id: int,
    extreme_type: Literal["heatwave", "freeze"],
    output_dir: str = "~/temperature_checks",
) -> None:
    """Save extreme mask as NetCDF file.

    Args:
        extreme_mask: Extreme event mask data array.
        case_id: Case ID number.
        extreme_type: Type of extreme event.
        output_dir: Directory to save the NetCDF file.
    """
    # Create output directory
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Fix data type issues for NetCDF compatibility
    clean_mask = extreme_mask.astype(float)

    # Add metadata
    clean_mask.attrs.update(
        {
            "case_id": case_id,
            "extreme_type": extreme_type,
            "description": f"Consecutive {extreme_type} days mask for case "
            f"{case_id:03d}",
            "units": "days",
            "long_name": f"consecutive_{extreme_type}_days",
        }
    )

    # Save as NetCDF
    mask_filename = f"{extreme_type}_mask_case_{case_id:03d}.nc"
    mask_path = output_path / mask_filename
    clean_mask.to_netcdf(mask_path)
    print(f"    Saved mask: {mask_path}")


def run_complete_temperature_analysis(
    output_dir: str = "~/temperature_checks",
    expand_degrees: float = 2.0,
    max_days: int = 20,
    min_consecutive_days: int = 3,
    max_iterations: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """Run complete temperature analysis for both heatwaves and freezes.

    Args:
        output_dir: Directory to save outputs.
        expand_degrees: Degrees to expand region bounds.
        max_days: Maximum days to process.
        min_consecutive_days: Minimum consecutive days for valid event.
        max_iterations: Maximum expansion iterations.

    Returns:
        Tuple of (heatwave_results, freeze_results).
    """
    print("=" * 80)
    print("COMPREHENSIVE TEMPERATURE EXTREME ANALYSIS")
    print("=" * 80)

    # Process heatwave events
    print("\n" + "=" * 50)
    print("PROCESSING HEATWAVE EVENTS")
    print("=" * 50)
    heatwave_results = process_all_temperature_events(
        "heatwave",
        output_dir,
        expand_degrees,
        max_days,
        min_consecutive_days,
        max_iterations,
    )

    # Process freeze events
    print("\n" + "=" * 50)
    print("PROCESSING FREEZE EVENTS")
    print("=" * 50)
    freeze_results = process_all_temperature_events(
        "freeze",
        output_dir,
        expand_degrees,
        max_days,
        min_consecutive_days,
        max_iterations,
    )

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"Heatwave events processed: {len(heatwave_results)}")
    print(f"Freeze events processed: {len(freeze_results)}")
    print(f"Output directory: {Path(output_dir).expanduser()}")

    return heatwave_results, freeze_results


# Main execution function
def main():
    """Main function to run complete temperature analysis."""
    # Run complete analysis
    heatwave_results, freeze_results = run_complete_temperature_analysis()

    print("\n🎉 Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    main()
