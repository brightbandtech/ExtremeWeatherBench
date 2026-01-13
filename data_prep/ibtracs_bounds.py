"""Script to calculate and update IBTrACS bounds for tropical cyclone cases."""

import datetime
import logging
import re
from importlib import resources

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib.patches import Rectangle

import extremeweatherbench.data
import extremeweatherbench.inputs as inputs
import extremeweatherbench.regions as regions
import extremeweatherbench.utils as utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_end_point(
    start_lat: float, start_lon: float, bearing: float, distance_km: float
) -> tuple[float, float]:
    """Calculate the end point given a starting point, bearing, and distance.

    Args:
        start_lat: Starting latitude in degrees.
        start_lon: Starting longitude in degrees.
        bearing: Bearing in degrees (0-360, where 0 is north, 90 is east).
        distance_km: Distance in kilometers.

    Returns:
        End point as (latitude, longitude) in degrees.
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1 = np.radians(start_lat)
    lon1 = np.radians(start_lon)
    bearing_rad = np.radians(bearing)

    # Calculate end point
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(distance_km / R)
        + np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing_rad)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_km / R) * np.cos(lat1),
        np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2),
    )

    # Convert back to degrees
    end_lat = np.degrees(lat2)
    end_lon = np.degrees(lon2)

    return end_lat, end_lon


def calculate_extent_bounds(
    left_lon: float,
    right_lon: float,
    bottom_lat: float,
    top_lat: float,
    extent_buffer: float = 250,
) -> regions.Region:
    """Calculate extent bounds with buffer.

    Args:
        left_lon: Left longitude boundary.
        right_lon: Right longitude boundary.
        bottom_lat: Bottom latitude boundary.
        top_lat: Top latitude boundary.
        extent_buffer: Buffer distance to add around bounds. Defaults to 250.
        extent_units: Units for the buffer ("degrees" or "km"). Defaults to "km".

    Returns:
        BoundingBoxRegion with buffered bounds.
    """
    new_bottom_lat, _ = np.round(
        calculate_end_point(bottom_lat, left_lon, 180, extent_buffer), 1
    )
    new_top_lat, _ = np.round(
        calculate_end_point(top_lat, right_lon, 0, extent_buffer), 1
    )
    _, new_left_lon = np.round(
        calculate_end_point(bottom_lat, left_lon, 270, extent_buffer), 1
    )
    _, new_right_lon = np.round(
        calculate_end_point(bottom_lat, right_lon, 90, extent_buffer), 1
    )

    new_left_lon = np.round(utils.convert_longitude_to_360(new_left_lon), 1)
    new_right_lon = np.round(utils.convert_longitude_to_360(new_right_lon), 1)
    new_box = regions.BoundingBoxRegion(
        new_bottom_lat, new_top_lat, new_left_lon, new_right_lon
    )
    return new_box


def plot_storm_bounds(storm_bounds):
    """Plot all of the tropical cyclone bounds.

    Args:
        storm_bounds: Dictionary or Series containing storm bounding box data.
    """
    _ = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)

    # Remove longitude labels from the top and latitude labels from the right
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Plot each storm's bounding box
    colors = plt.cm.tab20(np.linspace(0, 1, len(storm_bounds)))
    for i, (storm_name, bounds) in enumerate(storm_bounds.items()):
        # Extract bounds
        lon_min = bounds.longitude_min
        lon_max = bounds.longitude_max
        lat_min = bounds.latitude_min
        lat_max = bounds.latitude_max
        # Create rectangle patch
        width = lon_max - lon_min
        height = lat_max - lat_min
        if lon_max < lon_min:
            lon_min = lon_min - 360
            width = lon_max - lon_min
        rect = Rectangle(
            (lon_min, lat_min),
            width,
            height,
            linewidth=1,
            edgecolor=colors[i],
            facecolor="none",
            transform=ccrs.PlateCarree(),
            alpha=1,
        )
        ax.add_patch(rect)

    # Set global extent
    ax.set_global()

    plt.title("Storm Bounding Boxes from IBTrACS Data +250km Buffer", loc="left")
    plt.tight_layout()
    plt.show()


def load_and_process_ibtracs_data():
    """Load and process IBTrACS data for tropical cyclones from 2020-2025.

    Returns:
        Processed pandas DataFrame containing storm data with unified columns.
    """
    logger.info("Loading IBTrACS data...")

    IBTRACS = inputs.IBTrACS(
        source=inputs.IBTRACS_URI,
        variables=["vmax", "slp"],
        variable_mapping=inputs.IBTrACS_metadata_variable_mapping,
        storage_options={},
    )

    IBTRACS_lf = IBTRACS.open_and_maybe_preprocess_data_from_source()
    IBTRACS_lf = IBTRACS.maybe_map_variable_names(IBTRACS_lf)

    # Get all storms from 2020 - 2025 seasons
    all_storms_2020_2025_lf = IBTRACS_lf.filter(
        (pl.col("SEASON").cast(pl.Int32) >= 2020)
    ).select(inputs.IBTrACS_metadata_variable_mapping.values())

    schema = all_storms_2020_2025_lf.collect_schema()
    # Convert pressure and surface wind columns to float, replacing " " with null
    pressure_cols = [col for col in schema if "pressure" in col.lower()]
    wind_cols = [col for col in schema if "wind" in col.lower()]

    # Apply transformations to convert " " to null and cast to float
    all_storms_lf = all_storms_2020_2025_lf.with_columns(
        [
            pl.when(pl.col(col) == " ")
            .then(None)
            .otherwise(pl.col(col))
            .cast(pl.Float64, strict=False)
            .alias(col)
            for col in pressure_cols + wind_cols
        ]
    )

    # Drop rows where ALL columns are null
    all_storms_lf = all_storms_lf.filter(~pl.all_horizontal(pl.all().is_null()))

    # Create unified pressure and wind columns by preferring USA and WMO data
    wind_columns = [col for col in schema if "surface_wind_speed" in col]
    wind_priority = ["usa_surface_wind_speed", "wmo_surface_wind_speed"] + [
        col
        for col in wind_columns
        if col not in ["usa_surface_wind_speed", "wmo_surface_wind_speed"]
    ]

    pressure_columns = [
        col for col in schema if "air_pressure_at_mean_sea_level" in col
    ]
    pressure_priority = [
        "usa_air_pressure_at_mean_sea_level",
        "wmo_air_pressure_at_mean_sea_level",
    ] + [
        col
        for col in pressure_columns
        if col
        not in [
            "usa_air_pressure_at_mean_sea_level",
            "wmo_air_pressure_at_mean_sea_level",
        ]
    ]

    # Create unified columns using coalesce
    all_storms_lf = all_storms_lf.with_columns(
        [
            pl.coalesce(wind_priority).alias("surface_wind_speed"),
            pl.coalesce(pressure_priority).alias("air_pressure_at_mean_sea_level"),
        ]
    )

    # Select only the columns to keep
    columns_to_keep = [
        "storm_id",
        "valid_time",
        "tc_name",
        "latitude",
        "longitude",
        "surface_wind_speed",
        "air_pressure_at_mean_sea_level",
    ]

    all_storms_lf = all_storms_lf.select(columns_to_keep)

    # Drop rows where wind speed OR pressure are null
    all_storms_lf = all_storms_lf.filter(
        pl.col("surface_wind_speed").is_not_null()
        & pl.col("air_pressure_at_mean_sea_level").is_not_null()
    )

    logger.info("Missing values per column:")
    null_counts = all_storms_lf.select(
        [pl.col(col).null_count().alias(f"{col}_nulls") for col in columns_to_keep]
    ).collect()
    logger.info(null_counts)
    logger.info(
        "Total rows after filtering: %s",
        all_storms_lf.select(pl.len()).collect().item(),
    )

    # Convert to pandas for groupby operations
    all_storms_df = all_storms_lf.collect().to_pandas()
    all_storms_df["valid_time"] = pd.to_datetime(all_storms_df["valid_time"])

    # Final cleanup
    all_storms_df = all_storms_df.dropna(
        subset=["surface_wind_speed", "air_pressure_at_mean_sea_level"], how="any"
    )

    logger.info("After final cleanup: %s rows remaining", len(all_storms_df))
    logger.info("Missing values per column:")
    logger.info(all_storms_df.isnull().sum())

    return all_storms_df


def calculate_storm_bounds(all_storms_df):
    """Calculate bounding boxes for each storm.

    Args:
        all_storms_df: DataFrame containing storm data with columns for storm_id,
            tc_name, latitude, and longitude.

    Returns:
        Series with MultiIndex (storm_id, tc_name) containing BoundingBoxRegion
        objects for each storm.
    """
    logger.info("Calculating storm bounds...")

    # Group by tc_name and calculate extent bounds for each storm
    storm_bounds = all_storms_df.groupby(["storm_id", "tc_name"]).apply(
        lambda group: calculate_extent_bounds(
            left_lon=group["longitude"].min(),
            right_lon=group["longitude"].max(),
            bottom_lat=group["latitude"].min(),
            top_lat=group["latitude"].max(),
        )
    )

    logger.info("Calculated bounds for %s storms", len(storm_bounds))
    return storm_bounds


def calculate_geographic_overlap(case_bounds, storm_bounds):
    """Calculate geographic overlap between case bounds and storm bounds.

    Args:
        case_bounds: Dictionary with latitude_min, latitude_max, longitude_min,
            longitude_max.
        storm_bounds: Dictionary with latitude_min, latitude_max, longitude_min,
            longitude_max.

    Returns:
        Float representing overlap area (0 = no overlap, higher = more overlap).
    """
    # Calculate overlapping rectangle
    lat_overlap = max(
        0,
        min(case_bounds["latitude_max"], storm_bounds["latitude_max"])
        - max(case_bounds["latitude_min"], storm_bounds["latitude_min"]),
    )

    # Handle longitude wraparound
    lon_min_case = case_bounds["longitude_min"]
    lon_max_case = case_bounds["longitude_max"]
    lon_min_storm = storm_bounds["longitude_min"]
    lon_max_storm = storm_bounds["longitude_max"]

    # Simple longitude overlap calculation (not handling 180° wraparound)
    lon_overlap = max(
        0, min(lon_max_case, lon_max_storm) - max(lon_min_case, lon_min_storm)
    )

    return lat_overlap * lon_overlap


def find_best_storm_by_geography(storm_names, all_storms_df, case_location):
    """Find the best matching storm based on geographic proximity.

    Args:
        storm_names: List of storm names to search for.
        all_storms_df: DataFrame containing all storm data.
        case_location: Dictionary with case location bounds.

    Returns:
        Filtered DataFrame with the best matching storm data.
    """
    # Get all storms matching the names
    matching_storms = all_storms_df[all_storms_df["tc_name"].isin(storm_names)]

    if len(matching_storms) == 0:
        return matching_storms

    # If only one storm_id, return it
    unique_storm_ids = matching_storms["storm_id"].unique()
    if len(unique_storm_ids) == 1:
        return matching_storms

    logger.info(
        "Multiple storms found for %s, selecting by geographic proximity", storm_names
    )

    # Calculate bounds for each storm_id
    best_storm_id = None
    best_overlap = -1

    case_bounds = {
        "latitude_min": case_location["latitude_min"],
        "latitude_max": case_location["latitude_max"],
        "longitude_min": case_location["longitude_min"],
        "longitude_max": case_location["longitude_max"],
    }

    for storm_id in unique_storm_ids:
        storm_data = matching_storms[matching_storms["storm_id"] == storm_id]
        storm_bounds = {
            "latitude_min": storm_data["latitude"].min(),
            "latitude_max": storm_data["latitude"].max(),
            "longitude_min": storm_data["longitude"].min(),
            "longitude_max": storm_data["longitude"].max(),
        }

        overlap = calculate_geographic_overlap(case_bounds, storm_bounds)

        # Also check hemisphere match (more important than overlap)
        case_hemisphere = "N" if case_bounds["latitude_min"] >= 0 else "S"
        storm_hemisphere = "N" if storm_bounds["latitude_min"] >= 0 else "S"
        hemisphere_match = case_hemisphere == storm_hemisphere

        # Prioritize hemisphere match, then overlap
        score = (1000 if hemisphere_match else 0) + overlap

        logger.info(
            "Storm %s: hemisphere=%s, overlap=%.2f, score=%.2f",
            storm_id,
            storm_hemisphere,
            overlap,
            score,
        )

        if score > best_overlap:
            best_overlap = score
            best_storm_id = storm_id

    logger.info("Selected storm %s with score %.2f", best_storm_id, best_overlap)
    return matching_storms[matching_storms["storm_id"] == best_storm_id]


def find_storm_bounds_for_case(storm_name, storm_bounds, all_storms_df):
    """Find storm bounds for a given case, handling various name formats."""
    found_bounds = None
    storm_data = None

    # Check if storm name has parentheses and extract both versions
    if storm_name in storm_bounds.index.levels[1]:
        found_bounds = storm_bounds[
            storm_bounds.index.get_level_values("tc_name") == storm_name
        ]
        storm_names = [storm_name]
    elif "(" in storm_name and ")" in storm_name:
        # Extract name before parentheses
        name_before = storm_name.split("(")[0].strip()
        # Extract name inside parentheses
        name_in_parens = storm_name.split("(")[1].split(")")[0].strip()

        if name_before in storm_bounds.index.levels[1]:
            found_bounds = storm_bounds[
                storm_bounds.index.get_level_values("tc_name") == name_before
            ]
        elif name_in_parens in storm_bounds.index.levels[1]:
            found_bounds = storm_bounds[
                storm_bounds.index.get_level_values("tc_name") == name_in_parens
            ]
        storm_names = [name_before, name_in_parens]
    # Check if storm name contains 'AND' and try to find combined name with ':'
    elif " AND " in storm_name:
        # Split the names by 'AND' and search for each individually first
        bounds1 = None
        bounds2 = None
        names_parts = storm_name.split(" AND ")
        if len(names_parts) == 2:
            name1 = names_parts[0].strip()
            name2 = names_parts[1].strip()
            if name1 == "GULAB":
                name1 = "GULAB:SHAHEEN-GU"
                if name1 in storm_bounds.index.levels[1]:
                    bounds1 = storm_bounds[
                        storm_bounds.index.get_level_values("tc_name") == name1
                    ]
                name2 = None
            else:
                # Try to find each name individually
                if name1 in storm_bounds.index.levels[1]:
                    bounds1 = storm_bounds[
                        storm_bounds.index.get_level_values("tc_name") == name1
                    ]
                if name2 in storm_bounds.index.levels[1]:
                    bounds2 = storm_bounds[
                        storm_bounds.index.get_level_values("tc_name") == name2
                    ]

            # If we found both, merge them by taking the bounding box that
            # encompasses both
            if bounds1 is not None and bounds2 is not None:
                merged_bbox = regions.BoundingBoxRegion(
                    latitude_min=min(
                        bounds1.iloc[0].latitude_min, bounds2.iloc[0].latitude_min
                    ),
                    latitude_max=max(
                        bounds1.iloc[0].latitude_max, bounds2.iloc[0].latitude_max
                    ),
                    longitude_min=min(
                        bounds1.iloc[0].longitude_min, bounds2.iloc[0].longitude_min
                    ),
                    longitude_max=max(
                        bounds1.iloc[0].longitude_max, bounds2.iloc[0].longitude_max
                    ),
                )
                merged_bounds = pd.Series(merged_bbox)
                found_bounds = merged_bounds
                storm_names = [name1, name2]
            # If only one found, use that one
            elif bounds1 is not None:
                found_bounds = bounds1
                storm_names = [name1]
            elif bounds2 is not None:
                found_bounds = bounds2
                storm_names = [name2]
            else:
                # Fall back to trying combined name formats
                combined_name = f"{name1}:{name2}"
                if combined_name in storm_bounds.index.levels[1]:
                    found_bounds = storm_bounds[
                        storm_bounds.index.get_level_values("tc_name") == combined_name
                    ]
                # Also try with hyphen format
                combined_name_hyphen = f"{name1}-{name2}"
                if (
                    found_bounds is None
                    and combined_name_hyphen in storm_bounds.index.levels[1]
                ):
                    found_bounds = storm_bounds[
                        storm_bounds.index.get_level_values("tc_name")
                        == combined_name_hyphen
                    ]
                storm_names = [name1, name2]
    else:
        storm_names = [storm_name]

    if found_bounds is not None:
        # Get storm data for this storm to find first and last valid times
        storm_data = all_storms_df[all_storms_df["tc_name"].isin(storm_names)]
        if len(storm_data) == 0:
            # Try to find with different name formats
            for key in storm_bounds.keys():
                if any(name in key for name in storm_names):
                    storm_data = all_storms_df[all_storms_df["tc_name"] == key]
                    if len(storm_data) > 0:
                        break

    return found_bounds, storm_data, storm_names


def update_cases_with_storm_bounds(storm_bounds, all_storms_df):
    """Update cases in events.yaml with storm bounds from IBTrACS data.

    Args:
        storm_bounds: Series with MultiIndex containing storm bounds data.
        all_storms_df: DataFrame containing all storm data.

    Returns:
        Tuple containing:
            - cases_all: Complete events data dictionary.
            - cases_new: Updated list of cases.
    """
    logger.info("Updating cases with storm bounds...")

    cases_all = utils.load_events_yaml()
    cases_old = cases_all["cases"]
    cases_new = cases_old.copy()

    # Update the yaml cases with storm bounds from IBTrACS data
    for single_case in cases_new:
        if single_case["event_type"] == "tropical_cyclone":
            storm_name = single_case["title"].upper()

            found_bounds, storm_data, storm_names = find_storm_bounds_for_case(
                storm_name, storm_bounds, all_storms_df
            )
            # If multiple storms found, use geographic proximity to select best one
            if (
                storm_data is not None
                and len(storm_data) > 0
                and storm_data["storm_id"].nunique() > 1
            ):
                logger.info(
                    "Multiple storms found for %s, using geographic selection",
                    storm_name,
                )
                storm_data = find_best_storm_by_geography(
                    storm_names, all_storms_df, single_case["location"]["parameters"]
                )

            if found_bounds is not None:
                # Update the case with IBTrACS bounding box coordinates
                single_case["location"]["parameters"]["latitude_min"] = float(
                    found_bounds.iloc[0].latitude_min
                )
                single_case["location"]["parameters"]["latitude_max"] = float(
                    found_bounds.iloc[0].latitude_max
                )
                single_case["location"]["parameters"]["longitude_min"] = float(
                    found_bounds.iloc[0].longitude_min
                )
                single_case["location"]["parameters"]["longitude_max"] = float(
                    found_bounds.iloc[0].longitude_max
                )

                # Update start and end dates based on storm valid times +/- 48 hours
                if len(storm_data) > 0:
                    first_time = storm_data["valid_time"].min()
                    last_time = storm_data["valid_time"].max()

                    # Add/subtract 48 hours (2 days)
                    start_date = pd.to_datetime(first_time) - pd.Timedelta(hours=48)
                    end_date = pd.to_datetime(last_time) + pd.Timedelta(hours=48)

                    single_case["start_date"] = start_date
                    single_case["end_date"] = end_date

                    logger.info("Updated %s with bounds", storm_names)
                    logger.info("  Start date: %s, End date: %s", start_date, end_date)
                else:
                    logger.info("Updated %s with bounds: %s", storm_names, found_bounds)
                    logger.warning("Could not find storm data to update dates")
            else:
                logger.info(
                    "NOT updated: Storm %s not found in IBTrACS data", storm_name
                )

    return cases_all, cases_new


def convert_datetimes_to_strings(obj):
    """Recursively convert datetime objects to strings to avoid pickle issues.

    Args:
        obj: Object to convert (can be dict, list, datetime, or other types).

    Returns:
        Object with all datetime instances converted to strings in
        YYYY-MM-DD HH:MM:SS format.
    """
    if isinstance(obj, (datetime.datetime, pd.Timestamp)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, dict):
        return {key: convert_datetimes_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes_to_strings(item) for item in obj]
    else:
        return obj


def write_updated_yaml(cases_all, cases_new):
    """Write the updated events.yaml file.

    Args:
        cases_all: Complete events data dictionary.
        cases_new: Updated list of cases.
    """
    logger.info("Writing updated YAML file...")

    # Find changes between old and new cases
    cases_old = cases_all["cases"]
    updated_cases = []
    for i, case_new in enumerate(cases_new):
        case_old = cases_old[i] if i < len(cases_old) else None

        if case_old is None or case_new != case_old:
            updated_cases.append((i, case_new))
            logger.info(
                "Case %s (%s) has been updated",
                case_new["case_id_number"],
                case_new["title"],
            )

    # Replace only the updated cases in the original data
    for case_index, updated_case in updated_cases:
        if case_index < len(cases_old):
            cases_old[case_index] = updated_case

    # Write the updated events.yaml file
    cases_all["cases"] = cases_old

    # Convert all datetime objects to strings before YAML serialization
    events_data_clean = convert_datetimes_to_strings(cases_all)

    # Note: Writing to the events.yaml file in the package data directory
    # This updates the installed package's events.yaml file
    events_yaml_file = resources.files(extremeweatherbench.data).joinpath("events.yaml")
    with resources.as_file(events_yaml_file) as file_path:
        with open(file_path, "w") as f:
            # First, dump to YAML string
            yaml_content = yaml.dump(
                events_data_clean, default_flow_style=False, sort_keys=False, indent=2
            )

            # Pattern to match quoted datetime strings (YYYY-MM-DD HH:MM:SS)
            datetime_pattern = r"'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'"
            # Replace with unquoted version
            yaml_content = re.sub(datetime_pattern, r"\1", yaml_content)

            # Write the processed content
            f.write(yaml_content)

    logger.info("\nUpdated %s cases in events.yaml", len(updated_cases))


def main():
    """Main function to process IBTrACS data and update events.yaml.

    This function orchestrates the entire process:
    1. Loads and processes IBTrACS data
    2. Calculates storm bounds
    3. Updates cases with storm bounds
    4. Writes updated YAML file
    """
    logger.info("Starting IBTrACS bounds calculation and update process...")

    # Load and process IBTrACS data
    all_storms_df = load_and_process_ibtracs_data()

    # Calculate storm bounds
    storm_bounds = calculate_storm_bounds(all_storms_df)

    # Optionally plot storm bounds (uncomment to enable)
    # plot_storm_bounds(storm_bounds)

    # Update cases with storm bounds
    cases_all, cases_new = update_cases_with_storm_bounds(storm_bounds, all_storms_df)

    # Write updated YAML file
    write_updated_yaml(cases_all, cases_new)

    logger.info("Process completed successfully!")


if __name__ == "__main__":
    main()
