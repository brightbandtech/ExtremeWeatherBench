"""Create bounding boxes for PPH severe convection events.

This script:
1. Loads PPH sparse data with valid times
2. Loads severe convection events from events.yaml
3. For each PPH time that matches an event, creates a bounding box
   around the non-zero PPH values with a +250km buffer
4. Saves the bounding boxes to a file
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import label

from extremeweatherbench import calc, cases

# Radius of Earth in km (mean radius)
EARTH_RADIUS_KM = 6371.0


def has_case_for_dates(
    case_list: list[cases.IndividualCase],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    is_australia: bool,
) -> bool:
    """Check if there's a case covering the date range for the region.

    Args:
        case_list: list of all cases
        start_date: Start of date range to check
        end_date: End of date range to check
        is_australia: If True, look for Australia case; else US case

    Returns:
        True if matching case exists
    """
    for event in case_list:
        # Check date overlap (event end is exclusive)
        event_start = event.start_date
        event_end = event.end_date

        overlaps = not (end_date <= event_start or start_date >= event_end)
        if not overlaps:
            continue

        # Check if it's the right region
        is_aus_title = "australia" in event.title.lower()

        if is_australia and is_aus_title:
            return True
        elif not is_australia and not is_aus_title:
            return True

    return False


def get_connected_blob(
    mask: np.ndarray, start_indices: List[Tuple[int, int]]
) -> np.ndarray:
    """Get all connected points in a blob using flood fill.

    Args:
        mask: 2D boolean array
        start_indices: List of (row, col) tuples to start from

    Returns:
        2D boolean array of connected region
    """
    # Label connected components
    labeled, num_features = label(mask)

    # Find which labels correspond to our start indices
    blob_mask = np.zeros_like(mask, dtype=bool)

    for row, col in start_indices:
        if mask[row, col]:
            label_id = labeled[row, col]
            if label_id > 0:
                blob_mask |= labeled == label_id

    return blob_mask


def get_pph_bounding_box(
    pph_data: xr.DataArray,
    valid_time: pd.Timestamp,
    buffer_km: float = 250,
    threshold: float = 0.01,
    max_distance_km: float = 2000,
    max_blob_size_km: Optional[float] = None,
    case_title: str = "",
    all_cases: Optional[list[cases.IndividualCase]] = None,
) -> Optional[Dict[str, Any]]:
    """Calculate bounding box around PPH peak with distance constraint.

    Approach:
    1. Find peak PPH location
    2. Include all PPH within max_distance_km of peak
    3. For blobs partially cut by distance threshold, include entire blob
       if it's smaller than max_blob_size_km
    4. Add buffer_km buffer

    Args:
        pph_data: xarray DataArray with PPH data
        valid_time: Time to select from pph_data
        buffer_km: Buffer distance in kilometers (default 250)
        threshold: Minimum PPH value to consider (default 0.01)
        max_distance_km: Maximum distance from peak (default 2000)
        max_blob_size_km: Max blob diagonal for extension (default
                         0.75*max_distance_km)
        case_title: Case title to determine hemisphere
        all_cases: list of all cases (for
                   hemisphere checking)

    Returns:
        Dictionary with bounding box info or None if no data
    """
    # Set default blob size if not provided
    if max_blob_size_km is None:
        max_blob_size_km = max_distance_km * 0.75
    # Select time slice
    pph_slice = pph_data.sel(valid_time=valid_time, method="nearest")

    # Convert sparse to dense if needed
    if hasattr(pph_slice.data, "todense"):
        pph_dense = pph_slice.data.todense()
    else:
        pph_dense = pph_slice.values

    # Find non-zero locations
    mask = pph_dense >= threshold

    if not np.any(mask):
        return None

    # Get latitude and longitude arrays
    lats = pph_slice.latitude.values
    lons = pph_slice.longitude.values

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Find peak PPH location
    peak_value = np.max(pph_dense[mask])
    peak_idx = np.where((pph_dense == peak_value) & mask)
    peak_lat = lat_grid[peak_idx][0]
    peak_lon = lon_grid[peak_idx][0]

    # Calculate distance from peak for all grid points
    distances = calc.haversine_distance(
        [peak_lat, peak_lon], [lat_grid, lon_grid], units="km"
    )

    # Start with points within max_distance_km
    within_distance = distances <= max_distance_km
    include_mask = mask & within_distance

    # Extend to include complete blobs that are partially included
    # Label all connected components in the original mask
    labeled_all, num_all = label(mask)

    # Check each blob to see if we should include it entirely
    final_mask = include_mask.copy()

    for blob_id in range(1, num_all + 1):
        blob_mask = labeled_all == blob_id

        # Check how much of this blob is already included
        blob_included_count = np.sum(blob_mask & include_mask)
        blob_total_count = np.sum(blob_mask)

        # If this blob is partially included, consider extending
        if blob_included_count > 0 and blob_included_count < blob_total_count:
            # Get blob's geographic size
            blob_lats = lat_grid[blob_mask]
            blob_lons = lon_grid[blob_mask]

            blob_diagonal = calc.haversine_distance(
                [np.min(blob_lats), np.min(blob_lons)],
                [np.max(blob_lats), np.max(blob_lons)],
                units="km",
            )

            # Only extend blob if smaller than max_blob_size_km
            # This prevents connecting very distant regions while allowing
            # natural blob completion at edges
            if blob_diagonal < max_blob_size_km:
                final_mask |= blob_mask

    include_mask = final_mask

    # Apply hemisphere filtering if needed
    is_aus_case = "australia" in case_title.lower()

    if is_aus_case and all_cases:
        # Check for northern hemisphere data
        has_northern_included = np.any(lat_grid[include_mask] > 0)
        has_southern_included = np.any(lat_grid[include_mask] < 0)

        if has_northern_included:
            date_start = valid_time - pd.Timedelta(days=1)
            date_end = valid_time + pd.Timedelta(days=1)
            has_us_case = has_case_for_dates(
                all_cases, date_start, date_end, is_australia=False
            )

            if has_us_case:
                # Filter to southern hemisphere only
                include_mask = include_mask & (lat_grid < 0)
            elif not has_southern_included:
                # No southern data, skip
                return None

    elif not is_aus_case and all_cases:
        # US/other case
        has_southern_included = np.any(lat_grid[include_mask] < 0)
        has_northern_included = np.any(lat_grid[include_mask] > 0)

        if has_southern_included:
            date_start = valid_time - pd.Timedelta(days=1)
            date_end = valid_time + pd.Timedelta(days=1)
            has_aus_case = has_case_for_dates(
                all_cases, date_start, date_end, is_australia=True
            )

            if has_aus_case:
                # Filter to northern hemisphere only
                include_mask = include_mask & (lat_grid > 0)
            elif not has_northern_included:
                # No northern data, skip
                return None

    # Check if we still have points
    if not np.any(include_mask):
        return None

    # Get final coordinates
    active_lats = lat_grid[include_mask]
    active_lons = lon_grid[include_mask]

    # Create bounding box
    box = _create_single_bbox(active_lats, active_lons, buffer_km, valid_time)

    # Add peak information
    if box is not None:
        box["peak_lat"] = float(peak_lat)
        box["peak_lon"] = float(peak_lon)
        box["peak_value"] = float(peak_value)
        box["max_distance_km"] = max_distance_km

    return box


def _create_single_bbox(
    active_lats: np.ndarray,
    active_lons: np.ndarray,
    buffer_km: float,
    valid_time: pd.Timestamp,
) -> Dict[str, Any]:
    """Create a single bounding box from a set of coordinates.

    Args:
        active_lats: Array of latitudes
        active_lons: Array of longitudes
        buffer_km: Buffer distance in km
        valid_time: Timestamp

    Returns:
        Dictionary with bounding box info
    """
    # Find min/max
    min_lat = np.min(active_lats)
    max_lat = np.max(active_lats)
    min_lon = np.min(active_lons)
    max_lon = np.max(active_lons)

    # Calculate center point for longitude conversion
    center_lat = (min_lat + max_lat) / 2

    # Add buffer (convert km to degrees)
    lat_buffer = buffer_km / 111.0  # 1 degree latitude ~ 111 km
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(center_lat)))

    bbox_min_lat = min_lat - lat_buffer
    bbox_max_lat = max_lat + lat_buffer
    bbox_min_lon = min_lon - lon_buffer
    bbox_max_lon = max_lon + lon_buffer

    # Clamp to valid ranges
    bbox_min_lat = np.clip(bbox_min_lat, -90, 90)
    bbox_max_lat = np.clip(bbox_max_lat, -90, 90)

    # Handle longitude wrapping (0-360 convention)
    # Only wrap if the buffered extent goes outside [0, 360)
    # Check if region crosses antimeridian
    lon_span = max_lon - min_lon
    crosses_antimeridian = lon_span > 180

    if not crosses_antimeridian:
        # Simple case: doesn't cross antimeridian
        # Keep in [0, 360) range
        if bbox_min_lon < 0:
            bbox_min_lon = bbox_min_lon + 360
        elif bbox_min_lon >= 360:
            bbox_min_lon = bbox_min_lon % 360

        if bbox_max_lon < 0:
            bbox_max_lon = bbox_max_lon + 360
        elif bbox_max_lon >= 360:
            bbox_max_lon = bbox_max_lon % 360
    else:
        # Region crosses antimeridian - more complex handling
        bbox_min_lon = bbox_min_lon % 360
        bbox_max_lon = bbox_max_lon % 360

    return {
        "valid_time": pd.Timestamp(valid_time),
        "latitude_min": float(bbox_min_lat),
        "latitude_max": float(bbox_max_lat),
        "longitude_min": float(bbox_min_lon),
        "longitude_max": float(bbox_max_lon),
        "center_lat": float(center_lat),
        "center_lon": float((min_lon + max_lon) / 2),
        "original_extent": {
            "lat_min": float(min_lat),
            "lat_max": float(max_lat),
            "lon_min": float(min_lon),
            "lon_max": float(max_lon),
        },
        "buffer_km": buffer_km,
    }


def match_pph_times_to_events(
    pph_data: xr.DataArray, case_list: list[cases.IndividualCase]
) -> Dict[int, Dict[str, Any]]:
    """Match PPH valid times to severe convection events.

    Args:
        pph_data: xarray DataArray with PPH data
        case_collection: list of IndividualCase objects of severe convection events

    Returns:
        Dictionary mapping event case_id to list of valid times
    """
    pph_times = pd.to_datetime(pph_data.valid_time.values)

    event_time_map = {}

    for event in case_list:
        case_id = event.case_id_number
        start = event.start_date
        end = event.end_date

        # Find times within event window (end time is exclusive)
        matching_times = [t for t in pph_times if start <= t < end]

        if matching_times:
            event_time_map[case_id] = {
                "event": event,
                "times": matching_times,
            }

    return event_time_map


def create_bounding_boxes(
    pph_data: xr.DataArray,
    event_time_map: Dict[int, Dict[str, Any]],
    all_cases: list[cases.IndividualCase],
    buffer_km: float = 250,
    threshold: float = 0.01,
    max_distance_km: float = 2000,
    max_blob_size_km: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Create bounding boxes for matched events.

    For each event time:
    - Finds peak PPH location
    - Includes all PPH within max_distance_km
    - Includes connected blobs at boundary
    - Checks for opposite hemisphere case existence

    Args:
        pph_data: xarray DataArray with PPH data
        event_time_map: Dictionary from match_pph_times_to_events
        all_cases: list of all cases (for hemisphere checking)
        buffer_km: Buffer distance in kilometers
        threshold: Minimum PPH value to consider
        max_distance_km: Maximum distance from peak (default 2000)
        max_blob_size_km: Max blob diagonal for extension

    Returns:
        List of bounding box dictionaries
    """
    bounding_boxes = []

    for case_id, data in event_time_map.items():
        event = data["event"]
        times = data["times"]

        for time in times:
            bbox = get_pph_bounding_box(
                pph_data,
                time,
                buffer_km=buffer_km,
                threshold=threshold,
                max_distance_km=max_distance_km,
                max_blob_size_km=max_blob_size_km,
                case_title=event.title,
                all_cases=all_cases,
            )

            if bbox is not None:
                bbox["case_id_number"] = case_id
                bbox["case_title"] = event.title
                bbox["event_start_date"] = event.start_date
                bbox["event_end_date"] = event.end_date
                bounding_boxes.append(bbox)

    return bounding_boxes


def save_bounding_boxes(
    bounding_boxes: List[Dict[str, Any]], output_path: Union[str, Path]
) -> pd.DataFrame:
    """Save bounding boxes to YAML and CSV files.

    Args:
        bounding_boxes: List of bounding box dictionaries
        output_path: Path to save output (without extension)

    Returns:
        DataFrame with bounding box data
    """
    output_path_obj = Path(output_path)

    # Save as YAML
    yaml_path = output_path_obj.with_suffix(".yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"bounding_boxes": bounding_boxes}, f, default_flow_style=False)  # noqa: E501

    # Save as CSV (flattened)
    csv_data = []
    for bbox in bounding_boxes:
        row = {
            "case_id_number": bbox["case_id_number"],
            "case_title": bbox["case_title"],
            "valid_time": bbox["valid_time"],
            "event_start_date": bbox["event_start_date"],
            "event_end_date": bbox["event_end_date"],
            "latitude_min": bbox["latitude_min"],
            "latitude_max": bbox["latitude_max"],
            "longitude_min": bbox["longitude_min"],
            "longitude_max": bbox["longitude_max"],
            "center_lat": bbox["center_lat"],
            "center_lon": bbox["center_lon"],
            "buffer_km": bbox["buffer_km"],
            "original_lat_min": bbox["original_extent"]["lat_min"],
            "original_lat_max": bbox["original_extent"]["lat_max"],
            "original_lon_min": bbox["original_extent"]["lon_min"],
            "original_lon_max": bbox["original_extent"]["lon_max"],
            "peak_lat": bbox.get("peak_lat", None),
            "peak_lon": bbox.get("peak_lon", None),
            "peak_value": bbox.get("peak_value", None),
            "max_distance_km": bbox.get("max_distance_km", None),
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = output_path_obj.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    return df


def main(
    pph_data: Optional[Union[xr.DataArray, str]] = None,
    events_yaml_path: str = "src/extremeweatherbench/data/events.yaml",
    output_path: str = "data_prep/pph_severe_convection_bounding_boxes",
    buffer_km: float = 250,
    threshold: float = 0.01,
    max_distance_km: float = 2000,
    max_blob_size_km: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Main function to create bounding boxes.

    Args:
        pph_data: Either an xarray.DataArray with PPH data, or a string
                 path to load it from. If None, raises an error.
        events_yaml_path: Path to events.yaml
        output_path: Output file path (without extension)
        buffer_km: Buffer distance in kilometers
        threshold: Minimum PPH value to consider
        max_distance_km: Maximum distance from peak (default 2000km)
        max_blob_size_km: Max blob diagonal for extension (default
                         0.75*max_distance_km)

    Returns:
        Tuple of (bounding_boxes list, DataFrame with boxes)
    """
    # Load or use existing PPH data
    if pph_data is None:
        raise ValueError(
            "pph_data is required. Pass either:\n"
            "  - An xarray.DataArray object (e.g., pph_sparse)\n"
            "  - A string path to a netCDF file"
        )
    elif isinstance(pph_data, str):
        pph_data = xr.open_dataarray(pph_data)

    # Load all events and filter to severe convection
    all_cases = cases.load_individual_cases_from_yaml(events_yaml_path)
    severe_convection_cases = [
        n for n in all_cases if n.event_type == "severe_convection"
    ]

    # Match PPH times to events
    event_time_map = match_pph_times_to_events(pph_data, severe_convection_cases)
    # Create bounding boxes with peak-distance approach
    bounding_boxes = create_bounding_boxes(
        pph_data,
        event_time_map,
        all_cases,
        buffer_km=buffer_km,
        threshold=threshold,
        max_distance_km=max_distance_km,
        max_blob_size_km=max_blob_size_km,
    )

    # Save results
    df = save_bounding_boxes(bounding_boxes, output_path)

    return bounding_boxes, df


if __name__ == "__main__":
    # Example usage when run as script
    # You'll need to modify this to point to your PPH data file
    import sys

    if len(sys.argv) > 1:
        pph_file = sys.argv[1]
        bounding_boxes, df = main(pph_data=pph_file)
    else:
        print("Usage: python create_pph_bounding_boxes.py <pph_data_file>")
        print("\nOr import and use in a notebook/script:")
        print("  from create_pph_bounding_boxes import main")
        print("  bounding_boxes, df = main(pph_data=pph_sparse)")
        sys.exit(1)
