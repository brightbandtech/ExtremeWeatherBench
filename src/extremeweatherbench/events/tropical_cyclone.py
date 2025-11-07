"""Tropical cyclone track detection from gridded datasets."""

import hashlib
import logging
from collections import namedtuple
from itertools import product
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.spatial as spatial
import xarray as xr
from skimage import measure
from skimage.feature import peak_local_max

from extremeweatherbench import calc, utils

logger = logging.getLogger(__name__)


def generate_tc_tracks_by_init_time(
    sea_level_pressure: xr.DataArray,
    wind_speed: xr.DataArray,
    geopotential_thickness: Optional[xr.DataArray],
    tc_track_analysis_data: xr.DataArray,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    min_distance: int = 5,
    max_spatial_distance_degrees: float = 5.0,
    max_temporal_hours: float = 48,
    use_contour_validation: bool = True,
    min_track_length: int = 10,
) -> xr.Dataset:
    """Process all init_times and detect tropical cyclones.

    Args:
        sea_level_pressure: Sea level pressure DataArray
        wind_speed: Wind speed DataArray
        geopotential_thickness: Geopotential thickness DataArray (optional)
        tc_track_analysis_data: Tropical cyclone track analysis data
        max_temporal_hours: Maximum temporal window from init_time
        max_spatial_distance_degrees: Max spatial distance for TC track
            data filtering
        min_distance: Minimum distance between detected peaks
        slp_contour_magnitude: SLP contour threshold for validation
        dz_contour_magnitude: DZ contour threshold for validation
        use_contour_validation: Whether to use contour validation
        min_track_length: Minimum consecutive lead times for valid tracks

    Returns:
        xr.Dataset with detected tropical cyclone tracks
    """
    logger.info("Generating TC tracks by init_time")
    # Convert TC track data to DataFrame
    tc_track_data_df = tc_track_analysis_data.to_dataframe().reset_index()
    tc_track_data_df["valid_time"] = pd.to_datetime(tc_track_data_df["valid_time"])

    time_coord = sea_level_pressure.valid_time
    init_time_coord = sea_level_pressure.init_time
    latitude = sea_level_pressure.latitude
    longitude = sea_level_pressure.longitude

    # Filter tc_track_data_df to only include valid times in forecast
    logger.debug("tc_track_data_df shape before filtering: %s", tc_track_data_df.shape)
    logger.debug("Forecast time_coord values: %s...", time_coord.values[:5])

    tc_track_data_df = tc_track_data_df[
        pd.to_datetime(tc_track_data_df["valid_time"]).isin(
            pd.to_datetime(time_coord.values.ravel())
        )
    ]
    logger.debug("tc_track_data_df shape after filtering: %s", tc_track_data_df.shape)
    if len(tc_track_data_df) == 0:
        logger.warning("No TC track data matches forecast times - returning empty")

    # Transform data to have init_time as a dimension
    slp_transformed = utils.convert_valid_time_to_init_time(sea_level_pressure)
    wind_transformed = utils.convert_valid_time_to_init_time(wind_speed)
    if geopotential_thickness is not None:
        dz_transformed = utils.convert_valid_time_to_init_time(geopotential_thickness)
    else:
        # Create NaN-filled array as sentinel for "no dz data"
        dz_transformed = xr.full_like(slp_transformed, np.nan)

    # Create init_time_idx array for tracking
    n_init_times = len(slp_transformed.init_time)
    init_time_idx_array = xr.DataArray(
        np.arange(n_init_times),
        dims=["init_time"],
        coords={"init_time": slp_transformed.init_time},
    )

    # Use apply_ufunc to parallelize over init_time
    logger.debug("Processing %d init_times", n_init_times)
    use_validation = use_contour_validation and geopotential_thickness is not None
    results = xr.apply_ufunc(
        _process_single_init_time,
        slp_transformed,
        wind_transformed,
        dz_transformed,
        init_time_idx_array,
        kwargs={
            "init_time_values": slp_transformed.init_time.values,
            "init_time_coord": init_time_coord,
            "time_coord": time_coord,
            "latitude": latitude.values,
            "longitude": longitude.values,
            "tc_track_data_df": tc_track_data_df,
            "min_distance": min_distance,
            "max_spatial_distance_degrees": max_spatial_distance_degrees,
            "max_temporal_hours": max_temporal_hours,
            "slp_contour_magnitude": slp_contour_magnitude,
            "dz_contour_magnitude": dz_contour_magnitude,
            "use_contour_validation": use_validation,
            "min_track_length": min_track_length,
        },
        input_core_dims=[
            ["lead_time", "latitude", "longitude"],
            ["lead_time", "latitude", "longitude"],
            ["lead_time", "latitude", "longitude"],
            [],  # init_time_idx is scalar per init_time
        ],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Extract detections from results (already filtered by init_time)
    all_detections = []
    for result in results.values:
        all_detections.extend(result["detections"])

    logger.debug("Total detections after filtering: %s", len(all_detections))

    # Extract arrays from detections
    if all_detections:
        n_detections = len(all_detections)
        lt_indices = np.array([d["lt_idx"] for d in all_detections])
        vt_indices = np.array([d["vt_idx"] for d in all_detections])
        track_ids = np.array([d["track_id"] for d in all_detections])
        lats = np.array([d["latitude"] for d in all_detections])
        lons = np.array([d["longitude"] for d in all_detections])
        slp_vals = np.array([d["slp"] for d in all_detections])
        wind_vals = np.array([d["wind"] for d in all_detections])
    else:
        n_detections = 0
        lt_indices = np.array([])
        vt_indices = np.array([])
        track_ids = np.array([])
        lats = np.array([])
        lons = np.array([])
        slp_vals = np.array([])
        wind_vals = np.array([])

    # Convert to xarray Dataset
    return _convert_detections_to_dataset(
        n_detections,
        lt_indices,
        vt_indices,
        track_ids,
        lats,
        lons,
        slp_vals,
        wind_vals,
        sea_level_pressure.lead_time,
        sea_level_pressure.valid_time,
    )


def _process_single_init_time(
    slp_data: np.ndarray,  # (lead_time, lat, lon)
    wind_data: np.ndarray,  # (lead_time, lat, lon)
    dz_data: np.ndarray,  # (lead_time, lat, lon) or all NaN
    init_time_idx: int,
    init_time_values: np.ndarray,  # Unique init_time values from transform
    init_time_coord: xr.DataArray,  # Original 2D init_time coordinate
    time_coord: xr.DataArray,  # Original time coordinate
    latitude: np.ndarray,
    longitude: np.ndarray,
    tc_track_data_df: pd.DataFrame,
    min_distance: int,
    max_spatial_distance_degrees: float,
    max_temporal_hours: float,
    slp_contour_magnitude: float,
    dz_contour_magnitude: float,
    use_contour_validation: bool,
    min_track_length: int,
) -> dict:
    """Process TC detection for single init_time (vectorized version).

    Args:
        slp_data: Sea level pressure (lead_time, lat, lon)
        wind_data: Wind speed (lead_time, lat, lon)
        dz_data: Geopotential thickness (lead_time, lat, lon) or all NaN
        init_time_idx: Index of this init_time (for global track IDs)
        init_time_values: Unique init_time values from transformed data
        init_time_coord: Original 2D init_time coordinate
        time_coord: Original valid_time coordinate
        latitude: Latitude coordinates
        longitude: Longitude coordinates
        tc_track_data_df: TC track data for validation
        min_distance: Min distance between peaks
        max_spatial_distance_degrees: Max spatial distance
        max_temporal_hours: Max temporal buffer
        slp_contour_magnitude: SLP contour threshold
        dz_contour_magnitude: DZ contour threshold
        use_contour_validation: Whether to use contour validation
        min_track_length: Minimum consecutive lead times for valid tracks

    Returns:
        Dict containing filtered list of detections for this init_time
    """
    # Get the init_time value for this index
    current_init_time = init_time_values[init_time_idx]

    # Find all (lead_time, valid_time) pairs for this init_time
    mask = np.abs(init_time_coord - current_init_time) <= np.timedelta64(1, "s")
    lt_indices, vt_indices = np.where(mask)

    # Get valid times and sort by them
    valid_times = pd.to_datetime(
        [
            time_coord[vt].item() if hasattr(time_coord[vt], "item") else time_coord[vt]
            for vt in vt_indices
        ]
    )
    sort_order = np.argsort(valid_times)

    # Sorted indices and valid times
    lt_indices_seq = lt_indices[sort_order].astype(np.intp)
    vt_indices_seq = vt_indices[sort_order].astype(np.intp)
    valid_times_seq = [valid_times[i] for i in sort_order]
    n_timesteps = len(lt_indices_seq)

    # Data is already (lead_time, lat, lon) for this init_time
    # Check if dz_data is all NaN (sentinel for None)
    has_dz = not np.all(np.isnan(dz_data))
    if not has_dz:
        dz_data = None

    detections = []
    active_tracks: dict[int, dict[str, float]] = {}
    # Global track ID offset based on init_time_idx
    track_id_offset = init_time_idx * 10000

    # Find peaks for all timesteps at once
    all_peaks = []
    for timestep_idx in range(n_timesteps):
        slp_slice = slp_data[timestep_idx]
        wind_slice = wind_data[timestep_idx]
        dz_slice = dz_data[timestep_idx] if dz_data is not None else None

        peaks = _find_peaks_batch(
            slp_slice,
            wind_slice,
            dz_slice,
            timestep_idx,
            valid_times=valid_times_seq,
            tc_track_data_df=tc_track_data_df,
            min_distance=min_distance,
            latitude=latitude,
            longitude=longitude,
            max_spatial_distance_degrees=max_spatial_distance_degrees,
            max_temporal_hours=max_temporal_hours,
            slp_contour_magnitude=slp_contour_magnitude,
            dz_contour_magnitude=dz_contour_magnitude,
            use_contour_validation=use_contour_validation,
        )

        all_peaks.append(
            {
                "peaks": peaks,
                "slp_slice": slp_slice,
                "wind_slice": wind_slice,
                "timestep_idx": timestep_idx,
            }
        )

    # Now assign tracks sequentially
    next_track_id = 0
    for peak_data in all_peaks:
        peaks = peak_data["peaks"]
        slp_slice = peak_data["slp_slice"]
        wind_slice = peak_data["wind_slice"]
        timestep_idx = peak_data["timestep_idx"]

        lt_idx = lt_indices_seq[timestep_idx]
        vt_idx = vt_indices_seq[timestep_idx]

        if len(peaks) > 0:
            peak_lats = latitude[peaks[:, 0]]
            peak_lons = longitude[peaks[:, 1]]
            peak_slps = slp_slice[peaks[:, 0], peaks[:, 1]]
            peak_winds = wind_slice[peaks[:, 0], peaks[:, 1]]

            unassigned_peaks = list(range(len(peaks)))

            # Match to existing tracks (within 8 degrees)
            for track_id, last_pos in list(active_tracks.items()):
                if not unassigned_peaks:
                    break

                best_idx = None
                best_distance = float("inf")

                for peak_idx in unassigned_peaks:
                    distance = calc.haversine_distance(
                        [peak_lats[peak_idx], peak_lons[peak_idx]],
                        [last_pos["latitude"], last_pos["longitude"]],
                        units="deg",
                    )
                    if distance <= 8.0 and distance < best_distance:
                        best_distance = distance
                        best_idx = peak_idx

                if best_idx is not None:
                    detections.append(
                        {
                            "lt_idx": lt_idx,
                            "vt_idx": vt_idx,
                            "track_id": track_id + track_id_offset,
                            "latitude": peak_lats[best_idx],
                            "longitude": peak_lons[best_idx],
                            "slp": peak_slps[best_idx],
                            "wind": peak_winds[best_idx],
                        }
                    )

                    active_tracks[track_id] = {
                        "latitude": peak_lats[best_idx],
                        "longitude": peak_lons[best_idx],
                    }
                    unassigned_peaks.remove(best_idx)

            # Create new tracks for unassigned peaks
            for peak_idx in unassigned_peaks:
                track_id = next_track_id
                next_track_id += 1

                detections.append(
                    {
                        "lt_idx": lt_idx,
                        "vt_idx": vt_idx,
                        "track_id": track_id + track_id_offset,
                        "latitude": peak_lats[peak_idx],
                        "longitude": peak_lons[peak_idx],
                        "slp": peak_slps[peak_idx],
                        "wind": peak_winds[peak_idx],
                    }
                )

                active_tracks[track_id] = {
                    "latitude": peak_lats[peak_idx],
                    "longitude": peak_lons[peak_idx],
                }

    # Filter tracks by min_track_length consecutive lead times
    if detections and min_track_length > 1:
        # Group detections by track_id and collect lead time indices
        track_lead_times: dict[int, set[int]] = {}
        for detection in detections:
            track_id = detection["track_id"]
            lt_idx = detection["lt_idx"]

            if track_id not in track_lead_times:
                track_lead_times[track_id] = set()
            # Convert lt_idx to int
            lt_idx_int: int
            if hasattr(lt_idx, "item"):
                lt_idx_int = int(lt_idx.item())
            else:
                lt_idx_int = int(lt_idx)
            track_lead_times[track_id].add(lt_idx_int)

        # Check for consecutive lead times for each track
        valid_track_ids: set[int] = set()
        for track_id, lt_indices in track_lead_times.items():
            # Sort lead time indices to check for consecutive sequences
            sorted_lt_indices = sorted(lt_indices)

            # Find longest consecutive sequence
            max_consecutive = 1
            current_consecutive = 1

            for i in range(1, len(sorted_lt_indices)):
                if sorted_lt_indices[i] == sorted_lt_indices[i - 1] + 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1

            # Only keep tracks with min_track_length consecutive lead times
            if max_consecutive >= min_track_length:
                valid_track_ids.add(track_id)

        # Filter detections to only include valid tracks
        filtered_detections = [
            d for d in detections if d["track_id"] in valid_track_ids
        ]
        logger.debug(
            "Init %d: Filtered %d -> %d detections (min_track_length=%d)",
            init_time_idx,
            len(detections),
            len(filtered_detections),
            min_track_length,
        )
        detections = filtered_detections

    return {"detections": detections}


def _convert_detections_to_dataset(
    n_detections: int,
    lt_indices: np.ndarray,
    vt_indices: np.ndarray,
    track_ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    slp_vals: np.ndarray,
    wind_vals: np.ndarray,
    lead_time_coord: xr.DataArray,
    valid_time_coord: xr.DataArray,
) -> xr.Dataset:
    """Convert detection arrays to compact xarray Dataset."""

    # Handle case with no detections
    if n_detections == 0:
        # Create empty dataset with proper dimensions
        return xr.Dataset(
            {
                "track_id": (
                    ["lead_time", "valid_time"],
                    np.full((0, 0), -1, dtype=int),
                ),
                "air_pressure_at_mean_sea_level": (
                    ["lead_time", "valid_time"],
                    np.full((0, 0), np.nan),
                ),
                "surface_wind_speed": (
                    ["lead_time", "valid_time"],
                    np.full((0, 0), np.nan),
                ),
                "latitude": (["lead_time", "valid_time"], np.full((0, 0), np.nan)),
                "longitude": (["lead_time", "valid_time"], np.full((0, 0), np.nan)),
            },
            coords={
                "lead_time": [],
                "valid_time": [],
            },
        )

    # Get unique combinations of lead_time and valid_time indices
    unique_combinations: dict[tuple[int, int], list[int]] = {}
    for i in range(n_detections):
        # Convert numpy scalars to regular integers for use as dictionary keys
        lt_idx = int(lt_indices[i])
        vt_idx = int(vt_indices[i])
        key = (lt_idx, vt_idx)
        if key not in unique_combinations:
            unique_combinations[key] = []
        unique_combinations[key].append(i)

    # Determine output dimensions
    if unique_combinations:
        max_lt = max(key[0] for key in unique_combinations.keys())
        max_vt = max(key[1] for key in unique_combinations.keys())

        # Get the actual lead_time values
        lead_times = lead_time_coord.values[: max_lt + 1]

        # Get valid_time values
        # Flatten (works for any dimensionality) and get unique sorted values
        valid_times = np.unique(valid_time_coord.values.ravel())
        valid_times = valid_times[: max_vt + 1]
    else:
        lead_times = np.array([])
        valid_times = np.array([])

    # Handle multiple detections per time step by creating additional
    # track_id dimension
    max_tracks_per_timestep = (
        max(len(indices) for indices in unique_combinations.values())
        if unique_combinations
        else 0
    )

    # Initialize output arrays
    output_shape = (len(lead_times), len(valid_times), max_tracks_per_timestep)
    track_id_out = np.full(output_shape, -1, dtype=int)
    slp_out = np.full(output_shape, np.nan)
    wind_out = np.full(output_shape, np.nan)
    lat_out = np.full(output_shape, np.nan)
    lon_out = np.full(output_shape, np.nan)

    # Fill arrays
    for (lt_idx, vt_idx), detection_indices in unique_combinations.items():
        for track_idx, det_idx in enumerate(detection_indices):
            # Extract values from numpy arrays
            track_id_val = int(track_ids[det_idx])
            slp_val = float(slp_vals[det_idx])
            wind_val = float(wind_vals[det_idx])
            lat_val = float(lats[det_idx])
            lon_val = float(lons[det_idx])

            track_id_out[lt_idx, vt_idx, track_idx] = track_id_val
            slp_out[lt_idx, vt_idx, track_idx] = slp_val
            wind_out[lt_idx, vt_idx, track_idx] = wind_val
            lat_out[lt_idx, vt_idx, track_idx] = lat_val
            lon_out[lt_idx, vt_idx, track_idx] = lon_val

    # Create dataset
    return xr.Dataset(
        {
            "track_id": (["lead_time", "valid_time", "track"], track_id_out),
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time", "track"],
                slp_out,
            ),
            "surface_wind_speed": (["lead_time", "valid_time", "track"], wind_out),
            "latitude": (["lead_time", "valid_time", "track"], lat_out),
            "longitude": (["lead_time", "valid_time", "track"], lon_out),
        },
        coords={
            "lead_time": lead_times,
            "valid_time": valid_times,
            "track": np.arange(max_tracks_per_timestep),
        },
    )


Location = namedtuple("Location", ["latitude", "longitude"])
# Global cache for TC track data to avoid recomputation across child classes
_TC_TRACK_CACHE: Dict[str, xr.Dataset] = {}

# Global registry for tropical cyclone track data to be used in TC filtering
_TRACK_DATA_REGISTRY: Dict[str, xr.Dataset] = {}


def register_tc_track_data(case_id: str, tc_track_data: xr.Dataset) -> None:
    """Register tropical cyclone track data for a specific case to be used in TC
    filtering.

    Args:
        case_id: Unique identifier for the case.
        tc_track_data: tropical cyclone track dataset with valid_time, latitude,
            longitude.
    """
    global _TRACK_DATA_REGISTRY
    _TRACK_DATA_REGISTRY[case_id] = tc_track_data


def get_tc_track_data(case_id: str) -> Optional[xr.Dataset]:
    """Get registered tropical cyclone track data for a specific case.

    Args:
        case_id: Unique identifier for the case.

    Returns:
        tropical cyclone track dataset if available, None otherwise.
    """
    import logging

    logger = logging.getLogger(__name__)

    global _TRACK_DATA_REGISTRY
    logger.debug(
        "get_tc_track_data called with case_id=%s (type=%s)", case_id, type(case_id)
    )
    logger.debug("Registry keys: %s", list(_TRACK_DATA_REGISTRY.keys()))
    result = _TRACK_DATA_REGISTRY.get(case_id, None)
    logger.debug("Found data: %s", result is not None)
    return result


def clear_tc_track_data_registry() -> None:
    """Clear the tropical cyclone track data registry."""
    global _TRACK_DATA_REGISTRY
    _TRACK_DATA_REGISTRY.clear()


def _generate_cache_key(data: xr.Dataset) -> str:
    """Generate a hash key for the dataset to use as cache key."""
    # Create a hash based on data shape, coordinates, and first/last values
    # This is a lightweight way to create a unique key without hashing entire arrays
    key_components = []

    # Add dataset shape and coordinate info
    for var_name in [
        "air_pressure_at_mean_sea_level",
        "geopotential",
        "surface_eastward_wind",
        "surface_northward_wind",
    ]:
        if var_name in data.data_vars:
            var = data[var_name]
            key_components.extend(
                [
                    str(var.shape),
                    str(var.dims),
                    # Sample a few values for uniqueness (only if dims have size > 0)
                    str(
                        float(
                            var.isel(
                                {dim: 0 for dim in var.dims if var.sizes[dim] > 0}
                            ).values
                        )
                    )
                    if all(var.sizes[dim] > 0 for dim in var.dims)
                    else "empty",
                    str(
                        float(
                            var.isel(
                                {dim: -1 for dim in var.dims if var.sizes[dim] > 0}
                            ).values
                        )
                    )
                    if all(var.sizes[dim] > 0 for dim in var.dims)
                    else "empty",
                ]
            )

    # Add time and coordinate info
    if "time" in data.coords:
        key_components.extend([str(data.time.values[0]), str(data.time.values[-1])])

    # Create hash
    key_string = "|".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()


def find_furthest_contour_from_point(
    contour: Union[np.ndarray, Sequence[tuple[float, float]]],
    point: Union[np.ndarray, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Find the two points in a contour that are furthest apart.

    From
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-
    in-plot-with-many-points-in-python

    Args:
        contour: The contour to find the furthest point from.
        point: The point to find the furthest point from.

    Returns:
        The furthest point from the contour as a tuple of x,y coordinates.
    """

    # Convert inputs to numpy arrays if needed
    contour_array = (
        np.array(contour) if not isinstance(contour, np.ndarray) else contour
    )
    point_array = np.array(point) if not isinstance(point, np.ndarray) else point

    # Calculate distances from point to all points in contour
    distances = spatial.distance.cdist([point_array], contour_array)[0]
    # Find index of point with maximum distance
    furthest_idx = np.argmax(distances)

    # Get the furthest point
    furthest_point = contour_array[furthest_idx]
    return furthest_point, furthest_point  # Return as tuple of arrays


def find_contours_from_point_specified_field(
    field: xr.DataArray, point: tuple[float, float], level: float
) -> Sequence[Sequence[tuple[float, float]]]:
    """Find the contours from a point for a specified field.

    Args:
        field: The field to find the contours from.
        point: The point at which the field is subtracted from to find the
            anomaly contours.
        level: The anomaly level to find the contours at.

    Returns:
        The contours as a list of tuples of latitude and longitude.
    """
    field_at_point = field - field.isel(latitude=point[0], longitude=point[1])
    contours = measure.find_contours(
        field_at_point.values, level=level, positive_orientation="high"
    )
    return contours


def find_valid_contour_from_point(
    contour: Sequence[tuple[float, float]],
    point: tuple[float, float],
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> float:
    """Find the great circle distance from a point to a contour.

    Args:
        contour: The contour to find the great circle distance to
        point: The point to find the great circle distance to
        latitude: Latitude DataArray for coordinate mapping
        longitude: Longitude DataArray for coordinate mapping

    Returns:
        Great circle distance from the point to the contour
    """
    furthest_point, _ = find_furthest_contour_from_point(contour, point)
    gc_distance_point_latlon = calc.convert_from_cartesian_to_latlon(
        furthest_point, latitude, longitude
    )
    point_latlon = calc.convert_from_cartesian_to_latlon(point, latitude, longitude)
    gc_distance_contour_distance = calc.haversine_distance(
        [gc_distance_point_latlon[0], gc_distance_point_latlon[1]],
        [point_latlon[0], point_latlon[1]],
        units="degrees",
    )
    # Ensure we return a float
    if isinstance(gc_distance_contour_distance, xr.DataArray):
        return float(gc_distance_contour_distance.values)
    return float(gc_distance_contour_distance)


def find_valid_candidates(
    slp_contours: Sequence[Sequence[tuple[float, float]]],
    dz_contours: Sequence[Sequence[tuple[float, float]]],
    point: tuple[float, float],
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    time_counter: int,
    orography: Optional[xr.DataArray] = None,
    max_gc_distance_slp_contour: float = 5.5,
    max_gc_distance_dz_contour: float = 6.5,
    orography_filter_threshold: float = 150,
) -> Optional[Location]:
    """Find valid candidate coordinate for a TC.

    Args:
        slp_contours: List of SLP contours
        dz_contours: List of DZ contours
        point: Point to find the valid candidate for
        latitude: Latitude DataArray for coordinate mapping
        longitude: Longitude DataArray for coordinate mapping
        time_counter: Time counter
        orography: Orography DataArray (optional)
        max_gc_distance_slp_contour: Max great circle distance for SLP contour
        max_gc_distance_dz_contour: Max great circle distance for DZ contour
        orography_filter_threshold: Threshold for the orography filter

    Returns:
        Valid candidate Location or None
    """
    lat_val = calc.convert_from_cartesian_to_latlon(point, latitude, longitude)[0]
    lon_val = calc.convert_from_cartesian_to_latlon(point, latitude, longitude)[1]

    # Check orography filter if data is available
    if orography is not None:
        orography_filter = (
            orography.sel(latitude=lat_val, longitude=lon_val, method="nearest")
            .min()
            .values
            < orography_filter_threshold
            if time_counter < 8
            else True
        )
    else:
        orography_filter = True

    latitude_filter = abs(lat_val) < 50 if time_counter < 10 else True

    for slp_contour, dz_contour in product(slp_contours, dz_contours):
        if (
            all(np.isclose(slp_contour[-1], slp_contour[0]))
            and all(np.isclose(dz_contour[-1], dz_contour[0]))
            and measure.points_in_poly([[point[0], point[1]]], slp_contour)[0]
            and measure.points_in_poly([[point[0], point[1]]], dz_contour)[0]
            and find_valid_contour_from_point(slp_contour, point, latitude, longitude)
            < max_gc_distance_slp_contour
            and find_valid_contour_from_point(dz_contour, point, latitude, longitude)
            < max_gc_distance_dz_contour
            and orography_filter
            and latitude_filter
        ):
            return Location(latitude=lat_val, longitude=lon_val)
    return None


def _safe_extract_value(
    array_or_scalar: Union[xr.DataArray, np.ndarray, float],
) -> Union[float, str]:
    """Safely extract scalar value from DataArray or return as-is."""
    if hasattr(array_or_scalar, "item") and hasattr(array_or_scalar, "ndim"):
        # Handle numpy arrays and xarray DataArrays
        if array_or_scalar.ndim == 0:
            return array_or_scalar.item()
        else:
            # For arrays with more than 0 dimensions, return the first element
            if hasattr(array_or_scalar, "flat"):
                return array_or_scalar.flat[0]
            else:
                return array_or_scalar
    else:
        return array_or_scalar


def _find_peaks_batch(
    slp_slice: np.ndarray,
    wind_slice: np.ndarray,
    dz_slice: Optional[np.ndarray],
    timestep_idx: int,
    valid_times: list,
    tc_track_data_df: pd.DataFrame,
    min_distance: int,
    latitude: np.ndarray,
    longitude: np.ndarray,
    max_spatial_distance_degrees: float,
    max_temporal_hours: float,
    slp_contour_magnitude: float,
    dz_contour_magnitude: float,
    use_contour_validation: bool,
) -> np.ndarray:
    """Wrapper for vectorized peak finding across time slices.

    This function is designed to work with xr.apply_ufunc's vectorize mode.

    Args:
        slp_slice: Single time slice of SLP data
        wind_slice: Single time slice of wind data (unused here)
        dz_slice: Single time slice of DZ data
        timestep_idx: Index of current timestep
        valid_times: List of valid times for all timesteps
        tc_track_data_df: TC track data
        min_distance: Minimum distance between peaks
        latitude: Latitude coordinates
        longitude: Longitude coordinates
        max_spatial_distance_degrees: Max spatial distance
        max_temporal_hours: Max temporal buffer for genesis
        slp_contour_magnitude: SLP contour magnitude
        dz_contour_magnitude: DZ contour magnitude
        use_contour_validation: Whether to use contour validation

    Returns:
        Array of peak coordinates
    """
    # Convert scalar to int if needed
    if hasattr(timestep_idx, "item"):
        timestep_idx = int(timestep_idx.item())
    else:
        timestep_idx = int(timestep_idx)

    current_valid_time = pd.Timestamp(valid_times[timestep_idx])
    is_first_timestep = timestep_idx == 0

    return _find_peaks_for_time_slice(
        slp_slice,
        current_valid_time,
        tc_track_data_df,
        min_distance,
        latitude,
        longitude,
        max_spatial_distance_degrees,
        max_temporal_hours,
        dz_slice if dz_slice is not None else np.array([]),
        slp_contour_magnitude,
        dz_contour_magnitude,
        use_contour_validation,
        is_first_timestep,
    )


def _find_peaks_for_time_slice(
    slp_slice: np.ndarray,
    current_valid_time: pd.Timestamp,
    tc_track_data_df: pd.DataFrame,
    min_distance: int,
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    max_spatial_distance_degrees: float = 5.0,
    max_temporal_hours: float = 48.0,
    dz_slice: Optional[np.ndarray] = None,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    use_contour_validation: bool = True,
    is_first_timestep: bool = False,
) -> np.ndarray:
    """Find peaks with tropical cyclone track data filtering.

    Args:
        slp_slice: Sea level pressure slice
        current_valid_time: Current valid time being processed
        tc_track_data_df: Tropical cyclone track data
        min_distance: Minimum distance between peaks
        lat_coords: Latitude coordinates
        lon_coords: Longitude coordinates
        max_spatial_distance_degrees: Max spatial distance
        max_temporal_hours: Max temporal buffer for genesis
        dz_slice: Geopotential thickness slice
        slp_contour_magnitude: SLP contour magnitude
        dz_contour_magnitude: DZ contour magnitude
        use_contour_validation: Whether to use contour validation
        is_first_timestep: If True, applies temporal buffer
    """

    # Filter tropical cyclone track data temporally
    time_diff = np.abs(
        (tc_track_data_df["valid_time"] - current_valid_time).dt.total_seconds() / 3600
    )

    # For first timestep (genesis), allow buffer; otherwise exact
    if is_first_timestep:
        temporal_mask = time_diff <= max_temporal_hours
    else:
        temporal_mask = time_diff == 0

    nearby_tc_track_data = tc_track_data_df[temporal_mask]

    if len(nearby_tc_track_data) == 0:
        logger.debug("No TC track data at time %s", current_valid_time)
        return np.array([])
    else:
        n_rows = len(nearby_tc_track_data)
        logger.debug("Found %s TC track data rows at %s", n_rows, current_valid_time)

    # Check for valid data
    if np.all(np.isnan(slp_slice)):
        return np.array([])

    # Apply pressure threshold
    pressure_threshold = 100500  # 1005 hPa in Pa
    low_pressure_mask = slp_slice < pressure_threshold

    if not np.any(low_pressure_mask):
        return np.array([])

    # OPTIMIZED: Vectorized spatial masking
    if lat_coords is None or lon_coords is None:
        return np.array([])

    spatial_mask = _create_spatial_mask(
        lat_coords, lon_coords, nearby_tc_track_data, max_spatial_distance_degrees
    )

    # Combine spatial and pressure masks
    combined_mask = np.logical_and(spatial_mask, low_pressure_mask)

    if not np.any(combined_mask):
        return np.array([])

    # Apply combined mask for peak detection
    masked_slp = np.where(combined_mask, -slp_slice, -999999)
    # Find peaks
    peaks = peak_local_max(
        masked_slp,
        min_distance=min_distance,
        exclude_border=False,
        threshold_abs=-pressure_threshold,
    )

    # Apply contour validation if requested and DZ data is available
    if use_contour_validation and dz_slice is not None and len(peaks) > 0:
        validated_peaks = []

        # Create temporary DataArrays for contour analysis
        lat_da = xr.DataArray(lat_coords, dims=["latitude"])
        lon_da = xr.DataArray(lon_coords, dims=["longitude"])

        slp_da = xr.DataArray(
            slp_slice,
            dims=["latitude", "longitude"],
            coords={"latitude": lat_da, "longitude": lon_da},
        )
        dz_da = xr.DataArray(
            dz_slice,
            dims=["latitude", "longitude"],
            coords={"latitude": lat_da, "longitude": lon_da},
        )

        for peak in peaks:
            # Generate contours for this peak
            slp_contours = find_contours_from_point_specified_field(
                slp_da, peak, slp_contour_magnitude
            )
            dz_contours = find_contours_from_point_specified_field(
                dz_da, peak, dz_contour_magnitude
            )

            # Check if this peak passes contour validation
            candidate = find_valid_candidates(
                slp_contours,
                dz_contours,
                peak,
                lat_da,
                lon_da,
                time_counter=0,
                orography=None,
                max_gc_distance_slp_contour=5.5,
                max_gc_distance_dz_contour=6.5,
            )

            if candidate is not None:
                validated_peaks.append(peak)

        return (
            np.array(validated_peaks) if validated_peaks else np.array([]).reshape(0, 2)
        )

    return peaks


def _create_spatial_mask(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    nearby_tc_track_data: pd.DataFrame,
    max_distance_degrees: float,
):
    """Create spatial mask using vectorized distance calculation."""

    # Create meshgrid of all lat/lon coordinates
    lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing="ij")

    # Initialize mask
    spatial_mask = np.zeros_like(lat_grid, dtype=bool)

    # For each tropical cyclone track data point, compute distances to all grid points
    # vectorized
    for _, tc_track_data_row in nearby_tc_track_data.iterrows():
        tc_track_data_lat = tc_track_data_row["latitude"]
        tc_track_data_lon = tc_track_data_row["longitude"]

        # Vectorized distance calculation
        distances = calc.haversine_distance(
            [lat_grid, lon_grid],
            [tc_track_data_lat, tc_track_data_lon],
            units="degrees",
        )

        # Update mask where distance is within threshold
        spatial_mask |= distances <= max_distance_degrees

    return spatial_mask


def _create_empty_tracks_dataset() -> xr.Dataset:
    """Create an empty tracks dataset with proper structure."""
    return xr.Dataset(
        {
            "track_id": (["lead_time", "valid_time"], np.array([]).reshape(0, 0)),
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time"],
                np.array([]).reshape(0, 0),
            ),
            "surface_wind_speed": (
                ["lead_time", "valid_time"],
                np.array([]).reshape(0, 0),
            ),
            "latitude": (["lead_time", "valid_time"], np.array([]).reshape(0, 0)),
            "longitude": (["lead_time", "valid_time"], np.array([]).reshape(0, 0)),
        },
        coords={
            "lead_time": [],
            "valid_time": [],
        },
    )
