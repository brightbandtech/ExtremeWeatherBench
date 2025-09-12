import hashlib
import logging
import math
from collections import namedtuple
from itertools import product
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.spatial as spatial
import xarray as xr
from cartopy.io.shapereader import Reader, natural_earth
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from skimage import measure
from skimage.feature import peak_local_max

from extremeweatherbench import calc, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_landfall_metric(
    forecast: xr.Dataset,
    target: xr.Dataset,
    approach: Literal["first", "next", "all"] = "first",
    metric_type: Literal["displacement", "timing", "intensity"] = "displacement",
    aggregation: Literal["mean", "max", "min"] = "mean",
    units: Literal["hours", "days"] = "hours",
    intensity_var: str = "surface_wind_speed",
):
    """Unified function to compute landfall metrics with different approaches.

    Args:
        forecast: Forecast TC track dataset
        target: Target/analysis TC track dataset
        approach: Landfall detection approach ('first', 'next', 'all')
        metric_type: Type of metric ('displacement', 'timing', 'intensity')
        aggregation: How to aggregate multiple values
        units: Time units for timing metrics ('hours', 'days')
        intensity_var: Variable for intensity metrics

    Returns:
        xarray.DataArray with computed metric values
    """
    # Convert to datasets if needed
    if isinstance(forecast, xr.DataArray):
        forecast = forecast.to_dataset()
    if isinstance(target, xr.DataArray):
        target = target.to_dataset()

    if approach == "first":
        return _compute_first_landfall_metric(
            forecast, target, metric_type, units, intensity_var
        )
    elif approach == "next":
        return _compute_next_landfall_metric(
            forecast, target, metric_type, aggregation, units, intensity_var
        )
    elif approach == "all":
        return _compute_all_landfalls_metric(
            forecast, target, metric_type, aggregation, units, intensity_var
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")


def _compute_first_landfall_metric(
    forecast: xr.Dataset,
    target: xr.Dataset,
    metric_type: Literal["displacement", "timing", "intensity"],
    units: Literal["hours", "days"],
    intensity_var: str,
):
    """Compute metric using first landfall approach (classic)."""
    # Find landfall points using pure xarray
    forecast_landfall = find_landfall_xarray(forecast)
    target_landfall = find_landfall_xarray(target)

    if forecast_landfall is None or target_landfall is None:
        # Handle case where no landfall is found
        if "lead_time" in forecast.dims:
            # Calculate init_times from lead_time and valid_time
            init_times_calc = forecast.valid_time - forecast.lead_time
            unique_init_times = np.unique(init_times_calc.values)

            # Create NaN result with init_time dimension
            return xr.DataArray(
                np.full(len(unique_init_times), np.nan),
                dims=["init_time"],
                coords={"init_time": unique_init_times},
            )
        else:
            return xr.DataArray(np.nan)

    # Compute the specific metric
    if metric_type == "displacement":
        return _compute_displacement_metric(forecast_landfall, target_landfall)
    elif metric_type == "timing":
        return _compute_timing_metric(forecast_landfall, target_landfall, units)
    elif metric_type == "intensity":
        return _compute_intensity_metric(
            forecast_landfall, target_landfall, intensity_var
        )
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def _compute_next_landfall_metric(
    forecast: xr.Dataset,
    target: xr.Dataset,
    metric_type: Literal["displacement", "timing", "intensity"],
    aggregation: Literal["mean", "max", "min"],
    units: Literal["hours", "days"],
    intensity_var: str,
):
    """Compute metric using next upcoming landfall approach."""
    # Find all target landfalls
    target_landfalls = find_all_landfalls_xarray(target)
    if target_landfalls is None:
        # No target landfalls - return NaN for all init_times
        if "init_time" in forecast.coords:
            return xr.DataArray(
                np.full(len(forecast.init_time), np.nan),
                dims=["init_time"],
                coords={"init_time": forecast.init_time},
            )
        else:
            return xr.DataArray(np.nan)

    # Find next landfall for each init_time
    next_target_landfalls = find_next_landfall_for_init_time(forecast, target_landfalls)
    if next_target_landfalls is None:
        # No valid next landfalls
        if "init_time" in forecast.coords:
            return xr.DataArray(
                np.full(len(forecast.init_time), np.nan),
                dims=["init_time"],
                coords={"init_time": forecast.init_time},
            )
        else:
            return xr.DataArray(np.nan)

    # Find forecast landfalls
    forecast_landfalls = find_all_landfalls_xarray(forecast)
    if forecast_landfalls is None:
        # No forecast landfalls - return NaN
        return xr.DataArray(
            np.full(len(next_target_landfalls.init_time), np.nan),
            dims=["init_time"],
            coords={"init_time": next_target_landfalls.init_time},
        )

    # Compute metric for each init_time
    results = []
    init_times_out = []

    for i, init_time in enumerate(next_target_landfalls.init_time):
        target_landfall = next_target_landfalls.isel(init_time=i)
        target_time = target_landfall.valid_time.values

        # Find forecast landfall for this init_time
        if "init_time" in forecast_landfalls.dims:
            # Find matching init_time in forecast landfalls
            init_time_match = forecast_landfalls.init_time == init_time
            if not init_time_match.any():
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            forecast_for_init = forecast_landfalls.where(init_time_match, drop=True)

            if len(forecast_for_init.init_time) == 0:
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            # Find forecast landfall closest to target time
            time_diffs = np.abs(forecast_for_init.valid_time - target_time)
            closest_idx = time_diffs.argmin()
            closest_forecast = forecast_for_init.isel(landfall=closest_idx)
        else:
            # Single forecast landfall case
            closest_forecast = forecast_landfalls

        # Calculate the specific metric
        try:
            if metric_type == "displacement":
                result = _compute_displacement_metric(closest_forecast, target_landfall)
            elif metric_type == "timing":
                result = _compute_timing_metric(
                    closest_forecast, target_landfall, units
                )
            elif metric_type == "intensity":
                result = _compute_intensity_metric(
                    closest_forecast, target_landfall, intensity_var
                )
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")

            results.append(float(result.values))
        except Exception:
            results.append(np.nan)

        init_times_out.append(init_time.values)

    return xr.DataArray(
        results, dims=["init_time"], coords={"init_time": init_times_out}
    )


def _compute_all_landfalls_metric(
    forecast: xr.Dataset,
    target: xr.Dataset,
    metric_type: Literal["displacement", "timing", "intensity"],
    aggregation: Literal["mean", "max", "min"],
    units: Literal["hours", "days"],
    intensity_var: str,
):
    """Compute metric considering all landfalls with aggregation."""
    # For now, delegate to next approach - could expand later
    return _compute_next_landfall_metric(
        forecast, target, metric_type, aggregation, units, intensity_var
    )


def _compute_displacement_metric(
    forecast_landfall: xr.Dataset, target_landfall: xr.Dataset
):
    """Compute displacement between forecast and target landfall."""
    # Check dimensions and compute displacement
    if "init_time" in forecast_landfall.dims:
        # Vector case - compute distance for each init_time
        distances = []
        for i in range(len(forecast_landfall.init_time)):
            f_lat = forecast_landfall.latitude.isel(init_time=i).values
            f_lon = forecast_landfall.longitude.isel(init_time=i).values
            t_lat = target_landfall.latitude.values  # Target is scalar
            t_lon = target_landfall.longitude.values

            # Skip if any coordinates are NaN
            if np.isnan(f_lat) or np.isnan(f_lon) or np.isnan(t_lat) or np.isnan(t_lon):
                distances.append(np.nan)
            else:
                dist = calc.calculate_haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units="km"
                )
                # Ensure we append a scalar value
                if hasattr(dist, "item"):
                    distances.append(float(dist.item()))
                else:
                    distances.append(float(dist))

        return xr.DataArray(
            distances,
            dims=["init_time"],
            coords={"init_time": forecast_landfall.init_time},
        )
    else:
        # Scalar case
        return calculate_landfall_distance_km_xarray(forecast_landfall, target_landfall)


def _compute_timing_metric(
    forecast_landfall: xr.Dataset,
    target_landfall: xr.Dataset,
    units: Literal["hours", "days"],
):
    """Compute timing difference between forecast and target landfall."""
    # Calculate time difference in hours
    time_diff_hours = calculate_landfall_time_difference_hours_xarray(
        forecast_landfall, target_landfall
    )

    # Convert to specified units
    if units == "hours":
        return time_diff_hours
    elif units == "days":
        return time_diff_hours / 24.0


def _compute_intensity_metric(
    forecast_landfall: xr.Dataset,
    target_landfall: xr.Dataset,
    intensity_var: str,
):
    """Compute intensity difference between forecast and target landfall."""
    # Get intensity values
    if intensity_var not in forecast_landfall.data_vars:
        raise ValueError(f"Intensity variable '{intensity_var}' not found in forecast")
    if intensity_var not in target_landfall.data_vars:
        raise ValueError(f"Intensity variable '{intensity_var}' not found in target")

    forecast_intensity = forecast_landfall[intensity_var]
    target_intensity = target_landfall[intensity_var]

    # Calculate absolute error
    intensity_error = np.abs(forecast_intensity - target_intensity)

    return intensity_error


Location = namedtuple("Location", ["latitude", "longitude"])
# Global cache for TC track data to avoid recomputation across child classes
_TC_TRACK_CACHE: Dict[str, xr.Dataset] = {}

# Global registry for IBTrACS data to be used in TC filtering
_IBTRACS_DATA_REGISTRY: Dict[str, xr.Dataset] = {}


def register_ibtracs_data(case_id: str, ibtracs_data: xr.Dataset) -> None:
    """Register IBTrACS data for a specific case to be used in TC filtering.

    Args:
        case_id: Unique identifier for the case.
        ibtracs_data: IBTrACS dataset with valid_time, latitude, longitude.
    """
    global _IBTRACS_DATA_REGISTRY
    _IBTRACS_DATA_REGISTRY[case_id] = ibtracs_data


def get_ibtracs_data(case_id: str) -> Optional[xr.Dataset]:
    """Get registered IBTrACS data for a specific case.

    Args:
        case_id: Unique identifier for the case.

    Returns:
        IBTrACS dataset if available, None otherwise.
    """
    global _IBTRACS_DATA_REGISTRY
    return _IBTRACS_DATA_REGISTRY.get(case_id, None)


def clear_ibtracs_registry() -> None:
    """Clear the IBTrACS data registry."""
    global _IBTRACS_DATA_REGISTRY
    _IBTRACS_DATA_REGISTRY.clear()


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
    ds_mapping: xr.Dataset,
) -> float:
    """Find the great circle distance from a point to a contour.

    Args:
        contour: The contour to find the great circle distance to.
        point: The point to find the great circle distance to.
        ds_mapping: The xarray dataset to map the point to.

    Returns:
        The great circle distance from the point to the contour.
    """
    furthest_point, _ = find_furthest_contour_from_point(contour, point)
    gc_distance_point_latlon = calc.convert_from_cartesian_to_latlon(
        furthest_point, ds_mapping
    )
    point_latlon = calc.convert_from_cartesian_to_latlon(point, ds_mapping)
    gc_distance_contour_distance = calc.calculate_haversine_distance(
        gc_distance_point_latlon, point_latlon, units="degrees"
    )
    # Ensure we return a float
    if isinstance(gc_distance_contour_distance, xr.DataArray):
        return float(gc_distance_contour_distance.values)
    return float(gc_distance_contour_distance)


def find_valid_candidates(
    slp_contours: Sequence[Sequence[tuple[float, float]]],
    dz_contours: Sequence[Sequence[tuple[float, float]]],
    point: tuple[float, float],
    ds_mapping: xr.Dataset,
    time_counter: int,
    max_gc_distance_slp_contour: float = 5.5,
    max_gc_distance_dz_contour: float = 6.5,
    orography_filter_threshold: float = 150,
) -> Optional[Location]:
    """Find valid candidate coordinate for a TC.

    Defaults use the TempestExtremes criteria for TC track detection, where the
    slp must increase by at least n hPa within 5.5 great circle degrees from
    the center point, geopotential thickness must increase by at least m meters
    within 6.5 great circle degrees from the center point, and the terrain must
    be less than 150m.

    Args:
        slp_contours: List of SLP contours.
        dz_contours: List of DZ contours.
        point: The point to find the valid candidate for.
        ds_mapping: The xarray dataset to map the point to.
        time_counter: The time counter.
        max_gc_distance_slp_contour: The maximum great circle distance for the
            SLP contour.
        max_gc_distance_dz_contour: The maximum great circle distance for the
            DZ contour.
        orography_filter_threshold: The threshold for the orography filter.

    Returns:
        The valid candidate coordinate as a dict of timestamp and tuple of
        latitude and longitude.
    """
    latitude = calc.convert_from_cartesian_to_latlon(point, ds_mapping)[0]
    longitude = calc.convert_from_cartesian_to_latlon(point, ds_mapping)[1]
    # Check orography filter if orography data is available
    if "orography" in ds_mapping.data_vars:
        orography_filter = (
            ds_mapping["orography"]
            .sel(latitude=latitude, longitude=longitude, method="nearest")
            .min()
            .values
            < orography_filter_threshold
            if time_counter < 8
            else True
        )
    else:
        # If no orography data, skip the orography filter
        orography_filter = True
    latitude_filter = abs(latitude) < 50 if time_counter < 10 else True
    for slp_contour, dz_contour in product(slp_contours, dz_contours):
        if (
            # Only check closed contours
            all(np.isclose(slp_contour[-1], slp_contour[0]))
            and all(np.isclose(dz_contour[-1], dz_contour[0]))
            and
            # Check if point is inside both contour types
            measure.points_in_poly([[point[0], point[1]]], slp_contour)[0]
            and measure.points_in_poly([[point[0], point[1]]], dz_contour)[0]
            and
            # Check if the contour is within the max great circle distance
            find_valid_contour_from_point(slp_contour, point, ds_mapping)
            < max_gc_distance_slp_contour
            and find_valid_contour_from_point(dz_contour, point, ds_mapping)
            < max_gc_distance_dz_contour
            and orography_filter
            and latitude_filter
        ):
            valid_candidate = Location(latitude=latitude, longitude=longitude)
            return valid_candidate
    return None


def _process_all_init_times_vectorized(
    cyclone_dataset: xr.Dataset,
    time_coord_name: str,
    ibtracs_df: pd.DataFrame,
    max_temporal_hours: float,
    max_spatial_distance_degrees: float,
    min_distance: int,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    use_contour_validation: bool = True,
    min_track_length: int = 10,
) -> xr.Dataset:
    """Process all init_times using a single vectorized apply_ufunc call."""

    slp = cyclone_dataset["air_pressure_at_mean_sea_level"]
    time_coord = slp[time_coord_name]
    init_time_coord = slp.init_time

    # Determine the correct input core dims based on actual data dimensions
    spatial_dims = ["latitude", "longitude"]
    non_spatial_dims = [dim for dim in slp.dims if dim not in spatial_dims]
    ibtracs_df = ibtracs_df[
        pd.to_datetime(ibtracs_df["valid_time"]).isin(pd.to_datetime(time_coord.values))
    ]
    # Get wind speed data
    wind_speed = cyclone_dataset["surface_wind_speed"]

    # Get geopotential thickness data if available
    dz = cyclone_dataset.get("geopotential_thickness", None)

    # OPTIMIZED: Single apply_ufunc call that returns compact detection results
    input_vars = [
        slp,
        time_coord,
        init_time_coord,
        slp.latitude,
        slp.longitude,
        wind_speed,
    ]
    input_core_dims: list[list[str]] = [
        [str(dim) for dim in non_spatial_dims + spatial_dims],  # lead_time, valid_time
        [time_coord_name],  # e.g., ['valid_time']
        [str(dim) for dim in init_time_coord.dims],  # Use actual init_time dims
        ["latitude"],  # latitude coordinates
        ["longitude"],  # longitude coordinates
        [str(dim) for dim in non_spatial_dims + spatial_dims],  # wind speed dims
    ]

    # Always pass dz_array (either the data or None)
    if dz is not None:
        input_vars.append(dz)
        input_core_dims.append([str(dim) for dim in non_spatial_dims + spatial_dims])
    else:
        # Add None as a placeholder for dz_array
        input_vars.append(None)
        input_core_dims.append([])  # type: ignore

    detection_data = xr.apply_ufunc(
        _process_entire_dataset_compact,
        *input_vars,
        kwargs={
            "ibtracs_df": ibtracs_df,
            "max_spatial_distance_degrees": max_spatial_distance_degrees,
            "min_distance": min_distance,
            "max_temporal_hours": max_temporal_hours,
            "slp_contour_magnitude": slp_contour_magnitude,
            "dz_contour_magnitude": dz_contour_magnitude,
            "use_contour_validation": use_contour_validation and dz is not None,
            "min_track_length": min_track_length,
        },
        input_core_dims=input_core_dims,
        output_core_dims=[
            ["detection"],  # All detections as a single dimension
            ["detection"],  # Lead time indices for each detection
            ["detection"],  # Valid time indices for each detection
            ["detection"],  # Track IDs for each detection
            ["detection"],  # Latitudes for each detection
            ["detection"],  # Longitudes for each detection
            ["detection"],  # SLP values for each detection
            ["detection"],  # Wind speeds for each detection
        ],
        dask="allowed",
        output_dtypes=[
            int,  # n_detections
            int,  # lead_time_idx
            int,  # valid_time_idx
            int,  # track_id
            float,  # latitude
            float,  # longitude
            float,  # slp
            float,  # wind
        ],
    )

    # Unpack the results from apply_ufunc
    n_detections, lt_indices, vt_indices, track_ids, lats, lons, slp_vals, wind_vals = (
        detection_data
    )

    # Convert to a proper xarray Dataset with the requested structure
    return _convert_detections_to_dataset(
        n_detections,
        lt_indices,
        vt_indices,
        track_ids,
        lats,
        lons,
        slp_vals,
        wind_vals,
        cyclone_dataset,  # Pass the full dataset to access all coordinates
        time_coord_name,
        [str(dim) for dim in non_spatial_dims],  # Convert Hashable to str
    )


def _process_entire_dataset_compact(
    slp_array: np.ndarray,
    time_array: np.ndarray,
    init_time_array: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    wind_array: np.ndarray,
    dz_array: Optional[np.ndarray],  # Can be None when no DZ data available
    ibtracs_df: pd.DataFrame,
    max_spatial_distance_degrees: float,
    min_distance: int,
    max_temporal_hours: float,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    use_contour_validation: bool = True,
    min_track_length: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Process dataset and return compact detection arrays.

    Args:
        slp_array: Sea level pressure array (lead_time, valid_time, lat, lon)
        time_array: Valid time array
        init_time_array: Initialization time array (lead_time, valid_time)
        lat_array: Latitude coordinate array
        lon_array: Longitude coordinate array
        wind_array: Wind speed array (lead_time, valid_time, lat, lon)
        dz_array: Geopotential thickness array (lead_time, valid_time, lat, lon)
        ibtracs_df: IBTrACS dataframe for filtering
        max_spatial_distance_degrees: Max spatial distance for IBTrACS filtering
        min_distance: Minimum distance between detected peaks
        max_temporal_hours: Maximum temporal window from init_time
        slp_contour_magnitude: SLP contour threshold for validation
        dz_contour_magnitude: DZ contour threshold for validation
        use_contour_validation: Whether to use contour validation
        min_track_length: Minimum consecutive lead times for valid tracks
    """

    # Convert to numpy arrays if they're Dask arrays
    if hasattr(slp_array, "compute"):
        slp_array = slp_array.compute()
    if hasattr(time_array, "compute"):
        time_array = time_array.compute()
    if hasattr(init_time_array, "compute"):
        init_time_array = init_time_array.compute()
    if hasattr(wind_array, "compute"):
        wind_array = wind_array.compute()
    if dz_array is not None and hasattr(dz_array, "compute"):
        dz_array = dz_array.compute()

    # Ensure they're numpy arrays
    slp_array = np.asarray(slp_array)
    valid_time_array = np.asarray(time_array)
    init_time_array = np.asarray(init_time_array)
    wind_array = np.asarray(wind_array)
    if dz_array is not None:
        dz_array = np.asarray(dz_array)

    # Initialize arrays for (lead_time, valid_time, lat, lon)
    if slp_array.ndim == 4:  # (lead_time, valid_time, lat, lon)
        n_lead_times, n_valid_times = slp_array.shape[:2]
    else:
        raise ValueError(f"Expected 4D array, got {slp_array.ndim}D")

    # Collect all detections
    detections = []

    # Get unique init_times - handle potential datetime64 conversion issues
    unique_init_times = np.unique(init_time_array.flatten())

    # Global track ID counter to ensure unique IDs across all init_times
    global_next_track_id = 0

    # Process each unique init_time with proper storm tracking
    for current_init_time in unique_init_times:
        # Create mask for this init_time - handle datetime precision
        if init_time_array.dtype.kind == "M":  # datetime64 type
            # For datetime arrays, use np.isclose for timestamp comparison
            init_time_mask = np.abs(
                init_time_array - current_init_time
            ) <= np.timedelta64(1, "s")
        else:
            init_time_mask = init_time_array == current_init_time

        # Reset tracking for each init_time (but keep global track ID counter)
        active_tracks: dict[
            int, dict[str, float]
        ] = {}  # {track_id: last_known_position}

        # Get time indices for this init_time and sort them
        time_indices = []
        for lt_idx in range(n_lead_times):
            for vt_idx in range(n_valid_times):
                if init_time_mask[lt_idx, vt_idx]:
                    # Apply temporal filtering within max_temporal_hours
                    current_valid_time = valid_time_array[vt_idx]
                    if hasattr(current_valid_time, "item"):
                        current_valid_time = current_valid_time.item()

                    # Convert to pandas Timestamp for consistent datetime operations
                    current_valid_time = pd.Timestamp(current_valid_time)
                    current_init_time = pd.Timestamp(current_init_time)

                    # Calculate time difference in hours
                    time_diff = current_valid_time - current_init_time
                    time_diff_hours = time_diff.total_seconds() / 3600.0

                    # Only include if within max_temporal_hours
                    if time_diff_hours <= max_temporal_hours:
                        time_indices.append((lt_idx, vt_idx, current_valid_time))

        # Sort by valid_time to ensure temporal continuity
        time_indices.sort(key=lambda x: x[2])

        # Process time steps in chronological order
        for lt_idx, vt_idx, current_valid_time in time_indices:
            slp_slice = slp_array[lt_idx, vt_idx, :, :]
            wind_slice = wind_array[lt_idx, vt_idx, :, :]
            dz_slice = dz_array[lt_idx, vt_idx, :, :] if dz_array is not None else None
            current_valid_time = pd.Timestamp(current_valid_time)

            # Apply IBTrACS filtering with optional contour validation
            peaks = _find_peaks_for_time_slice(
                slp_slice,
                current_valid_time,
                ibtracs_df,
                0.0,  # Exact time match only
                min_distance,
                lat_array,
                lon_array,
                max_spatial_distance_degrees,
                dz_slice if dz_slice is not None else np.array([]),
                slp_contour_magnitude,
                dz_contour_magnitude,
                use_contour_validation,
            )

            if len(peaks) > 0:
                # Get coordinates and values for all peaks
                peak_lats = lat_array[peaks[:, 0]]
                peak_lons = lon_array[peaks[:, 1]]
                peak_slps = slp_slice[peaks[:, 0], peaks[:, 1]]
                peak_winds = wind_slice[peaks[:, 0], peaks[:, 1]]

                # Process each detected peak as a potential storm
                unassigned_peaks = list(range(len(peaks)))

                # First, try to match peaks to existing active tracks (within 8°)
                for track_id, last_pos in list(active_tracks.items()):
                    if not unassigned_peaks:
                        break

                    # Find closest unassigned peak to this track
                    best_idx = None
                    best_distance = float("inf")

                    for peak_idx in unassigned_peaks:
                        distance = _calculate_great_circle_distance(
                            peak_lats[peak_idx],
                            peak_lons[peak_idx],
                            last_pos["lat"],
                            last_pos["lon"],
                        )
                        if distance <= 8.0 and distance < best_distance:
                            best_distance = distance
                            best_idx = peak_idx

                    if best_idx is not None:
                        # Record this detection
                        detections.append(
                            {
                                "lt_idx": lt_idx,
                                "vt_idx": vt_idx,
                                "track_id": track_id,
                                "lat": peak_lats[best_idx],
                                "lon": peak_lons[best_idx],
                                "slp": peak_slps[best_idx],
                                "wind": peak_winds[best_idx],
                            }
                        )

                        # Update active track position
                        active_tracks[track_id] = {
                            "lat": peak_lats[best_idx],
                            "lon": peak_lons[best_idx],
                        }
                        unassigned_peaks.remove(best_idx)

                # Create new tracks for remaining unassigned peaks
                for peak_idx in unassigned_peaks:
                    track_id = global_next_track_id
                    global_next_track_id += 1

                    # Record this detection
                    detections.append(
                        {
                            "lt_idx": lt_idx,
                            "vt_idx": vt_idx,
                            "track_id": track_id,
                            "lat": peak_lats[peak_idx],
                            "lon": peak_lons[peak_idx],
                            "slp": peak_slps[peak_idx],
                            "wind": peak_winds[peak_idx],
                        }
                    )

                    active_tracks[track_id] = {
                        "lat": peak_lats[peak_idx],
                        "lon": peak_lons[peak_idx],
                    }

    # Filter out tracks with fewer than min_track_length consecutive lead times
    if detections:
        # Group detections by (track_id, init_time) and collect lead time indices
        # This ensures tracks from different forecasts don't get mixed
        track_lead_times: dict[tuple[int, pd.Timestamp], set[int]] = {}
        for detection in detections:
            track_id = detection["track_id"]
            lt_idx = detection["lt_idx"]
            vt_idx = detection["vt_idx"]

            # Get init_time for this detection
            detection_init_time = init_time_array[lt_idx, vt_idx]
            # Convert to Python datetime for consistent hashing
            if hasattr(detection_init_time, "item"):
                detection_init_time = detection_init_time.item()
            track_key = (track_id, detection_init_time)

            if track_key not in track_lead_times:
                track_lead_times[track_key] = set()
            # Ensure lt_idx is converted to int for the set
            lt_idx_int: int
            if hasattr(lt_idx, "item"):
                lt_idx_int = int(lt_idx.item())
            else:
                lt_idx_int = int(lt_idx)
            track_lead_times[track_key].add(lt_idx_int)

        # Check for consecutive lead times for each track
        valid_track_keys: set[tuple[int, pd.Timestamp]] = set()
        for track_key, lt_indices in track_lead_times.items():
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

            # Only keep tracks with at least min_track_length consecutive lead times
            if max_consecutive >= min_track_length:
                valid_track_keys.add(track_key)

        # Filter detections to only include valid tracks
        filtered_detections = []
        for d in detections:
            track_id = d["track_id"]
            lt_idx = d["lt_idx"]
            vt_idx = d["vt_idx"]
            detection_init_time = init_time_array[lt_idx, vt_idx]
            # Convert to Python datetime for consistent hashing
            if hasattr(detection_init_time, "item"):
                detection_init_time = detection_init_time.item()
            track_key = (track_id, detection_init_time)

            if track_key in valid_track_keys:
                filtered_detections.append(d)
    else:
        filtered_detections = []

    # Convert filtered data to arrays
    n_detections = len(filtered_detections)
    if n_detections == 0:
        # Return empty arrays
        empty_array = np.array([])
        return (
            np.array([0]),  # n_detections
            empty_array.astype(int),  # lt_indices
            empty_array.astype(int),  # vt_indices
            empty_array.astype(int),  # track_ids
            empty_array.astype(float),  # lats
            empty_array.astype(float),  # lons
            empty_array.astype(float),  # slp_vals
            empty_array.astype(float),  # wind_vals
        )

    # Pack filtered detection data into arrays
    lt_indices = np.array([d["lt_idx"] for d in filtered_detections])
    vt_indices = np.array([d["vt_idx"] for d in filtered_detections])
    track_ids = np.array([d["track_id"] for d in filtered_detections])
    lats = np.array([d["lat"] for d in filtered_detections])
    lons = np.array([d["lon"] for d in filtered_detections])
    slp_vals = np.array([d["slp"] for d in filtered_detections])
    wind_vals = np.array([d["wind"] for d in filtered_detections])

    return (
        np.array([n_detections]),
        lt_indices,
        vt_indices,
        track_ids,
        lats,
        lons,
        slp_vals,
        wind_vals,
    )


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


def _convert_detections_to_dataset(
    n_detections: np.ndarray,
    lt_indices: np.ndarray,
    vt_indices: np.ndarray,
    track_ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    slp_vals: np.ndarray,
    wind_vals: np.ndarray,
    original_dataset: xr.Dataset,
    time_coord_name: str,
    non_spatial_dims: list[str],
) -> xr.Dataset:
    """Convert detection arrays to compact xarray Dataset."""

    # Handle case with no detections
    n_dets_val = _safe_extract_value(n_detections[0])
    if n_dets_val == 0:
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
    for i in range(int(n_dets_val)):
        # Convert DataArrays to regular integers for use as dictionary keys
        lt_idx = _safe_extract_value(lt_indices[i])
        vt_idx = _safe_extract_value(vt_indices[i])
        key = (int(lt_idx), int(vt_idx))
        if key not in unique_combinations:
            unique_combinations[key] = []
        unique_combinations[key].append(i)

    # Determine output dimensions
    if unique_combinations:
        max_lt = max(key[0] for key in unique_combinations.keys())
        max_vt = max(key[1] for key in unique_combinations.keys())

        # Get the actual lead_time values from the original dataset
        lead_time_coord_name = non_spatial_dims[0] if non_spatial_dims else "lead_time"
        if lead_time_coord_name in original_dataset.coords:
            lead_times = original_dataset.coords[lead_time_coord_name].values[
                : max_lt + 1
            ]
        else:
            # Fallback to integer indices if no lead_time coordinate found
            lead_times = np.arange(max_lt + 1)

        # Get valid_time values
        valid_times = original_dataset.coords[time_coord_name].values[: max_vt + 1]
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
            # Convert DataArrays to values for array indexing
            track_id_val = _safe_extract_value(track_ids[det_idx])
            slp_val = _safe_extract_value(slp_vals[det_idx])
            wind_val = _safe_extract_value(wind_vals[det_idx])
            lat_val = _safe_extract_value(lats[det_idx])
            lon_val = _safe_extract_value(lons[det_idx])

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


def _find_peaks_for_time_slice(
    slp_slice: np.ndarray,
    current_valid_time: pd.Timestamp,
    ibtracs_df: pd.DataFrame,
    max_temporal_hours: float,
    min_distance: int,
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    max_spatial_distance_degrees: float = 5.0,
    dz_slice: Optional[np.ndarray] = None,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    use_contour_validation: bool = True,
) -> np.ndarray:
    """Find peaks for a single time slice with IBTrACS filtering."""

    # Filter IBTrACS data temporally
    time_diff = np.abs(
        (ibtracs_df["valid_time"] - current_valid_time).dt.total_seconds() / 3600
    )
    temporal_mask = time_diff <= max_temporal_hours
    nearby_ibtracs = ibtracs_df[temporal_mask]

    if len(nearby_ibtracs) == 0:
        return np.array([])

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

    spatial_mask = _create_spatial_mask_vectorized(
        lat_coords, lon_coords, nearby_ibtracs, max_spatial_distance_degrees
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

        # Create temporary datasets for contour analysis
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

        # Create a simple dataset for mapping
        ds_mapping = xr.Dataset({"latitude": lat_da, "longitude": lon_da})

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
                ds_mapping,
                0,  # time_counter=0
                max_gc_distance_slp_contour=5.5,
                max_gc_distance_dz_contour=6.5,
            )

            if candidate is not None:
                validated_peaks.append(peak)

        return (
            np.array(validated_peaks) if validated_peaks else np.array([]).reshape(0, 2)
        )

    return peaks


def _create_spatial_mask_vectorized(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    nearby_ibtracs: pd.DataFrame,
    max_distance_degrees: float,
):
    """Create spatial mask using vectorized distance calculation."""

    # Create meshgrid of all lat/lon coordinates
    lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing="ij")

    # Initialize mask
    spatial_mask = np.zeros_like(lat_grid, dtype=bool)

    # For each IBTrACS point, compute distances to all grid points vectorized
    for _, ibtracs_row in nearby_ibtracs.iterrows():
        ibtracs_lat = ibtracs_row["latitude"]
        ibtracs_lon = ibtracs_row["longitude"]

        # Vectorized distance calculation
        distances = _calculate_great_circle_distance_vectorized(
            lat_grid, lon_grid, ibtracs_lat, ibtracs_lon
        )

        # Update mask where distance is within threshold
        spatial_mask |= distances <= max_distance_degrees

    return spatial_mask


def _calculate_great_circle_distance_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
):
    """Vectorized great circle distance calculation."""
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula - vectorized
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    # Convert back to degrees
    distance_degrees = np.degrees(c)

    return distance_degrees


def _calculate_great_circle_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in degrees using haversine formula."""
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Convert back to degrees (Earth radius = 6371 km, 1 degree ≈ 111 km)
    distance_degrees = math.degrees(c)

    return distance_degrees


def _create_tctracks_optimized_with_ibtracs(
    cyclone_dataset: xr.Dataset,
    ibtracs_df: pd.DataFrame,
    slp_contour_magnitude: float,
    dz_contour_magnitude: float,
    min_distance: int,
    max_spatial_distance_degrees: float,
    max_temporal_hours: float,
    use_contour_validation: bool = True,
    min_track_length: int = 10,
) -> xr.Dataset:
    """Optimized TC track creation using apply_ufunc and IBTrACS filtering.

    This version combines peak detection with optional contour validation.
    If geopotential_thickness data is available and use_contour_validation=True,
    detected peaks are validated using TempestExtremes-style SLP and DZ contour
    criteria for improved track quality.
    """
    # Handle datasets with init_time coordinate
    if "init_time" in cyclone_dataset.coords:
        # Vectorized approach: process all init_times at once using groupby
        # Group by init_time and process each group simultaneously
        time_coord_name = (
            "valid_time" if "valid_time" in cyclone_dataset.dims else "time"
        )

        # Create a combined dataset for vectorized processing
        storm_ds = _process_all_init_times_vectorized(
            cyclone_dataset,
            time_coord_name,
            ibtracs_df,
            max_temporal_hours,
            max_spatial_distance_degrees,
            min_distance,
            slp_contour_magnitude,
            dz_contour_magnitude,
            use_contour_validation,
            min_track_length,
        )
        return storm_ds
    else:
        # Handle datasets without init_time coordinate
        # Use standard coordinate names
        time_dim = "valid_time"
        time_coord = cyclone_dataset.valid_time

        lead_time_dim = "lead_time"
        lead_time_coord = cyclone_dataset.lead_time

        # Create empty dataset as fallback for now
        return xr.Dataset(
            {
                "tc_slp": (
                    [time_dim, lead_time_dim],
                    np.full((len(time_coord), len(lead_time_coord)), np.nan),
                ),
                "tc_latitude": (
                    [time_dim, lead_time_dim],
                    np.full((len(time_coord), len(lead_time_coord)), np.nan),
                ),
                "tc_longitude": (
                    [time_dim, lead_time_dim],
                    np.full((len(time_coord), len(lead_time_coord)), np.nan),
                ),
                "tc_vmax": (
                    [time_dim, lead_time_dim],
                    np.full((len(time_coord), len(lead_time_coord)), np.nan),
                ),
            },
            coords={
                time_dim: time_coord,
                lead_time_dim: lead_time_coord,
            },
        )


def create_tctracks_from_dataset_with_ibtracs_filter(
    cyclone_dataset: xr.Dataset,
    ibtracs_data: xr.Dataset,
    slp_contour_magnitude: float = 200,
    dz_contour_magnitude: float = -6,
    min_distance: int = 5,
    max_spatial_distance_degrees: float = 5.0,
    max_temporal_hours: float = 120,
    use_contour_validation: bool = True,
    min_track_length: int = 10,
    exclude_post_landfall_init_times: bool = False,
) -> xr.Dataset:
    """Create storm tracks from a cyclone dataset with IBTrACS proximity filtering.

    For forecast data, this function only considers TC candidates that are within
    5 great circle degrees of IBTrACS points and within 120 hours of the
    valid_time. Optionally uses xr.apply_ufunc for improved performance.

    Args:
        cyclone_dataset: The cyclone dataset.
        ibtracs_data: IBTrACS dataset with valid_time, latitude, longitude.
        slp_contour_magnitude: The SLP contour magnitude.
        dz_contour_magnitude: The DZ contour magnitude.
        min_distance: The minimum distance between TCs.
        max_spatial_distance_degrees: Maximum great circle distance from
            IBTrACS points.
        max_temporal_hours: Maximum time difference from IBTrACS valid times.
        use_contour_validation: Whether to use contour validation.
        min_track_length: Minimum number of points required for a track to be
            included in results (default: 10).
        exclude_post_landfall_init_times: Whether to exclude init_times that occur
            after all IBTrACS landfall events (default: True).

    Returns:
        An xarray Dataset with detected storm tracks.
    """
    # Filter out init_times that occur after all IBTrACS landfalls
    if exclude_post_landfall_init_times:
        filtered_dataset = _filter_post_landfall_init_times(
            cyclone_dataset, ibtracs_data
        )

        # If no valid init_times remain, return empty dataset
        if filtered_dataset is None:
            return _create_empty_tracks_dataset()

        # Check if dataset has init_time after filtering
        if (
            "init_time" in filtered_dataset.coords
            and len(filtered_dataset.init_time) == 0
        ):
            return _create_empty_tracks_dataset()

        cyclone_dataset = filtered_dataset

    # Convert IBTrACS to pandas for easier temporal filtering
    ibtracs_df = ibtracs_data.to_dataframe().reset_index()
    ibtracs_df["valid_time"] = pd.to_datetime(ibtracs_df["valid_time"])
    logger.info("Generating TC tracks")
    return _create_tctracks_optimized_with_ibtracs(
        cyclone_dataset,
        ibtracs_df,
        slp_contour_magnitude,
        dz_contour_magnitude,
        min_distance,
        max_spatial_distance_degrees,
        max_temporal_hours,
        use_contour_validation,
        min_track_length,
    )


def _filter_post_landfall_init_times(
    cyclone_dataset: xr.Dataset, ibtracs_data: xr.Dataset
) -> Optional[xr.Dataset]:
    """
    Filter out init_times that occur after all IBTrACS landfall events.

    Args:
        cyclone_dataset: Forecast dataset with init_time dimension
        ibtracs_data: IBTrACS dataset

    Returns:
        Filtered dataset with only valid init_times, or None if no valid init_times
    """
    # Find all IBTrACS landfalls
    ibtracs_landfalls = find_all_landfalls_xarray(ibtracs_data)

    if ibtracs_landfalls is None:
        # No IBTrACS landfalls detected, keep all init_times
        return cyclone_dataset

    # Get all IBTrACS landfall times
    landfall_times = ibtracs_landfalls.valid_time.values
    latest_landfall = np.max(landfall_times)

    # Convert cyclone_dataset to have init_time coordinate if it doesn't already
    if "init_time" not in cyclone_dataset.coords:
        cyclone_dataset = convert_valid_time_to_init_time(cyclone_dataset)

    # Filter to only include init_times before the latest landfall
    valid_init_times = cyclone_dataset.init_time < latest_landfall

    if not valid_init_times.any():
        # No valid init_times
        return None

    # Return filtered dataset
    return cyclone_dataset.where(valid_init_times, drop=True)


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


def find_landfall_xarray(track_dataset: xr.Dataset) -> Optional[xr.Dataset]:
    """Finds the first point where a tropical cyclone track intersects with land
    by linearly interpolating between track points using pure xarray operations.

    Based on the original find_landfall function, handles two cases:
    1. IBTrACS data: single track with valid_time dimension
    2. Forecast data: tracks with lead_time, valid_time dimensions processed by
    init_time

    Args:
        track_dataset: xarray Dataset containing track data with variables:
                      - latitude, longitude, surface_wind_speed,
                        air_pressure_at_mean_sea_level
                      Case 1: (valid_time,) for IBTrACS
                      Case 2: (lead_time, valid_time) for forecasts

    Returns:
        xarray Dataset with landfall point data, or None if no landfall found.
        For forecast data, includes init_time dimension.
    """
    # Pre-load coastline data once
    land = natural_earth(category="physical", name="land", resolution="10m")
    land_geom = list(Reader(land).geometries())[0]

    # Case 1: IBTrACS data with valid_time dimension
    if "valid_time" in track_dataset.dims and "lead_time" not in track_dataset.dims:
        return _find_landfall_ibtracs(track_dataset, land_geom)

    # Case 2: Forecast data with lead_time and valid_time dimensions
    elif "lead_time" in track_dataset.dims and "valid_time" in track_dataset.dims:
        return _find_landfall_forecast(track_dataset, land_geom)

    else:
        raise ValueError(
            f"Unsupported track dataset structure. Expected either "
            f"(valid_time,) for IBTrACS or (lead_time, valid_time) for forecasts. "
            f"Got dimensions: {list(track_dataset.dims)}"
        )


def find_next_landfall_for_init_time(
    forecast_dataset: xr.Dataset, target_landfalls: xr.Dataset
) -> Optional[xr.Dataset]:
    """For each init_time in the forecast, find the next upcoming landfall from target
    data.

    Args:
        forecast_dataset: Forecast dataset with init_time dimension
        target_landfalls: Target landfall dataset (from find_all_landfalls_xarray)

    Returns:
        Dataset with next landfall for each init_time, or None if no target landfalls
    """
    if target_landfalls is None:
        return None

    # Convert forecast to have init_time if needed
    if "init_time" not in forecast_dataset.coords:
        forecast_dataset = convert_valid_time_to_init_time(forecast_dataset)

    # Get all target landfall times
    target_times = target_landfalls.valid_time.values

    next_landfalls = []
    init_times_out = []

    for init_time in forecast_dataset.init_time:
        init_time_val = init_time.values

        # Find target landfalls that occur after this init_time
        future_landfall_mask = target_times > init_time_val

        if not future_landfall_mask.any():
            # No future landfalls for this init_time - skip it
            continue

        # Get the earliest future landfall (next landfall)
        future_landfall_indices = np.where(future_landfall_mask)[0]
        next_landfall_idx = future_landfall_indices[
            0
        ]  # First (earliest) future landfall

        # Extract the next landfall data
        next_landfall = target_landfalls.isel(landfall=next_landfall_idx)
        next_landfalls.append(next_landfall)
        init_times_out.append(init_time_val)

    if not next_landfalls:
        return None

    # Combine into a dataset with init_time dimension
    combined_data = {}
    for var in next_landfalls[0].data_vars:
        values = [landfall[var].values for landfall in next_landfalls]
        combined_data[var] = (["init_time"], values)

    # Handle coordinates
    coords = {"init_time": init_times_out}
    if "valid_time" in next_landfalls[0].coords:
        valid_times = [landfall.valid_time.values for landfall in next_landfalls]
        combined_data["valid_time"] = (["init_time"], valid_times)

    return xr.Dataset(combined_data, coords=coords)


def find_all_landfalls_xarray(track_dataset: xr.Dataset) -> Optional[xr.Dataset]:
    """Finds ALL points where a tropical cyclone track intersects with land
    by linearly interpolating between track points using pure xarray operations.

    Args:
        track_dataset: xarray Dataset containing track data with variables:
                      - latitude, longitude, surface_wind_speed,
                        air_pressure_at_mean_sea_level
                      Case 1: (valid_time,) for IBTrACS
                      Case 2: (lead_time, valid_time) for forecasts

    Returns:
        xarray Dataset with ALL landfall point data, or None if no landfall found.
        For forecast data, includes init_time dimension.
        Dataset has a "landfall" dimension for multiple landfall points.
    """
    # Pre-load coastline data with buffer for better detection

    # Use 110m resolution (faster) but add buffer for better coastal detection
    land = natural_earth(category="physical", name="land", resolution="110m")
    land_geoms = list(Reader(land).geometries())
    land_geom = unary_union(land_geoms)

    # Add a small buffer (0.1 degrees ~11km) to catch near-coastal points
    land_geom = land_geom.buffer(0.1)

    # Case 1: IBTrACS data with valid_time dimension
    if "valid_time" in track_dataset.dims and "lead_time" not in track_dataset.dims:
        return _find_all_landfalls_ibtracs(track_dataset, land_geom)

    # Case 2: Forecast data with lead_time and valid_time dimensions
    elif "lead_time" in track_dataset.dims and "valid_time" in track_dataset.dims:
        return _find_all_landfalls_forecast(track_dataset, land_geom)

    else:
        raise ValueError(
            f"Unsupported track dataset structure. Expected either "
            f"(valid_time,) for IBTrACS or (lead_time, valid_time) for forecasts. "
            f"Got dimensions: {list(track_dataset.dims)}"
        )


def _find_landfall_ibtracs(
    track_dataset: xr.Dataset, land_geom
) -> Optional[xr.Dataset]:
    """Find landfall for IBTrACS data (single track with valid_time dimension).
    Based on the original find_landfall function.
    """
    # Filter out NaN values and sort by time
    valid_mask = (
        ~np.isnan(track_dataset["latitude"])
        & ~np.isnan(track_dataset["longitude"])
        & ~np.isnan(track_dataset["surface_wind_speed"])
        & ~np.isnan(track_dataset["air_pressure_at_mean_sea_level"])
    )
    track_data = track_dataset.where(valid_mask, drop=True).sortby("valid_time")

    # Extract data
    lats = track_data["latitude"].values
    lons = track_data["longitude"].values
    valid_times = track_data["valid_time"].values
    vmax = track_data["surface_wind_speed"].values
    slp = track_data["air_pressure_at_mean_sea_level"].values

    if len(lats) < 2:
        return None

    # Get coastlines from Natural Earth
    land = natural_earth(category="physical", name="land", resolution="10m")
    land_geom = list(Reader(land).geometries())[0]

    # Convert to -180 to 180 degree longitude (for now) - from original code
    lons_180 = (lons + 180) % 360 - 180

    # Check each track segment - from original code
    landfall_data = []

    for i in range(len(valid_times) - 1):
        try:
            # Create line segment between consecutive points
            segment = LineString(
                [(lons_180[i], lats[i]), (lons_180[i + 1], lats[i + 1])]
            )

            # Check if this is a true landfall (ocean to land movement)
            if _is_true_landfall(
                lons_180[i], lats[i], lons_180[i + 1], lats[i + 1], land_geom
            ):
                intersection = segment.intersection(land_geom)
                landfall_lon, landfall_lat = intersection.coords[0]

                # Linearly interpolate time based on distance along segment - from
                # original code
                full_dist = segment.length
                landfall_dist = LineString(
                    [(lons_180[i], lats[i]), (landfall_lon, landfall_lat)]
                ).length
                frac = landfall_dist / full_dist

                landfall_time = valid_times[i] + frac * (
                    valid_times[i + 1] - valid_times[i]
                )
                landfall_vmax = vmax[i] + frac * (vmax[i + 1] - vmax[i])
                landfall_slp = slp[i] + frac * (slp[i + 1] - slp[i])

                # Store landfall data
                landfall_data.append(
                    {
                        "latitude": landfall_lat,
                        "longitude": utils.convert_longitude_to_360(landfall_lon),
                        "surface_wind_speed": landfall_vmax,
                        "air_pressure_at_mean_sea_level": landfall_slp,
                        "valid_time": landfall_time,
                    }
                )
        except Exception as e:
            print(f"Error finding landfall: {e}")
            # Skip this segment if any error occurs
            continue

    # Convert list of dictionaries to proper xarray Dataset
    if not landfall_data:
        return None

    # Always return first landfall point as scalar for metrics compatibility
    data_point = landfall_data[0]
    return xr.Dataset(
        {
            "latitude": ([], data_point["latitude"]),
            "longitude": ([], data_point["longitude"]),
            "surface_wind_speed": ([], data_point["surface_wind_speed"]),
            "air_pressure_at_mean_sea_level": (
                [],
                data_point["air_pressure_at_mean_sea_level"],
            ),
        },
        coords={"valid_time": data_point["valid_time"]},
    )


def _find_all_landfalls_ibtracs(
    track_dataset: xr.Dataset, land_geom: Polygon
) -> Optional[xr.Dataset]:
    """Find ALL landfall points for IBTrACS data (single track with valid_time
    dimension)."""
    # Filter out NaN values and sort by time
    valid_mask = (
        ~np.isnan(track_dataset["latitude"])
        & ~np.isnan(track_dataset["longitude"])
        & ~np.isnan(track_dataset["surface_wind_speed"])
        & ~np.isnan(track_dataset["air_pressure_at_mean_sea_level"])
    )
    track_data = track_dataset.where(valid_mask, drop=True).sortby("valid_time")

    # Extract data
    lats = track_data["latitude"].values
    lons = track_data["longitude"].values
    valid_times = track_data["valid_time"].values
    vmax = track_data["surface_wind_speed"].values
    slp = track_data["air_pressure_at_mean_sea_level"].values

    if len(lats) < 2:
        return None

    # Convert to -180 to 180 degree longitude
    lons_180 = (lons + 180) % 360 - 180

    # Check each track segment and collect ALL landfall points
    landfall_data = []

    for i in range(len(valid_times) - 1):
        try:
            # Create line segment between consecutive points
            segment = LineString(
                [(lons_180[i], lats[i]), (lons_180[i + 1], lats[i + 1])]
            )

            # Check if this is a true landfall (ocean to land movement)
            if _is_true_landfall(
                lons_180[i], lats[i], lons_180[i + 1], lats[i + 1], land_geom
            ):
                intersection = segment.intersection(land_geom)
                landfall_lon, landfall_lat = intersection.coords[0]

                # Linearly interpolate time based on distance along segment
                full_dist = segment.length
                landfall_dist = LineString(
                    [(lons_180[i], lats[i]), (landfall_lon, landfall_lat)]
                ).length
                frac = landfall_dist / full_dist

                landfall_time = valid_times[i] + frac * (
                    valid_times[i + 1] - valid_times[i]
                )
                landfall_vmax = vmax[i] + frac * (vmax[i + 1] - vmax[i])
                landfall_slp = slp[i] + frac * (slp[i + 1] - slp[i])

                # Store landfall data
                landfall_data.append(
                    {
                        "latitude": landfall_lat,
                        "longitude": utils.convert_longitude_to_360(landfall_lon),
                        "surface_wind_speed": landfall_vmax,
                        "air_pressure_at_mean_sea_level": landfall_slp,
                        "valid_time": landfall_time,
                    }
                )
        except Exception as e:
            print(f"Error finding landfall: {e}")
            # Skip this segment if any error occurs
            continue

    # Convert list of dictionaries to proper xarray Dataset
    if not landfall_data:
        return None

    # Return ALL landfall points with landfall dimension
    n_landfalls = len(landfall_data)

    latitudes = np.array([d["latitude"] for d in landfall_data])
    longitudes = np.array([d["longitude"] for d in landfall_data])
    wind_speeds = np.array([d["surface_wind_speed"] for d in landfall_data])
    pressures = np.array([d["air_pressure_at_mean_sea_level"] for d in landfall_data])
    times = np.array([d["valid_time"] for d in landfall_data])

    return xr.Dataset(
        {
            "latitude": (["landfall"], latitudes),
            "longitude": (["landfall"], longitudes),
            "surface_wind_speed": (["landfall"], wind_speeds),
            "air_pressure_at_mean_sea_level": (["landfall"], pressures),
            "valid_time": (["landfall"], times),
        },
        coords={"landfall": np.arange(n_landfalls)},
    )


def _find_landfall_forecast(
    track_dataset: xr.Dataset, land_geom
) -> Optional[xr.Dataset]:
    """Find landfall for forecast data (lead_time, valid_time dimensions).
    Simple version that processes by unique init_time.
    """
    # Squeeze out track dimension if it exists and has size 1
    if "track" in track_dataset.dims and track_dataset.sizes["track"] == 1:
        track_dataset = track_dataset.squeeze("track", drop=True)

    landfall_results = []
    valid_init_times = []
    track_dataset = convert_valid_time_to_init_time(track_dataset)
    # Process each lead_time separately (each represents a different init_time)
    for init_time in track_dataset.init_time:
        # Extract data for this lead_time
        single_forecast = track_dataset.sel(init_time=init_time)

        # Convert to numpy arrays
        lats = single_forecast.latitude.values
        lons = single_forecast.longitude.values
        vmax = single_forecast.surface_wind_speed.values
        slp = single_forecast.air_pressure_at_mean_sea_level.values
        times = single_forecast.valid_time.values

        # Remove NaN values
        valid_idx = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(vmax) | np.isnan(slp))
        if np.sum(valid_idx) < 2:
            continue  # Skip if insufficient data

        lats = lats[valid_idx]
        lons = lons[valid_idx]
        vmax = vmax[valid_idx]
        slp = slp[valid_idx]
        times = times[valid_idx]

        # Sort by time
        sort_idx = np.argsort(times)
        lats = lats[sort_idx]
        lons = lons[sort_idx]
        vmax = vmax[sort_idx]
        slp = slp[sort_idx]
        times = times[sort_idx]

        # Find landfall using optimized approach
        landfall = _find_landfall_optimized(lats, lons, vmax, slp, times, land_geom)
        if landfall is not None:
            landfall = landfall.assign_coords(init_time=init_time)
            landfall_results.append(landfall)
            # Use the first init_time value (they should all be the same for this
            # lead_time)
            valid_init_times.append(
                init_time.values if hasattr(init_time, "values") else init_time
            )

    if not landfall_results:
        return None

    # Concatenate along init_time dimension
    combined_landfall = xr.concat(landfall_results, dim="init_time")
    return combined_landfall


def _find_all_landfalls_forecast(
    track_dataset: xr.Dataset, land_geom: Polygon
) -> Optional[xr.Dataset]:
    """Find ALL landfall points for forecast data (lead_time, valid_time dimensions).
    Returns all landfall points for each init_time,
    with init_time and landfall dimensions.
    """
    # Squeeze out track dimension if it exists and has size 1
    if "track" in track_dataset.dims and track_dataset.sizes["track"] == 1:
        track_dataset = track_dataset.squeeze("track", drop=True)

    landfall_results = []
    track_dataset = convert_valid_time_to_init_time(track_dataset)

    # Process each init_time separately
    for init_time in track_dataset.init_time:
        # Extract data for this init_time
        single_forecast = track_dataset.sel(init_time=init_time)

        # Convert to numpy arrays
        lats = single_forecast.latitude.values
        lons = single_forecast.longitude.values
        vmax = single_forecast.surface_wind_speed.values
        slp = single_forecast.air_pressure_at_mean_sea_level.values
        times = single_forecast.valid_time.values

        # Remove NaN values
        valid_idx = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(vmax) | np.isnan(slp))
        if np.sum(valid_idx) < 2:
            continue  # Skip if insufficient data

        lats = lats[valid_idx]
        lons = lons[valid_idx]
        vmax = vmax[valid_idx]
        slp = slp[valid_idx]
        times = times[valid_idx]

        # Sort by time
        sort_idx = np.argsort(times)
        lats = lats[sort_idx]
        lons = lons[sort_idx]
        vmax = vmax[sort_idx]
        slp = slp[sort_idx]
        times = times[sort_idx]

        # Find ALL landfalls for this init_time
        all_landfalls = _find_all_landfalls_optimized(
            lats, lons, vmax, slp, times, land_geom
        )
        if all_landfalls is not None:
            all_landfalls = all_landfalls.assign_coords(init_time=init_time)
            landfall_results.append(all_landfalls)

    if not landfall_results:
        return None

    # Concatenate along init_time dimension
    combined_landfall = xr.concat(landfall_results, dim="init_time")
    return combined_landfall


def _is_true_landfall(
    lon1: float, lat1: float, lon2: float, lat2: float, land_geom: Polygon
):
    """Clean landfall detection following exact requirements:
    1. Ocean -> Land = LANDFALL
    2. Ocean -> Ocean (with land between) = LANDFALL
    3. Land -> Ocean = NOT LANDFALL
    4. Ocean -> Ocean (no land between) = NOT LANDFALL

    Args:
        lon1, lat1: Starting point coordinates
        lon2, lat2: Ending point coordinates
        land_geom: Land geometry for intersection testing

    Returns:
        bool: True if this represents a landfall, False otherwise
    """
    try:
        # Create points
        start_point = Point(lon1, lat1)
        end_point = Point(lon2, lat2)

        # Check if points are over land or ocean
        start_over_land = land_geom.contains(start_point)
        end_over_land = land_geom.contains(end_point)

        # Case 1: Ocean -> Land = LANDFALL
        if not start_over_land and end_over_land:
            return True

        # Case 2: Ocean -> Ocean, check if land is between
        if not start_over_land and not end_over_land:
            segment = LineString([(lon1, lat1), (lon2, lat2)])
            if segment.intersects(land_geom):
                return True

        # Case 3: Land -> Ocean = NOT LANDFALL
        # Case 4: Ocean -> Ocean (no intersection) = NOT LANDFALL
        return False

    except Exception:
        return False

    # ORIGINAL LOGIC (DISABLED FOR NOW):
    # try:
    #     # Create points for start and end positions
    #     start_point = Point(lon1, lat1)
    #     end_point = Point(lon2, lat2)

    #     # Check if start point is over ocean and end point is over land
    #     start_over_ocean = not land_geom.contains(start_point)
    #     end_over_land = land_geom.contains(end_point)

    #     # Case 1: Ocean to Land - clear landfall
    #     if start_over_ocean and end_over_land:
    #         return True

    #     # Case 2: Ocean to Ocean - check if track crosses land (e.g., island crossing)
    #     end_over_ocean = not land_geom.contains(end_point)
    #     if start_over_ocean and end_over_ocean:
    #         segment = LineString([(lon1, lat1), (lon2, lat2)])
    #         if segment.intersects(land_geom):
    #             # Track crosses land between ocean points - this is a landfall
    #             return True

    #     # Case 3: Land to Ocean - this is NOT a landfall (it's an exit)
    #     # Case 4: Land to Land - this is NOT a landfall (already on land)
    #     return False

    # except Exception:
    #     # If there are geometry errors, fall back to simple intersection check
    #     segment = LineString([(lon1, lat1), (lon2, lat2)])
    #     return segment.intersects(land_geom)


def _find_landfall_optimized(
    lats: np.ndarray,
    lons: np.ndarray,
    vmax: np.ndarray,
    slp: np.ndarray,
    times: np.ndarray,
    land_geom: Polygon,
):
    """Optimized landfall detection for pre-processed coordinate arrays.
    Uses simple intersection detection for forecast data.
    """
    if len(lats) < 2:
        return None

    # Convert longitude to -180 to 180 for geometry operations
    lons_180 = (lons + 180) % 360 - 180

    # Check segments for landfall
    for i in range(len(lats) - 1):
        try:
            # Skip invalid coordinates
            if (
                np.isnan(lats[i])
                or np.isnan(lats[i + 1])
                or np.isnan(lons_180[i])
                or np.isnan(lons_180[i + 1])
                or (lats[i] == lats[i + 1] and lons_180[i] == lons_180[i + 1])
            ):
                continue

            # Check if this is a true landfall (ocean to land movement)
            if _is_true_landfall(
                lons_180[i], lats[i], lons_180[i + 1], lats[i + 1], land_geom
            ):
                # Create line segment and get intersection for true landfall
                segment = LineString(
                    [(lons_180[i], lats[i]), (lons_180[i + 1], lats[i + 1])]
                )
                intersection = segment.intersection(land_geom)

                # Extract landfall coordinates - use centroid for better interpolation
                landfall_lon, landfall_lat = None, None
                try:
                    if hasattr(intersection, "coords") and len(intersection.coords) > 0:
                        # For simple point intersections
                        landfall_lon, landfall_lat = intersection.coords[0]
                    elif hasattr(intersection, "centroid"):
                        # Use centroid for complex geometries (better interpolation)
                        centroid = intersection.centroid
                        landfall_lon, landfall_lat = centroid.x, centroid.y
                    elif hasattr(intersection, "geoms") and len(intersection.geoms) > 0:
                        # Multi-geometry case - use first geometry's centroid
                        first_geom = list(intersection.geoms)[0]
                        if hasattr(first_geom, "centroid"):
                            centroid = first_geom.centroid
                            landfall_lon, landfall_lat = centroid.x, centroid.y
                        elif (
                            hasattr(first_geom, "coords") and len(first_geom.coords) > 0
                        ):
                            landfall_lon, landfall_lat = first_geom.coords[0]
                except Exception:
                    # Fallback to segment midpoint if geometry extraction fails
                    landfall_lon = (lons_180[i] + lons_180[i + 1]) / 2
                    landfall_lat = (lats[i] + lats[i + 1]) / 2

                if (
                    landfall_lon is None
                    or np.isnan(landfall_lon)
                    or np.isnan(landfall_lat)
                ):
                    continue

                # Interpolate landfall values
                full_dist = segment.length
                landfall_dist = LineString(
                    [(lons_180[i], lats[i]), (landfall_lon, landfall_lat)]
                ).length
                frac = landfall_dist / full_dist if full_dist > 0 else 0

                landfall_time = times[i] + frac * (times[i + 1] - times[i])
                landfall_vmax = vmax[i] + frac * (vmax[i + 1] - vmax[i])
                landfall_slp = slp[i] + frac * (slp[i + 1] - slp[i])

                # Return landfall dataset
                return xr.Dataset(
                    {
                        "latitude": ([], landfall_lat),
                        "longitude": ([], utils.convert_longitude_to_360(landfall_lon)),
                        "surface_wind_speed": ([], landfall_vmax),
                        "air_pressure_at_mean_sea_level": ([], landfall_slp),
                    },
                    coords={"valid_time": landfall_time},
                )

        except Exception:
            continue

    return None


def _find_landfall_intersection_point(lon1, lat1, lon2, lat2, land_geom):
    """Find the FIRST intersection point where a track segment enters land.
    For ocean->ocean crossings, this returns the entry point, not the exit.
    """

    try:
        segment = LineString([(lon1, lat1), (lon2, lat2)])
        intersection = segment.intersection(land_geom)

        # Collect all intersection points
        intersection_points = []

        if hasattr(intersection, "coords") and len(intersection.coords) > 0:
            # Single point intersection
            intersection_points = list(intersection.coords)
        elif hasattr(intersection, "geoms") and len(intersection.geoms) > 0:
            # Multiple geometries - collect all points
            for geom in intersection.geoms:
                if hasattr(geom, "coords") and len(geom.coords) > 0:
                    intersection_points.extend(list(geom.coords))

        if not intersection_points:
            # No intersection points found, use midpoint
            return ((lon1 + lon2) / 2, (lat1 + lat2) / 2)

        # Find the intersection point closest to the start of the segment
        start_point = np.array([lon1, lat1])
        closest_point = None
        min_distance = float("inf")

        for point in intersection_points:
            point_array = np.array(point)
            distance = np.linalg.norm(point_array - start_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        return (
            closest_point if closest_point else ((lon1 + lon2) / 2, (lat1 + lat2) / 2)
        )

    except Exception:
        # Fallback to segment midpoint
        return ((lon1 + lon2) / 2, (lat1 + lat2) / 2)


def _find_all_landfalls_optimized(lats, lons, vmax, slp, times, land_geom):
    """Optimized detection for ALL true landfall points along pre-processed coordinate
    arrays.

    Only detects ocean to land linestrings.
    """
    if len(lats) < 2:
        return None

    # Convert longitude to -180 to 180 for geometry operations
    lons_180 = (lons + 180) % 360 - 180

    # Find the first landfall only
    for i in range(len(lats) - 1):
        # Skip invalid coordinates
        if (
            np.isnan(lats[i])
            or np.isnan(lats[i + 1])
            or np.isnan(lons_180[i])
            or np.isnan(lons_180[i + 1])
            or (lats[i] == lats[i + 1] and lons_180[i] == lons_180[i + 1])
        ):
            continue

        # Check if this segment represents a landfall
        if _is_true_landfall(
            lons_180[i], lats[i], lons_180[i + 1], lats[i + 1], land_geom
        ):
            # Get the exact intersection point
            landfall_lon, landfall_lat = _find_landfall_intersection_point(
                lons_180[i], lats[i], lons_180[i + 1], lats[i + 1], land_geom
            )

            segment = LineString(
                [(lons_180[i], lats[i]), (lons_180[i + 1], lats[i + 1])]
            )
            full_dist = segment.length

            if full_dist > 0:
                landfall_dist = LineString(
                    [(lons_180[i], lats[i]), (landfall_lon, landfall_lat)]
                ).length
                frac = landfall_dist / full_dist
            else:
                frac = 0.5  # Midpoint if no distance

            # Interpolate time and other values
            landfall_time = times[i] + frac * (times[i + 1] - times[i])
            landfall_vmax = vmax[i] + frac * (vmax[i + 1] - vmax[i])
            landfall_slp = slp[i] + frac * (slp[i + 1] - slp[i])

            # Return the first landfall found
            return xr.Dataset(
                {
                    "latitude": (["landfall"], [landfall_lat]),
                    "longitude": (
                        ["landfall"],
                        [utils.convert_longitude_to_360(landfall_lon)],
                    ),
                    "surface_wind_speed": (["landfall"], [landfall_vmax]),
                    "air_pressure_at_mean_sea_level": (["landfall"], [landfall_slp]),
                },
                coords={
                    "landfall": [0],
                    "valid_time": (["landfall"], [landfall_time]),
                },
            )

    # No landfall found
    return None


def calculate_landfall_time_difference_hours_xarray(
    landfall1: xr.Dataset, landfall2: xr.Dataset
) -> xr.DataArray:
    """Calculate the time difference between two landfall points in hours.

    Args:
        landfall1: First landfall xarray Dataset
        landfall2: Second landfall xarray Dataset

    Returns:
        Time difference in hours (landfall1 - landfall2) as xarray DataArray
        with init_time as the sole dimension
    """
    if landfall1 is None or landfall2 is None:
        return xr.DataArray(np.nan)

    # Get time values from landfall datasets
    time1 = landfall1.valid_time if "valid_time" in landfall1 else landfall1.time
    time2 = landfall2.valid_time if "valid_time" in landfall2 else landfall2.time

    # Calculate time difference in hours
    time_diff = time1 - time2
    time_diff_hours = time_diff / np.timedelta64(1, "h")

    # Return with init_time as the only dimension
    if "init_time" in landfall1.dims:
        # Ensure the values are properly shaped for init_time dimension
        if time_diff_hours.dims == ():
            # Scalar case - broadcast to all init_times
            values = np.full(len(landfall1.init_time), float(time_diff_hours.values))
        else:
            # Already has the right dimensions
            values = time_diff_hours.values

        return xr.DataArray(
            values,
            dims=["init_time"],
            coords={"init_time": landfall1.init_time},
        )
    else:
        # Scalar case - no init_time dimension
        return time_diff_hours


def calculate_landfall_distance_km_xarray(
    landfall1: xr.Dataset, landfall2: xr.Dataset
) -> xr.DataArray:
    """Calculate the distance between two landfall points in kilometers.
    Handles both scalar and multi-dimensional (with init_time) datasets.

    Args:
        landfall1: First landfall xarray Dataset
        landfall2: Second landfall xarray Dataset

    Returns:
        Distance in kilometers as xarray DataArray
    """
    if landfall1 is None or landfall2 is None:
        return xr.DataArray(np.nan)

    # Use xarray operations to handle multi-dimensional case
    distance_degrees = calc.calculate_haversine_distance(
        (landfall1.latitude, landfall1.longitude),
        (landfall2.latitude, landfall2.longitude),
        units="degrees",
    )

    # Convert from degrees to kilometers (1 degree ≈ 111 km at equator)
    distance_km = np.radians(distance_degrees) * 6371

    return distance_km


def generate_tc_variables(ds: xr.Dataset) -> xr.Dataset:
    """Generate the variables needed for the TC track calculation.

    Args:
        ds: The xarray dataset to subset from.

    Returns:
        The subset variables as an xarray Dataset.
    """

    output_vars = {
        "air_pressure_at_mean_sea_level": ds["air_pressure_at_mean_sea_level"],
        "surface_wind_speed": calc.calculate_wind_speed(ds),
    }

    # Only add geopotential thickness if the dataset has level data
    if "level" in ds.dims and "geopotential" in ds.data_vars:
        output_vars["geopotential_thickness"] = calc.generate_geopotential_thickness(
            ds, top_level_value=300, bottom_level_value=500
        )

    output = xr.Dataset(output_vars)
    return output


def convert_valid_time_to_init_time(ds: xr.Dataset) -> xr.Dataset:
    """Convert the valid_time coordinate to a init_time coordinate.

    Args:
        ds: The dataset to convert with lead_time and valid_time coordinates.

    Returns:
        The dataset with a init_time coordinate.
    """
    init_time = xr.DataArray(
        ds.valid_time, coords={"valid_time": ds.valid_time}
    ) - xr.DataArray(ds.lead_time, coords={"lead_time": ds.lead_time})
    ds = ds.assign_coords(init_time=init_time)
    return xr.concat(
        [
            ds.sel(lead_time=lead).swap_dims({"valid_time": "init_time"})
            for lead in ds.lead_time
        ],
        "lead_time",
    )
