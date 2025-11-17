from typing import Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import regionmask
import scores.categorical as categorical
import shapely
import xarray as xr

from extremeweatherbench import utils

epsilon: float = 0.6219569100577033  # Ratio of molecular weights (H2O/dry air)
sat_press_0c: float = 6.112  # Saturation vapor pressure at 0°C (hPa)
g0: float = 9.80665  # Standard gravity (m/s^2)


def convert_from_cartesian_to_latlon(
    input_point: Union[np.ndarray, tuple[float, float]],
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> tuple[float, float]:
    """Convert point from cartesian coordinate system to lat/lon.

    Args:
        input_point: Point as tuple (y, x) in cartesian system
        latitude: Latitude DataArray
        longitude: Longitude DataArray

    Returns:
        Tuple (latitude, longitude) in degrees
    """
    lat_idx = int(input_point[0])
    lon_idx = int(input_point[1])
    return (
        latitude.isel(latitude=lat_idx).values,
        longitude.isel(longitude=lon_idx).values,
    )


def mixing_ratio(
    partial_pressure: float | npt.NDArray, total_pressure: float | npt.NDArray
) -> float | npt.NDArray[np.float64]:
    r"""Calculate the mixing ratio of water vapor in air.

    Uses the formula: $w = (\epsilon * e) / (p - e)$ where $\epsilon = 0.622$.

    Args:
        partial_pressure: Water vapor partial pressure in hPa.
        total_pressure: Total atmospheric pressure in hPa.

    Returns:
        numpy.ndarray: Mixing ratio in kg/kg (dimensionless).

    Notes:
        - Mixing ratio is approximately constant with height for unsaturated air
        - Values typically range from 0 to ~0.025 kg/kg in the atmosphere
        - ε (epsilon) = 0.622 is the ratio of molecular weights (H2O/dry air)
    """
    # Suppress warnings for this specific calculation
    with np.errstate(divide="ignore", invalid="ignore"):
        return epsilon * partial_pressure / (total_pressure - partial_pressure)


def saturation_vapor_pressure(
    temperature: float | npt.NDArray,
) -> npt.NDArray[np.float64]:
    r"""Calculate saturation vapor pressure using the Clausius-Clapeyron equation.

    Uses the Magnus formula approximation which is accurate for temperatures
    between -40°C and +50°C. Formula:

    $$e_ = 6.112 \times \exp\left(\frac{17.67*T}{T+243.5}\right)$$

    Args:
        temperature: Temperature values in Celsius (can be scalar or array).

    Returns:
        numpy.ndarray: Saturation vapor pressure values in hPa.

    Notes:
        - Based on the Magnus formula which is accurate within ±0.1% for typical
          atmospheric temperatures
        - Saturation vapor pressure increases exponentially with temperature
        - At 0°C: ~6.11 hPa, at 20°C: ~23.4 hPa, at 30°C: ~42.4 hPa
    """
    # Suppress overflow warnings for this calculation
    with np.errstate(over="ignore", invalid="ignore"):
        return sat_press_0c * np.exp(17.67 * temperature / (temperature + 243.5))


def saturation_mixing_ratio(
    pressure: float | npt.NDArray, temperature: float | npt.NDArray
) -> npt.NDArray[np.float64]:
    """Calculates the saturation mixing ratio of a parcel.

    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in C

    Returns:
        Saturation mixing ratio values in kg/kg
    """
    return mixing_ratio(saturation_vapor_pressure(temperature), pressure)


def haversine_distance(
    input_a: Sequence[Union[float, xr.DataArray]],
    input_b: Sequence[Union[float, xr.DataArray]],
    units: Literal["km", "kilometers", "deg", "degrees"] = "km",
) -> Union[float, xr.DataArray]:
    """Calculate the great-circle/haversine distance between two points on the Earth's
    surface.

    Args:
        input_a: The first point, represented as an ndarray of shape (2,n) in
            degrees lat/lon.
        input_b: The second point(s), represented as an ndarray of shape (2,n)
            in degrees lat/lon.

    Returns:
        The great-circle distance between the two points in kilometers.
    """
    # Convert to radians for calculations
    lat1 = np.radians(input_a[0])
    lon1 = np.radians(input_a[1])
    lat2 = np.radians(input_b[0])
    lon2 = np.radians(input_b[1])

    # Haversine formula for great circle distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    if units == "km" or units == "kilometers":
        return 6371 * c
    elif units == "deg" or units == "degrees":
        return np.degrees(c)  # Convert back to degrees
    else:
        raise ValueError(f"Invalid units: {units}")


def great_circle_mask(
    ds: xr.Dataset, latlon_point: tuple[float, float], radius_degrees: float
) -> xr.DataArray:
    """Create a circular mask based on great circle distance for an xarray dataset.

    Args:
        ds: Dataset with 'latitude' and 'longitude' coordinates.
        latlon_point: Tuple containing (latitude, longitude) of the center point.
        radius_degrees: Radius in degrees of great circle distance.

    Returns:
        Boolean mask where True indicates points within the radius.
    """

    distance = haversine_distance(
        latlon_point, (ds.latitude, ds.longitude), units="deg"
    )
    # Create mask as xarray DataArray
    if isinstance(distance, xr.DataArray):
        mask = distance <= radius_degrees
    else:
        # If distance is a scalar, create a DataArray mask
        mask = xr.full_like(ds.latitude, distance <= radius_degrees, dtype=bool)

    return mask


def orography(ds: xr.Dataset) -> xr.DataArray:
    """Calculate the orography from the geopotential at the surface using ERA5.

    Args:
        ds: The potential xarray dataset to calculate the orography from.

    Returns:
        The orography as an xarray DataArray.
    """

    if "geopotential_at_surface" in ds.variables:
        return ds["geopotential_at_surface"].isel(time=0) / g0
    else:
        # Import inputs here to avoid circular import
        from extremeweatherbench import inputs

        era5 = xr.open_zarr(
            inputs.ARCO_ERA5_FULL_URI,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return (
            era5.isel(time=1000000)["geopotential_at_surface"].sel(
                latitude=ds.latitude, longitude=ds.longitude
            )
            / g0
        )


def pressure_at_surface(orography_da: xr.DataArray) -> xr.DataArray:
    """Calculate the pressure at the surface, based on orography.

    The dataarray is orography (geopotential at the surface/g0).

    Args:
        orography_da: The orography dataarray.

    Returns:
        The pressure at the surface of the dataarray in Pa.
    """
    return 101325 * (1 - 2.25577e-5 * orography_da) ** 5.25579


def maybe_calculate_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    """Maybe prepare wind data by computing wind speed.

    If the wind speed is not already present, it will be computed from the eastward
    and northward wind components (u and v). If the wind speed is already present, the
    dataset is returned as is.

    Args:
        ds: The dataset to prepare the wind data from.

    Returns:
        A dataset with the wind speed computed if it is not already present.
    """

    has_wind_speed = "surface_wind_speed" in ds.data_vars
    has_wind_components = (
        "surface_eastward_wind" in ds.data_vars
        and "surface_northward_wind" in ds.data_vars
    )

    # If we don't have wind speed but have components, compute it
    if not has_wind_speed and has_wind_components:
        ds["surface_wind_speed"] = np.hypot(
            ds["surface_eastward_wind"], ds["surface_northward_wind"]
        )

    return ds


def geopotential_thickness(
    da: xr.DataArray,
    top_level_value: int = 300,
    bottom_level_value: int = 500,
    geopotential: bool = False,
) -> xr.DataArray:
    """Generate the geopotential thickness from the geopotential heights.

    Args:
        da: The xarray DataArray to generate the geopotential thickness from.
        top_level_value: The value of the top level to generate the geopotential
            thickness from.
        bottom_level_value: The value of the bottom level to generate the
            geopotential thickness from.
        geopotential: Whether the input DataArray is geopotential height or
        geopotential (default is geopotential height).

    Returns:
        The geopotential thickness as an xarray DataArray.
    """
    geopotential_heights = da.sel({"level": top_level_value})
    geopotential_height_bottom = da.sel({"level": bottom_level_value})
    if geopotential:
        geopotential_thickness = (
            geopotential_heights - geopotential_height_bottom
        ) / g0
    else:
        geopotential_thickness = geopotential_heights - geopotential_height_bottom
    geopotential_thickness.attrs = dict(
        description="Geopotential thickness of level and 500 hPa", units="m"
    )
    return geopotential_thickness


def nantrapezoid(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
):
    """Trapezoid rule for arrays with nans.

    Identical to np.trapezoid but with nans handled correctly in the summation.
    """
    y = np.asanyarray(y)
    if x is None:
        # Create an array of the step size
        d = np.full(y.shape[axis] - 1, dx) if y.shape[axis] > 1 else np.array([dx])
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = d.shape[0]
        d = d.reshape(shape)
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    if y.ndim != d.ndim:
        d = np.expand_dims(d, axis=1)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        # This is the only location different from np.trapezoid
        ret = np.nansum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret


def specific_humidity_from_relative_humidity(
    air_temperature: xr.DataArray, relative_humidity: xr.DataArray, levels: xr.DataArray
) -> xr.DataArray:
    """Compute specific humidity from relative humidity and air temperature.

    Args:
        air_temperature: Air temperature in Kelvin.
        relative_humidity: Relative humidity (0-1 or 0-100 depending on data).
        levels: Pressure levels in hPa.

    Returns:
        A DataArray of specific humidity in kg/kg.
    """
    # Compute saturation mixing ratio; air temperature must be in Kelvin
    sat_mixing_ratio = saturation_mixing_ratio(levels, air_temperature - 273.15)

    # Calculate specific humidity using saturation mixing ratio, epsilon,
    # and relative humidity
    mixing_ratio = (
        epsilon
        * sat_mixing_ratio
        * relative_humidity
        / (epsilon + sat_mixing_ratio * (1 - relative_humidity))
    )
    specific_humidity = mixing_ratio / (1 + mixing_ratio)
    return specific_humidity


def find_land_intersection(
    mask: xr.DataArray, land_mask: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Find points where a data mask intersects with a land mask.

    Args:
        mask: a boolean mask of data locations that includes latitude and longitude
        land_mask: a boolean mask of land locations

    Returns:
        a mask of points where AR overlaps with land
    """
    if land_mask is None:
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
            mask.longitude, mask.latitude
        )
        land_mask = land_mask.where(np.isnan(land_mask), 1).where(land_mask == 0, 0)

    # Use the scores.categorical library to compute the binary mask (true positives)
    contingency_manager = categorical.BinaryContingencyManager(mask, land_mask)
    # return the true positive mask, where mask is true and land is true
    return contingency_manager.tp


def dewpoint_from_specific_humidity(pressure: float, specific_humidity: float) -> float:
    r"""Calculate dewpoint from specific humidity.

    This computation follows the methodology used in
    `metpy.calc.dewpoint_from_specific_humidity`. Given specific humidity $q$, mixing
    ratio $w$, and pressure $p$, we compute first $w = q / (1 - q)$ and
    $e = p w / (w + \epsilon)$ where $\epsilon=0.622$ is the ratio of the molecular
    weights of water and dry air and $e$ is the partial pressure of water vapor.

    Then, we invert the Bolton (1980) formula to get the dewpoint $T_d$ given $e$:

    $$T_d =\frac{243.5 \log(e / 6.112)}{17.67 - \log(e / 6.112)}$$

    Args:
        pressure: The ambient pressure in hPa.
        specific_humidity: The specific humidity in kg/kg.

    Returns:
        The dewpoint in Kelvin.
    """
    # NOTE: there are some pathological cases where the specific humidity is negative
    # in some NWP and MLWP data sources. For safety, you should restrict the specific
    # humidity to the interval [1e-10, 1], which addresses this and should preserve the
    # remaining physical constraints.
    w = specific_humidity / (1.0 - specific_humidity)
    e = pressure * w / (w + epsilon)
    T_d = 243.5 * np.log(e / sat_press_0c) / (17.67 - np.log(e / sat_press_0c))
    return T_d + 273.15


def find_landfalls(
    track_data: xr.DataArray,
    land_geom: Optional[shapely.geometry.Polygon] = None,
    return_all_landfalls: bool = False,
) -> xr.DataArray:
    """Find landfall point(s) where tracked object intersects land.

    Generalized landfall detection for any object (TC, AR, etc).
    Expects DataArray with latitude, longitude, valid_time as coords.

    Args:
        track_data: Track DataArray with latitude, longitude,
            valid_time coords. Data values are interpolated at landfall.
            Shape: (valid_time,) or (lead_time, valid_time)
        return_all_landfalls: If True, return all landfalls; else first
        land_geom: Shapely geometry for land intersection testing

    Returns:
        DataArray with landfall values and lat/lon/time as coords,
        or None if no landfall found
    """
    # If no land geometry is provided, use default
    if land_geom is None:
        land_geom = utils.load_land_geometry()

    # Squeeze track dimension if needed
    if "track" in track_data.dims and track_data.sizes["track"] == 1:
        track_data = track_data.squeeze("track", drop=True)

    # Detect forecast vs single track data
    is_forecast = "lead_time" in track_data.dims and "valid_time" in track_data.dims

    if is_forecast:
        # Convert to init_time coords for boundary detection
        track_data = utils.convert_valid_time_to_init_time(track_data)

        # Flatten to single time dimension for vectorized processing
        track_data_flat = track_data.stack(time=("lead_time", "init_time"))

        # Vectorized landfall detection
        landfall_mask = _detect_landfalls_wrapper(track_data_flat, land_geom)

        # Mask init_time boundaries
        landfall_mask = _mask_init_time_boundaries(landfall_mask, track_data_flat)

        # Interpolate at landfall points
        result = _interpolate_and_format_landfalls(
            track_data_flat,
            landfall_mask,
            land_geom,
            return_all_landfalls,
            group_by="init_time",
        )

        return result
    else:
        # Process single track data
        landfall_mask = _detect_landfalls_wrapper(track_data, land_geom)

        return _interpolate_and_format_landfalls(
            track_data,
            landfall_mask,
            land_geom,
            return_all_landfalls,
            group_by=None,
        )


def find_next_landfall_for_init_time(
    forecast_data: xr.DataArray,
    target_landfalls: xr.DataArray,
) -> Optional[xr.DataArray]:
    """Find next upcoming landfall from target data for each init_time.

    For each forecast initialization time, finds the next landfall
    event in the target data that occurs after that init_time.

    Args:
        forecast_data: Forecast DataArray with init_time dimension
        target_landfalls: Target landfall DataArray from find_landfalls
            with return_all=True (has landfall dimension)

    Returns:
        DataArray with next landfall for each init_time, or None if
        no future landfalls exist
    """
    # Extract init_times from forecast
    if "init_time" in forecast_data.dims:
        init_times = forecast_data.init_time.values
    elif "lead_time" in forecast_data.coords and "valid_time" in forecast_data.coords:
        # Calculate init_times from lead_time and valid_time
        init_times_calc = forecast_data.coords["valid_time"] - forecast_data.lead_time
        init_times = np.unique(init_times_calc.values)
    else:
        return None

    # Get target landfall times
    target_times = target_landfalls.coords["valid_time"].values

    # Use searchsorted to find next landfall index for each init_time
    next_indices = np.searchsorted(target_times, init_times, side="right")

    # Filter to only valid indices (where future landfall exists)
    valid_mask = next_indices < len(target_times)

    if not valid_mask.any():
        return None

    # Select the next landfalls and corresponding init_times
    selected_landfalls = target_landfalls.isel(landfall=next_indices[valid_mask])
    selected_init_times = init_times[valid_mask]

    # Assign init_time coordinate and return
    return selected_landfalls.assign_coords(
        init_time=("landfall", selected_init_times)
    ).swap_dims({"landfall": "init_time"})


def _detect_landfalls_wrapper(
    track_data: xr.DataArray,
    land_geom: shapely.geometry.Polygon,
) -> xr.DataArray:
    """Vectorized landfall detection across consecutive point pairs.

    Args:
        track_data: Track data with latitude, longitude coords
        land_geom: Land geometry for intersection testing

    Returns:
        Boolean DataArray where True indicates a landfall between
        point i and point i+1
    """
    # Determine time dimension name
    time_dims = [d for d in track_data.dims if "time" in d.lower()]
    if not time_dims:
        return xr.DataArray(
            np.array([], dtype=bool),
            dims=track_data.dims,
        )
    time_dim = time_dims[0]

    # Extract coordinates
    lats = track_data.coords["latitude"]
    lons = track_data.coords["longitude"]

    # Convert to -180/180
    lons_180 = (lons + 180) % 360 - 180

    # Get shifted versions for consecutive pairs
    lats_next = lats.shift({time_dim: -1})
    lons_180_next = lons_180.shift({time_dim: -1})

    # Vectorize the landfall check function
    def _check_landfall_scalar(lon1, lat1, lon2, lat2):
        """Scalar landfall check for vectorization."""
        # Handle NaN values (from shift at boundaries)
        if np.isnan(lon1) or np.isnan(lat1) or np.isnan(lon2) or np.isnan(lat2):
            return False
        return _is_true_landfall(lon1, lat1, lon2, lat2, land_geom)

    # Vectorize the function
    check_vectorized = np.vectorize(_check_landfall_scalar, otypes=[bool])

    # Apply to get landfall mask
    landfall_mask = xr.apply_ufunc(
        check_vectorized,
        lons_180,
        lats,
        lons_180_next,
        lats_next,
        vectorize=False,  # We already vectorized with np.vectorize
        dask="parallelized",
        output_dtypes=[bool],
    )

    return landfall_mask


def _mask_init_time_boundaries(
    landfall_mask: xr.DataArray,
    track_data: xr.DataArray,
) -> xr.DataArray:
    """Mask out landfalls at init_time boundaries.

    Prevents comparing last point of one init_time with
    first point of next init_time.

    Args:
        landfall_mask: Boolean mask of potential landfalls
        track_data: Original track data with init_time coord

    Returns:
        Masked landfall array with boundaries set to False
    """
    if "init_time" not in track_data.coords:
        return landfall_mask

    # Determine time dimension
    time_dims = [d for d in track_data.dims if "time" in d.lower()]
    if not time_dims:
        return landfall_mask
    time_dim = time_dims[0]

    # Get init_time for each point and the next point
    init_curr = track_data.coords["init_time"]
    init_next = init_curr.shift({time_dim: -1})

    # Only keep landfalls where init_time doesn't change
    same_init = init_curr == init_next

    # Also handle NaN from shift operation
    same_init = same_init.fillna(False)

    return landfall_mask & same_init


def _interpolate_and_format_landfalls(
    track_data: xr.DataArray,
    landfall_mask: xr.DataArray,
    land_geom: shapely.geometry.Polygon,
    return_all_landfalls: bool,
    group_by: Optional[str] = None,
) -> Optional[xr.DataArray]:
    """Interpolate landfall points and format output.

    Args:
        track_data: Original track data
        landfall_mask: Boolean mask of landfall locations
        land_geom: Land geometry for intersection
        return_all_landfalls: Whether to return all or just first
        group_by: If not None, group by this coord (e.g., "init_time")

    Returns:
        Formatted landfall DataArray or None if no landfalls
    """
    # Determine time dimension
    time_dims = [d for d in track_data.dims if "time" in d.lower()]
    if not time_dims:
        return None

    # Get indices where landfalls occur
    landfall_indices = np.where(landfall_mask.values)[0]

    if len(landfall_indices) == 0:
        return None

    # Extract coordinate arrays
    lats_vals = track_data.coords["latitude"].values
    lons_vals = track_data.coords["longitude"].values
    times_vals = track_data.coords["valid_time"].values
    track_vals = track_data.values

    # Convert longitudes to -180/180
    lons_180 = (lons_vals + 180) % 360 - 180

    # Get init_time if available
    init_times = None
    if "init_time" in track_data.coords:
        init_times = track_data.coords["init_time"].values

    # Process each landfall
    landfall_data = []

    for i in landfall_indices:
        try:
            # Create segment
            segment = shapely.geometry.LineString(
                [(lons_180[i], lats_vals[i]), (lons_180[i + 1], lats_vals[i + 1])]
            )

            # Get intersection point
            intersection = segment.intersection(land_geom)

            # Handle different intersection types
            if intersection.is_empty:
                continue

            # Get first intersection point based on geometry type
            geom_type = intersection.geom_type

            if geom_type == "Point":
                landfall_lon, landfall_lat = intersection.x, intersection.y
            elif geom_type == "LineString":
                landfall_lon, landfall_lat = intersection.coords[0]
            elif geom_type in ("MultiPoint", "MultiLineString", "GeometryCollection"):
                # Get first geometry from collection
                first_geom = intersection.geoms[0]
                if first_geom.geom_type == "Point":
                    landfall_lon, landfall_lat = first_geom.x, first_geom.y
                else:
                    landfall_lon, landfall_lat = first_geom.coords[0]
            else:
                continue

            # Interpolate
            full_dist = segment.length
            if full_dist == 0:
                continue

            landfall_dist = shapely.geometry.LineString(
                [(lons_180[i], lats_vals[i]), (landfall_lon, landfall_lat)]
            ).length
            frac = landfall_dist / full_dist

            landfall_point = {
                "latitude": landfall_lat,
                "longitude": utils.convert_longitude_to_360(landfall_lon),
                "valid_time": times_vals[i]
                + frac * (times_vals[i + 1] - times_vals[i]),
                "value": track_vals[i] + frac * (track_vals[i + 1] - track_vals[i])
                if not np.isnan(track_vals[i])
                else np.nan,
            }

            # Add init_time if available
            if init_times is not None:
                landfall_point["init_time"] = init_times[i]

            landfall_data.append(landfall_point)

        except (IndexError, AttributeError, ValueError, TypeError, ZeroDivisionError):
            continue

    if not landfall_data:
        return None

    # Format output based on grouping
    if group_by == "init_time" and init_times is not None:
        # Group by init_time and optionally keep first per group
        results = []
        unique_inits = np.unique([d["init_time"] for d in landfall_data])

        for init_t in unique_inits:
            init_landfalls = [d for d in landfall_data if d["init_time"] == init_t]

            if not return_all_landfalls:
                # Keep only first landfall for this init_time
                init_landfalls = [init_landfalls[0]]

            # Create DataArray for this init_time
            if len(init_landfalls) == 1:
                d = init_landfalls[0]
                da = xr.DataArray(
                    d["value"],
                    coords={
                        "latitude": d["latitude"],
                        "longitude": d["longitude"],
                        "valid_time": d["valid_time"],
                        "init_time": init_t,
                    },
                    name=track_data.name or "landfall_value",
                )
            else:
                # Multiple landfalls for this init_time
                values = [d["value"] for d in init_landfalls]
                coords = {
                    "latitude": (["landfall"], [d["latitude"] for d in init_landfalls]),
                    "longitude": (
                        ["landfall"],
                        [d["longitude"] for d in init_landfalls],
                    ),
                    "valid_time": (
                        ["landfall"],
                        [d["valid_time"] for d in init_landfalls],
                    ),
                    "landfall": np.arange(len(init_landfalls)),
                    "init_time": init_t,
                }
                da = xr.DataArray(
                    values,
                    dims=["landfall"],
                    coords=coords,
                    name=track_data.name or "landfall_value",
                )
            results.append(da)

        return xr.concat(results, dim="init_time") if results else None

    else:
        # No grouping - format as single or multiple landfalls
        if return_all_landfalls:
            # Return with landfall dimension
            values = [d["value"] for d in landfall_data]
            coords = {
                "latitude": (["landfall"], [d["latitude"] for d in landfall_data]),
                "longitude": (["landfall"], [d["longitude"] for d in landfall_data]),
                "valid_time": (["landfall"], [d["valid_time"] for d in landfall_data]),
                "landfall": np.arange(len(landfall_data)),
            }
            return xr.DataArray(
                values,
                dims=["landfall"],
                coords=coords,
                name=track_data.name or "landfall_value",
            )
        else:
            # Return first landfall as scalar
            d = landfall_data[0]
            coords = {
                "latitude": d["latitude"],
                "longitude": d["longitude"],
                "valid_time": d["valid_time"],
            }
            if "init_time" in d:
                coords["init_time"] = d["init_time"]
            return xr.DataArray(
                d["value"],
                coords=coords,
                name=track_data.name or "landfall_value",
            )


# Keep existing _process_single_track_landfall for backward compatibility
# or remove if fully replaced


def _is_true_landfall(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    land_geom: shapely.geometry.Polygon,
) -> bool:
    """Detect true landfall (ocean to land movement).

    This function is a way to detect if a track crosses land. There are a few scenarios,
    being ocean to land, ocean to ocean, and land to ocean. Ocean to ocean can have a
    landfall if there is land between the two points.

    Args:
        lon1, lat1: Starting point coordinates
        lon2, lat2: Ending point coordinates
        land_geom: Land geometry for intersection testing

    Returns:
        True if this represents a landfall, False otherwise
    """
    try:
        start_point = shapely.geometry.Point(lon1, lat1)
        end_point = shapely.geometry.Point(lon2, lat2)

        start_over_land = land_geom.contains(start_point)
        end_over_land = land_geom.contains(end_point)

        # Ocean -> Land = LANDFALL
        if not start_over_land and end_over_land:
            return True

        # Ocean -> Ocean, check if land is between
        if not start_over_land and not end_over_land:
            segment = shapely.geometry.LineString([(lon1, lat1), (lon2, lat2)])
            if segment.intersects(land_geom):
                return True

        # Land -> Ocean or Ocean -> Ocean (no intersection) = NOT LANDFALL
        return False

    except (AttributeError, ValueError, TypeError):
        # Return False if geometry operations fail:
        # - AttributeError: invalid/None geometry
        # - ValueError/TypeError: invalid coordinate values
        return False
