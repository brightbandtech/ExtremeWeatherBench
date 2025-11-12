from typing import Literal, Optional, Sequence, Union

import cartopy.io.shapereader as shpreader
import numpy as np
import shapely
import xarray as xr

from extremeweatherbench import utils


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
        return ds["geopotential_at_surface"].isel(time=0) / 9.80665
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
            / 9.80665
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
        ) / 9.80665
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


def find_landfalls(
    track_data: xr.DataArray,
    land_geom: Optional[shapely.geometry.Polygon] = None,
    return_all_landfalls: bool = False,
) -> Optional[xr.DataArray]:
    """Find landfall point(s) where tracked object intersects land.

    Generalized landfall detection for any object (TC, AR, etc).
    Expects DataArray with latitude, longitude, valid_time as coords.

    Args:
        track_data: Track DataArray with latitude, longitude,
            valid_time coords. Data values are interpolated at landfall.
            Shape: (valid_time,) or (lead_time, valid_time)
        return_all: If True, return all landfalls; if False, first only

    Returns:
        DataArray with landfall values and lat/lon/time as coords,
        or None if no landfall found
    """

    # If no land geometry is provided, use the default 10m resolution land geometry
    if land_geom is None:
        # Use 10m resolution with buffer for better coastal detection
        land = shpreader.natural_earth(
            category="physical", name="land", resolution="10m"
        )
        land_geoms = list(shpreader.Reader(land).geometries())
        land_geom = shapely.ops.unary_union(land_geoms).buffer(0.1)

    # Squeeze track dimension if needed
    if "track" in track_data.dims and track_data.sizes["track"] == 1:
        track_data = track_data.squeeze("track", drop=True)

    # Detect forecast vs single track data
    is_forecast = "lead_time" in track_data.dims and "valid_time" in track_data.dims

    if is_forecast:
        # Convert to init_time indexing
        temp_ds = xr.Dataset({"track_data": track_data})
        temp_ds = utils.convert_valid_time_to_init_time(temp_ds)
        results = []

        for init_time in temp_ds.init_time:
            init_time_val = (
                init_time.values if hasattr(init_time, "values") else init_time
            )

            single_track = temp_ds["track_data"].sel(init_time=init_time)

            da = _process_single_track_landfall(
                single_track,
                land_geom,
                return_all_landfalls,
                init_time=init_time_val,
            )
            if da is not None:
                results.append(da)

        return xr.concat(results, dim="init_time") if results else None

    else:
        # Process single track data
        return _process_single_track_landfall(
            track_data, land_geom, return_all_landfalls
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
    # Convert forecast to have init_time if needed
    if "init_time" not in forecast_data.coords:
        # Wrap in Dataset, convert, then extract
        temp_ds = xr.Dataset({"data": forecast_data})
        temp_ds = utils.convert_valid_time_to_init_time(temp_ds)
        forecast_data = temp_ds["data"]

    # Vectorized search for next landfall for each init_time
    target_times = target_landfalls.coords["valid_time"].values
    init_times = forecast_data.init_time.values

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


def _process_single_track_landfall(
    track_data: xr.DataArray,
    land_geom,
    return_all_landfalls: bool = False,
    init_time: Optional[np.datetime64] = None,
) -> Optional[xr.DataArray]:
    """Process a single track to find and format landfall data.

    Args:
        track_data: Track DataArray with latitude, longitude,
            valid_time coords
        land_geom: Shapely geometry for land intersection testing
        return_all: If True, return all landfalls; if False, first only
        init_time: If provided, add init_time coordinate to output

    Returns:
        DataArray with landfall values and lat/lon/time as coords,
        or None if no landfalls
    """
    # Filter NaN values and extract coordinate values
    valid_mask = ~np.isnan(track_data)
    track_filtered = track_data.where(valid_mask, drop=True)

    # Get values directly from filtered data
    lats_vals = track_filtered.coords["latitude"].values.flatten()
    lons_vals = track_filtered.coords["longitude"].values.flatten()
    times_vals = track_filtered.coords["valid_time"].values.flatten()
    track_vals = track_filtered.values.flatten()

    # Need at least 2 points for landfall detection
    if len(lats_vals) < 2:
        return None

    # Convert to -180 to 180 longitude
    lons_180 = (lons_vals + 180) % 360 - 180

    # Find landfalls
    landfall_data = []

    for i in range(len(times_vals) - 1):
        try:
            segment = shapely.geometry.LineString(
                [(lons_180[i], lats_vals[i]), (lons_180[i + 1], lats_vals[i + 1])]
            )

            if _is_true_landfall(
                lons_180[i], lats_vals[i], lons_180[i + 1], lats_vals[i + 1], land_geom
            ):
                intersection = segment.intersection(land_geom)
                landfall_lon, landfall_lat = intersection.coords[0]

                # Interpolate landfall values
                full_dist = segment.length
                landfall_dist = shapely.geometry.LineString(
                    [(lons_180[i], lats_vals[i]), (landfall_lon, landfall_lat)]
                ).length
                frac = landfall_dist / full_dist

                landfall_point = {
                    "latitude": landfall_lat,
                    "longitude": utils.convert_longitude_to_360(landfall_lon),
                    "valid_time": times_vals[i]
                    + frac * (times_vals[i + 1] - times_vals[i]),
                }

                # Interpolate the track data value itself
                if track_data.name:
                    landfall_point["value"] = track_vals[i] + frac * (
                        track_vals[i + 1] - track_vals[i]
                    )
                else:
                    # Use NaN if no named variable
                    landfall_point["value"] = np.nan

                landfall_data.append(landfall_point)

                # Stop after first landfall if that's all we need
                if not return_all_landfalls:
                    break
        # Skip this segment if geometry operations fail:
        # - IndexError: empty intersection coords
        # - AttributeError: invalid geometry attributes
        # - ValueError/TypeError: invalid coordinate values
        # - ZeroDivisionError: zero-length segment
        except (IndexError, AttributeError, ValueError, TypeError, ZeroDivisionError):
            continue

    # Format output
    if not landfall_data:
        return None

    if return_all_landfalls:
        # Multiple landfalls with landfall dimension
        values = [d["value"] for d in landfall_data]
        coords = {
            "latitude": (["landfall"], [d["latitude"] for d in landfall_data]),
            "longitude": (["landfall"], [d["longitude"] for d in landfall_data]),
            "valid_time": (["landfall"], [d["valid_time"] for d in landfall_data]),
            "landfall": np.arange(len(landfall_data)),
        }
        da = xr.DataArray(
            values,
            dims=["landfall"],
            coords=coords,
            name=track_data.name if track_data.name else "landfall_value",
        )
    else:
        # Single (first) landfall as scalar
        d = landfall_data[0]
        coords = {
            "latitude": d["latitude"],
            "longitude": d["longitude"],
            "valid_time": d["valid_time"],
        }
        da = xr.DataArray(
            d["value"],
            coords=coords,
            name=track_data.name if track_data.name else "landfall_value",
        )

    # Add init_time if provided
    if init_time is not None:
        da = da.assign_coords(init_time=init_time)

    return da


def _is_true_landfall(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    land_geom: shapely.geometry.Polygon,
) -> bool:
    """Detect true landfall (ocean to land movement).

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
