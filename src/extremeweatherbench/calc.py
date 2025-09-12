from typing import Literal, Sequence, Union

import numpy as np
import xarray as xr


def convert_from_cartesian_to_latlon(
    input_point: Union[np.ndarray, tuple[float, float]], ds_mapping: xr.Dataset
) -> tuple[float, float]:
    """Convert a point from the cartesian coordinate system to the lat/lon coordinate
    system.

    Args:
        input_point: The point to convert, represented as a tuple (y, x) in the
            cartesian coordinate system.
        ds_mapping: The dataset containing the latitude and longitude
            coordinates.

    Returns:
        The point in the lat/lon coordinate system, represented as a tuple
        (latitude, longitude) in degrees.
    """
    return (
        ds_mapping.isel(
            latitude=int(input_point[0]), longitude=int(input_point[1])
        ).latitude.values,
        ds_mapping.isel(
            latitude=int(input_point[0]), longitude=int(input_point[1])
        ).longitude.values,
    )


def calculate_haversine_distance(
    input_a: Sequence[float],
    input_b: Sequence[Union[float, xr.DataArray]],
    units: Literal["km", "kilometers", "deg", "degrees"] = "km",
) -> Union[float, xr.DataArray]:
    """Calculate the great-circle distance between two points on the Earth's surface.

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


def create_great_circle_mask(
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

    distance = calculate_haversine_distance(
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
        from extremeweatherbench.inputs import ARCO_ERA5_FULL_URI

        era5 = xr.open_zarr(
            ARCO_ERA5_FULL_URI,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return (
            era5.isel(time=1000000)["geopotential_at_surface"].sel(
                latitude=ds.latitude, longitude=ds.longitude
            )
            / 9.80665
        )


def calculate_pressure_at_surface(orography_da: xr.DataArray) -> xr.DataArray:
    """Calculate the pressure at the surface, based on orography.

    The dataarray is orography (geopotential at the surface/g0).

    Args:
        orography_da: The orography dataarray.

    Returns:
        The pressure at the surface of the dataarray in Pa.
    """
    return 101325 * (1 - 2.25577e-5 * orography_da) ** 5.25579


def maybe_calculate_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """Prepare wind data by computing wind speed.

    If the wind speed is not already present, it will be computed from the eastward
    and northward wind components. If the wind speed is already present, the dataset
    is returned as is.

    Args:
        ds: The xarray dataset to prepare the wind data from.

    Returns:
        An xarray dataset with the wind speed computed if it is not already present.
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


def generate_geopotential_thickness(
    ds: xr.Dataset,
    var_name: str = "geopotential",
    level_name: str = "level",
    top_level_value: int = 300,
    bottom_level_value: int = 500,
) -> xr.DataArray:
    """Generate the geopotential thickness from the geopotential heights.

    Args:
        ds: The xarray dataset to generate the geopotential thickness from.
        var_name: The name of the variable to generate the geopotential thickness from.
        level_name: The name of the level to generate the geopotential thickness from.
        top_level_value: The value of the top level to generate the geopotential
            thickness from.
        bottom_level_value: The value of the bottom level to generate the
            geopotential thickness from.

    Returns:
        The geopotential thickness as an xarray DataArray.
    """
    geopotential_heights = ds[var_name].sel({level_name: top_level_value})
    geopotential_height_bottom = ds[var_name].sel({level_name: bottom_level_value})
    geopotential_thickness = (
        geopotential_heights - geopotential_height_bottom
    ) / 9.80665
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
