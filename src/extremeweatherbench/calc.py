from typing import Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import regionmask
import scores.categorical as categorical
import xarray as xr

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


def specific_humidity_from_relative_humidity(
    air_temperature: xr.DataArray, relative_humidity: xr.DataArray, levels: xr.DataArray
) -> xr.DataArray:
    """Compute specific humidity from relative humidity and air temperature.

    Args:
        data: The xarray dataset to compute the specific humidity from containing
        levels (hPa), air_temperature (Kelvin), and relative_humidity. If level is not
        included in the dataset, assumed to be surface pressure.

    Returns:
        A DataArray of specific humidity.
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
    e = pressure * w / (w + 0.622)
    T_d = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    return T_d + 273.15
