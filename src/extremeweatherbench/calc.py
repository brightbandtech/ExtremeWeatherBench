from typing import Literal, Sequence, Union

import numpy as np
import xarray as xr

from extremeweatherbench._cape import (
    _compute_batch_parallel as compute_ml_cape_cin_parallel,
)
from extremeweatherbench._cape import (
    _compute_batch_serial as compute_ml_cape_cin_serial,
)


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


def calculate_pressure_at_surface(orography_da: xr.DataArray) -> xr.DataArray:
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


def dewpoint_from_specific_humidity(pressure: float, specific_humidity: float) -> float:
    r"""Calculate dewpoint from specific humidity.

    This computation follows the methodology used in `metpy.calc.dewpoint_from_specific_humidity`.
    Given specific humidity $q$, mixing ratio $w$, and pressure $p$, we compute first
    $w = q / (1 - q)$ and $e = p w / (w + \epsilon)$ where $\epsilon=0.622$ is the ratio
    of the molecular weights of water and dry air and $e$ is the partial pressure of water vapor.

    Then, we invert the Bolton (1980) formula to get the dewpoint $T_d$ given $e$:

    $$T_d =\frac{243.5 \log(e / 6.112)}{17.67 - \log(e / 6.112)}$$

    Args:
        pressure: The ambient pressure in hPa.
        specific_humidity: The specific humidity in kg/kg.

    Returns:
        The dewpoint in Kelvin.
    """
    # NOTE: there are some pathological cases where the specific humidity is negative in some
    # NWP and MLWP data sources. For safety, you should restrict the specific humidity to the
    # interval [1e-10, 1], which addresses this and should preserve the remaining physical
    # constraints.
    w = specific_humidity / (1.0 - specific_humidity)
    e = pressure * w / (w + 0.622)
    T_d = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    return T_d + 273.15


def compute_mixed_layer_cape(
    pressure: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    geopotential: xr.DataArray,
    pressure_dim: str = "level",
    depth: float = 100.0,
    parallel: bool = True,
) -> xr.DataArray:
    """Compute mixed-layer CAPE from thermodynamic profiles.

    This function applies the optimized CAPE/CIN calculation to xarray DataArrays,
    handling both in-memory and Dask-backed arrays automatically. It supports arbitrary
    dimensional layouts including multi-dimensional grids.

    In general, we recommend using parallel=False whenever you are processing data with
    less than ~1,000 profiles per chunk; otherwise, use parallel=True to take advantage
    of batching at the chunk level.

    We don't return the computed CIN here because it's not yet implemented correctly.

    Args:
        pressure: Pressure in hPa. Must have pressure_dim as one of its dimensions.
        temperature: Temperature in Kelvin. Must be broadcastable against pressure.
        dewpoint: Dewpoint in Kelvin. Must be broadcastable against pressure.
        geopotential: Geopotential in m2/s2. Must be broadcastable against pressure.
        pressure_dim: Name of the pressure level dimension (default: 'level').
        depth: Mixed layer depth in hPa (default: 100.0).
        parallel: If True, use Numba parallel processing within chunks (default: True).
            Uses batched computation with prange for multi-threaded computation. Set to
            False for serial processing

    Returns:
        Tuple of (cape, cin) as xarray DataArrays with pressure_dim removed.

    Examples:
        # Basic usage with parallel processing (default, recommended)
        >>> cape, cin = compute_cape_cin(
        ...     ds["pressure"],
        ...     ds["temperature"],
        ...     ds["dewpoint"],
        ...     ds["geopotential"],
        ...     pressure_dim="level",
        ... )

        # Dask distributed with large chunks - use parallel (excellent performance)
        >>> # Dataset: 61 timesteps, 160 lat * 280 lon = ~45k profiles/chunk
        >>> ds = xr.open_zarr("era5.zarr", chunks={"time": 1})
        >>> cape, cin = compute_cape_cin(
        ...     ds["pressure"],
        ...     ds["temperature"],
        ...     ds["dewpoint"],
        ...     ds["geopotential"],
        ...     parallel=True,  # Multi-threaded within each chunk
        ... )

        # Small chunks or debugging - use serial
        >>> ds_small = xr.open_zarr("data.zarr", chunks={"profile": 100})
        >>> cape, cin = compute_cape_cin(
        ...     ds_small["pressure"],
        ...     ds_small["temperature"],
        ...     ds_small["dewpoint"],
        ...     ds_small["geopotential"],
        ...     parallel=False,  # Single-threaded, simpler behavior
        ... )

        # Isobaric data (pressure is 1D, others are multi-dimensional)
        >>> cape, cin = compute_cape_cin(
        ...     ds["pressure"],  # shape: (level,) - isobaric levels
        ...     ds["temperature"],  # shape: (time, lat, lon, level)
        ...     ds["dewpoint"],  # shape: (time, lat, lon, level)
        ...     ds["geopotential"],  # shape: (time, lat, lon, level)
        ...     pressure_dim="level",
        ... )
        # xarray automatically broadcasts pressure to match other dimensions

    Notes:
        - Temperature, dewpoint, and geopotential must have matching dimensions/shape
        - Pressure can be 1D (just level) for isobaric data - will be broadcast automatically
        - The pressure_dim must be present in all input arrays with the same size
        - Pressure levels should be in descending order (surface to top)
        - For Dask arrays, pressure_dim MUST be in a single chunk
        - Recommended chunking: All spatial dims in one chunk, 1 time step per chunk
          Example: (time=1, lat=*, lon=*, level=*) where * means all values
        - For large ERA5-like grids (lat * lon > 10k points): parallel=True is optimal
        - Dask distributed + large chunks + parallel=True: ~8-10x speedup vs serial
        - NumPy arrays: parallel=True provides multi-core speedup automatically
        - The parallel overhead is negligible for chunks >1000 profiles
    """

    cape_batch_func = (
        compute_ml_cape_cin_parallel if parallel else compute_ml_cape_cin_serial
    )

    # Define the wrapper function that will be applied
    def _compute_cape_cin_wrapper(p, t, td, z):
        """Wrapper to handle the conversion and calling of Numba function.

        This expects p, t, td, z with shape (..., n_levels) where ... represents
        any number of batch dimensions that will be flattened into n_profiles.

        Note: Pressure (p) might be 1D (level,) for isobaric data while others
        are multi-dimensional. We broadcast it to match if needed.
        """

        # Get the original shape (excluding the pressure level dimension which is last)
        # Use temperature's shape as reference
        original_shape = t.shape[:-1]
        n_levels = t.shape[-1]

        # Broadcast pressure if it's 1D (isobaric data)
        if p.ndim == 1:
            # Pressure is 1D: (level,) - broadcast to match temperature shape
            # Create the target shape: (..., n_levels)
            target_shape = t.shape
            p = np.broadcast_to(p, target_shape)

        # Reshape to (n_profiles, n_levels) by flattening all batch dimensions
        n_profiles = np.prod(original_shape, dtype=int)

        p_batch = p.reshape(n_profiles, n_levels)
        t_batch = t.reshape(n_profiles, n_levels)
        td_batch = td.reshape(n_profiles, n_levels)
        z_batch = z.reshape(n_profiles, n_levels)

        # Ensure arrays are contiguous and correct dtype
        p_batch = np.ascontiguousarray(p_batch, dtype=np.float64)
        t_batch = np.ascontiguousarray(t_batch, dtype=np.float64)
        td_batch = np.ascontiguousarray(td_batch, dtype=np.float64)
        z_batch = np.ascontiguousarray(z_batch, dtype=np.float64)

        # Call the selected Numba batch function (parallel or serial)
        cape_array, _ = cape_batch_func(
            p_batch, t_batch, td_batch, z_batch, depth=depth
        )

        # Reshape back to original batch dimensions (removing pressure dimension)
        cape_array = cape_array.reshape(original_shape)

        return cape_array

    # Use xarray.apply_ufunc to apply the wrapped interface
    cape = xr.apply_ufunc(
        _compute_cape_cin_wrapper,
        pressure,
        temperature,
        dewpoint,
        geopotential,
        input_core_dims=[
            [pressure_dim],
            [pressure_dim],
            [pressure_dim],
            [pressure_dim],
        ],
        output_core_dims=[[]],  # CAPE has no pressure dimension
        vectorize=False,  # We handle the batching ourselves
        dask="parallelized",  # Enable Dask support
        output_dtypes=[np.float64],  # Specify output types
        dask_gufunc_kwargs={
            "output_sizes": {},  # No new dimensions created
        },
    )

    # Add metadata
    cape.name = "cape"
    cape.attrs = {
        "long_name": "Convective Available Potential Energy",
        "units": "J/kg",
        "description": f"Mixed-layer CAPE computed over {depth} hPa depth",
    }
    return cape
