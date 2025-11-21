"""Severe convection atmospheric physics calculations for ExtremeWeatherBench.

This module contains constants and functions for calculating atmospheric
convection parameters including CAPE, CIN, and severe weather indices.
Most functions are adapted from the MetPy library (https://github.com/Unidata/MetPy)
with optimizations for gridded meteorological data.

The module supports calculation of:
- Mixed layer convective available potential energy (MLCAPE)
- Convective inhibition (CIN)
- Lifting condensation level (LCL)
- Level of free convection (LFC) and equilibrium level (EL)
- Craven-Brooks significant severe parameter
- Low-level wind shear
- Various thermodynamic quantities

All temperature inputs are in Celsius unless otherwise noted. Inputs are expected
to be in Kelvin for main entrypoints to Craven-Brooks significant severe and MLCAPE.
All pressure inputs are in hPa (hectopascals).
All wind inputs are in m/s.
Energy outputs (CAPE/CIN) are in J/kg.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from extremeweatherbench._cape import (
    _compute_batch_parallel as compute_ml_cape_cin_parallel,
)
from extremeweatherbench._cape import (
    _compute_batch_serial as compute_ml_cape_cin_serial,
)

# Atmospheric Physics Constants
# All constants follow standard atmospheric physics values from CODATA and WMO

# Temperature and pressure reference values
gamma: float = 6.5  # Standard atmospheric temperature lapse rate (K/km)
p0: float = 1000  # Reference pressure (hPa)
p0_stp: float = 1013.25  # Standard atmospheric pressure at sea level (hPa)
t0: float = 288.0  # Standard temperature at sea level (K)
T0: float = 273.15  # Temperature at 0°C in Kelvin (K)

# Gas constants and thermodynamic properties
Rd: float = 287.04749097718457  # Specific gas constant for dry air (J/kg/K)
R: float = 8.314462618  # Universal gas constant (J/mol/K)
Mw: float = 18.015268  # Molecular weight of water (g/mol)
Rv: float = (R / Mw) * 1000  # Specific gas constant for water vapor (J/kg/K)
epsilon: float = 0.6219569100577033  # Ratio of molecular weights (H2O/dry air)
kappa: float = 0.28571428571428564  # Poisson constant (Rd/Cp_d) for dry air

# Specific heat capacities
Cp_d: float = (
    1004.6662184201462  # Specific heat of dry air at constant pressure (J/kg/K)
)
Cp_l: float = 4219.4  # Specific heat of liquid water (J/kg/K)
Cp_v: float = (
    1860.078011865639  # Specific heat of water vapor at constant pressure (J/kg/K)
)

# Physical constants
g: float = 9.81  # Gravitational acceleration (m/s²)
Lv: float = 2500840  # Latent heat of vaporization of water at 0°C (J/kg)
sat_press_0c: float = 6.112  # Saturation vapor pressure at 0°C (hPa)


def craven_brooks_significant_severe(
    air_temperature: xr.DataArray,
    dewpoint_temperature: xr.DataArray,
    geopotential: xr.DataArray,
    pressure: xr.DataArray,
    eastward_wind: xr.DataArray,
    northward_wind: xr.DataArray,
    surface_eastward_wind: xr.DataArray,
    surface_northward_wind: xr.DataArray,
    layer_depth: float = 100,
) -> xr.DataArray:
    """Calculate the Craven-Brooks Significant Severe (CBSS) parameter.

    The CBSS parameter combines thermodynamic (CAPE) and kinematic (wind shear)
    factors to identify environments favorable for significant severe weather
    including supercells, large hail, and tornadoes.

    Formula: CBSS = MLCAPE × 0-6km_Shear

    The parameter integrates two key ingredients for severe thunderstorms:
    - Mixed Layer CAPE: Provides buoyancy for strong updrafts
    - Low-level shear: Promotes storm organization and rotation

    Args:
        air_temperature: Air temperature in Kelvin.
        dewpoint_temperature: Dewpoint temperature in Kelvin.
        geopotential: Geopotential in m2/s2.
        pressure: Pressure in hPa.
        eastward_wind: Eastward wind in m/s.
        northward_wind: Northward wind in m/s.
        surface_eastward_wind: Surface eastward wind in m/s.
        surface_northward_wind: Surface northward wind in m/s.
        layer_depth: Mixed layer depth in hPa for CAPE calculation (default: 100).

    Returns:
        xr.DataArray: CBSS parameter values in m³/s³.

    Notes:
        Interpretation thresholds:
        - CBSS < 10,000 m³/s³: Low severe weather potential
        - CBSS 10,000-22,500 m³/s³: Marginal severe weather potential
        - CBSS > 22,500 m³/s³: Significant severe weather potential
        - CBSS > 50,000 m³/s³: Extreme severe weather potential

        The 0-6 km shear uses winds at 500 hPa (~5.5 km) as proxy for 6 km level.

    References:
        Craven, J. P., and H. E. Brooks, 2004: Baseline climatology of sounding
        derived parameters associated with deep moist convection. Natl. Wea.
        Digest, 28, 13-24.
    """
    # CIN not needed for CBSS
    cape = compute_mixed_layer_cape(
        pressure,
        air_temperature,
        dewpoint_temperature,
        geopotential,
        depth=layer_depth,
    )
    shear = low_level_shear(
        eastward_wind,
        northward_wind,
        surface_eastward_wind,
        surface_northward_wind,
    )
    cbss = cape * shear
    return cbss


def low_level_shear(
    eastward_wind: xr.DataArray,
    northward_wind: xr.DataArray,
    surface_eastward_wind: xr.DataArray,
    surface_northward_wind: xr.DataArray,
) -> xr.DataArray:
    """Calculates the low level (0-6 km) shear of a dataset (Lepore et al 2021).

    Args:
        ds: Dataset containing eastward and northward (u and v) wind vectors

    Returns:
        ll_shear: ndarray of low level shear values in m/s
    """
    ll_shear = np.sqrt(
        (eastward_wind.sel(level=500) - surface_eastward_wind) ** 2
        + (northward_wind.sel(level=500) - surface_northward_wind) ** 2
    )
    return ll_shear


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
        CAPE as xarray DataArrays with pressure_dim removed.

    Examples:
        # Basic usage with parallel processing (default, recommended)
        >>> cape = compute_mixed_layer_cape(
        ...     ds["pressure"],
        ...     ds["temperature"],
        ...     ds["dewpoint"],
        ...     ds["geopotential"],
        ...     pressure_dim="level",
        ... )

        # Dask distributed with large chunks - use parallel (excellent performance)
        >>> # Dataset: 61 timesteps, 160 lat * 280 lon = ~45k profiles/chunk
        >>> ds = xr.open_zarr("era5.zarr", chunks={"time": 1})
        >>> cape = compute_mixed_layer_cape(
        ...     ds["pressure"],
        ...     ds["temperature"],
        ...     ds["dewpoint"],
        ...     ds["geopotential"],
        ...     parallel=True,  # Multi-threaded within each chunk
        ... )

        # Small chunks or debugging - use serial
        >>> ds_small = xr.open_zarr("data.zarr", chunks={"profile": 100})
        >>> cape = compute_mixed_layer_cape(
        ...     ds_small["pressure"],
        ...     ds_small["temperature"],
        ...     ds_small["dewpoint"],
        ...     ds_small["geopotential"],
        ...     parallel=False,  # Single-threaded, simpler behavior
        ... )

        # Isobaric data (pressure is 1D, others are multi-dimensional)
        >>> cape = compute_mixed_layer_cape(
        ...     ds["pressure"],  # shape: (level,) - isobaric levels
        ...     ds["temperature"],  # shape: (time, lat, lon, level)
        ...     ds["dewpoint"],  # shape: (time, lat, lon, level)
        ...     ds["geopotential"],  # shape: (time, lat, lon, level)
        ...     pressure_dim="level",
        ... )
        # xarray automatically broadcasts pressure to match other dimensions

    Notes:
        - Temperature, dewpoint, and geopotential must have matching dimensions/shape
        - Pressure can be 1D (just level) for isobaric data - will be broadcast
            automatically
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
        dask="allowed",  # Enable Dask support
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
