"""Module for calculating convective available potential energy (CAPE).

This module implements several functions to compute CAPE from collections of
atmospheric profiles retrieved from NWP or MLWP forecast datasets. In general,
we follow the methodology implemented by MetPy [1], although we make a few
modifications to simplify the computation. The implementation here is most appropriate
for analyzing the coarser profiles typical of model output rather than profiles
from rawinsonde observations.

Users can use the functions in this module directly, although we expect that they will
find it easier to use the xarray-based interfaces in the `cape.py` module.


[1]: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.mixed_layer_cape_cin.html
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange

# ============================================================================
# Physical Constants
# ============================================================================

# Thermodynamic constants
KAPPA = 2.0 / 7.0  # Poisson constant R/Cp for dry air (dimensionless)
GRAVITY = 9.80665  # Gravitational acceleration (m/s^2)
Rd = 287.058  # Gas constant for dry air (J/kg/K)
Cp = 1005.7  # Specific heat at constant pressure for dry air (J/kg/K)

# Water vapor constants
EPSILON = 0.622  # Ratio of molecular weights of water vapor to dry air (dimensionless)
VIRTUAL_TEMP_COEFF = (
    0.61  # Coefficient for virtual temperature calculation (dimensionless)
)

# Latent heat constants
L_V_0 = 2.501e6  # Latent heat of vaporization at 0°C (J/kg)
L_V_TEMP_COEFF = 2370.0  # Temperature dependence of latent heat (J/kg/K)

# Reference values
P_REF = 1000.0  # Reference pressure for potential temperature (hPa)
KELVIN_TO_CELSIUS = 273.15  # Conversion factor from Kelvin to Celsius (K)

# Bolton (1980) formula constants for saturation vapor pressure
# e_s = E0 * exp(A * T_c / (T_c + B))
# where T_c is temperature in Celsius
E0_BOLTON = 6.112  # Reference vapor pressure (hPa)
A_BOLTON = 17.67  # Empirical constant (dimensionless)
B_BOLTON = 243.5  # Empirical constant (°C)

# Bolton (1980) LCL formula constants
LCL_OFFSET = 56.0  # Empirical constant for LCL calculation (K)
LCL_DENOM = 800.0  # Empirical constant for LCL calculation (K)

# Numerical integration parameters
MOIST_ASCENT_STEPS = 50  # Number of steps for moist adiabat integration

# Data processing parameters
RADIUS_DEG = 2.0  # Default radius for sample data extraction (degrees)


# ============================================================================
# Inlined Helper Functions
#
# Each of these functions is intended to be inlined by the Numba/LLVM compiler
# to improve performance.
# ============================================================================


@njit(inline="always", fastmath=True)
def saturation_vapor_pressure_inline(temperature: float) -> float:
    """Inline saturation vapor pressure calculation (Bolton 1980).

    Args:
        temperature: The temperature in Kelvin.

    Returns:
        The saturation vapor pressure in hPa.
    """
    t_celsius = temperature - KELVIN_TO_CELSIUS
    return E0_BOLTON * np.exp(A_BOLTON * t_celsius / (t_celsius + B_BOLTON))


@njit(inline="always", fastmath=True)
def mixing_ratio_inline(pressure: float, vapor_pressure: float) -> float:
    """Inline mixing ratio calculation with divide-by-zero protection.

    Ensures denominator is always positive by capping vapor_pressure at
    99.99% of pressure to handle supersaturation and numerical precision issues.

    Args:
        pressure: The pressure in hPa.
        vapor_pressure: The vapor pressure in hPa.

    Returns:
        The mixing ratio in kg/kg.
    """
    # Prevent supersaturation: cap vapor pressure at 0.9999 * pressure
    # This handles both real supersaturation in data and numerical precision issues
    max_vapor_pressure = 0.9999 * pressure
    if vapor_pressure > max_vapor_pressure:
        vapor_pressure = max_vapor_pressure
    return EPSILON * vapor_pressure / (pressure - vapor_pressure)


@njit(inline="always", fastmath=True)
def virtual_temperature_inline(temperature: float, w: float) -> float:
    """Inline virtual temperature from mixing ratio.

    Args:
        temperature: The temperature in Kelvin.
        w: The mixing ratio in kg/kg.

    Returns:
        The virtual temperature in Kelvin.
    """
    return temperature * (1.0 + 0.61 * w)


@njit(inline="always", fastmath=True)
def potential_temperature_inline(temperature: float, pressure: float) -> float:
    """Inline potential temperature calculation.

    Args:
        temperature: The temperature in Kelvin.
        pressure: The pressure in hPa.

    Returns:
        The potential temperature in Kelvin.
    """
    return temperature * (P_REF / pressure) ** KAPPA


@njit(inline="always", fastmath=True)
def temperature_from_theta_inline(theta: float, pressure: float) -> float:
    """Inline temperature from potential temperature.

    Args:
        theta: The potential temperature in Kelvin.
        pressure: The pressure in hPa.

    Returns:
        The temperature in Kelvin.
    """
    return theta * (pressure / P_REF) ** KAPPA


@njit(inline="always", fastmath=True)
def compute_buoyancy_energy_inline(
    parcel_tv_avg: float, env_tv_avg: float, dz: float
) -> float:
    """Safely compute buoyancy energy with divide-by-zero protection.

    Returns 0.0 if environment virtual temperature is too small or invalid.
    This handles edge cases from corrupted data or extreme atmospheric conditions.

    Args:
        parcel_tv_avg: The average virtual temperature of the parcel in Kelvin.
        env_tv_avg: The average virtual temperature of the environment in Kelvin.
        dz: The height difference in meters.

    Returns:
        The buoyancy energy in J/kg.
    """
    # Minimum reasonable virtual temperature (in K) to avoid division issues
    # ~100 K is well below any realistic atmospheric temperature
    MIN_TV = 100.0

    if env_tv_avg < MIN_TV or not np.isfinite(env_tv_avg):
        return 0.0

    return GRAVITY * (parcel_tv_avg - env_tv_avg) / env_tv_avg * dz


# ============================================================================
# Core Functions
# ============================================================================


@njit(fastmath=True)
def lcl(pressure: float, temperature: float, dewpoint: float) -> tuple[float, float]:
    """Fast LCL calculation with inline math, following Bolton (1980).

    Args:
        pressure: The pressure in hPa.
        temperature: The temperature in Kelvin.
        dewpoint: The dewpoint in Kelvin.

    Returns:
        The LCL pressure and temperature in hPa and Kelvin, respectively.
    """
    # LCL temperature (Bolton 1980, eq. 15)
    t_lcl = (
        1.0 / (1.0 / (dewpoint - 56.0) + np.log(temperature / dewpoint) / 800.0) + 56.0
    )

    # LCL pressure (Bolton 1980, eq. 22)
    theta = potential_temperature_inline(temperature, pressure)
    p_lcl = P_REF * (t_lcl / theta) ** (1.0 / KAPPA)

    return p_lcl, t_lcl


@njit(fastmath=True)
def insert_lcl_level(
    pressure: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    dewpoint: npt.NDArray[np.float64],
    geopotential: npt.NDArray[np.float64],
    p_lcl: float,
    t_lcl: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int,
]:
    """Insert LCL as a new level in the profile for better resolution.

    This function consumes 1D profiles of pressure, temperature, dewpoint, and geopotential
    and determines the index at which thermodynamic values at the LCL should be inserted
    into each profile. We directly compute the LCL pressure and temperature, and
    logarithmically interpolate the dewpoint and geopotential to this level.

    Args:
        pressure: The pressure in hPa.
        temperature: The temperature in Kelvin.
        dewpoint: The dewpoint in Kelvin.
        geopotential: The geopotential in meters.
        p_lcl: The LCL pressure in hPa.
        t_lcl: The LCL temperature in Kelvin.

    Returns:
        The pressure, temperature, dewpoint, geopotential, and insert index.
    """
    n = len(pressure)

    # Find where to insert LCL
    insert_idx = 0
    for i in range(n):
        if pressure[i] >= p_lcl:
            insert_idx = i + 1
        else:
            break

    # Check if LCL is already at an existing level (within 0.1 hPa)
    if insert_idx < n and abs(pressure[insert_idx] - p_lcl) < 0.1:
        # Copy arrays to maintain consistent types
        new_p = np.empty(n, dtype=np.float64)
        new_t = np.empty(n, dtype=np.float64)
        new_td = np.empty(n, dtype=np.float64)
        new_z = np.empty(n, dtype=np.float64)
        for i in range(n):
            new_p[i] = pressure[i]
            new_t[i] = temperature[i]
            new_td[i] = dewpoint[i]
            new_z[i] = geopotential[i]
        return new_p, new_t, new_td, new_z, insert_idx

    if insert_idx > 0 and abs(pressure[insert_idx - 1] - p_lcl) < 0.1:
        new_p = np.empty(n, dtype=np.float64)
        new_t = np.empty(n, dtype=np.float64)
        new_td = np.empty(n, dtype=np.float64)
        new_z = np.empty(n, dtype=np.float64)
        for i in range(n):
            new_p[i] = pressure[i]
            new_t[i] = temperature[i]
            new_td[i] = dewpoint[i]
            new_z[i] = geopotential[i]
        return new_p, new_t, new_td, new_z, insert_idx - 1

    # Pre-allocate new arrays
    new_pressure = np.empty(n + 1, dtype=np.float64)
    new_temperature = np.empty(n + 1, dtype=np.float64)
    new_dewpoint = np.empty(n + 1, dtype=np.float64)
    new_geopotential = np.empty(n + 1, dtype=np.float64)

    # Copy data before LCL
    for i in range(insert_idx):
        new_pressure[i] = pressure[i]
        new_temperature[i] = temperature[i]
        new_dewpoint[i] = dewpoint[i]
        new_geopotential[i] = geopotential[i]

    # Insert LCL level
    new_pressure[insert_idx] = p_lcl
    new_temperature[insert_idx] = t_lcl

    # Interpolate dewpoint and geopotential at LCL
    if insert_idx > 0 and insert_idx < n:
        log_p_below = np.log(pressure[insert_idx - 1])
        log_p_above = np.log(pressure[insert_idx])
        log_p_lcl = np.log(p_lcl)

        frac = (log_p_lcl - log_p_below) / (log_p_above - log_p_below)
        new_dewpoint[insert_idx] = dewpoint[insert_idx - 1] + frac * (
            dewpoint[insert_idx] - dewpoint[insert_idx - 1]
        )
        new_geopotential[insert_idx] = geopotential[insert_idx - 1] + frac * (
            geopotential[insert_idx] - geopotential[insert_idx - 1]
        )
    elif insert_idx == 0:
        new_dewpoint[insert_idx] = dewpoint[0]
        new_geopotential[insert_idx] = geopotential[0]
    else:
        new_dewpoint[insert_idx] = dewpoint[-1]
        new_geopotential[insert_idx] = geopotential[-1]

    # Copy data after LCL
    for i in range(insert_idx, n):
        new_pressure[i + 1] = pressure[i]
        new_temperature[i + 1] = temperature[i]
        new_dewpoint[i + 1] = dewpoint[i]
        new_geopotential[i + 1] = geopotential[i]

    return new_pressure, new_temperature, new_dewpoint, new_geopotential, insert_idx


@njit(fastmath=True)
def moist_ascent(p_target: float, p_lcl: float, t_lcl: float) -> float:
    """Compute the temperature of a parcel that is ascending moist adiabatically from the LCL.

    Args:
        p_target: The target pressure in hPa.
        p_lcl: The LCL pressure in hPa.
        t_lcl: The LCL temperature in Kelvin.

    Returns:
        The temperature in Kelvin.
    """
    if p_target >= p_lcl:
        return t_lcl * (p_target / p_lcl) ** KAPPA

    # Logarithmic pressure integration
    log_p_start = np.log(p_lcl)
    log_p_end = np.log(p_target)
    d_log_p = (log_p_end - log_p_start) / MOIST_ASCENT_STEPS

    t_current = t_lcl
    log_p_current = log_p_start

    for _ in range(MOIST_ASCENT_STEPS):
        p_current = np.exp(log_p_current)

        # Inlined saturation vapor pressure and mixing ratio
        e_s = saturation_vapor_pressure_inline(t_current)
        w_s = mixing_ratio_inline(p_current, e_s)

        # Latent heat and moist adiabatic factor
        L_v = L_V_0 - L_V_TEMP_COEFF * (t_current - KELVIN_TO_CELSIUS)

        numerator = 1.0 + L_v * w_s / (Rd * t_current)
        denominator = 1.0 + L_v * L_v * w_s * EPSILON / (
            Cp * Rd * t_current * t_current
        )

        dt_dlogp = KAPPA * t_current * numerator / denominator

        t_current += dt_dlogp * d_log_p
        log_p_current += d_log_p

    return t_current


@njit(fastmath=True)
def compute_ml_cape_cin_from_profile(
    pressure: np.ndarray,
    temperature: npt.NDArray[np.float64],
    dewpoint: npt.NDArray[np.float64],
    geopotential: npt.NDArray[np.float64],
    depth: float = 100.0,
) -> tuple[float, float]:
    """Compute CAPE/CIN for a given thermodynamic profile.

    This function operates on a single thermodynamic profile at a time, and uses a
    variety of inlined helper functions to ensure that the computation is as fast
    as possible.

    WARNING: The CIN computation is not yet implemented correctly and may give
    erroneous results.

    Args:
        pressure: The pressure in hPa.
        temperature: The temperature in Kelvin.
        dewpoint: The dewpoint in Kelvin.
        geopotential: The geopotential in meters.
        depth: The depth of the mixed layer in hPa.

    Returns:
        The CAPE and CIN in J/kg.
    """
    n_levels = len(pressure)

    # Step 1: Mixed layer properties
    p_surface = pressure[0]
    p_bottom = p_surface - depth

    sum_theta_weighted = 0.0
    sum_w_weighted = 0.0
    sum_weights = 0.0

    for i in range(n_levels - 1):
        if pressure[i] >= p_bottom:
            weight = pressure[i] - max(pressure[i + 1], p_bottom)

            if weight > 0:
                theta_i = potential_temperature_inline(temperature[i], pressure[i])
                e_i = saturation_vapor_pressure_inline(dewpoint[i])
                w_i = mixing_ratio_inline(pressure[i], e_i)

                sum_theta_weighted += theta_i * weight
                sum_w_weighted += w_i * weight
                sum_weights += weight
        else:
            break

    if sum_weights == 0:
        return 0.0, 0.0

    ml_theta = sum_theta_weighted / sum_weights
    w_ml = sum_w_weighted / sum_weights
    ml_temp = temperature_from_theta_inline(ml_theta, p_surface)

    # Mixed layer dewpoint from mixing ratio
    e_ml = p_surface * w_ml / (EPSILON + w_ml)
    ml_dewpoint = (
        B_BOLTON * np.log(e_ml / E0_BOLTON) / (A_BOLTON - np.log(e_ml / E0_BOLTON))
        + KELVIN_TO_CELSIUS
    )

    # Step 2: LCL
    p_lcl, t_lcl = lcl(p_surface, ml_temp, ml_dewpoint)

    # Step 2b: Insert LCL into profile for better resolution
    pressure, temperature, dewpoint, geopotential, lcl_idx = insert_lcl_level(
        pressure, temperature, dewpoint, geopotential, p_lcl, t_lcl
    )

    # Update n_levels after potential LCL insertion
    n_levels = len(pressure)

    # Step 3: Parcel and environment virtual temperatures
    parcel_tv = np.empty(n_levels, dtype=np.float64)
    env_tv = np.empty(n_levels, dtype=np.float64)

    for i in range(n_levels):
        p = pressure[i]

        # Parcel temperature
        if p > p_lcl:
            t_parcel = ml_temp * (p / p_surface) ** KAPPA
            w_parcel = w_ml
        else:
            t_parcel = moist_ascent(p, p_lcl, t_lcl)
            e_parcel = saturation_vapor_pressure_inline(t_parcel)
            w_parcel = mixing_ratio_inline(p, e_parcel)

        # Virtual temperatures (inlined)
        parcel_tv[i] = virtual_temperature_inline(t_parcel, w_parcel)

        e_env = saturation_vapor_pressure_inline(dewpoint[i])
        w_env = mixing_ratio_inline(p, e_env)
        env_tv[i] = virtual_temperature_inline(temperature[i], w_env)

    # Step 4: CAPE and CIN with zero-crossing interpolation
    heights = geopotential / GRAVITY

    # Find zero crossings
    max_crossings = n_levels - 1
    crossing_pressures = np.empty(max_crossings, dtype=np.float64)
    crossing_heights = np.empty(max_crossings, dtype=np.float64)
    crossing_indices = np.empty(max_crossings, dtype=np.int64)
    crossing_directions = np.empty(max_crossings, dtype=np.int64)
    n_crossings = 0

    for i in range(n_levels - 1):
        buoy_i = parcel_tv[i] - env_tv[i]
        buoy_i1 = parcel_tv[i + 1] - env_tv[i + 1]

        if buoy_i * buoy_i1 < 0:
            log_p_i = np.log(pressure[i])
            log_p_i1 = np.log(pressure[i + 1])

            frac = buoy_i / (buoy_i - buoy_i1)
            log_p_cross = log_p_i + frac * (log_p_i1 - log_p_i)
            p_cross = np.exp(log_p_cross)

            z_cross = heights[i] + frac * (heights[i + 1] - heights[i])

            direction = 1 if buoy_i1 > buoy_i else -1

            crossing_pressures[n_crossings] = p_cross
            crossing_heights[n_crossings] = z_cross
            crossing_indices[n_crossings] = i
            crossing_directions[n_crossings] = direction
            n_crossings += 1

    # Find LFC: first increasing crossing
    lfc_pressure = -1.0
    lfc_height = -1.0
    lfc_level_idx = -1

    for i in range(n_crossings):
        if crossing_directions[i] == 1:
            lfc_pressure = crossing_pressures[i]
            lfc_height = crossing_heights[i]
            lfc_level_idx = crossing_indices[i]
            break

    # If no increasing crossing but surface is warm, use surface as LFC
    if lfc_pressure < 0 and parcel_tv[0] > env_tv[0]:
        lfc_pressure = pressure[0]
        lfc_height = heights[0]
        lfc_level_idx = 0

    # Find EL: last decreasing crossing after LFC
    el_pressure = -1.0
    el_height = -1.0
    el_level_idx = -1

    if lfc_level_idx >= 0:
        for i in range(n_crossings):
            if crossing_directions[i] == -1 and crossing_indices[i] >= lfc_level_idx:
                el_pressure = crossing_pressures[i]
                el_height = crossing_heights[i]
                el_level_idx = crossing_indices[i]

    cape = 0.0
    cin = 0.0

    # No LFC case
    if lfc_pressure < 0:
        for i in range(n_levels - 1):
            dz = heights[i + 1] - heights[i]
            tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
            parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5

            energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
            if energy < 0:
                cin += energy

        return 0.0, cin

    # Integrate CIN from surface to LFC
    for i in range(n_levels - 1):
        if i >= lfc_level_idx:
            if i == lfc_level_idx:
                dz = lfc_height - heights[i]
                tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
                parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5
                energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
                cin += energy
            break

        dz = heights[i + 1] - heights[i]
        tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
        parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5
        energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
        cin += energy

    # Integrate CAPE from LFC to EL (or top)
    if el_pressure > 0:
        for i in range(lfc_level_idx, n_levels - 1):
            if i == lfc_level_idx:
                z1 = lfc_height
                z2 = heights[i + 1]
            else:
                z1 = heights[i]
                z2 = heights[i + 1]

            if i >= el_level_idx:
                if i == el_level_idx:
                    z1 = heights[i] if i > lfc_level_idx else lfc_height
                    z2 = el_height
                    dz = z2 - z1
                    tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
                    parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5
                    energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
                    if energy > 0:
                        cape += energy
                break

            dz = z2 - z1
            tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
            parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5
            energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
            if energy > 0:
                cape += energy
    else:
        # No EL - integrate to top
        for i in range(lfc_level_idx, n_levels - 1):
            if i == lfc_level_idx:
                z1 = lfc_height
                z2 = heights[i + 1]
            else:
                z1 = heights[i]
                z2 = heights[i + 1]

            dz = z2 - z1
            tv_avg = (env_tv[i] + env_tv[i + 1]) * 0.5
            parcel_avg = (parcel_tv[i] + parcel_tv[i + 1]) * 0.5
            energy = compute_buoyancy_energy_inline(parcel_avg, tv_avg, dz)
            if energy > 0:
                cape += energy

    return cape, cin


# ============================================================================
# Batch Processing Functions
#
# Depending on the number of profiles the user wishes to analyze, it may be
# more efficient to process them in batches. Numba provides some automation which
# tries to optimize the calculation for running in parallel. Otherwise, we will
# simply use a parallel range loop to process the profiles.
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _compute_batch_parallel(
    pressure_batch: npt.NDArray[np.float64],
    temperature_batch: npt.NDArray[np.float64],
    dewpoint_batch: npt.NDArray[np.float64],
    geopotential_batch: npt.NDArray[np.float64],
    depth: float = 100.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute CAPE/CIN for multiple profiles in parallel.

    This uses Numba's parallel processing to compute multiple profiles simultaneously.

    Args:
        pressure_batch: 2D array of shape (n_profiles, n_levels)
        temperature_batch: 2D array of shape (n_profiles, n_levels)
        dewpoint_batch: 2D array of shape (n_profiles, n_levels)
        geopotential_batch: 2D array of shape (n_profiles, n_levels)
        depth: Mixed layer depth in hPa

    Returns:
        tuple of (cape_array, cin_array) with shape (n_profiles,)
    """
    n_profiles = pressure_batch.shape[0]

    cape_array = np.empty(n_profiles, dtype=np.float64)
    cin_array = np.empty(n_profiles, dtype=np.float64)

    # Parallel loop over profiles
    for i in prange(n_profiles):
        cape_array[i], cin_array[i] = compute_ml_cape_cin_from_profile(
            pressure_batch[i],
            temperature_batch[i],
            dewpoint_batch[i],
            geopotential_batch[i],
            depth,
        )

    return cape_array, cin_array


@njit(fastmath=True, cache=True)
def _compute_batch_serial(
    pressure_batch: npt.NDArray[np.float64],
    temperature_batch: npt.NDArray[np.float64],
    dewpoint_batch: npt.NDArray[np.float64],
    geopotential_batch: npt.NDArray[np.float64],
    depth: float = 100.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute CAPE/CIN for multiple profiles serially.

    Use this for smaller batches where parallel overhead isn't worth it.

    Args:
        pressure_batch: 2D array of shape (n_profiles, n_levels)
        temperature_batch: 2D array of shape (n_profiles, n_levels)
        dewpoint_batch: 2D array of shape (n_profiles, n_levels)
        geopotential_batch: 2D array of shape (n_profiles, n_levels)
        depth: Mixed layer depth in hPa

    Returns:
        tuple of (cape_array, cin_array) with shape (n_profiles,)
    """
    n_profiles = pressure_batch.shape[0]

    cape_array = np.empty(n_profiles, dtype=np.float64)
    cin_array = np.empty(n_profiles, dtype=np.float64)

    for i in range(n_profiles):
        cape_array[i], cin_array[i] = compute_ml_cape_cin_from_profile(
            pressure_batch[i],
            temperature_batch[i],
            dewpoint_batch[i],
            geopotential_batch[i],
            depth,
        )

    return cape_array, cin_array


def compute_ml_cape_cin_batched(
    pressure: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    dewpoint: npt.NDArray[np.float64],
    geopotential: npt.NDArray[np.float64],
    depth: float = 100.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Automatically choose serial or parallel based on batch size.

    Args:
        pressure: 2D array of shape (n_profiles, n_levels)
        temperature: 2D array of shape (n_profiles, n_levels)
        dewpoint: 2D array of shape (n_profiles, n_levels)
        geopotential: 2D array of shape (n_profiles, n_levels)
        depth: Mixed layer depth in hPa

    Returns:
        tuple of (cape_array, cin_array) each with shape (n_profiles,)
    """
    n_profiles = pressure.shape[0]

    # Use parallel for large batches (>100 profiles)
    if n_profiles > 100:
        return _compute_batch_parallel(
            pressure, temperature, dewpoint, geopotential, depth
        )
    else:
        return _compute_batch_serial(
            pressure, temperature, dewpoint, geopotential, depth
        )
