"""
Severe convection atmospheric physics calculations for ExtremeWeatherBench.

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

All temperature inputs are in Celsius unless otherwise noted.
All pressure inputs are in hPa (hectopascals).
All wind inputs are in m/s.
Energy outputs (CAPE/CIN) are in J/kg.

Constants defined in this module follow standard atmospheric physics:
- Dry air gas constant (Rd): 287.05 J/(kg·K)
- Specific heat at constant pressure (Cp_d): 1004.67 J/(kg·K)
- Gravitational acceleration (g): 9.81 m/s²
- Water vapor gas constant (Rv): 461.5 J/(kg·K)
"""

import itertools
from importlib import resources

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.optimize as so
import xarray as xr
from scipy.interpolate import interp1d
from scipy.special import lambertw

# Physical constants following standard atmospheric physics values
gamma = 6.5  # Standard atmospheric temperature lapse rate (K/km)
p0 = 1000  # Reference pressure (hPa)
p0_stp = 1013.25  # Standard atmospheric pressure at sea level (hPa)
t0 = 288.0  # Standard temperature at sea level (K)
Rd = 287.04749097718457  # Specific gas constant for dry air (J/kg/K)
depth = 100  # Default mixed layer depth (hPa)
epsilon = 0.6219569100577033  # Ratio of molecular weights (water vapor/dry air)
sat_press_0c = 6.112  # Saturation vapor pressure at 0°C (hPa)
kappa = 0.28571428571428564  # Poisson constant (Rd/Cp_d) for dry air
g = 9.81  # Gravitational acceleration (m/s²)
Lv = 2500840  # Latent heat of vaporization of water at 0°C (J/kg)
Cp_d = 1004.6662184201462  # Specific heat of dry air at constant pressure (J/kg/K)
R = 8.314462618  # Universal gas constant (J/mol/K)
Mw = 18.015268  # Molecular weight of water (g/mol)
Rv = (R / Mw) * 1000  # Specific gas constant for water vapor (J/kg/K)
Cp_l = 4219.4  # Specific heat of liquid water (J/kg/K)
Cp_v = 1860.078011865639  # Specific heat of water vapor at constant pressure (J/kg/K)
T0 = 273.15  # Temperature at 0°C in Kelvin (K)


def load_moist_lapse_lookup():
    """Load the moist lapse lookup table for pseudoadiabatic calculations.

    Loads a precomputed lookup table containing moist adiabatic temperature
    profiles for various starting conditions. This table is used to efficiently
    calculate parcel temperatures above the LCL without iterative computation.

    Returns:
        pd.DataFrame: Lookup table with pressure levels as index and temperature
            profiles as columns. Each column represents a different starting
            temperature condition for moist adiabatic ascent.

    Notes:
        - The lookup table is stored as a parquet file for efficient loading
        - Used by moist_lapse_lookup() for interpolating parcel temperatures
        - Critical for CAPE/CIN calculations above the lifting condensation level
    """
    import extremeweatherbench.data

    moist_lapse_lookup_table = resources.files(extremeweatherbench.data).joinpath(
        "moist_lapse_lookup.parq"
    )
    moist_lapse_lookup_df = pd.read_parquet(moist_lapse_lookup_table)
    return moist_lapse_lookup_df


def _basic_ds_checks(ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare and validate dataset for atmospheric convection calculations.

    Performs essential data quality checks and transformations:
    1. Ensures pressure levels are in descending order (surface to top)
    2. Moves 'level' dimension to last position for calculation efficiency
    3. Filters out stratospheric levels (< 50 hPa) where calculations are invalid

    Args:
        ds: Input xarray Dataset containing atmospheric profile data.
            Must have 'level' coordinate in hPa.

    Returns:
        xr.Dataset: Validated and properly ordered dataset ready for calculations.

    Notes:
        - Pressure levels must be in hPa
        - Removes levels above 50 hPa as convective calculations are less reliable
          in the stratosphere
        - Sorts pressure levels in descending order (high to low pressure)
    """

    # make sure the pressure level is descending. If not, sort it
    if ds["level"][0] < ds["level"][-1]:
        ds = ds.sortby("level", ascending=False)
        for var in ds.data_vars:
            if "level" in ds[var].dims:
                ds[var] = ds[var].sortby("level", ascending=False)

    # Make sure dims are in order of latitude, longitude, level at the end.
    if "level" in ds.dims:
        # Get all dimensions and reorder them properly
        dims = list(ds.dims)
        ordered_dims = []

        # Add dimensions that are not lat, lon, or level first
        for dim in dims:
            if dim not in ["latitude", "longitude", "level"]:
                ordered_dims.append(dim)

        # Add latitude and longitude if they exist
        if "latitude" in dims:
            ordered_dims.append("latitude")
        if "longitude" in dims:
            ordered_dims.append("longitude")

        # Add level last
        if "level" in dims:
            ordered_dims.append("level")

        ds = ds.transpose(*ordered_dims)

    # Make sure the level is at least 50 hPa. If not, drop the levels below 50 hPa
    # Calculations for CAPE (e.g. virtual temperature) are less relevant above the
    # troposphere
    if any(ds["level"] < 50):
        ds = ds.sel(level=slice(ds["level"].max(), 50))
    return ds


def _interp_integrate(pressure, pressure_interp, layer_depth, vars, axis=0):
    vars_interp = log_interpolate(pressure_interp, pressure, vars)
    integration = np.trapezoid(vars_interp, pressure_interp, axis=axis) / -layer_depth
    return integration


def moist_lapse_lookup(target_pressure, target_temp, reference_pressure=None):
    """
    Find the column in test_df that matches the closest temperature and pressure.

    Parameters:
    -----------
    target_temp : float or ndarray
        Target temperature(s) in Celsius
    target_pressure : float or ndarray
        Target pressure(s) in hPa
    reference_pressure: float or ndarray
        Pressure(s) to start lifting from in hPa
    table_path : str
        Location of lookup table

    Returns:
    --------
    ndarray
        Array of temperature profiles that best match the target conditions
    """

    moist_lapse_lookup_df = load_moist_lapse_lookup()
    # Convert inputs to arrays if they aren't already
    target_temp = np.asarray(target_temp)
    target_pressure = np.asarray(target_pressure)

    # Handle empty arrays
    if target_pressure.size == 0 or target_temp.size == 0:
        return np.array([])

    if target_temp.ndim > 1:
        target_temp = target_temp.flatten()

    # Ensure target_pressure is at least 2D
    if target_pressure.ndim == 1:
        target_pressure = target_pressure.reshape(1, -1)

    # Flatten target pressure and temp arrays to 2D
    target_pressure_reshaped = target_pressure.reshape(-1, target_pressure.shape[-1])
    # Get the pressure level closest to each target_pressure
    if reference_pressure is None:
        reference_pressure = target_pressure_reshaped[:, 0]
    else:
        # round reference pressure to be able to get the index in the lookup table
        reference_pressure = np.round(reference_pressure, 2)

    # TODO: doesn't work with > 1 dimension, fix this so it can
    # Reshape reference pressure to 1D for indexing
    reference_pressure_flat = reference_pressure.flatten()
    pressure_indices = moist_lapse_lookup_df.index.get_indexer(
        reference_pressure_flat, method="nearest"
    )
    # Get the temperature at those pressure levels for each column
    temps_at_pressure = moist_lapse_lookup_df.iloc[pressure_indices]
    # Vectorized computation of closest column indices
    temp_diff = np.abs(temps_at_pressure.values - target_temp.flatten()[:, np.newaxis])
    closest_col_indices = np.argmin(temp_diff, axis=1)

    # Get the corresponding temperature profiles
    profiles = np.array(
        [moist_lapse_lookup_df.iloc[:, idx].values for idx in closest_col_indices]
    )
    profiles = profiles.reshape(*target_temp.shape, -1)

    # Vectorized interpolation using apply_ufunc
    def _interp_moist_lapse(target_p, profile):
        return np.interp(
            target_p[::-1], moist_lapse_lookup_df.index[::-1], profile[::-1]
        )[::-1]

    target_p_da = xr.DataArray(
        target_pressure_reshaped,
        dims=[
            *[f"dim_{i}" for i in range(target_pressure_reshaped.ndim - 1)],
            "pressure",
        ],
    )
    profiles_da = xr.DataArray(
        profiles, dims=[*[f"dim_{i}" for i in range(profiles.ndim - 1)], "level"]
    )

    interpolated_profiles = xr.apply_ufunc(
        _interp_moist_lapse,
        target_p_da,
        profiles_da,
        input_core_dims=[["pressure"], ["level"]],
        output_core_dims=[["pressure"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).values.reshape(target_pressure.shape)

    # remove values where nans exist in target_pressure_reshaped
    interpolated_profiles = np.where(
        np.isnan(target_pressure), np.nan, interpolated_profiles
    )
    return interpolated_profiles


def mixing_ratio(partial_pressure, total_pressure):
    """Calculate the mixing ratio of water vapor in air.

    The mixing ratio represents the mass of water vapor per unit mass of dry air.
    Uses the formula: w = ε * e / (p - e) where ε = 0.622.

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


def vapor_pressure(pressure, mixing_ratio):
    """Calculates the vapor pressure of a parcel.

    Args:
        pressure: Pressure values
        mixing_ratio: Mixing ratio values in kg/kg

    Returns:
        Vapor pressure values in provided pressure units
    """
    # Suppress warnings for this specific calculation
    with np.errstate(divide="ignore", invalid="ignore"):
        return pressure * mixing_ratio / (epsilon + mixing_ratio)


def saturation_vapor_pressure(temperature):
    """Calculate saturation vapor pressure using the Clausius-Clapeyron equation.

    Uses the Magnus formula approximation which is accurate for temperatures
    between -40°C and +50°C. Formula: es = 6.112 * exp(17.67*T/(T+243.5))

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


def exner_function(pressure):
    """Calculates the Exner function of a parcel.

    Args:
        pressure: Pressure values in hPa

    Returns:
        Exner function values
    """
    return (pressure / p0) ** kappa


def get_pressure_height(pressure):
    """Calculates the pressure and height of a parcel.

    Args:
        pressure: Pressure values in hPa

    Returns:
        pressure: Pressure values in hPa
        height: Height values in m
    """
    pressure = np.atleast_1d(pressure)
    height = (t0 / gamma) * (1 - (pressure / p0) ** (Rd * gamma / g))
    return pressure, height


def potential_temperature(temperature, pressure, units="C"):
    """Calculates the potential temperature of a parcel.

    Args:
        temperature: Temperature values in Celsius
        pressure: Pressure values in hPa

    Returns:
        Potential temperature values in K
    """
    if units == "C":
        temperature = temperature + 273.15
    elif units == "K":
        pass
    else:
        raise ValueError(f"Unknown units: {units}")
    theta = temperature / exner_function(pressure)
    return theta


def dewpoint_from_vapor_pressure(vapor_pressure):
    """Calculates the dewpoint of a parcel.

    Args:
        vapor_pressure: Vapor pressure values in hPa

    Returns:
        Dewpoint values in C
    """
    # Suppress warnings for this calculation
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.log(vapor_pressure / sat_press_0c)
        return 243.5 * val / (17.67 - val)


def dry_lapse(pressure, temperature):
    """Calculates the temperature of a parcel given the dry adiabatic lapse rate.

    If pressure is a 1D array, the temperature is calculated for each pressure value.
    If pressure is a 2D array, the temperature is calculated for each pressure value in
    the first dimension, while the second dimension is broadcasted.
    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in C

    Returns:
        Temperature values in C
    """
    if pressure.ndim == 1:
        return temperature * (pressure / pressure[0]) ** kappa
    else:
        return temperature * (pressure / pressure[..., 0:1]) ** kappa


def saturation_mixing_ratio(pressure, temperature):
    """Calculates the saturation mixing ratio of a parcel.

    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in C

    Returns:
        Saturation mixing ratio values in kg/kg
    """
    return mixing_ratio(saturation_vapor_pressure(temperature), pressure)


def _lcl_iter(pressure, pressure_0, mixing_ratio, temperature, nan_mask_list):
    """Iterative function to calculate the LCL pressure."""
    td = (
        dewpoint_from_vapor_pressure(vapor_pressure(pressure / 100, mixing_ratio))
        + 273.15
    )
    pressure_new = pressure_0 * (td / temperature) ** (1.0 / kappa)
    nan_mask_list[0] = nan_mask_list[0] | np.isnan(pressure_new)

    return np.where(np.isnan(pressure_new), pressure, pressure_new)


def virtual_temperature_from_dewpoint(pressure, temperature, dewpoint):
    """Calculate virtual temperature from dewpoint.

    This function calculates virtual temperature from dewpoint, temperature, and
    pressure.

    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in Celsius
        dewpoint: Dewpoint values in Celsius

    Returns:
        Virtual temperature values in Celsius
    """

    # Convert dewpoint to mixing ratio
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    # Calculate virtual temperature with given parameters
    return virtual_temperature(temperature, mixing_ratio)


def virtual_temperature(temperature, mixing_ratio):
    """Calculates the virtual temperature of a parcel.

    Args:
        temperature: Temperature values in Celsius
        mixing_ratio: Mixing ratio values in kg/kg

    Returns:
        Virtual temperature values in Celsius
    """
    return temperature * ((mixing_ratio + epsilon) / (epsilon * (1 + mixing_ratio)))


# TODO: update to metpy 1.7 with direct solution
def lifting_condensation_level(pressure_prof, temp_prof, dew_prof):
    """
    Calculates the LCL pressure of a parcel.

    Args:
        pressure_prof: Pressure values in hPa
        temp_prof: Temperature values in C
        dew_prof: Dewpoint values in C

    Returns:
        LCL pressure values in hPa
        LCL dewpoint values in C
    """
    pressure_prof_pa = pressure_prof * 100  # convert to Pa
    temp_prof_k = temp_prof + 273.15  # convert to K
    # Handle nans by creating a mask that gets set by our _lcl_iter function if it
    # ever encounters a nan, at which point pressure is set to p, stopping iteration.
    nan_mask_list = np.isnan(pressure_prof)  # Use a mutable list to store the mask
    es = saturation_vapor_pressure(dew_prof)
    w = mixing_ratio(es, pressure_prof)
    try:
        lcl_p = so.fixed_point(
            _lcl_iter,
            pressure_prof_pa,
            args=(pressure_prof_pa, w, temp_prof_k, nan_mask_list),
            xtol=1e-5,
            maxiter=50,
        )
    except ValueError:
        lcl_p = np.zeros_like(pressure_prof_pa) * np.nan
    lcl_p = np.where(nan_mask_list[0], np.nan, lcl_p) / 100  # convert to hPa

    calculated_lcl_p = np.atleast_1d(
        np.where(np.isclose(lcl_p, pressure_prof), pressure_prof, lcl_p)
    )
    calculated_lcl_td = np.atleast_1d(
        dewpoint_from_vapor_pressure(vapor_pressure(lcl_p, w))
    )
    return calculated_lcl_p, calculated_lcl_td


def moist_air_specific_heat_pressure(specific_humidity_prof):
    """
    Calculates the specific heat of moist air at constant pressure.

    Args:
        specific_humidity_prof: Specific humidity values in kg/kg

    Returns:
        Specific heat of moist air at constant pressure in J/kg/K
    """
    return Cp_d + specific_humidity_prof * (Cp_v - Cp_d)


def moist_air_gas_constant(specific_humidity_prof):
    """
    Calculates the gas constant of moist air.

    Args:
        specific_humidity_prof: Specific humidity values in kg/kg

    Returns:
        Gas constant of moist air in J/kg/K
    """
    return Rd + specific_humidity_prof * (Rv - Rd)


def new_lcl(pressure_prof, temp_prof, dew_prof):
    """
    Calculates the LCL pressure of a parcel.

    Args:
        pressure_prof: Pressure values in hPa
        temp_prof: Temperature values in K
        dew_prof: Dewpoint values in K
        specific_humidity_prof: Specific humidity values in kg/kg

    Returns:
        LCL pressure values in hPa
        LCL temperature values in C
    """

    e = saturation_vapor_pressure(dew_prof - 273.15)
    es = saturation_vapor_pressure(temp_prof - 273.15)
    relative_humidity = e / es
    sat_mixing_ratio = saturation_mixing_ratio(
        pressure_prof, temp_prof - 273.15
    )  # convert to C for saturation mixing ratio
    specific_humidity_prof = sat_mixing_ratio / (1 + sat_mixing_ratio)

    moist_heat_ratio = moist_air_specific_heat_pressure(
        specific_humidity_prof
    ) / moist_air_gas_constant(specific_humidity_prof)
    spec_heat_diff = Cp_l - Cp_v

    a = moist_heat_ratio + spec_heat_diff / Rv
    b = -(Lv + spec_heat_diff * T0) / (Rv * temp_prof)
    c = b / a

    w_minus1 = lambertw((relative_humidity ** (1 / a) * c * np.exp(c)), k=-1).real

    t_lcl = c / w_minus1 * temp_prof
    p_lcl = pressure_prof * (t_lcl / temp_prof) ** moist_heat_ratio

    return p_lcl, t_lcl


def insert_lcl_level_fast(pressure, temperature, lcl_pressure):
    """Inserts the LCL pressure height into a temperature profile.

    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in C
        lcl_pressure: LCL pressure values in hPa

    Returns:
        Temperature values with LCL pressure height inserted
    """
    pressure_flat = pressure.reshape(-1, pressure.shape[-1])
    temperature_flat = temperature.reshape(-1, temperature.shape[-1])

    # Handle lcl_pressure dimensions - ensure it has the same number of dimensions as pressure
    # but with the last dimension being 1
    if lcl_pressure.ndim == pressure.ndim - 1:
        # lcl_pressure needs one more dimension
        lcl_pressure_expanded = np.expand_dims(lcl_pressure, axis=-1)
    elif lcl_pressure.ndim == pressure.ndim and lcl_pressure.shape[-1] == 1:
        # lcl_pressure already has the right shape
        lcl_pressure_expanded = lcl_pressure
    else:
        # Reshape lcl_pressure to match pressure shape except last dimension
        target_shape = list(pressure.shape[:-1]) + [1]
        lcl_pressure_expanded = lcl_pressure.reshape(target_shape)

    lcl_pressure_flat = lcl_pressure_expanded.reshape(-1, 1)

    # Insert calculated_lcl_p into pressure_prof and sort
    # First append the LCL pressure to get combined array
    combined_pressure = np.append(pressure, lcl_pressure_expanded, axis=-1)
    combined_pressure_flat = combined_pressure.reshape(-1, combined_pressure.shape[-1])
    combined_pressure_flat = np.sort(combined_pressure_flat, axis=-1)
    # Store the original indices of LCL pressure values
    lcl_indices = np.where(combined_pressure_flat == lcl_pressure_flat)

    # Vectorized interpolation using apply_ufunc
    def _interp_single_profile(combined_p, pressure_p, temperature_p):
        return np.interp(combined_p, pressure_p[::-1], temperature_p[::-1])

    combined_p_da = xr.DataArray(
        combined_pressure_flat,
        dims=[*[f"dim_{i}" for i in range(combined_pressure_flat.ndim - 1)], "level"],
    )
    pressure_da = xr.DataArray(
        pressure_flat,
        dims=[*[f"dim_{i}" for i in range(pressure_flat.ndim - 1)], "pressure"],
    )
    temperature_da = xr.DataArray(
        temperature_flat,
        dims=[*[f"dim_{i}" for i in range(temperature_flat.ndim - 1)], "pressure"],
    )

    interp_temp = xr.apply_ufunc(
        _interp_single_profile,
        combined_p_da,
        pressure_da,
        temperature_da,
        input_core_dims=[["level"], ["pressure"], ["pressure"]],
        output_core_dims=[["level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).values

    # Create a copy of temp_prof to modify
    calculated_new_temp_flat = np.copy(temperature_flat)

    # Insert the interpolated temperature at the LCL pressure point
    # Reshape arrays to handle insertion along the pressure dim
    orig_shape = calculated_new_temp_flat.shape
    new_shape = (orig_shape[0], orig_shape[1] + 1)
    result = np.zeros(new_shape) * np.nan
    result[lcl_indices] = interp_temp[lcl_indices]
    temperature_flat_w_nan = np.full(result.shape, np.nan)
    temperature_flat_w_nan[:, : orig_shape[1]] = temperature_flat[..., ::-1]

    # create a mask of the nans in the result with lcl_heights applied
    nan_mask = np.isnan(result)
    num_nans = np.sum(nan_mask, axis=1)

    # This is the replacement for the loop.
    # Create a 1D array representing column indices [0, 1, 2, ..., 33]
    col_indices = np.arange(result.shape[1])

    # Use broadcasting to compare each row's num_nans value against the column indices.
    # num_nans[:, np.newaxis] reshapes (204525,) to (204525, 1)
    # This comparison broadcasts to a (204525, 34) boolean mask.
    source_selection_mask = col_indices < num_nans[:, np.newaxis]

    values_to_assign = temperature_flat_w_nan[source_selection_mask]
    result[nan_mask] = values_to_assign

    return result.reshape(combined_pressure.shape)[..., ::-1]


def insert_lcl_level(pressure, temperature, lcl_pressure):
    """Inserts the LCL pressure height into a temperature profile.
    Deprecated in favor of insert_lcl_level_fast.
    Args:
        pressure: Pressure values in hPa
        temperature: Temperature values in C
        lcl_pressure: LCL pressure values in hPa

    Returns:
        Temperature values with LCL pressure height inserted
    """
    # Insert calculated_lcl_p into pressure_prof and sort
    # First append the LCL pressure to get combined array
    combined_pressure = np.append(pressure, lcl_pressure, axis=-1)
    # Get indices that would sort the array
    sort_indices = np.argsort(combined_pressure, axis=-1)[::-1]
    # Apply the sort while preserving the original indices
    combined_pressure = np.take_along_axis(combined_pressure, sort_indices, axis=-1)
    # Store the original indices of LCL pressure values
    lcl_indices = np.where(combined_pressure == lcl_pressure)

    # Vectorized interpolation using apply_ufunc
    def _interp_single_profile_3d(combined_p, pressure_p, temperature_p):
        return np.interp(combined_p, pressure_p[::-1], temperature_p[::-1])[::-1]

    combined_p_da = xr.DataArray(
        combined_pressure,
        dims=[*[f"dim_{i}" for i in range(combined_pressure.ndim - 1)], "level"],
    )
    pressure_da = xr.DataArray(
        pressure, dims=[*[f"dim_{i}" for i in range(pressure.ndim - 1)], "pressure"]
    )
    temperature_da = xr.DataArray(
        temperature,
        dims=[*[f"dim_{i}" for i in range(temperature.ndim - 1)], "pressure"],
    )

    interp_temp = xr.apply_ufunc(
        _interp_single_profile_3d,
        combined_p_da,
        pressure_da,
        temperature_da,
        input_core_dims=[["level"], ["pressure"], ["pressure"]],
        output_core_dims=[["level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).values
    # Find the index where calculated_lcl_p was inserted into combined_pressure

    # Create a copy of temp_prof to modify
    calculated_new_temp = np.copy(temperature)

    # Insert the interpolated temperature at the LCL pressure point
    # Reshape arrays to handle insertion along pressure dimension
    orig_shape = calculated_new_temp.shape
    new_shape = list(orig_shape)
    new_shape[-1] += 1  # Increase the last dimension by 1
    new_shape = tuple(new_shape)

    # Create output array with new shape
    result = np.zeros(new_shape) * np.nan
    # Check if we have enough dimensions and if all LCL indices are at position 0
    last_dim_index = len(lcl_indices) - 1
    if (
        last_dim_index >= 0
        and len(lcl_indices[last_dim_index]) > 0
        and all(lcl_indices[last_dim_index] == 0)
    ):
        result = np.insert(
            calculated_new_temp[..., ::-1], 0, interp_temp[..., ::-1][..., 0], axis=-1
        )
    else:
        # For multi-dimensional arrays, this deprecated function has complex logic
        # Since it's deprecated in favor of insert_lcl_level_fast,
        # we'll return the interpolated result as a fallback
        result = interp_temp

    calculated_new_temp = result[..., ::-1]
    return calculated_new_temp


def log_interpolate(x, xp, var):
    """
    Interpolates data with logarithmic x-scale over a specified axis.
    Assumes all inputs are in descending order and need to be reversed.

    Args:
        x: Desired interpolated values
        xp: x-coordinates of the data points
        var: Data to be interpolated
        axis: Axis to interpolate over

    Returns:
        Interpolated values
    """

    # Reverse and take log of x and xp
    x_log = np.log(x[::-1])
    xp_log = np.log(xp[::-1])

    # Create output array shape
    out_shape = list(var.shape)
    out_shape[-1] = len(x_log)

    # Reshape input to 2D array with interpolation axis last
    var_2d = np.array(var).reshape(-1, var.shape[-1])

    # Use scipy's interp1d for vectorized interpolation
    f = interp1d(xp_log, var_2d[..., ::-1], axis=-1, bounds_error=False)
    result_2d = f(x_log)

    # Reshape back to original dimensions
    result = result_2d.reshape(out_shape)
    # Swap back and reverse the result
    return result


def combine_profiles(
    calculated_press_lower,
    calculated_lcl_p,
    calculated_press_upper,
    calculated_temp_lower,
    calculated_lcl_td,
    calculated_temp_upper,
    axis=0,
):
    """Combine pressure and temperature profiles, handling empty arrays."""

    # Handle empty upper arrays by reshaping to match lower arrays
    if calculated_press_upper.size == 0:
        target_shape = list(calculated_press_lower.shape)
        target_shape[axis] = 0
        calculated_press_upper = np.empty(target_shape)
        calculated_temp_upper = np.empty(target_shape)

    calculated_new_pressure = np.concatenate(
        (
            calculated_press_lower,
            calculated_lcl_p,
            calculated_press_upper,
        ),
        axis=axis,
    )

    calculated_prof_dewpoint = np.concatenate(
        (
            calculated_temp_lower,
            calculated_lcl_td,
            calculated_temp_upper,
        ),
        axis=axis,
    )

    return calculated_new_pressure, calculated_prof_dewpoint


def mixed_parcel(
    ds: xr.Dataset,
    layer_depth: float = 100,
    temperature_units: str = "K",
):
    """Calculates the mixed parcel properties of a dataset.

    Args:
        ds: Dataset containing pressure, temperature, and dewpoint variables
        pressure_var: Name of the pressure variable in the dataset
        temperature_var: Name of the temperature variable in the dataset
        temperature_dewpoint_var: Name of the dewpoint variable in the dataset
        layer_depth: Depth of the mixed layer in hPa

    Returns:
        calculated_parcel_start_pressure: ndarray of the pressure at the start of the
        mixed layer
        calculated_parcel_temp: ndarray of the temperature of the mixed parcel
        calculated_parcel_dewpoint: ndarray of the dewpoint of the mixed parcel
    """

    theta = potential_temperature(
        ds["air_temperature"], ds["pressure"], units=temperature_units
    )
    # convert temperature to celsius
    es = saturation_vapor_pressure(ds["dewpoint_temperature"] - 273.15)
    mixing_ratio_g_g = mixing_ratio(es, ds["pressure"])
    # because pressure is the same across the domain, we can use a single column
    pressure = ds["level"]
    # begin mixed layer
    bottom_pressure, _ = get_pressure_height(np.atleast_1d(pressure[0]))
    top = bottom_pressure - layer_depth  # hPa
    top_pressure, _ = get_pressure_height(top)

    # Get the mask of pressures in between bottom (high) pressure and top (low) pressure
    pressure_mask = (pressure >= (top_pressure[0])) & (pressure <= (bottom_pressure[0]))
    p_interp = pressure[pressure_mask]

    if not np.any(np.isclose(top_pressure, p_interp)):
        p_interp = np.sort(np.append(p_interp, top_pressure))
    if not np.any(np.isclose(bottom_pressure, p_interp)):
        p_interp = np.sort(np.append(p_interp, bottom_pressure))
    else:
        p_interp = np.sort(p_interp)

    p_interp = p_interp[::-1]
    layer_depth = abs(p_interp[0] - p_interp[-1])
    mean_theta = _interp_integrate(pressure, p_interp, layer_depth, theta, axis=-1)
    mean_mixing_ratio = _interp_integrate(
        pressure, p_interp, layer_depth, mixing_ratio_g_g, axis=-1
    )
    calculated_parcel_start_pressure = pressure[0].values

    calculated_parcel_temp_kelvin = (mean_theta) * exner_function(
        calculated_parcel_start_pressure
    )
    vapor_pres = vapor_pressure(calculated_parcel_start_pressure, mean_mixing_ratio)
    calculated_parcel_dewpoint_kelvin = (
        dewpoint_from_vapor_pressure(vapor_pres) + 273.15
    )

    return (
        calculated_parcel_start_pressure,
        calculated_parcel_temp_kelvin,
        calculated_parcel_dewpoint_kelvin,
    )


def find_intersection(x, y1, y2):
    """
    Finds the intersection points of two y-arrays given a common x-array.

    Args:
      x: A 1D numpy array representing the common x-values.
      y1: A 1D numpy array representing the first set of y-values.
      y2: A 1D numpy array representing the second set of y-values.

    Returns:
      A tuple containing two numpy arrays:
        - x_intersection: x-values where the intersection occurs.
        - y_intersection: y-values at the intersection points.
      Returns (None, None) if no intersection is found.
    """

    # Find indices where y1 and y2 are equal
    indices = np.where(y1 == y2)[0]

    if len(indices) > 0:
        return x[indices], y1[indices]
    else:
        return None, None


# Next steps are to get lfc translated then keep moving in the cape/cin code.
# finished cleaning up parcel_profile_with_lcl...next is to reproduce find_intersections


def _next_non_masked_element(a, idx):
    """Return the next non masked element of a masked array.

    If an array is masked, return the next non-masked element (if the given index is
    masked).
    If no other unmasked points are after the given masked point, returns none.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values
    idx : integer
        Index of requested element

    Returns
    -------
        Index of next non-masked element and next non-masked element

    """
    try:
        next_idx = idx + a[idx:].mask.argmin()
        if ma.is_masked(a[next_idx]):
            return None, None
        else:
            return next_idx, a[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, a[idx]


def find_intersections(x, y1, y2, direction="all"):
    """Finds the intersection points of two y-arrays, given their x-arrays.

    Args:
        x (array-like): x-coordinates of the first curve.
        y1 (array-like): y-coordinates of the first curve.
        y2 (array-like): y-coordinates of the second curve.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - intersect_x: x-coordinates of the intersection points.
    """
    x = np.log(x)
    diff = y1 - y2

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    closest_idx = np.nonzero(np.diff(np.sign(diff)))
    next_idx = closest_idx[0] + 1
    sign_change = np.sign(y1[next_idx] - y2[next_idx])
    # x-values around each intersection
    _, x0 = _next_non_masked_element(x, closest_idx)
    _, x1 = _next_non_masked_element(x, next_idx)

    # y-values around each intersection for the first line
    _, y10 = _next_non_masked_element(y1, closest_idx)
    _, y11 = _next_non_masked_element(y1, next_idx)

    # y-values around each intersection for the second line
    _, y20 = _next_non_masked_element(y2, closest_idx)
    _, y21 = _next_non_masked_element(y2, next_idx)
    # Calculate the x-intersection. This comes from finding the equations of the two
    # lines, one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1,
    # b1), finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = y10 - y20
    delta_y1 = y11 - y21
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points. One could solve for y like x above, but this
    # causes weirder unit behavior and seems a little less good numerically.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (y11 - y10) + y10
    # If there's no intersections, return
    if len(intersect_x) == 0:
        return intersect_x, intersect_y

    intersect_x = np.exp(intersect_x)

    # Check for duplicates
    duplicate_mask = np.ediff1d(intersect_x, to_end=1) != 0

    # Make a mask based on the direction of sign change desired
    if direction == "increasing":
        mask = sign_change > 0
    elif direction == "decreasing":
        mask = sign_change < 0
    elif direction == "all":
        return intersect_x[duplicate_mask], intersect_y[duplicate_mask]
    else:
        raise ValueError(f"Unknown option for direction: {direction}")

    return intersect_x[mask & duplicate_mask], intersect_y[mask & duplicate_mask]


def equilibrium_level(pressure, temperature, dewpoint, parcel_profile):
    """Finds the equilibrium level of the parcel profile.

    Args:
        pressure: numpy array of pressure values in hPa
        temperature: numpy array of temperature values in C
        dewpoint: numpy array of dewpoint values in C
        parcel_profile: numpy array of parcel profile values in C

    Returns:
        x: numpy array of EL pressure values
        y: numpy array of EL temperature values in C
    """

    if pressure.ndim == 1:
        if parcel_profile[-1] > temperature[-1]:
            return np.nan, np.nan
    x, y = find_intersections(
        pressure[1:], parcel_profile[1:], temperature[1:], direction="decreasing"
    )
    lcl_pressure, _ = new_lcl(
        pressure[0], temperature[0] + 273.15, dewpoint[0] + 273.15
    )  # new lcl function takes kelvin
    if len(x) > 0 and x[-1] < lcl_pressure:
        idx = x < lcl_pressure
        x, y = x[idx][-1], y[idx][-1]
        return x, y
    else:
        return np.nan, np.nan


def level_free_convection(pressure, temperature, dewpoint, parcel_profile):
    """
    Finds the LFC of the parcel profile.
    Args:
        pressure: numpy array of pressure values in hPa
        temperature: numpy array of temperature values in C
        dewpoint: numpy array of dewpoint values in C
        parcel_profile: numpy array of parcel profile values in C
    Returns:
        x: numpy array of LFC pressure values
    """
    x, y = find_intersections(
        pressure, parcel_profile, temperature, direction="increasing"
    )
    lcl_pressure, lcl_temperature_c = new_lcl(
        pressure[0], parcel_profile[0] + 273.15, dewpoint[0] + 273.15
    )
    if len(x) == 0:
        if np.all(pressure < lcl_pressure):
            return np.nan, np.nan
        else:
            x, y = lcl_pressure, lcl_temperature_c
        return x, y
    else:
        idx = x < lcl_pressure
        if not any(idx):
            el_pressure, _ = find_intersections(
                pressure[1:],
                parcel_profile[1:],
                temperature[1:],
                direction="decreasing",
            )
            if len(el_pressure) > 0 and np.min(el_pressure) > lcl_pressure:
                return np.nan, np.nan
            else:
                x, y = lcl_pressure, lcl_temperature_c
                return x, y
        else:
            x, y = x[idx][0], y[idx][0]
            return x, y


def _cape_cin_single_profile(
    pressure_profile, temperature_profile, dewpoint_profile, parcel_profile_profile
):
    """Calculate CAPE and CIN for a single atmospheric profile.

    This function is designed to be called by apply_ufunc for vectorization.
    """
    # Initialize outputs
    cape = 0.0
    cin = 0.0

    # Get LFC
    lfc_pressure, _ = level_free_convection(
        pressure_profile,
        temperature_profile,
        dewpoint_profile,
        parcel_profile_profile,
    )

    if np.isnan(lfc_pressure):
        return cape, cin

    # Get EL
    el_pressure, _ = equilibrium_level(
        pressure_profile,
        temperature_profile,
        dewpoint_profile,
        parcel_profile_profile,
    )

    # Calculate temperature difference
    y = parcel_profile_profile - temperature_profile

    # Find intersections
    x_crossing, y_crossing = find_intersections(
        pressure_profile[1:], y[1:], np.zeros_like(y[1:]), direction="all"
    )

    # Combine pressure and intersection points
    x = np.concatenate([np.atleast_1d(pressure_profile), x_crossing])
    y = np.concatenate([np.atleast_1d(y), y_crossing])

    # Sort and remove duplicates
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    keep_idx = np.ediff1d(x, to_end=[1]) > 1e-6
    x = x[keep_idx]
    y = y[keep_idx]

    # Calculate CAPE
    cape_mask = ((x < lfc_pressure) | np.isclose(x, lfc_pressure)) & (
        (x > el_pressure) | np.isclose(x, el_pressure)
    )
    x_clipped = x[cape_mask]
    y_clipped = y[cape_mask]
    if len(x_clipped) > 0:
        cape = Rd * np.trapezoid(y_clipped, np.log(x_clipped))
        cape = max(0, cape)  # Set CAPE to 0 if negative

    # Calculate CIN
    cin_mask = (x > lfc_pressure) | np.isclose(x, lfc_pressure)
    x_clipped = x[cin_mask]
    y_clipped = y[cin_mask]
    if len(x_clipped) > 0:
        cin = Rd * np.trapezoid(y_clipped, np.log(x_clipped))
        cin = min(0, cin)  # Set CIN to 0 if positive

    return cape, cin


def mlcape_cin(pressure, temperature, dewpoint, parcel_profile):
    """Calculates the convective available potential energy (CAPE) and convective
    inhibition (CIN) of a dataset.

    Args:
        pressure: numpy array of pressure values in hPa
        temperature: numpy array of temperature values in C
        dewpoint: numpy array of dewpoint values in C
        parcel_profile: numpy array of parcel profile values in C

    Returns:
        cape: numpy array of CAPE values in J/kg
        cin: numpy array of CIN values in J/kg
    """
    # Handle empty arrays
    if pressure.size == 0 or pressure.shape[-1] == 0:
        target_shape = pressure.shape[:-1] if pressure.size > 0 else (1,)
        return np.full(target_shape, np.nan), np.full(target_shape, np.nan)

    # Get the LCL pressure for each profile
    pressure_lcl, _ = new_lcl(
        pressure[..., 0], temperature[..., 0] + 273.15, dewpoint[..., 0] + 273.15
    )  # new lcl function takes kelvin
    below_lcl = pressure > np.expand_dims(pressure_lcl, axis=-1)
    parcel_mixing_ratio = np.where(
        below_lcl,
        saturation_mixing_ratio(pressure, dewpoint),
        saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    tv_td = virtual_temperature_from_dewpoint(pressure, temperature, dewpoint)
    tv_parcel_profile = virtual_temperature(parcel_profile.copy(), parcel_mixing_ratio)

    # Convert to xarray DataArrays for apply_ufunc
    pressure_da = xr.DataArray(
        pressure, dims=[*[f"dim_{i}" for i in range(pressure.ndim - 1)], "pressure"]
    )
    tv_td_da = xr.DataArray(
        tv_td, dims=[*[f"dim_{i}" for i in range(tv_td.ndim - 1)], "pressure"]
    )
    dewpoint_da = xr.DataArray(
        dewpoint, dims=[*[f"dim_{i}" for i in range(dewpoint.ndim - 1)], "pressure"]
    )
    tv_parcel_profile_da = xr.DataArray(
        tv_parcel_profile,
        dims=[*[f"dim_{i}" for i in range(tv_parcel_profile.ndim - 1)], "pressure"],
    )

    # Use xarray.apply_ufunc for vectorization
    cape, cin = xr.apply_ufunc(
        _cape_cin_single_profile,
        pressure_da,
        tv_td_da,
        dewpoint_da,
        tv_parcel_profile_da,
        input_core_dims=[["pressure"], ["pressure"], ["pressure"], ["pressure"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    return cape.values, cin.values


def dewpoint_from_specific_humidity(
    specific_humidity: xr.DataArray, pressure: xr.DataArray
) -> xr.DataArray:
    """Calculate dewpoint from specific humidity and pressure.

    The pressure DataArray must be the same shape as the specific humidity DataArray.

    Args:
        specific_humidity: The specific humidity in kg/kg.
        pressure: The pressure in hPa.

    Returns:
        The dewpoint DataArray in Celsius.
    """
    mixing_ratio = specific_humidity / (1 - specific_humidity)
    e = pressure * mixing_ratio / (epsilon + mixing_ratio)

    return dewpoint_from_vapor_pressure(e) + 273.15


def craven_brooks_significant_severe(
    ds: xr.Dataset,
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
        ds: xarray Dataset containing 3D atmospheric data with pressure levels.
        pressure_var: Name of pressure variable in ds (hPa).
        temperature_var: Name of temperature variable in ds (°C).
        temperature_dewpoint_var: Name of dewpoint temperature variable in ds (°C).
        eastward_wind_var: Name of u-component wind variable (m/s).
        northward_wind_var: Name of v-component wind variable (m/s).
        surface_eastward_wind_var: Name of surface u-component wind (m/s).
        surface_northward_wind_var: Name of surface v-component wind (m/s).
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
    # Check for prerequisites to ensure successful execution
    ds = _basic_ds_checks(ds)
    # CIN not needed for CBSS
    cape, _ = mixed_layer_cape_cin(
        ds,
        layer_depth,
    )
    shear = low_level_shear(
        ds,
    )
    cbss = cape * shear
    return cbss


def low_level_shear(
    ds: xr.Dataset,
) -> xr.DataArray:
    """Calculates the low level (0-6 km) shear of a dataset (Lepore et al 2021).

    Args:
        ds: Dataset containing eastward and northward (u and v) wind vectors

    Returns:
        ll_shear: ndarray of low level shear values in m/s
    """
    ll_shear = np.sqrt(
        (ds["eastward_wind"].sel(level=500) - ds["surface_eastward_wind"]) ** 2
        + (ds["northward_wind"].sel(level=500) - ds["surface_northward_wind"]) ** 2
    )
    return ll_shear


def mixed_layer_cape_cin(
    ds: xr.Dataset,
    layer_depth: float = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mixed layer CAPE and CIN for severe weather forecasting.

    Computes Convective Available Potential Energy (CAPE) and Convective Inhibition
    (CIN) by lifting a mixed layer parcel through the atmosphere. The mixed layer
    is defined as the average properties over a specified layer_depth from the surface.

    Uses optimized lookup tables for moist pseudoadiabatic calculations to handle
    large gridded datasets efficiently. The calculation follows these steps:
    1. Create a mixed layer parcel from the specified layer_depth
    2. Lift the parcel dry adiabatically to the LCL
    3. Lift moist adiabatically above the LCL using lookup tables
    4. Calculate CAPE (positive area) and CIN (negative area) between parcel
       and environment temperatures

    Args:
        ds: xarray Dataset containing 3D atmospheric profile data.
            Must have pressure levels as the last dimension.
        pressure_var: Name of pressure variable in ds (hPa).
        temperature_var: Name of temperature variable in ds (°C).
        temperature_dewpoint_var: Name of dewpoint temperature variable in ds (°C).
        layer_depth: Mixed layer layer_depth in hPa (typically 50-100 hPa for severe
        weather).

    Returns:
        tuple[np.ndarray, np.ndarray]: CAPE and CIN values both in J/kg.
            - CAPE: Positive values indicating convective potential
            - CIN: Negative values indicating convective inhibition

    Notes:
        - CAPE > 2500 J/kg indicates strong convective potential
        - CAPE > 4000 J/kg indicates extreme convective potential
        - CIN < -100 J/kg can suppress convection initiation
        - Mixed layer depth of 100 hPa is standard for severe weather applications
        - Requires pressure levels in descending order (surface to top)

    References:
        Doswell, C. A., and E. N. Rasmussen, 1994: The effect of neglecting the
        virtual temperature correction on CAPE calculations. Wea. Forecasting,
        9, 625–629.
    """

    # Check if we need to combine time dimensions
    # Expected: (time_dims..., latitude, longitude, level)
    # Need to merge all time dims into single dimension for processing

    # Identify spatial and level dimensions
    spatial_dims = ["latitude", "longitude", "lat", "lon", "y", "x"]
    found_spatial = [dim for dim in ds.dims if dim in spatial_dims]

    # All dimensions except spatial and level are considered time dimensions
    time_dims = [dim for dim in ds.dims if dim != "level" and dim not in found_spatial]

    # If we have multiple time dimensions, we need to stack them
    if len(time_dims) > 1:
        # Store original shapes for reshaping results
        original_time_shape = tuple(ds.sizes[dim] for dim in time_dims)
        spatial_shape = tuple(ds.sizes[dim] for dim in found_spatial)

        # Stack all time dimensions into single 'time' dimension
        stacked_ds = ds.stack(time=time_dims)

        # Run the calculation on the stacked dataset
        cape_flat, cin_flat = _run_cape_calculation(stacked_ds, layer_depth)

        # Reshape results back to original time structure
        # cape_flat shape: (time, spatial...) -> (time_dims..., spatial...)
        cape_reshaped = cape_flat.reshape(original_time_shape + spatial_shape)
        cin_reshaped = cin_flat.reshape(original_time_shape + spatial_shape)

        return cape_reshaped, cin_reshaped

    # Single or no time dimension - run calculation directly
    return _run_cape_calculation(ds, layer_depth)


def _run_cape_calculation(
    ds: xr.Dataset, layer_depth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run the core CAPE calculation on a dataset with standard dimensions."""
    ds = _basic_ds_checks(ds)
    pressure = ds["level"]
    mixed_layer_mask = ds["pressure"] < (pressure[0] - layer_depth)
    # Get the indices where the condition is True along the last dimension
    valid_indices = np.any(
        mixed_layer_mask, axis=tuple(range(mixed_layer_mask.ndim - 1))
    )

    (
        calculated_parcel_start_pressure,
        calculated_parcel_temp,
        calculated_parcel_dewpoint,
    ) = mixed_parcel(ds, layer_depth)
    parcel_temp_reshaped = np.expand_dims(calculated_parcel_temp, axis=-1)
    parcel_dewpoint_reshaped = np.expand_dims(calculated_parcel_dewpoint, axis=-1)

    # Extract valid pressure, temperature and dewpoint profiles
    pressure_prof = ds["pressure"][..., valid_indices]
    temp_prof = ds["air_temperature"][..., valid_indices]
    dew_prof = ds["dewpoint_temperature"][..., valid_indices]
    # Concatenate the mixed parcel properties with the profiles
    parcel_start_pressure_reshaped = np.full(
        (*pressure_prof.shape[:-1], 1),
        np.atleast_1d(calculated_parcel_start_pressure)[0],
    )

    # Now concatenate along the first dimension
    pressure_prof = np.concatenate(
        [parcel_start_pressure_reshaped, pressure_prof], axis=-1
    )
    temp_prof = np.concatenate([parcel_temp_reshaped, temp_prof], axis=-1)
    dew_prof = np.concatenate([parcel_dewpoint_reshaped, dew_prof], axis=-1)
    calculated_lcl_pressure, calculated_lcl_td = new_lcl(
        pressure_prof[..., 0], temp_prof[..., 0], dew_prof[..., 0]
    )
    calculated_lcl_pressure = np.expand_dims(calculated_lcl_pressure, axis=-1)
    calculated_lcl_td = np.expand_dims(calculated_lcl_td, axis=-1)

    # Create profiles at or below LCL
    at_or_below_lcl_pressure_mask = pressure_prof >= calculated_lcl_pressure
    pressure_prof_at_or_below_lcl = np.empty_like(pressure_prof) * np.nan
    pressure_prof_at_or_below_lcl[at_or_below_lcl_pressure_mask] = pressure_prof[
        at_or_below_lcl_pressure_mask
    ]
    pressure_prof_at_or_below_lcl = pressure_prof_at_or_below_lcl[
        ...,
        np.any(
            ~np.isnan(pressure_prof_at_or_below_lcl),
            axis=tuple(range(0, pressure_prof_at_or_below_lcl.ndim - 1)),
        ),
    ]
    pressure_prof_at_or_below_lcl = np.concatenate(
        (pressure_prof_at_or_below_lcl, calculated_lcl_pressure), axis=-1
    )

    temp_prof_at_or_below_lcl = dry_lapse(
        pressure_prof_at_or_below_lcl,
        np.expand_dims(temp_prof[..., 0], axis=-1),
    )

    # Create profiles above LCL
    above_lcl_pressure_mask = pressure_prof < calculated_lcl_pressure
    pressure_prof_above_lcl = np.empty_like(pressure_prof) * np.nan
    pressure_prof_above_lcl[above_lcl_pressure_mask] = pressure_prof[
        above_lcl_pressure_mask
    ]
    pressure_prof_above_lcl = pressure_prof_above_lcl[
        ...,
        np.any(
            ~np.isnan(pressure_prof_above_lcl),
            axis=tuple(range(0, pressure_prof_above_lcl.ndim - 1)),
        ),
    ]

    temp_above_lcl = (
        moist_lapse_lookup(
            pressure_prof_above_lcl,
            temp_prof_at_or_below_lcl[..., -1] - 273.15,
            reference_pressure=calculated_lcl_pressure,
        )
        + 273.15
    )
    # Combine profiles at or below LCL and above LCL
    combined_all_pressure_w_lcl, combined_all_temp_w_lcl = combine_profiles(
        pressure_prof_at_or_below_lcl[..., :-1],
        calculated_lcl_pressure,
        pressure_prof_above_lcl,
        temp_prof_at_or_below_lcl[..., :-1],
        calculated_lcl_td,
        temp_above_lcl,
        axis=-1,
    )

    # Insert LCL level into profiles
    calculated_new_temp = insert_lcl_level_fast(
        pressure_prof, temp_prof, calculated_lcl_pressure
    )
    # calculated_new_temp = insert_lcl_level(
    #     pressure_prof, temp_prof, calculated_lcl_pressure
    # )
    calculated_new_dewpoint = insert_lcl_level_fast(
        pressure_prof, dew_prof, calculated_lcl_pressure
    )
    # calculated_new_dewpoint = insert_lcl_level(
    #     pressure_prof, dew_prof, calculated_lcl_pressure
    # )

    # Get unique values and indices for pressure array
    orig_shape = combined_all_pressure_w_lcl.shape
    reshaped_pressures = combined_all_pressure_w_lcl[..., ::-1].reshape(
        -1, orig_shape[-1]
    )
    reshaped_temps = combined_all_temp_w_lcl[..., ::-1].reshape(-1, orig_shape[-1])

    # Reshape array to 2D for unique operation, then reshape back
    unique_pressures, unique_indices = np.unique(
        reshaped_pressures.round(decimals=4), return_index=True, axis=-1, equal_nan=True
    )
    combined_all_pressure_w_lcl = unique_pressures.reshape(*orig_shape[:-1], -1)

    # Use the same indices to select corresponding temperature values
    unique_temps = reshaped_temps[..., unique_indices]
    combined_all_temp_w_lcl = unique_temps.reshape(*orig_shape[:-1], -1)

    # Sort profiles by pressure
    sorted_indices = np.argsort(combined_all_pressure_w_lcl, axis=-1)
    combined_all_pressure_w_lcl = np.take_along_axis(
        combined_all_pressure_w_lcl, sorted_indices, axis=-1
    )
    combined_all_temp_w_lcl = np.take_along_axis(
        combined_all_temp_w_lcl, sorted_indices, axis=-1
    )

    # Find indices where all values are NaN in the last dimension
    nan_mask_pressure = np.all(
        np.isnan(combined_all_pressure_w_lcl),
        axis=tuple(range(combined_all_pressure_w_lcl.ndim - 1)),
    )
    nan_mask_temp = np.all(
        np.isnan(combined_all_temp_w_lcl),
        axis=tuple(range(combined_all_temp_w_lcl.ndim - 1)),
    )

    # Combine masks - if either pressure or temp is all NaN, we want to drop that row
    combined_nan_mask = np.logical_or(nan_mask_pressure, nan_mask_temp)

    # Create new arrays without the all-NaN rows
    combined_all_pressure_w_lcl = combined_all_pressure_w_lcl[..., ~combined_nan_mask][
        ..., ::-1
    ]
    combined_all_temp_w_lcl = combined_all_temp_w_lcl[..., ~combined_nan_mask][
        ..., ::-1
    ]

    # Finally, calculate CAPE and CIN
    # Check if we have valid data to work with
    if (
        combined_all_pressure_w_lcl.size == 0
        or combined_all_pressure_w_lcl.shape[-1] == 0
    ):
        # Return NaN for invalid cases with appropriate shape
        target_shape = calculated_new_temp.shape[:-1]
        return np.full(target_shape, np.nan), np.full(target_shape, np.nan)

    # Convert temps back to Celsius
    cape, cin = mlcape_cin(
        combined_all_pressure_w_lcl,
        calculated_new_temp - 273.15,
        calculated_new_dewpoint - 273.15,
        combined_all_temp_w_lcl - 273.15,
    )
    return cape, cin
