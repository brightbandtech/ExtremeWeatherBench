"""
This module contains constants and functions that are part of derived variables.
Most functions are adapted from the MetPy library (https://github.com/Unidata/MetPy).
"""

import itertools

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.optimize as so
import xarray as xr
from scipy.interpolate import interp1d
from scipy.special import lambertw

gamma = 6.5  # K/km
p0 = 1000  # hPa
p0_stp = 1013.25  # hPa
t0 = 288.0  # K
Rd = 287.04749097718457  # J/kg/K
depth = 100  # hPa
epsilon = 0.6219569100577033
sat_press_0c = 6.112  # hPa
kappa = 0.28571428571428564
g = 9.81  # m/s^2
Lv = 2500840  # J/kg
Cp_d = 1004.6662184201462  # J/kgK
R = 8.314462618  # J/mol/K
Mw = 18.015268  # g/mol
Rv = (R / Mw) * 1000  # J/kg/K
Cp_l = 4219.4  # J/g
Cp_v = 1860.078011865639  # J/kg
T0 = 273.15  # K


def load_moist_lapse_lookup():
    """Load the moist lapse lookup table."""
    import os

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the lookup table
    lookup_path = os.path.join(current_dir, "moist_lapse_lookup.parq")
    moist_lapse_lookup_df = pd.read_parquet(lookup_path)
    return moist_lapse_lookup_df


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
    if target_temp.ndim > 1:
        target_temp = target_temp.flatten()
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
    closest_col_indices = np.array(
        [
            abs(temps_at_pressure.iloc[i].values - temp).argsort()[0]
            for i, temp in enumerate(target_temp.flatten())
        ]
    )

    # Get the corresponding temperature profiles
    profiles = np.array(
        [moist_lapse_lookup_df.iloc[:, idx].values for idx in closest_col_indices]
    )
    profiles = profiles.reshape(*target_temp.shape, -1)
    # Interpolate and reshape profiles to match target_pressure_reshaped levels
    interpolated_profiles = np.array(
        [
            np.interp(
                target_pressure_reshaped[i][::-1],
                moist_lapse_lookup_df.index[::-1],
                profiles[i][::-1],
            )[::-1]
            for i in range(target_pressure_reshaped.shape[0])
        ]
    ).reshape(target_pressure.shape)

    # remove values where nans exist in target_pressure_reshaped
    interpolated_profiles = np.where(
        np.isnan(target_pressure), np.nan, interpolated_profiles
    )
    return interpolated_profiles


def mixing_ratio(partial_pressure, total_pressure):
    """Calculates the mixing ratio of a parcel.

    Args:
        partial_press: Partial pressure values
        total_press: Total pressure values

    Returns:
        Mixing ratio values in kg/kg
    """
    return epsilon * partial_pressure / (total_pressure - partial_pressure)


def vapor_pressure(pressure, mixing_ratio):
    """Calculates the vapor pressure of a parcel.

    Args:
        pressure: Pressure values
        mixing_ratio: Mixing ratio values in kg/kg

    Returns:
        Vapor pressure values in provided pressure units
    """
    return pressure * mixing_ratio / (epsilon + mixing_ratio)


def saturation_vapor_pressure(temperature):
    """Calculates the saturation vapor pressure of a parcel.

    Args:
        temperature: Temperature values in Celsius

    Returns:
        Saturation vapor pressure values in hPa
    """
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
    val = np.log(vapor_pressure / sat_press_0c)
    return 243.5 * val / (17.67 - val)


def dry_lapse(pressure, temperature):
    """Calculates the temperature of a parcel given the dry adiabatic lapse rate.

    If pressure is a 1D array, the temperature is calculated for each pressure value.
    If pressure is a 2D array, the temperature is calculated for each pressure value in the first dimension,
    while the second dimension is broadcasted.
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

    This function calculates virtual temperature from dewpoint, temperature, and pressure.
    """

    # Convert dewpoint to mixing ratio
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    # Calculate virtual temperature with given parameters
    return virtual_temperature(temperature, mixing_ratio)


def virtual_temperature(temperature, mixing_ratio):
    """Calculates the virtual temperature of a parcel.

    Args:
        temperature: Temperature values in K
        mixing_ratio: Mixing ratio values in kg/kg

    Returns:
        Virtual temperature values in K
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
    lcl_pressure_flat = lcl_pressure.reshape(-1, lcl_pressure.shape[-1])
    # Insert calculated_lcl_p into pressure_prof and sort
    # First append the LCL pressure to get combined array
    combined_pressure = np.append(pressure, lcl_pressure, axis=-1)
    combined_pressure_flat = combined_pressure.reshape(-1, combined_pressure.shape[-1])
    combined_pressure_flat = np.sort(combined_pressure_flat, axis=-1)
    # Store the original indices of LCL pressure values
    lcl_indices = np.where(combined_pressure_flat == lcl_pressure_flat)
    interp_temp = np.array(
        [
            np.interp(
                combined_pressure_flat[i],
                pressure_flat[i][::-1],
                temperature_flat[i][::-1],
            )
            for i in range(pressure_flat.shape[0])
        ]
    ).reshape(combined_pressure_flat.shape)

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

    interp_temp = np.array(
        [
            np.interp(
                combined_pressure[i, j, k],
                pressure[i, j, k][::-1],
                temperature[i, j, k][::-1],
            )[::-1]
            for i in range(pressure.shape[0])
            for j in range(pressure.shape[1])
            for k in range(pressure.shape[2])
        ]
    ).reshape(combined_pressure.shape)
    # Find the index where calculated_lcl_p was inserted into combined_pressure

    # Create a copy of temp_prof to modify
    calculated_new_temp = np.copy(temperature)

    # Insert the interpolated temperature at the LCL pressure point
    # Reshape arrays to handle insertion along pressure dimension
    orig_shape = calculated_new_temp.shape
    new_shape = (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3] + 1)

    # Create output array with new shape
    result = np.zeros(new_shape) * np.nan
    if all(lcl_indices[3] == 0):
        result = np.insert(
            calculated_new_temp[..., ::-1], 0, interp_temp[..., ::-1][..., 0], axis=-1
        )
    else:
        # For each x,y,z coordinate, insert the interpolated temperature at the LCL pressure point
        for i, j, k in itertools.product(
            range(orig_shape[0]), range(orig_shape[1]), range(orig_shape[2])
        ):
            if not np.isnan(calculated_new_temp[i, j, k]).all():
                # Get the LCL index for this x,y,z coordinate
                lcl_idx = lcl_indices[3][
                    np.where(
                        (lcl_indices[0] == i)
                        & (lcl_indices[1] == j)
                        & (lcl_indices[2] == k)
                    )[0][0]
                ]
                # Insert the interpolated temperature
                result[i, j, k] = np.insert(
                    calculated_new_temp[..., ::-1][i, j, k],
                    lcl_idx,
                    interp_temp[..., ::-1][i, j, k, lcl_idx],
                )
            else:
                # in the case where there's missing data, the result will be nans
                result[i, j, k] = np.zeros_like(calculated_new_temp[i, j, k]) * np.nan

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
    calculated_new_pressure = np.concatenate(
        (
            np.atleast_1d(calculated_press_lower),
            np.atleast_1d(calculated_lcl_p),
            np.atleast_1d(calculated_press_upper),
        ),
        axis=axis,
    )

    calculated_prof_dewpoint = np.concatenate(
        (
            np.atleast_1d(calculated_temp_lower),
            np.atleast_1d(calculated_lcl_td),
            np.atleast_1d(calculated_temp_upper),
        ),
        axis=axis,
    )

    return calculated_new_pressure, calculated_prof_dewpoint


def _basic_ds_checks(ds: xr.Dataset):
    """
    Checks the dataset for basic issues that could cause problems with the CAPE/CIN calculation.
    """

    # make sure the pressure level is descending. If not, sort it
    if ds["level"][0] < ds["level"][-1]:
        ds = ds.sortby("level", ascending=False)
        for var in ds.data_vars:
            if "level" in ds[var].dims:
                ds[var] = ds[var].sortby("level", ascending=False)

    # Make sure level is the last dimension. If not, transpose it to the end.
    if "level" in ds.dims and list(ds.dims).index("level") != len(ds.dims) - 1:
        # Get all dimensions and move level to the end
        dims = list(ds.dims)
        dims.remove("level")
        dims.append("level")
        ds = ds.transpose(*dims)

    # Make sure the level is at least 50 hPa. If not, drop the levels below 50 hPa
    # Calculations for CAPE (e.g. virtual temperature) are less relevant above the troposphere
    if any(ds["level"] < 50):
        ds = ds.sel(level=slice(ds["level"].max(), 50))
    return ds


def mixed_parcel(
    ds: xr.Dataset,
    pressure_var: str = "pressure",
    temperature_var: str = "temperature",
    temperature_dewpoint_var: str = "dewpoint",
    depth: float = 100,
    temperature_units: str = "K",
):
    """Calculates the mixed parcel properties of a dataset.

    Args:
        ds: Dataset containing pressure, temperature, and dewpoint variables
        pressure_var: Name of the pressure variable in the dataset
        temperature_var: Name of the temperature variable in the dataset
        temperature_dewpoint_var: Name of the dewpoint variable in the dataset
        depth: Depth of the mixed layer in hPa

    Returns:
        calculated_parcel_start_pressure: ndarray of the pressure at the start of the mixed layer
        calculated_parcel_temp: ndarray of the temperature of the mixed parcel
        calculated_parcel_dewpoint: ndarray of the dewpoint of the mixed parcel
    """

    theta = potential_temperature(
        ds[temperature_var], ds[pressure_var], units=temperature_units
    )
    # convert temperature to celsius
    es = saturation_vapor_pressure(ds[temperature_dewpoint_var] - 273.15)
    mixing_ratio_g_g = mixing_ratio(es, ds[pressure_var])
    # because pressure is the same across the domain, we can use a single column
    pressure = ds["level"]
    # begin mixed layer
    bottom_pressure, _ = get_pressure_height(np.atleast_1d(pressure[0]))
    top = bottom_pressure - depth  # hPa
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

    If an array is masked, return the next non-masked element (if the given index is masked).
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
    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
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
            if np.min(el_pressure) > lcl_pressure:
                return np.nan, np.nan
            else:
                x, y = lcl_pressure, lcl_temperature_c
                return x, y
        else:
            x, y = x[idx][0], y[idx][0]
            return x, y


def mlcape_cin(pressure, temperature, dewpoint, parcel_profile):
    """Calculates the convective available potential energy (CAPE) and convective inhibition (CIN) of a dataset.

    Args:
        pressure: numpy array of pressure values in hPa
        temperature: numpy array of temperature values in C
        dewpoint: numpy array of dewpoint values in C
        parcel_profile: numpy array of parcel profile values in C

    Returns:
        cape: numpy array of CAPE values in J/kg
        cin: numpy array of CIN values in J/kg
    """
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

    pressure_flat = pressure.reshape(-1, pressure.shape[-1])
    temperature_flat = tv_td.reshape(-1, tv_td.shape[-1])
    dewpoint_flat = dewpoint.reshape(-1, dewpoint.shape[-1])
    parcel_profile_flat = tv_parcel_profile.reshape(-1, tv_parcel_profile.shape[-1])
    cape_flat = np.zeros(len(pressure_flat))
    cin_flat = np.zeros(len(pressure_flat))
    # Loop through each profile and calculate CAPE and CIN
    # TODO: Determine if there's a way to vectorize this
    for individual_profile in range(len(pressure_flat)):
        pressure_individual = pressure_flat[individual_profile, :]
        temperature_individual = temperature_flat[individual_profile]
        dewpoint_individual = dewpoint_flat[individual_profile]
        parcel_profile_individual = parcel_profile_flat[individual_profile]
        lfc_pressure, _ = level_free_convection(
            pressure_individual,
            temperature_individual,
            dewpoint_individual,
            parcel_profile_individual,
        )
        if np.isnan(lfc_pressure):
            # return a cape and cin of 0
            cape_flat[individual_profile] = 0
            cin_flat[individual_profile] = 0
            continue

        el_pressure, _ = equilibrium_level(
            pressure_individual,
            temperature_individual,
            dewpoint_individual,
            parcel_profile_individual,
        )
        y = parcel_profile_individual - temperature_individual

        x_crossing, y_crossing = find_intersections(
            pressure_individual[1:], y[1:], np.zeros_like(y[1:]), direction="all"
        )
        x = np.concatenate([np.atleast_1d(pressure_individual), x_crossing])
        y = np.concatenate([np.atleast_1d(y), y_crossing])
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        keep_idx = np.ediff1d(x, to_end=[1]) > 1e-6
        x = x[keep_idx]
        y = y[keep_idx]
        cape_mask = ((x < lfc_pressure) | np.isclose(x, lfc_pressure)) & (
            (x > el_pressure) | np.isclose(x, el_pressure)
        )
        x_clipped = x[cape_mask]
        y_clipped = y[cape_mask]
        cape = Rd * np.trapezoid(y_clipped, np.log(x_clipped))
        cin_mask = (x > lfc_pressure) | np.isclose(x, lfc_pressure)
        x_clipped = x[cin_mask]
        y_clipped = y[cin_mask]
        cin = Rd * np.trapezoid(y_clipped, np.log(x_clipped))
        # Set CIN to 0 if it's returned as a positive value
        cin_flat[individual_profile] = 0 if cin > 0 else cin
        # Set CAPE to 0 if it's returned as a negative value
        cape_flat[individual_profile] = 0 if cape < 0 else cape
    cape = cape_flat.reshape(pressure.shape[:-1])
    cin = cin_flat.reshape(pressure.shape[:-1])
    return cape, cin
