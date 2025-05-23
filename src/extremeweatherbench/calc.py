"""
This module contains constants and functions that are part of derived variables.
"""

import numpy as np
import scipy.integrate as si
import scipy.optimize as so
from tqdm import tqdm
from metpy.units import units

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


def mixing_ratio(partial_press, total_press):
    """Calculate the mixing ratio from the partial pressure and total pressure.

    Args:
        partial_press: The partial pressure of the water vapor in hPa.
        total_press: The total pressure in hPa.

    Returns:
        The mixing ratio in kg/kg.
    """
    return epsilon * partial_press / (total_press - partial_press)


def vapor_pressure(pressure, mixing_ratio):
    # pressure in hPa, mixing_ratio in kg/kg
    return pressure * mixing_ratio / (epsilon + mixing_ratio)


def saturation_vapor_pressure(temperature):
    # temperature in celsius
    return sat_press_0c * np.exp(17.67 * temperature / (temperature + 243.5))


def exner_function(pressure):
    # pressure in hPa
    return (pressure / p0) ** kappa


def get_pressure_height(pressure):
    pressure = np.atleast_1d(pressure)
    height = (t0 / gamma) * (1 - (pressure / p0) ** (Rd * gamma / g))
    return pressure, height


def potential_temperature(temperature, pressure):
    # assume incoming temp in celsius, output in kelvin
    theta = (temperature + 273.15) / exner_function(pressure)
    return theta


def dewpoint(vapor_pressure):
    # vapor pressure in hPa
    val = np.log(vapor_pressure / sat_press_0c)
    return 243.5 * val / (17.67 - val)


def dry_lapse(pressure, temperature):
    return temperature * (pressure / pressure[0]) ** kappa


def saturation_mixing_ratio(pressure, temperature):
    return mixing_ratio(saturation_vapor_pressure(temperature), pressure)


def _lcl_iter(p, p0, w, t, nan_mask_list):
    td = dewpoint(vapor_pressure(p / 100, w)) + 273.15
    p_new = p0 * (td / t) ** (1.0 / kappa)
    nan_mask_list[0] = nan_mask_list[0] | np.isnan(p_new)

    return np.where(np.isnan(p_new), p, p_new)


def insert_lcl_level(pressure, temperature, lcl_pressure):
    if pressure.ndim == 1:
        # Insert calculated_lcl_p into pressure_prof and sort
        combined_pressure = np.sort(np.append(pressure, lcl_pressure))[::-1]
        test_temp = np.interp(combined_pressure, pressure[::-1], temperature[::-1])
        # Find the index where calculated_lcl_p was inserted into combined_pressure
        lcl_index = np.where(combined_pressure == lcl_pressure)[0][0]

        # Create a copy of temp_prof to modify
        calculated_new_temp = np.copy(temperature)

        # Insert the interpolated temperature at the LCL pressure point
        calculated_new_temp = np.insert(
            calculated_new_temp, lcl_index, test_temp[lcl_index]
        )
    else:
        # Insert calculated_lcl_p into pressure_prof and sort
        # Append lcl_pressure to pressure along the first dimension
        # pressure shape: (p,x,y,z), lcl_pressure shape: (x,y,z)
        # Reshape lcl_pressure to (1,x,y,z) to append along first dimension
        lcl_pressure_reshaped = np.expand_dims(lcl_pressure, axis=0)
        combined_pressure = np.append(pressure, lcl_pressure_reshaped, axis=0)
        combined_pressure = np.sort(combined_pressure, axis=0)[::-1]
        # Handle multidimensional arrays properly
        # Reshape for broadcasting
        combined_pressure_flat = combined_pressure.reshape(
            combined_pressure.shape[0], -1
        )
        pressure_flat = pressure.reshape(pressure.shape[0], -1)
        temperature_flat = temperature.reshape(temperature.shape[0], -1)
        # Initialize output array
        interp_temp = np.zeros_like(combined_pressure_flat)
        # Interpolate for each column
        for i in range(combined_pressure_flat.shape[1]):
            interp_temp[:, i] = np.interp(
                combined_pressure_flat[:, i],
                pressure_flat[::-1, i],
                temperature_flat[::-1, i],
            )

            # Find the index where calculated_lcl_p was inserted into combined_pressure
        lcl_index = np.where(combined_pressure == lcl_pressure)[0][0]

        # Create a copy of temp_prof to modify
        calculated_new_temp = np.copy(temperature)
        # Get the shape of calculated_new_temp
        orig_shape = calculated_new_temp.shape
        # Reshape for insertion along first axis
        flat_shape = (orig_shape[0], -1)
        calculated_new_temp_flat = calculated_new_temp.reshape(flat_shape)
        interp_temp_flat = interp_temp.reshape(interp_temp.shape[0], -1)

        # Insert values for each column
        result = np.zeros((orig_shape[0] + 1, *orig_shape[1:]))
        result_flat = result.reshape(result.shape[0], -1)

        for i in range(calculated_new_temp_flat.shape[1]):
            col_result = np.insert(
                calculated_new_temp_flat[:, i],
                lcl_index,
                interp_temp_flat[lcl_index, i],
            )
            result_flat[:, i] = col_result

        calculated_new_temp = result.reshape((orig_shape[0] + 1, *orig_shape[1:]))

    return calculated_new_temp


def vectorize_lcl(p, T, Td):
    """Vectorized LCL function based on metpy.calc.lcl. Needs the bottom layer but is dimension agnostic.

    Pressure needs to be in Pa, temperature and dewpoint needs to be in K.
    """

    def _lcl_iter(p, p0, w, t, nan_mask_list):
        td = dewpoint(vapor_pressure(p / 100, w)) + 273.15
        p_new = p0 * (td / t) ** (1.0 / kappa)
        nan_mask_list[0] = nan_mask_list[0] | np.isnan(p_new)

        return np.where(np.isnan(p_new), p, p_new)

    # Handle nans by creating a mask that gets set by our _lcl_iter function if it
    # ever encounters a nan, at which point pressure is set to p, stopping iteration.
    nan_mask_list = [False]  # Use a mutable list to store the mask
    es = saturation_vapor_pressure(Td - 273.15)
    w = mixing_ratio(es, p / 100)
    lcl_p = so.fixed_point(
        _lcl_iter, p, args=(p, w, T, nan_mask_list), xtol=1e-5, maxiter=50
    )
    lcl_p = np.where(nan_mask_list[0], np.nan, lcl_p) / 100

    # np.isclose needed if surface is LCL due to precision error with np.log in dewpoint.
    # Causes issues with parcel_profile_with_lcl if removed. Issue #1187
    lcl_p = np.where(np.isclose(lcl_p, p), p, lcl_p)
    lcl_td = dewpoint(vapor_pressure(lcl_p, w))
    return lcl_p, lcl_td


def virtual_temperature_from_dewpoint(pressure, temperature, dewpoint):
    """Calculate virtual temperature from dewpoint.

    This function calculates virtual temperature from dewpoint, temperature, and pressure.
    """

    # Convert dewpoint to mixing ratio
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    # Calculate virtual temperature with given parameters
    return virtual_temperature(temperature, mixing_ratio)


def virtual_temperature(temperature, mixing_ratio):
    if np.any(temperature < 100):  # rough celsius to kelvin conversion check
        temperature = temperature + 273.15
    # temperature in kelvin, mixing_ratio in kg/kg
    return temperature * ((mixing_ratio + epsilon) / (epsilon * (1 + mixing_ratio)))


def dt(p, t):
    rs = saturation_mixing_ratio(p, t - 273.15)  # t to Celsius
    frac = (Rd * t + Lv * rs) / (Cp_d + (Lv * Lv * rs * epsilon / (Rd * t**2)))
    return frac / p


def vectorized_moist_lapse(pressure, temperature, reference_pressure=None):
    pressure_flattened = pressure.reshape(pressure.shape[0], -1)
    temperature_flattened = temperature.flatten()
    reference_pressure_flattened = (
        reference_pressure.flatten() if reference_pressure is not None else None
    )
    moist_lapse_results = np.zeros(pressure_flattened.shape)
    # Create arrays for all cells at once
    references = (
        reference_pressure_flattened
        if reference_pressure is not None
        else np.full(len(pressure_flattened[-1]), None)
    )

    with tqdm(total=len(pressure_flattened[-1])) as pbar:
        # Use numpy's vectorize to apply moist_lapse across all cells
        vectorized_moist_lapse_func = np.vectorize(
            lambda p_col, t_val, pbar, ref_p: moist_lapse(
                p_col, t_val, pbar=pbar, reference_pressure=ref_p
            ),
            signature="(n),(),(),()->(n)",
        )
        # Apply the vectorized function to all columns at once
        moist_lapse_results = vectorized_moist_lapse_func(
            np.transpose(
                pressure_flattened
            ),  # Transpose to get columns as first dimension
            temperature_flattened,
            pbar,
            references,
        ).T  # Transpose back to original orientation
    return moist_lapse_results.reshape(pressure.shape)


def moist_lapse(pressure, temperature, pbar, reference_pressure=None):
    pbar.update(1)
    temperature = np.atleast_1d(temperature)
    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]

    if np.isnan(reference_pressure) or np.all(np.isnan(temperature)):
        return np.full((temperature.size, pressure.size), np.nan)

    pres_decreasing = pressure[0] > pressure[-1]
    if pres_decreasing:
        # Everything is easier if pressures are in increasing order
        pressure = pressure[::-1]

    # It would be preferable to use a regular solver like RK45, but as of scipy 1.8.0
    # anything other than LSODA goes into an infinite loop when given NaNs for y0.
    solver_args = {
        "fun": dt,
        "y0": temperature,
        "method": "RK45",
        "atol": 1e-7,
        "rtol": 1.5e-8,
    }
    # Need to handle close points to avoid an error in the solver
    close = np.isclose(pressure, reference_pressure)
    if np.any(close):
        ret = np.broadcast_to(
            temperature[:, np.newaxis], (temperature.size, np.sum(close))
        )
    else:
        ret = np.empty((temperature.size, 0), dtype=temperature.dtype)

    # Do we have any points above the reference pressure
    points_above = (pressure < reference_pressure) & ~close
    if np.any(points_above):
        # Integrate upward--need to flip so values are properly ordered from ref to min
        press_side = pressure[points_above][::-1]

        # Flip on exit so t values correspond to increasing pressure
        result = si.solve_ivp(
            t_span=(reference_pressure, press_side[-1]),
            t_eval=press_side,
            **solver_args,
        )
        if result.success:
            ret = np.concatenate((result.y[..., ::-1], ret), axis=-1)
        else:
            raise ValueError(
                "ODE Integration failed. This is likely due to trying to "
                "calculate at too small values of pressure."
            )

    # Do we have any points below the reference pressure
    points_below = ~points_above & ~close
    if np.any(points_below):
        # Integrate downward
        press_side = pressure[points_below]
        result = si.solve_ivp(
            t_span=(reference_pressure, press_side[-1]),
            t_eval=press_side,
            **solver_args,
        )
        if result.success:
            ret = np.concatenate((ret, result.y), axis=-1)
        else:
            raise ValueError(
                "ODE Integration failed. This is likely due to trying to "
                "calculate at too small values of pressure."
            )
    if pres_decreasing:
        ret = ret[..., ::-1]
    return ret.squeeze()


# pressure needs to be in Pa, temperature and dewpoint needs to be in K
def lcl(pressure_prof, temp_prof, dew_prof):
    """Temperature needs to be in Celsius, pressure needs to be in hPa"""
    pressure_prof_pa = pressure_prof * 100  # convert to Pa
    temp_prof_k = temp_prof + 273.15  # convert to K
    # Handle nans by creating a mask that gets set by our _lcl_iter function if it
    # ever encounters a nan, at which point pressure is set to p, stopping iteration.
    nan_mask_list = [False]  # Use a mutable list to store the mask
    es = saturation_vapor_pressure(dew_prof)
    w = mixing_ratio(es, pressure_prof)
    lcl_p = so.fixed_point(
        _lcl_iter,
        pressure_prof_pa,
        args=(pressure_prof_pa, w, temp_prof_k, nan_mask_list),
        xtol=1e-5,
        maxiter=50,
    )
    lcl_p = np.where(nan_mask_list[0], np.nan, lcl_p) / 100  # convert to hPa

    # np.isclose needed if surface is LCL due to precision error with np.log in dewpoint.
    # Causes issues with parcel_profile_with_lcl if removed. Issue #1187
    calculated_lcl_p = np.atleast_1d(
        np.where(np.isclose(lcl_p, pressure_prof), pressure_prof, lcl_p)
    )
    calculated_lcl_td = np.atleast_1d(dewpoint(vapor_pressure(lcl_p, w)))
    return calculated_lcl_p, calculated_lcl_td


def _parcel_profile_helper(pressure_prof, temperature, dewpoint, axis=0):
    calculated_lcl_p, calculated_lcl_td = lcl(pressure_prof[0], temperature, dewpoint)
    temp_prof_k = temperature + 273.15  # convert to K
    print("calculated_lcl_p shape", calculated_lcl_p.shape)
    if pressure_prof.ndim == 1:
        calculated_press_lower = np.concatenate(
            (
                pressure_prof[pressure_prof >= calculated_lcl_p],
                np.atleast_1d(calculated_lcl_p),
            )
        )
    else:
        mask = pressure_prof >= calculated_lcl_p
        # Apply mask along axis 0 while preserving the shape of pressure_prof
        pressure_prof_indices = np.zeros_like(pressure_prof)
        pressure_prof_indices[mask] = pressure_prof[mask]
        # Filter out zeros to get only the valid pressure values
        pressure_prof_indices = pressure_prof_indices[
            np.any(
                pressure_prof_indices != 0,
                axis=tuple(range(1, pressure_prof_indices.ndim)),
            )
        ]
        calculated_press_lower = np.concatenate(
            (pressure_prof_indices, np.expand_dims(calculated_lcl_p, axis=axis))
        )
        calculated_press_lower[calculated_press_lower == 0] = np.nan

    print("calculated_press_lower shape", calculated_press_lower.shape)
    calculated_temp_lower = dry_lapse(calculated_press_lower, temp_prof_k)
    # If the pressure profile doesn't make it to the lcl, we can stop here
    if pressure_prof.ndim == 1:
        if np.isclose(np.nanmin(pressure_prof), calculated_lcl_p):
            return (
                calculated_press_lower[:-1],
                calculated_lcl_p,
                units.Quantity(np.array([]), calculated_press_lower.units),
                calculated_temp_lower[:-1],
                calculated_lcl_td,
                units.Quantity(np.array([]), calculated_temp_lower.units),
            )
        else:
            # Establish profile above LCL
            calculated_press_upper = np.concatenate(
                (calculated_lcl_p, pressure_prof[pressure_prof < calculated_lcl_p])
            )
            # Remove duplicate pressure values from remaining profile. Needed for solve_ivp in
            # moist_lapse. unique will return remaining values sorted ascending.
            unique, indices, counts = np.unique(
                calculated_press_upper, return_inverse=True, return_counts=True
            )
            # Find moist pseudo-adiabatic profile starting at the LCL, reversing above sorting
            calculated_temp_upper = vectorized_moist_lapse(
                unique[::-1], calculated_temp_lower[-1]
            )
            calculated_temp_upper = calculated_temp_upper[::-1][indices]
    else:
        mask = pressure_prof < calculated_lcl_p
        valid_indices = np.any(mask, axis=tuple(range(1, mask.ndim)))
        pressure_prof = pressure_prof[valid_indices]
        calculated_press_upper = np.concatenate(
            (np.expand_dims(calculated_lcl_p, axis=0), pressure_prof)
        )
        calculated_press_upper = np.sort(calculated_press_upper, axis=0)[
            ::-1
        ]  # Sort descending along 0th axis

        # Sort the pressure profile in descending order
        unique, indices, counts = np.unique(
            calculated_press_upper, return_inverse=True, return_counts=True, axis=0
        )
        calculated_temp_upper = vectorized_moist_lapse(
            unique[::-1], calculated_temp_lower[-1]
        )
        calculated_temp_upper = calculated_temp_upper[::-1][indices]

    return (
        calculated_press_lower[:-1],
        calculated_lcl_p,
        calculated_press_upper[1:],
        calculated_temp_lower[:-1],
        calculated_lcl_td,
        calculated_temp_upper[1:],
    )


def log_interpolate(x, xp, var, axis=0):
    """
    Interpolates data with logarithmic x-scale over a specified axis.
    Assumes all inputs are in descending order and need to be reversed.

    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    var : array-like
        The data to be interpolated.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.

    Returns
    -------
    array-like
        Interpolated values.
    """
    # Reverse and take log of x and xp
    x_log = np.log(x[::-1])
    xp_log = np.log(xp[::-1])

    # Handle different axes by moving the interpolation axis to the first dimension
    var_swapped = np.swapaxes(var, 0, axis)

    # Create output array
    out_shape = list(var_swapped.shape)
    out_shape[0] = len(x_log)
    result = np.empty(out_shape)

    # Iterate over all other dimensions and apply interpolation
    for idx in np.ndindex(var_swapped.shape[1:]):
        # Get the slice along the interpolation axis
        var_slice = var_swapped[(slice(None),) + idx]
        # Interpolate and store the result
        result[(slice(None),) + idx] = np.interp(x_log, xp_log, var_slice[::-1])

    # Swap back and reverse the result
    return np.swapaxes(result, 0, axis)[::-1]


def combine_profiles(
    calculated_press_lower,
    calculated_lcl_p,
    calculated_press_upper,
    calculated_temp_lower,
    calculated_lcl_td,
    calculated_temp_upper,
):
    calculated_new_press = np.concatenate(
        (
            np.atleast_1d(calculated_press_lower),
            np.expand_dims(calculated_lcl_p, axis=0)
            if calculated_lcl_p.ndim > 1
            else np.atleast_1d(calculated_lcl_p),
            np.atleast_1d(calculated_press_upper),
        )
    )
    calculated_prof_dewpoint = np.concatenate(
        (
            np.atleast_1d(calculated_temp_lower),
            np.expand_dims(calculated_lcl_td, axis=0)
            if calculated_lcl_td.ndim > 1
            else np.atleast_1d(calculated_lcl_td),
            np.atleast_1d(calculated_temp_upper),
        )
    )

    return calculated_new_press, calculated_prof_dewpoint


# All steps combined to make "parcel_profile_with_lcl"
def parcel_profile_with_lcl(pressure, temperature, dewpoint):
    (
        calculated_p_l,
        calculated_p_lcl,
        calculated_p_u,
        calculated_t_l,
        calculated_t_lcl,
        calculated_t_u,
    ) = _parcel_profile_helper(pressure, temperature[0], dewpoint[0])
    calculated_new_press, calculated_prof_temp = combine_profiles(
        calculated_p_l,
        calculated_p_lcl,
        calculated_p_u,
        calculated_t_l,
        calculated_t_lcl,
        calculated_t_u,
    )
    calculated_new_temp = insert_lcl_level(pressure, temperature, calculated_p_lcl)
    calculated_new_dewp = insert_lcl_level(pressure, dewpoint, calculated_p_lcl)
    return (
        calculated_new_press,
        calculated_new_temp,
        calculated_new_dewp,
        calculated_prof_temp,
    )


def _interp_integrate(pressure, pressure_interp, layer_depth, vars, axis=0):
    vars_interp = log_interpolate(pressure_interp, pressure, vars, axis=axis)
    integration = np.trapezoid(vars_interp, pressure_interp, axis=axis) / -layer_depth
    return integration


def mixed_parcel(pressure, temperature, temperature_dewpoint):
    theta = potential_temperature(temperature, pressure)
    es = saturation_vapor_pressure(temperature_dewpoint)
    mixing_ratio_g_g = mixing_ratio(es, pressure)
    # begin mixed layer
    if len(pressure.shape) > 1:
        pressure = pressure[:, 0, 0, 0]
    bottom_pressure, bottom_height = get_pressure_height(np.atleast_1d(pressure[0]))
    top = bottom_pressure - depth  # hPa
    top_pressure, top_height = get_pressure_height(top)
    pressure_mask = (pressure >= (top_pressure[0])) & (pressure <= (bottom_pressure[0]))
    p_interp = pressure[pressure_mask]
    if not np.any(np.isclose(top_pressure, p_interp)):
        p_interp = np.sort(np.append(p_interp, top_pressure))
    if not np.any(np.isclose(bottom_pressure, p_interp)):
        p_interp = np.sort(np.append(p_interp, bottom_pressure))
    p_interp = p_interp[::-1]
    layer_depth = abs(p_interp[0] - p_interp[-1])
    mean_theta = _interp_integrate(pressure, p_interp, layer_depth, theta, axis=0)
    mean_mixing_ratio = _interp_integrate(
        pressure, p_interp, layer_depth, mixing_ratio_g_g, axis=0
    )

    # end mixed layer
    calculated_parcel_start_pressure = pressure[0]
    calculated_parcel_temp_kelvin = (mean_theta) * exner_function(pressure[0])
    calculated_parcel_temp = calculated_parcel_temp_kelvin - 273.15
    vapor_pres = vapor_pressure(pressure[0], mean_mixing_ratio)
    calculated_parcel_dewpoint = dewpoint(vapor_pres)
    return (
        calculated_parcel_start_pressure,
        calculated_parcel_temp,
        calculated_parcel_dewpoint,
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
