"""
Derived variables for ExtremeWeatherBench's case evaluations.
"""

import logging

import numpy as np
import xarray as xr

from docs.notebooks import calc

np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def dewpoint_from_specific_humidity(
    specific_humidity: xr.DataArray, pressure: xr.DataArray
) -> xr.DataArray:
    """Calculate dewpoint from specific humidity and pressure.

    The pressure DataArray must be the same shape as the specific humidity DataArray.

    Args:
        specific_humidity: The specific humidity in kg/kg.
        pressure: The pressure in hPa.

    Returns:
        The dewpoint DataArray.
    """
    mixing_ratio = specific_humidity / (1 - specific_humidity)
    e = pressure * mixing_ratio / (calc.epsilon + mixing_ratio)

    return calc.dewpoint_from_vapor_pressure(e)


def craven_brooks_sig_svr(
    ds: xr.Dataset,
    pressure_var: str = "pressure",
    temperature_var: str = "temperature",
    temperature_dewpoint_var: str = "dewpoint",
    eastward_wind_var: str = "eastward_wind",
    northward_wind_var: str = "northward_wind",
    surface_eastward_wind_var: str = "surface_eastward_wind",
    surface_northward_wind_var: str = "surface_northward_wind",
    depth: float = 100,
) -> xr.DataArray:
    """Calculates the Craven-Brooks Significant Severe (CBSS) parameter.

    CBSS is the product of mixed layer CAPE, typically at 100 hPa mixed layer depth, and low level (0-6 km) shear.
    Values over ~22,500 m3/s3 indicate higher likelihood of severe convection, hail, and tornadoes. More information
    available at Craven, J. P., and H. E. Brooks, 2004: Baseline climatology of sounding derived parameters associated
    with deep moist convection. Natl. Wea. Digest, 28, 13-24.

    Args:
        ds: Dataset containing pressure, temperature, and dewpoint variables
        pressure_var: Name of the pressure variable in the dataset
        temperature_var: Name of the temperature variable in the dataset
        temperature_dewpoint_var: Name of the dewpoint variable in the dataset
        depth: Depth of the mixed layer in hPa

    Returns:
        sig_svr: ndarray of Significant Severe parameter values
    """
    # Check for prerequisites to ensure successful execution
    ds = calc._basic_ds_checks(ds)
    # CIN not needed for CBSS
    cape, _ = mixed_layer_cape_cin(
        ds,
        pressure_var,
        temperature_var,
        temperature_dewpoint_var,
        depth,
    )
    shear = low_level_shear(
        ds,
        eastward_wind_var,
        northward_wind_var,
        surface_eastward_wind_var,
        surface_northward_wind_var,
    )
    cbss = cape * shear
    return cbss


def low_level_shear(
    ds: xr.Dataset,
    eastward_wind_var: str = "eastward_wind",
    northward_wind_var: str = "northward_wind",
    surface_eastward_wind_var: str = "surface_eastward_wind",
    surface_northward_wind_var: str = "surface_northward_wind",
) -> xr.DataArray:
    """Calculates the low level (0-6 km) shear of a dataset (Lepore et al 2021).

    Args:
        ds: Dataset containing eastward and northward (u and v) wind vectors

    Returns:
        ll_shear: ndarray of low level shear values in m/s
    """
    ll_shear = np.sqrt(
        (ds[eastward_wind_var].sel(level=500) - ds[surface_eastward_wind_var]) ** 2
        + (ds[northward_wind_var].sel(level=500) - ds[surface_northward_wind_var]) ** 2
    )
    return ll_shear


def mixed_layer_cape_cin(
    ds: xr.Dataset,
    pressure_var: str = "pressure",
    temperature_var: str = "temperature",
    temperature_dewpoint_var: str = "dewpoint",
    depth: float = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the mixed layer CAPE and CIN of a dataset.

    Uses a lookup table for the moist pseudoadiabats to calculate the convective available
    potential energy and convective inhibition given a mixed layer with a prescribed depth,
    e.g. 100 hPa.

    Currently requires a dataset with the pressure levels as the last dimension and time,
    latitude, and longitude as the first three dimensions (in any order). Temperature and dewpoint
    must be in Celsius, pressure must be in hPa.

    Args:
        ds: Dataset containing pressure, temperature, and dewpoint variables
        pressure_var: Name of the pressure variable in the dataset
        temperature_var: Name of the temperature variable in the dataset
        temperature_dewpoint_var: Name of the dewpoint variable in the dataset
        depth: Depth of the mixed layer in hPa

    Returns:
        cape: ndarray of CAPE values in J/kg
        cin: ndarray of CIN values in J/kg
    """
    ds = calc._basic_ds_checks(ds)
    pressure = ds["level"]
    mixed_layer_mask = ds[pressure_var] < (pressure[0] - depth)
    # Get the indices where the condition is True along the last dimension
    valid_indices = np.any(
        mixed_layer_mask, axis=tuple(range(mixed_layer_mask.ndim - 1))
    )

    (
        calculated_parcel_start_pressure,
        calculated_parcel_temp,
        calculated_parcel_dewpoint,
    ) = calc.mixed_parcel(
        ds, pressure_var, temperature_var, temperature_dewpoint_var, depth
    )
    parcel_temp_reshaped = np.expand_dims(calculated_parcel_temp, axis=-1)
    parcel_dewpoint_reshaped = np.expand_dims(calculated_parcel_dewpoint, axis=-1)

    # Extract valid pressure, temperature and dewpoint profiles
    pressure_prof = ds[pressure_var][..., valid_indices]
    temp_prof = ds[temperature_var][..., valid_indices]
    dew_prof = ds[temperature_dewpoint_var][..., valid_indices]
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
    calculated_lcl_pressure, calculated_lcl_td = calc.new_lcl(
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

    temp_prof_at_or_below_lcl = calc.dry_lapse(
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
        calc.moist_lapse_lookup(
            pressure_prof_above_lcl,
            temp_prof_at_or_below_lcl[..., -1] - 273.15,
            reference_pressure=calculated_lcl_pressure,
        )
        + 273.15
    )
    # Combine profiles at or below LCL and above LCL
    combined_all_pressure_w_lcl, combined_all_temp_w_lcl = calc.combine_profiles(
        pressure_prof_at_or_below_lcl[..., :-1],
        calculated_lcl_pressure,
        pressure_prof_above_lcl,
        temp_prof_at_or_below_lcl[..., :-1],
        calculated_lcl_td,
        temp_above_lcl,
        axis=-1,
    )

    # Insert LCL level into profiles
    calculated_new_temp = calc.insert_lcl_level_fast(
        pressure_prof, temp_prof, calculated_lcl_pressure
    )
    # calculated_new_temp = calc.insert_lcl_level(
    #     pressure_prof, temp_prof, calculated_lcl_pressure
    # )
    calculated_new_dewpoint = calc.insert_lcl_level_fast(
        pressure_prof, dew_prof, calculated_lcl_pressure
    )
    # calculated_new_dewpoint = calc.insert_lcl_level(
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
    nan_mask_pressure = np.all(np.isnan(combined_all_pressure_w_lcl), axis=(0, 1, 2))
    nan_mask_temp = np.all(np.isnan(combined_all_temp_w_lcl), axis=(0, 1, 2))

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
    # Convert temps back to Celsius
    # TODO: sanity check combined_all_temp_w_lcl as the ml_profile. might be too hot
    cape, cin = calc.mlcape_cin(
        combined_all_pressure_w_lcl,
        calculated_new_temp - 273.15,
        calculated_new_dewpoint - 273.15,
        combined_all_temp_w_lcl - 273.15,
    )
    return cape, cin
