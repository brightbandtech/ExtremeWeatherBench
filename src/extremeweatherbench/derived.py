import xarray as xr
import logging
import numpy as np
from metpy.calc import (
    dewpoint_from_relative_humidity,
    mixed_layer_cape_cin,
    relative_humidity_from_specific_humidity,
    relative_humidity_from_dewpoint,
)
from metpy.units import units
from typing import Callable
from enum import StrEnum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def relative_humidity(
    ds: xr.Dataset, original_variable: str = "specific_humidity"
) -> xr.DataArray:
    """Compute relative humidity from specific humidity.
    Args:
        ds: An xarray Dataset containing temperature and humidity (relative and/or specific).
        from: The variable to compute relative humidity from.
    Returns:
        rh: An xarray DataArray containing the relative humidity.
    """
    if original_variable == "specific_humidity":
        return relative_humidity_from_specific_humidity(
            ds["level"].values * units.hPa,
            ds["air_temperature"].sel(level=ds["level"].values * units.hPa)
            * units.degK,
            ds["specific_humidity"].sel(level=ds["level"].values * units.hPa)
            * units("g/kg"),
        )
    elif original_variable == "dewpoint_temperature":
        return relative_humidity_from_dewpoint(
            ds["air_temperature"].sel(level=ds["level"].values * units.hPa)
            * units.degK,
            ds["dewpoint_temperature"].sel(level=ds["level"].values * units.hPa)
            * units.degK,
        )


# Craven, J. P., and H. E. Brooks, 2004: Baseline climatology
# of sounding derived parameters associated with deep moist convection. Natl. Wea. Digest, 28, 13-24.
def craven_sigsvr(ds: xr.Dataset) -> xr.DataArray:
    """Compute the Craven Significant Severe Parameter. Values over ~22,500 m3/s3 indicate higher
    likelihood of severe convection, hail, and tornadoes.
    Args:
        ds: An xarray Dataset containing winds, temperature, and humidity (relative and/or specific).

    Returns:
        sigsvr: An xarray DataArray containing the Craven Significant Severe Parameter in m3/s3.
    """
    # Compute estimated 0-6 km shear from Lepore et al 2021
    shear_0_6_km = np.sqrt(
        (ds["eastward_wind"].sel(level=500) - ds["surface_eastward_wind"]) ** 2
        + (ds["northward_wind"].sel(level=500) - ds["surface_northward_wind"]) ** 2
    )
    pressure_levels = ds["level"].values * units.hPa
    temperature = ds["temperature"].sel(level=pressure_levels) * units.degK

    # temperature
    if "dewpoint_temperature" in ds.variables:
        dewpoint_temperature = (
            ds["dewpoint_temperature"].sel(level=pressure_levels) * units.degK
        )
    elif "relative_humidity" in ds.variables:
        rh = ds["relative_humidity"].sel(level=pressure_levels) * units.dimensionless
        dewpoint_temperature = dewpoint_from_relative_humidity(temperature, rh)
    elif "specific_humidity" in ds.variables:
        rh = relative_humidity(ds, original_variable="specific_humidity")
        dewpoint_temperature = dewpoint_from_relative_humidity(temperature, rh)
    else:
        raise ValueError("No humidity variable found in dataset")

    mlcape, _ = mixed_layer_cape_cin(pressure_levels, temperature, dewpoint_temperature)
    sigsvr = mlcape * shear_0_6_km
    return sigsvr


class DerivedVariable(StrEnum):
    """Enum class for the different types of derived variables."""

    CRAVEN_SIGSVR = "craven_sigsvr"
    RELATIVE_HUMIDITY = "relative_humidity"


DERIVED_VARIABLE_MATCHER: dict[DerivedVariable, Callable] = {
    DerivedVariable.CRAVEN_SIGSVR: craven_sigsvr,
    DerivedVariable.RELATIVE_HUMIDITY: relative_humidity,
}


def get_derived_variable(variable: str) -> Callable:
    """Get the derived variable from the derived variable matcher."""
    return DERIVED_VARIABLE_MATCHER[DerivedVariable(variable)]