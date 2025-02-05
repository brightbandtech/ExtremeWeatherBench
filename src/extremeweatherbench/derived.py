import xarray as xr
import logging
import numpy as np
# from metpy.calc import dewpoint_from_relative_humidity, mixed_layer_cape_cin
# from metpy.units import units

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def craven_sigsvr(ds: xr.DataArray) -> xr.DataArray:
    """Compute the Craven Sigsvr parameter.

    Args:
        forecast: An xarray DataArray containing forecast data.
    """
    # Compute estimated 0-6 km shear from Lepore et al 2021
    shear_0_6_km = np.sqrt(
        (ds["eastward_wind"].sel(level=500) - ds["surface_eastward_wind"]) ** 2
        + (ds["northward_wind"].sel(level=500) - ds["surface_northward_wind"]) ** 2
    )
    return shear_0_6_km
