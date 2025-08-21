import logging
from abc import ABC, abstractmethod
from typing import List, Type, Union

import numpy as np
import xarray as xr

from extremeweatherbench import calc, utils
from extremeweatherbench.events import tropical_cyclone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DerivedVariable(ABC):
    """An abstract base class defining the interface for ExtremeWeatherBench
    derived variables.

    A DerivedVariable is any variable that requires extra computation than what
    is provided in analysis or forecast data. Some examples include the
    practically perfect hindcast, MLCAPE, IVT, or atmospheric river masks.

    Attributes:
        name: The name that is used for applications of derived variables.
            Defaults to the class name.
        required_variables: A list of variables that are used to build the
            variable.
        build: A method that builds the variable from the required variables.
            Build is used specifically to distinguish from the compute method in
            xarray, which eagerly processes the data and loads into memory;
            build is used to lazily process the data and return a dataset that
            can be used later to compute the variable.
        derive_variable: An abstract method that defines the computation to
            derive the variable from required_variables.
    """

    required_variables: List[str]

    @property
    def name(self) -> str:
        """A name for the derived variable.

        Defaults to the class name.
        """
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the variable from the required variables.

        The output of the derivation must be a single variable output returned as
        a DataArray.

        Args:
            data: The dataset to derive the variable from.

        Returns:
            A DataArray with the derived variable.
        """
        pass

    @classmethod
    def compute(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.
        It checks that the data has the variables required to build the variable,
        and then derives the variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.
            *args: Additional positional arguments to pass to derive_variable.
            **kwargs: Additional keyword arguments to pass to derive_variable.

        Returns:
            A DataArray with the derived variable.
        """
        for v in cls.required_variables:
            if v not in data.data_vars:
                raise ValueError(f"Input variable {v} not found in data")
        return cls.derive_variable(data, *args, **kwargs)


# TODO: add the AR mask calculations
class AtmosphericRiverMask(DerivedVariable):
    """A derived variable that computes the atmospheric river mask."""

    required_variables = ["air_pressure_at_mean_sea_level"]

    # TODO: add the AR mask calculations
    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the atmospheric river mask."""
        raise NotImplementedError("Atmospheric river mask not yet implemented")


# TODO: add the IVT calculations for ARs
class IntegratedVaporTransport(DerivedVariable):
    """Calculates the IVT (Integrated Vapor Transport) from a dataset, using the method
    described in Newell et al.

    1992 and elsewhere (e.g. Mo 2024).
    """

    required_variables = [
        "eastward_wind",
        "northward_wind",
        "specific_humidity",
    ]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the integrated vapor transport.

        Args:
            data: The input xarray dataset.

        Returns:
            The IVT (Integrated Vapor Transport) quantity as a DataArray.
        """
        coords_dict = {dim: data.coords[dim] for dim in data.dims if dim != "level"}
        if "surface_standard_pressure" not in data.data_vars:
            data["surface_standard_pressure"] = calc.calculate_pressure_at_surface(
                calc.orography(data)
            )

        # Find the axis corresponding to "level", assuming all variables have
        # the same dimension order
        level_axis = list(data.dims).index("level")

        # TODO: REMOVE COMPUTE BEFORE MERGE, this is to speed up testing
        data = data.compute()

        _, level_broadcast, sfc_pres_broadcast = xr.broadcast(
            data, data["level"], data["surface_standard_pressure"]
        )
        data["adjusted_level"] = xr.where(
            level_broadcast * 100 < sfc_pres_broadcast, data["level"], np.nan
        )
        data["vertical_integral_of_eastward_water_vapour_flux"] = xr.DataArray(
            calc.nantrapezoid(
                data["eastward_wind"] * data["specific_humidity"],
                x=data.adjusted_level * 100,
                axis=level_axis,
            )
            / 9.80665,
            coords=coords_dict,
            dims=coords_dict.keys(),
        )
        data["vertical_integral_of_northward_water_vapour_flux"] = xr.DataArray(
            calc.nantrapezoid(
                data["northward_wind"] * data["specific_humidity"],
                x=data.adjusted_level * 100,
                axis=level_axis,
            )
            / 9.80665,
            coords=coords_dict,
            dims=coords_dict.keys(),
        )

        ivt_magnitude = np.hypot(
            data["vertical_integral_of_eastward_water_vapour_flux"],
            data["vertical_integral_of_northward_water_vapour_flux"],
        )
        return xr.DataArray(
            ivt_magnitude,
            coords=coords_dict,
            dims=coords_dict.keys(),
        )


# TODO: add the IVT Laplacian calculations for ARs
class IntegratedVaporTransportLaplacian(DerivedVariable):
    """A derived variable that computes the integrated vapor transport Jacobian."""

    required_variables = [
        "surface_eastward_wind",
        "surface_northward_wind",
        "specific_humidity",
    ]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the integrated vapor transport Jacobian."""
        raise NotImplementedError("IVT Laplacian not yet implemented")


class GeopotentialThickness(DerivedVariable):
    """A derived variable that computes the geopotential thickness."""

    required_variables = ["geopotential"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the geopotential thickness.

        Args:
            data: The input xarray dataset.
            top_level: The top level of the geopotential thickness in hPa.
            bottom_level: The bottom level of the geopotential thickness in hPa.

        Returns:
            The geopotential thickness as a DataArray.
        """
        # Handle both positional and keyword arguments
        if len(args) >= 2:
            top_level = args[0]
            bottom_level = args[1]
        else:
            top_level = kwargs.get("top_level", 300)
            bottom_level = kwargs.get("bottom_level", 500)
        top_level_geopotential = data["geopotential"].sel(level=top_level)
        bottom_level_geopotential = data["geopotential"].sel(level=bottom_level)
        geopotential_thickness = (
            top_level_geopotential - bottom_level_geopotential
        ) / 9.80665

        geopotential_thickness.attrs = dict(
            description=f"Geopotential thickness of {top_level} and {bottom_level} hPa",
            units="m",
        )
        return geopotential_thickness


class SurfaceWindSpeed(DerivedVariable):
    """A derived variable that computes the surface wind speed."""

    required_variables = ["surface_eastward_wind", "surface_northward_wind"]

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the surface wind speed."""
        return np.hypot(data["surface_eastward_wind"], data["surface_northward_wind"])


class TropicalCycloneTrackVariable(DerivedVariable):
    """A derived variable abstract class for tropical cyclone (TC) variables.

    This class serves as a parent for TC-related derived variables and provides
    shared track computation with caching to avoid reprocessing the same data
    multiple times across different child classes.

    The track data is computed once and cached, then child classes can extract
    specific variables (like sea level pressure, wind speed) from the cached
    track dataset.

    Deriving the track locations using default TempestExtremes criteria:
    https://doi.org/10.5194/gmd-14-5023-2021

    For forecast data, when IBTrACS data is provided, the valid candidates
    approach is filtered to only include candidates within 5 great circle
    degrees of IBTrACS points and within 120 hours of the valid_time.
    """

    # required variables for TC track identification
    required_variables = [
        "air_pressure_at_mean_sea_level",
        "geopotential",
        "surface_eastward_wind",
        "surface_northward_wind",
    ]

    @classmethod
    def _get_or_compute_tracks(cls, data: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Get cached track data or compute if not already cached.

        This method handles the caching logic to ensure track computation
        is only done once per unique dataset.

        Args:
            data: Input dataset containing required variables

        Returns:
            3D dataset containing tropical cyclone track information
        """
        cache_key = tropical_cyclone._generate_cache_key(data)

        # Return cached result if available
        if cache_key in tropical_cyclone._TC_TRACK_CACHE:
            return tropical_cyclone._TC_TRACK_CACHE[cache_key]

        # Compute tracks if not cached
        def _prepare_wind_data(data: xr.Dataset) -> xr.Dataset:
            """Prepare wind data by computing wind speed if needed."""
            # Make a copy to avoid modifying original
            prepared_data = data.copy()

            has_wind_speed = "surface_wind_speed" in data.data_vars
            has_wind_components = (
                "surface_eastward_wind" in data.data_vars
                and "surface_northward_wind" in data.data_vars
            )

            # If we don't have wind speed but have components, compute it
            if not has_wind_speed and has_wind_components:
                prepared_data["surface_wind_speed"] = np.hypot(
                    data["surface_eastward_wind"], data["surface_northward_wind"]
                )

            return prepared_data

        # Prepare the data with wind variables as needed
        prepared_data = _prepare_wind_data(data)

        # Generates the variables needed for the TC track calculation
        # (geop. thickness, winds, temps, slp)
        cyclone_dataset = tropical_cyclone.generate_tc_variables(prepared_data)

        # Check if we should apply IBTrACS filtering
        # First check kwargs, then the global registry
        ibtracs_data = kwargs.get("ibtracs_data", None)
        if ibtracs_data is None:
            # Try to get from registry using case_id if provided
            case_id = kwargs.get("case_id", None)
            if case_id is not None:
                ibtracs_data = tropical_cyclone.get_ibtracs_data(case_id)
            else:
                raise ValueError("No IBTrACS data provided to constrain TC tracks.")
        # Use IBTrACS-filtered TC detection
        tctracks_ds = tropical_cyclone.create_tctracks_from_dataset_with_ibtracs_filter(
            cyclone_dataset, ibtracs_data
        )
        # Cache the result
        tropical_cyclone._TC_TRACK_CACHE[cache_key] = tctracks_ds

        return tctracks_ds

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the TC track variables.

        This base method returns the full track dataset. Child classes should
        override this method to extract specific variables from the track data.

        Args:
            data: Input dataset containing required meteorological variables

        Returns:
            DataArray containing the derived variable
        """
        # Get the cached or computed track data
        tracks_dataset = cls._get_or_compute_tracks(data, *args, **kwargs)
        tracks_dataset = utils.convert_valid_time_to_init_time(tracks_dataset)

        return tracks_dataset

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the global track cache.

        Useful for memory management or when processing completely different datasets.
        """
        tropical_cyclone._TC_TRACK_CACHE.clear()


def maybe_derive_variables(
    ds: xr.Dataset, variables: list[str | DerivedVariable], **kwargs
) -> xr.Dataset:
    """Derive variables from the data if any exist in a list of variables.

    Derived variables must maintain the same spatial dimensions as the original
    dataset.

    Args:
        ds: The dataset, ideally already subset in case of in memory operations
            in the derived variables.
        variables: The potential variables to derive as a list of strings or
            DerivedVariable objects.

    Returns:
        A dataset with derived variables, if any exist, else the original
        dataset.
    """

    derived_variables = [v for v in variables if not isinstance(v, str)]
    derived_data = {}
    if derived_variables:
        for v in derived_variables:
            output_da = v.compute(data=ds, **kwargs)
            # Ensure the DataArray has the correct name
            if output_da.name is None:
                output_da.name = v.name
            derived_data[v.name] = output_da
    # TODO consider removing data variables only used for derivation
    ds = ds.merge(derived_data)
    return ds


def maybe_pull_required_variables_from_derived_input(
    incoming_variables: list[Union[str, DerivedVariable, Type[DerivedVariable]]],
) -> list[str]:
    """Pull the required variables from a derived input and add to the list of
    variables to pull.

    Args:
        incoming_variables: a list of string and/or derived variables.

    Returns:
        A list of variables possibly including derived variables' required
        variables.
    """
    string_variables = [v for v in incoming_variables if isinstance(v, str)]

    derived_required_variables = []
    for v in incoming_variables:
        if isinstance(v, DerivedVariable):
            # Handle instances of DerivedVariable
            derived_required_variables.extend(v.required_variables)
        elif isinstance(v, type) and issubclass(v, DerivedVariable):
            # Handle classes that inherit from DerivedVariable
            derived_required_variables.extend(v.required_variables)

    return string_variables + derived_required_variables
