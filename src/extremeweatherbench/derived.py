import logging
from abc import ABC, abstractmethod
from typing import List, Type, Union

import numpy as np
import xarray as xr

import extremeweatherbench.events.severe_convection as sc
from extremeweatherbench import calc

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
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
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
    def compute(cls, data: xr.Dataset) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.
        It checks that the data has the variables required to build the variable,
        and then derives the variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.

        Returns:
            A DataArray with the derived variable.
        """
        for v in cls.required_variables:
            if v not in data.data_vars:
                raise ValueError(f"Input variable {v} not found in data")
        return cls.derive_variable(data)


class CravenBrooksSignificantSevere(DerivedVariable):
    """A derived variable that computes the Craven-Brooks significant severe
    convection index.
    """

    required_variables = [
        "air_temperature",
        "eastward_wind",
        "northward_wind",
        "surface_eastward_wind",
        "surface_northward_wind",
        "specific_humidity",
    ]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Derive the Craven-Brooks significant severe convection index."""
        data["dewpoint_temperature"] = sc.dewpoint_from_specific_humidity(
            data["specific_humidity"], data["air_pressure_at_mean_sea_level"]
        )
        # create broadcasted pressure variable
        data["pressure"] = xr.broadcast(
            data["level"], data["air_pressure_at_mean_sea_level"]
        )[0]
        return sc.craven_brooks_significant_severe(data)


# TODO: add the AR mask calculations
class AtmosphericRiverMask(DerivedVariable):
    """A derived variable that computes the atmospheric river mask."""

    required_variables = ["air_pressure_at_mean_sea_level"]

    # TODO: add the AR mask calculations
    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
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
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
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
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Derive the integrated vapor transport Jacobian."""
        raise NotImplementedError("IVT Laplacian not yet implemented")


# TODO: finish TC track calculation port
class TCTrackVariables(DerivedVariable):
    """A derived variable that computes the TC track outputs.

    This derived variable is used to compute the TC track outputs.
    It is a flexible variable that can be used to compute the TC track outputs
    based on the data availability.

    Deriving the track locations using default TempestExtremes criteria:
    https://doi.org/10.5194/gmd-14-5023-2021
    """

    required_variables = [
        "air_pressure_at_mean_sea_level",
        "geopotential",
        "surface_wind_speed",
        "surface_eastward_wind",
        "surface_northward_wind",
    ]

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.DataArray:
        """Derive the TC track variables."""

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
        cyclone_dataset = calc.generate_tc_variables(prepared_data)
        tctracks = calc.create_tctracks_from_dataset(cyclone_dataset)
        tctracks_ds_3d = calc.tctracks_to_3d_dataset(tctracks)

        return tctracks_ds_3d  # type: ignore[return-value]


def maybe_derive_variables(
    ds: xr.Dataset, variables: list[str | DerivedVariable]
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
            output_da = v.compute(data=ds)
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
