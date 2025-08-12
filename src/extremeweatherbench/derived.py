from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import xarray as xr

from extremeweatherbench import calc, utils


class DerivedVariable(ABC):
    """An abstract base class defining the interface for ExtremeWeatherBench derived variables.

    A DerivedVariable is any variable that requires extra computation than what is provided in analysis
    or forecast data. Some examples include the practically perfect hindcast, MLCAPE, IVT, or atmospheric river masks.

    Attributes:
        name: The name that is used for applications of derived variables. Defaults to the class name.
        required_variables: A list of variables that are used to build the variable.
        build: A method that builds the variable from the required variables. Build is used specifically to distinguish
        from the compute method in xarray, which eagerly processes the data and loads into memory; build is used to
        lazily process the data and return a dataset that can be used later to compute the variable.
        derive_variable: An abstract method that defines the computation to derive the variable from required_variables.
    """

    @property
    def name(self) -> str:
        """A name for the derived variable. Defaults to the class name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def required_variables(self) -> List[str]:
        """A list of variables that are used to compute the variable.

        Each derived variable is a product of one or more variables in an incoming dataset.
        The required variables are the names of the variables in the incoming dataset.

        """
        pass

    @abstractmethod
    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the variable from the required variables.

        The output of the derivation must be a single variable output returned as
        a DataArray.

        Args:
            data: The dataset to derive the variable from.

        Returns:
            A DataArray with the derived variable.
        """
        pass

    def build(self, data: xr.Dataset) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.
        It checks that the data has the variables required to build the variable,
        and then derives the variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.

        Returns:
            A DataArray with the derived variable.
        """
        utils.check_data_for_variables(data, self.input_variables)
        return self.derive_variable(data)


class SurfaceWindSpeed(DerivedVariable):
    """A derived variable that computes the surface wind speed."""

    name = "surface_wind_speed"
    required_variables = ["surface_eastward_wind", "surface_northward_wind"]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the surface wind speed."""
        return np.hypot(
            data[self.required_variables[0]], data[self.required_variables[1]]
        )


class AtmosphericRiverMask(DerivedVariable):
    """A derived variable that computes the atmospheric river mask."""

    name = "atmospheric_river_mask"
    input_variables = ["msl"]

    # TODO: add the AR mask calculations
    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the atmospheric river mask."""
        return data[self.input_variables[0]] < 1000


class IntegratedVaporTransport(DerivedVariable):
    """A derived variable that computes the integrated vapor transport."""

    name = "integrated_vapor_transport"
    input_variables = ["u", "v", "q"]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the integrated vapor transport."""
        return (
            data[self.input_variables[0]]
            * data[self.input_variables[1]]
            * data[self.input_variables[2]]
        )


class IntegratedVaporTransportJacobian(DerivedVariable):
    """A derived variable that computes the integrated vapor transport Jacobian."""

    name = "integrated_vapor_transport_jacobian"
    input_variables = ["u", "v", "q"]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the integrated vapor transport Jacobian."""
        return (
            data[self.input_variables[0]]
            * data[self.input_variables[1]]
            * data[self.input_variables[2]]
        )


class TCTrack(DerivedVariable):
    """A derived variable that computes the TC track outputs."""

    def __init__(self, prefer_wind_speed: bool = True):
        """Initialize TCTrack with flexible wind variable handling.

        Args:
            prefer_wind_speed: If True, prefer surface_wind_speed over component winds.
            If False, prefer component winds over wind speed.
        """
        self.prefer_wind_speed = prefer_wind_speed

    @property
    def required_variables(self) -> List[str]:
        """Return flexible input variables based on data availability."""
        # Base required variables
        base_vars = [
            "air_pressure_at_mean_sea_level",
            "geopotential",
        ]

        # Wind variables - we'll check availability in get_required_variables
        wind_vars = [
            "surface_wind_speed",  # Preferred if available
            "surface_eastward_wind",  # Fallback u component
            "surface_northward_wind",  # Fallback v component
        ]

        return base_vars + wind_vars

    def get_required_variables(self, data: xr.Dataset) -> List[str]:
        """Get the actually required variables based on what's available in the data."""
        base_vars = ["air_pressure_at_mean_sea_level", "geopotential"]

        # Check wind variable availability
        has_wind_speed = "surface_wind_speed" in data.data_vars
        has_wind_components = (
            "surface_eastward_wind" in data.data_vars
            and "surface_northward_wind" in data.data_vars
        )

        if self.prefer_wind_speed and has_wind_speed:
            wind_vars = ["surface_wind_speed"]
        elif has_wind_components:
            wind_vars = ["surface_eastward_wind", "surface_northward_wind"]
        elif has_wind_speed:
            wind_vars = ["surface_wind_speed"]
        else:
            raise ValueError(
                "Neither 'surface_wind_speed' nor both 'surface_eastward_wind' and "
                "'surface_northward_wind' are available in the dataset"
            )

        return base_vars + wind_vars

    def _prepare_wind_data(self, data: xr.Dataset) -> xr.Dataset:
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

    def build(self, data: xr.Dataset) -> xr.DataArray:
        """Build the derived variable with flexible input checking."""
        # Get the actually required variables for this dataset
        required_vars = self.get_required_variables(data)

        # Check that we have the required variables
        utils.check_data_for_variables(data, required_vars)
        # Prepare the data with wind variables as needed
        prepared_data = self._prepare_wind_data(data)

        return self.derive_variable(prepared_data)

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the TC track."""
        cyclone_dataset_builder = calc.CycloneDatasetBuilder()

        # Generates the variables needed for the TC track calculation (geop. thickness, winds, temps, slp)
        cyclone_dataset = cyclone_dataset_builder.generate_variables(data)
        tctracks = calc.create_tctracks_from_dataset(cyclone_dataset)
        tctracks_ds_3d = calc.tctracks_to_3d_dataset(tctracks)

        return tctracks_ds_3d


def maybe_derive_variables(
    ds: xr.Dataset, variables: list[str | DerivedVariable]
) -> xr.Dataset:
    """Derive variables from the data if any exist in a list of variables.

    Derived variables must maintain the same spatial dimensions as the original dataset.

    Args:
        ds: The dataset, ideally already subset in case of in memory operations in the derived variables.
        case: The case to derive the variables for.
        variables: The potential variables to derive as a list of strings or DerivedVariable objects.

    Returns:
        A dataset with derived variables, if any exist, else the original dataset.
    """
    derived_variables = {}

    derived_variables = [v for v in variables if not isinstance(v, str)]
    if derived_variables:
        for v in derived_variables:
            derived_variable = v() if isinstance(v, type) else v
            derived_data = derived_variable.build(data=ds)
            ds[derived_variable.name] = derived_data

    return ds


def maybe_pull_required_variables_from_derived_input(
    incoming_variables: list[Union[str, DerivedVariable]],
) -> list[str]:
    """Pull the required variables from a derived input and add to the list of variables to pull.

    Args:
        incoming_variables: a list of string and/or derived variables.

    Returns:
        A list of variables possibly including derived variables' required variables.
    """
    string_variables = [v for v in incoming_variables if isinstance(v, str)]
    derived_required_variables = []

    for v in incoming_variables:
        if not isinstance(v, str):
            derived_variable = v() if isinstance(v, type) else v
            if isinstance(derived_variable, DerivedVariable):
                derived_required_variables.extend(derived_variable.required_variables)

    return string_variables + derived_required_variables
