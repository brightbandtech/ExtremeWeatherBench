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

    required_variables: List[str]

    @property
    def name(self) -> str:
        """A name for the derived variable. Defaults to the class name."""
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
        utils.check_data_for_variables(data, cls.required_variables)
        return cls.derive_variable(data)


# TODO: add the AR mask calculations
class AtmosphericRiverMask(DerivedVariable):
    """A derived variable that computes the atmospheric river mask."""

    required_variables = ["air_pressure_at_mean_sea_level"]

    # TODO: add the AR mask calculations
    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the atmospheric river mask."""
        return data[self.input_variables[0]] < 1000


# TODO: add the IVT calculations for ARs
class IntegratedVaporTransport(DerivedVariable):
    """A derived variable that computes the integrated vapor transport."""

    required_variables = [
        "surface_eastward_wind",
        "surface_northward_wind",
        "specific_humidity",
    ]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the integrated vapor transport."""
        return (
            data[self.input_variables[0]]
            * data[self.input_variables[1]]
            * data[self.input_variables[2]]
        )


# TODO: add the IVT Laplacian calculations for ARs
class IntegratedVaporTransportLaplacian(DerivedVariable):
    """A derived variable that computes the integrated vapor transport Jacobian."""

    required_variables = [
        "surface_eastward_wind",
        "surface_northward_wind",
        "specific_humidity",
    ]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the integrated vapor transport Jacobian."""
        return (
            data[self.input_variables[0]]
            * data[self.input_variables[1]]
            * data[self.input_variables[2]]
        )


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

    def get_required_variables(self, data: xr.Dataset) -> List[str]:
        """Get the actually required variables based on what's available in the data."""
        base_vars = ["air_pressure_at_mean_sea_level", "geopotential"]

        # Check wind variable availability
        has_wind_speed = "surface_wind_speed" in data.data_vars
        has_wind_components = (
            "surface_eastward_wind" in data.data_vars
            and "surface_northward_wind" in data.data_vars
        )

        if has_wind_speed:
            wind_vars = ["surface_wind_speed"]
        elif has_wind_components:
            wind_vars = ["surface_eastward_wind", "surface_northward_wind"]
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

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the TC track variables."""

        # Get the actually required variables for this dataset
        required_vars = self.get_required_variables(data)

        # Check that we have the required variables
        utils.check_data_for_variables(data, required_vars)

        # Prepare the data with wind variables as needed
        prepared_data = self._prepare_wind_data(data)

        # Generates the variables needed for the TC track calculation (geop. thickness, winds, temps, slp)
        cyclone_dataset = calc.generate_tc_variables(prepared_data)
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
        variables: The potential variables to derive as a list of strings or DerivedVariable objects.

    Returns:
        A dataset with derived variables, if any exist, else the original dataset.
    """

    derived_variables = [v for v in variables if not isinstance(v, str)]
    derived_data = {v.name: v.compute(data=ds) for v in derived_variables}
    # TODO check logic for merging derived data
    ds = ds.merge(derived_data)
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
