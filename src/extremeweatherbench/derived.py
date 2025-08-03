from abc import ABC, abstractmethod
from typing import List

import xarray as xr


class DerivedVariable(ABC):
    """A abstract base class defining the interface for ExtremeWeatherBench derived variables.

    A DerivedVariable is any variable that requires extra computation than what is provided in analysis
    or forecast data. Some examples include the practically perfect hindcast, MLCAPE, IVT, or atmospheric river masks.

    Attributes:
        name: The name that is used for applications of derived variables. Defaults to the class name.
        input_variables: A list of variables that are used to build the variable.
        build: A method that builds the variable from the input variables. Build is used specifically to distinguish
        from the compute method in xarray, which eagerly processes the data and loads into memory; build is used to
        lazily process the data and return a dataset that can be used later to compute the variable.
        _check_data_for_variables: A method that checks that the data has the variables required to build the variable,
        using input_variables.
        derive_variable: An abstractmethod that defines the computation to derive the variable from input_variables.
    """

    @property
    def name(self) -> str:
        """A name for the derived variable. Defaults to the class name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """A list of variables that are used to compute the variable.

        Each derived variable is a product of one or more variables in an incoming dataset.
        The input variables are the names of the variables in the incoming dataset.

        """
        pass

    @abstractmethod
    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the variable from the input variables.

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
        self._check_data_for_variables(data)
        return self.derive_variable(data)

    def _check_data_for_variables(self, data: xr.Dataset):
        """Check that the data has the variables required to build the variable, based on assigned input variables."""
        for v in self.input_variables:
            if v not in data.data_vars:
                raise ValueError(f"Input variable {v} not found in data")
