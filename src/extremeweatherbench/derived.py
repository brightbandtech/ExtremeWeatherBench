import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import xarray as xr

import extremeweatherbench.events.atmospheric_river as ar

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

    required_variables: List[str | Type["DerivedVariable"]]
    optional_variables: Optional[List[str | Type["DerivedVariable"]]] = None

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
    def compute(cls, data: xr.Dataset, **kwargs) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.
        It checks that the data has the variables required to build the variable,
        and then derives the variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.
            **kwargs: Additional keyword arguments to pass to the derived variable.

        Returns:
            A DataArray with the derived variable.
        """
        for v in cls.required_variables:
            if isinstance(v, str):
                if v not in data.data_vars:
                    raise ValueError(f"Input variable {v} not found in data")
            elif issubclass(v, DerivedVariable):
                if v.name not in data.data_vars:
                    raise ValueError(f"Input variable {v.name} not found in data")
            else:
                raise ValueError(f"Invalid input variable type: {type(v)}")
        return cls.derive_variable(data)


class AtmosphericRiverMask(DerivedVariable):
    """A derived variable that computes the atmospheric river mask.

    Calculates the IVT (Integrated Vapor Transport) and its Laplacian from a dataset.IVT
    is calculated using the method described in Newell et al. 1992 and elsewhere
    (e.g. Mo 2024).

    The Laplacian of IVT is calculated using a Gaussian blurring kernel with a
    sigma of 3 grid points, meant to smooth out 0.25 degree grid scale features.
    """

    required_variables = [
        "eastward_wind",
        "northward_wind",
        "specific_humidity",
    ]
    name = "atmospheric_river_mask"

    @classmethod
    def derive_variable(cls, data: xr.Dataset) -> xr.Dataset:
        """Derive the atmospheric river mask using xr.apply_ufunc approach."""
        return ar.compute_atmospheric_river_mask_ufunc(data)


def maybe_derive_variables(
    dataset: xr.Dataset,
    variables: list[Union[str, DerivedVariable, Type[DerivedVariable]]],
    **kwargs,
) -> xr.Dataset:
    """Derive variable from the data if it exists in a list of variables.

    Derived variables do not need to maintain the same spatial dimensions as the
    original dataset. Expected behavior is that an EvaluationObject has one derived
    variable. If there are multiple derived variables, the first one will be used.

    Args:
        ds: The dataset, ideally already subset in case of in memory operations
            in the derived variables.
        variables: The potential variables to derive as a list of strings or
            DerivedVariable objects.
        **kwargs: Additional keyword arguments to pass to the derived variables.

    Returns:
        A dataset with derived variables, if any exist, else the original
        dataset.
    """

    maybe_derived_variables = [v for v in variables if not isinstance(v, str)]

    if not maybe_derived_variables:
        logger.debug("No derived variables for dataset type.")
        return dataset

    if len(maybe_derived_variables) > 1:
        logger.warning(
            "Multiple derived variables provided. Only the first one will be "
            "computed. Users must use separate EvaluationObjects to derive "
            "each variable."
        )

    # Take the first derived variable and process it
    derived_variable = maybe_derived_variables[0]
    output = derived_variable.compute(data=dataset, **kwargs)

    # Ensure the DataArray has the correct name and is a DataArray.
    # Some derived variables return a dataset (multiple variables), so we need
    # to check
    if isinstance(output, xr.DataArray):
        if output.name is None:
            logger.debug(
                "Derived variable %s has no name, using class name.",
                derived_variable.name,
            )
            output.name = derived_variable.name
        # Merge the derived variable into the dataset
        return output.to_dataset()

    elif isinstance(output, xr.Dataset):
        # Check if derived dataset dimensions are compatible for merging
        return output

    # If output is neither DataArray nor Dataset, return original
    logger.warning(
        f"Derived variable {derived_variable.name} returned neither DataArray nor "
        "Dataset. Returning original dataset."
    )
    return dataset


def maybe_pull_variables_from_derived_input(
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
            # Recursively pull required variables from derived variables
            derived_required_variables.extend(
                maybe_pull_variables_from_derived_input(v.required_variables)
            )

    # Remove duplicates by converting to set and back to list
    all_variables = string_variables + derived_required_variables
    return list(set(all_variables))
