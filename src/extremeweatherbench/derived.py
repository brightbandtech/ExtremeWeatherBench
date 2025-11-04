import abc
import logging
from typing import List, Optional, Sequence, Union

import xarray as xr

import extremeweatherbench.events.atmospheric_river as ar

logger = logging.getLogger(__name__)


class DerivedVariable(abc.ABC):
    """An abstract base class defining the interface for ExtremeWeatherBench
    derived variables.

    A DerivedVariable is any variable or transform that requires extra computation than
    what is provided in analysis or forecast data. Some examples include the
    practically perfect hindcast, MLCAPE, IVT, or atmospheric river masks.

    Attributes:
        name: The name that is used for applications of derived variables.
            Defaults to the class name.
        variables: A list of variables that are used to build the
            derived variable.
        compute: A method that generates the derived variable from the variables.
        derive_variable: An abstract method that defines the computation to
            derive the derived_variable from variables.
    """

    variables: List[str]

    def __init__(self, name: Optional[str] = None):
        """Initialize the derived variable.

        Args:
            name: The name of the derived variable. Defaults to class-level
                name attribute if present, otherwise the class name.
        """
        self.name = name or getattr(self.__class__, "name", self.__class__.__name__)

    @abc.abstractmethod
    def derive_variable(self, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the variable from the required variables.

        The output of the derivation must be a single variable output returned as
        a DataArray.

        Args:
            data: The dataset to derive the variable from.

        Returns:
            A DataArray with the derived variable.
        """
        pass

    def compute(self, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.
            *args: Additional positional arguments to pass to derive_variable.
            **kwargs: Additional keyword arguments to pass to derive_variable.

        Returns:
            A DataArray with the derived variable.
        """
        return self.derive_variable(data, *args, **kwargs)


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
    optional_variables = [
        "integrated_vapor_transport",
        "surface_standard_pressure",
        "relative_humidity",
        "air_temperature",
    ]
    optional_variables_mapping = {
        "integrated_vapor_transport": [
            "eastward_wind",
            "northward_wind",
            "specific_humidity",
        ],
        "surface_standard_pressure": ["surface_standard_pressure"],
        "relative_humidity": ["specific_humidity", "air_temperature"],
    }

    # Note: this name is used for the intersection between the AR mask and land, not the
    # mask. The mask is named "atmospheric_river_mask" in the atmospheric_river module.
    # EWB's atmospheric river event detection uses the intersection for evaluation.
    name = "atmospheric_river_land_intersection"

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Derive the atmospheric river mask using xr.apply_ufunc approach."""
        return ar.build_atmospheric_river_mask_and_land_intersection(data)


def maybe_derive_variables(
    data: xr.Dataset,
    variables: list[Union[str, DerivedVariable]],
    **kwargs,
) -> xr.Dataset:
    """Derive variable from the data if it exists in a list of variables.

    Derived variables do not need to maintain the same spatial dimensions as the
    original dataset. Expected behavior is that an EvaluationObject has one derived
    variable. If there are multiple derived variables, the first one will be used.

    Args:
        data: The data, ideally already subset in case of in memory operations
            in the derived variables.
        variables: The potential variables to derive as a list of strings or
            DerivedVariable objects.
        **kwargs: Additional keyword arguments to pass to the derived variables.

    Returns:
        A dataset with derived variables, if any exist, else the original
        dataset.
    """
    # If there are no valid times, return the dataset unaltered; saves time as case will
    # be skipped
    if data.valid_time.size == 0:
        logger.debug("No valid times found in the dataset.")
        return data

    maybe_derived_variables = [v for v in variables if not isinstance(v, str)]

    if not maybe_derived_variables:
        logger.debug("No derived variables for dataset type.")
        return data

    if len(maybe_derived_variables) > 1:
        logger.warning(
            "Multiple derived variables provided. Only the first one will be "
            "computed. Users must use separate EvaluationObjects to derive "
            "each variable."
        )

    # Take the first derived variable and process it
    derived_variable = maybe_derived_variables[0]
    output = derived_variable.compute(data=data, **kwargs)

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
    return data


def maybe_include_variables_from_derived_input(
    incoming_variables: Sequence[Union[str, DerivedVariable]],
) -> list[str]:
    """Identify and return variables that a derived variable needs to compute.

    Args:
        incoming_variables: a list of string and/or derived variables.

    Returns:
        A list of variables possibly including derived variables' required
        variables.
    """
    string_variables = [v for v in incoming_variables if isinstance(v, str)]

    derived_variables: list[str] = []
    for v in incoming_variables:
        if isinstance(v, DerivedVariable):
            # Handle instances of DerivedVariable
            derived_variables.extend(v.variables)

    return list(set(string_variables + derived_variables))


def _maybe_convert_variable_to_string(
    variable: Union[str, DerivedVariable],
) -> str:
    """Convert a variable to its string representation.

    Args:
        variable: Either a string or a DerivedVariable instance.

    Returns:
        The string representation of the variable.
    """
    if isinstance(variable, str):
        return variable
    # variable is a DerivedVariable instance with .name set in __init__
    return str(variable.name)
