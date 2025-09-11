import logging
from abc import ABC, abstractmethod
from typing import Sequence, Type, TypeGuard, Union

import xarray as xr

import extremeweatherbench.events.severe_convection as sc

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

    required_variables: list[str]
    optional_variables: list[str] = []
    optional_variables_mapping: dict = {}

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

        Args:
            data: The dataset to build the derived variable from.
            **kwargs: Additional keyword arguments to pass to the derived variable.

        Returns:
            A DataArray with the derived variable.
        """
        return cls.derive_variable(data, *args, **kwargs)


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
        "air_pressure_at_mean_sea_level",
    ]
    optional_variables = [
        "dewpoint_temperature",
        "relative_humidity",
        "specific_humidity",
    ]
    optional_variables_mapping = {
        "dewpoint_temperature": ["specific_humidity"],
        "relative_humidity": ["specific_humidity"],
        "specific_humidity": ["relative_humidity"],
    }
    name = "craven_brooks_significant_severe"

    @classmethod
    def derive_variable(cls, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the Craven-Brooks significant severe convection index."""
        # create broadcasted pressure variable, output target is always last
        _, data["pressure"] = xr.broadcast(data["air_temperature"], data["level"])
        # calculate dewpoint temperature if not present
        if "dewpoint_temperature" not in data.data_vars:
            # using relative humidity if present
            if "relative_humidity" in data.data_vars:
                data["dewpoint_temperature"] = data[
                    "relative_humidity"
                ] * sc.saturation_vapor_pressure(data["air_temperature"])
            # or using specific humidity if present
            elif "specific_humidity" in data.data_vars:
                data["dewpoint_temperature"] = sc.dewpoint_from_specific_humidity(
                    data["specific_humidity"], data["pressure"]
                )
            # and if neither are present, raise an error
            else:
                raise KeyError("No variable to compute dewpoint temperature.")

        cbss = sc.craven_brooks_significant_severe(data)
        coords = {dim: data.coords[dim] for dim in data.sizes.keys() if dim != "level"}
        return xr.DataArray(
            cbss,
            coords=coords,
            dims=coords.keys(),
            name=cls.name,
        )


def maybe_derive_variables(
    data: xr.Dataset,
    variables: list[Union[str, DerivedVariable, Type[DerivedVariable]]],
    **kwargs,
) -> xr.Dataset:
    """Derive variable from the data if it exists in a list of variables.

    Derived variables do not need to maintain the same spatial dimensions as the
    original dataset. Expected behavior is that an EvaluationObject has one derived
    variable. If there are multiple derived variables, the first one will be used.

    Args:
        data: The dataset, ideally already subset in case of in memory operations
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
    incoming_variables: Sequence[Union[str, Type[DerivedVariable]]],
) -> list[str]:
    """Identify and return variables that a derived variable needs to compute.

    Args:
        incoming_variables: a list of string and/or derived variables.

    Returns:
        A list of variables possibly including derived variables' required
        variables.
    """
    string_variables = [v for v in incoming_variables if isinstance(v, str)]

    derived_required_variables = []
    for v in incoming_variables:
        # TODO: change to is_derived_variable
        if isinstance(v, DerivedVariable):
            # Handle instances of DerivedVariable
            derived_required_variables.extend(v.required_variables)
        elif isinstance(v, type) and issubclass(v, DerivedVariable):
            # Handle classes that inherit from DerivedVariable
            # Recursively pull required variables from derived variables
            derived_required_variables.extend(
                maybe_include_variables_from_derived_input(v.required_variables)
            )

    return string_variables + derived_required_variables


def is_derived_variable(
    variable: Union[str, Type[DerivedVariable]],
) -> TypeGuard[Type[DerivedVariable]]:
    """Checks whether the incoming variable is a string or a DerivedVariable.

    Args:
        variable: a single string or DerivedVariable object

    Returns:
        True if the variable is a DerivedVariable object, False otherwise
    """

    return isinstance(variable, type) and issubclass(variable, DerivedVariable)
