import abc
import logging
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import xarray as xr

import extremeweatherbench.events.atmospheric_river as ar
import extremeweatherbench.events.severe_convection as sc
from extremeweatherbench import calc

if TYPE_CHECKING:
    from extremeweatherbench import cases


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

    def __init__(
        self,
        output_variables: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the derived variable.

        Args:
            output_variables: Optional list of variable names that specify
                which outputs to use from the derived computation.
            name: The name of the derived variable. Defaults to class-level
                name attribute if present, otherwise the class name.
        """
        self.name = name or getattr(self.__class__, "name", self.__class__.__name__)
        self.output_variables = output_variables

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


class CravenBrooksSignificantSevere(DerivedVariable):
    """A derived variable that computes the Craven-Brooks significant severe
    convection index.
    """

    variables = [
        "air_temperature",
        "eastward_wind",
        "northward_wind",
        "surface_eastward_wind",
        "surface_northward_wind",
        "air_pressure_at_mean_sea_level",
        "geopotential",
        "specific_humidity",
    ]

    def __init__(
        self,
        name: Optional[str] = "craven_brooks_significant_severe",
        layer_depth: float = 100,
    ):
        super().__init__(name=name)
        self.layer_depth = layer_depth

    def derive_variable(
        self, data: xr.Dataset, case_metadata: "cases.IndividualCase", *args, **kwargs
    ) -> xr.DataArray:
        """Derive the Craven-Brooks significant severe parameter."""
        # Ensure 'level' is the last dimension for proper processing
        if "level" in data.dims and list(data.dims)[-1] != "level":
            # Get all dimensions and move 'level' to the end
            dims = list(data.dims)
            dims.remove("level")
            dims.append("level")
            data = data.transpose(*dims)
        # Check if pressure levels need to be reversed
        # CAPE expects descending order (surface to top)
        needs_reverse = data["level"][0] < data["level"][-1]
        if needs_reverse:
            # Reverse and load to ensure contiguous arrays for Numba
            data = data.isel(level=slice(None, None, -1))
        # calculate dewpoint temperature if not present
        if "dewpoint_temperature" not in data.data_vars:
            # using relative humidity if present
            if "relative_humidity" in data.data_vars:
                data["dewpoint_temperature"] = data[
                    "relative_humidity"
                ] * calc.saturation_vapor_pressure(data["air_temperature"])
            # or using specific humidity if present
            elif "specific_humidity" in data.data_vars:
                data["dewpoint_temperature"] = calc.dewpoint_from_specific_humidity(
                    data["level"], data["specific_humidity"]
                )
            # and if neither are present, raise an error
            else:
                raise KeyError("No variable to compute dewpoint temperature.")
        layer_depth = self.layer_depth
        # Compute CAPE (geopotential in m²/s²)
        cape = sc.compute_mixed_layer_cape(
            pressure=data["level"],
            temperature=data["air_temperature"],
            dewpoint=data["dewpoint_temperature"],
            geopotential=data["geopotential"],
            pressure_dim="level",
            depth=layer_depth,
        )

        shear = sc.low_level_shear(
            eastward_wind=data["eastward_wind"],
            northward_wind=data["northward_wind"],
            surface_eastward_wind=data["surface_eastward_wind"],
            surface_northward_wind=data["surface_northward_wind"],
        )

        cbss = cape * shear
        cbss = cbss.max(
            dim="valid_time",
        )
        cbss = cbss.expand_dims(valid_time=[case_metadata.start_date])
        coords = {dim: cbss.coords[dim] for dim in cbss.sizes.keys() if dim != "level"}
        return xr.DataArray(
            cbss,
            coords=coords,
            dims=coords.keys(),
            name=self.name,
        )


class AtmosphericRiverVariables(DerivedVariable):
    """A derived variable that computes atmospheric river related variables.

    Calculates the IVT (Integrated Vapor Transport), atmospheric river mask, and land
    intersection. IVT is calculated using the method described in Newell et al. 1992 and
    elsewhere (e.g. Mo 2024).

    Output variables are: integrated_vapor_transport, atmospheric_river_mask, and
    atmospheric_river_land_intersection. Users must declare at least one of the output
    variables they want when calling the derived variable.

    The Laplacian of IVT is calculated using a Gaussian blurring kernel with a
    sigma of 3 grid points, meant to smooth out 0.25 degree grid scale features.
    """

    variables = [
        "eastward_wind",
        "northward_wind",
        "specific_humidity",
    ]

    name = "atmospheric_river"

    def derive_variable(self, data: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Derive the atmospheric river mask and land intersection."""
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
    # be skipped. Check for multiple possible time coordinate names.
    time_coords = ["valid_time", "time", "init_time"]
    has_time_data = any(
        coord in data.coords and data[coord].size > 0 for coord in time_coords
    )
    if not has_time_data:
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
        output_ds = output.to_dataset()

    elif isinstance(output, xr.Dataset):
        # Check if derived dataset dimensions are compatible for merging
        output_ds = output

    else:
        # If output is neither DataArray nor Dataset, return original
        logger.warning(
            f"Derived variable {derived_variable.name} returned neither "
            "DataArray nor Dataset. Returning original dataset."
        )
        return data

    # Subset to output_variables if specified
    if (
        hasattr(derived_variable, "output_variables")
        and derived_variable.output_variables
    ):
        missing_vars = set(derived_variable.output_variables) - set(output_ds.data_vars)
        if missing_vars:
            logger.warning(
                f"Derived variable {derived_variable.name} specified "
                f"output_variables {derived_variable.output_variables}, but "
                f"computed output is missing: {missing_vars}"
            )
        # Subset to only the requested output variables
        available_vars = [
            v for v in derived_variable.output_variables if v in output_ds.data_vars
        ]
        if available_vars:
            output_ds = output_ds[available_vars]
        else:
            logger.warning(
                f"None of the specified output_variables "
                f"{derived_variable.output_variables} are in the computed "
                f"dataset. Returning original dataset."
            )
            return data

    return output_ds


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
