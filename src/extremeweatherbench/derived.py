import abc
import logging
from typing import List, Optional, Sequence, Union

import xarray as xr

from extremeweatherbench import calc
from extremeweatherbench.events import tropical_cyclone

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


class TropicalCycloneTrackVariables(DerivedVariable):
    """A derived variable abstract class for tropical cyclone (TC) variables.

    This class serves as a parent for TC-related derived variables and provides
    shared track computation with caching to avoid reprocessing the same data
    multiple times across different child classes.

    The track data is computed once and cached, then child classes can extract
    specific variables (like sea level pressure, wind speed) from the cached
    track dataset.

    Deriving the track locations using default TempestExtremes criteria:
    https://doi.org/10.5194/gmd-14-5023-2021

    For forecast data, when track data data is provided, the valid candidates
    approach is filtered to only include candidates within 5 great circle
    degrees of track data points and within 48 hours of the valid_time.
    """

    # required variables for TC track identification
    variables = [
        "air_pressure_at_mean_sea_level",
        "geopotential_thickness",
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

        # Prepare the data with wind variables as needed
        prepared_data = calc.maybe_calculate_wind_speed(data)

        # Generates the variables needed for the TC track calculation
        # (geop. thickness, winds, temps, slp)

        # Check if we should apply track data filtering
        # First check kwargs, then the global registry
        tc_track_data = kwargs.get("tc_track_data", None)
        case_metadata = kwargs.get("case_metadata", None)
        case_id_number = case_metadata.case_id_number if case_metadata else None

        logger.debug(
            "TC derived variable: case_id_number=%s, tc_track_data from kwargs=%s",
            case_id_number,
            tc_track_data is not None,
        )

        if tc_track_data is None and case_id_number is not None:
            # Convert to string to match registry key type
            case_id_str = str(case_id_number)
            logger.debug("Looking up tc_track_data for case %s", case_id_str)
            tc_track_data = tropical_cyclone.get_tc_track_data(case_id_str)
            logger.debug("Retrieved tc_track_data: %s", tc_track_data is not None)

        if tc_track_data is None:
            logger.error("No track data found for case %s", case_id_number)
            raise ValueError("No track data data provided to constrain TC tracks.")

        tctracks_ds = tropical_cyclone.generate_tc_tracks_by_init_time(
            prepared_data["air_pressure_at_mean_sea_level"],
            prepared_data["surface_wind_speed"],
            prepared_data.get("geopotential_thickness", None),
            tc_track_data,
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

        # Squeeze the dataset to remove the track dimension if only one track is present
        return tracks_dataset.squeeze()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the global track cache.

        Useful for memory management or when processing completely different datasets.
        """
        tropical_cyclone._TC_TRACK_CACHE.clear()


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
