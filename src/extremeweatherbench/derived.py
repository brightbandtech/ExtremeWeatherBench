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
        requires_target_dataset: If True, target dataset will be passed to
            this derived variable via kwargs. Set to False for memory efficiency
            when target data is not needed.
        compute: A method that generates the derived variable from the variables.
        derive_variable: An abstract method that defines the computation to
            derive the derived_variable from variables.
    """

    variables: List[str]
    requires_target_dataset: bool = False

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

    For forecast data, when track data is provided, the valid candidates
    approach is filtered to only include candidates within 5 great circle
    degrees of track data points and within 48 hours of the valid_time.

    Track data is automatically obtained from the target dataset when using
    the evaluation pipeline (via `requires_target_dataset=True` flag).
    """

    # required variables for TC track identification
    variables = [
        "air_pressure_at_mean_sea_level",
        "geopotential_thickness",
        "surface_eastward_wind",
        "surface_northward_wind",
    ]
    # Needs target data for track filtering
    requires_target_dataset = True

    def get_or_compute_tracks(self, data: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Get cached track data or compute if not already cached.

        This method handles the caching logic to ensure track computation
        is only done once per unique dataset.

        Track data is automatically obtained from `_target_dataset` in kwargs,
        which is provided by the evaluation pipeline when
        `requires_target_dataset=True`.

        Args:
            data: Input dataset containing required variables
            **kwargs: Must include:
                - _target_dataset: Target dataset with lat/lon/valid_time
                - case_metadata: IndividualCase with case_id_number (optional)

        Returns:
            3D dataset containing tropical cyclone track information

        Raises:
            ValueError: If _target_dataset is missing or lacks required vars
        """

        # Prepare the data with wind variables as needed
        prepared_data = calc.maybe_calculate_wind_speed(data)

        # Generates the variables needed for the TC track calculation
        # (geop. thickness, winds, temps, slp)

        # Get track data from target dataset (auto-provided by pipeline)
        tc_track_data = kwargs.get("_target_dataset", None)

        if tc_track_data is not None:
            # Verify it has the required variables for TC tracking
            required = ["latitude", "longitude", "valid_time"]
            has_required = all(
                var in tc_track_data.coords or var in tc_track_data.data_vars
                for var in required
            )
            if not has_required:
                case_metadata = kwargs.get("case_metadata", None)
                case_id = case_metadata.case_id_number if case_metadata else "unknown"
                raise ValueError(
                    f"Target dataset for case {case_id} missing required "
                    f"track variables (latitude, longitude, valid_time). "
                    f"Available coords: {list(tc_track_data.coords.keys())}, "
                    f"vars: {list(tc_track_data.data_vars.keys())}"
                )
            logger.debug("Using target dataset as track data for TC detection")
        else:
            case_metadata = kwargs.get("case_metadata", None)
            case_id = case_metadata.case_id_number if case_metadata else "unknown"
            raise ValueError(
                f"No track data provided for case {case_id}. "
                "Ensure requires_target_dataset=True is set and target data "
                "is available in the evaluation pipeline."
            )

        tctracks_ds = tropical_cyclone.generate_tc_tracks_by_init_time(
            prepared_data["air_pressure_at_mean_sea_level"],
            prepared_data["surface_wind_speed"],
            prepared_data.get("geopotential_thickness", None),
            tc_track_data,
        )
        return tctracks_ds

    def derive_variable(self, data: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        """Derive the TC track variables.

        This base method returns the full track dataset. Child classes should
        override this method to extract specific variables from the track data.

        Args:
            data: Input dataset containing required meteorological variables

        Returns:
            DataArray containing the derived variable
        """
        # Get the cached or computed track data
        tracks_dataset = self.get_or_compute_tracks(data, *args, **kwargs)

        # Squeeze the dataset to remove the track dimension if only one track is present
        return tracks_dataset.squeeze()


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
