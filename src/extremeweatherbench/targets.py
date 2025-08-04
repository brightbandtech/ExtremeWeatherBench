from abc import ABC, abstractmethod
from typing import Optional

import xarray as xr

from extremeweatherbench import derived  # type: ignore
from extremeweatherbench.case import CaseOperator

# will be an import from derived once it is implemented
from extremeweatherbench.utils import IncomingDataInput, maybe_map_variable_names


class TargetBase(ABC):
    """
    An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """

    source: str

    @abstractmethod
    def open_data_from_source(
        self, storage_options: Optional[dict] = None
    ) -> IncomingDataInput:
        """
        Open the target data from the source, opting to avoid loading the entire dataset into memory if possible.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.

        Returns:
            The target data with a type determined by the user.
        """

    @abstractmethod
    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case: CaseOperator,
    ) -> IncomingDataInput:
        """
        Subset the target data to the case information provided in CaseOperator.

        Time information, spatial bounds, and variables are captured in the case metadata
        where this method is used to subset.

        Args:
            data: The observation data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
            pandas dataframe, or numpy array.
            case: The case operator to subset the data to; includes time information, spatial bounds, and variables.

        Returns:
            The target data with the variables subset to the case metadata.
        """

    def maybe_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """
        Convert the target data to an xarray dataset if it is not already.

        This method handles the common conversion cases automatically. Override this method
        only if you need custom conversion logic beyond the standard cases.

        Args:
            data: The target data to convert.

        Returns:
            The target data as an xarray dataset.
        """
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            return data.to_dataset()
        else:
            # For other data types, try to use a custom conversion method if available
            return self._custom_convert_to_dataset(data)

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """
        Hook method for custom conversion logic. Override this method in subclasses
        if you need custom conversion behavior for non-xarray data types.

        By default, this raises a NotImplementedError to encourage explicit handling
        of custom data types.

        Args:
            data: The target data to convert.

        Returns:
            The target data as an xarray dataset.
        """
        raise NotImplementedError(
            f"Conversion from {type(data)} to xarray.Dataset not implemented. "
            f"Override _custom_convert_to_dataset in your TargetBase subclass."
        )

    def run_pipeline(
        self,
        case: CaseOperator,
        target_storage_options: Optional[dict] = None,
        target_variable_mapping: dict = {},
    ) -> xr.Dataset:
        """
        Shared method for running the target pipeline.

        Args:
            source: The source of the target data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.
            target_variables: The variables to include in the target. Some target objects may not have variables, or
            only have a singular variable; thus, this is optional.
            target_variable_mapping: A dictionary of variable names to map to the target data.
            **kwargs: Additional keyword arguments to pass in as needed.

        Returns:
            The target data with a type determined by the user.
        """

        # Open data and process through pipeline steps
        data = (
            # opens data from user-defined source
            self.open_data_from_source(
                storage_options=target_storage_options,
            )
            # maps variable names to the target data if not already using EWB naming conventions
            .pipe(
                maybe_map_variable_names,
                variable_mapping=target_variable_mapping,
            )
            # subsets the target data using the caseoperator metadata
            .pipe(
                self.subset_data_to_case,
                case=case,
            )
            # converts the target data to an xarray dataset if it is not already
            .pipe(self.maybe_convert_to_dataset)
            # derives variables from the target data if derived variables are defined
            .pipe(derived.maybe_derive_variables, variables=case.target_variables)
        )
        return data
