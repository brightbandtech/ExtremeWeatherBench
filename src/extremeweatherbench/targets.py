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

    @abstractmethod
    def maybe_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """
        Abstract method to convert the target data to an xarray dataset.

        In the case of a target object already being a dataset, this method should return the data unchanged.
        Some data, such as a LazyFrame or DataFrame, could have more complex steps to convert to a dataset
        with the proper dimensions; this method is primarily used to handle these cases.

        Args:
            data: The target data already run through _subset_data_to_case.

        Returns:
            The target data as an xarray dataset.
        """
        pass

    def _maybe_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """
        Convert the target data to an xarray dataset if it is not already.

        Args:
            data: The target data already run through

        Returns:
            The target data as an xarray dataset.
        """
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            return data.to_dataset()
        else:
            return self._maybe_convert_to_dataset(data)

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
            # subsets the target data to the case information
            .pipe(
                self.subset_data_to_case,
                case=case,
            )
            # converts the target data to an xarray dataset if it is not already
            .pipe(self._maybe_convert_to_dataset)
            # derives variables from the target data if derived variables are defined
            .pipe(derived.maybe_derive_variables, variables=case.target_variables)
        )
        return data
