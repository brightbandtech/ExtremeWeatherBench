import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl
import xarray as xr

from extremeweatherbench import case, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#: Storage/access options for gridded observation datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point observation dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"

IBTRACS_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"  # noqa: E501

# type hint for the data input to the observation classes
ObservationDataInput = Union[
    xr.Dataset, xr.DataArray, pl.LazyFrame, pd.DataFrame, np.ndarray
]


# TODO: add a derived variable class
class DerivedVariable(ABC):
    """
    Abstract base class for derived variables.
    """

    @abstractmethod
    def compute(self, **kwargs) -> xr.DataArray:
        """
        Compute the derived variable from the observation data.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the derived variable.
        """


class Observation(ABC):
    """
    Abstract base class for all observation types.

    An Observation is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Observations in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """

    source: str

    @abstractmethod
    def _open_data_from_source(
        self, storage_options: Optional[dict] = None, **kwargs
    ) -> ObservationDataInput:
        """
        Open the observation data from the source, opting to avoid loading the entire dataset into memory if possible.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.

        Returns:
            The observation data with a type determined by the user.
        """

    @abstractmethod
    def _subset_data_to_case(
        self,
        data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
        **kwargs,
    ) -> ObservationDataInput:
        """
        Subset the observation data to the case information provided in IndividualCase.

        Time information, spatial bounds, and variables are captured in the case metadata
        where this method is used to subset.

        Args:
            data: The observation data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
            pandas dataframe, or numpy array.
            variables: The variables to include in the observation. Some observations may not have variables, or
            only have a singular variable; thus, this is optional.

        Returns:
            The observation data with the variables subset to the case metadata.
        """

    @abstractmethod
    def _maybe_convert_to_dataset(
        self, data: ObservationDataInput, **kwargs
    ) -> xr.Dataset:
        """
        Convert the observation data to an xarray dataset if it is not already.

        If this method is used prior to _subset_data_to_case, OOM errors are possible
        prior to subsetting.

        Args:
            data: The observation data already run through _subset_data_to_case.

        Returns:
            The observation data as an xarray dataset.
        """

    @abstractmethod
    def _maybe_map_variable_names(
        self,
        data: ObservationDataInput,
        variable_mapping: Optional[dict] = None,
        **kwargs,
    ) -> ObservationDataInput:
        """
        Map the variable names to the observation data, if required.
        """

    def _maybe_derive_variables(
        self,
        data: xr.Dataset,
        case: case.IndividualCase,
        variables: list[str | DerivedVariable],
        **kwargs,
    ) -> xr.Dataset:
        """
        Derive variables from the observation data if any exist in variables.

        Args:
            data: The observation data already run through _subset_data_to_case.
            variables: The variables to derive.

        Returns:
            The observation data with the derived variables.
        """
        for v in variables:
            # there should only be strings or derived variables in the list
            if not isinstance(v, str):
                if not issubclass(v, DerivedVariable):
                    raise ValueError(f"Expected str or DerivedVariable, got {type(v)}")
                derived_data = v().compute(
                    data=data, single_case=case, variables=variables
                )
                return derived_data
        return data

    def run_pipeline(
        self,
        case: case.IndividualCase,
        storage_options: Optional[dict] = None,
        variables: Optional[list[str | DerivedVariable]] = None,
        variable_mapping: dict = {},
        **kwargs,
    ) -> xr.Dataset:
        """
        Shared method for running the observation pipeline.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.
            variables: The variables to include in the observation. Some observations may not have variables, or
            only have a singular variable; thus, this is optional.
            variable_mapping: A dictionary of variable names to map to the observation data.
            **kwargs: Additional keyword arguments to pass in as needed.

        Returns:
            The observation data with a type determined by the user.
        """

        # Open data and process through pipeline steps
        data = (
            self._open_data_from_source(
                storage_options=storage_options,
                **kwargs,
            )
            .pipe(
                self._maybe_map_variable_names,
                variable_mapping=variable_mapping,
                **kwargs,
            )
            .pipe(
                self._subset_data_to_case,
                case=case,
                variables=variables,
                **kwargs,
            )
            .pipe(self._maybe_convert_to_dataset, **kwargs)
            .pipe(
                self._maybe_derive_variables,
                case=case,
                variables=variables or [],
                **kwargs,
            )
        )
        return data


class ERA5(Observation):
    """
    Observation class for ERA5 gridded data.

    The easiest approach to using this class
    is to use the ARCO ERA5 dataset provided by Google for a source. Otherwise, either a
    different zarr source or modifying the _open_data_from_source method to open the data
    using another method is required.
    """

    source: str = ARCO_ERA5_FULL_URI

    def _open_data_from_source(
        self,
        storage_options: Optional[dict] = None,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
        **kwargs,
    ) -> ObservationDataInput:
        data = xr.open_zarr(
            self.source,
            storage_options=storage_options,
            chunks=None,
        )
        return data

    def _subset_data_to_case(
        self,
        data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
        **kwargs,
    ) -> ObservationDataInput:
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise ValueError(f"Expected xarray Dataset or DataArray, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_data = data.sel(time=slice(case.start_date, case.end_date))

        # check that the variables are in the observation data
        if variables is not None and any(
            var not in subset_time_data.data_vars for var in variables
        ):
            raise ValueError(f"Variables {variables} not found in observation data")
        # subset the variables
        elif variables is not None:
            subset_time_variable_data = subset_time_data[variables]
        else:
            raise ValueError(
                "Variables not defined for ERA5. Please list at least one variable to select."
            )
        # # calling chunk here to avoid loading subset_data into memory
        chunks = kwargs.get("chunks", {"time": 48, "latitude": 721, "longitude": 1440})
        subset_time_variable_data = subset_time_variable_data.chunk(chunks)
        # mask the data to the case location
        fully_subset_data = case.location.mask(subset_time_variable_data, drop=True)

        return fully_subset_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput, **kwargs):
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        return data

    def _maybe_map_variable_names(
        self,
        data: ObservationDataInput,
        variable_mapping: Optional[dict] = None,
        **kwargs,
    ) -> ObservationDataInput:
        """
        Map the variable names to the observation data, if required.
        """
        if variable_mapping is None:
            return data
        # Filter the mapping to only include variables that exist in the dataset
        filtered_mapping = {
            v: k for k, v in variable_mapping.items() if v in data.data_vars
        }
        if filtered_mapping:
            data = data.rename(filtered_mapping)
        return data


class GHCN(Observation):
    """
    Observation class for GHCN tabular data.

    Data is processed using polars to maintain the lazy loading
    paradigm in _open_data_from_source and to separate the subsetting
    into _subset_data_to_case.
    """

    source: str = DEFAULT_GHCN_URI

    def _open_data_from_source(
        self, storage_options: Optional[dict] = None, **kwargs
    ) -> ObservationDataInput:
        observation_data: pl.LazyFrame = pl.scan_parquet(
            self.source, storage_options=storage_options
        )

        return observation_data

    def _subset_data_to_case(
        self,
        observation_data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
        **kwargs,
    ) -> ObservationDataInput:
        # Create filter expressions for LazyFrame
        time_min = case.start_date - pd.Timedelta(days=2)
        time_max = case.end_date + pd.Timedelta(days=2)

        if not isinstance(observation_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(observation_data)}")

        # Apply filters using proper polars expressions
        subset_observation_data = observation_data.filter(
            (pl.col("time") >= time_min)
            & (pl.col("time") <= time_max)
            & (pl.col("latitude") >= case.location.latitude_min)
            & (pl.col("latitude") <= case.location.latitude_max)
            & (pl.col("longitude") >= case.location.longitude_min)
            & (pl.col("longitude") <= case.location.longitude_max)
        )

        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        if variables is None:
            all_variables = ["time", "latitude", "longitude"]
        else:
            all_variables = variables + ["time", "latitude", "longitude"]

        # check that the variables are in the observation data
        schema_fields = [field for field in subset_observation_data.collect_schema()]
        if variables is not None and any(
            var not in schema_fields for var in all_variables
        ):
            raise ValueError(f"Variables {all_variables} not found in observation data")

        # subset the variables
        if variables is not None:
            subset_observation_data = subset_observation_data.select(all_variables)

        return subset_observation_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput, **kwargs):
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_pandas()
            data = data.set_index(["time", "latitude", "longitude"])
            # GHCN data can have duplicate values right now, dropping here if it occurs
            try:
                data = data.to_xarray()
            except ValueError as e:
                if "non-unique" in str(e):
                    logger.warning(
                        "ValueError when converting to xarray due to duplicate indexes"
                    )
                data = data.drop_duplicates().to_xarray()
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")

    def _maybe_map_variable_names(
        self, data: ObservationDataInput, variable_mapping: dict, **kwargs
    ) -> ObservationDataInput:
        """
        Map the variable names to the observation data, if required.
        """
        # Filter the mapping to only include variables that exist in the dataset
        filtered_mapping = {
            v: k for k, v in variable_mapping.items() if v in data.columns
        }
        if filtered_mapping:
            data = data.rename(filtered_mapping)
        return data


class LSR(Observation):
    """
    Observation class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods.
    """

    source: str = LSR_URI

    def _open_data_from_source(
        self, storage_options: Optional[dict] = None, **kwargs
    ) -> ObservationDataInput:
        # force LSR to use anon token to prevent google reauth issues for users
        observation_data = pd.read_parquet(
            self.source, storage_options={"token": "anon"}
        )

        return observation_data

    def _subset_data_to_case(
        self,
        observation_data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
        **kwargs,
    ) -> ObservationDataInput:
        if not isinstance(observation_data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(observation_data)}")

        # latitude, longitude are strings by default, convert to float
        observation_data["lat"] = observation_data["lat"].astype(float)
        observation_data["lon"] = observation_data["lon"].astype(float)
        observation_data["time"] = pd.to_datetime(observation_data["time"])

        filters = (
            (observation_data["time"] >= case.start_date)
            & (observation_data["time"] <= case.end_date)
            & (observation_data["lat"] >= case.location.latitude_min)
            & (observation_data["lat"] <= case.location.latitude_max)
            & (
                observation_data["lon"]
                >= utils.convert_longitude_to_180(case.location.longitude_min)
            )
            & (
                observation_data["lon"]
                <= utils.convert_longitude_to_180(case.location.longitude_max)
            )
        )

        subset_observation_data = observation_data.loc[filters]

        subset_observation_data = subset_observation_data.rename(
            columns={"lat": "latitude", "lon": "longitude", "time": "valid_time"}
        )

        return subset_observation_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput, **kwargs):
        if isinstance(data, pd.DataFrame):
            data = data.set_index(["valid_time", "latitude", "longitude"])
            data = xr.Dataset.from_dataframe(
                data[~data.index.duplicated(keep="first")], sparse=True
            )
            return data
        else:
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")

    def _maybe_map_variable_names(
        self, data: ObservationDataInput, variable_mapping: dict, **kwargs
    ) -> ObservationDataInput:
        """
        Map the variable names to the observation data, if required.
        """
        # Filter the mapping to only include variables that exist in the dataset
        filtered_mapping = {
            v: k for k, v in variable_mapping.items() if v in data.columns
        }
        if filtered_mapping:
            data = data.rename(filtered_mapping)
        return data


class IBTrACS(Observation):
    """
    Observation class for IBTrACS data.
    """

    source: str = IBTRACS_URI

    def _open_data_from_source(
        self, storage_options: Optional[dict] = None, **kwargs
    ) -> ObservationDataInput:
        # not using storage_options in this case due to NetCDF4Backend not supporting them
        observation_data: pl.LazyFrame = pl.scan_csv(
            self.source, storage_options=storage_options
        )
        return observation_data

    def _subset_data_to_case(
        self,
        observation_data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
        **kwargs,
    ) -> ObservationDataInput:
        # Create filter expressions for LazyFrame
        year = case.start_date.year

        if not isinstance(observation_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(observation_data)}")

        # Apply filters using proper polars expressions
        subset_observation_data = observation_data.filter(
            (pl.col("NAME") == case.title.upper())
        )

        all_variables = [
            "SEASON",
            "NUMBER",
            "NAME",
            "ISO_TIME",
            "LAT",
            "LON",
            "WMO_WIND",
            "USA_WIND",
            "WMO_PRES",
            "USA_PRES",
        ]
        # Get the season (year) from the case start date, cast as string as polars is interpreting the schema as strings
        season = str(year)

        # First filter by name to get the storm data
        subset_observation_data = observation_data.filter(
            (pl.col("NAME") == case.title.upper())
        )

        # Create a subquery to find all storm numbers in the same season
        matching_numbers = (
            subset_observation_data.filter(pl.col("SEASON") == season)
            .select("NUMBER")
            .unique()
        )

        # Apply the filter to get all data for storms with the same number in the same season
        # This maintains the lazy evaluation
        subset_observation_data = observation_data.join(
            matching_numbers, on="NUMBER", how="inner"
        ).filter((pl.col("NAME") == case.title.upper()) & (pl.col("SEASON") == season))

        # check that the variables are in the observation data
        schema_fields = [field for field in subset_observation_data.collect_schema()]
        if variables is not None and any(
            var not in schema_fields for var in all_variables
        ):
            raise ValueError(f"Variables {all_variables} not found in observation data")

        # subset the variables
        if variables is not None:
            subset_observation_data = subset_observation_data.select(all_variables)

        return subset_observation_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput, **kwargs):
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_pandas()
            data = data.set_index(["ISO_TIME"])
            try:
                data = data.to_xarray()
            except ValueError as e:
                if "non-unique" in str(e):
                    logger.warning(
                        "ValueError when converting to xarray due to duplicate indexes"
                    )
                data = data.drop_duplicates().to_xarray()
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")

    def _maybe_map_variable_names(
        self, data: ObservationDataInput, variable_mapping: dict, **kwargs
    ) -> ObservationDataInput:
        """
        Map the variable names to the observation data, if required.
        """
        # Filter the mapping to only include variables that exist in the dataset
        filtered_mapping = {
            v: k for k, v in variable_mapping.items() if v in data.columns
        }
        if filtered_mapping:
            data = data.rename(filtered_mapping)
        return data
