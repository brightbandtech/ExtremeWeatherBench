import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd  # type: ignore
import polars as pl
import xarray as xr

from extremeweatherbench import case

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
    def compute(self, data: xr.Dataset) -> xr.DataArray:
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
        self, storage_options: Optional[dict] = None
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
    def _maybe_convert_to_dataset(self, data: ObservationDataInput) -> xr.Dataset:
        """
        Convert the observation data to an xarray dataset if it is not already.

        If this method is used prior to _subset_data_to_case, OOM errors are possible
        prior to subsetting.

        Args:
            data: The observation data already run through _subset_data_to_case.

        Returns:
            The observation data as an xarray dataset.
        """

    def _maybe_derive_variables(
        self, data: xr.Dataset, variables: list[str | DerivedVariable]
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
            if isinstance(v, DerivedVariable):
                derived_variable = v.compute(data)
                data[v.name()] = derived_variable
        return data

    def run_pipeline(
        self,
        case: case.IndividualCase,
        storage_options: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Shared method for running the observation pipeline.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.
            variables: The variables to include in the observation. Some observations may not have variables, or
            only have a singular variable; thus, this is optional.

        Returns:
            The observation data with a type determined by the user.
        """

        # Open data and process through pipeline steps
        data = (
            self._open_data_from_source(
                storage_options=storage_options,
            )
            .pipe(
                self._subset_data_to_case,
                case=case,
                variables=[v for v in case.data_vars if isinstance(v, str)]
                if case.data_vars
                else None,
            )
            .pipe(self._maybe_convert_to_dataset)
            .pipe(self._maybe_derive_variables, variables=case.data_vars or [])
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
        self, storage_options: Optional[dict] = None
    ) -> ObservationDataInput:
        data = xr.open_zarr(
            self.source,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return data

    def _subset_data_to_case(
        self,
        data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
    ) -> ObservationDataInput:
        # TODO: fix case to automatically apply these; currently stand-in for now
        case.latitude_min = case.location.latitude - case.bounding_box_degrees / 2
        case.latitude_max = case.location.latitude + case.bounding_box_degrees / 2
        case.longitude_min = np.mod(
            case.location.longitude - case.bounding_box_degrees / 2, 360
        )
        case.longitude_max = np.mod(
            case.location.longitude + case.bounding_box_degrees / 2, 360
        )

        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise ValueError(f"Expected xarray Dataset or DataArray, got {type(data)}")

        subset_data = data.sel(
            time=slice(case.start_date, case.end_date),
            # latitudes are sliced from max to min
            latitude=slice(case.latitude_max, case.latitude_min),
            longitude=slice(case.longitude_min, case.longitude_max),
        )

        # check that the variables are in the observation data
        if variables is not None and any(
            var not in subset_data.data_vars for var in variables
        ):
            raise ValueError(f"Variables {variables} not found in observation data")

        # subset the variables
        if variables is not None:
            subset_data = subset_data[variables]

        return subset_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput):
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
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
        self, storage_options: Optional[dict] = None
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
    ) -> ObservationDataInput:
        # Create filter expressions for LazyFrame
        time_min = case.start_date - pd.Timedelta(days=2)
        time_max = case.end_date + pd.Timedelta(days=2)

        # TODO: fix case to automatically apply these; currently stand-in for now
        case.latitude_min = case.location.latitude - case.bounding_box_degrees / 2
        case.latitude_max = case.location.latitude + case.bounding_box_degrees / 2
        case.longitude_min = case.location.longitude - case.bounding_box_degrees / 2
        case.longitude_max = case.location.longitude + case.bounding_box_degrees / 2

        if not isinstance(observation_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(observation_data)}")

        # Apply filters using proper polars expressions
        subset_observation_data = observation_data.filter(
            (pl.col("time") >= time_min)
            & (pl.col("time") <= time_max)
            & (pl.col("latitude") >= case.latitude_min)
            & (pl.col("latitude") <= case.latitude_max)
            & (pl.col("longitude") >= case.longitude_min)
            & (pl.col("longitude") <= case.longitude_max)
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

    def _maybe_convert_to_dataset(self, data: ObservationDataInput):
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


class LSR(Observation):
    """
    Observation class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods.
    """

    source: str = LSR_URI

    def _open_data_from_source(
        self, storage_options: Optional[dict] = None
    ) -> ObservationDataInput:
        observation_data = pd.read_parquet(self.source, storage_options=storage_options)

        return observation_data

    def _subset_data_to_case(
        self,
        observation_data: ObservationDataInput,
        case: case.IndividualCase,
        variables: Optional[list[str]] = None,
    ) -> ObservationDataInput:
        if not isinstance(observation_data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(observation_data)}")

        # latitude, longitude are strings by default, convert to float
        observation_data["lat"] = observation_data["lat"].astype(float)
        observation_data["lon"] = observation_data["lon"].astype(float)
        observation_data["time"] = pd.to_datetime(observation_data["time"])

        # TODO: fix case to automatically apply these; currently stand-in for now
        case.latitude_min = case.location.latitude - case.bounding_box_degrees / 2
        case.latitude_max = case.location.latitude + case.bounding_box_degrees / 2
        case.longitude_min = np.mod(
            case.location.longitude - case.bounding_box_degrees / 2, 360
        )
        case.longitude_max = np.mod(
            case.location.longitude + case.bounding_box_degrees / 2, 360
        )

        filters = (
            (observation_data["time"] >= case.start_date)
            & (observation_data["time"] <= case.end_date)
            & (observation_data["lat"] >= case.latitude_min)
            & (observation_data["lat"] <= case.latitude_max)
            & (observation_data["lon"] >= case.longitude_min)
            & (observation_data["lon"] <= case.longitude_max)
        )

        subset_observation_data = observation_data.loc[filters]

        subset_observation_data = subset_observation_data.rename(
            columns={"lat": "latitude", "lon": "longitude", "time": "valid_time"}
        )

        return subset_observation_data

    def _maybe_convert_to_dataset(self, data: ObservationDataInput):
        if isinstance(data, pd.DataFrame):
            data = data.set_index(["valid_time", "latitude", "longitude"])
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
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")


class IBTrACS(Observation):
    """
    Observation class for IBTrACS data.
    """

    source: str = IBTRACS_URI

    def _open_data_from_source(
        self, storage_options: Optional[dict] = None
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
    ) -> ObservationDataInput:
        raise NotImplementedError("IBTrACS data subset is not implemented yet")

    def _maybe_convert_to_dataset(self, data: ObservationDataInput):
        if isinstance(data, pd.DataFrame):
            data = data.set_index(["valid_time", "latitude", "longitude"])
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
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")
