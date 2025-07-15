import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import case

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Observation(ABC):
    """
    Abstract base class for all observation types.

    An Observation is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or a reference dataset. Observations are not required to be the same
    variable as the forecast dataset, but they must be in the same coordinate system for evaluation.

    Attributes:
        source: The source of the observation.
        case: The case that the observation is associated with.
        variables: The variables to include in the observation.
    """

    def __init__(
        self, case: case.IndividualCase
    ):  # TODO: add Variable type to include here alongside str
        self.case = case

    @abstractmethod
    def _open_data_from_source(
        self, source: str, storage_options: Optional[dict] = None
    ):
        """
        Open the observation data from the source, opting to avoid loading the entire dataset into memory if possible.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.

        Returns:
            The observation data with a type determined by the user.
        """

    @abstractmethod
    def _subset_data_to_case(self, data, variables: Optional[list[str]] = None):
        """
        Subset the observation data to the case.

        Args:
            data: The observation data to subset.
            variables: The variables to include in the observation. Some observations may not have variables, or
            only have a singular variable; thus, this is optional.

        Returns:
            The observation data with the variables subset to the case.
        """

    @abstractmethod
    def _maybe_convert_to_dataset(self):
        """
        Convert the observation data to an xarray dataset if it is not already.
        """

    def run_pipeline(
        self,
        source: str,
        storage_options: Optional[dict] = None,
        variables: Optional[list[str]] = None,
    ) -> xr.Dataset:
        """
        Shared method for running the observation pipeline.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.

        Returns:
            The observation data with a type determined by the user.
        """
        data = self._open_data_from_source(
            source=source, storage_options=storage_options
        )
        data = self._subset_data_to_case(data, variables=variables)
        data = self._maybe_convert_to_dataset(data)
        return data


class ERA5(Observation):
    """
    Observation class for ERA5 gridded data.
    """

    def __init__(self, case: case.IndividualCase):
        super().__init__(case)

    def _open_data_from_source(
        self, source: str, storage_options: Optional[dict] = None
    ):
        data = xr.open_zarr(
            source,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return data

    def _subset_data_to_case(self, data, variables: list[str]):
        """
        Subset the observation data to the case.
        """

        # TODO: fix case to automatically apply these; currently stand-in for now
        self.case.latitude_min = (
            self.case.location.latitude - self.case.bounding_box_degrees / 2
        )
        self.case.latitude_max = (
            self.case.location.latitude + self.case.bounding_box_degrees / 2
        )
        self.case.longitude_min = np.mod(
            self.case.location.longitude - self.case.bounding_box_degrees / 2, 360
        )
        self.case.longitude_max = np.mod(
            self.case.location.longitude + self.case.bounding_box_degrees / 2, 360
        )

        subset_data = data.sel(
            time=slice(self.case.start_date, self.case.end_date),
            # latitudes are sliced from max to min
            latitude=slice(self.case.latitude_max, self.case.latitude_min),
            longitude=slice(self.case.longitude_min, self.case.longitude_max),
        )

        # check that the variables are in the observation data
        if any(var not in subset_data.data_vars for var in variables):
            raise ValueError(f"Variables {variables} not found in observation data")

        # subset the variables
        subset_data = subset_data[variables]

        return subset_data

    def _maybe_convert_to_dataset(self, data):
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        return data


class GHCN(Observation):
    """
    Observation class for GHCN gridded data.
    """

    def __init__(self, case: case.IndividualCase):
        super().__init__(case)

    def _open_data_from_source(
        self, source: str, storage_options: Optional[dict] = None
    ):
        """
        Open the observation data from the source.

        Args:
            source: The source of the observation data.
            storage_options: Optional storage options for the source.

        Returns:
            The observation data as a polars LazyFrame.
        """
        observation_data: pl.LazyFrame = pl.scan_parquet(
            source, storage_options=storage_options
        )

        return observation_data

    def _subset_data_to_case(
        self, observation_data: pl.LazyFrame, variables: list[str]
    ):
        """
        Subset the observation data to the case.

        Args:
            observation_data: The observation data to subset to the case.
            variables: The variables to include in the observation.

        Returns:
            The subset observation data.
        """
        # Create filter expressions for LazyFrame
        time_min = self.case.start_date - pd.Timedelta(days=2)
        time_max = self.case.end_date + pd.Timedelta(days=2)

        # TODO: fix case to automatically apply these; currently stand-in for now
        self.case.latitude_min = (
            self.case.location.latitude - self.case.bounding_box_degrees / 2
        )
        self.case.latitude_max = (
            self.case.location.latitude + self.case.bounding_box_degrees / 2
        )
        self.case.longitude_min = (
            self.case.location.longitude - self.case.bounding_box_degrees / 2
        )
        self.case.longitude_max = (
            self.case.location.longitude + self.case.bounding_box_degrees / 2
        )

        # Apply filters using proper polars expressions
        subset_observation_data = observation_data.filter(
            (pl.col("time") >= time_min)
            & (pl.col("time") <= time_max)
            & (pl.col("latitude") >= self.case.latitude_min)
            & (pl.col("latitude") <= self.case.latitude_max)
            & (pl.col("longitude") >= self.case.longitude_min)
            & (pl.col("longitude") <= self.case.longitude_max)
        )

        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        all_variables = variables + ["time", "latitude", "longitude"]

        # check that the variables are in the observation data
        schema_fields = [field for field in subset_observation_data.collect_schema()]
        if any(var not in schema_fields for var in all_variables):
            raise ValueError(f"Variables {all_variables} not found in observation data")

        # subset the variables
        subset_observation_data = subset_observation_data.select(all_variables)

        return subset_observation_data

    def _maybe_convert_to_dataset(self, data):
        """
        Convert the observation data to an xarray dataset

        Args:
            data: The observation data to convert to an xarray dataset.

        Returns:
            The observation data as an xarray dataset.
        """
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
    Observation class for local storm report data.

    Returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods.
    """

    def __init__(self, case: case.IndividualCase):
        super().__init__(case)

    def _open_data_from_source(
        self, source: str, storage_options: Optional[dict] = None
    ):
        """
        Open the observation data from the source.

        Args:
            source: The source of the observation data.
            **kwargs: Additional keyword arguments to pass to the data loading function.

        Returns:
            The observation data.
        """

        observation_data = pd.read_parquet(source, storage_options=storage_options)

        return observation_data

    def _subset_data_to_case(
        self, observation_data: pd.DataFrame, variables: list[str]
    ):
        """
        Subset the observation data to the case.

        Args:
            observation_data: The observation data to subset to the case.
            variables: The variables to subset.

        Returns:
            The subset observation data.
        """

        # latitude, longitude are strings by default, convert to float
        observation_data["lat"] = observation_data["lat"].astype(float)
        observation_data["lon"] = observation_data["lon"].astype(float)
        observation_data["time"] = pd.to_datetime(observation_data["time"])

        central_conus_bbox = [24.0, 49.0, -109.0, -89.0]  # Mississippi River to Rockies
        # TODO: fix case to automatically apply these; currently stand-in for now
        self.case.latitude_min = central_conus_bbox[0]
        self.case.latitude_max = central_conus_bbox[1]
        self.case.longitude_min = central_conus_bbox[2]
        self.case.longitude_max = central_conus_bbox[3]

        filters = (
            (observation_data["time"] >= self.case.start_date)
            & (observation_data["time"] <= self.case.end_date)
            & (observation_data["lat"] >= self.case.latitude_min)
            & (observation_data["lat"] <= self.case.latitude_max)
            & (observation_data["lon"] >= self.case.longitude_min)
            & (observation_data["lon"] <= self.case.longitude_max)
        )

        subset_observation_data = observation_data.loc[filters]

        subset_observation_data = subset_observation_data.rename(
            columns={"lat": "latitude", "lon": "longitude", "time": "valid_time"}
        )
        return subset_observation_data

    def _maybe_convert_to_dataset(self, data):
        """
        Convert the observation data to an xarray dataset

        Args:
            data: The observation data to convert to an xarray dataset.

        Returns:
            The observation data as an xarray dataset.
        """
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

    def __init__(self, case: case.IndividualCase):
        super().__init__(case)

    def _open_data_from_source(
        self, source: str, storage_options: Optional[dict] = None
    ):
        """
        Open the IBTrACS data from the source.

        Args:
            source: The source of the IBTrACS data.
            storage_options: Optional storage options for the source.

        Returns:
            The IBTrACS data as an xarray Dataset.
        """
        # not using storage_options in this case due to NetCDF4Backend not supporting them
        observation_data: xr.Dataset = xr.open_dataset(
            source, engine="h5netcdf", chunks="auto"
        )
        return observation_data

    def _subset_data_to_case(self, observation_data: xr.Dataset):
        """
        Subset the observation data to the case.

        Args:
            observation_data: The observation data to subset.

        Returns:
            The subsetted observation data.
        """

    def _maybe_convert_to_dataset(self, data):
        """
        Convert the observation data to an xarray dataset if it is not already.

        Args:
            data: The observation data to convert.

        Returns:
            The observation data as an xarray dataset.
        """
        if not isinstance(data, xr.Dataset):
            data = data.to_dataset()
        return data

    def harmonize_forecast_data(
        self, forecast_data: xr.Dataset, forecast_variable: str
    ):
        """
        Harmonize the forecast data to the observation data.

        Args:
            forecast_data: The forecast data to harmonize.
            forecast_variable: The forecast variable to harmonize.

        Returns:
            The harmonized forecast data.
        """
        # Implementation to be added
        pass

    def run_pipeline(self, source: str, storage_options: Optional[dict] = None):
        """
        Run the observation pipeline.

        Args:
            source: The source of the observation data.
            storage_options: Optional storage options for the source.

        Returns:
            The processed observation data.
        """
        observation_data = self._open_data_from_source(source, storage_options)
        observation_data = self.subset_data_to_case(observation_data)
        observation_data = self._maybe_convert_to_dataset(observation_data)
        return observation_data
