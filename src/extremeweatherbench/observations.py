# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import logging
from abc import ABC, abstractmethod
from typing import Tuple
from datetime import datetime

# %%
import xarray as xr
import polars as pl
import pandas as pd

# %%
from extremeweatherbench import case, utils, config

# %%
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %%
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
        self, source: str, case: case.IndividualCase, variables: list[str]
    ):  # TODO: add Variable type to include here alongside str
        self.source = source
        self.case = case
        self.variables = variables

    @abstractmethod
    def open_data_from_source(self):
        """
        Open the observation data from the source, opting to avoid loading the entire dataset into memory.
        """
        pass

    @abstractmethod
    def subset_data_to_case(self):
        """
        Subset the observation data to the case.
        """
        pass

    @abstractmethod
    def maybe_convert_to_dataset(self):
        """
        Convert the observation data to an xarray dataset if it is not already.
        """
        pass

    @abstractmethod
    def harmonize_forecast_data(
        self, forecast_data: xr.Dataset, forecast_variable: str
    ) -> Tuple[xr.Dataset, xr.Dataset]:  # TODO: add Variable type
        """
        Harmonize the forecast data to the observation data.

        Args:
            forecast_data: The forecast data to harmonize.
            forecast_variable: The variable to harmonize. Can be a single variable or a derived variable.

        Returns:
            A tuple of the harmonized forecast data and the harmonized observation data.
        """
        pass
    
    @abstractmethod
    def run_pipeline(self) -> xr.Dataset:
        """
        Run the observation pipeline. 
        
        The simple pipeline is:
        open_data_from_source()
        subset_data_to_case()
        maybe_convert_to_dataset()

        But this method can be customized to include more steps.
        """
        self.open_data_from_source()
        self.subset_data_to_case()
        self.maybe_convert_to_dataset()


# %%
class ERA5(Observation):
    """
    Observation class for ERA5 gridded data.
    """

    def __init__(self, source: str, case: case.IndividualCase, variables: list[str]):
        super().__init__(source, case, variables)

    def open_data_from_source(self):
        self.observation_data = xr.open_zarr(
            self.source,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        self.data_pattern = self.observation_data.dims

    def subset_data_to_case(self):
        self.subset_observation_data = self.observation_data.sel(
            time=slice(self.case.start_date, self.case.end_date),
            latitude=slice(self.case.latitude_min, self.case.latitude_max),
            longitude=slice(self.case.longitude_min, self.case.longitude_max),
        )

        # check that the variables are in the observation data
        if any(var not in self.observation_data.data_vars for var in self.variables):
            raise ValueError(
                f"Variables {self.variables} not found in observation data"
            )

        # subset the variables
        self.subset_observation_data = self.subset_observation_data[self.variables]

    def harmonize_forecast_data(
        self, forecast_data: xr.Dataset, forecast_variable: str
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Harmonize the forecast data to the observation data.
        """
        # Check if observation data exists before proceeding
        if not hasattr(self, "observation_data"):
            logger.error("Observation data not loaded. Call open_data_from_source() first.")
            raise ValueError("Observation data not loaded. Call open_data_from_source() first.")

        harmonized_forecast, harmonized_obs = xr.align(
            self.observation_data,
            forecast_data,
            join="inner",
        )
    
    def maybe_convert_to_dataset(self):
        if not isinstance(self.observation_data, xr.Dataset):
            self.observation_data = self.observation_data.to_dataset()
    
    def run_pipeline(self):
        self.open_data_from_source()
        self.subset_data_to_case()
        self.maybe_convert_to_dataset()


# %%
class GHCN(Observation):
    """
    Observation class for GHCN gridded data.
    """

    def __init__(self, source: str, case: case.IndividualCase, variables: list[str]):
        super().__init__(source, case, variables)

        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        self.variables = self.variables + ["time", "latitude", "longitude"]
    def open_data_from_source(self):
        observation_data: pl.LazyFrame = pl.scan_parquet(
            self.source
        )

        return observation_data
        
    def subset_data_to_case(self, observation_data: pl.LazyFrame):
        """
        Subset the observation data to the case.

        Args:
            observation_data: The observation data to subset to the case.

        Returns:
            The subset observation data.
        """
        # Create filter expressions for LazyFrame
        time_min = self.case.start_date - pd.Timedelta(days=2)
        time_max = self.case.end_date + pd.Timedelta(days=2)

        #TODO: fix case to automatically apply these; currently stand-in for now
        lat_min = self.case.location.latitude - self.case.bounding_box_degrees/2
        lat_max = self.case.location.latitude + self.case.bounding_box_degrees/2
        lon_min = self.case.location.longitude - self.case.bounding_box_degrees/2
        lon_max = self.case.location.longitude + self.case.bounding_box_degrees/2
        
        # Apply filters using proper polars expressions
        subset_observation_data = observation_data.filter(
            (pl.col("time") >= time_min) &
            (pl.col("time") <= time_max) &
            (pl.col("latitude") >= lat_min) &
            (pl.col("latitude") <= lat_max) &
            (pl.col("longitude") >= lon_min) &
            (pl.col("longitude") <= lon_max)
        )

        # check that the variables are in the observation data
        if any(var not in subset_observation_data.collect_schema() for var in self.variables):
            raise ValueError(
                f"Variables {self.variables} not found in observation data"
            )

        # subset the variables
        subset_observation_data = subset_observation_data.select(self.variables)

        return subset_observation_data
    
    def harmonize_forecast_data(
        self, observation_data: xr.Dataset, forecast_data: xr.Dataset, forecast_variable: str
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Harmonize the forecast data to the observation data.
        """
        # Check if observation data exists before proceeding
        if not hasattr(self, "observation_data"):
            logger.error("Observation data not loaded. Call open_data_from_source() first.")
            raise ValueError("Observation data not loaded. Call open_data_from_source() first.")

        harmonized_forecast, harmonized_obs = xr.align(
            observation_data,
            forecast_data,
            join="inner",
        )
    
    def maybe_convert_to_dataset(self, observation_data: pl.LazyFrame):
        """
        Convert the observation data to an xarray dataset

        Args:
            observation_data: The observation data to convert to an xarray dataset.

        Returns:
            The observation data as an xarray dataset.
        """
        observation_data = observation_data.collect().to_pandas()
        observation_data = observation_data.set_index(['time', 'latitude', 'longitude'])
        observation_data = observation_data.to_xarray()
        return observation_data

    def run_pipeline(self):
        """
        Run GHCN obs pipeline.
        """
        observation_data = self.open_data_from_source()
        observation_data = self.subset_data_to_case(observation_data)
        observation_data = self.maybe_convert_to_dataset(observation_data)

        return observation_data


# %%
class LSR(Observation):
    """
    Observation class for local storm report data. 

    Returns a dataset with LSRs and practically perfect hindcast gridded
    probability data.
    """

    def __init__(self, source: str, case: case.IndividualCase, variables: list[str]):
        super().__init__(source, case, variables)

    def open_data_from_source(self):
        """
        Open the observation data from the source.
        """
        observation_data: pl.LazyFrame = pd.read_csv(
            self.source
        )

        return observation_data
    
    def subset_data_to_case(self, observation_data: pd.DataFrame):
        """
        Subset the observation data to the case.

        Args:
            observation_data: The observation data to subset to the case.

        Returns:
            The subset observation data.
        """
        filters = [
            (observation_data['time'] >= self.case.start_date) &
            (observation_data['time'] <= self.case.end_date) &
            (observation_data['latitude'] >= self.case.latitude_min) &
            (observation_data['latitude'] <= self.case.latitude_max) &
            (observation_data['longitude'] >= self.case.longitude_min) &
            (observation_data['longitude'] <= self.case.longitude_max)
        ]

        subset_observation_data = observation_data.loc[filters]

        return subset_observation_data
    
    def maybe_convert_to_dataset(self, observation_data: pd.DataFrame):
        """
        Convert the observation data to an xarray dataset

        Args:
            observation_data: The observation data to convert to an xarray dataset.

        Returns:
            The observation data as an xarray dataset.
        """
        observation_data = observation_data.set_index(['time', 'latitude', 'longitude'])
        observation_data = observation_data.to_xarray()
        return observation_data
    
    def harmonize_forecast_data(self, forecast_data: xr.Dataset, forecast_variable: str):
        """
        Harmonize the forecast data to the observation data.
        """
        pass
    
    def run_pipeline(self):
        """
        Run the observation pipeline.
        """
        observation_data = self.open_data_from_source()
        observation_data = self.subset_data_to_case(observation_data)
        observation_data = self.maybe_convert_to_dataset(observation_data)
        return observation_data



# %%
class IBTrACS(Observation):
    """
    Observation class for IBTrACS data.
    """

    def __init__(self, source: str, case: case.IndividualCase, variables: list[str]):
        super().__init__(source, case, variables)

    def open_data_from_source(self):
        observation_data: xr.Dataset = xr.open_dataset(
            self.source,
            engine="netcdf4", 
            chunks='auto'
        )
        return observation_data
    
    def subset_data_to_case(self, observation_data: xr.Dataset):
        """
        Subset the observation data to the case.
        """
        pass
    
    def maybe_convert_to_dataset(self, observation_data: xr.Dataset):
        """
        Convert the observation data to a dataset if it is not already.
        """
        if not isinstance(observation_data, xr.Dataset):
            observation_data = observation_data.to_dataset()
        return observation_data
    
    def harmonize_forecast_data(self, forecast_data: xr.Dataset, forecast_variable: str):
        """
        Harmonize the forecast data to the observation data.
        """
        pass

    def run_pipeline(self):
        """
        Run the observation pipeline.
        """
        observation_data = self.open_data_from_source()
        observation_data = self.subset_data_to_case(observation_data)
        observation_data = self.maybe_convert_to_dataset(observation_data)
        return observation_data
