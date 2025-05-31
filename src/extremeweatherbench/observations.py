import abc
import logging
from typing import Dict, Type

import pandas as pd
import xarray as xr

from extremeweatherbench import case

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Storage location for IBTrACS data.
IBTRACS_URI = "gs://extremeweatherbench/IBTrACS.since1980.v04r01.nc"

#: Storage location for GHCN.
GHCN_URI = "gs://extremeweatherbench/ghcnh.parquet"

#: Storage location for ERA5.
ERA5_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

#: Storage location for storm reports.
STORM_REPORT_URI = "gs://extremeweatherbench/storm_reports.parq"


class ObservationHandler(abc.ABC):
    """An abstract base class for different observation types.

    This class defines the interface for loading and subsetting observations
    for a given event type. Subclasses must implement the `observation_type`,
    `load_observations`, and `subset_observation_for_case` methods.

    Args:
        observation_type: The type of observation to load.
        event_type: The type of event the observation is for.
    """

    def __init__(self, observation_type: str, event_type: str):
        self.observation_type = observation_type
        self.event_type = event_type

    @property
    @abc.abstractmethod
    def observation_type(self) -> str:
        pass

    @abc.abstractmethod
    def load_observations(self) -> pd.DataFrame | xr.Dataset:
        pass

    @abc.abstractmethod
    def subset_observation_for_case(
        self, case: case.IndividualCase
    ) -> pd.DataFrame | xr.Dataset:
        """Subset the observation for a given case."""
        pass


class IBTrACSHandler(ObservationHandler):
    """A subclass of ObservationHandler that loads IBTrACS (hurricane track data) observations."""

    def __init__(self, observation_type: str, event_type: str):
        super().__init__(observation_type, event_type)

    @property
    def observation_type(self) -> str:
        return "ibtracs"

    def load_observations(self) -> pd.DataFrame:
        """Load IBTrACS observations from GCS.

        Args:
            cases: A list of IndividualCase objects.

        Returns:
            A pandas DataFrame containing the IBTrACS observations.
        """

        # Open the IBTrACS dataset from GCS using fsspec
        ds = xr.open_dataset(IBTRACS_URI, storage_options=dict(token="anon"))

        # Convert to pandas DataFrame
        df = ds.to_dataframe().reset_index()

        return df


class GHCNHandler(ObservationHandler):
    """A subclass of ObservationHandler that loads gridded Global Historical Climatology Network (GHCN) observations."""

    def __init__(self, observation_type: str, event_type: str):
        super().__init__(observation_type, event_type)

    @property
    def observation_type(self) -> str:
        return "ghcn"

    def load_observations(self) -> pd.DataFrame:
        """Load GHCN observations from GCS.

        Args:
            cases: A list of IndividualCase objects.

        Returns:
            A pandas DataFrame containing the GHCN observations.
        """
        # Open the IBTrACS dataset from GCS using fsspec
        raw_point_obs = pd.read_parquet(GHCN_URI, storage_options=dict(token="anon"))
        return raw_point_obs


class ERA5Handler(ObservationHandler):
    """A subclass of ObservationHandler that loads ERA5 gridded observations
    from the Google Cloud Storage bucket."""

    def __init__(self, observation_type: str, event_type: str):
        super().__init__(observation_type, event_type)

    @property
    def observation_type(self) -> str:
        return "era5"

    def load_observations(self) -> xr.Dataset:
        """Load ERA5 observations from GCS.

        Args:
            cases: A list of IndividualCase objects.

        Returns:
            A xarray Dataset containing the ERA5 observations.
        """
        ds = xr.open_zarr(
            ERA5_URI,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return ds


class StormReportHandler(ObservationHandler):
    """A subclass of ObservationHandler that loads severe storm reports."""

    def __init__(self, observation_type: str, event_type: str):
        super().__init__(observation_type, event_type)

    @property
    def observation_type(self) -> str:
        return "storm_report"

    def load_observations(self) -> pd.DataFrame:
        """Load storm report observations from GCS.

        Args:
            cases: A list of IndividualCase objects.

        Returns:
            A pandas DataFrame containing the storm report observations.
        """
        # Open the storm report dataset from GCS using fsspec
        df = pd.read_parquet(STORM_REPORT_URI, storage_options=dict(token="anon"))

        return df


# Create a mapping of observation type strings to their handler classes
OBSERVATION_HANDLERS: Dict[str, Type[ObservationHandler]] = {
    "ghcn": GHCNHandler,
    "era5": ERA5Handler,
    "ibtracs": IBTrACSHandler,
    "storm_report": StormReportHandler,
}


def create_observation_handler(
    observation_type: str, event_type: str
) -> ObservationHandler:
    """Create an observation handler instance based on the observation type string.

    Args:
        observation_type: String identifier for the observation type (e.g. "ghcn")
        event_type: The type of event this observation is for

    Returns:
        An instance of the appropriate ObservationHandler subclass

    Raises:
        ValueError: If the observation_type is not recognized
    """
    handler_class = OBSERVATION_HANDLERS.get(observation_type.lower())
    if handler_class is None:
        raise ValueError(f"Unknown observation type: {observation_type}")
    return handler_class(observation_type, event_type)
