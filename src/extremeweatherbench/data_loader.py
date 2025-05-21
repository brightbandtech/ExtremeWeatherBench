import abc
import dataclasses
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from extremeweatherbench import case, config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Storage location for IBTrACS data.
IBTRACS_URI = "gs://extremeweatherbench/IBTrACS.since1980.v04r01.nc"

#: Storage location for GHCN.
GHCN_URI = "gs://extremeweatherbench/ghcnh.parquet"

#: Storage location for ERA5.
ERA5_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

#: Storage location for storm reports. TODO: add storm reports to bucket
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

    def load_observations(self, cases: List[case.IndividualCase]) -> pd.DataFrame:
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

        # Filter to only include cases of interest
        case_dates = [pd.Timestamp(c.start_date) for c in cases]
        df = df[df.time.isin(case_dates)]

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


def open_and_preprocess_forecast_dataset(
    eval_config: config.Config,
) -> xr.Dataset:
    """Open the forecast dataset specified for evaluation.

    If a URI is provided (e.g. s3://bucket/path/to/forecast), the filesystem
    will be inferred from the provided source (in this case, s3). Otherwise,
    the filesystem will assumed to be local.

    Preprocessing examples:
        A typical preprocess function handles metadata changes:

        def _preprocess_cira_forecast_dataset(
            ds: xr.Dataset
        ) -> xr.Dataset:
            ds = ds.rename({"time": "lead_time"})
            return ds

        The preprocess function is applied before variable renaming occurs, so it should
        reference the original variable names in the forecast dataset, not the standardized
        names defined in the ForecastSchemaConfig.

    Args:
        eval_config: The evaluation configuration.
        forecast_schema_config: The forecast schema configuration.
        preprocess: A function that preprocesses the forecast dataset.

    Returns:
        The opened forecast dataset.
    """
    logger.debug("Opening forecast dataset")
    if "zarr" in eval_config.forecast_dir:
        forecast_ds = xr.open_zarr(eval_config.forecast_dir, chunks="auto")
    elif (
        "parq" in eval_config.forecast_dir
        or "json" in eval_config.forecast_dir
        or "parquet" in eval_config.forecast_dir
    ):
        forecast_ds = open_kerchunk_reference(eval_config)
    else:
        raise TypeError(
            "Unknown file type found in forecast path, only json, parquet, and zarr are supported."
        )
    forecast_ds = eval_config.forecast_preprocess(forecast_ds)
    forecast_ds = _rename_forecast_dataset(
        forecast_ds, eval_config.forecast_schema_config
    )
    forecast_ds = _maybe_convert_dataset_lead_time_to_int(eval_config, forecast_ds)

    return forecast_ds


def open_kerchunk_reference(
    eval_config: config.Config,
) -> xr.Dataset:
    """Open a dataset from a kerchunked reference file in parquet or json format.
    This has been built for the CIRA MLWP S3 bucket's data (https://registry.opendata.aws/aiwp/),
    but can work with other data in the future. Currently only supports CIRA data unless
    schema is identical to the CIRA schema.

    Args:
        file: The path to the kerchunked reference file.
        remote_protocol: The remote protocol to use.

    Returns:
        The opened dataset.
    """
    if "parq" in eval_config.forecast_dir or "parquet" in eval_config.forecast_dir:
        storage_options = {
            "remote_protocol": eval_config.remote_protocol,
            "remote_options": {"anon": True},
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        kerchunk_ds = xr.open_dataset(
            eval_config.forecast_dir,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
        kerchunk_ds = kerchunk_ds.compute()
    elif "json" in eval_config.forecast_dir:
        kerchunk_ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": eval_config.forecast_dir,
                    "remote_protocol": eval_config.remote_protocol,
                    "remote_options": {"anon": True},
                },
            },
        )
    else:
        raise TypeError(
            "Unknown kerchunk file type found in forecast path, only json and parquet are supported."
        )
    return kerchunk_ds


def open_obs_datasets(
    eval_config: config.Config,
) -> Tuple[Optional[pd.DataFrame], Optional[xr.Dataset]]:
    """Open the observation datasets specified for evaluation.

    Args:
        eval_config: The evaluation configuration.
        point_obs_schema_config: The point observation schema configuration.

    Returns:
        The point observation dataset and the gridded observation dataset.
    """
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path:
        raw_point_obs = pd.read_parquet(
            eval_config.point_obs_path, storage_options=dict(token="anon")
        )
        point_obs = _rename_point_obs_dataset(
            raw_point_obs, eval_config.point_obs_schema_config
        )
        point_obs.attrs = {
            "metadata_vars": eval_config.point_obs_schema_config.mapped_metadata_vars
        }
    if eval_config.gridded_obs_path:
        gridded_obs = xr.open_zarr(
            eval_config.gridded_obs_path,
            chunks=None,
            storage_options=dict(token="anon"),
        )
    return point_obs, gridded_obs


def _rename_point_obs_dataset(
    point_obs: pd.DataFrame,
    point_obs_schema_config: config.PointObservationSchemaConfig,
) -> pd.DataFrame:
    """Rename the point observation dataset to the correct names expected by the evaluation routines.

    Args:
        point_obs: The point observation dataframe to rename.
        point_obs_schema_config: The point observation schema configuration.

    Returns:
        The renamed point observation dataframe.
    """
    # Mapping here is used to rename the incoming data variables to the correct
    # names expected by the evaluation routines.
    mapping = {
        getattr(point_obs_schema_config, field.name): field.name
        for field in dataclasses.fields(point_obs_schema_config)
        if field.type is not List[str]
    }

    # Application of the mapping to coords and data variables.
    filtered_mapping_columns = {
        old: new for old, new in mapping.items() if old in point_obs.columns
    }

    # Dataframes need "columns=" unlike xarray datasets
    renamed_point_obs = point_obs.rename(columns=filtered_mapping_columns)
    return renamed_point_obs


def _rename_forecast_dataset(
    forecast_ds: xr.Dataset, forecast_schema_config: config.ForecastSchemaConfig
) -> xr.Dataset:
    """Rename the forecast dataset to the correct names expected by the evaluation routines.

    Args:
        forecast_ds: The forecast dataset to rename.
        forecast_schema_config: The forecast schema configuration.

    Returns:
        The renamed forecast dataset.
    """
    # Mapping here is used to rename the incoming data variables to the correct
    # names expected by the evaluation routines.
    mapping = {
        getattr(forecast_schema_config, field.name): field.name
        for field in dataclasses.fields(forecast_schema_config)
    }

    # Application of the mapping to coords and data variables.
    filtered_mapping_data_vars = {
        old: new for old, new in mapping.items() if old in forecast_ds.data_vars
    }
    filtered_mapping_coords = {
        old: new for old, new in mapping.items() if old in forecast_ds.coords
    }

    # Combine the mapping for coords and data variables and rename them in the forecast dataset.
    filtered_mapping = {**filtered_mapping_data_vars, **filtered_mapping_coords}
    forecast_ds = forecast_ds.rename(filtered_mapping)

    return forecast_ds


def _preprocess_cira_forecast_dataset(
    eval_config: config.Config, ds: xr.Dataset
) -> xr.Dataset:
    """An example preprocess function that renames the time coordinate to lead_time
    and sets the lead time range and resolution.

    Args:
        eval_config: The evaluation configuration.
        ds: The forecast dataset to rename.

    Returns:
        The renamed forecast dataset.
    """
    ds = ds.rename({"time": "lead_time"})

    # The evaluation configuration is used to set the lead time range and resolution.
    ds["lead_time"] = range(
        eval_config.init_forecast_hour,
        (
            eval_config.init_forecast_hour
            + eval_config.output_timesteps * eval_config.temporal_resolution_hours
        ),
        eval_config.temporal_resolution_hours,
    )
    return ds


def _maybe_convert_dataset_lead_time_to_int(
    eval_config: config.Config, dataset: xr.Dataset
) -> xr.Dataset:
    """Convert types of variables in an xarray Dataset based on the schema,
    ensuring that, for example, the variable representing lead_time is of type int.

    Args:
        dataset: The input xarray Dataset that uses the schema's variable names.

    Returns:
        An xarray Dataset with adjusted types.
    """

    var = dataset["lead_time"]
    if var.dtype == np.dtype("timedelta64[ns]"):
        dataset["lead_time"] = (var / np.timedelta64(1, "h")).astype(int)
    elif var.dtype == np.dtype("int64"):
        logger.info("lead_time is already an int, skipping conversion")
    else:
        logger.info(
            "lead_time is not a timedelta64[ns] or int64, creating range based on "
            "init_forecast_hour, output_timesteps, and temporal_resolution_hours"
        )
        dataset["lead_time"] = range(
            eval_config.init_forecast_hour,
            (
                eval_config.init_forecast_hour
                + eval_config.output_timesteps * eval_config.temporal_resolution_hours
            ),
            eval_config.temporal_resolution_hours,
        )
    return dataset
