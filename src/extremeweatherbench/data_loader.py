import xarray as xr
from extremeweatherbench import config, utils
import pandas as pd
import logging
import dataclasses
from typing import Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def open_forecast_dataset(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig,
) -> xr.Dataset:
    """Open the forecast dataset specified for evaluation.

    If a URI is provided (e.g. s3://bucket/path/to/forecast), the filesystem
    will be inferred from the provided source (in this case, s3). Otherwise,
    the filesystem will assumed to be local.
    Args:
        eval_config: The evaluation configuration.
        forecast_schema_config: The forecast schema configuration.

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
    forecast_ds = _rename_forecast_dataset(forecast_ds, forecast_schema_config)
    forecast_ds = utils.maybe_convert_dataset_lead_time_to_int(forecast_ds)

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
    if eval_config.forecast_source == "cira":
        kerchunk_ds = kerchunk_ds.rename({"time": "lead_time"})

        # The evaluation configuration is used to set the lead time range and resolution.
        kerchunk_ds["lead_time"] = range(
            eval_config.timestep_begin,
            (
                eval_config.timestep_begin
                + eval_config.output_timesteps * eval_config.temporal_resolution_hours
            ),
            eval_config.temporal_resolution_hours,
        )
    else:
        raise NotImplementedError(
            "Only CIRA data is currently supported for kerchunked reference files."
        )
    return kerchunk_ds


def open_obs_datasets(
    eval_config: config.Config,
) -> Tuple[Optional[pd.DataFrame], Optional[xr.Dataset]]:
    """Open the observation datasets specified for evaluation.

    Args:
        eval_config: The evaluation configuration.

    Returns:
        The point observation dataset and the gridded observation dataset.
    """
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path:
        point_obs = pd.read_parquet(
            eval_config.point_obs_path, storage_options=dict(token="anon")
        )
    if eval_config.gridded_obs_path:
        gridded_obs = xr.open_zarr(
            eval_config.gridded_obs_path,
            chunks=None,
            storage_options=dict(token="anon"),
        )
    return point_obs, gridded_obs


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
