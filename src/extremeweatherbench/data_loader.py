import fsspec
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

    Args:
        eval_config: The evaluation configuration.
        forecast_schema_config: The forecast schema configuration.

    Returns:
        The opened forecast dataset.
    """
    logger.debug("Opening forecast dataset")
    if "://" in eval_config.forecast_dir:
        filesystem = eval_config.forecast_dir.split("://")[0]
    else:
        filesystem = "file"
    fs = fsspec.filesystem(filesystem)
    file_list = fs.ls(eval_config.forecast_dir)
    file_types = set([file.split(".")[-1] for file in file_list])

    if len(file_types) > 1:
        if (
            "parq" not in eval_config.forecast_dir
            and "zarr" not in eval_config.forecast_dir
        ):
            raise TypeError("Multiple file types found in forecast path.")
        if "parq" in eval_config.forecast_dir:
            # kerchunked parq refs seem to consistently need two "compute" calls
            # the first one doesn't actually load to memory
            forecast_ds = open_kerchunk_reference(
                eval_config.forecast_dir, forecast_schema_config
            )
        elif "zarr" in eval_config.forecast_dir:
            forecast_ds = xr.open_zarr(eval_config.forecast_dir, chunks="auto")
    elif len(file_types) == 1:
        if "zarr" in file_types:
            forecast_ds = xr.open_zarr(file_list, chunks="auto")
        elif "json" in file_types:
            forecast_ds = open_kerchunk_reference(file_list[0], forecast_schema_config)
        else:
            raise TypeError(
                "Unknown file type found in forecast path, only json, parquet, and zarr are supported."
            )
    else:
        raise FileNotFoundError("No files found in forecast path.")

    mapping = {
        getattr(forecast_schema_config, field.name): field.name
        for field in dataclasses.fields(forecast_schema_config)
    }

    filtered_mapping_data_vars = {
        old: new for old, new in mapping.items() if old in forecast_ds.data_vars
    }
    filtered_mapping_coords = {
        old: new for old, new in mapping.items() if old in forecast_ds.coords
    }
    filtered_mapping = {**filtered_mapping_data_vars, **filtered_mapping_coords}
    forecast_ds = forecast_ds.rename(filtered_mapping)
    forecast_ds = utils.convert_dataset_lead_time_to_int(forecast_ds)

    return forecast_ds


def open_kerchunk_reference(
    file,
    forecast_schema_config: config.ForecastSchemaConfig,
    remote_protocol: str = "s3",
) -> xr.Dataset:
    """Open a dataset from a kerchunked reference file in parquet or json format.
    This has been built for the CIRA MLWP S3 bucket's data, but can work with other
    data as well if it is 6 hourly.

    Args:
        file: The path to the kerchunked reference file.
        forecast_schema_config: The forecast schema configuration.
        remote_protocol: The remote protocol to use.

    Returns:
        The opened dataset.
    """
    if "parq" in file:
        storage_options = {
            "remote_protocol": remote_protocol,
            "remote_options": {"anon": True},
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        kerchunk_ds = xr.open_dataset(
            file,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
        kerchunk_ds = kerchunk_ds.compute()
    elif "json" in file:
        kerchunk_ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": file,
                    "remote_protocol": remote_protocol,
                    "remote_options": {"anon": True},
                },
            },
        )
    else:
        raise TypeError(
            "Unknown file type found in forecast path, only json and parquet are supported."
        )
    kerchunk_ds = kerchunk_ds.rename({"time": "lead_time"})
    kerchunk_ds["lead_time"] = range(0, 241, 6)
    for variable in forecast_schema_config.__dict__:
        attr_value = getattr(forecast_schema_config, variable)
        if attr_value in kerchunk_ds.data_vars:
            kerchunk_ds = kerchunk_ds.rename({attr_value: variable})
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
    if point_obs is None and gridded_obs is None:
        raise FileNotFoundError("No gridded or point observation data provided.")
    return point_obs, gridded_obs
