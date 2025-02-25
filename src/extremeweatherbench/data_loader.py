import fsspec
import xarray as xr
from extremeweatherbench import config
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def open_forecast_dataset(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig,
):
    """Open the forecast dataset specified for evaluation."""
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
            forecast_dataset = open_mlwp_kerchunk_reference(
                eval_config.forecast_dir, forecast_schema_config
            )
        elif "zarr" in eval_config.forecast_dir:
            forecast_dataset = xr.open_zarr(eval_config.forecast_dir, chunks="auto")
    elif len(file_types) == 1:
        if "zarr" in file_types:
            forecast_dataset = xr.open_zarr(file_list, chunks="auto")
        elif "json" in file_types:
            forecast_dataset = open_mlwp_kerchunk_reference(
                file_list[0], forecast_schema_config
            )
        else:
            raise TypeError("Unknown file type found in forecast path.")
    else:
        raise FileNotFoundError("No files found in forecast path.")

    mapping = forecast_schema_config.to_mapping()
    filtered_mapping_data_vars = {
        old: new for old, new in mapping.items() if old in forecast_dataset.data_vars
    }
    filtered_mapping_coords = {
        old: new for old, new in mapping.items() if old in forecast_dataset.coords
    }
    filtered_mapping = {**filtered_mapping_data_vars, **filtered_mapping_coords}
    forecast_dataset = forecast_dataset.rename(filtered_mapping)
    forecast_dataset = forecast_schema_config.conversions(forecast_dataset)

    return forecast_dataset


def open_mlwp_kerchunk_reference(
    file,
    forecast_schema_config: config.ForecastSchemaConfig,
    remote_protocol: str = "s3",
) -> xr.Dataset:
    """Open a dataset from a kerchunked reference file for the OAR MLWP S3 bucket."""
    if "parq" in file:
        storage_options = {
            "remote_protocol": remote_protocol,
            "remote_options": {"anon": True},
        }  # options passed to fsspec
        open_dataset_options: dict = {"chunks": {}}  # opens passed to xarray

        ds = xr.open_dataset(
            file,
            engine="kerchunk",
            storage_options=storage_options,
            open_dataset_options=open_dataset_options,
        )
        ds = ds.compute()
    else:
        ds = xr.open_dataset(
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
    ds = ds.rename({"time": "lead_time"})
    ds["lead_time"] = range(0, 241, 6)
    for variable in forecast_schema_config.__dict__:
        attr_value = getattr(forecast_schema_config, variable)
        if attr_value in ds.data_vars:
            ds = ds.rename({attr_value: variable})
    return ds


def open_obs_datasets(eval_config: config.Config):
    """Open the observation datasets specified for evaluation."""
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
