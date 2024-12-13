import xarray as xr
import pandas as pd
import logging
import fsspec
import numpy as np
from typing import Optional
from . import events
from . import config
from . import utils
from . import case

def evaluate(eval_config: config.Config, 
             forecast_schema_config: Optional[config.ForecastSchemaConfig] = config.ForecastSchemaConfig()):
    """
    Evaluate the forecast data against the observed data, looping
    through each case for each event type selected.
    Attributes:
        eval_config: config.Config: the configuration object for the evaluation
    """
    point_obs, gridded_obs = _open_obs_datasets(eval_config)
    forecast_dataset = _open_forecast_dataset(eval_config, forecast_schema_config)
    for event in eval_config.event_types:
        _evaluate_event_loop(event, forecast_dataset, gridded_obs, point_obs)


def _evaluate_event_loop(event: events._Event, 
                         forecast_dataset: xr.Dataset,
                         gridded_obs: Optional[xr.Dataset],
                         point_obs: Optional[pd.DataFrame], 
                         ):
    case_container = case.Case(event, individual_case)
    for individual_case in event.case_df.itertuples():
        if point_obs is not None:
            raise NotImplementedError("Point observation data evaluation not yet implemented.")
        if gridded_obs is not None:
            
            results = _evaluate_case_gridded(case, forecast_dataset, gridded_obs)

def _evaluate_case_gridded(case, forecast_dataset: xr.Dataset, gridded_obs: xr.Dataset):



def _open_obs_datasets(eval_config: config.Config):
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path is not None:
        point_obs = pd.read_parquet(eval_config.point_obs_path, chunks='auto')
    if eval_config.gridded_obs_path is not None:
        gridded_obs = xr.open_zarr(eval_config.gridded_obs_path, chunks='auto')
    if point_obs is None and gridded_obs is None:
        raise ValueError("No grided or point observation data provided.")
    return point_obs, gridded_obs

def _open_forecast_dataset(eval_config: config.Config, forecast_schema_config: Optional[config.ForecastSchemaConfig] = None):
    logging.info("Opening forecast dataset")
    if eval_config.forecast_path.startswith("s3://"):
        fs = fsspec.filesystem('s3')
    elif eval_config.forecast_path.startswith("gcs://") or eval_config.forecast_path.startswith("gs://"):
        fs = fsspec.filesystem('gcs')
    else:
        fs = fsspec.filesystem('file')

    file_list = fs.ls(eval_config.forecast_path)
    file_types = set([file.split('.')[-1] for file in file_list])
    if len(file_types) > 1:
        raise ValueError("Multiple file types found in forecast path.")
    
    if 'zarr' in file_types and len(file_list) == 1:
        forecast_dataset = xr.open_zarr(file_list, chunks='auto')
    elif 'zarr' in file_types and len(file_list) > 1:
        raise ValueError("Multiple zarr files found in forecast path, please provide a single zarr file.")
    
    if 'nc' in file_types:
        logging.warning("NetCDF files are not recommended for large datasets. Consider converting to zarr.")
        forecast_dataset = xr.open_mfdataset(file_list, chunks='auto')

    if 'json' in file_types:
        forecast_dataset = _open_kerchunk_zarr_reference_jsons(file_list, forecast_schema_config)

    return forecast_dataset

def _open_kerchunk_zarr_reference_jsons(file_list, forecast_schema_config):
    xarray_datasets = []
    for json_file in file_list:
        fs_ = fsspec.filesystem("reference", fo=json_file, ref_storage_args={'skip_instance_cache':True},
                        remote_protocol='gcs', remote_options={'anon':True})
        m = fs_.get_mapper("")
        ds = xr.open_dataset(m, engine="zarr", backend_kwargs={'consolidated':False}, chunks='auto')
        if 'initialization_time' not in ds.attrs:
            raise ValueError("Initialization time not found in dataset attributes. \
                             Please add initialization_time to the dataset attributes.")
        else:
            model_run_time = np.datetime64(pd.to_datetime(ds.attrs['initialization_time']))
        ds[forecast_schema_config.init_time] = model_run_time.astype('datetime64[ns]')
        fhours = ds[forecast_schema_config.time] - model_run_time
        fhours = fhours.values / np.timedelta64(1, 'h')
        ds[forecast_schema_config.fhour] = fhours
        ds = ds.set_coords(forecast_schema_config.init_time)
        ds = ds.expand_dims(forecast_schema_config.init_time)
        for data_vars in ds.data_vars:
            if forecast_schema_config.time in ds[data_vars].dims:
                ds[data_vars] = ds[data_vars].swap_dims({forecast_schema_config.time:forecast_schema_config.fhour})
        ds = ds.transpose(forecast_schema_config.init_time, 
                          forecast_schema_config.time, 
                          forecast_schema_config.fhour, 
                          forecast_schema_config.latitude, 
                          forecast_schema_config.longitude, 
                          forecast_schema_config.level)
        xarray_datasets.append(ds)

    return xr.concat(xarray_datasets, dim=forecast_schema_config.init_time)