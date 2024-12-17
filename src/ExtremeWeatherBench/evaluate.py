'''Implements evaluation routines for the ExtremeWeatherBench library.'''
import xarray as xr
import pandas as pd

#TODO: logging
import logging
import fsspec
import numpy as np
import os
from typing import Optional, List

#TODO: once uv/ruff/pyproject.toml is set up, remove relative imports
from . import events
from . import config
from . import case
from . import utils
#: default configuration for mapping forecast dataset schema
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()

def evaluate(eval_config: config.Config, 
             forecast_schema_config: config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG,
             dry_run: bool = False) -> List[xr.Dataset]:
    """ Evaluate the metrics, looping through each case for each event type.
    Arguments:
        eval_config: the configuration object for the evaluation
        forecast_schema_config: the configuration object for the forecast variable naming schema
        dry_run: a bool that is used to validate the cases without running the evaluation
    """

    point_obs, gridded_obs = utils._open_obs_datasets(eval_config) #TODO: more elegant design approach?
    forecast_dataset = utils._open_forecast_dataset(eval_config, forecast_schema_config)

    #TODO: use importlib resources for this when uv/ruff/pyproject.toml is set up
    base_dir = os.path.dirname(os.path.abspath(__file__))
    events_file_path = os.path.join(base_dir, '../../assets/data/events.yaml')
    for event in eval_config.event_types:
        cases = event.from_yaml(events_file_path)
        cases.build_metrics()
        if dry_run: # temporary validation for the cases
            return cases
        else:
            results = _evaluate_cases_loop(cases, forecast_dataset, gridded_obs, point_obs)
    return results

def _evaluate_cases_loop(event: events.Event, 
                         forecast_dataset: xr.Dataset,
                         gridded_obs: Optional[xr.Dataset],
                         point_obs: Optional[pd.DataFrame] 
                         ) -> List[xr.Dataset]:
    """ Loops through each case for each event type selected.
    Arguments:
        eval_config: the configuration object for the evaluation
        forecast_schema_config: the configuration object for the forecast variable naming schema
        dry_run: a bool that is used to validate the cases without running the evaluation
    """
    results = []
    for individual_case in event.cases:
            results.append(_evaluate_case(individual_case, forecast_dataset, gridded_obs, point_obs))
    return results

def _evaluate_case(individual_case: case.IndividualCase, 
                   forecast_dataset, 
                   gridded_obs, 
                   point_obs): #strongly type returns
    """ Evaluate the forecast data against the observed data for a single case.
    Arguments:
        individual_case: the container object for one case within an event
        forecast_dataset: the forecast xarray Dataset
        gridded_obs: the gridded observation xarray Dataset
        point_obs: the point observation pandas DataFrame
    """
    #TODO: address incomplete functions
    if point_obs is not None:
        pass 
    if gridded_obs is not None:
        pass