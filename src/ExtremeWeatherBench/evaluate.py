"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

# TODO(taylor): Incorporate logging diagnostics throughout evaluation stack.
# import logging
import os
from typing import Optional

import pandas as pd
import xarray as xr

# TODO(taylor): once uv/ruff/pyproject.toml is set up, remove relative imports
from . import case, config, events, utils

#: Default mapping for forecast dataset schema.
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()


def evaluate(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG,
    dry_run: bool = False,
) -> dict[str, list[xr.Dataset]]
    """Driver for evaluating a collection of Cases across a set of Events.

    Args:
        eval_config: A configuration object defining the evaluation run.
        forecast_schema_config: A mapping of the forecast variable naming schema to use
            when reading / decoding forecast data in the analysis.
        dry_run: Flag to disable performing actual calculations (but still validate
            case configurations). Defaults to "False."
    
    Returns:
        A dictionary mapping event types to lists of xarray Datasets containing the
        evaluation results for each case within the event type.
    """

    point_obs, gridded_obs = utils._open_obs_datasets(
        eval_config
    )  # TODO: more elegant design approach?
    forecast_dataset = utils._open_forecast_dataset(eval_config, forecast_schema_config)

    # TODO: use importlib resources for this when uv/ruff/pyproject.toml is set up
    base_dir = os.path.dirname(os.path.abspath(__file__))
    events_file_path = os.path.join(base_dir, "../../assets/data/events.yaml")
    all_results = {}
    for event in eval_config.event_types:
        cases = event.from_yaml(events_file_path)
        cases.build_metrics()
        if dry_run:  # temporary validation for the cases
            return cases
        else:
            results = _evaluate_cases_loop(
                cases, forecast_dataset, gridded_obs, point_obs
            )
            # NOTE(daniel): This is a bit of a hack, but it's a quick way to get the
            # event name for the dictionary key; can do something later, since we
            # probably don't want to make Event objects hashable.
            all_results[event.__name__] = results
    return all_results


def _evaluate_cases_loop(
    event: events.Event,
    forecast_dataset: xr.Dataset,
    gridded_obs: Optional[xr.Dataset] = None,
    point_obs: Optional[pd.DataFrame] = None,
) -> list[xr.Dataset]:
    """Sequentially loop over and evalute all cases for a specific event type.

    Args:
        event: The Event object containing the cases to evaluate.
        forecast_dataset: The forecast dataset to evaluate against.
        gridded_obs: The gridded observation dataset to use for evaluation.
        point_obs: The point observation dataset to use for evaluation.

    Returns:
        A list of xarray Datasets containing the evaluation results for each case
        in the Event of interest.
    """
    results = []
    for individual_case in event.cases:
        results.append(
            _evaluate_case(individual_case, forecast_dataset, gridded_obs, point_obs)
        )
    return results


def _evaluate_case(
    individual_case: case.IndividualCase, forecast_dataset, gridded_obs, point_obs
) -> xr.Dataset: 
    """Evalaute a single case given forecast data and observations.

    Args:
        individual_case: A configuration object defining the case to evaluate.
        forecast_dataset: The forecast dataset to evaluate against.
        gridded_obs: The gridded observation dataset to use for evaluation.
        point_obs: The point observation dataset to use for evaluation.

    Returns:
        An xarray Dataset containing the evaluation results for the case.
    """
    # TODO(taylor): Implement the actual evaluation logic here.
    if point_obs is not None:
        pass
    if gridded_obs is not None:
        pass
