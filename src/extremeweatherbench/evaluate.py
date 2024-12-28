"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

# TODO(taylor): Incorporate logging diagnostics throughout evaluation stack.
import logging
import fsspec
import os
from typing import Optional
import logging
import pandas as pd
import xarray as xr
from importlib import resources
from extremeweatherbench import config, events, case, utils

#: Default mapping for forecast dataset schema.
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()


def evaluate(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG,
    dry_run: bool = False,
) -> dict[str, list[xr.Dataset]]:
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

    point_obs, gridded_obs = _open_obs_datasets(
        eval_config
    )  # TODO: more elegant design approach?
    forecast_dataset = _open_forecast_dataset(eval_config, forecast_schema_config)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    events_file_path = os.path.join(base_dir, "../../assets/data/events.yaml")
    all_results = {}

    for event in eval_config.event_types:
        event_runner = event(path=events_file_path)
        # cases = event.from_yaml(events_file_path)
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
            _evaluate_case(
                individual_case,
                event.metrics,
                forecast_dataset,
                gridded_obs,
                point_obs,
            )
        )
    return results


def _evaluate_case(
    individual_case: case.IndividualCase,
    metrics: list,
    forecast_dataset,
    gridded_obs,
    point_obs,
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
    # Each case has a unique region and event type which can be
    # assessed here.

    # TODO(taylor): Implement the actual evaluation logic here.
    if point_obs is not None:
        pass
    if gridded_obs is not None:
        for metric in metrics:
            result = metric.compute(forecast_dataset, gridded_obs)
            return result


# TODO simplify to one paradigm, don't use nc, zarr, AND json
def _open_forecast_dataset(
    eval_config: config.Config,
    forecast_schema_config: Optional[config.ForecastSchemaConfig] = None,
):
    logging.info("Opening forecast dataset")
    if eval_config.forecast_dir.startswith("s3://"):
        fs = fsspec.filesystem("s3")
    elif eval_config.forecast_dir.startswith(
        "gcs://"
    ) or eval_config.forecast_dir.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
    else:
        fs = fsspec.filesystem("file")

    file_list = fs.ls(eval_config.forecast_dir)
    file_types = set([file.split(".")[-1] for file in file_list])
    if len(file_types) > 1:
        raise ValueError("Multiple file types found in forecast path.")

    if "zarr" in file_types and len(file_list) == 1:
        forecast_dataset = xr.open_zarr(file_list, chunks="auto")
    elif "zarr" in file_types and len(file_list) > 1:
        raise ValueError(
            "Multiple zarr files found in forecast path, please provide a single zarr file."
        )

    if "nc" in file_types:
        raise NotImplementedError("NetCDF file reading not implemented.")

    if "json" in file_types:
        forecast_dataset = utils._open_kerchunk_zarr_reference_jsons(
            file_list, forecast_schema_config
        )

    return forecast_dataset


def _open_obs_datasets(eval_config: config.Config):
    """Open the observation datasets specified for evaluation."""
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path is not None:
        point_obs = pd.read_parquet(eval_config.point_obs_path, chunks="auto")
    if eval_config.gridded_obs_path is not None:
        gridded_obs = xr.open_zarr(
            eval_config.gridded_obs_path,
            chunks=None,
            storage_options=dict(token="anon"),
        )
    if point_obs is None and gridded_obs is None:
        raise ValueError("No grided or point observation data provided.")
    return point_obs, gridded_obs
