"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

# TODO(taylor): Incorporate logging diagnostics throughout evaluation stack.
import logging
import fsspec
import os
from typing import Optional
import logging
import pandas as pd
import xarray as xr
from extremeweatherbench import config, events, case, utils
import dacite
import yaml

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

    # TODO: aligning forecast and obs dataset using more robust method
    point_obs, gridded_obs = _open_obs_datasets(eval_config)
    forecast_dataset = _open_forecast_dataset(eval_config, forecast_schema_config)
    if gridded_obs:
        gridded_obs = utils.map_era5_vars_to_forecast(
            DEFAULT_FORECAST_SCHEMA_CONFIG, forecast_dataset, gridded_obs
        )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    events_file_path = os.path.join(base_dir, "../../assets/data/events.yaml")
    all_results = {}
    with open(events_file_path, "r") as file:
        yaml_event_case = yaml.safe_load(file)

    for event in eval_config.event_types:
        cases = dacite.from_dict(data_class=event, data=yaml_event_case)
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
    event: events.EventContainer,
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
    results = {}
    for individual_case in event.cases:
        results[individual_case.id] = _evaluate_case(
            individual_case,
            forecast_dataset,
            gridded_obs,
            point_obs,
        )

    return results


def _evaluate_case(
    individual_case: case.IndividualCase,
    forecast_dataset: xr.Dataset,
    gridded_obs: xr.Dataset,
    point_obs: pd.DataFrame,
) -> dict:
    """Evaluate a single case given forecast data and observations.

    Args:
        individual_case: A configuration object defining the case to evaluate.
        forecast_dataset: The forecast dataset to evaluate against.
        gridded_obs: The gridded observation dataset to use for evaluation.
        point_obs: The point observation dataset to use for evaluation.

    Returns:
        An xarray Dataset containing the evaluation results for the case.
    """
    variable_subset_ds = individual_case._subset_data_vars(forecast_dataset)
    time_subset_forecast_ds = individual_case._subset_valid_times(variable_subset_ds)
    # Check if forecast data is available for the case, if not, return None
    forecast_exists = individual_case._check_for_forecast_data_availability(
        time_subset_forecast_ds
    )
    if not forecast_exists:
        return None
    if gridded_obs is not None:
        data_vars = {}
        variable_subset_gridded_obs = individual_case._subset_data_vars(gridded_obs)
        time_subset_gridded_obs_ds = variable_subset_gridded_obs.sel(
            time=slice(individual_case.start_date, individual_case.end_date)
        )
        time_subset_gridded_obs_ds = individual_case.perform_subsetting_procedure(
            time_subset_gridded_obs_ds
        )
        # Align gridded_obs and forecast_dataset by time
        # TODO: test if doing ERA5 first might speed up the compute
        time_subset_gridded_obs_ds, spatiotemporal_subset_ds = xr.align(
            time_subset_gridded_obs_ds,
            time_subset_forecast_ds[list(time_subset_forecast_ds.keys())],
            join="inner",
        )
        spatiotemporal_subset_ds = spatiotemporal_subset_ds.compute()
        time_subset_gridded_obs_ds = time_subset_gridded_obs_ds.compute()
        for metric in individual_case.metrics_list:
            metric_instance = metric()
            result = metric_instance.compute(
                spatiotemporal_subset_ds, time_subset_gridded_obs_ds
            )
            data_vars[metric_instance.name()] = result
        return data_vars
    else:
        # Each event type has a unique subsetting procedure
        spatiotemporal_subset_ds = individual_case.perform_subsetting_procedure(
            time_subset_forecast_ds
        )
    if point_obs is not None:
        pass


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
    if len(file_types) > 1 and "parq" not in eval_config.forecast_dir:
        raise ValueError("Multiple file types found in forecast path.")

    if "zarr" in file_types and len(file_list) == 1:
        forecast_dataset = xr.open_zarr(file_list, chunks="auto")
    elif "zarr" in file_types and len(file_list) > 1:
        raise ValueError(
            "Multiple zarr files found in forecast path, please provide a single zarr file."
        )

    if "nc" in file_types:
        raise NotImplementedError("NetCDF file reading not implemented.")

    if "json" or "parq" in file_types:
        forecast_dataset = utils._open_mlwp_kerchunk_references(
            file_list, forecast_schema_config
        )
        forecast_dataset = utils.convert_longitude_to_180(forecast_dataset)
    return forecast_dataset


def _open_obs_datasets(eval_config: config.Config):
    """Open the observation datasets specified for evaluation."""
    point_obs = None
    gridded_obs = None
    if eval_config.point_obs_path:
        point_obs = pd.read_parquet(eval_config.point_obs_path, chunks="auto")
    if eval_config.gridded_obs_path:
        gridded_obs = xr.open_zarr(
            eval_config.gridded_obs_path,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        gridded_obs = utils.convert_longitude_to_180(gridded_obs)
    if point_obs is None and gridded_obs is None:
        raise ValueError("No grided or point observation data provided.")
    return point_obs, gridded_obs
