"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import logging
import fsspec
from typing import Optional, Any, Literal
import pandas as pd
import xarray as xr
from extremeweatherbench import config, events, case, utils
import dacite


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Default mapping for forecast dataset schema.
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()


def evaluate(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG,
    dry_run: bool = False,
    dry_run_event_type: Optional[str] = "HeatWave",
) -> dict[Any, dict[Any, Optional[dict[str, Any]]]]:
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

    all_results = {}
    yaml_event_case = utils.load_events_yaml()
    for k, v in yaml_event_case.items():
        if k == "cases":
            for individual_case in v:
                if "location" in individual_case:
                    individual_case["location"]["longitude"] = (
                        utils.convert_longitude_to_360(
                            individual_case["location"]["longitude"]
                        )
                    )
                    individual_case["location"] = utils.Location(
                        **individual_case["location"]
                    )
    if dry_run:
        logger.debug(
            "Dry run invoked for %s, not running evaluation", dry_run_event_type
        )
        for event in eval_config.event_types:
            if event.__name__ == dry_run_event_type:
                cases: dict = dacite.from_dict(
                    data_class=event,
                    data=yaml_event_case,
                )
                return cases
    logger.debug("Evaluation starting")
    for event in eval_config.event_types:
        cases = dacite.from_dict(
            data_class=event,
            data=yaml_event_case,
        )
        logger.debug("Cases loaded for %s", event.event_type)
        point_obs, gridded_obs = _open_obs_datasets(eval_config)
        forecast_dataset = _open_forecast_dataset(eval_config, forecast_schema_config)

        # Manages some of the quirkiness of the parquets and avoids loading in memory overloads
        # from the json kerchunk references
        if "json" not in eval_config.forecast_dir:
            logger.warning(
                "json not detected for %s at %s, loading part of forecast dataset into memory",
                event.event_type,
                eval_config.forecast_dir,
            )
            forecast_dataset = forecast_dataset.compute()

        if gridded_obs:
            logger.info(
                "gridded obs detected, mapping variables in gridded obs to forecast"
            )
            gridded_obs = utils.map_era5_vars_to_forecast(
                DEFAULT_FORECAST_SCHEMA_CONFIG, forecast_dataset, gridded_obs
            )
        logger.debug("beginning evaluation loop for %s", event.event_type)
        results = _evaluate_cases_loop(cases, forecast_dataset, gridded_obs, point_obs)
        all_results[event.event_type] = results
        logger.debug("evaluation loop complete for %s", event.event_type)
    logger.info(
        "\nVerification Summary:\n"
        "- Processed %s event types\n"
        "- Observation types verified against: %s\n"
        "- Generated results for %s cases\n"
        "Evaluation complete.",
        len(eval_config.event_types),
        ", ".join(
            [
                x
                for x in [
                    "point" if point_obs is not None else "",
                    "gridded" if gridded_obs is not None else "",
                ]
                if x
            ]
        ),
        sum(len(results) for results in all_results.values() if results is not None),
    )
    return all_results


def _evaluate_cases_loop(
    event: events.EventContainer,
    forecast_dataset: xr.Dataset,
    gridded_obs: Optional[xr.Dataset] = None,
    point_obs: Optional[pd.DataFrame] = None,
) -> dict[Any, Optional[dict[str, Any]]]:
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
    forecast_dataset: Optional[xr.Dataset],
    gridded_obs: Optional[xr.Dataset],
    point_obs: Optional[pd.DataFrame],
) -> Optional[dict[str, xr.Dataset]]:
    """Evaluate a single case given forecast data and observations.

    Args:
        individual_case: A configuration object defining the case to evaluate.
        forecast_dataset: The forecast dataset to evaluate against.
        gridded_obs: The gridded observation dataset to use for evaluation.
        point_obs: The point observation dataset to use for evaluation.

    Returns:
        An xarray Dataset containing the evaluation results for the case.
    """

    logger.info("Evaluating case %s, %s", individual_case.id, individual_case.title)
    variable_subset_ds = individual_case._subset_data_vars(forecast_dataset)
    time_subset_forecast_ds = individual_case._subset_valid_times(variable_subset_ds)
    # Check if forecast data is available for the case, if not, return None
    lead_time_len = len(time_subset_forecast_ds.init_time)
    if lead_time_len == 0:
        logger.warning(
            "No forecast data available for case %s, skipping", individual_case.id
        )
        return None
    elif lead_time_len < (individual_case.end_date - individual_case.start_date).days:
        logger.warning(
            "Fewer valid times in forecast than days in case %s, results likely unreliable",
            individual_case.id,
        )
    logger.info("Forecast data available for case %s", individual_case.id)
    case_results: dict[str, dict[str, Any]] = {}
    if gridded_obs is not None:
        case_results["gridded"] = {}
        subset_gridded_obs = _subset_gridded_obs(gridded_obs, individual_case)

        # Align gridded_obs and forecast_dataset by time
        subset_gridded_obs, forecast_ds = xr.align(
            subset_gridded_obs,
            time_subset_forecast_ds[list(time_subset_forecast_ds.keys())],
            join="inner",
        )
        for data_var in individual_case.data_vars:
            case_results["gridded"][data_var] = {}
            forecast_da = forecast_ds[data_var]
            gridded_obs_da = subset_gridded_obs[data_var]
            forecast_da = forecast_da.compute()
            gridded_obs_da = gridded_obs_da.compute()
            for metric in individual_case.metrics_list:
                metric_instance = metric()
                logging.debug(
                    "gridded metric %s computing for %s", metric_instance.name, data_var
                )
                result = metric_instance.compute(forecast_da, gridded_obs_da)
                case_results["gridded"][data_var][metric_instance.name] = result
                logger.debug(
                    "gridded, %s, %s, %s", data_var, metric_instance.name, result
                )
    if point_obs is not None:
        case_results["point"] = {}
        case_subset_point_obs = point_obs.loc[point_obs["id"] == individual_case.id]
        case_subset_point_obs = case_subset_point_obs.rename(columns=utils.ISD_MAPPING)
        case_subset_point_obs["longitude"] = utils.convert_longitude_to_360(
            case_subset_point_obs["longitude"]
        )
        case_subset_point_obs = utils.unit_check(case_subset_point_obs)
        case_subset_point_obs = utils.location_subset_point_obs(
            case_subset_point_obs,
            spatiotemporal_subset_forecast_ds["latitude"].min().values,
            spatiotemporal_subset_forecast_ds["latitude"].max().values,
            spatiotemporal_subset_forecast_ds["longitude"].min().values,
            spatiotemporal_subset_forecast_ds["longitude"].max().values,
        )

        for data_var in individual_case.data_vars:
            case_results["point"][data_var] = {}
            forecast_da = spatiotemporal_subset_forecast_ds[data_var]
            case_subset_point_obs_df = case_subset_point_obs[
                utils.POINT_OBS_METADATA_VARS + [data_var]
            ]

            forecast_da, case_subset_point_obs_da = utils.align_point_obs_from_gridded(
                forecast_da, case_subset_point_obs_df, utils.POINT_OBS_METADATA_VARS
            )  # rename forecast_da to something more readable/descriptive

            forecast_da = forecast_da.compute()
            case_subset_point_obs_da = case_subset_point_obs_da.compute()

            forecast_da = (
                forecast_da.groupby(
                    ["init_time", "lead_time", "latitude", "longitude"]
                ).mean()  # change to mean([["init_time", "lead_time", "latitude", "longitude"]])
            )
            case_subset_point_obs_da = case_subset_point_obs_da.groupby(
                ["time", "latitude", "longitude"]
            ).first()
            # TODO(aaTman): #64 define where and how loading to memory will occur.
            # where diverging occurs between EWB and libraries like WBX is the
            # philosophical core of what's being computed, in a way. Though having
            # users define when they want to load into memory is important,
            # there are clearly defined places in code, such as here, that loading to memory
            # would minimize overhead from large graphs.
            for metric in individual_case.metrics_list:
                metric_instance = metric()
                logging.debug(
                    "point metric %s computing for %s", metric_instance.name, data_var
                )
                result = metric_instance.compute(forecast_da, case_subset_point_obs_da)
                case_results["point"][data_var][metric_instance.name] = result
                logger.debug(
                    "point, %s, %s, %s", data_var, metric_instance.name, result
                )
        # intentionally not returning anything or producing outputs for point obs
    return case_results


def _open_forecast_dataset(
    eval_config: config.Config,
    forecast_schema_config: config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG,
):
    """Open the forecast dataset specified for evaluation."""
    logging.debug("Opening forecast dataset")
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
    if "parq" in file_types or any("parq" in ft for ft in file_types):
        forecast_dataset = utils._open_mlwp_kerchunk_reference(
            eval_config.forecast_dir, forecast_schema_config
        )
    if "json" in file_types:
        forecast_dataset = utils._open_mlwp_kerchunk_reference(
            file_list[0], forecast_schema_config
        )
    return forecast_dataset


def _open_obs_datasets(eval_config: config.Config):
    """Open the observation datasets specified for evaluation."""
    point_obs = None
    gridded_obs = None
    if eval_config.gridded_obs_path:
        gridded_obs = xr.open_zarr(
            eval_config.gridded_obs_path,
            chunks=None,
            storage_options=dict(token="anon"),
        )
    if point_obs is None and gridded_obs is None:
        raise ValueError("No gridded or point observation data provided.")
    return point_obs, gridded_obs


def _subset_gridded_obs(
    gridded_obs: xr.Dataset, individual_case: case.IndividualCase
) -> xr.Dataset:
    variable_subset_gridded_obs = individual_case._subset_data_vars(gridded_obs)
    time_subset_gridded_obs_ds = variable_subset_gridded_obs.sel(
        time=slice(individual_case.start_date, individual_case.end_date)
    )
    time_subset_gridded_obs_ds = individual_case.perform_subsetting_procedure(
        time_subset_gridded_obs_ds
    )
    return time_subset_gridded_obs_ds
