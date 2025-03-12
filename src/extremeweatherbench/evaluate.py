"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import logging
from typing import Optional, Any, Literal, Union
import pandas as pd
import xarray as xr
from extremeweatherbench import config, events, case, utils, data_loader
import dacite
import dataclasses
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Default mapping for forecast dataset schema.
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()


@dataclasses.dataclass
class CaseEvaluationInput:
    """
    A dataclass for storing the inputs of an evaluation.

    Attributes:
        observation_type: The type of observation to evaluate (gridded or point).
        observation: The observation dataarray to evaluate.
        forecast: The forecast dataarray to evaluate.
    """

    observation_type: Literal["gridded", "point"]
    observation: Optional[xr.DataArray] = None
    forecast: Optional[xr.DataArray] = None

    def load_data(self):
        """Load the evaluation inputs into memory."""
        logger.debug("Loading evaluation inputs into memory")
        self.observation = self.observation.compute()
        self.forecast = self.forecast.compute()


@dataclasses.dataclass
class CaseEvaluationData:
    """
    This class is designed to be used in conjunction with the `case.IndividualCase` class to build
    datasets for evaluation of individual Cases across a set of Events.

    Attributes:
        individual_case: The `case.IndividualCase` object to evaluate.
        forecast: The forecast dataset to evaluate.
        gridded_observation: The gridded observation dataset to evaluate.
        point_observation: The point observation dataset to evaluate.
    """

    individual_case: case.IndividualCase
    forecast: xr.Dataset
    observation_type: Literal["gridded", "point"]
    observation: Optional[Union[xr.Dataset | pd.DataFrame]] = None


def build_dataarray_subsets(
    case_evaluation_data: CaseEvaluationData,
    compute: bool = True,
    existing_forecast: Optional[xr.Dataset] = None,
) -> CaseEvaluationInput:
    """Build the subsets of the gridded and point observations for a given data variable.
    Computation occurs based on what observation datasets are provided in the Evaluation object.

    Args:
        data_var: The data variable to evaluate.
        compute: Flag to disable performing actual calculations (but still validate
            case configurations). Defaults to "False."
        existing_forecast: If a forecast dataset is already loaded, this can be passed
            in to avoid recomputing the forecast data into memory.
    Returns:
        A tuple of xarray DataArrays containing the gridded and point observation subsets.
    """
    if existing_forecast is not None:
        logger.debug(
            "Using existing forecast dataset for %s",
            case_evaluation_data.observation_type,
        )
        case_evaluation_data.forecast = existing_forecast
    else:
        case_evaluation_data.forecast = _check_and_subset_forecast_availability(
            case_evaluation_data
        )
    if (
        case_evaluation_data.forecast is None
        or case_evaluation_data.observation is None
    ):
        return CaseEvaluationInput(
            observation_type=case_evaluation_data.observation_type,
            observation=None,
            forecast=None,
        )
    else:
        if case_evaluation_data.observation_type == "gridded":
            evaluation_result = _subset_gridded_obs(case_evaluation_data)
        elif case_evaluation_data.observation_type == "point":
            # point obs needs to be computed if compute is True due to complex subsetting operations
            evaluation_result = _subset_point_obs(case_evaluation_data, compute=compute)
        if compute:
            evaluation_result.load_data()
        return evaluation_result


def _subset_gridded_obs(
    case_evaluation_data: CaseEvaluationData,
) -> CaseEvaluationInput:
    """Subset the gridded observation dataarray for a given data variable."""

    if case_evaluation_data.observation is None:
        raise ValueError("Gridded observation cannot be None")
    var_subset_gridded_obs_ds = case_evaluation_data.observation[
        case_evaluation_data.individual_case.data_vars
    ]
    time_var_subset_gridded_obs_ds = var_subset_gridded_obs_ds.sel(
        time=slice(
            case_evaluation_data.individual_case.start_date,
            case_evaluation_data.individual_case.end_date,
        )
    )
    completed_subset_gridded_obs_ds = (
        case_evaluation_data.individual_case.perform_subsetting_procedure(
            time_var_subset_gridded_obs_ds
        )
    )
    # Align gridded_obs and forecast_dataset by time
    subset_gridded_obs, forecast_ds = xr.align(
        completed_subset_gridded_obs_ds,
        case_evaluation_data.forecast,
        join="inner",
    )
    return CaseEvaluationInput(
        "gridded", observation=subset_gridded_obs, forecast=forecast_ds
    )


def _subset_point_obs(
    case_evaluation_data: CaseEvaluationData, compute: bool = True
) -> CaseEvaluationInput:
    """Subset the point observation dataarray for a given data variable."""
    if case_evaluation_data.observation is None:
        raise ValueError("Point observation cannot be None")
    renamed_observations = case_evaluation_data.observation.rename(
        columns=utils.ISD_MAPPING
    )
    var_subset_point_obs = renamed_observations[
        utils.POINT_OBS_METADATA_VARS + case_evaluation_data.individual_case.data_vars
    ]
    var_id_subset_point_obs = var_subset_point_obs.loc[
        var_subset_point_obs["id"] == case_evaluation_data.individual_case.id
    ]
    mapped_var_id_subset_point_obs = var_id_subset_point_obs.rename(
        columns=utils.ISD_MAPPING
    )
    mapped_var_id_subset_point_obs["longitude"] = utils.convert_longitude_to_360(
        mapped_var_id_subset_point_obs["longitude"]
    )

    # this saves a significant amount of time if done prior to alignment with point obs
    if compute:
        logger.debug("Computing forecast dataset in point obs subsetting")
        case_evaluation_data.forecast = case_evaluation_data.forecast.compute()

    point_forecast_ds, subset_point_obs_ds = utils.align_point_obs_from_gridded(
        forecast_ds=case_evaluation_data.forecast,
        case_subset_point_obs_df=mapped_var_id_subset_point_obs,
        data_var=case_evaluation_data.individual_case.data_vars,
        point_obs_metadata_vars=utils.POINT_OBS_METADATA_VARS,
    )
    point_forecast_df = point_forecast_ds.to_dataframe()
    subset_point_obs_df = subset_point_obs_ds.to_dataframe()

    # pandas groupby is significantly faster than xarray groupby, so we use that here
    point_forecast_recompiled_ds = (
        point_forecast_df.reset_index()
        .groupby(["init_time", "lead_time", "latitude", "longitude"])
        .first()
        .to_xarray()
    )

    subset_point_obs_recompiled_ds = (
        subset_point_obs_df.reset_index()
        .groupby(["time", "latitude", "longitude"])
        .first()
        .to_xarray()
    )

    return CaseEvaluationInput(
        "point",
        observation=subset_point_obs_recompiled_ds,
        forecast=point_forecast_recompiled_ds,
    )


def _check_and_subset_forecast_availability(
    case_evaluation_data: CaseEvaluationData,
) -> Optional[xr.DataArray]:
    if (
        len(case_evaluation_data.forecast.lead_time) == 0
        or len(case_evaluation_data.forecast.init_time) == 0
    ):
        raise ValueError("No forecast data available, check forecast dataset.")

    # subset the forecast to the valid times of the case
    forecast_time_subset = case_evaluation_data.individual_case._subset_valid_times(
        case_evaluation_data.forecast
    )
    forecast_spatial_subset = (
        case_evaluation_data.individual_case.perform_subsetting_procedure(
            forecast_time_subset
        )
    )
    # subset the forecast to the data variables for the event type/metric
    forecast = forecast_spatial_subset[case_evaluation_data.individual_case.data_vars]
    lead_time_len = len(forecast.init_time)
    if lead_time_len == 0:
        logger.warning(
            "No forecast data available for case %s, skipping",
            case_evaluation_data.individual_case.id,
        )
        return None
    elif (
        lead_time_len
        < (
            case_evaluation_data.individual_case.end_date
            - case_evaluation_data.individual_case.start_date
        ).days
    ):
        logger.warning(
            "Fewer valid times in forecast than days in case %s, results likely unreliable",
            case_evaluation_data.individual_case.id,
        )
    logger.info(
        "Forecast data available for case %s", case_evaluation_data.individual_case.id
    )

    return forecast


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
    point_obs, gridded_obs = data_loader.open_obs_datasets(eval_config)
    forecast_dataset = data_loader.open_forecast_dataset(
        eval_config, forecast_schema_config
    )
    logger.debug("Forecast and observation datasets loaded")
    logger.debug(
        "Observation data: Point %s, Gridded %s",
        point_obs is not None,
        gridded_obs is not None,
    )
    for event in eval_config.event_types:
        cases = dacite.from_dict(
            data_class=event,
            data=yaml_event_case,
        )

        # map era5 vars by renaming and dropping extra vars
        if gridded_obs is not None:
            gridded_obs = utils.map_era5_vars_to_forecast(
                forecast_schema_config,
                forecast_dataset=forecast_dataset,
                era5_dataset=gridded_obs,
            )
        logger.debug("beginning evaluation loop for %s", event.event_type)
        results = _maybe_evaluate_individual_cases_loop(
            cases, forecast_dataset, gridded_obs, point_obs
        )
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
        # TODO(aaTman): #63 loop does not work properly, it does not skip None's outputting the sum of all cases.
        # Needs to skip None values
        sum(len(results) for results in all_results.values() if results is not None),
    )
    return all_results


def _maybe_evaluate_individual_cases_loop(
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
        results[individual_case.id] = _maybe_evaluate_individual_case(
            individual_case,
            forecast_dataset,
            gridded_obs,
            point_obs,
        )

    return results


def _maybe_evaluate_individual_case(
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

    Raises:
        ValueError: If no forecast data is available.
    """
    case_results: dict[str, dict[str, Any]] = {}
    logger.info("Evaluating case %s, %s", individual_case.id, individual_case.title)
    gridded_obs_evaluation = CaseEvaluationData(
        individual_case=individual_case,
        observation_type="gridded",
        observation=gridded_obs,
        forecast=forecast_dataset,
    )
    point_obs_evaluation = CaseEvaluationData(
        individual_case=individual_case,
        observation_type="point",
        observation=point_obs,
        forecast=forecast_dataset,
    )

    gridded_case_eval = build_dataarray_subsets(gridded_obs_evaluation, compute=True)
    point_case_eval = build_dataarray_subsets(
        point_obs_evaluation,
        compute=True,
        existing_forecast=gridded_case_eval.forecast,
    )
    for data_var, metric in itertools.product(
        individual_case.data_vars, individual_case.metrics_list
    ):
        case_results[data_var] = {}
        metric_instance = metric()
        logging.debug("metric %s computing", metric_instance.name)
        # TODO(aaTman): Create metric container object for gridded and point obs
        # in the meantime, forcing a check for Nones in gridded and point eval objects
        result = {}
        for eval in [gridded_case_eval, point_case_eval]:
            if eval.observation is not None and eval.forecast is not None:
                result[eval.observation_type] = metric_instance.compute(
                    eval.forecast[data_var], eval.observation[data_var]
                )
            else:
                result[eval.observation_type] = None
        case_results[data_var][metric_instance.name] = result
    return case_results
