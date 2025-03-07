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

POINT_OBS_STORAGE_OPTIONS = dict(token="anon")

GRIDDED_OBS_STORAGE_OPTIONS = dict(token="anon")


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

    def compute(self):
        """Load the evaluation inputs into memory."""
        self.observation = self.observation.compute()
        self.forecast = self.forecast.compute()


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

    def __init__(
        self,
        individual_case: case.IndividualCase,
        observation_type: Literal["gridded", "point"],
        forecast: xr.Dataset,
        observation: Optional[Union[xr.Dataset | pd.DataFrame]],
    ):
        self.observation_type = observation_type
        self.observation = observation
        self.forecast = forecast
        self.individual_case = individual_case

    def build_dataarray_subsets(
        self,
        compute: bool = True,
        point_obs_mapping: dict = utils.ISD_MAPPING,
    ) -> CaseEvaluationInput:
        """Build the subsets of the gridded and point observations for a given data variable.
        Computation occurs based on what observation datasets are provided in the Evaluation object.

        Args:
            data_var: The data variable to evaluate.
            compute: Flag to disable performing actual calculations (but still validate
                case configurations). Defaults to "False."

        Returns:
            A tuple of xarray DataArrays containing the gridded and point observation subsets.
        """
        self.forecast = self._check_forecast_data_availability()
        if self.forecast is None:
            return CaseEvaluationInput(
                observation_type=self.observation_type,
                observation=None,
                forecast=None,
            )
        if self.observation is None:
            return CaseEvaluationInput(
                observation_type=self.observation_type,
                observation=None,
                forecast=None,
            )
        if self.observation_type == "gridded":
            subset_gridded_obs_da = self.observation[self.individual_case.data_vars]
            evaluation_result = self._subset_gridded_obs(subset_gridded_obs_da)
        elif self.observation_type == "point":
            renamed_observations = self.observation.rename(columns=point_obs_mapping)
            subset_point_obs = renamed_observations[
                utils.POINT_OBS_METADATA_VARS + self.individual_case.data_vars
            ]
            evaluation_result = self._subset_point_obs(subset_point_obs, compute)
        if compute:
            evaluation_result.compute()
        return evaluation_result

    def _subset_gridded_obs(self, gridded_obs: xr.Dataset) -> CaseEvaluationInput:
        """Subset the gridded observation dataarray for a given data variable."""
        time_subset_gridded_obs_ds = gridded_obs.sel(
            time=slice(self.individual_case.start_date, self.individual_case.end_date)
        )
        time_spatial_subset_gridded_obs_ds = (
            self.individual_case.perform_subsetting_procedure(
                time_subset_gridded_obs_ds
            )
        )
        # Align gridded_obs and forecast_dataset by time
        time_spatial_subset_gridded_obs_ds, forecast_da = xr.align(
            time_spatial_subset_gridded_obs_ds,
            self.forecast,
            join="inner",
        )
        return CaseEvaluationInput(
            "gridded",
            observation=time_spatial_subset_gridded_obs_ds,
            forecast=forecast_da,
        )

    def _subset_point_obs(
        self, point_obs: pd.DataFrame, compute: bool
    ) -> CaseEvaluationInput:
        """Subset the point observation dataarray for a given data variable."""
        subset_id_point_obs = point_obs.loc[point_obs["id"] == self.individual_case.id]
        mapped_subset_id_point_obs = subset_id_point_obs.rename(
            columns=utils.ISD_MAPPING
        )
        mapped_subset_id_point_obs["longitude"] = utils.convert_longitude_to_360(
            mapped_subset_id_point_obs["longitude"]
        )
        mapped_subset_id_point_obs = utils.location_subset_point_obs(
            mapped_subset_id_point_obs,
            self.forecast["latitude"].min().values,
            self.forecast["latitude"].max().values,
            self.forecast["longitude"].min().values,
            self.forecast["longitude"].max().values,
        )
        point_forecast_ds, subset_point_obs_ds = utils.align_point_obs_from_gridded(
            self.forecast,
            mapped_subset_id_point_obs,
            self.individual_case.data_vars,
            utils.POINT_OBS_METADATA_VARS,
        )
        # groupby function is not lazy, so we need to compute the datasets before grouping
        # even with flox this method is slow without prior computation
        if compute:
            point_forecast_ds = point_forecast_ds.compute()
            subset_point_obs_ds = subset_point_obs_ds.compute()
        point_forecast_ds = point_forecast_ds.groupby(
            ["init_time", "lead_time", "latitude", "longitude"]
        ).mean()
        subset_point_obs_ds = subset_point_obs_ds.groupby(
            ["time", "latitude", "longitude"]
        ).first()
        return CaseEvaluationInput(
            "point", observation=subset_point_obs_ds, forecast=point_forecast_ds
        )

    def _check_forecast_data_availability(self):
        if len(self.forecast.lead_time) == 0 or len(self.forecast.init_time) == 0:
            raise ValueError("No forecast data available, check forecast dataset.")
        variable_subset_forecast_ds = self.individual_case._subset_data_vars(
            self.forecast
        )
        time_variable_subset_forecast_ds = self.individual_case._subset_valid_times(
            variable_subset_forecast_ds
        )
        lead_time_len = len(time_variable_subset_forecast_ds.init_time)
        if lead_time_len == 0:
            logger.warning(
                "No forecast data available for case %s, skipping",
                self.individual_case.id,
            )
            return None
        elif (
            lead_time_len
            < (self.individual_case.end_date - self.individual_case.start_date).days
        ):
            logger.warning(
                "Fewer valid times in forecast than days in case %s, results likely unreliable",
                self.individual_case.id,
            )
        else:
            logger.info(
                "Forecast data available for case %s, %s obs",
                self.individual_case.id,
                self.observation_type,
            )
        return time_variable_subset_forecast_ds


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
        logger.debug("Cases loaded for %s", event.event_type)
        if gridded_obs:
            logger.info(
                "gridded obs detected, mapping variables in gridded obs to forecast"
            )
            gridded_obs = utils.map_era5_vars_to_forecast(
                forecast_schema_config,
                forecast_dataset=forecast_dataset,
                era5_dataset=gridded_obs,
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
        # TODO(aaTman): #63 loop does not work properly, it does not skip None's outputting the sum of all cases.
        # Needs to skip None values
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
    case_results: dict[str, dict[str, Any]] = {}
    logger.info("Evaluating case %s, %s", individual_case.id, individual_case.title)
    gridded_obs_evaluation = CaseEvaluationData(
        individual_case, "gridded", observation=gridded_obs, forecast=forecast_dataset
    )
    point_obs_evaluation = CaseEvaluationData(
        individual_case, "point", observation=point_obs, forecast=forecast_dataset
    )
    gridded_case_eval = gridded_obs_evaluation.build_dataarray_subsets(compute=True)
    point_case_eval = point_obs_evaluation.build_dataarray_subsets(compute=True)
    for data_var, eval, metric in itertools.product(
        individual_case.data_vars,
        [gridded_case_eval, point_case_eval],
        individual_case.metrics_list,
    ):
        if data_var not in case_results:
            case_results[data_var] = {}
        if eval.observation_type not in case_results[data_var]:
            case_results[data_var][eval.observation_type] = {}

        metric_instance = metric()

        # TODO(aaTman): Create metric container object for gridded and point obs
        # in the meantime, forcing a check for Nones in gridded and point eval objects

        if eval.observation is not None and eval.forecast is not None:
            logging.info("metric %s computing", metric_instance.name)
            case_results[data_var][eval.observation_type][metric_instance.name] = (
                metric_instance.compute(
                    eval.forecast[data_var], eval.observation[data_var]
                )
            )
        else:
            case_results[data_var][eval.observation_type][metric_instance.name] = None
    return case_results
