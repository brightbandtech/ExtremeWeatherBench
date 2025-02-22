"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import logging
import fsspec
from typing import Optional, Any, Literal, Union
import pandas as pd
import xarray as xr
from extremeweatherbench import config, events, case, utils
import dacite
import dataclasses

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Default mapping for forecast dataset schema.
DEFAULT_FORECAST_SCHEMA_CONFIG = config.ForecastSchemaConfig()


@dataclasses.dataclass
class CaseEvaluationInput:
    """
    A dataclass for storing the inputs of an evaluation.
    """

    observation_type: Literal["gridded", "point"]
    observation: xr.DataArray
    forecast: xr.DataArray

    def compute(self):
        self.observation = self.observation.compute()
        self.forecast = self.forecast.compute()


class CaseEvaluation:
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
        forecast: xr.DataArray,
        observation: Optional[Union[xr.DataArray | pd.DataFrame]],
    ):
        self.observation_type = observation_type
        self.observation = observation
        self.forecast = forecast
        self.individual_case = individual_case

    def build_dataarray_subsets(
        self,
        compute: bool = True,
        point_obs_mapping: dict = utils.ISD_MAPPING,
    ) -> Optional[CaseEvaluationInput]:
        """Build the subsets of the gridded and point observations for a given data variable.
        Computation occurs based on what observation datasets are provided in the Evaluation object.

        Args:
            data_var: The data variable to evaluate.
            compute: Flag to disable performing actual calculations (but still validate
                case configurations). Defaults to "False."

        Returns:
            A tuple of xarray DataArrays containing the gridded and point observation subsets.
        """
        forecast = self._check_forecast_data_availability()
        if forecast is None:
            return None
        if self.observation is None:
            return None
        if self.observation_type == "gridded":
            subset_gridded_obs_da = self.observation[self.individual_case.data_vars]
            evaluation_result = self._subset_gridded_obs(subset_gridded_obs_da)
        elif self.observation_type == "point":
            renamed_observations = self.observation.rename(columns=point_obs_mapping)
            subset_point_obs = renamed_observations[
                utils.POINT_OBS_METADATA_VARS + [self.individual_case.data_vars]
            ]
            evaluation_result = self._subset_point_obs(subset_point_obs)
        if compute:
            evaluation_result.compute()
        return evaluation_result

    def _subset_gridded_obs(self, gridded_obs: xr.Dataset) -> CaseEvaluationInput:
        time_subset_gridded_obs_ds = gridded_obs.sel(
            time=slice(self.individual_case.start_date, self.individual_case.end_date)
        )
        time_subset_gridded_obs_ds = self.individual_case.perform_subsetting_procedure(
            gridded_obs
        )
        # Align gridded_obs and forecast_dataset by time
        subset_gridded_obs, forecast_ds = xr.align(
            time_subset_gridded_obs_ds,
            self.forecast,
            join="inner",
        )
        return CaseEvaluationInput(
            "gridded", observation=subset_gridded_obs, forecast=forecast_ds
        )

    def _subset_point_obs(self, point_obs: pd.DataFrame) -> CaseEvaluationInput:
        subset_id_point_obs = point_obs.loc[point_obs["id"] == self.individual_case.id]
        mapped_subset_id_point_obs = subset_id_point_obs.rename(
            columns=utils.ISD_MAPPING
        )
        mapped_subset_id_point_obs["longitude"] = utils.convert_longitude_to_360(
            mapped_subset_id_point_obs["longitude"]
        )
        mapped_subset_id_point_obs = utils.unit_check(mapped_subset_id_point_obs)
        mapped_subset_id_point_obs = utils.location_subset_point_obs(
            mapped_subset_id_point_obs,
            self.forecast["latitude"].min().values,
            self.forecast["latitude"].max().values,
            self.forecast["longitude"].min().values,
            self.forecast["longitude"].max().values,
        )
        point_forecast_da, subset_point_obs_da = utils.align_point_obs_from_gridded(
            self.forecast, mapped_subset_id_point_obs, utils.POINT_OBS_METADATA_VARS
        )  # rename forecast_da to something more readable/descriptive

        point_forecast_da = (
            point_forecast_da.groupby(
                ["init_time", "lead_time", "latitude", "longitude"]
            ).mean()  # change to mean([["init_time", "lead_time", "latitude", "longitude"]])
        )
        subset_point_obs_da = subset_point_obs_da.groupby(
            ["time", "latitude", "longitude"]
        ).first()
        return CaseEvaluationInput(
            "point", observation=subset_point_obs_da, forecast=point_forecast_da
        )

    def _check_forecast_data_availability(self):
        forecast = self.individual_case._subset_valid_times(self.forecast)
        lead_time_len = len(forecast.init_time)
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
        logger.info("Forecast data available for case %s", self.individual_case.id)
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
    case_results: dict[str, dict[str, Any]] = {}
    logger.info("Evaluating case %s, %s", individual_case.id, individual_case.title)
    gridded_obs_evaluation = CaseEvaluation(
        "gridded", gridded_obs, forecast_dataset, individual_case
    )
    point_obs_evaluation = CaseEvaluation(
        "point", point_obs, forecast_dataset, individual_case
    )

    gridded_case_eval = gridded_obs_evaluation.build_dataarray_subsets(compute=True)
    point_case_eval = point_obs_evaluation.build_dataarray_subsets(compute=True)
    for data_var in individual_case.data_vars:
        case_results[data_var] = {}
        for metric in individual_case.metrics_list:
            metric_instance = metric()
            logging.debug("metric %s computing", metric_instance.name)
            # TODO(aaTman): Create metric container object for gridded and point obs
            result = [
                metric_instance.compute(
                    eval.forecast[data_var], eval.observation[data_var]
                )
                for eval in [gridded_case_eval, point_case_eval]
            ]
            case_results[data_var][metric_instance.name] = result
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
