"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd
import polars as pl
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import cases, derived, utils

if TYPE_CHECKING:
    from extremeweatherbench import inputs, metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtremeWeatherBench:
    """A class to run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is a wrapper around the
    case operators and metrics to create either a serial loop or will return the built
    case operators to run in parallel as defined by the user.

    Attributes:
        cases: A dictionary of cases to run.
        metrics: A list of metrics to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the workflow.
    """

    def __init__(
        self,
        cases: dict[str, list],
        metrics: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.cases = cases
        self.metrics = metrics
        self.cache_dir = cache_dir

    # case operators as a property are a convenience method for users to use them outside the class
    # if desired for a parallel workflow
    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        return cases.build_case_operators(self.cases, self.metrics)

    def run(
        self,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the ExtremeWeatherBench workflow.

        This method will run the workflow in the order of the case operators, optionally
        caching the mid-flight outputs of the workflow if cache_dir was provided.

        Keyword arguments are passed to the metric computations if there are specific requirements
        needed for metrics such as threshold arguments.
        """
        # instantiate the cache directory if caching and build it if it does not exist
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                self.cache_dir = Path(self.cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        run_results = []
        with logging_redirect_tqdm():
            for case_operator in tqdm(self.case_operators):
                run_results.append(compute_case_operator(case_operator, **kwargs))

                # store the results of each case operator if caching
                if self.cache_dir:
                    pd.concat(run_results).to_pickle(
                        self.cache_dir / "case_results.pkl"
                    )
        return pd.concat(run_results, ignore_index=True)


def compute_case_operator(case_operator: "cases.CaseOperator", **kwargs):
    """Compute the results of a case operator.

    This method will compute the results of a case operator. It will build the target and forecast datasets,
    align them, compute the metrics, and return a concatenated dataframe of the results.

    Args:
        case_operator: The case operator to compute the results of.
        kwargs: Keyword arguments to pass to the metric computations.

    Returns:
        A concatenated dataframe of the results of the case operator.
    """
    target_ds, forecast_ds = _build_datasets(case_operator)

    # spatiotemporally align the target and forecast datasets dependent on the forecast
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # compute and cache the datasets if requested
    if kwargs.get("pre_compute", False):
        aligned_forecast_ds, aligned_target_ds = _compute_and_maybe_cache(
            aligned_forecast_ds,
            aligned_target_ds,
            cache_dir=kwargs.get("cache_dir", None),
        )

    time_aligned_target_ds, time_aligned_forecast_ds = [
        derived.maybe_derive_variables(ds, variables)
        for ds, variables in zip(
            [time_aligned_target_ds, time_aligned_forecast_ds],
            [case_operator.target.variables, case_operator.forecast.variables],
        )
    ]
    logger.info(f"datasets built for case {case_operator.case_metadata.case_id_number}")
    results = []
    # TODO: determine if derived variables need to be pushed here or at pre-compute
    for variables, metric in itertools.product(
        zip(
            case_operator.forecast.variables,
            case_operator.target.variables,
        ),
        case_operator.metric,
    ):
        results.append(
            _evaluate_metric_and_return_df(
                forecast_ds=aligned_forecast_ds,
                target_ds=aligned_target_ds,
                forecast_variable=variables[0],
                target_variable=variables[1],
                metric=metric,
                target_name=case_operator.target.name,
                case_id_number=case_operator.case_metadata.case_id_number,
                event_type=case_operator.case_metadata.event_type,
                **kwargs,
            )
        )

        # cache the results of each metric if caching
        if kwargs.get("cache_dir", None):
            results.to_pickle(kwargs.get("cache_dir") / "results.pkl")

    return pd.concat(results, ignore_index=True)


def _evaluate_metric_and_return_df(
    forecast_ds: xr.Dataset,
    target_ds: xr.Dataset,
    forecast_variable: Union[str, "derived.DerivedVariable"],
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_id_number: int,
    event_type: str,
    **kwargs,
):
    """Evaluate a metric and return a dataframe of the results.

    Args:
        forecast_ds: The forecast dataset.
        target_ds: The target dataset.
        forecast_variable: The forecast variable to evaluate.
        target_variable: The target variable to evaluate.
        metric: The metric to evaluate.
        case_id_number: The case id number.
        event_type: The event type.

    Returns:
        A dataframe of the results of the metric evaluation.
    """
    metric = metric()
    logger.info(f"computing metric {metric.name}")
    metric_result = metric.compute_metric(
        forecast_ds[forecast_variable],
        target_ds[target_variable],
        **kwargs,
    )

    # Convert to DataFrame and add metadata
    df = metric_result.to_dataframe(name="value").reset_index()
    df["target_variable"] = target_variable
    df["metric"] = metric.name
    df["target_source"] = target_ds.attrs["source"]
    df["forecast_source"] = forecast_ds.attrs["source"]
    df["case_id_number"] = case_id_number
    df["event_type"] = event_type
    return df


def _build_datasets(
    case_operator: "cases.CaseOperator",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target and forecast datasets,
    including preprocessing, variable renaming, and subsetting.
    """
    logger.info("running target pipeline")
    target_ds = run_pipeline(case_operator, "target")
    logger.info("running forecast pipeline")
    forecast_ds = run_pipeline(case_operator, "forecast")
    return (target_ds, forecast_ds)


def _compute_and_maybe_cache(
    *datasets: xr.Dataset, cache_dir: Optional[Union[str, Path]]
) -> list[xr.Dataset]:
    """Compute and cache the datasets if caching."""
    logger.info("computing datasets")
    computed_datasets = [dataset.compute() for dataset in datasets]
    if cache_dir:
        raise NotImplementedError("Caching is not implemented yet")
        # (computed_dataset.to_netcdf(self.cache_dir) for computed_dataset in computed_datasets)
    return computed_datasets


def run_pipeline(
    case_operator: "cases.CaseOperator",
    input_source: Literal["target", "forecast"],
) -> xr.Dataset:
    """
    Shared method for running the target pipeline.

    Args:
        case_operator: The case operator to run the pipeline on.
        input_source: The input source to run the pipeline on.

    Returns:
        The target data with a type determined by the user.
    """

    if input_source == "target":
        input_data = case_operator.target
    elif input_source == "forecast":
        input_data = case_operator.forecast
    else:
        raise ValueError(f"Invalid input source: {input_source}")

    # Open data and process through pipeline steps
    data = (
        # opens data from user-defined source
        input_data.open_and_maybe_preprocess_data_from_source()
        # maps variable names to the target data if not already using EWB naming conventions
        .pipe(
            utils.maybe_map_variable_names,
            variable_mapping=input_data.variable_mapping,
        )
        # subsets the target data using the caseoperator metadata
        .pipe(
            input_data.subset_data_to_case,
            case_operator=case_operator,
        )
        # converts the target data to an xarray dataset if it is not already
        .pipe(input_data.maybe_convert_to_dataset)
        .pipe(input_data.add_source_to_dataset_attrs)
    )
    return data


def maybe_map_variable_names(
    data: utils.IncomingDataInput, case_operator: "case.CaseOperator"
) -> utils.IncomingDataInput:
    """Map the variable names to the target data, if required.

    Args:
        data: The incoming data in the form of an object that has a rename method for data variables/columns.
        case_operator: The case operator to run the pipeline on.

    Returns:
        A dataset with mapped variable names, if any exist, else the original data.
    """
    variable_mapping = case_operator.target.variable_mapping
    target_and_maybe_derived_variables = (
        derived.maybe_pull_required_variables_from_derived_input(
            variable_mapping.keys()
        )
    )
    # check that the variables are in the target data
    if target_and_maybe_derived_variables and any(
        var not in data.variables for var in target_and_maybe_derived_variables
    ):
        raise ValueError(
            f"Variables {target_and_maybe_derived_variables} not found in target data"
        )
    # subset the variables if they exist in the target data
    elif target_and_maybe_derived_variables:
        variable_subset_data = data[target_and_maybe_derived_variables]
    else:
        raise ValueError(
            "Variables not defined. Please list at least one target variable to select."
        )
    if variable_mapping is None:
        return variable_subset_data
    # Filter the mapping to only include variables that exist in the dataset
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        subset_variable_mapping = {
            v: k
            for v, k in variable_mapping.items()
            if v in variable_subset_data.keys()
        }
    elif isinstance(data, (pl.DataFrame, pd.DataFrame)):
        subset_variable_mapping = {
            v: k
            for v, k in variable_mapping.items()
            if v in variable_subset_data.columns
        }
    elif isinstance(data, pl.LazyFrame):
        subset_variable_mapping = {
            v: k
            for v, k in variable_mapping.items()
            if v in variable_subset_data.collect_schema().names()
        }
    else:
        raise ValueError(
            (
                "Data is not a dataset, data array, lazy frame, dataframe, or pandas dataframe:"
                f"{type(variable_subset_data)}"
            )
        )
    if subset_variable_mapping:
        variable_subset_data = variable_subset_data.rename(subset_variable_mapping)
    return variable_subset_data
