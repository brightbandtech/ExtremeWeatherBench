"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import cases, derived, inputs

if TYPE_CHECKING:
    from extremeweatherbench import metrics


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
        self.cache_dir = Path(cache_dir) if cache_dir else None

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
        if run_results:
            return pd.concat(run_results, ignore_index=True)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "value",
                    "target_variable",
                    "metric",
                    "target_source",
                    "forecast_source",
                    "case_id_number",
                    "event_type",
                ]
            )


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
    forecast_ds, target_ds = _build_datasets(case_operator)

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

    aligned_forecast_ds, aligned_target_ds = [
        derived.maybe_derive_variables(ds, variables)
        for ds, variables in zip(
            [aligned_forecast_ds, aligned_target_ds],
            [case_operator.forecast.variables, case_operator.target.variables],
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
                case_id_number=case_operator.case_metadata.case_id_number,
                event_type=case_operator.case_metadata.event_type,
                **kwargs,
            )
        )

        # cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            pd.concat(results, ignore_index=True).to_pickle(cache_path / "results.pkl")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "value",
                "target_variable",
                "metric",
                "target_source",
                "forecast_source",
                "case_id_number",
                "event_type",
            ]
        )


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
    logger.info("running forecast pipeline")
    forecast_ds = run_pipeline(case_operator, "forecast")
    logger.info("running target pipeline")
    target_ds = run_pipeline(case_operator, "target")
    return (forecast_ds, target_ds)


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
    """Shared method for running the target pipeline.

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
        .pipe(input_data.maybe_map_variable_names)
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
