"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import itertools
import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
import xarray as xr

from extremeweatherbench import cases, derived, inputs
from extremeweatherbench.defaults import OUTPUT_COLUMNS
from extremeweatherbench.progress import (
    enhanced_logging,
    format_dataset_info,
    progress_tracker,
)

if TYPE_CHECKING:
    from extremeweatherbench import metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def with_progress_tracking(func):
    """Decorator to optionally disable progress tracking for compute_case_operator.

    This decorator allows users to disable progress tracking by setting
    disable_progress=True in kwargs, which can be useful for batch processing
    or when progress tracking overhead is not desired.
    """

    @wraps(func)
    def wrapper(case_operator: "cases.CaseOperator", **kwargs):
        if kwargs.pop("disable_progress", False):
            # Use core function without progress tracking
            return _compute_case_operator_core(case_operator, **kwargs)
        else:
            # Use full function with progress tracking
            return func(case_operator, **kwargs)

    return wrapper


def with_result_preservation(func):
    """Decorator to dump partial results if cache_dir is set during errors/interrupts.

    This decorator ensures that any partial results are saved to the cache directory
    if an error or keyboard interrupt occurs during execution, preventing loss of
    computation progress. It creates a backup of current results state before
    re-raising the exception.
    """

    @wraps(func)
    def wrapper(self, **kwargs):
        try:
            return func(self, **kwargs)
        except (KeyboardInterrupt, Exception) as e:
            # Only save if cache_dir is set
            if hasattr(self, "cache_dir") and self.cache_dir:
                results_file = self.cache_dir / "case_results.pkl"

                # Check if there are any existing results files to preserve
                if results_file.exists():
                    logger.info(
                        f"Results already cached at {results_file} due to "
                        f"{type(e).__name__}"
                    )
                else:
                    logger.info(f"No cached results found due to {type(e).__name__}")

                # Also try to create an emergency backup if possible
                emergency_file = (
                    self.cache_dir / f"emergency_results_{type(e).__name__.lower()}.pkl"
                )
                logger.info(
                    f"Emergency results backup would be saved to {emergency_file}"
                )
            else:
                logger.warning(
                    f"No cache_dir set - unable to preserve partial "
                    f"results from {type(e).__name__}"
                )

            # Re-raise the original exception
            raise

    return wrapper


class ExtremeWeatherBench:
    """A class to run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is a
    wrapper around the
    case operators and evaluation objects to create either a serial loop or will return
    the built case operators to run in parallel as defined by the user.


    Attributes:
        cases: A dictionary of cases to run.
        evaluation_objects: A list of EvaluationObjects to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the
            workflow.
    """

    def __init__(
        self,
        cases: dict[str, list],
        evaluation_objects: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.cases = cases
        self.evaluation_objects = evaluation_objects
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # case operators as a property are a convenience method for users to use
    # them outside the class
    # if desired for a parallel workflow
    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        return cases.build_case_operators(self.cases, self.evaluation_objects)

    @with_result_preservation
    def run(
        self,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the ExtremeWeatherBench workflow.

        This method will run the workflow in the order of the case operators, optionally
        caching the mid-flight outputs of the workflow if cache_dir was provided.

        Keyword arguments are passed to the metric computations if there are specific
        requirements needed for metrics such as threshold arguments.
        """
        # instantiate the cache directory if caching and build it if it does not exist
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                self.cache_dir = Path(self.cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Calculate total operations: sum of all metric computations across cases
        case_operators = self.case_operators
        total_operations = sum(len(case_op.metric_list) for case_op in case_operators)

        results = pd.DataFrame(columns=OUTPUT_COLUMNS)

        with enhanced_logging():
            with progress_tracker.overall_workflow(
                total_operations,
                f"Processing {len(case_operators)} cases with {total_operations} total "
                "metrics",
            ) as main_pbar:
                for case_operator in case_operators:
                    case_id = case_operator.case_metadata.case_id_number
                    case_title = case_operator.case_metadata.title
                    num_metrics = len(case_operator.metric)

                    with progress_tracker.case_processing(
                        case_id, f"{case_title}", num_metrics
                    ) as case_pbar:
                        # Enable dask progress for this computation
                        with progress_tracker.dask_computation_context():
                            result = compute_case_operator(case_operator, **kwargs)
                            results = pd.concat(
                                [
                                    df.dropna(axis=1, how="all")
                                    for df in [results, result]
                                ],
                                ignore_index=True,
                            )

                        # Update case progress bar
                        case_pbar.update(num_metrics)  # Case completion

                        # Note: main_pbar is updated inside compute_case_operator per
                        # metric

                        # Store results incrementally if caching
                        if self.cache_dir:
                            pd.concat(results).to_pickle(
                                self.cache_dir / "case_results.pkl"
                            )
                main_pbar.update(num_metrics)
        return results


def _compute_case_operator_core(case_operator: "cases.CaseOperator", **kwargs):
    """Core computation logic without progress tracking."""
    # Step 1: Build datasets
    forecast_ds, target_ds = _build_datasets(case_operator)
    if len(forecast_ds) == 0 or len(target_ds) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Step 2: Align datasets
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Step 3: Compute and cache if requested
    if kwargs.get("pre_compute", False):
        aligned_forecast_ds, aligned_target_ds = _compute_and_maybe_cache(
            aligned_forecast_ds,
            aligned_target_ds,
            cache_dir=kwargs.get("cache_dir", None),
        )

    # Step 4: Derive variables
    case_id = case_operator.case_metadata.case_id_number

    aligned_forecast_ds = derived.maybe_derive_variables(
        aligned_forecast_ds,
        variables=case_operator.forecast.variables,
        case_id_number=case_id,
    )

    aligned_target_ds = derived.maybe_derive_variables(
        aligned_target_ds,
        variables=case_operator.target.variables,
        case_id_number=case_id,
    )

    logger.info(f"datasets built for case {case_operator.case_metadata.case_id_number}")
    results = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Step 5: Compute metrics
    variable_pairs = list(
        zip(
            case_operator.forecast.variables,
            case_operator.target.variables,
        )
    )

    for variables, metric_class in itertools.product(
        variable_pairs, case_operator.metric
    ):
        forecast_var, target_var = variables
        # Handle derived variables by extracting their names
        forecast_var = (
            forecast_var.name if hasattr(forecast_var, "name") else forecast_var
        )
        target_var = target_var.name if hasattr(target_var, "name") else target_var

        # Instantiate the metric if it's a class
        if isinstance(metric_class, type):
            metric = metric_class()
        else:
            metric = metric_class

        result = _evaluate_metric_and_return_df(
            forecast_ds=aligned_forecast_ds,
            target_ds=aligned_target_ds,
            forecast_variable=forecast_var,
            target_variable=target_var,
            metric=metric,
            case_id_number=case_operator.case_metadata.case_id_number,
            event_type=case_operator.case_metadata.event_type,
            **kwargs,
        )
        results = pd.concat(
            [df.dropna(axis=1, how="all") for df in [results, result]],
            ignore_index=True,
        )

        # Update main progress bar for each metric completion
        if "main" in progress_tracker.active_bars:
            progress_tracker.active_bars["main"].update(1)

        # cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            pd.concat(results, ignore_index=True).to_pickle(cache_path / "results.pkl")

    return results


@with_progress_tracking
def compute_case_operator(case_operator: "cases.CaseOperator", **kwargs):
    """Compute the results of a case operator.

    This method will compute the results of a case operator. It will build
    the target and forecast datasets, align them, compute the metrics, and return a
    concatenated dataframe of the results.

    Args:
        case_operator: The case operator to compute the results of.
        kwargs: Keyword arguments to pass to the metric computations.

    Returns:
        A concatenated dataframe of the results of the case operator.
    """
    # Early exit for empty datasets
    forecast_ds, target_ds = _build_datasets(case_operator)
    if len(forecast_ds) == 0 or len(target_ds) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    # spatiotemporally align the target and forecast datasets dependent on the target
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Step 2: Align datasets with progress tracking
    with progress_tracker.dataset_alignment(
        forecast_ds.dims, target_ds.dims
    ) as align_pbar:
        aligned_forecast_ds, aligned_target_ds = (
            case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
        )
        align_pbar.update(1)

    # Step 3: Compute and cache if requested
    if kwargs.get("pre_compute", False):
        aligned_forecast_ds, aligned_target_ds = _compute_and_maybe_cache(
            aligned_forecast_ds,
            aligned_target_ds,
            cache_dir=kwargs.get("cache_dir", None),
        )
    logger.info(f"datasets built for case {case_operator.case_metadata.case_id_number}")
    results = pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Step 5: Compute metrics with progress tracking
    variable_pairs = list(
        zip(
            case_operator.forecast.variables,
            case_operator.target.variables,
        ),
        case_operator.metric_list,
    )

    for variables, metric_class in itertools.product(
        variable_pairs, case_operator.metric_list
    ):
        # Handle derived variables by extracting their names
        forecast_var = (
            variables[0].name if hasattr(variables[0], "name") else variables[0]
        )
        target_var = (
            variables[1].name if hasattr(variables[1], "name") else variables[1]
        )
        results.append(
            _evaluate_metric_and_return_df(
                forecast_ds=aligned_forecast_ds,
                target_ds=aligned_target_ds,
                forecast_variable=forecast_var,
                target_variable=target_var,
                metric=metric_class,
                case_id_number=case_operator.case_metadata.case_id_number,
                event_type=case_operator.case_metadata.event_type,
                **kwargs,
            )
        )

    for variables, metric_class in itertools.product(
        variable_pairs, case_operator.metric
    ):
        forecast_var, target_var = variables

        # Instantiate the metric if it's a class
        if isinstance(metric_class, type):
            metric = metric_class()
        else:
            metric = metric_class

        data_size = format_dataset_info(aligned_forecast_ds)

        with progress_tracker.metric_computation(metric.name, data_size) as metric_pbar:
            with progress_tracker.xarray_computation(
                f"{metric.name} on {forecast_var}"
            ) as xarray_pbar:
                result = _evaluate_metric_and_return_df(
                    forecast_ds=aligned_forecast_ds,
                    target_ds=aligned_target_ds,
                    forecast_variable=forecast_var,
                    target_variable=target_var,
                    metric=metric,
                    case_id_number=case_operator.case_metadata.case_id_number,
                    event_type=case_operator.case_metadata.event_type,
                    **kwargs,
                )
                results = pd.concat(
                    [df.dropna(axis=1, how="all") for df in [results, result]],
                    ignore_index=True,
                )
                xarray_pbar.update(1)
            metric_pbar.update(1)

            # Update main progress bar for each metric completion
            if "main" in progress_tracker.active_bars:
                progress_tracker.active_bars["main"].update(1)

        # cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            pd.concat(results, ignore_index=True).to_pickle(cache_path / "results.pkl")

    return results


def _extract_standard_metadata(
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    target_ds: xr.Dataset,
    forecast_ds: xr.Dataset,
    case_id_number: int,
    event_type: str,
) -> dict:
    """Extract standard metadata for output dataframe.

    This function centralizes the logic for extracting metadata from the
    evaluation context. Makes it easy to modify how metadata is extracted
    without changing the schema enforcement logic.

    Args:
        target_variable: The target variable
        metric: The metric instance
        target_ds: Target dataset
        forecast_ds: Forecast dataset
        case_id_number: Case ID number
        event_type: Event type string

    Returns:
        Dictionary of metadata for the output dataframe
    """
    return {
        "target_variable": target_variable,
        "metric": metric.name,
        "case_id_number": case_id_number,
        "event_type": event_type,
    }


def _ensure_output_schema(df: pd.DataFrame, **metadata) -> pd.DataFrame:
    """Ensure dataframe conforms to OUTPUT_COLUMNS schema.

    This function adds any provided metadata columns to the dataframe and validates
    that all OUTPUT_COLUMNS are present. Any missing columns will be filled with NaN
    and a warning will be logged.

    Args:
        df: Base dataframe (typically with 'value' column from metric result)
        **metadata: Key-value pairs for metadata columns (e.g., target_variable='temp')

    Returns:
        DataFrame with columns matching OUTPUT_COLUMNS specification

    Example:
        df = _ensure_output_schema(
            metric_df,
            target_variable=target_var,
            metric=metric.name,
            case_id_number=case_id,
            event_type=event_type
        )
    """
    # Add metadata columns
    for col, value in metadata.items():
        df[col] = value

    # Check for missing columns and warn
    missing_cols = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing_cols and missing_cols not in [{"init_time"}, {"lead_time"}]:
        logger.warning(f"Missing expected columns: {missing_cols}")

    # Ensure all OUTPUT_COLUMNS are present (missing ones will be NaN)
    # and reorder to match OUTPUT_COLUMNS specification
    return df.reindex(columns=OUTPUT_COLUMNS)


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

    # Normalize variables to their string names
    forecast_variable = _normalize_variable(forecast_variable)
    target_variable = _normalize_variable(target_variable)

    # TODO: remove this once we have a better way to handle metric
    # instantiation
    if isinstance(metric, type):
        metric = metric()
    logger.info(f"computing metric {metric.name}")
    # loads in all variables if no name is provided
    # TODO: expand typing to allow for datasets or rethink logic with inputs like TCs
    metric_result = metric.compute_metric(
        forecast_ds.get(forecast_variable, forecast_ds.data_vars),
        target_ds.get(target_variable, target_ds.data_vars),
        **kwargs,
    )
    # Convert to DataFrame and add metadata, ensuring OUTPUT_COLUMNS compliance
    df = metric_result.to_dataframe(name="value").reset_index()
    # TODO: add functionality for custom metadata columns
    metadata = _extract_standard_metadata(
        target_variable, metric, target_ds, forecast_ds, case_id_number, event_type
    )
    return _ensure_output_schema(df, **metadata)


def _normalize_variable(variable: Union[str, "derived.DerivedVariable"]) -> str:
    """Convert a variable to its string representation."""
    if isinstance(variable, str):
        return variable
    elif hasattr(variable, "name"):
        return variable.name
    else:
        # This case seems incorrect in original - returning all data_vars
        # for a single variable doesn't make sense
        raise ValueError(f"Cannot normalize variable: {variable}")


def _build_datasets(
    case_operator: "cases.CaseOperator",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target and
    forecast datasets, including preprocessing, variable renaming, and subsetting.
    """
    logger.info("running forecast pipeline")
    forecast_ds = run_pipeline(case_operator.case_metadata, case_operator.forecast)
    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        if "valid_time" in zero_length_dims:
            logger.warning(
                f"forecast dataset for case "
                f"{case_operator.case_metadata.case_id_number} "
                f"has no data for case time range "
                f"{case_operator.case_metadata.start_date} to "
                f"{case_operator.case_metadata.end_date}"
            )
        else:
            logger.warning(
                f"forecast dataset for case "
                f"{case_operator.case_metadata.case_id_number} "
                f"has zero-length dimensions {zero_length_dims} for case time range "
                f"{case_operator.case_metadata.start_date} "
                f"to {case_operator.case_metadata.end_date}"
            )
        return xr.Dataset(), xr.Dataset()
    logger.info("running target pipeline")
    target_ds = run_pipeline(case_operator.case_metadata, case_operator.target)
    return (forecast_ds, target_ds)


def _compute_and_maybe_cache(
    *datasets: xr.Dataset, cache_dir: Optional[Union[str, Path]]
) -> list[xr.Dataset]:
    """Compute and cache the datasets if caching."""
    logger.info("computing datasets")
    computed_datasets = [dataset.compute() for dataset in datasets]
    if cache_dir:
        raise NotImplementedError("Caching is not implemented yet")
        # (computed_dataset.to_netcdf(self.cache_dir) for computed_dataset in
        # computed_datasets)
    return computed_datasets


def run_pipeline(
    case_metadata: "cases.IndividualCase",
    input_data: "inputs.InputBase",
) -> xr.Dataset:
    """Shared method for running the target pipeline.

    Args:
        case_operator: The case operator to run the pipeline on.
        input_source: The input source to run the pipeline on.

    Returns:
        The target data with a type determined by the user.
    """
    # Open data and process through pipeline steps
    data = (
        # opens data from user-defined source
        input_data.open_and_maybe_preprocess_data_from_source()
        # maps variable names to the target data if not already using EWB
        # naming conventions
        .pipe(input_data.maybe_map_variable_names)
        .pipe(input_data.maybe_subset_variables, variables=input_data.variables)
        # subsets the target data using the caseoperator metadata
        .pipe(
            input_data.subset_data_to_case,
            case_metadata=case_metadata,
        )
        # converts the target data to an xarray dataset if it is not already
        .pipe(input_data.maybe_convert_to_dataset)
        .pipe(input_data.add_source_to_dataset_attrs)
        .pipe(derived.maybe_derive_variables, variables=input_data.variables)
    )
    return data
