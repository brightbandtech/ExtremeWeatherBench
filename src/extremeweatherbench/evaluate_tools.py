"""Evaluation tools for metric-level parallelization.

This module provides functions for building and executing metric jobs with
parallelization at both the case operator and metric levels.

Key design principles:
- No global state - all caching via joblib.Memory passed as parameter
- Small, focused, testable functions
- Parallelize all independent loops
"""

import contextlib
import copy
import dataclasses
import logging
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Union

import dask.array
import joblib
import pandas as pd
import sparse
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.dask import TqdmCallback

from extremeweatherbench import cases, derived, inputs, metrics, sources, utils

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


OUTPUT_COLUMNS = [
    "value",
    "lead_time",
    "init_time",
    "target_variable",
    "metric",
    "forecast_source",
    "target_source",
    "case_id_number",
    "event_type",
]


@dataclasses.dataclass
class MetricJob:
    """A single metric computation job for parallel execution.

    Attributes:
        case_operator: The case operator containing metadata and input sources.
        metric: The metric instance to compute.
        forecast_var: The forecast variable name.
        target_var: The target variable name.
        metric_kwargs: Pre-computed kwargs for metric evaluation.
    """

    case_operator: "cases.CaseOperator"
    metric: "metrics.BaseMetric"
    forecast_var: str
    target_var: str
    metric_kwargs: dict


@dataclasses.dataclass
class PreparedDatasets:
    """Container for aligned forecast and target datasets.

    Attributes:
        forecast: Aligned forecast dataset.
        target: Aligned target dataset.
    """

    forecast: xr.Dataset
    target: xr.Dataset


def evaluate_case_operators(
    case_operators: list["cases.CaseOperator"],
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in parallel or serial.

    Args:
        case_operators: List of case operators to run.
        cache_dir: Optional directory for caching.
        **kwargs: Additional arguments, may include 'parallel_config' dict.

    Returns:
        List of result DataFrames.
    """
    with logging_redirect_tqdm():
        with _dataset_cache(cache_dir) as memory:
            kwargs["memory"] = memory
            metric_jobs = _run_parallel_or_serial(
                items=case_operators,
                func=_create_metric_jobs_for_case_operator,
                cache_dir=cache_dir,
                desc="Creating metric jobs",
                return_as="list",
                total_tasks=len(case_operators),
                **kwargs,
            )
            return _run_parallel_or_serial(
                items=metric_jobs,
                func=_compute_single_metric,
                cache_dir=cache_dir,
                desc="Computing metrics",
                return_as="list",
                total_tasks=len(metric_jobs),
                **kwargs,
            )


def _create_metric_jobs_for_case_operator(
    case_operator: "cases.CaseOperator",
    memory: joblib.Memory,
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[MetricJob]:
    """Create MetricJob objects for all metrics in a case operator.

    This is the unit of work for parallel processing across case operators.

    Args:
        case_operator: The case operator to process.
        memory: joblib.Memory instance for caching.
        cache_dir: Optional directory for zarr caching.
        **kwargs: Additional kwargs passed to pipeline.

    Returns:
        List of MetricJob objects, one per metric/variable combination.
    """
    case_operator = _validate_case_operator_metrics(case_operator)

    datasets = _prepare_aligned_datasets(
        case_operator, memory, cache_dir=cache_dir, **kwargs
    )

    if datasets is None:
        logger.info(
            "Skipping case %s: empty datasets",
            case_operator.case_metadata.case_id_number,
        )
        return []

    logger.info(
        "Datasets built for case %s", case_operator.case_metadata.case_id_number
    )

    metric_jobs = []

    # Collect claimed variables
    claimed_forecast, claimed_target = _collect_claimed_variables(
        case_operator.metric_list
    )

    for metric in case_operator.metric_list:
        # Expand composite metrics
        metrics_to_evaluate = metric.maybe_expand_composite()

        # Get variable pairs for this metric
        variable_pairs = _get_variable_pairs_for_metric(
            metric, case_operator, claimed_forecast, claimed_target
        )

        for forecast_var, target_var in variable_pairs:
            # Prepare metric kwargs
            metric_kwargs = metric.maybe_prepare_composite_kwargs(
                forecast_data=datasets.forecast[forecast_var],
                target_data=datasets.target[target_var],
            )

            # Create job for each expanded metric
            for single_metric in metrics_to_evaluate:
                metric_jobs.append(
                    MetricJob(
                        case_operator=case_operator,
                        metric=single_metric,
                        forecast_var=forecast_var,
                        target_var=target_var,
                        metric_kwargs=metric_kwargs,
                    )
                )

    return metric_jobs


def _prepare_aligned_datasets(
    case_operator: "cases.CaseOperator",
    memory: joblib.Memory,
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> Optional[PreparedDatasets]:
    """Build and align datasets for a case operator with caching.

    Uses joblib.Memory for caching the dataset building step.

    Args:
        case_operator: The case operator to build datasets for.
        memory: joblib.Memory instance for caching.
        cache_dir: Optional directory for zarr caching of aligned datasets.
        **kwargs: Additional kwargs passed to pipeline.

    Returns:
        PreparedDatasets with aligned forecast and target, or None if empty.
    """

    # Cache the build_case_operator_datasets function to avoid re-running the pipeline for the same
    # case operator
    cached_build = memory.cache(build_case_operator_datasets)
    forecast_ds, target_ds = cached_build(case_operator, **kwargs)

    # Check for empty datasets
    if 0 in forecast_ds.sizes.values() or 0 in target_ds.sizes.values():
        return None
    if len(forecast_ds.sizes) == 0 or len(target_ds.sizes) == 0:
        return None

    # Align datasets
    aligned_forecast, aligned_target = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Optionally cache to zarr
    if cache_dir is not None:
        aligned_forecast = utils.maybe_cache_and_compute(
            aligned_forecast,
            cache_dir=cache_dir,
            name=f"{case_operator.case_metadata.case_id_number}_"
            f"{case_operator.forecast.name}",
        )
        aligned_target = utils.maybe_cache_and_compute(
            aligned_target,
            cache_dir=cache_dir,
            name=f"{case_operator.case_metadata.case_id_number}_"
            f"{case_operator.target.name}",
        )

    return PreparedDatasets(forecast=aligned_forecast, target=aligned_target)


def _compute_single_metric(
    job: MetricJob,
    memory: joblib.Memory,
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute a single metric from a MetricJob.

    This function reloads cached datasets and computes one metric.

    Args:
        job: The MetricJob containing computation details.
        memory: joblib.Memory instance for caching.
        cache_dir: Optional directory for zarr caching.
        **kwargs: Additional kwargs passed to pipeline.

    Returns:
        DataFrame with metric results.
    """

    # Use cache to call this function again and get datasets
    datasets = _prepare_aligned_datasets(
        job.case_operator, memory, cache_dir=cache_dir, **kwargs
    )

    if datasets is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return evaluate_metric_and_return_df(
        datasets=datasets,
        forecast_variable=job.forecast_var,
        target_variable=job.target_var,
        metric=job.metric,
        case_operator=job.case_operator,
        **job.metric_kwargs,
    )


def build_case_operator_datasets(
    case_operator: "cases.CaseOperator",
    **kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target
    and forecast datasets, including preprocessing, variable renaming, and
    subsetting. It augments the InputBase variables with any variables defined
    in metrics to ensure all required variables are loaded and derived.

    If any forecast variable has `requires_target_dataset=True`, the target
    dataset will be passed to the forecast pipeline via `_target_dataset` in
    kwargs.

    Args:
        case_operator: The case operator containing metadata and input sources.
        **kwargs: Additional keyword arguments to pass to pipeline steps.

    Returns:
        A tuple containing (forecast_dataset, target_dataset). If either dataset
        has no dimensions, both will be empty datasets.
    """
    metric_forecast_vars, metric_target_vars = _collect_metric_variables(
        case_operator.metric_list
    )

    # Get all output_variables from DerivedVariables in InputBase
    forecast_derived_outputs = _get_all_derived_output_variables(
        case_operator.forecast.variables
    )
    target_derived_outputs = _get_all_derived_output_variables(
        case_operator.target.variables
    )

    # Filter out string variables that are output_variables of existing
    # DerivedVariables
    filtered_forecast_vars = {
        v
        for v in metric_forecast_vars
        if not (isinstance(v, str) and v in forecast_derived_outputs)
    }
    filtered_target_vars = {
        v
        for v in metric_target_vars
        if not (isinstance(v, str) and v in target_derived_outputs)
    }

    # Create augmented copies of InputBase objects with combined variables
    augmented_forecast = copy.copy(case_operator.forecast)
    augmented_target = copy.copy(case_operator.target)

    # Combine InputBase variables with metric-specific variables (filtered)
    augmented_forecast.variables = list(
        set(case_operator.forecast.variables) | filtered_forecast_vars
    )
    augmented_target.variables = list(
        set(case_operator.target.variables) | filtered_target_vars
    )

    logger.info("Running target pipeline... ")
    with TqdmCallback(
        desc=f"Running target pipeline for case "
        f"{case_operator.case_metadata.case_id_number}"
    ):
        target_ds = run_input_data_pipeline(
            case_operator.case_metadata, augmented_target, **kwargs
        )

    # Pass target dataset to forecast pipeline only if needed
    needs_target = any(
        getattr(var, "requires_target_dataset", False)
        for var in case_operator.forecast.variables
        if hasattr(var, "requires_target_dataset")
    )
    if needs_target:
        kwargs["_target_dataset"] = target_ds
        logger.debug(
            "Passing target dataset to forecast pipeline (required by derived variable)"
        )

    logger.info("Running forecast pipeline... ")
    with TqdmCallback(
        desc=f"Running forecast pipeline for case "
        f"{case_operator.case_metadata.case_id_number}"
    ):
        forecast_ds = run_input_data_pipeline(
            case_operator.case_metadata, augmented_forecast, **kwargs
        )

    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        if "valid_time" in zero_length_dims:
            logger.warning(
                "Forecast dataset %s for case %s has no data for case time range "
                "%s to %s."
                % (
                    case_operator.forecast.name,
                    case_operator.case_metadata.case_id_number,
                    case_operator.case_metadata.start_date,
                    case_operator.case_metadata.end_date,
                )
            )
        else:
            logger.warning(
                "Forecast dataset %s for case %s has zero-length dimensions %s for "
                "case time range %s to %s."
                % (
                    case_operator.forecast.name,
                    case_operator.case_metadata.case_id_number,
                    zero_length_dims,
                    case_operator.case_metadata.start_date,
                    case_operator.case_metadata.end_date,
                )
            )
        return xr.Dataset(), xr.Dataset()
    return (forecast_ds, target_ds)


@contextlib.contextmanager
def _dataset_cache(cache_dir: Optional[Union[str, pathlib.Path]] = None):
    """Context manager that provides a joblib.Memory cache.

    If cache_dir is provided, uses that directory for caching.
    Otherwise, creates a temporary directory that is cleaned up on exit.

    Args:
        cache_dir: Optional directory for caching. If None, uses a temp dir.

    Yields:
        joblib.Memory instance for caching.

    Example:
        with _dataset_cache() as memory:
            cached_func = memory.cache(expensive_function)
            result = cached_func(args)
    """
    if cache_dir is not None:
        cache_path = pathlib.Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        yield joblib.Memory(str(cache_path), verbose=0)
    else:
        with tempfile.TemporaryDirectory(prefix="ewb_cache_") as tmpdir:
            yield joblib.Memory(tmpdir, verbose=0)


def _validate_case_operator_metrics(
    case_operator: "cases.CaseOperator",
) -> "cases.CaseOperator":
    """Validate and normalize metrics in a case operator.

    Instantiates any metric classes that weren't instantiated, and validates
    that all metrics are BaseMetric instances.

    Args:
        case_operator: The case operator to validate.

    Returns:
        CaseOperator with validated metric_list.

    Raises:
        TypeError: If any metric is not a BaseMetric instance.
    """
    metric_list = list(case_operator.metric_list)

    for i, metric in enumerate(metric_list):
        if isinstance(metric, type):
            metric_list[i] = metric()
            logger.warning(
                "Metric %s instantiated with default parameters",
                metric_list[i].name,
            )
        if not isinstance(metric_list[i], metrics.BaseMetric):
            raise TypeError(f"Metric must be a BaseMetric instance, got {type(metric)}")

    return dataclasses.replace(case_operator, metric_list=metric_list)


def _collect_claimed_variables(
    metric_list: Sequence["metrics.BaseMetric"],
) -> tuple[set[str], set[str]]:
    """Collect explicitly claimed variables from metrics.

    Variables are "claimed" when a metric explicitly specifies forecast_variable
    and target_variable. These should not be used by metrics without explicit
    variable specifications.

    Args:
        metric_list: List of metrics to collect claimed variables from.

    Returns:
        Tuple of (claimed_forecast_vars, claimed_target_vars) as sets of strings.
    """
    claimed_forecast = set()
    claimed_target = set()

    for metric in metric_list:
        if metric.forecast_variable is not None and metric.target_variable is not None:
            claimed_forecast.update(
                _maybe_expand_variable_to_string(metric.forecast_variable)
            )
            claimed_target.update(
                _maybe_expand_variable_to_string(metric.target_variable)
            )

    return claimed_forecast, claimed_target


def _get_variable_pairs_for_metric(
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
    claimed_forecast_vars: set[str],
    claimed_target_vars: set[str],
) -> list[tuple[str, str]]:
    """Get variable pairs for a single metric.

    If the metric has explicit variables, uses those. Otherwise, uses all
    available variables from the InputBase objects, excluding claimed ones.

    Args:
        metric: The metric to get variable pairs for.
        case_operator: The case operator with input sources.
        claimed_forecast_vars: Variables claimed by other metrics.
        claimed_target_vars: Variables claimed by other metrics.

    Returns:
        List of (forecast_var, target_var) string tuples.
    """
    if metric.forecast_variable is not None and metric.target_variable is not None:
        forecast_vars = _maybe_expand_variable_to_string(metric.forecast_variable)
        target_vars = _maybe_expand_variable_to_string(metric.target_variable)
        return list(zip(forecast_vars, target_vars))

    # Use all InputBase variable pairs, excluding claimed variables
    forecast_vars = []
    for var in case_operator.forecast.variables:
        forecast_vars.extend(_maybe_expand_variable_to_string(var))

    target_vars = []
    for var in case_operator.target.variables:
        target_vars.extend(_maybe_expand_variable_to_string(var))

    forecast_available = [v for v in forecast_vars if v not in claimed_forecast_vars]
    target_available = [v for v in target_vars if v not in claimed_target_vars]

    return list(zip(forecast_available, target_available))


def _maybe_expand_variable_to_string(
    variable: Union[str, "derived.DerivedVariable"],
) -> list[str]:
    """Expand a variable to its string names.

    Args:
        variable: Either a string variable name or a DerivedVariable.

    Returns:
        List of string variable names.
    """
    if isinstance(variable, str):
        return [variable]
    elif isinstance(variable, derived.DerivedVariable):
        if hasattr(variable, "output_variables") and variable.output_variables:
            return variable.output_variables
        else:
            return [str(variable.name)]
    else:
        return [str(variable)]


def _get_all_derived_output_variables(
    variables: Sequence[Union[str, "derived.DerivedVariable"]],
) -> set[str]:
    """Get all output_variables from DerivedVariables in a list.

    Args:
        variables: Sequence that may contain DerivedVariable instances.

    Returns:
        Set of all output_variable names from DerivedVariables.
    """
    output_vars = set()
    for var in variables:
        if isinstance(var, derived.DerivedVariable):
            if hasattr(var, "output_variables") and var.output_variables:
                output_vars.update(var.output_variables)
    return output_vars


def _collect_metric_variables(
    metric_list: Sequence["metrics.BaseMetric"],
) -> tuple[
    set[Union[str, "derived.DerivedVariable"]],
    set[Union[str, "derived.DerivedVariable"]],
]:
    """Collect unique variables from metrics that have them defined.

    When a metric has a DerivedVariable with output_variables defined,
    the DerivedVariable instance is added to ensure it gets computed
    during pipeline execution.

    Args:
        metric_list: Sequence of metrics to extract variables from.

    Returns:
        Tuple of (forecast_variables, target_variables) as sets.
    """
    forecast_vars = set()
    target_vars = set()

    for metric in metric_list:
        if metric.forecast_variable is not None:
            forecast_vars.add(metric.forecast_variable)
        if metric.target_variable is not None:
            target_vars.add(metric.target_variable)

    return forecast_vars, target_vars


def run_input_data_pipeline(
    case_metadata: "cases.IndividualCase",
    input_data: "inputs.InputBase",
    **kwargs,
) -> xr.Dataset:
    """Shared method for running an input pipeline.

    Args:
        case_metadata: The case metadata to run the pipeline on.
        input_data: The input data to run the pipeline on.

    Returns:
        The processed input data as an xarray dataset.
    """
    # Open data and process through pipeline steps
    data = input_data.open_and_maybe_preprocess_data_from_source().pipe(
        lambda ds: input_data.maybe_map_variable_names(ds)
    )

    # Get the appropriate source module for the data type
    source_module = sources.get_backend_module(type(data))

    # Checks if the data has valid times and spatial overlap
    if inputs.check_for_missing_data(
        data,
        case_metadata,
        source_module=source_module,
    ):
        valid_data = (
            inputs.maybe_subset_variables(
                data,
                variables=input_data.variables,
                source_module=source_module,
            )
            .pipe(
                lambda ds: input_data.subset_data_to_case(ds, case_metadata, **kwargs)
            )
            .pipe(input_data.maybe_convert_to_dataset)
            .pipe(input_data.add_source_to_dataset_attrs)
            .pipe(
                lambda ds: derived.maybe_derive_variables(
                    ds,
                    variables=input_data.variables,
                    case_metadata=case_metadata,
                    **kwargs,
                )
            )
        )
        return valid_data
    else:
        logger.warning(
            "Data input %s for case %s has no data for case time range %s to %s."
            % (
                input_data.name,
                case_metadata.case_id_number,
                case_metadata.start_date.strftime("%Y-%m-%d %H:%M:%S"),
                case_metadata.end_date.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        return xr.Dataset()


def extract_standard_metadata(
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
) -> dict:
    """Extract standard metadata for output dataframe.

    Args:
        target_variable: The target variable
        metric: The metric instance
        case_operator: The CaseOperator holding associated case metadata

    Returns:
        Dictionary of metadata for the output dataframe
    """
    return {
        "target_variable": target_variable,
        "metric": metric.name,
        "target_source": case_operator.target.name,
        "forecast_source": case_operator.forecast.name,
        "case_id_number": case_operator.case_metadata.case_id_number,
        "event_type": case_operator.case_metadata.event_type,
    }


def _ensure_output_schema(df: pd.DataFrame, **metadata) -> pd.DataFrame:
    """Ensure dataframe conforms to OUTPUT_COLUMNS schema.

    This function adds any provided metadata columns to the dataframe and
    validates that all OUTPUT_COLUMNS are present.

    Args:
        df: Base dataframe (typically with 'value' column from metric result)
        **metadata: Key-value pairs for metadata columns

    Returns:
        DataFrame with columns matching OUTPUT_COLUMNS specification
    """
    for col, value in metadata.items():
        df[col] = value

    missing_cols = set(OUTPUT_COLUMNS) - set(df.columns)

    # One of init_time or lead_time should be present, but not both
    init_time_missing = "init_time" in missing_cols
    lead_time_missing = "lead_time" in missing_cols

    if init_time_missing != lead_time_missing:
        missing_cols.discard("init_time")
        missing_cols.discard("lead_time")

    if missing_cols:
        logger.warning("Missing expected columns: %s.", missing_cols)

    return df.reindex(columns=OUTPUT_COLUMNS)


def safe_concat(
    dataframes: list[pd.DataFrame], ignore_index: bool = True
) -> pd.DataFrame:
    """Safely concatenate DataFrames, filtering out empty ones.

    This function prevents FutureWarnings from pd.concat when dealing with
    empty or all-NA DataFrames by filtering them out before concatenation.

    Args:
        dataframes: List of DataFrames to concatenate
        ignore_index: Whether to ignore index during concatenation

    Returns:
        Concatenated DataFrame, or empty DataFrame with OUTPUT_COLUMNS if all
        input DataFrames are empty.
    """
    valid_dfs = []
    for i, df in enumerate(dataframes):
        if df.empty:
            logger.debug("Skipping empty DataFrame %s", i)
            continue
        if df.isna().all().all():
            logger.debug("Skipping all-NA DataFrame %s", i)
            continue
        if len(df.columns) > 0 and all(df[col].isna().all() for col in df.columns):
            logger.debug("Skipping DataFrame %s with all-NA columns", i)
            continue

        valid_dfs.append(df)

    if valid_dfs:
        if len(valid_dfs) > 1:
            reference_df = valid_dfs[0]
            has_dtype_mismatch = False

            for df in valid_dfs[1:]:
                for col in reference_df.columns:
                    if col in df.columns:
                        if reference_df[col].dtype != df[col].dtype:
                            has_dtype_mismatch = True
                            break
                if has_dtype_mismatch:
                    break

            if has_dtype_mismatch:
                consistent_dfs = [df.astype(object) for df in valid_dfs]
                return pd.concat(consistent_dfs, ignore_index=ignore_index)

        return pd.concat(valid_dfs, ignore_index=ignore_index)
    else:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _run_parallel_or_serial(
    items: Sequence,
    func: Callable,
    parallel_config: Optional[dict] = None,
    desc: str = "Processing",
    kwargs_extractor: Optional[Callable[[Any], dict]] = None,
    return_as: Literal["list", "generator_ordered", "generator_unordered"] = "list",
    total_tasks: Optional[int] = None,
    **common_kwargs,
) -> list:
    """Generic parallel/serial executor based on parallel_config.

    This function provides a unified interface for running tasks either in
    parallel (using joblib) or serially (using a simple loop), depending on
    the provided parallel_config.

    Args:
        items: Sequence of items to process.
        func: Function to apply to each item. Should accept an item as the
            first argument plus any kwargs.
        parallel_config: Optional dictionary of joblib parallel configuration.
            If None, runs serially. If provided, runs in parallel using joblib.
        desc: Description for the progress bar.
        kwargs_extractor: Optional callable that takes an item and returns a
            dict of additional kwargs specific to that item. Used when
            different items need different arguments (e.g., forecast building
            needs per-case target datasets).
        **common_kwargs: Additional keyword arguments passed to func for all
            items.

    Returns:
        List of results from applying func to each item.
    """

    # Serial execution, return before parallel execution
    if parallel_config is None:
        # Serial execution with tqdm progress bar
        results = []
        for item in tqdm(items, desc=desc):
            item_kwargs = common_kwargs.copy()
            if kwargs_extractor is not None:
                item_kwargs.update(kwargs_extractor(item))
            results.append(func(item, **item_kwargs))
        return results


    # Parallel execution; warn if no number of jobs is provided
    if parallel_config.get("n_jobs") is None:
        logger.warning("No number of jobs provided, using joblib backend default.")

    dask_client = _maybe_create_dask_client(parallel_config)

    # Use logging redirection to try to get progress bar working separate from
    # logging outputs
    with joblib.parallel_config(**parallel_config):
        # If we have a kwargs_extractor, build delayed calls with per-item kwargs
        if kwargs_extractor is not None:
            # Build delayed calls with per-item kwargs
            delayed_calls = []
            for item in items:
                item_kwargs = common_kwargs.copy()
                item_kwargs.update(kwargs_extractor(item))
                delayed_calls.append(joblib.delayed(func)(item, **item_kwargs))
        # Simple case: same kwargs for all items        
        else:
            delayed_calls = (joblib.delayed(func)(item, **common_kwargs) for item in items)
        
        # Run the delayed calls in parallel and return the results
        results = utils.ParallelTqdm(
                    total_tasks=total_tasks, desc=desc, return_as=return_as
                )(delayed_calls)

    # If we created a dask client, close it
    if dask_client is not None:
        logger.info("Closing dask client")
        dask_client.close()

    # Flatten the results if it's a nested list. joblib should return a consistent
    # structure, i.e. if the first item is a list, all items should be lists.
    # If it's not a list, return the results as is. Generators are the primary
    # alternative output when `return_as` is set to "generator_unordered" or "generator"
    if isinstance(results, list) and results and isinstance(results[0], list):
        return [result for results in results for result in results]
    else:
        return results


def _maybe_create_dask_client(parallel_config: dict):
    """Create a dask client if the parallel_config is set to dask."""
    if parallel_config.get("backend") == "dask":
        try:
            from dask.distributed import Client, LocalCluster

            # Check if a client already exists
            try:
                Client.current()
                logger.info("Using existing dask client")
            except ValueError:
                # No client exists, create a local one
                logger.info("Creating local dask client for parallel execution")
                dask_client = Client(LocalCluster(processes=True, silence_logs=False))
                logger.info("Dask client created: %s", dask_client)
        except ImportError:
            raise ImportError(
                "Dask is required for dask backend. "
                "Install with: pip install dask[distributed]"
            )
        return dask_client
    else:
        return None


def evaluate_metric_and_return_df(
    datasets: PreparedDatasets,
    forecast_variable: Union[str, "derived.DerivedVariable"],
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
    **kwargs,
) -> pd.DataFrame:
    """Evaluate a metric and return a dataframe of the results.

    Args:
        datasets: The prepared datasets.
        forecast_variable: The forecast variable to evaluate.
        target_variable: The target variable to evaluate.
        metric: The metric to evaluate.
        case_operator: The case operator with metadata for evaluation.
        **kwargs: Additional keyword arguments to pass to metric computation.

    Returns:
        A dataframe of the results with standard output schema columns.
    """
    # Normalize variables to their string names if needed
    forecast_variable = derived._maybe_convert_variable_to_string(forecast_variable)
    target_variable = derived._maybe_convert_variable_to_string(target_variable)

    logger.info("Computing metric %s... ", metric.name)

    # Extract the appropriate data for the metric
    if forecast_variable not in datasets.forecast.data_vars:
        raise ValueError(
            f"Variable '{forecast_variable}' not found in forecast dataset. "
            f"Available variables: {list(datasets.forecast.data_vars)}"
        )

    if target_variable not in datasets.target.data_vars:
        raise ValueError(
            f"Variable '{target_variable}' not found in target dataset. "
            f"Available variables: {list(datasets.target.data_vars)}"
        )

    forecast_data = datasets.forecast[forecast_variable]
    target_data = datasets.target[target_variable]

    metric_result = metric.compute_metric(
        forecast_data,
        target_data,
        **kwargs,
    )

    # If data is sparse, densify it
    if isinstance(metric_result.data, sparse.COO):
        metric_result.data = metric_result.data.maybe_densify()
    elif isinstance(metric_result.data, dask.array.Array) and isinstance(
        metric_result.data._meta, sparse.COO
    ):
        metric_result.data = metric_result.data.map_blocks(
            lambda x: x.maybe_densify(), dtype=metric_result.data.dtype
        )

    df = metric_result.to_dataframe(name="value").reset_index()
    metadata = extract_standard_metadata(target_variable, metric, case_operator)
    return _ensure_output_schema(df, **metadata)
