"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import copy
import logging
import pathlib
from typing import TYPE_CHECKING, Optional, Sequence, Union

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
    from extremeweatherbench import regions

logger = logging.getLogger(__name__)

# Columns for the evaluation output dataframe
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


class ExtremeWeatherBench:
    """A class to build and run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is ultimately a
    wrapper around case operators and evaluation objects to create a parallel or
    serial run to evaluate cases and metrics, returning a concatenated dataframe of the
    results.

    Attributes:
        case_metadata: A dictionary of cases or an IndividualCaseCollection to run.
        evaluation_objects: A list of evaluation objects to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the
            workflow for serial runs.
        region_subsetter: An optional region subsetter to subset the cases that are part
        of the evaluation to a Region object or a dictionary of lat/lon bounds.
    """

    def __init__(
        self,
        case_metadata: Union[dict[str, list], "cases.IndividualCaseCollection"],
        evaluation_objects: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        region_subsetter: Optional["regions.RegionSubsetter"] = None,
    ):
        if isinstance(case_metadata, dict):
            self.case_metadata = cases.load_individual_cases(case_metadata)
        elif isinstance(case_metadata, cases.IndividualCaseCollection):
            self.case_metadata = case_metadata
        else:
            raise TypeError(
                "case_metadata must be a dictionary of cases or an "
                "IndividualCaseCollection"
            )
        self.evaluation_objects = evaluation_objects
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        self.region_subsetter = region_subsetter

    # Case operators as a property can be used as a convenience method for a workflow
    # independent of the class.
    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        """Build the CaseOperator objects from case_metadata and evaluation_objects."""

        # Subset the cases if a region subsetter was provided
        if self.region_subsetter:
            subset_collection = self.region_subsetter.subset_case_collection(
                self.case_metadata
            )
        else:
            subset_collection = self.case_metadata
        return cases.build_case_operators(subset_collection, self.evaluation_objects)

    def run(
        self,
        n_jobs: Optional[int] = None,
        parallel_config: Optional[dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the ExtremeWeatherBench workflow.

        This method will run the workflow in the order of the case operators, optionally
        caching the mid-flight outputs of the workflow if cache_dir was provided for
        serial runs.

        Args:
            n_jobs: The number of jobs to run in parallel. If None, defaults to the
                joblib backend default value. If 1, the workflow will run serially.
                Ignored if parallel_config is provided.
            parallel_config: Optional dictionary of joblib parallel configuration.
                If provided, this takes precedence over n_jobs. If not provided and
                n_jobs is specified, a default config with threading backend is used.

        Returns:
            A concatenated dataframe of the evaluation results.
        """
        logger.info("Running ExtremeWeatherBench workflow...")

        # Determine if running in serial or parallel mode
        # Serial: n_jobs=1 or (parallel_config with n_jobs=1)
        # Parallel: n_jobs>1 or (parallel_config with n_jobs>1)
        is_serial = (
            (n_jobs == 1)
            or (parallel_config is not None and parallel_config.get("n_jobs") == 1)
            or (n_jobs is None and parallel_config is None)
        )
        logger.debug("Running in %s mode.", "serial" if is_serial else "parallel")

        if not is_serial:
            # Build parallel_config if not provided
            if parallel_config is None and n_jobs is not None:
                logger.debug(
                    "No parallel_config provided, using threading backend and %s jobs.",
                    n_jobs,
                )
                parallel_config = {"backend": "threading", "n_jobs": n_jobs}
            kwargs["parallel_config"] = parallel_config

            # Caching does not work in parallel mode as of now
            if self.cache_dir:
                logger.warning(
                    "Caching is not supported in parallel mode, ignoring cache_dir"
                )
        else:
            # Running in serial mode - instantiate cache dir if needed
            if self.cache_dir:
                if not self.cache_dir.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

        run_results = _run_case_operators(self.case_operators, self.cache_dir, **kwargs)

        # If there are results, concatenate them and return, else return an empty
        # DataFrame with the expected columns
        if run_results:
            return _safe_concat(run_results, ignore_index=True)
        else:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _run_case_operators(
    case_operators: list["cases.CaseOperator"],
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in parallel or serial.

    Args:
        case_operators: List of case operators to run.
        cache_dir: Optional directory for caching (serial mode only).
        **kwargs: Additional arguments, may include 'parallel_config' dict.

    Returns:
        List of result DataFrames.
    """
    with logging_redirect_tqdm():
        # Check if parallel_config is provided
        parallel_config = kwargs.get("parallel_config", None)

        # Run in parallel if parallel_config exists and n_jobs != 1
        if parallel_config is not None:
            logger.info("Running case operators in parallel...")
            return _run_parallel(case_operators, **kwargs)
        else:
            logger.info("Running case operators in serial...")
            return _run_serial(case_operators, cache_dir, **kwargs)


def _run_serial(
    case_operators: list["cases.CaseOperator"],
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in serial."""
    run_results = []

    # Loop over the case operators
    for case_operator in tqdm(case_operators):
        run_results.append(compute_case_operator(case_operator, cache_dir, **kwargs))
    return run_results


def _run_parallel(
    case_operators: list["cases.CaseOperator"],
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in parallel.

    Args:
        case_operators: List of case operators to run.
        **kwargs: Additional arguments, must include 'parallel_config' dict.

    Returns:
        List of result DataFrames.
    """
    parallel_config = kwargs.pop("parallel_config", None)

    if parallel_config is None:
        raise ValueError("parallel_config must be provided to _run_parallel")

    if parallel_config.get("n_jobs") is None:
        logger.warning("No number of jobs provided, using joblib backend default.")

    # Handle dask backend - create client if needed
    dask_client = None
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

    try:
        # TODO(198): return a generator and compute at a higher level
        with joblib.parallel_config(**parallel_config):
            run_results = utils.ParallelTqdm(total_tasks=len(case_operators))(
                # None is the cache_dir, we can't cache in parallel mode
                joblib.delayed(compute_case_operator)(case_operator, None, **kwargs)
                for case_operator in case_operators
            )
        return run_results
    finally:
        # Clean up the dask client if we created it
        if dask_client is not None:
            logger.info("Closing dask client")
            dask_client.close()


def compute_case_operator(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute the resulting evaluation of a case operator.

    This method will compute the results of a case operator. It validates
    that all metrics are properly instantiated, builds the target and forecast
    datasets, aligns them, and computes each metric with appropriate variable
    pairs. Metrics with their own forecast_variable and target_variable use
    only those variables; metrics without will use all InputBase variable pairs.

    Args:
        case_operator: The case operator to compute the results of.
        cache_dir: The directory to cache mid-flight outputs (serial mode).
        kwargs: Keyword arguments to pass to the metric computations.

    Returns:
        A pd.DataFrame of results from the case operator.

    Raises:
        TypeError: If any metric is not properly instantiated (i.e. isn't an instance
        or child class of BaseMetric).
    """
    # Validate that all metrics are instantiated (not classes or callables)
    for i, metric in enumerate(case_operator.metric_list):
        if isinstance(metric, type):
            case_operator.metric_list[i] = metric()
            logger.warning(
                "Metric %s instantiated with default parameters",
                case_operator.metric_list[i].name,
            )
        if not isinstance(case_operator.metric_list[i], metrics.BaseMetric):
            raise TypeError(f"Metric must be a BaseMetric instance, got {type(metric)}")

    forecast_ds, target_ds = _build_datasets(case_operator, **kwargs)

    # Check if any dimension has zero length
    if 0 in forecast_ds.sizes.values() or 0 in target_ds.sizes.values():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Or, check if there aren't any dimensions
    elif len(forecast_ds.sizes) == 0 or len(target_ds.sizes) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # spatiotemporally align the target and forecast datasets dependent on the target
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Compute and cache the datasets if requested
    if kwargs.get("pre_compute", False):
        aligned_forecast_ds, aligned_target_ds = _compute_and_maybe_cache(
            aligned_forecast_ds,
            aligned_target_ds,
            cache_dir=kwargs.get("cache_dir", None),
        )
    logger.info(
        "Datasets built for case %s.", case_operator.case_metadata.case_id_number
    )
    results = []

    # Collect all explicitly specified variables across all metrics
    # These variables are "claimed" and should not be used by metrics
    # without specific variables
    explicitly_claimed_forecast_vars = set()
    explicitly_claimed_target_vars = set()

    for metric in case_operator.metric_list:
        # Determine which variable pairs to evaluate for this metric
        if (metric.forecast_variable is not None) and (
            metric.target_variable is not None
        ):
            # Expand and collect claimed variables
            explicitly_claimed_forecast_vars.update(
                _maybe_expand_derived_variable_to_output_variables(
                    metric.forecast_variable
                )
            )
            explicitly_claimed_target_vars.update(
                _maybe_expand_derived_variable_to_output_variables(
                    metric.target_variable
                )
            )

    for metric in case_operator.metric_list:
        # Expand composite metrics into individual metrics
        # (returns [self] for non-composite metrics)
        metrics_to_evaluate = metric.maybe_expand_composite()

        # Determine which variable pairs to evaluate for this metric
        if metric.forecast_variable is not None and metric.target_variable is not None:
            # Expand DerivedVariable to output_variables if applicable
            forecast_vars = _maybe_expand_derived_variable_to_output_variables(
                metric.forecast_variable
            )
            target_vars = _maybe_expand_derived_variable_to_output_variables(
                metric.target_variable
            )

            # Create pairs from expanded variables
            variable_pairs = list(zip(forecast_vars, target_vars))
        else:
            # Use all InputBase variable pairs for this metric
            # Expand any DerivedVariables in the InputBase variables
            forecast_vars_expanded = []
            for var in case_operator.forecast.variables:
                forecast_vars_expanded.extend(
                    _maybe_expand_derived_variable_to_output_variables(var)
                )

            target_vars_expanded = []
            for var in case_operator.target.variables:
                target_vars_expanded.extend(
                    _maybe_expand_derived_variable_to_output_variables(var)
                )

            # Exclude variables that are explicitly claimed by other metrics
            forecast_vars_available = [
                v
                for v in forecast_vars_expanded
                if v not in explicitly_claimed_forecast_vars
            ]
            target_vars_available = [
                v
                for v in target_vars_expanded
                if v not in explicitly_claimed_target_vars
            ]

            variable_pairs = list(zip(forecast_vars_available, target_vars_available))

        # Evaluate the metric(s) for each variable pair
        for forecast_var, target_var in variable_pairs:
            # Prepare kwargs for metric evaluation (handles composite setup)
            forecast_var_str = derived._maybe_convert_variable_to_string(forecast_var)
            target_var_str = derived._maybe_convert_variable_to_string(target_var)

            metric_kwargs = metric.maybe_prepare_composite_kwargs(
                forecast_data=aligned_forecast_ds[forecast_var_str],
                target_data=aligned_target_ds[target_var_str],
                **kwargs,
            )

            # Evaluate each expanded metric
            for single_metric in metrics_to_evaluate:
                results.append(
                    _evaluate_metric_and_return_df(
                        forecast_ds=aligned_forecast_ds,
                        target_ds=aligned_target_ds,
                        forecast_variable=forecast_var,
                        target_variable=target_var,
                        metric=single_metric,
                        case_operator=case_operator,
                        **metric_kwargs,
                    )
                )

        # Cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = (
                pathlib.Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            )
            concatenated = _safe_concat(results, ignore_index=True)
            if not concatenated.empty:
                concatenated.to_pickle(cache_path / "results.pkl")

    return _safe_concat(results, ignore_index=True)


def _extract_standard_metadata(
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
) -> dict:
    """Extract standard metadata for output dataframe.

    This function centralizes the logic for extracting metadata from the
    evaluation context. Makes it easy to modify how metadata is extracted
    without changing the schema enforcement logic.

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

    # An output requires one of init_time or lead_time. init_time will be present for a
    # metric that assesses something in an entire model run, such as the onset error of
    # an event. Lead_time will be present for a metric that assesses something at a
    # specific forecast hour, such as RMSE. If neither are present, the output is
    # invalid. Both should not be present for one metric. Thus, one should always be
    # missing, which is intended behavior.
    init_time_missing = "init_time" in missing_cols
    lead_time_missing = "lead_time" in missing_cols

    # Check if exactly one of init_time or lead_time is missing
    if init_time_missing != lead_time_missing:
        missing_cols.discard("init_time")
        missing_cols.discard("lead_time")

    if missing_cols:
        logger.warning("Missing expected columns: %s.", missing_cols)

    # Ensure all OUTPUT_COLUMNS are present (missing ones will be NaN)
    # and reorder to match OUTPUT_COLUMNS specification
    return df.reindex(columns=OUTPUT_COLUMNS)


def _evaluate_metric_and_return_df(
    forecast_ds: xr.Dataset,
    target_ds: xr.Dataset,
    forecast_variable: Union[str, "derived.DerivedVariable"],
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
    **kwargs,
) -> pd.DataFrame:
    """Evaluate a metric and return a dataframe of the results.

    Args:
        forecast_ds: The forecast dataset.
        target_ds: The target dataset.
        forecast_variable: The forecast variable to evaluate.
        target_variable: The target variable to evaluate.
        metric: The metric to evaluate.
        case_operator: The case operator with metadata for evaluation.
        **kwargs: Additional keyword arguments to pass to metric
            computation.

    Returns:
        A dataframe of the results with standard output schema
        columns.
    """

    # Normalize variables to their string names if needed
    forecast_variable = derived._maybe_convert_variable_to_string(forecast_variable)
    target_variable = derived._maybe_convert_variable_to_string(target_variable)

    logger.info("Computing metric %s... ", metric.name)

    # Extract the appropriate data for the metric
    # Variables should already be present at this point in the pipeline
    if forecast_variable not in forecast_ds.data_vars:
        raise ValueError(
            f"Variable '{forecast_variable}' not found in forecast dataset. "
            f"Available variables: {list(forecast_ds.data_vars)}"
        )

    if target_variable not in target_ds.data_vars:
        raise ValueError(
            f"Variable '{target_variable}' not found in target dataset. "
            f"Available variables: {list(target_ds.data_vars)}"
        )

    forecast_data = forecast_ds[forecast_variable]
    target_data = target_ds[target_variable]

    metric_result = metric.compute_metric(
        forecast_data,
        target_data,
        **kwargs,
    )
    # If data is sparse, densify it
    if isinstance(metric_result.data, dask.array.Array) and isinstance(
        metric_result.data._meta, sparse.COO
    ):
        metric_result.data = metric_result.data.compute().maybe_densify()
    # Convert to DataFrame and add metadata, ensuring OUTPUT_COLUMNS compliance
    df = metric_result.to_dataframe(name="value").reset_index()
    # TODO: add functionality for custom metadata columns
    metadata = _extract_standard_metadata(target_variable, metric, case_operator)
    return _ensure_output_schema(df, **metadata)


def _maybe_expand_derived_variable_to_output_variables(
    variable: Union[str, "derived.DerivedVariable"],
) -> list[str]:
    """Expand a variable to its output_variables if it's a DerivedVariable.

    Args:
        variable: Either a string variable name or a DerivedVariable
            instance.

    Returns:
        List of variable names. For strings, returns [variable]. For
        DerivedVariable with output_variables, returns those. For
        DerivedVariable without output_variables, returns [variable.name].
    """
    if isinstance(variable, str):
        return [variable]
    elif isinstance(variable, derived.DerivedVariable):
        if hasattr(variable, "output_variables") and variable.output_variables:
            return variable.output_variables
        else:
            # DerivedVariable without output_variables, use name
            return [str(variable.name)]
    else:
        # Fallback to string conversion, this should never happen
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
    metric_list: list["metrics.BaseMetric"],
) -> tuple[
    set[Union[str, "derived.DerivedVariable"]],
    set[Union[str, "derived.DerivedVariable"]],
]:
    """Collect unique variables from metrics that have them defined.

    When a metric has a DerivedVariable with output_variables defined,
    the DerivedVariable instance is added to ensure it gets computed
    during pipeline execution.

    Args:
        metric_list: List of metrics to extract variables from.

    Returns:
        Tuple of (forecast_variables, target_variables) as sets.
    """
    forecast_vars = set()
    target_vars = set()

    for metric in metric_list:
        # Check if metric has variables defined (not None)
        if metric.forecast_variable is not None:
            forecast_vars.add(metric.forecast_variable)
        if metric.target_variable is not None:
            target_vars.add(metric.target_variable)

    return forecast_vars, target_vars


def _build_datasets(
    case_operator: "cases.CaseOperator",
    **kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target and
    forecast datasets, including preprocessing, variable renaming, and subsetting.
    It augments the InputBase variables with any variables defined in metrics to
    ensure all required variables are loaded and derived.

    If any forecast variable has `requires_target_dataset=True`, the target dataset
    will be passed to the forecast pipeline via `_target_dataset` in kwargs. This
    allows derived variables to automatically access target/reference data when needed.

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
    # These should NOT be added separately as they'll be created by derivation
    forecast_derived_outputs = _get_all_derived_output_variables(
        case_operator.forecast.variables
    )
    target_derived_outputs = _get_all_derived_output_variables(
        case_operator.target.variables
    )

    # Filter out string variables that are output_variables of existing DerivedVariables
    # Only add metric variables that are not already covered by DerivedVariable outputs
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
        target_ds = run_pipeline(
            case_operator.case_metadata, augmented_target, **kwargs
        )

    # Pass target dataset to forecast pipeline only if needed
    # Check if any forecast variable requires target dataset
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
        forecast_ds = run_pipeline(
            case_operator.case_metadata, augmented_forecast, **kwargs
        )

    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        if "valid_time" in zero_length_dims:
            logger.warning(
                "Forecast dataset for case %s has no data for case time range %s to %s."
                % (
                    case_operator.case_metadata.case_id_number,
                    case_operator.case_metadata.start_date,
                    case_operator.case_metadata.end_date,
                )
            )
        else:
            logger.warning(
                "Forecast dataset for case %s has zero-length dimensions %s for case "
                "time range %s to %s."
                % (
                    case_operator.case_metadata.case_id_number,
                    zero_length_dims,
                    case_operator.case_metadata.start_date,
                    case_operator.case_metadata.end_date,
                )
            )
        return xr.Dataset(), xr.Dataset()
    return (forecast_ds, target_ds)


def _compute_and_maybe_cache(
    *datasets: xr.Dataset, cache_dir: Optional[Union[str, pathlib.Path]]
) -> list[xr.Dataset]:
    """Compute and cache the datasets if caching."""
    logger.info("Computing datasets...")
    computed_datasets = [dataset.compute() for dataset in datasets]
    if cache_dir:
        raise NotImplementedError("Caching is not implemented yet")
    return computed_datasets


def run_pipeline(
    case_metadata: "cases.IndividualCase",
    input_data: "inputs.InputBase",
    **kwargs,
) -> xr.Dataset:
    """Shared method for running an input pipeline.

    Args:
        case_metadata: The case metadata to run the pipeline on.
        input_data: The input data to run the pipeline on.
        **kwargs: Additional keyword arguments to pass to pipeline steps.

    Returns:
        The processed input data as an xarray dataset.
    """
    # Open data and process through pipeline steps
    data = input_data.open_and_maybe_preprocess_data_from_source().pipe(
        lambda ds: input_data.maybe_map_variable_names(ds)
    )

    # Get the appropriate source module for the data type
    source_module = sources.get_backend_module(type(data))

    # Checks if the data has valid times and spatial overlap. This must come after
    # maybe_map_variable_names to ensure variable names are mapped correctly.
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
            "Forecast dataset for case %s has no data for case time range %s to %s."
            % (
                case_metadata.case_id_number,
                case_metadata.start_date.strftime("%Y-%m-%d %H:%M:%S"),
                case_metadata.end_date.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        return xr.Dataset()


def _safe_concat(
    dataframes: list[pd.DataFrame], ignore_index: bool = True
) -> pd.DataFrame:
    """Safely concatenate DataFrames, filtering out empty ones.

    This function prevents FutureWarnings from pd.concat when dealing with
    empty or all-NA DataFrames by filtering them out before concatenation.
    It also handles dtype mismatches by converting to object dtype only when
    necessary to prevent concatenation warnings.

    Args:
        dataframes: List of DataFrames to concatenate
        ignore_index: Whether to ignore index during concatenation

    Returns:
        Concatenated DataFrame, or empty DataFrame with OUTPUT_COLUMNS if all
        input DataFrames are empty. Preserves original dtypes when consistent
        across DataFrames, converts to object dtype only when there are
        dtype mismatches.
    """
    # Filter out problematic DataFrames that would trigger FutureWarning
    valid_dfs = []
    for i, df in enumerate(dataframes):
        # Skip empty DataFrames
        if df.empty:
            logger.debug("Skipping empty DataFrame %s", i)
            continue
        # Skip DataFrames where all values are NA
        if df.isna().all().all():
            logger.debug("Skipping all-NA DataFrame %s", i)
            continue
        # Skip DataFrames where all columns are empty/NA
        if len(df.columns) > 0 and all(df[col].isna().all() for col in df.columns):
            logger.debug("Skipping DataFrame %s with all-NA columns", i)
            continue

        valid_dfs.append(df)

    if valid_dfs:
        # Check for dtype inconsistencies that cause FutureWarning
        if len(valid_dfs) > 1:
            # Check if there are dtype mismatches between DataFrames
            reference_df = valid_dfs[0]
            has_dtype_mismatch = False

            for df in valid_dfs[1:]:
                # Check if columns have different dtypes across DataFrames
                for col in reference_df.columns:
                    if col in df.columns:
                        if reference_df[col].dtype != df[col].dtype:
                            has_dtype_mismatch = True
                            break
                if has_dtype_mismatch:
                    break

            if has_dtype_mismatch:
                # Only convert to object dtype if there are mismatches
                consistent_dfs = [df.astype(object) for df in valid_dfs]
                return pd.concat(consistent_dfs, ignore_index=ignore_index)

        # No dtype mismatches, concatenate normally
        return pd.concat(valid_dfs, ignore_index=ignore_index)
    else:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
