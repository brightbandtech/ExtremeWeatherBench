"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import contextlib
import contextvars
import copy
import dataclasses
import logging
import multiprocessing
import pathlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import dask.array as da
import joblib
import pandas as pd
import sparse
import xarray as xr
from tqdm.contrib.logging import logging_redirect_tqdm

import extremeweatherbench.cases as cases
import extremeweatherbench.derived as derived
import extremeweatherbench.inputs as inputs
import extremeweatherbench.metrics as metrics
import extremeweatherbench.outputs as outputs
import extremeweatherbench.progress as progress_module
import extremeweatherbench.sources as sources
import extremeweatherbench.utils as utils

if TYPE_CHECKING:
    import extremeweatherbench.regions as regions

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility.
OUTPUT_COLUMNS = outputs.OUTPUT_COLUMNS

# Process-local reuse of forecast/target datasets within one evaluation.
_pipeline_cache_var: contextvars.ContextVar[Optional[dict[tuple, xr.Dataset]]] = (
    contextvars.ContextVar("_pipeline_cache", default=None)
)


class ExtremeWeatherBench:
    """A class to build and run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is ultimately a
    wrapper around case operators and evaluation objects to create a parallel or
    serial run to evaluate cases and metrics, returning a concatenated dataframe of the
    results.

    Attributes:
        case_metadata: A list of case dicts or IndividualCase objects to run.
        evaluation_objects: A list of evaluation objects to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the
            workflow for serial runs.
        region_subsetter: An optional region subsetter to subset the cases that are
            part of the evaluation to a Region object or a dictionary of lat/lon
            bounds.
    """

    def __init__(
        self,
        case_metadata: Union[list[dict[str, Any]], "list[cases.IndividualCase]"],
        evaluation_objects: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        region_subsetter: Optional["regions.RegionSubsetter"] = None,
    ):
        """Initialize the ExtremeWeatherBench workflow.

        Args:
            case_metadata: List of case dicts or IndividualCase objects.
            evaluation_objects: List of evaluation objects to run.
            cache_dir: Optional directory for caching mid-flight outputs in
                serial runs.
            region_subsetter: Optional RegionSubsetter to filter cases by
                spatial region.
        """
        # Load the case metadata from the input
        self.case_metadata = cases.load_individual_cases(case_metadata)
        self.evaluation_objects = evaluation_objects
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        # Instantiate cache dir if needed
        if self.cache_dir:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.region_subsetter = region_subsetter

    # Case operators as a property can be used as a convenience method for a workflow
    # independent of the class.
    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        """Build the CaseOperator objects from case_metadata and evaluation_objects."""

        # Subset the cases if a region subsetter was provided
        if self.region_subsetter:
            subset_list = self.region_subsetter.subset_case_list(self.case_metadata)
        else:
            subset_list = self.case_metadata
        return cases.build_case_operators(subset_list, self.evaluation_objects)

    def run(
        self,
        n_jobs: Optional[int] = None,
        parallel_config: Optional[dict] = None,
        progress: bool = True,
        output_format: str = "pandas",
        sparse: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """Deprecated alias for :meth:`run_evaluation`."""
        logger.warning("The run method is deprecated. Use run_evaluation instead.")
        return self.run_evaluation(
            n_jobs=n_jobs,
            parallel_config=parallel_config,
            progress=progress,
            output_format=output_format,
            sparse=sparse,
            **kwargs,
        )

    def run_evaluation(
        self,
        n_jobs: Optional[int] = None,
        parallel_config: Optional[dict] = None,
        progress: bool = True,
        output_format: str = "pandas",
        sparse: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """Runs the ExtremeWeatherBench evaluation workflow.

        This method will run the evaluation workflow in the order of the case operators,
        optionally caching the mid-flight outputs of the workflow if cache_dir was
        provided for serial runs.

        Args:
            n_jobs: The number of jobs to run in parallel. If None, defaults to the
                joblib backend default value. If 1, the workflow will run serially.
                Ignored if parallel_config is provided.
            parallel_config: Optional dictionary of joblib parallel configuration.
                If provided, this takes precedence over n_jobs. If not provided and
                n_jobs is specified, a default config with the loky backend is used.
            progress: Whether to display progress bars. Defaults to True.
            output_format: "pandas" for the long-form DataFrame, or "xarray"
                for the flat Dataset from outputs.results_to_dataset.
            sparse: Forwarded to outputs.results_to_dataset when
                output_format is "xarray".
            **kwargs: Additional arguments to pass to compute_case_operator.
        Returns:
            The evaluation results in the requested output_format.

        Raises:
            ValueError: If output_format is not "pandas" or "xarray".
        """
        _validate_output_format(output_format)
        logger.info("Running ExtremeWeatherBench evaluations...")

        # Check for serial or parallel configuration
        parallel_config = _parallel_serial_config_check(n_jobs, parallel_config)

        run_results = _run_evaluation(
            self.case_operators,
            cache_dir=self.cache_dir,
            parallel_config=parallel_config,
            progress=progress,
            **kwargs,
        )

        flattened_results = [
            result for case_results in run_results for result in case_results
        ]
        return _convert_results(flattened_results, output_format, sparse=sparse)


def _validate_output_format(output_format: str) -> None:
    """Raise ValueError unless output_format is pandas or xarray."""
    if output_format not in ("pandas", "xarray"):
        raise ValueError(
            f"Unknown output_format '{output_format}'. Expected 'pandas' or "
            "'xarray'."
        )


def _convert_results(
    results: list[xr.DataArray],
    output_format: str,
    sparse: bool = False,
) -> Union[pd.DataFrame, xr.Dataset]:
    """Convert a flat list of annotated metric results to output_format.

    Args:
        results: A flat list of annotated metric results.
        output_format: "pandas" for the long-form DataFrame, or "xarray"
            for the flat Dataset from outputs.results_to_dataset.
        sparse: Forwarded to outputs.results_to_dataset when output_format
            is "xarray".

    Returns:
        The results converted to the requested output_format.

    Raises:
        ValueError: If output_format is not "pandas" or "xarray".
    """
    _validate_output_format(output_format)
    if output_format == "pandas":
        return outputs.results_to_dataframe(results)
    return outputs.results_to_dataset(results, sparse=sparse)


def _parallel_serial_config_check(
    n_jobs: Optional[int] = None,
    parallel_config: Optional[dict] = None,
) -> Optional[dict]:
    """Check if running in serial or parallel mode.

    Args:
        n_jobs: The number of jobs to run in parallel. If None, defaults to the
            joblib backend default value. If 1, the workflow will run serially.
        parallel_config: Optional dictionary of joblib parallel configuration. If
            provided, this takes precedence over n_jobs. If not provided and n_jobs is
            specified, a default config with loky backend is used.
    Returns:
        None if running in serial mode, otherwise a dictionary of joblib parallel
        configuration.
    """
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
                "No parallel_config provided, using loky backend and %s jobs.",
                n_jobs,
            )
            parallel_config = {"backend": "loky", "n_jobs": n_jobs}
    # If running in serial mode, set parallel_config to None if not already
    else:
        parallel_config = None
    # Return the maybe updated kwargs
    return parallel_config


def _run_evaluation(
    case_operators: list["cases.CaseOperator"],
    cache_dir: Optional[pathlib.Path] = None,
    parallel_config: Optional[dict] = None,
    progress: bool = True,
    **kwargs,
) -> list[list[xr.DataArray]]:
    """Run the case operators in parallel or serial.

    Args:
        case_operators: List of case operators to run.
        cache_dir: Optional directory for caching (serial mode only).
        parallel_config: Optional dict of joblib parallel configuration.
        progress: Whether to display progress bars. Defaults to True.
        **kwargs: Additional keyword arguments passed to case operators.

    Returns:
        List of per-case-operator lists of annotated metric results.
    """
    with progress_module.captured_warnings(), logging_redirect_tqdm():
        if parallel_config is not None:
            logger.info("Running case operators in parallel...")
            run_results = _run_parallel_evaluation(
                case_operators,
                cache_dir=cache_dir,
                parallel_config=parallel_config,
                progress=progress,
                **kwargs,
            )
        else:
            logger.info("Running case operators in serial...")
            run_results = []
            cache_token = _pipeline_cache_var.set({})
            case_bar = progress_module.make_case_bar(
                len(case_operators), disable=not progress
            )
            task_bar = progress_module.make_dask_task_bar(disable=not progress)
            try:
                with progress_module.registered_bar(
                    case_bar, allow_phase_updates=True
                ):
                    with progress_module.DaskTaskBar(task_bar):
                        for case_operator in case_operators:
                            step_bar = progress_module.make_case_step_bar(
                                case_operator.case_metadata.case_id_number,
                                _count_case_steps(case_operator, cache_dir),
                                disable=not progress,
                            )
                            progress_module.register_step_bar(step_bar)
                            try:
                                run_results.append(
                                    _compute_case_operator_results(
                                        case_operator, cache_dir, **kwargs
                                    )
                                )
                            finally:
                                progress_module.register_step_bar(None)
                                step_bar.close()
                            case_bar.update(1)
            finally:
                task_bar.close()
                case_bar.close()
                _pipeline_cache_var.reset(cache_token)

    return run_results


def _run_parallel_evaluation(
    case_operators: list["cases.CaseOperator"],
    parallel_config: dict,
    cache_dir: Optional[pathlib.Path] = None,
    progress: bool = True,
    **kwargs,
) -> list[list[xr.DataArray]]:
    """Run the case operators in parallel.

    Args:
        case_operators: List of case operators to run.
        parallel_config: Joblib parallel configuration dict.
        cache_dir: Optional directory for caching (unused in parallel mode).
        progress: Whether to display progress bars. Defaults to True.
        **kwargs: Additional arguments, must include 'parallel_config' dict.

    Returns:
        List of per-case-operator lists of annotated metric results.
    """
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

    parallel_tqdm_kwargs: dict[str, Any] = {"total_tasks": len(case_operators)}
    if not progress:
        parallel_tqdm_kwargs["disable_progressbar"] = True

    manager = None
    event_queue = None
    renderer = None
    log_queue = None
    log_listener = None
    if progress and progress_module.supports_nested_bars():
        n_jobs = parallel_config.get("n_jobs")
        # n_jobs may be negative (all-but-N cores) or None (backend
        # default), so resolve it to a real, bounded slot count.
        n_slots = joblib.effective_n_jobs(n_jobs) if n_jobs else 1
        manager = multiprocessing.Manager()
        event_queue = manager.Queue()
        renderer = progress_module.WorkerSlotRenderer(n_slots=n_slots)
        renderer.start(event_queue)
        # A separate queue from event_queue: log records and progress
        # events have different payloads and drain logic.
        log_queue = manager.Queue()
        log_listener = progress_module.LogQueueListener()
        log_listener.start(log_queue)

    def _close_worker_progress() -> None:
        # Flush trailing log records before the bars they'd otherwise
        # tear through close, then close the slot bars themselves.
        # Passed to ParallelTqdm as pre_close so it runs before the
        # case bar closes; also called as a safety net below in case
        # ParallelTqdm's own finally never ran (idempotent either way).
        if log_listener is not None:
            log_listener.close()
        if renderer is not None:
            renderer.close()

    try:
        # TODO(198): return a generator and compute at a higher level
        # Group operators that share a case and forecast so one worker can
        # reuse the forecast dataset (PPH + LSR on the same HRES object).
        groups = _group_operators_sharing_forecast(case_operators)
        parallel_tqdm_kwargs["total_tasks"] = len(groups)
        with joblib.parallel_config(**parallel_config):
            nested_results = utils.ParallelTqdm(
                pre_close=_close_worker_progress, **parallel_tqdm_kwargs
            )(
                joblib.delayed(_compute_operator_group_with_progress)(
                    group,
                    cache_dir=cache_dir,
                    event_queue=event_queue,
                    log_queue=log_queue,
                    **kwargs,
                )
                for group in groups
            )
        return _scatter_group_results(
            groups, nested_results, len(case_operators)
        )
    finally:
        _close_worker_progress()
        if manager is not None:
            manager.shutdown()
        # Clean up the dask client if we created it
        if dask_client is not None:
            logger.info("Closing dask client")
            dask_client.close()


@dataclasses.dataclass(frozen=True)
class IndexedOperator:
    """An operator tagged with its original input index."""

    index: int
    operator: "cases.CaseOperator"


def _group_operators_sharing_forecast(
    case_operators: list["cases.CaseOperator"],
) -> list[list[IndexedOperator]]:
    """Group operators that share a case and forecast source.

    PPH and LSR for the same case can reuse one forecast pipeline this way.
    """
    groups: OrderedDict[tuple, list[IndexedOperator]] = OrderedDict()
    for i, op in enumerate(case_operators):
        key = (
            op.case_metadata.case_id_number,
            op.forecast.name,
            op.forecast.source,
        )
        groups.setdefault(key, []).append(
            IndexedOperator(index=i, operator=op)
        )
    return list(groups.values())


def _scatter_group_results(
    groups: list[list[IndexedOperator]],
    nested_results: list[list[list[xr.DataArray]]],
    n_operators: int,
) -> list[list[xr.DataArray]]:
    """Restore original operator order from grouped parallel results.

    Each job returns per-operator results in group order. Indices stay
    on the parent in ``groups``.
    """
    run_results: list[list[xr.DataArray]] = [[] for _ in range(n_operators)]
    for group, group_results in zip(groups, nested_results, strict=True):
        for indexed, res in zip(group, group_results, strict=True):
            run_results[indexed.index] = res
    return run_results


def _variable_cache_token(var: Any) -> str:
    """Stable cache token for a pipeline variable."""
    if isinstance(var, str):
        return var
    return f"{type(var).__name__}:{getattr(var, 'name', '')}"


def _pipeline_cache_key(
    case_metadata: "cases.IndividualCase",
    input_data: "inputs.InputBase",
) -> tuple:
    """Key for reusing a pipeline dataset within one evaluation run."""
    var_key = tuple(sorted(_variable_cache_token(v) for v in input_data.variables))
    return (
        case_metadata.case_id_number,
        type(input_data).__name__,
        input_data.name,
        input_data.source,
        var_key,
    )


def _run_pipeline_maybe_cached(
    case_metadata: "cases.IndividualCase",
    input_data: "inputs.InputBase",
    pipeline_cache: Optional[dict[tuple, xr.Dataset]],
    extra_key: tuple = (),
    **kwargs,
) -> xr.Dataset:
    """Run an input pipeline, reusing a dataset when the cache hits."""
    if pipeline_cache is None:
        return run_pipeline(case_metadata, input_data, **kwargs)
    key = _pipeline_cache_key(case_metadata, input_data) + extra_key
    cached = pipeline_cache.get(key)
    if cached is not None:
        logger.debug(
            "Reusing cached %s dataset for case %s",
            input_data.name,
            case_metadata.case_id_number,
        )
        return cached
    ds = run_pipeline(case_metadata, input_data, **kwargs)
    pipeline_cache[key] = ds
    return ds


def _compute_operator_group_with_progress(
    indexed_ops: list[IndexedOperator],
    cache_dir: Optional[pathlib.Path] = None,
    event_queue=None,
    log_queue=None,
    **kwargs,
) -> list[list[xr.DataArray]]:
    """Run operators that share a forecast in one worker.

    A process-local pipeline cache lets the second operator skip a second
    forecast derive (for example CBSS for PPH then LSR). Results stay in
    group order; the parent scatters them with ``groups``.
    """
    kwargs = dict(kwargs)
    token = _pipeline_cache_var.set({})
    try:
        return [
            _compute_case_operator_with_progress(
                indexed.operator,
                cache_dir=cache_dir,
                event_queue=event_queue,
                log_queue=log_queue,
                dispatch_id=indexed.index,
                **kwargs,
            )
            for indexed in indexed_ops
        ]
    finally:
        _pipeline_cache_var.reset(token)


def _plan_metric_evaluations(
    case_operator: "cases.CaseOperator",
) -> list[tuple["metrics.BaseMetric", Sequence["metrics.BaseMetric"], list[tuple]]]:
    """Enumerate the metric evaluations a case operator will perform.

    Depends only on metric and input metadata, never on the data, so the
    result is available before any pipeline runs and can size a progress
    bar.

    Args:
        case_operator: The case operator to plan evaluations for.

    Returns:
        One (metric, expanded_metrics, variable_pairs) tuple per metric,
        in the same order the evaluation loop will consume them.
    """
    explicitly_claimed_forecast_vars = set()
    explicitly_claimed_target_vars = set()
    for metric in case_operator.metric_list:
        if (metric.forecast_variable is not None) and (
            metric.target_variable is not None
        ):
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

    plan = []
    for metric in case_operator.metric_list:
        metrics_to_evaluate = metric.maybe_expand_composite()
        if metric.forecast_variable is not None and metric.target_variable is not None:
            forecast_vars = _maybe_expand_derived_variable_to_output_variables(
                metric.forecast_variable
            )
            target_vars = _maybe_expand_derived_variable_to_output_variables(
                metric.target_variable
            )
            variable_pairs = list(zip(forecast_vars, target_vars))
        else:
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
        plan.append((metric, metrics_to_evaluate, variable_pairs))
    return plan


def _count_metric_evaluations(case_operator: "cases.CaseOperator") -> int:
    """Count the metric evaluations a case operator will perform.

    Args:
        case_operator: The case operator to count evaluations for.

    Returns:
        The number of individual metric evaluations.
    """
    return sum(
        len(expanded) * len(pairs)
        for _, expanded, pairs in _plan_metric_evaluations(case_operator)
    )


def _count_case_steps(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
) -> int:
    """Count the coarse steps a case will report progress for.

    Two pipeline runs, two cache writes when caching is on, and one step
    per metric evaluation. Mirrors the set_phase call sites.

    Args:
        case_operator: The case operator about to be computed.
        cache_dir: The cache directory, if caching is enabled.

    Returns:
        The total number of steps.
    """
    cache_steps = 2 if cache_dir is not None else 0
    return 2 + cache_steps + _count_metric_evaluations(case_operator)


def compute_case_operator(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
    output_format: str = "pandas",
    **kwargs,
) -> Union[pd.DataFrame, xr.Dataset]:
    """Compute the resulting evaluation of a case operator.

    This method will compute the results of a case operator. It validates
    that all metrics are properly instantiated, builds the target and forecast
    datasets, aligns them, and computes each metric with appropriate variable
    pairs. Metrics with their own forecast_variable and target_variable use
    only those variables; metrics without will use all InputBase variable pairs.

    Args:
        case_operator: The case operator to compute the results of.
        cache_dir: The directory to cache mid-flight outputs (serial mode).
        output_format: "pandas" for the long-form DataFrame, or "xarray"
            for the flat Dataset from outputs.results_to_dataset.

    Returns:
        The case operator's results in the requested output_format.

    Raises:
        TypeError: If any metric is not properly instantiated (i.e. isn't an
            instance or child class of BaseMetric).
        ValueError: If output_format is not "pandas" or "xarray".
    """
    _validate_output_format(output_format)
    results = _compute_case_operator_results(case_operator, cache_dir, **kwargs)
    return _convert_results(results, output_format)


def _compute_case_operator_results(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[xr.DataArray]:
    """Compute the annotated metric results for a case operator.

    This method validates that all metrics are properly instantiated, builds
    the target and forecast datasets, aligns them, and computes each metric
    with appropriate variable pairs. Metrics with their own forecast_variable
    and target_variable use only those variables; metrics without will use
    all InputBase variable pairs.

    Args:
        case_operator: The case operator to compute the results of.
        cache_dir: The directory to cache mid-flight outputs (serial mode).

    Returns:
        A list of annotated metric result DataArrays.

    Raises:
        TypeError: If any metric is not properly instantiated (i.e. isn't an
            instance or child class of BaseMetric).
    """
    # Validate that all metrics are instantiated (not classes or callables)
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
    case_operator = dataclasses.replace(case_operator, metric_list=metric_list)

    forecast_ds, target_ds = _build_datasets(case_operator, **kwargs)

    # Check if any dimension has zero length
    if 0 in forecast_ds.sizes.values() or 0 in target_ds.sizes.values():
        return []

    # Or, check if there aren't any dimensions
    elif len(forecast_ds.sizes) == 0 or len(target_ds.sizes) == 0:
        return []

    # spatiotemporally align the target and forecast datasets dependent on the target
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Compute and cache the datasets if cache_dir is set
    aligned_forecast_ds = utils.maybe_cache_and_compute(
        aligned_forecast_ds,
        cache_dir=cache_dir,
        name=f"{case_operator.case_metadata.case_id_number}_{case_operator.forecast.name}",
    )
    aligned_target_ds = utils.maybe_cache_and_compute(
        aligned_target_ds,
        cache_dir=cache_dir,
        name=f"{case_operator.case_metadata.case_id_number}_{case_operator.target.name}",
    )
    # Compute once so each metric does not rebuild the same dask graph.
    if cache_dir is None:
        aligned_forecast_ds = aligned_forecast_ds.compute()
        aligned_target_ds = aligned_target_ds.compute()
    logger.info(
        "Datasets built for case %s.", case_operator.case_metadata.case_id_number
    )
    results: list[xr.DataArray] = []

    for metric, metrics_to_evaluate, variable_pairs in _plan_metric_evaluations(
        case_operator
    ):
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
                    _evaluate_metric(
                        forecast_ds=aligned_forecast_ds,
                        target_ds=aligned_target_ds,
                        forecast_variable=forecast_var,
                        target_variable=target_var,
                        metric=single_metric,
                        case_operator=case_operator,
                        **metric_kwargs,
                    )
                )

    if cache_dir:
        concatenated = outputs.results_to_dataframe(results)
        if not concatenated.empty:
            concatenated.to_pickle(
                cache_dir
                / f"case_{case_operator.case_metadata.case_id_number}_results.pkl"
            )

    return results


def _compute_case_operator_with_progress(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
    event_queue=None,
    log_queue=None,
    dispatch_id=None,
    **kwargs,
) -> list[xr.DataArray]:
    """Compute a case operator, publishing progress to event_queue.

    Runs inside a worker process. Falls back to plain computation when no
    queue is supplied so serial callers are unaffected.

    Args:
        case_operator: The case operator to compute.
        cache_dir: The directory to cache mid-flight outputs.
        event_queue: Cross-process queue for progress events, or None.
        log_queue: Cross-process queue for log records, or None. When
            set, this process's log records and warnings are forwarded
            there instead of going to this worker's own stderr.
        dispatch_id: Unique key for this dispatch, assigned by the
            caller via enumerate(). build_case_operators can emit
            several CaseOperators sharing one case_id (one case, many
            EvaluationObjects), so case_id alone can't key a slot.
        **kwargs: Additional arguments for compute_case_operator.

    Returns:
        A list of annotated metric result DataArrays.
    """
    if event_queue is None:
        return _compute_case_operator_results(case_operator, cache_dir, **kwargs)

    case_id = case_operator.case_metadata.case_id_number
    slot_key = dispatch_id if dispatch_id is not None else case_id
    target_name = getattr(case_operator.target, "name", None) or "target"
    forecast_name = getattr(case_operator.forecast, "name", None) or "forecast"
    # One case fans out into a dispatch per EvaluationObject, so the
    # case id alone doesn't say which slot is which; the target and
    # forecast names are what actually distinguish them.
    label = f"case {case_id} | {target_name} | {forecast_name}"
    sink = progress_module.QueueSink(event_queue)
    progress_module.register_sink(
        sink,
        case_id=case_id,
        total_steps=_count_case_steps(case_operator, cache_dir),
        slot_key=slot_key,
        label=label,
    )
    log_context = (
        progress_module.forwarding_logs_to(log_queue)
        if log_queue is not None
        else contextlib.nullcontext()
    )
    try:
        with log_context:
            with progress_module.DaskTaskSink(sink, case_id):
                return _compute_case_operator_results(
                    case_operator, cache_dir, **kwargs
                )
    finally:
        sink(
            progress_module.ProgressEvent(
                case_id=case_id, slot_key=slot_key, finished=True
            )
        )
        progress_module.register_sink(None)


def _extract_standard_metadata(
    forecast_variable: Union[str, "derived.DerivedVariable"],
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
) -> dict:
    """Extract standard metadata for an annotated metric result.

    This function centralizes the logic for extracting metadata from the
    evaluation context. Makes it easy to modify how metadata is extracted
    without changing the schema enforcement logic.

    Args:
        forecast_variable: The forecast variable
        target_variable: The target variable
        metric: The metric instance
        case_operator: The CaseOperator holding associated case metadata

    Returns:
        Dictionary of metadata for the metric result
    """
    return {
        "target_variable": target_variable,
        "forecast_variable": forecast_variable,
        "metric": metric.name,
        "target_source": case_operator.target.name,
        "forecast_source": case_operator.forecast.name,
        "case_id_number": case_operator.case_metadata.case_id_number,
        "event_type": case_operator.case_metadata.event_type,
    }


def _evaluate_metric(
    forecast_ds: xr.Dataset,
    target_ds: xr.Dataset,
    forecast_variable: Union[str, "derived.DerivedVariable"],
    target_variable: Union[str, "derived.DerivedVariable"],
    metric: "metrics.BaseMetric",
    case_operator: "cases.CaseOperator",
    **kwargs,
) -> xr.DataArray:
    """Evaluate a metric and return the annotated result.

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
        The metric result with standard metadata attached as coords.
    """

    # Normalize variables to their string names if needed
    forecast_variable = derived._maybe_convert_variable_to_string(forecast_variable)
    target_variable = derived._maybe_convert_variable_to_string(target_variable)

    logger.info(
        "Computing metric %s for case %s... ",
        metric.name,
        case_operator.case_metadata.case_id_number,
    )
    progress_module.set_phase(
        f"case {case_operator.case_metadata.case_id_number} | {metric.name}"
    )

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
        case_metadata=case_operator.case_metadata,
        **kwargs,
    )
    # If data is sparse, densify it
    if isinstance(metric_result.data, sparse.COO):
        metric_result.data = metric_result.data.maybe_densify()
    elif isinstance(metric_result.data, da.Array) and isinstance(
        metric_result.data._meta, sparse.COO
    ):
        # Dask array with sparse.COO chunks - densify chunks
        metric_result.data = metric_result.data.map_blocks(
            lambda x: x.maybe_densify(), dtype=metric_result.data.dtype
        )

    # TODO: add functionality for custom metadata columns
    metadata = _extract_standard_metadata(
        forecast_variable, target_variable, metric, case_operator
    )
    annotated_result = outputs.annotate_metric_result(metric_result, **metadata)
    # Avoid pickling dask graphs back from worker processes.
    if isinstance(annotated_result.data, da.Array):
        annotated_result = annotated_result.compute()
    return annotated_result


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

    logger.info(
        "Running target pipeline for case %s... ",
        case_operator.case_metadata.case_id_number,
    )
    progress_module.set_phase(
        f"case {case_operator.case_metadata.case_id_number} | target pipeline"
    )
    pipeline_cache = _pipeline_cache_var.get()
    target_ds = _run_pipeline_maybe_cached(
        case_operator.case_metadata,
        augmented_target,
        pipeline_cache,
        **kwargs,
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

    logger.info(
        "Running forecast pipeline for case %s... ",
        case_operator.case_metadata.case_id_number,
    )
    progress_module.set_phase(
        f"case {case_operator.case_metadata.case_id_number} | forecast pipeline"
    )
    forecast_extra = (augmented_target.name,) if needs_target else ()
    forecast_ds = _run_pipeline_maybe_cached(
        case_operator.case_metadata,
        augmented_forecast,
        pipeline_cache,
        extra_key=forecast_extra,
        **kwargs,
    )

    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        if "valid_time" in zero_length_dims:
            logger.warning(
                "Forecast dataset %s for case %s has no data for case time range %s to "
                "%s."
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


def run_pipeline(
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
    # Gridded: map names, subset case, preprocess, then subset variables so
    # preprocess can add fields such as geopotential thickness. Tabular
    # sources such as IBTrACS need original column names inside preprocess.
    if isinstance(input_data, inputs.LSR):
        data = input_data._open_data_from_source(case_metadata=case_metadata)
    else:
        data = input_data._open_data_from_source()
    is_gridded = isinstance(data, (xr.Dataset, xr.DataArray))
    if not is_gridded:
        data = input_data.preprocess(data)
    data = input_data.maybe_map_variable_names(data)

    # Get the appropriate source module for the data type
    source_module = sources.get_backend_module(type(data))

    # Checks if the data has valid times and spatial overlap. This must come after
    # maybe_map_variable_names to ensure variable names are mapped correctly.
    if inputs.check_for_missing_data(
        data,
        case_metadata,
        source_module=source_module,
    ):
        # Gridded: time/space subset and preprocess before variable subset
        # so preprocess can create fields such as geopotential thickness.
        to_subset = data
        if not is_gridded:
            to_subset = inputs.maybe_subset_variables(
                data,
                variables=input_data.variables,
                source_module=source_module,
            )
        valid_data = (
            to_subset.pipe(
                lambda ds: input_data.subset_data_to_case(ds, case_metadata, **kwargs)
            )
            .pipe(input_data.maybe_convert_to_dataset)
            .pipe(input_data.add_source_to_dataset_attrs)
        )
        if is_gridded:
            valid_data = input_data.preprocess(valid_data)
            valid_data = inputs.maybe_subset_variables(
                valid_data,
                variables=input_data.variables,
                source_module=source_module,
            )
        valid_data = derived.maybe_derive_variables(
            valid_data,
            variables=input_data.variables,
            case_metadata=case_metadata,
            **kwargs,
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
