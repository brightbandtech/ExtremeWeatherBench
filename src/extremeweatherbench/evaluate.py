"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import itertools
import logging
import pathlib
from typing import TYPE_CHECKING, Optional, Type, Union

import pandas as pd
import sparse
import xarray as xr
from joblib import delayed
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import cases, derived, inputs, utils
from extremeweatherbench.defaults import OUTPUT_COLUMNS

if TYPE_CHECKING:
    from extremeweatherbench import metrics

logger = logging.getLogger(__name__)


class ExtremeWeatherBench:
    """A class to build and run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is ultimately a
    wrapper around case operators and evaluation objects to create either a parallel or
    serial run to evaluate cases and metrics, returning a concatenated dataframe of the
    results.

    Attributes:
        cases: A dictionary of cases to run.
        evaluation_objects: A list of evaluation objects to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the
            workflow for serial runs.
    """

    def __init__(
        self,
        cases: dict[str, list],
        evaluation_objects: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Initialize the ExtremeWeatherBench class.

        Args:
            cases: A dictionary of cases to run.
            evaluation_objects: A list of evaluation objects to run.
            cache_dir: An optional directory to cache the mid-flight outputs of the
                workflow for serial runs.
        """
        self.cases = cases
        self.evaluation_objects = evaluation_objects
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

    # Case operators as a property can be used as a convenience method for a workflow
    # independent of the class.
    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        return cases.build_case_operators(self.cases, self.evaluation_objects)

    def run(
        self,
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the ExtremeWeatherBench workflow.

        This method will run the workflow in the order of the case operators, optionally
        caching the mid-flight outputs of the workflow if cache_dir was provided for
        serial runs.

        Args:
            n_jobs: The number of jobs to run in parallel. If None, defaults to the
            joblib backend default value. If 1, the workflow will run serially.

        Returns:
            A concatenated dataframe of the evaluation results.
        """
        # Caching does not work in parallel mode as of now, so ignore the cache_dir
        # but raise a warning for the user
        if self.cache_dir and n_jobs != 1:
            logger.warning(
                "Caching is not supported in parallel mode, ignoring cache_dir"
            )
        # Instantiate the cache directory if caching and build it if it does not exist
        elif self.cache_dir:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        run_results = _run_case_operators(
            self.case_operators, n_jobs, self.cache_dir, **kwargs
        )
        if run_results:
            return utils._safe_concat(run_results, ignore_index=True)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _run_case_operators(
    case_operators: list["cases.CaseOperator"],
    n_jobs: Optional[int] = None,
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in parallel or serial."""
    with logging_redirect_tqdm():
        if n_jobs != 1:
            return _run_parallel(case_operators, n_jobs, **kwargs)
        else:
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
    n_jobs: Optional[int] = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Run the case operators in parallel."""

    if n_jobs is None:
        logger.warning("No number of jobs provided, using joblib backend default.")
    run_results = utils.ParallelTqdm(
        n_jobs=n_jobs, total_tasks=len(case_operators), desc="Case Operators"
    )(
        # None is the cache_dir, we can't cache in parallel mode
        delayed(compute_case_operator)(case_operator, None, **kwargs)
        for case_operator in case_operators
    )
    return run_results


def compute_case_operator(
    case_operator: "cases.CaseOperator",
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute the results of a case operator.

    This method will compute the results of a case operator. It will build
    the target and forecast datasets,
    align them, compute the metrics, and return a concatenated dataframe of the results.

    Args:
        case_operator: The case operator to compute the results of.
        cache_dir: The directory to cache the mid-flight outputs of the workflow if
        in serial mode.
        kwargs: Keyword arguments to pass to the metric computations.

    Returns:
        A concatenated dataframe of the results of the case operator.
    """
    forecast_ds, target_ds = _build_datasets(case_operator)
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
    # TODO: determine if derived variables need to be pushed here or at pre-compute
    for variables, metric in itertools.product(
        zip(
            case_operator.forecast.variables,
            case_operator.target.variables,
        ),
        case_operator.metric_list,
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

        # Cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = (
                pathlib.Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            )
            concatenated = utils._safe_concat(results, ignore_index=True)
            if not concatenated.empty:
                concatenated.to_pickle(cache_path / "results.pkl")

    return utils._safe_concat(results, ignore_index=True)


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
        "target_source": target_ds.attrs["source"],
        "forecast_source": forecast_ds.attrs["source"],
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
    forecast_variable: Union[str, Type["derived.DerivedVariable"]],
    target_variable: Union[str, Type["derived.DerivedVariable"]],
    metric: "metrics.BaseMetric",
    case_id_number: int,
    event_type: str,
    **kwargs,
) -> pd.DataFrame:
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

    # Normalize variables to their string names if needed
    forecast_variable = _maybe_convert_variable_to_string(forecast_variable)
    target_variable = _maybe_convert_variable_to_string(target_variable)

    # TODO: remove this once we have a better way to handle metric
    # instantiation
    if isinstance(metric, type):
        metric = metric()
    logger.info("Computing metric %s... ", metric.name)
    metric_result = metric.compute_metric(
        forecast_ds.get(forecast_variable, forecast_ds.data_vars),
        target_ds.get(target_variable, target_ds.data_vars),
        **kwargs,
    )
    # If data is sparse, densify it
    if isinstance(metric_result.data, sparse.COO):
        metric_result.data = metric_result.data.maybe_densify()
    # Convert to DataFrame and add metadata, ensuring OUTPUT_COLUMNS compliance
    df = metric_result.to_dataframe(name="value").reset_index()
    # TODO: add functionality for custom metadata columns
    metadata = _extract_standard_metadata(
        target_variable, metric, target_ds, forecast_ds, case_id_number, event_type
    )
    return _ensure_output_schema(df, **metadata)


def _maybe_convert_variable_to_string(
    variable: Union[str, Type["derived.DerivedVariable"]],
) -> str:
    """Convert a variable to its string representation."""
    if derived.is_derived_variable(variable):
        return variable.name  # type: ignore
    else:
        return variable  # type: ignore


def _build_datasets(
    case_operator: "cases.CaseOperator",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target and
    forecast datasets, including preprocessing, variable renaming, and subsetting.
    """
    logger.info("Running target pipeline... ")
    target_ds = run_pipeline(case_operator.case_metadata, case_operator.target)
    logger.info("Running forecast pipeline... ")
    forecast_ds = run_pipeline(case_operator.case_metadata, case_operator.forecast)
    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        if "valid_time" in zero_length_dims:
            logger.warning(
                f"Forecast dataset for case "
                f"{case_operator.case_metadata.case_id_number} "
                f"has no data for case time range "
                f"{case_operator.case_metadata.start_date} to "
                f"{case_operator.case_metadata.end_date}."
            )
        else:
            logger.warning(
                f"Forecast dataset for case "
                f"{case_operator.case_metadata.case_id_number} "
                f"has zero-length dimensions {zero_length_dims} for case time range "
                f"{case_operator.case_metadata.start_date} "
                f"to {case_operator.case_metadata.end_date}."
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
) -> xr.Dataset:
    """Shared method for running an input pipeline.

    Args:
        case_metadata: The case metadata to run the pipeline on.
        input_data: The input data to run the pipeline on.

    Returns:
        The processed input data as an xarray dataset.
    """
    # Open data and process through pipeline steps
    data = (
        # Opens data from user-defined source
        input_data.open_and_maybe_preprocess_data_from_source()
        # Maps variable names to the input data if not already using EWB
        # naming conventions
        .pipe(input_data.maybe_map_variable_names)
        # subsets the input data to the variables defined in the input data
        .pipe(inputs.maybe_subset_variables, variables=input_data.variables)
        # Subsets the input data using case metadata
        .pipe(
            input_data.subset_data_to_case,
            case_metadata=case_metadata,
        )
        # Converts the input data to an xarray dataset if it is not already
        .pipe(input_data.maybe_convert_to_dataset)
        # Adds the name of the dataset to the dataset attributes
        .pipe(input_data.add_source_to_dataset_attrs)
        # Derives variables if needed
        .pipe(
            derived.maybe_derive_variables,
            variables=input_data.variables,
            case_metadata=case_metadata,
        )
    )
    return data
