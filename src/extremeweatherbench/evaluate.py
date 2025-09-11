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
from extremeweatherbench.defaults import OUTPUT_COLUMNS

if TYPE_CHECKING:
    from extremeweatherbench import metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.debug(f"Skipping empty DataFrame {i}")
            continue
        # Skip DataFrames where all values are NA
        if df.isna().all().all():
            logger.debug(f"Skipping all-NA DataFrame {i}")
            continue
        # Skip DataFrames where all columns are empty/NA
        if len(df.columns) > 0 and all(df[col].isna().all() for col in df.columns):
            logger.debug(f"Skipping DataFrame {i} with all-NA columns")
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


class ExtremeWeatherBench:
    """A class to run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is a
    wrapper around the
    case operators and evaluation objects to create either a serial loop or will return
    the built case operators to run in parallel as defined by the user.


    Attributes:
        cases: A dictionary of cases to run.
        metrics: A list of metrics to run.
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

        run_results = []
        with logging_redirect_tqdm():
            for case_operator in tqdm(self.case_operators, desc="Processing cases"):
                run_results.append(compute_case_operator(case_operator, **kwargs))

                # store the results of each case operator if caching
                if self.cache_dir:
                    concatenated = _safe_concat(run_results, ignore_index=False)
                    if not concatenated.empty:
                        concatenated.to_pickle(self.cache_dir / "case_results.pkl")

        return _safe_concat(run_results, ignore_index=True)


def compute_case_operator(case_operator: "cases.CaseOperator", **kwargs):
    """Compute the results of a case operator.

    This method will compute the results of a case operator. It will build
    the target and forecast datasets,
    align them, compute the metrics, and return a concatenated dataframe of the results.

    Args:
        case_operator: The case operator to compute the results of.
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

    # Derive the variables for the forecast and target datasets independently
    aligned_forecast_ds = derived.maybe_derive_variables(
        aligned_forecast_ds, variables=case_operator.forecast.variables
    )
    aligned_target_ds = derived.maybe_derive_variables(
        aligned_target_ds, variables=case_operator.target.variables
    )

    logger.info(f"datasets built for case {case_operator.case_metadata.case_id_number}")
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

        # cache the results of each metric if caching
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir:
            cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            concatenated = _safe_concat(results, ignore_index=True)
            if not concatenated.empty:
                concatenated.to_pickle(cache_path / "results.pkl")

    return _safe_concat(results, ignore_index=True)


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
        "target_source": target_ds.attrs["source"],
        "forecast_source": forecast_ds.attrs["source"],
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
            metric_list=metric.name,
            target_source=target_ds.attrs["source"],
            forecast_source=forecast_ds.attrs["source"],
            case_id_number=case_id,
            event_type=event_type
        )
    """
    # Add metadata columns
    for col, value in metadata.items():
        df[col] = value

    # Check for missing columns and warn
    missing_cols = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing_cols:
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
    metric = metric()
    logger.info(f"computing metric {metric.name}")
    metric_result = metric.compute_metric(
        forecast_ds[forecast_variable],
        target_ds[target_variable],
        **kwargs,
    )

    # Convert to DataFrame and add metadata, ensuring OUTPUT_COLUMNS compliance
    df = metric_result.to_dataframe(name="value").reset_index()
    # TODO: add functionality for custom metadata columns
    metadata = _extract_standard_metadata(
        target_variable, metric, target_ds, forecast_ds, case_id_number, event_type
    )
    return _ensure_output_schema(df, **metadata)


def _build_datasets(
    case_operator: "cases.CaseOperator",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the target and forecast datasets for a case operator.

    This method will process through all stages of the pipeline for the target and
    forecast datasets, including preprocessing, variable renaming, and subsetting.
    """
    logger.info("running forecast pipeline")
    forecast_ds = run_pipeline(case_operator, "forecast")

    # Check if any dimension has zero length
    zero_length_dims = [dim for dim, size in forecast_ds.sizes.items() if size == 0]
    if zero_length_dims:
        logger.warning(
            f"forecast dataset for case {case_operator.case_metadata.case_id_number} "
            f"has zero-length dimensions {zero_length_dims} for case time range "
            f"{case_operator.case_metadata.start_date} to "
            f"{case_operator.case_metadata.end_date}"
        )
        return xr.Dataset(), xr.Dataset()
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
        # (computed_dataset.to_netcdf(self.cache_dir) for computed_dataset in
        # computed_datasets)
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
        # maps variable names to the target data if not already using EWB
        # naming conventions
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
