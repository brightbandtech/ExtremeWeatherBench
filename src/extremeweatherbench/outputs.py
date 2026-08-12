"""The output contract for evaluation results: xarray in, pandas out."""

import logging

import pandas as pd
import xarray as xr

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

# forecast_variable is dropped when converting to a dataframe.
METADATA_COORDS = (
    "metric",
    "target_variable",
    "forecast_variable",
    "forecast_source",
    "target_source",
    "case_id_number",
    "event_type",
)


def annotate_metric_result(result: xr.DataArray, **metadata) -> xr.DataArray:
    """Attach evaluation metadata to a metric result as scalar coords.

    Args:
        result: The metric result to annotate.
        **metadata: Scalar metadata to attach, typically the fields in
            METADATA_COORDS.

    Returns:
        The result with metadata assigned as non-dim coords.
    """
    return result.assign_coords(metadata)


def _ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe conforms to OUTPUT_COLUMNS schema.

    Args:
        df: Dataframe produced from an annotated metric result, with
            metadata already present as columns.

    Returns:
        DataFrame with columns matching OUTPUT_COLUMNS specification,
        followed by any extra columns (e.g. landfall metadata).
    """
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

    extra_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    return df.reindex(columns=OUTPUT_COLUMNS + extra_cols)


def results_to_dataframe(results: list[xr.DataArray]) -> pd.DataFrame:
    """Convert annotated metric results into the long-form output dataframe.

    Args:
        results: Annotated metric results, e.g. from annotate_metric_result.

    Returns:
        Concatenated long-form DataFrame matching the OUTPUT_COLUMNS schema.
    """
    if not results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    dataframes = []
    for result in results:
        df = result.to_dataframe(name="value").reset_index()
        df = df.drop(columns="forecast_variable", errors="ignore")
        dataframes.append(_ensure_output_schema(df))
    return _safe_concat(dataframes, ignore_index=True)


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
