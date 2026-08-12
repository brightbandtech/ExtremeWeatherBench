"""The output contract for evaluation results: xarray in, pandas/xarray out."""

import logging
import pathlib
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd
import sparse
import xarray as xr

import extremeweatherbench.utils as utils

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

# These become dimensions of the merged Dataset; event_type rides along
# case_id_number instead, and target/forecast_variable become the data
# variable name rather than dimensions.
_METADATA_DIMS = ("case_id_number", "metric", "forecast_source", "target_source")

# A dense case_id_number x metric x ... x latitude x longitude hypercube
# grows fast for unaggregated spatial results; past this many elements,
# nudge the caller towards sparse=True instead of exhausting memory.
_DENSE_SIZE_WARNING_THRESHOLD = 1_000_000


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


def _variable_name(forecast_variable: str, target_variable: str) -> str:
    """Name a merged data variable from its forecast/target variable pair."""
    if forecast_variable == target_variable:
        return target_variable
    return f"{forecast_variable}_vs_{target_variable}"


def _sentinel_for_dtype(dtype: np.dtype) -> object:
    """Pick an out-of-band coordinate value to pad a dim a result lacks.

    Args:
        dtype: The dtype of the dimension coordinate elsewhere in the run.

    Returns:
        NaT for datetime/timedelta dtypes, otherwise NaN.
    """
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT", "ns")
    if np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64("NaT", "ns")
    return np.nan


def _collect_own_dim_dtypes(results: list[xr.DataArray]) -> dict[str, np.dtype]:
    """Find every non-metadata dim used anywhere in the run, with its dtype."""
    dim_dtypes: dict[str, np.dtype] = {}
    for result in results:
        for dim in result.dims:
            if dim in _METADATA_DIMS or dim in dim_dtypes:
                continue
            dim_dtypes[dim] = (
                result.coords[dim].dtype
                if dim in result.coords
                else np.dtype("float64")
            )
    return dim_dtypes


def _result_to_padded_dataset(
    result: xr.DataArray, own_dim_dtypes: dict[str, np.dtype]
) -> xr.Dataset:
    """Convert one annotated result into a slab ready to merge into a run.

    Promotes non-dim coords other than metadata into data variables (they
    otherwise conflict across results that share a dim label but not the
    coord's value, e.g. landfall coords riding along init_time), pads in
    any dim used elsewhere in the run but absent from this result (so
    xarray can't silently broadcast this result's values across it), then
    expand_dims the scalar metadata coords into length-1 dims.

    Args:
        result: A single annotated metric result.
        own_dim_dtypes: Dtypes of every non-metadata dim seen in the run,
            from _collect_own_dim_dtypes.

    Returns:
        A Dataset with one data variable per (variable pair, extra coord),
        padded to case_id_number/metric/forecast_source/target_source.
    """
    event_type = result.coords["event_type"].item()
    var_name = _variable_name(
        result.coords["forecast_variable"].item(),
        result.coords["target_variable"].item(),
    )

    extra_coords = [
        c for c in result.coords if c not in METADATA_COORDS and c not in result.dims
    ]
    ds = result.to_dataset(name=var_name)
    if extra_coords:
        ds = ds.reset_coords(extra_coords)
    ds = ds.drop_vars(["target_variable", "forecast_variable"])

    for dim in own_dim_dtypes:
        if dim not in ds.dims:
            ds = ds.expand_dims({dim: [_sentinel_for_dtype(own_dim_dtypes[dim])]})

    ds = ds.expand_dims(list(_METADATA_DIMS))
    return ds.assign_coords(event_type=("case_id_number", [event_type]))


def _estimate_dense_size(datasets: list[xr.Dataset]) -> int:
    """Estimate the element count of the dense hypercube these would pad to."""
    dim_labels: dict[str, set] = {}
    for ds in datasets:
        for dim, index in ds.indexes.items():
            dim_labels.setdefault(dim, set()).update(index.values.tolist())
    size = 1
    for labels in dim_labels.values():
        size *= max(len(labels), 1)
    return size


def _maybe_chunk(ds: xr.Dataset) -> xr.Dataset:
    """Chunk a Dataset with dask unless it is already dask-backed."""
    if any(isinstance(v.data, da.Array) for v in ds.data_vars.values()):
        return ds
    return ds.chunk()


def _sparsify(ds: xr.Dataset) -> xr.Dataset:
    """Back each floating-point data variable with sparse.COO, fill=NaN."""

    def _to_sparse(variable: xr.DataArray) -> xr.DataArray:
        if not np.issubdtype(variable.dtype, np.floating):
            return variable
        dense = np.asarray(variable.data)
        return variable.copy(data=sparse.COO.from_numpy(dense, fill_value=np.nan))

    return ds.map(_to_sparse)


def results_to_dataset(results: list[xr.DataArray], sparse: bool = False) -> xr.Dataset:
    """Build one flat Dataset from a whole run's annotated metric results.

    Each result becomes a length-1 slab along case_id_number, metric,
    forecast_source, and target_source (via expand_dims on the scalar
    metadata coords annotate_metric_result attached), keeping whatever
    dims the metric itself preserved (lead_time, init_time, latitude,
    longitude, level, ...). Slabs are combined with an outer join so
    each result occupies its own disjoint hyper-slab; everything else is
    NaN (or absent, if sparse=True).

    Args:
        results: A flat list of annotated metric results for a run.
        sparse: If True, back data variables with sparse.COO (fill_value
            NaN) instead of densifying the padded hypercube.

    Returns:
        The merged Dataset, or an empty Dataset if results is empty.
    """
    if not results:
        return xr.Dataset()

    own_dim_dtypes = _collect_own_dim_dtypes(results)
    datasets = [_result_to_padded_dataset(r, own_dim_dtypes) for r in results]

    if sparse:
        datasets = [_sparsify(ds) for ds in datasets]
    else:
        dense_size = _estimate_dense_size(datasets)
        if dense_size > _DENSE_SIZE_WARNING_THRESHOLD:
            logger.warning(
                "Densifying this run's results would require an estimated "
                "%d elements per data variable. Pass sparse=True to avoid "
                "materializing the padded hypercube.",
                dense_size,
            )
        datasets = [_maybe_chunk(ds) for ds in datasets]

    return xr.merge(datasets, join="outer", compat="no_conflicts")


def write_results(
    results: Union[list[xr.DataArray], pd.DataFrame, xr.Dataset],
    path: Union[str, pathlib.Path],
    output_format: str = "csv",
    sparse: bool = False,
) -> None:
    """Write evaluation results to disk in the given format.

    Args:
        results: A flat list of annotated metric results, or the
            already-converted output: a DataFrame for output_format
            "csv", or a Dataset for output_format "netcdf"/"zarr".
        path: Destination path.
        output_format: One of "csv", "netcdf", or "zarr".
        sparse: Forwarded to results_to_dataset when results is a list
            and output_format is "netcdf" or "zarr".

    Raises:
        ValueError: If output_format is not "csv", "netcdf", or "zarr".
    """
    if output_format == "csv":
        df = (
            results
            if isinstance(results, pd.DataFrame)
            else results_to_dataframe(results)
        )
        df.to_csv(path, index=False)
        return

    if output_format in ("netcdf", "zarr"):
        ds = (
            results
            if isinstance(results, xr.Dataset)
            else results_to_dataset(results, sparse=sparse)
        )
        # netCDF/zarr can't store sparse.COO directly; dask stays lazy.
        ds = ds.map(utils.maybe_densify_dataarray)
        # Coords are cheap to materialize and avoid a dtype-inference
        # warning from writing them as dask arrays.
        dask_coords = {
            c: ds.coords[c].compute()
            for c in ds.coords
            if isinstance(ds.coords[c].data, da.Array)
        }
        if dask_coords:
            ds = ds.assign_coords(dask_coords)
        if output_format == "netcdf":
            ds.to_netcdf(path)
        else:
            ds.to_zarr(path, mode="w")
        return

    raise ValueError(
        f"Unknown output_format '{output_format}'. Expected one of: "
        "'csv', 'netcdf', 'zarr'."
    )
