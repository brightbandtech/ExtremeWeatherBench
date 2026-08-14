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
    # invalid, so a metric supplying only one of them is intended behavior. Peak metrics
    # supply both, because they reduce over a run and then report against the lead time
    # at which that run verified against the target's extreme.
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
    valid_dfs = []
    for i, df in enumerate(dataframes):
        if df.empty or df.isna().all().all():
            logger.debug("Skipping empty or all-NA DataFrame %s", i)
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


_PEAK_TIME_COORDS = ("init_time", "lead_time")


def _scatter_peak_time_coord(result: xr.DataArray) -> xr.DataArray:
    """Turn a peak metric's verifying lead/init time into a real dim.

    Peak metrics reduce over a run but report against the lead_time (or
    init_time) at which that run verified: one of the pair is the
    result's own dim, the other rides along it as a non-dim coord. A
    MultiIndex over both, then unstacked, scatters each value to its
    true position instead of leaving the pair to collide downstream.

    Args:
        result: A single annotated metric result.

    Returns:
        result unchanged, unless it carries exactly one of init_time/
        lead_time as its own dim and the other as a non-dim coord along
        it, in which case both become dims of the returned result.
    """
    own_dim = next((d for d in _PEAK_TIME_COORDS if d in result.dims), None)
    other = next((c for c in _PEAK_TIME_COORDS if c != own_dim), None)
    if own_dim is None or other not in result.coords or other in result.dims:
        return result
    return result.set_index(__scatter_idx=(own_dim, other)).unstack("__scatter_idx")


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


def _sparsifiable(dtype: np.dtype) -> bool:
    """Whether sparse.COO can hold dtype with a well-defined fill value."""
    return (
        np.issubdtype(dtype, np.floating)
        or np.issubdtype(dtype, np.datetime64)
        or np.issubdtype(dtype, np.timedelta64)
    )


def _collect_data_var_info(
    datasets: list[xr.Dataset],
) -> dict[str, tuple[tuple[str, ...], np.dtype]]:
    """First-seen dims and dtype for every data variable across all slabs."""
    info: dict[str, tuple[tuple[str, ...], np.dtype]] = {}
    for ds in datasets:
        for name, variable in ds.data_vars.items():
            if name not in info:
                info[name] = (variable.dims, variable.dtype)
    return info


def _positions_in_union(index: pd.Index, values: np.ndarray) -> np.ndarray:
    """Map values onto integer positions in index, resolving null padding.

    get_indexer can't match a null placeholder (NaT/NaN) by value, so an
    unmatched query is pointed at index's own (unique) null entry.
    """
    positions = index.get_indexer(values)
    unmatched = positions == -1
    if unmatched.any():
        positions[unmatched] = np.flatnonzero(index.isna())[0]
    return positions


def _scatter_variable(
    dims: tuple[str, ...],
    dtype: np.dtype,
    name: str,
    datasets: list[xr.Dataset],
    dim_index: dict[str, pd.Index],
) -> Union[sparse.COO, np.ndarray]:
    """Scatter one data variable's values from every slab that has it.

    Positions are resolved per slab, per dim, straight into the run-wide
    union index, so this never needs xr.merge's fillna-based combine.
    """
    shape = tuple(len(dim_index[d]) for d in dims)
    fill_value = _sentinel_for_dtype(dtype)

    coord_rows: list[list[np.ndarray]] = [[] for _ in dims]
    value_chunks: list[np.ndarray] = []
    for ds in datasets:
        if name not in ds.data_vars:
            continue
        variable = ds.data_vars[name]
        positions = {
            d: _positions_in_union(dim_index[d], ds.indexes[d].values)
            for d in variable.dims
        }
        grids = np.meshgrid(*(positions[d] for d in variable.dims), indexing="ij")
        for row, d in zip(coord_rows, dims):
            row.append(grids[variable.dims.index(d)].ravel())
        value_chunks.append(np.asarray(variable.data).ravel())

    coords = np.array([np.concatenate(row) for row in coord_rows], dtype=np.intp)
    values = np.concatenate(value_chunks)

    if _sparsifiable(dtype):
        return sparse.COO(
            coords=coords, data=values, shape=shape, fill_value=fill_value
        )
    dense = np.full(shape, fill_value, dtype=object)
    dense[tuple(coords)] = values
    return dense


def _scatter_to_sparse_dataset(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Combine padded slabs into one Dataset without going through xr.merge.

    xr.merge's compat="no_conflicts" combines same-named variables via
    fillna, which broadcasts through duck_array_ops.transpose whenever two
    slabs disagree on that variable's dim order -- exactly the case where a
    variable shows up from metrics that preserve different dims. sparse
    0.19.1 has no module-level transpose, so that combine crashes for any
    sparse-backed variable. Scattering each variable's values directly into
    its own array, in the run's outer-joined dim order, avoids it.

    Non-sparsifiable data variables (anything but floating/datetime64/
    timedelta64) are merged the ordinary way instead, since they're never
    sparse-backed and so can't trip the same bug.
    """
    var_info = _collect_data_var_info(datasets)
    sparsifiable = {n for n, (_, dt) in var_info.items() if _sparsifiable(dt)}

    dense_slabs = [ds.drop_vars(sparsifiable, errors="ignore") for ds in datasets]
    dense_ds = xr.merge(dense_slabs, join="outer", compat="no_conflicts")

    dim_index = {dim: dense_ds.indexes[dim] for dim in dense_ds.dims}
    scattered = {
        name: (dims, _scatter_variable(dims, dtype, name, datasets, dim_index))
        for name, (dims, dtype) in var_info.items()
        if name in sparsifiable
    }
    return xr.Dataset(
        data_vars={**dense_ds.data_vars, **scattered}, coords=dense_ds.coords
    )


def results_to_dataset(results: list[xr.DataArray], sparse: bool = False) -> xr.Dataset:
    """Build one flat Dataset from a whole run's annotated metric results.

    Each result becomes a length-1 slab along case_id_number, metric,
    forecast_source, and target_source (via expand_dims on the scalar
    metadata coords annotate_metric_result attached), keeping whatever
    dims the metric itself preserved (lead_time, init_time, latitude,
    longitude, level, ...). A peak metric result that carries both
    init_time and lead_time (see _scatter_peak_time_coord) occupies
    both dims. Slabs are combined with an outer join so each result
    occupies its own disjoint hyper-slab; everything else is NaN (or
    absent, if sparse=True).

    Args:
        results: A flat list of annotated metric results for a run.
        sparse: If True, back data variables with sparse.COO (fill_value
            NaN) instead of densifying the padded hypercube.

    Returns:
        The merged Dataset, or an empty Dataset if results is empty.
    """
    if not results:
        return xr.Dataset()

    results = [_scatter_peak_time_coord(r) for r in results]
    own_dim_dtypes = _collect_own_dim_dtypes(results)
    datasets = [_result_to_padded_dataset(r, own_dim_dtypes) for r in results]

    if sparse:
        return _scatter_to_sparse_dataset(datasets)

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


def _label_has_data(obj: Union[xr.Dataset, xr.DataArray], dim: str) -> xr.DataArray:
    """Find which labels along dim have at least one non-NaN value."""
    other_dims = [d for d in obj.dims if d != dim]
    reduced = obj.notnull().any(other_dims) if other_dims else obj.notnull()
    if isinstance(reduced, xr.Dataset):
        reduced = reduced.to_array().any("variable")
    return reduced


def drop_empty_slices(
    obj: Union[xr.Dataset, xr.DataArray],
) -> Union[xr.Dataset, xr.DataArray]:
    """Collapse the placeholder padding results_to_dataset introduces.

    Meant to be called after selecting down to a single metric (and,
    typically, a single case), not on the full run's cube: there, every
    dim mixes real and placeholder labels, so nothing is all-NaN and
    this is close to a no-op.

    For each dim, drops labels that are entirely NaN, then drops any
    dim left with only a single null (placeholder) label. Applied to
    an RMSE selection, this removes the untouched init_time labels and
    then the now-single-label placeholder init_time dim, leaving a
    clean series over lead_time; a DurationMeanError selection
    collapses symmetrically to a clean series over init_time.

    Does not work on sparse.COO-backed input (from results_to_dataset's
    sparse=True): indexing with a boolean mask ends up calling .values,
    which sparse.COO refuses to densify implicitly. Densify first with
    utils.maybe_densify_dataarray if obj may be sparse-backed.

    Args:
        obj: A Dataset or DataArray, typically a selection out of
            results_to_dataset's output.

    Returns:
        The same type as obj, with empty slices and placeholder-only
        dims dropped.
    """
    result = obj
    for dim in list(result.dims):
        has_data = _label_has_data(result, dim)
        result = result.isel({dim: has_data.compute().values})

    placeholder_dims = [
        dim
        for dim in result.dims
        if result.sizes[dim] == 1 and pd.isnull(result[dim].values[0])
    ]
    return result.squeeze(placeholder_dims, drop=True)


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
