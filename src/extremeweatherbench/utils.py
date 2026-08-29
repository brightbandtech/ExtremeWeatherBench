"""Utility functions for the ExtremeWeatherBench package that don't fit into any other
specialized package."""

import datetime
import inspect
import logging
import operator
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import pooch
import regionmask
import shapely
import sparse
import tqdm
import xarray as xr
import yaml  # type: ignore[import]
from joblib import Parallel

from extremeweatherbench import progress

logger = logging.getLogger(__name__)

# Natural Earth vector data has been hosted on S3 since 2021; see
# https://github.com/nvkelso/natural-earth-vector/issues/445
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/{resolution}_{category}/"
    "ne_{resolution}_{name}.zip"
)

operators = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _empty_init_time_array() -> xr.DataArray:
    """Empty float DataArray with an init_time dimension."""
    return xr.DataArray(
        np.array([], dtype=float),
        dims=["init_time"],
        coords={"init_time": np.array([], dtype="datetime64[ns]")},
    )


def maybe_get_operator(
    operator_method: Literal[">", ">=", "<", "<=", "==", "!="] | Callable,
) -> Callable:
    """Get the operator function from the operator string. If the operator_method is a
    callable, return it.

    Args:
        operator_method: The operator method to get. Can be a string or a callable.

    Returns:
        The operator function.
    """
    if isinstance(operator_method, str):
        return operators[operator_method]
    return operator_method


def find_common_init_times(
    forecast_landfall: xr.DataArray, target_landfall: xr.DataArray
) -> list[datetime.datetime]:
    """Find the common init_times between forecast and target landfalls.

    Args:
        forecast_landfall: The forecast landfall DataArray.
        target_landfall: The target landfall DataArray.

    Returns:
        Sorted list of init_times present in both forecast and target.
    """
    forecast_init_times = set(forecast_landfall.coords["init_time"].values)
    target_init_times = set(target_landfall.coords["init_time"].values)
    common_init_times = sorted(forecast_init_times.intersection(target_init_times))
    return common_init_times


def is_valid_landfall(landfall: xr.DataArray | None) -> bool:
    """Check if a landfall DataArray is valid for processing.

    A valid landfall has dimensions and contains the init_time coordinate
    needed for landfall metric calculations. Also checks that the data
    contains at least some non-NaN values.

    Args:
        landfall: The landfall DataArray to check

    Returns:
        True if the landfall is valid, False otherwise
    """
    if landfall is None or landfall.ndim == 0:
        return False
    if "init_time" not in landfall.coords:
        return False
    if landfall.size == 0:
        return False
    # notnull().any() reduces chunk by chunk. Reading .values first would pull
    # every chunk into one array and build a full-size boolean beside it, all
    # to answer a single yes-or-no question.
    return bool(landfall.notnull().any())


def _create_nan_dataarray(
    preserved_dims: str | list[str] = "init_time",
) -> xr.DataArray:
    """Create a NaN DataArray with the given dimension(s).

    Args:
        preserved_dims: The dimension(s) to create the NaN DataArray for.
            Can be a single string or a list of strings.

    Returns:
        A DataArray with the given dimension(s) and NaN values.
    """
    if isinstance(preserved_dims, str):
        preserved_dims = [preserved_dims]

    # Create shape with one element per dimension
    shape = [1] * len(preserved_dims)
    nan_values = np.full(shape, np.nan)
    nan_da = xr.DataArray(nan_values, dims=preserved_dims)
    return nan_da


def convert_longitude_to_360(longitude: float) -> float:
    """Convert a longitude from the range [-180, 180) to [0, 360)."""
    return np.mod(longitude, 360)


def convert_longitude_to_180(
    longitude: float | xr.Dataset | xr.DataArray,
    longitude_name: str = "longitude",
) -> float | xr.Dataset | xr.DataArray:
    """Convert a longitude from the range [0, 360) to [-180, 180).

    Datasets are coerced to [-180, 180) and sorted by longitude.
    """
    if isinstance(longitude, (xr.Dataset, xr.DataArray)):
        longitude.coords[longitude_name] = (
            longitude.coords[longitude_name] + 180
        ) % 360 - 180
        longitude = longitude.sortby(longitude_name)
        return longitude
    else:
        return np.mod(longitude - 180, 360) - 180


# Cached Natural Earth land masks keyed by lat/lon byte values.
_REGIONMASK_LAND_CACHE: dict[tuple[bytes, bytes], xr.DataArray] = {}


def regionmask_land_110(
    latitude: xr.DataArray | npt.NDArray, longitude: xr.DataArray | npt.NDArray
) -> xr.DataArray:
    """Land mask from Natural Earth 110m. 0 is land. Cached per grid."""
    lat_vals = np.asarray(latitude)
    lon_vals = np.asarray(longitude)
    key = (lat_vals.tobytes(), lon_vals.tobytes())
    mask = _REGIONMASK_LAND_CACHE.get(key)
    if mask is None:
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask = land.mask(longitude, latitude)
        _REGIONMASK_LAND_CACHE[key] = mask
    return mask


def remove_ocean_gridpoints(dataset: xr.Dataset) -> xr.Dataset:
    """Subset a dataset to only include land gridpoints based on a land-sea mask.

    Args:
        dataset: The input xarray dataset.

    Returns:
        The dataset masked to only land gridpoints.
    """
    land_sea_mask = regionmask_land_110(dataset.latitude, dataset.longitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    return dataset.where(land_mask)


def read_event_yaml(input_pth: str | pathlib.Path) -> dict:
    """Read events yaml from data."""
    logger.warning(
        "This function is deprecated and will be removed in a future release. "
        "Please use cases.read_incoming_yaml instead."
    )
    input_pth = pathlib.Path(input_pth)
    with open(input_pth, "rb") as f:
        yaml_event_case = yaml.safe_load(f)
    return yaml_event_case


def derive_indices_from_init_time_and_lead_time(
    dataset: xr.Dataset,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> tuple[np.ndarray[Any, Any], ...]:
    """Derive the indices of valid times in a dataset when the dataset has init_time and
    lead_time coordinates.

    Args:
        dataset: The dataset to derive the indices from.
        start_date: The start date to derive the indices from.
        end_date: The end date to derive the indices from.

    Returns:
        The indices of valid times in the dataset.

    Example:
        >>> import xarray as xr
        >>> import datetime
        >>> import pandas as pd
        >>> from extremeweatherbench.utils import (
        ...     derive_indices_from_init_time_and_lead_time,
        ... )
        >>> ds = xr.Dataset(
        ...     coords={
        ...         "init_time": pd.date_range("2020-01-01", "2020-01-03"),
        ...         "lead_time": [0, 24, 48],  # hours
        ...     }
        ... )
        >>> start = datetime.datetime(2020, 1, 1)
        >>> end = datetime.datetime(2020, 1, 4)
        >>> indices = derive_indices_from_init_time_and_lead_time(ds, start, end)
        >>> print(indices)
        array([0, 0, 1, 1, 2])
    """
    lead_time_grid, init_time_grid = np.meshgrid(dataset.lead_time, dataset.init_time)
    valid_times = (
        init_time_grid.flatten()
        + pd.to_timedelta(lead_time_grid.flatten(), unit="h").to_numpy()
    )
    valid_times_reshaped = valid_times.reshape(
        (
            dataset.init_time.shape[0],
            dataset.lead_time.shape[0],
        )
    )
    valid_time_mask = (valid_times_reshaped >= pd.to_datetime(start_date)) & (
        valid_times_reshaped <= pd.to_datetime(end_date)
    )
    valid_time_indices = np.asarray(valid_time_mask).nonzero()

    return valid_time_indices


def filter_kwargs_for_callable(kwargs: dict, callable_obj: Callable) -> dict:
    """Filter kwargs to only include arguments that the callable can accept.

    This method uses introspection to determine which arguments the callable
    can accept and filters kwargs accordingly.

    Args:
        kwargs: The full kwargs dictionary to filter
        callable_obj: The callable (function, method, etc.) to check against

    Returns:
        A filtered dictionary containing only the kwargs that the callable can accept
    """
    # Get the signature of the callable
    sig = inspect.signature(callable_obj)

    # Get the parameter names that the callable accepts
    # For bound methods, 'self' is already excluded from the signature
    accepted_params = list(sig.parameters.keys())

    # Filter kwargs to only include accepted parameters
    filtered_kwargs = {}
    for param_name in accepted_params:
        if param_name in kwargs:
            filtered_kwargs[param_name] = kwargs[param_name]

    return filtered_kwargs


def min_if_all_timesteps_present(
    da: xr.DataArray,
    time_resolution_hours: float,
) -> xr.DataArray:
    """Return the minimum value of a DataArray if all timesteps of a day are present.

    Counts values that are actually present rather than the length of the time
    coordinate, so a day padded out with NaNs is correctly rejected as
    incomplete.

    Args:
        da: The input DataArray.
        time_resolution_hours: The spacing of the data in hours.

    Returns:
        The minimum value of the DataArray if all timesteps are present,
        otherwise NaN.
    """
    timesteps_per_day = 24 / time_resolution_hours
    # Comparing lazily rather than in a Python `if` keeps a dask-backed day
    # out of memory: nothing here is computed until the caller asks for it.
    timesteps_present = da.notnull().sum()
    return da.min().where(timesteps_present == timesteps_per_day)


def min_if_all_timesteps_present_forecast(
    da: xr.DataArray, time_resolution_hours: float
) -> xr.DataArray:
    """Return the minimum value of a DataArray if all timesteps of a day are present
    given a dataset with lead_time and valid_time dimensions.

    The completeness check is made per lead time against the number of values
    actually present. Checking the length of the valid_time coordinate instead
    would pass for every lead time, because that coordinate is the union over
    all lead times and is denser than any single lead time's sampling.

    Args:
        da: The input DataArray.
        time_resolution_hours: The spacing of the data in hours.

    Returns:
        The minimum along valid_time for lead times holding a full day of
        values, and NaN for the rest.
    """
    timesteps_per_day = 24 / time_resolution_hours
    timesteps_present = da.notnull().sum("valid_time")
    return da.min("valid_time").where(timesteps_present == timesteps_per_day)


def expected_timesteps_per_day(forecast: xr.DataArray) -> int:
    """Number of forecast steps that make up one day at the lead_time spacing.

    This is the per-init sampling rate, which is the meaningful one when asking
    whether a single forecast run covers a full day. It is unrelated to the
    spacing of the valid_time axis, which is the union over all lead times.

    Args:
        forecast: A forecast DataArray with a lead_time coordinate.

    Returns:
        The number of lead times spanning 24 hours, at least 1.
    """
    lead_time = _lead_time_as_timedelta(forecast.lead_time)
    steps = np.abs(np.diff(lead_time.values)) / np.timedelta64(1, "h")
    steps = steps[steps > 0]
    if steps.size == 0:
        return 1
    return max(round(24 / steps.min()), 1)


def reduce_forecast_over_window_per_init(
    forecast: xr.DataArray,
    center_time: Any,
    tolerance_range_hours: int,
    method: str = "max",
    required_timesteps: int = 1,
) -> xr.DataArray:
    """Reduce a forecast over a time window separately for each initialization.

    Peak metrics ask what extreme value a forecast predicted near the time the
    target's extreme occurred. That is a reduction along a single run's
    trajectory, so it must be taken over lead_time at fixed init_time. Reducing
    over valid_time at fixed lead_time instead samples a different run at every
    step, and when the initialization interval is wider than the window it
    degenerates to a single instantaneous value whose time of day is set by
    ``lead_time mod 24 h``.

    The result is indexed by the lead time of the target's extreme relative to
    each initialization, ``center_time - init_time``, and keeps init_time as a
    coordinate so the provenance of each value survives into the output.

    Args:
        forecast: Forecast DataArray with lead_time and valid_time dimensions,
            already reduced over any spatial dimensions.
        center_time: The target time the window is centered on.
        tolerance_range_hours: Full width of the window in hours.
        method: The reduction to apply, e.g. "max" or "min".
        required_timesteps: Minimum number of values an initialization must
            contribute inside the window to be scored. Initializations with
            fewer are dropped rather than scored on partial coverage.

    Returns:
        A DataArray with a lead_time dimension and an init_time coordinate.
    """
    half_window = np.timedelta64(tolerance_range_hours // 2, "h")
    center = np.ravel(np.asarray(center_time))[0]

    forecast_by_init = convert_valid_time_to_init_time(forecast)
    windowed = forecast_by_init.where(
        (forecast_by_init.valid_time >= center - half_window)
        & (forecast_by_init.valid_time <= center + half_window)
    )
    windowed = windowed.where(
        windowed.notnull().sum("lead_time") >= required_timesteps, drop=True
    )

    reduced = getattr(windowed, method)("lead_time")
    reduced = reduced.assign_coords(
        lead_time=("init_time", (center - reduced.init_time).data)
    )
    # A run launched after the target's extreme did not forecast it, and a
    # negative lead time would be meaningless in the output.
    reduced = reduced.where(reduced.lead_time >= np.timedelta64(0, "h"), drop=True)
    return reduced.swap_dims({"init_time": "lead_time"}).sortby("lead_time")


def determine_temporal_resolution(
    data: xr.Dataset | xr.DataArray,
) -> float | None:
    """Determine the temporal resolution of the data.

    Args:
        data: The input dataset with a valid_time dimension or coordinate.

    Returns:
        The temporal resolution of the data as a float in hours.
    """
    # Read the times off the index where there is one. Going through
    # data.valid_time builds a DataArray and diffs it through xarray's
    # dispatch, which costs several times more than diffing the index.
    if "valid_time" in data.indexes:
        valid_time = data.indexes["valid_time"].to_numpy()
    else:
        valid_time = np.asarray(data["valid_time"].values)

    num_timesteps = np.unique(np.diff(valid_time)).astype("timedelta64[h]").astype(int)
    if len(num_timesteps) > 1:
        logger.warning(
            "Multiple time resolutions found in dataset, data may be missing in "
            "forecast or target datasets. Returning the highest time resolution."
        )
    # likely missing any data for valid time
    if len(num_timesteps) == 0:
        return None

    # return the minimum (highest time resolution) in hours
    # this is the most likely to be correct if there are multiple resolutions
    # present, likely due to missing data
    return np.min(num_timesteps).astype(float)


def _lead_time_as_timedelta(lead_time: xr.DataArray) -> xr.DataArray:
    """Coerce an integer-hours lead_time to timedelta64."""
    if np.issubdtype(lead_time.dtype, np.timedelta64):
        return lead_time
    return lead_time.copy(data=pd.to_timedelta(lead_time.values, unit="h"))


def convert_init_time_to_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """Convert the init_time coordinate to a valid_time coordinate.

    Each lead is reindexed onto the union of valid times, then concatenated.
    That matches concat(..., join="outer") without growing aligns.

    Args:
        ds: The dataset to convert with lead_time and init_time coordinates.

    Returns:
        The dataset with a valid_time coordinate.
    """
    lead_time = _lead_time_as_timedelta(ds.lead_time)
    ds = ds.assign_coords(valid_time=ds.init_time + lead_time)
    vt_union = np.unique(ds.valid_time.values.reshape(-1))
    pieces = [
        ds.isel(lead_time=i)
        .swap_dims({"init_time": "valid_time"})
        .reindex(valid_time=vt_union)
        for i in range(ds.sizes["lead_time"])
    ]
    out = xr.concat(
        pieces,
        dim="lead_time",
        coords="different",
        compat="equals",
        join="outer",
    )
    out = out.assign_coords(lead_time=("lead_time", ds.lead_time.values))
    if "valid_time_mask" in out.coords:
        mask = out.valid_time_mask
        out = out.assign_coords(
            valid_time_mask=xr.where(mask.isnull(), False, mask).astype(bool)
        )
    return out


def values_as_bool(values) -> npt.NDArray[np.bool_]:
    """Coerce an array to bool; NaN / non-finite fills become False."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return np.isfinite(arr) & (arr != 0)
    return arr.astype(bool)


def stack_valid_time_pairs(
    obj: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Keep only valid (lead_time, valid_time) pairs as a sample dim.

    Uses the ``valid_time_mask`` coordinate so fill cells from
    ``convert_init_time_to_valid_time`` never enter the dask graph.
    """
    if not {"lead_time", "valid_time"} <= set(obj.dims):
        return obj
    if "valid_time_mask" not in obj.coords:
        return obj
    stacked = obj.stack(sample=("lead_time", "valid_time"))
    keep = values_as_bool(stacked["valid_time_mask"]).ravel()
    return stacked.isel(sample=np.flatnonzero(keep))


def unstack_valid_time_pairs(
    obj: xr.Dataset | xr.DataArray,
    like: xr.Dataset | xr.DataArray | None = None,
) -> xr.Dataset | xr.DataArray:
    """Restore lead_time × valid_time from a valid-pair sample dim.

    When ``like`` is given, reindex onto its lead_time and valid_time so
    fill cells come back as NaN on the original dense grid.
    """
    if "sample" not in obj.dims:
        return obj
    out = obj.unstack("sample")
    if like is not None and {"lead_time", "valid_time"} <= set(like.dims):
        out = out.reindex(lead_time=like.lead_time, valid_time=like.valid_time)
    return out


def convert_valid_time_to_init_time(da: xr.DataArray) -> xr.DataArray:
    """Convert the valid_time dimension to a init_time dimension.

    Args:
        da: The dataarray to convert with lead_time and valid_time dimensions.

    Returns:
        The dataarray with an init_time dimension.
    """
    lead_time = _lead_time_as_timedelta(da.lead_time)
    init_time = xr.DataArray(
        da.valid_time, coords={"valid_time": da.valid_time}
    ) - xr.DataArray(lead_time, coords={"lead_time": da.lead_time})
    da = da.assign_coords(init_time=init_time)
    return xr.concat(
        [
            da.sel(lead_time=lead).swap_dims({"valid_time": "init_time"})
            for lead in da.lead_time
        ],
        "lead_time",
        coords="different",
        compat="equals",
        join="outer",
    )


def maybe_get_closest_timestamp_to_center_of_valid_times(
    output_times: xr.DataArray,
    valid_time_values: xr.DataArray,
) -> xr.DataArray:
    if output_times.size > 1:
        # This is a temporary fix to handle the case where there are multiple
        # max/min target values. It's assumed the target value closest to the center
        # of the forecast valid time is the most relevant.
        center_time = valid_time_values.values[valid_time_values.size // 2]
        time_diffs = np.abs(output_times - center_time)
        closest_idx = np.argmin(time_diffs.data)
        output_times = output_times[closest_idx]
    # Pass through the original output times and values if there is only one
    return output_times


# Extract all possible names from the title to handle cases with
# multiple names in formats: "name1 (name2)" or "name1 and name2"
def extract_tc_names(title: str) -> list[str]:
    """Extract tropical cyclone names from case title."""
    import re

    names = []
    title_upper = title.upper()

    # Pattern 1: "name1 (name2)" - extract both names
    paren_match = re.search(r"^(.+?)\s*\((.+?)\)$", title_upper)
    if paren_match:
        names.extend([paren_match.group(1).strip(), paren_match.group(2).strip()])
    # Pattern 2: "name1 and name2" - extract both names
    elif " AND " in title_upper:
        parts = title_upper.split(" AND ")
        names.extend([part.strip() for part in parts])
    else:
        # Single name or other format
        names.append(title_upper)

    return names


def stack_dataarray_from_dims(
    da: xr.DataArray,
    stack_dims: list[str],
    max_size: float = 1e9,
    coords: npt.NDArray | None = None,
) -> xr.DataArray:
    """Stack sparse data with n-dimensions.

    In cases where sparse.COO data is in da.data, this function will stack the
    dimensions and return a densified dataarray using reduce_dims.

    Args:
        da: An xarray dataarray with sparse.COO data
        stack_dims: The dimensions to stack.
        max_size: The maximum size of records to densify; default is 100000.

    Returns:
        The densified xarray dataarray reduced to (time, location).
    """
    if coords is None and isinstance(da.data, sparse.COO):
        coords = da.data.coords
    elif coords is None:
        if da.size != 0:
            return da.stack(stacked=stack_dims)
        return da

    # Check if da dimensions size equals number of rows in coords
    if len(da.dims) != coords.shape[0]:
        # Add a new dimension to coords for the missing dimension
        missing_dim_size = da.shape[0]
        nnz = coords.shape[1]
        # Create indices for the missing dimension (all values)
        new_dim_indices = np.repeat(np.arange(missing_dim_size), nnz)
        # Replicate existing coords for each value in the missing dim
        expanded_coords = np.tile(coords, missing_dim_size)
        # Prepend the new dimension indices
        coords = np.vstack([new_dim_indices, expanded_coords])

    # Get the indices of the dimensions to stack
    reduce_dim_indices = [da.dims.index(dim) for dim in stack_dims]
    reduce_dim_names = [da.dims[n] for n in reduce_dim_indices]
    indices_from_coords = [coords[n] for n in reduce_dim_indices]
    # Create pairs and get unique combinations
    idx_pairs = list(zip(*indices_from_coords))
    unique_idx_pairs = list(set(idx_pairs))
    # Extract coordinate values for each unique pair
    # Each pair represents coordinates for the dimensions being reduced
    coord_values = []
    for pair in unique_idx_pairs:
        # Get actual coordinate values for each dimension
        coord_tuple = tuple(
            da[dim].values[idx] for dim, idx in zip(reduce_dim_names, pair)
        )
        coord_values.append(coord_tuple)

    # If the data is not empty, stack and select the unique coordinates; otherwise,
    # return the data densified as an empty dataarray
    if da.size != 0:
        da = da.stack(stacked=reduce_dim_names).sel(stacked=coord_values)

    # Densify the sparse data using the utility function
    return maybe_densify_dataarray(da, max_size=max_size)


def check_for_vars(variable_list: list[str], source: Sequence) -> str | None:
    """Check if the variable is in the source.

    Args:
        variable_list: The list of variables to check for.
        source: The source to check for the variables.

    Returns:
        The variable if it is in the source, otherwise None.
    """
    for variable in variable_list:
        if variable in source:
            return variable
    return None


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar
    From: https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de
    Attributes:
        total_tasks: int, default: None
            the number of expected jobs. Used in the tqdm progressbar.
            If None, try to infer from the length of the called iterator, and
            fallback to use the number of remaining items as soon as we finish
            dispatching.
            Note: use a list instead of an iterator if you want the total_tasks
            to be inferred from its length.

        desc: str, default: None
            the description used in the tqdm progressbar.

        disable_progressbar: bool, default: False
            If True, a tqdm progressbar is not used.

        show_joblib_header: bool, default: False
            If True, show joblib header before the progressbar.

        pre_close: Callable[[], None] | None, default: None
            Invoked at the top of __call__'s finally block, before the
            case bar closes. Callers with bars nested below the case
            bar (e.g. parallel-mode worker slot bars) must close those
            here, since closing the case bar first leaves them
            redrawing into space the case bar has already vacated.

    Example:
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(0.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        pre_close: Callable[[], None] | None = None,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use disable_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.pre_close = pre_close
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            if self.pre_close is not None:
                self.pre_close()
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()
                progress.clear_bar()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = progress.make_case_bar(
                self.total_tasks, disable=self.disable_progressbar
            )
            # Disallow phase updates: concurrent cases would otherwise
            # fight over a single postfix.
            progress.register_bar(self.progress_bar, allow_phase_updates=False)
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # Check if progress_bar has been initialized
        if self.progress_bar is None:
            return
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def idx_to_coords(
    lat_idx: npt.NDArray,
    lon_idx: npt.NDArray,
    lat_coords: npt.NDArray,
    lon_coords: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert indices to coordinates, handling NaN indices.

    Args:
        lat_idx: The latitude indices.
        lon_idx: The longitude indices.
        lat_coords: The latitude coordinates.
        lon_coords: The longitude coordinates.

    Returns:
        The latitude and longitude coordinates.
    """
    # Create output arrays with NaN
    lat_coords_out = np.full_like(lat_idx, np.nan)
    lon_coords_out = np.full_like(lon_idx, np.nan)

    # Find valid (non-NaN) indices
    valid_mask = ~(np.isnan(lat_idx) | np.isnan(lon_idx))

    if valid_mask.any():
        # Convert to integer indices only where valid
        int_lat_idx = np.where(valid_mask, lat_idx.astype(int), 0)
        int_lon_idx = np.where(valid_mask, lon_idx.astype(int), 0)

        # Use advanced indexing to get coordinates
        lat_coords_out[valid_mask] = lat_coords[int_lat_idx[valid_mask]]
        lon_coords_out[valid_mask] = lon_coords[int_lon_idx[valid_mask]]

    return lat_coords_out, lon_coords_out


def convert_day_yearofday_to_time(
    dataset: xr.Dataset | xr.DataArray, year: int
) -> xr.Dataset | xr.DataArray:
    """Convert dayofyear and hour to new time coordinate.

    Args:
        dataset: The input xarray dataset or dataarray.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset or dataarray with a new time coordinate.
    """
    stacked = dataset.stack(valid_time=("dayofyear", "hour"))
    times = pd.to_datetime(
        year * 1000 + stacked.dayofyear.values.astype(int), format="%Y%j"
    ) + pd.to_timedelta(stacked.hour.values.astype(int), unit="h")
    return stacked.reset_index("valid_time", drop=True).assign_coords(
        valid_time=("valid_time", times)
    )


_TIME_DIM_NAMES = frozenset({"time", "valid_time", "lead_time", "init_time"})
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_POINT_DIM_NAMES = ("location", "station", "sample", "stacked")


def _lat_lon_names(
    obj: xr.Dataset | xr.DataArray,
) -> tuple[str | None, str | None]:
    """Return latitude and longitude coordinate names if present."""
    names = list(obj.dims) + list(obj.coords)
    lat = next((n for n in _LAT_NAMES if n in names), None)
    lon = next((n for n in _LON_NAMES if n in names), None)
    return lat, lon


def infer_spatial_layout(
    obj: xr.Dataset | xr.DataArray,
) -> Literal["grid", "points"]:
    """Classify a Dataset or DataArray as a spatial grid or point samples.

    Points are paired (lat, lon) samples: a shared non-time dimension such
    as ``location``, or a sparse lat × lon cube of occupied stations.
    Independent latitude and longitude dimensions with dense data are a
    grid.

    Args:
        obj: Dataset or DataArray to classify.

    Returns:
        ``"points"`` if lat/lon are paired samples, otherwise ``"grid"``.
        Objects with no lat/lon coordinates are treated as ``"grid"``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from extremeweatherbench import utils
        >>> times = pd.date_range("2021-01-01", periods=2, freq="6h")
        >>> grid = xr.Dataset(
        ...     {"t": (["valid_time", "latitude", "longitude"], np.zeros((2, 3, 4)))},
        ...     coords={
        ...         "valid_time": times,
        ...         "latitude": [10.0, 11.0, 12.0],
        ...         "longitude": [100.0, 101.0, 102.0, 103.0],
        ...     },
        ... )
        >>> utils.infer_spatial_layout(grid)
        'grid'
        >>> pts = utils.point_frame_to_dataset(
        ...     pd.DataFrame(
        ...         {
        ...             "valid_time": [times[0], times[0]],
        ...             "latitude": [10.0, 12.0],
        ...             "longitude": [100.0, 103.0],
        ...             "t": [273.0, 275.0],
        ...         }
        ...     )
        ... )
        >>> utils.infer_spatial_layout(pts)
        'points'
    """
    lat_name, lon_name = _lat_lon_names(obj)
    if lat_name is None or lon_name is None:
        return "grid"

    arrays: list[xr.DataArray]
    if isinstance(obj, xr.DataArray):
        arrays = [obj]
    else:
        arrays = [obj[v] for v in obj.data_vars]

    for da in arrays:
        if (
            isinstance(da.data, sparse.COO)
            and lat_name in da.dims
            and lon_name in da.dims
        ):
            return "points"

    for dim in _POINT_DIM_NAMES:
        if dim not in obj.dims:
            continue
        if dim in obj[lat_name].dims and dim in obj[lon_name].dims:
            return "points"

    lat_dims = set(obj[lat_name].dims) - _TIME_DIM_NAMES
    lon_dims = set(obj[lon_name].dims) - _TIME_DIM_NAMES
    if lat_dims and lat_dims == lon_dims and lat_name not in obj.dims:
        return "points"

    return "grid"


def _point_dim_name(obj: xr.Dataset | xr.DataArray) -> str | None:
    """Name of the sample dimension for point-like data, if any."""
    for dim in _POINT_DIM_NAMES:
        if dim in obj.dims:
            return dim
    lat_name, lon_name = _lat_lon_names(obj)
    if lat_name is None:
        return None
    skip = _TIME_DIM_NAMES | {lat_name, lon_name}
    spatial = set(obj[lat_name].dims) - skip
    if len(spatial) == 1:
        return str(next(iter(spatial)))
    return None


def point_frame_to_dataset(
    df: pd.DataFrame,
    *,
    time_col: str = "valid_time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    location_dim: str = "location",
) -> xr.Dataset:
    """Convert station rows to (time, location) with lat/lon as coords.

    Each unique (latitude, longitude) pair becomes one location. This does
    not build a unique-lat × unique-lon Cartesian product, which would
    fill empty grid crossings and can exhaust memory.

    Args:
        df: Point observations with time, latitude, longitude, and value
            columns.
        time_col: Name of the valid-time column.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        location_dim: Name of the sample dimension to create.

    Returns:
        Dataset with dimensions ``(time_col, location_dim)``. Latitude
        and longitude are non-dimension coordinates on ``location_dim``.
        Duplicate (time, lat, lon) rows keep the first value.

    Examples:
        >>> import pandas as pd
        >>> from extremeweatherbench import utils
        >>> df = pd.DataFrame(
        ...     {
        ...         "valid_time": ["2021-02-01", "2021-02-01"],
        ...         "latitude": [10.0, 30.0],
        ...         "longitude": [20.0, 40.0],
        ...         "surface_air_temperature": [273.1, 268.4],
        ...     }
        ... )
        >>> ds = utils.point_frame_to_dataset(df)
        >>> list(ds.dims)
        ['valid_time', 'location']
        >>> ds.sizes["location"]
        2
    """
    frame = df.copy()
    key_cols = [time_col, lat_col, lon_col]
    frame = frame.drop_duplicates(subset=key_cols, keep="first")
    stations = frame[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    stations[location_dim] = np.arange(len(stations))
    frame = frame.merge(stations, on=[lat_col, lon_col], how="left")
    skip = {time_col, lat_col, lon_col, location_dim}
    value_cols = [c for c in frame.columns if c not in skip]
    indexed = frame.set_index([time_col, location_dim])[value_cols]
    ds = xr.Dataset.from_dataframe(indexed)
    stations = stations.set_index(location_dim)
    return ds.assign_coords(
        latitude=(location_dim, stations[lat_col].to_numpy()),
        longitude=(location_dim, stations[lon_col].to_numpy()),
    )


def sparse_cube_to_location(
    ds: xr.Dataset, location_dim: str = "location"
) -> xr.Dataset:
    """Collapse a sparse lat × lon cube onto occupied station pairs.

    Use this when point data was stored as a unique-lat × unique-lon
    product with values only at real stations. The result has a sample
    dimension instead of separate latitude and longitude dimensions.

    Args:
        ds: Dataset with latitude and longitude dimensions. Sparse COO
            variables are densified after occupied pairs are collected.
        location_dim: Name of the sample dimension to create. Defaults
            to ``"location"``.

    Returns:
        Dataset indexed by occupied (lat, lon) pairs along
        ``location_dim``, with latitude and longitude as coordinates.
        If lat/lon names are missing, ``ds`` is returned unchanged.
    """
    lat_name, lon_name = _lat_lon_names(ds)
    if lat_name is None or lon_name is None:
        return ds
    da = next(iter(ds.data_vars.values()))
    pairs: np.ndarray | None = None
    if isinstance(da.data, sparse.COO):
        dims = list(da.dims)
        lat_i = da.data.coords[dims.index(lat_name)]
        lon_i = da.data.coords[dims.index(lon_name)]
        pairs = np.unique(np.column_stack([lat_i, lon_i]), axis=0)
        ds = ds.map(
            lambda v: (
                v.copy(data=v.data.todense()) if isinstance(v.data, sparse.COO) else v
            )
        )
    if pairs is None:
        stacked = ds.stack({location_dim: (lat_name, lon_name)})
        keep = (
            stacked[da.name]
            .notnull()
            .any(dim=[d for d in stacked[da.name].dims if d != location_dim])
        )
        return stacked.isel({location_dim: keep})

    loc = xr.DataArray(np.arange(len(pairs)), dims=location_dim)
    out = ds.isel(
        {
            lat_name: xr.DataArray(pairs[:, 0], dims=location_dim),
            lon_name: xr.DataArray(pairs[:, 1], dims=location_dim),
        }
    )
    return out.assign_coords({location_dim: loc})


def as_point_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Return point data with a ``location`` dim and lat/lon coords.

    Accepts already-point Datasets (``location``, ``station``,
    ``sample``, or ``stacked``) or a sparse lat × lon cube of occupied
    stations. Grid Datasets are returned unchanged.

    Args:
        ds: Point or gridded Dataset.

    Returns:
        Point Dataset with a ``location`` dimension and ``latitude`` /
        ``longitude`` coordinates, or ``ds`` if it is a spatial grid.
    """
    if _point_dim_name(ds) is not None and infer_spatial_layout(ds) == "points":
        lat_name, lon_name = _lat_lon_names(ds)
        point_dim = _point_dim_name(ds)
        if point_dim != "location" and point_dim is not None:
            ds = ds.rename({point_dim: "location"})
        if lat_name and lat_name != "latitude":
            ds = ds.rename({lat_name: "latitude"})
        if lon_name and lon_name != "longitude":
            ds = ds.rename({lon_name: "longitude"})
        return ds
    if infer_spatial_layout(ds) == "points":
        return sparse_cube_to_location(ds)
    return ds


def sample_field_at_points(
    field: xr.Dataset,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    method: str = "nearest",
) -> xr.Dataset:
    """Sample a gridded field at paired latitude/longitude coordinates.

    Interpolating onto 1D lat and lon *dimension* coords builds a full
    lat × lon mesh. This instead evaluates the field at each (lat, lon)
    pair, so the output has a ``location`` dimension.

    Args:
        field: Gridded Dataset with latitude and longitude dimensions.
        latitude: Sample latitudes. Should share a ``location`` dim with
            ``longitude`` when already a DataArray.
        longitude: Sample longitudes paired with ``latitude``.
        method: Interpolation method passed to ``Dataset.interp``.
            ``"nearest"`` or ``"linear"``. Defaults to ``"nearest"``.

    Returns:
        ``field`` interpolated at the given pairs, with a ``location``
        dimension. If ``field`` has no lat/lon coordinates, it is
        returned unchanged.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from extremeweatherbench import utils
        >>> t2m = np.arange(6.0).reshape(2, 3)
        >>> forecast = xr.Dataset(
        ...     {"t2m": (("latitude", "longitude"), t2m)},
        ...     coords={
        ...         "latitude": [10.0, 20.0],
        ...         "longitude": [1.0, 2.0, 3.0],
        ...     },
        ... )
        >>> lat = xr.DataArray([10.0, 20.0], dims="location")
        >>> lon = xr.DataArray([1.0, 3.0], dims="location")
        >>> sampled = utils.sample_field_at_points(forecast, lat, lon)
        >>> list(sampled.dims)
        ['location']
    """
    lat_name, lon_name = _lat_lon_names(field)
    if lat_name is None or lon_name is None:
        return field
    interp_method: Literal["nearest", "linear"] = (
        "nearest" if method == "nearest" else "linear"
    )
    lat_da = latitude
    lon_da = longitude
    if "location" not in lat_da.dims:
        loc = np.arange(lat_da.size)
        lat_da = xr.DataArray(
            np.asarray(lat_da),
            dims="location",
            coords={"location": loc},
        )
        lon_da = xr.DataArray(
            np.asarray(lon_da),
            dims="location",
            coords={"location": loc},
        )
    return field.interp(
        {lat_name: lat_da, lon_name: lon_da},
        method=interp_method,
        kwargs={"fill_value": None},
    )


def _colocate_points(forecast_pts: xr.Dataset, target_pts: xr.Dataset) -> xr.Dataset:
    """Map forecast stations onto target stations by nearest lat/lon."""
    fc_dim = _point_dim_name(forecast_pts) or "location"
    tg_dim = _point_dim_name(target_pts) or "location"
    fc_lat = np.asarray(forecast_pts["latitude"])
    fc_lon = np.asarray(forecast_pts["longitude"])
    tg_lat = np.asarray(target_pts["latitude"])
    tg_lon = np.asarray(target_pts["longitude"])
    dist = (fc_lat[:, None] - tg_lat[None, :]) ** 2 + (
        fc_lon[:, None] - tg_lon[None, :]
    ) ** 2
    idx = dist.argmin(axis=0)
    mapped = forecast_pts.isel({fc_dim: idx})
    mapped = mapped.assign_coords(
        {
            tg_dim: target_pts[tg_dim].values,
            "latitude": target_pts["latitude"],
            "longitude": target_pts["longitude"],
        }
    )
    if fc_dim != tg_dim:
        mapped = mapped.rename({fc_dim: tg_dim})
    return mapped


def interp_climatology_to_target(
    target: xr.DataArray, climatology: xr.DataArray
) -> xr.DataArray:
    """Interpolate a climatology to a target data array.

    Args:
        target: The target data array to interpolate the climatology to.
        climatology: The climatology data array to interpolate.

    Returns:
        The interpolated climatology data array. If the target is sparse, the
        climatology is interpolated to the target coordinates. If the target is not
        sparse, the climatology is interpolated to the target coordinates.
    """
    point_dim = _point_dim_name(target)
    if point_dim is not None and "latitude" in target.coords:
        return climatology.interp(
            latitude=target["latitude"],
            longitude=target["longitude"],
            method="nearest",
            kwargs={"fill_value": None},
        )
    # If the target is sparse or has less than 3 dimensions, interpolate the
    # climatology using stacked dim
    if isinstance(target.data, sparse.COO) or target.ndim < 3:
        return climatology.interp(
            # stacked as a data variable is enforced by stack_dataarray_from_dims
            latitude=target["stacked"]["latitude"],
            longitude=target["stacked"]["longitude"],
            method="nearest",
            kwargs={"fill_value": None},
        )
    # Otherwise, interpolate the climatology to the target coordinates
    return climatology.interp_like(
        target, method="nearest", kwargs={"fill_value": None}
    )


def maybe_densify_dataarray(da: xr.DataArray, max_size: float = 1e9) -> xr.DataArray:
    """Densify a dataarray's data if it is sparse.

    Args:
        da: The xarray dataarray to densify.
        max_size: Max size for densification. Default is 1e9 to
            avoid issues with Dask and sparse.

    Returns:
        The densified xarray dataarray.
    """
    if isinstance(da.data, sparse.COO):
        # Assigning to da.data would rewrite the Variable this DataArray shares
        # with whatever it came from, so pulling a variable out of a dataset and
        # densifying it turned the dataset dense too.
        return da.copy(data=da.data.maybe_densify(max_size=max_size))
    return da


def reduce_dataarray(
    da: xr.DataArray,
    method: str | Callable,
    reduce_dims: list[str],
    compute: bool = True,
    **method_kwargs: Any,
) -> xr.DataArray:
    """Reduce using xarray methods or numpy functions.

    This function can utilize xarray's optimized methods (e.g., mean, sum) or
    numpy/callable reductions. Using the built-in methods xarray provides can be more
    efficient than using numpy functions.

    If compute is True, the dataarray will be computed before returning.
    This is useful to avoid dask exceptions when indexing with a boolean mask.

    Args:
        da: The xarray dataarray to reduce.
        method: Either an xarray method name (e.g., 'mean', 'sum') or
            a callable function (e.g., np.nanmean).
        reduce_dims: The dimensions to reduce.
        compute: Whether to compute the dataarray before returning. Defaults to True.

    Returns:
        The reduced xarray dataarray.
    """
    reduce_dims = list(reduce_dims)
    present = [d for d in reduce_dims if d in da.dims]
    if not present:
        alt = _point_dim_name(da)
        if alt is not None:
            reduce_dims = [alt]
        elif isinstance(da.data, sparse.COO):
            da = stack_dataarray_from_dims(da, reduce_dims)
            reduce_dims = ["stacked"]
    else:
        reduce_dims = present

    if (
        isinstance(da.data, sparse.COO)
        and reduce_dims != ["stacked"]
        and all(d in da.dims for d in reduce_dims)
    ):
        da = stack_dataarray_from_dims(da, reduce_dims)
        reduce_dims = ["stacked"]

    if callable(method):
        # Use numpy function or other callable (original behavior)
        return (
            da.reduce(method, dim=reduce_dims).compute()
            if compute
            else da.reduce(method, dim=reduce_dims)
        )
    elif isinstance(method, str):
        # Use xarray built-in method
        if not hasattr(da, method):
            raise ValueError(f"DataArray has no method '{method}'")

        method_func = getattr(da, method)
        return (
            method_func(dim=reduce_dims, **method_kwargs).compute()
            if compute
            else method_func(dim=reduce_dims, **method_kwargs)
        )
    else:
        raise TypeError(f"method must be str or callable, got {type(method)}")


def load_natural_earth_geometries(
    name: str,
    resolution: Literal["10m", "50m", "110m"] = "50m",
    category: str = "physical",
) -> list[shapely.geometry.base.BaseGeometry]:
    """Download and read a Natural Earth vector layer.

    The zipped shapefile is cached on disk by pooch, so repeat calls within and
    across sessions do not re-download it.

    Args:
        name: Natural Earth layer name, e.g. 'land', 'lakes', or 'ocean'.
        resolution: Natural Earth resolution ('10m', '50m', or '110m').
            Defaults to '50m'.
        category: Natural Earth category. Defaults to 'physical'.

    Returns:
        The layer's geometries in EPSG:4326.
    """
    path = pooch.retrieve(
        url=NATURAL_EARTH_URL.format(
            resolution=resolution, category=category, name=name
        ),
        known_hash=None,
        path=pooch.os_cache("extremeweatherbench"),
    )
    return list(gpd.read_file(f"zip://{path}").geometry)


def load_land_geometry(
    resolution: Literal["10m", "50m", "110m"] = "50m",
) -> shapely.geometry.Polygon:
    """Load the land geometry, excluding lakes and non-ocean bodies of water.

    Args:
        resolution: Natural Earth resolution ('10m', '50m', or '110m').
            Defaults to '50m'.

    Returns:
        The land geometry as a shapely Polygon with lakes and
        ocean-connected water bodies (bays, estuaries, seas) excluded.
    """
    land_geoms = load_natural_earth_geometries("land", resolution=resolution)
    land_union = shapely.ops.unary_union(land_geoms)

    # Exclude lakes to avoid false landfall detections
    try:
        lake_geoms = load_natural_earth_geometries("lakes", resolution=resolution)
        if lake_geoms:
            lakes_union = shapely.ops.unary_union(lake_geoms)
            land_union = land_union.difference(lakes_union)
    except (OSError, ValueError):
        pass

    # Exclude ocean-connected water bodies (bays, estuaries, coastal seas)
    try:
        ocean_union = load_ocean_geometry(resolution=resolution)
        land_union = land_union.difference(ocean_union)
    except (OSError, ValueError):
        pass

    return land_union


def load_ocean_geometry(
    resolution: Literal["10m", "50m", "110m"] = "50m",
) -> shapely.geometry.Polygon:
    """Load the ocean geometry from Natural Earth.

    Args:
        resolution: Natural Earth resolution ('10m', '50m', or '110m').
            Defaults to '50m'.

    Returns:
        The ocean geometry as a unified shapely Polygon.
    """
    ocean_geoms = load_natural_earth_geometries("ocean", resolution=resolution)
    return shapely.ops.unary_union(ocean_geoms)


def _cache_maybe_densify_helper(
    data: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Helper function to maybe densify a dataset's variables or dataarray for caching.

    Args:
        data: The dataset or dataarray to format.

    Returns:
        The formatted dataset or dataarray.
    """
    # If the data is a dataset, map the helper function to each dataarray
    if isinstance(data, xr.Dataset):
        return data.map(_cache_maybe_densify_helper)

    # If the data is a dataarray, densify the data if it is sparse
    elif isinstance(data, xr.DataArray):
        return maybe_densify_dataarray(da=data)

    # Otherwise, raise an error
    else:
        raise TypeError(f"data must be xr.Dataset or xr.DataArray, got {type(data)}")


def maybe_cache_and_compute(
    data: xr.Dataset | xr.DataArray,
    name: str,
    cache_dir: str | pathlib.Path | None = None,
) -> xr.Dataset | xr.DataArray:
    """Compute and cache datasets if cache_dir is provided.

    Data is returned as technically lazily loaded from the cache, but will significantly
    speed up subsequent computations with a copy of the data in memory. Note that if
    many cases or cases with large spatiotemporal domains are to be computed, it may be
    better to avoid caching with a limited disk size.

    Args:
        data: The dataset or dataarray to compute and cache.
        name: The name of the dataset for naming cached files.
        cache_dir: The directory to cache the datasets. If provided,
            datasets or dataarrays will be cached as zarrs and loaded from the cache.
            Default is None.

    Returns:
        The computed dataset if cache_dir is set, otherwise the
        original dataset.
    """
    # If no caching, return as dataset or dataarray
    if cache_dir is None:
        return data

    # Compute and cache dataset or dataarray
    logger.info("Computing datasets and storing at %s...", cache_dir)
    cache_path = pathlib.Path(cache_dir)

    # If the cache file does not exist, maybe densify the data and cache it. Sparse data
    # must be densified to be stored in zarrs
    if not (cache_path / f"{name}.zarr").exists():
        progress.set_phase(f"caching {name}")
        _cache_maybe_densify_helper(data).to_zarr(
            cache_path / f"{name}.zarr", zarr_format=2, mode="w"
        )

    # Load the data from the cache, matching the type of the input data
    if isinstance(data, xr.Dataset):
        return xr.open_dataset(cache_path / f"{name}.zarr")
    else:
        return xr.open_dataarray(cache_path / f"{name}.zarr")
