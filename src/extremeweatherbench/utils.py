"""Utility functions for the ExtremeWeatherBench package that don't fit into any other
specialized package."""

import datetime
import importlib
import inspect
import logging
import operator
import pathlib
from typing import Any, Callable, Literal, Optional, Sequence, Union

import cartopy.io.shapereader as shpreader
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import regionmask
import shapely
import sparse
import tqdm
import xarray as xr
import yaml  # type: ignore[import]
from joblib import Parallel

logger = logging.getLogger(__name__)

operators = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def maybe_get_operator(
    operator_method: Union[Literal[">", ">=", "<", "<=", "==", "!="], Callable],
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
    # Find common init_times between forecast and target
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
    # Check that we have actual data (not all NaN)
    if np.isnan(landfall.values).all():
        return False
    return True


def _create_nan_dataarray(
    preserved_dims: Union[str, list[str]] = "init_time",
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
    longitude: float | Union[xr.Dataset, xr.DataArray],
    longitude_name: str = "longitude",
) -> float | Union[xr.Dataset, xr.DataArray]:
    """Convert a longitude from the range [0, 360) to [-180, 180).

    Datasets are coerced to [-180, 180) and sorted by longitude.
    """
    if isinstance(longitude, xr.Dataset) or isinstance(longitude, xr.DataArray):
        longitude.coords[longitude_name] = (
            longitude.coords[longitude_name] + 180
        ) % 360 - 180
        longitude = longitude.sortby(longitude_name)
        return longitude
    else:
        return np.mod(longitude - 180, 360) - 180


def remove_ocean_gridpoints(dataset: xr.Dataset) -> xr.Dataset:
    """Subset a dataset to only include land gridpoints based on a land-sea mask.

    Args:
        dataset: The input xarray dataset.

    Returns:
        The dataset masked to only land gridpoints.
    """
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_sea_mask = land.mask(dataset.longitude, dataset.latitude)
    land_mask = land_sea_mask == 0
    # Subset the dataset to only include land gridpoints
    return dataset.where(land_mask)


def load_events_yaml():
    """Load the events yaml file."""
    logger.warning(
        "This function is deprecated and will be removed in a future release. "
        "Please use cases.load_ewb_events_yaml_into_case_collection instead."
    )
    import extremeweatherbench.data

    events_yaml_file = importlib.resources.files(extremeweatherbench.data).joinpath(
        "events.yaml"
    )
    with importlib.resources.as_file(events_yaml_file) as file:
        yaml_event_case = read_event_yaml(file)

    return yaml_event_case


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

    Args:
        da: The input DataArray.

    Returns:
        The minimum value of the DataArray if all timesteps are present,
        otherwise the original DataArray.
    """
    timesteps_per_day = 24 / time_resolution_hours
    if da.values.size == timesteps_per_day:
        return da.min()
    else:
        return xr.DataArray(np.nan)


def min_if_all_timesteps_present_forecast(
    da: xr.DataArray, time_resolution_hours: float
) -> xr.DataArray:
    """Return the minimum value of a DataArray if all timesteps of a day are present
    given a dataset with lead_time and valid_time dimensions.

    Args:
        da: The input DataArray.

    Returns:
        The minimum value of the DataArray if all timesteps are present,
        otherwise the original DataArray.
    """
    timesteps_per_day = 24 / time_resolution_hours
    if da.valid_time.size == timesteps_per_day:
        return da.min("valid_time")
    else:
        # Return an array with the same lead_time dimension but filled with NaNs
        return xr.DataArray(
            np.full(da.lead_time.size, np.nan),
            coords={"lead_time": da.lead_time},
            dims=["lead_time"],
        )


def determine_temporal_resolution(
    data: xr.Dataset | xr.DataArray,
) -> Optional[float]:
    """Determine the temporal resolution of the data.

    Args:
        data: The input dataset with a valid_time dimension or coordinate.

    Returns:
        The temporal resolution of the data as a float in hours.
    """
    num_timesteps = (
        np.unique(np.diff(data.valid_time)).astype("timedelta64[h]").astype(int)
    )
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


def convert_init_time_to_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """Convert the init_time coordinate to a valid_time coordinate.

    Args:
        ds: The dataset to convert with lead_time and init_time coordinates.

    Returns:
        The dataset with a valid_time coordinate.
    """
    valid_time = xr.DataArray(
        ds.init_time, coords={"init_time": ds.init_time}
    ) + xr.DataArray(ds.lead_time, coords={"lead_time": ds.lead_time})
    ds = ds.assign_coords(valid_time=valid_time)
    return xr.concat(
        [
            ds.sel(lead_time=lead).swap_dims({"init_time": "valid_time"})
            for lead in ds.lead_time
        ],
        "lead_time",
        coords="different",
        compat="equals",
        join="outer",
    )


def convert_valid_time_to_init_time(da: xr.DataArray) -> xr.DataArray:
    """Convert the valid_time coordinate to a init_time coordinate.

    Args:
        ds: The dataset to convert with lead_time and valid_time coordinates.

    Returns:
        The dataset with a init_time coordinate.
    """
    init_time = xr.DataArray(
        da.valid_time, coords={"valid_time": da.valid_time}
    ) - xr.DataArray(da.lead_time, coords={"lead_time": da.lead_time})
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
    coords: Optional[npt.NDArray] = None,
) -> xr.DataArray:
    """Stack sparse data with n-dimensions.

    In cases where sparse.COO data is in da.data, this function will stack the
    dimensions and return a densified dataarray using reduce_dims.

    Args:
        da: An xarray dataarray with sparse.COO data
        reduce_dims: The dimensions to reduce.
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


def check_for_vars(variable_list: list[str], source: Sequence) -> Optional[str]:
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
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
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
    dataset: Union[xr.Dataset, xr.DataArray], year: int
) -> Union[xr.Dataset, xr.DataArray]:
    """Convert dayofyear and hour to new time coordinate.

    Args:
        dataset: The input xarray dataset or dataarray.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset or dataarray with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f"{year}-01-01",
        periods=len(dataset["dayofyear"]) * len(dataset["hour"]),
        freq="6h",
    )
    dataset = dataset.stack(valid_time=("dayofyear", "hour")).drop_vars(
        ["dayofyear", "hour"]
    )
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(valid_time=time_dim)

    return dataset


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
        da.data = da.data.maybe_densify(max_size=max_size)
    return da


def reduce_dataarray(
    da: xr.DataArray,
    method: str | Callable,
    reduce_dims: list[str],
    compute: bool = True,
    **method_kwargs,
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
        **method_kwargs: Additional kwargs for the method. Only used
            when method is a string (xarray method).

    Returns:
        The reduced xarray dataarray.
    """
    if isinstance(da.data, sparse.COO):
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


def load_land_geometry(resolution: str = "10m") -> shapely.geometry.Polygon:
    """Load the land geometry, excluding lakes.

    Returns:
        The land geometry as a shapely Polygon with lakes excluded.
    """
    land = shpreader.natural_earth(
        category="physical", name="land", resolution=resolution
    )
    land_geoms = list(shpreader.Reader(land).geometries())
    land_union = shapely.ops.unary_union(land_geoms)

    # Exclude lakes to avoid false landfall detections in lakes/bays
    try:
        lakes = shpreader.natural_earth(
            category="physical", name="lakes", resolution=resolution
        )
        lake_geoms = list(shpreader.Reader(lakes).geometries())
        if lake_geoms:
            lakes_union = shapely.ops.unary_union(lake_geoms)
            land_union = land_union.difference(lakes_union)
    except (OSError, ValueError):
        # If lakes dataset is not available, return land without lakes
        # This can happen if the dataset doesn't exist for the resolution
        pass

    return land_union
