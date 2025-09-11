"""Utility functions for the ExtremeWeatherBench package that don't fit into any other
specialized package."""

import datetime
import inspect
import logging
import threading
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import regionmask
import xarray as xr
import yaml  # type: ignore[import]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ThreadSafeDict:
    """A thread-safe dictionary implementation using locks.

    This class provides a thread-safe wrapper around a standard dictionary,
    ensuring atomic operations for getting, setting, and deleting items.
    Useful for caching data that needs to be shared between threads safely.
    """

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def keys(self):
        with self._lock:
            return list(self._data.keys())
            # Return a copy to prevent concurrent modification issues during
            # iteration

    def values(self):
        with self._lock:
            return list(self._data.values())  # Return a copy

    def items(self):
        with self._lock:
            return list(self._data.items())  # Return a copy


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

    events_yaml_file = resources.files(extremeweatherbench.data).joinpath("events.yaml")
    with resources.as_file(events_yaml_file) as file:
        yaml_event_case = read_event_yaml(file)

    return yaml_event_case


def read_event_yaml(input_pth: str | Path) -> dict:
    """Read events yaml from data."""
    logger.warning(
        "This function is deprecated and will be removed in a future release. "
        "Please use cases.read_incoming_yaml instead."
    )
    input_pth = Path(input_pth)
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
    valid_time_mask = (valid_times_reshaped > pd.to_datetime(start_date)) & (
        valid_times_reshaped < pd.to_datetime(end_date)
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
