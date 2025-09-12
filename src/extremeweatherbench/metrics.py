import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import derived, evaluate, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global cache for transformed contingency managers
_GLOBAL_CONTINGENCY_CACHE = utils.ThreadSafeDict()  # type: ignore


def get_cached_transformed_manager(
    forecast: xr.Dataset,
    target: xr.Dataset,
    forecast_threshold: float = 0.5,
    target_threshold: float = 0.5,
    preserve_dims: str = "lead_time",
) -> cat.BasicContingencyManager:
    """Get cached transformed contingency manager, creating if needed.

    This function provides a global cache that can be used by any metric
    with the same thresholds and data, regardless of how the metrics are created.
    """
    # Create cache key from data content hash and parameters
    forecast_hash = hash(forecast.to_array().values.tobytes())
    target_hash = hash(target.to_array().values.tobytes())

    cache_key = (
        forecast_hash,
        target_hash,
        forecast_threshold,
        target_threshold,
        preserve_dims,
    )

    # Return cached result if available
    if cache_key in _GLOBAL_CONTINGENCY_CACHE:
        logger.info(f"Cache found for {cache_key}")
        return _GLOBAL_CONTINGENCY_CACHE[cache_key]

    # Apply thresholds to binarize the data
    binary_forecast = (forecast >= forecast_threshold).astype(float)
    binary_target = (target >= target_threshold).astype(float)

    # Create and transform contingency manager
    binary_contingency_manager = cat.BinaryContingencyManager(
        binary_forecast, binary_target
    )
    transformed = binary_contingency_manager.transform(preserve_dims=preserve_dims)

    # Cache the result
    _GLOBAL_CONTINGENCY_CACHE[cache_key] = transformed

    return transformed


def clear_contingency_cache():
    """Clear the global contingency manager cache."""
    global _GLOBAL_CONTINGENCY_CACHE
    _GLOBAL_CONTINGENCY_CACHE.clear()


class BaseMetric(ABC):
    """A BaseMetric class is an abstract class that defines the foundational interface
    for all metrics.

    Metrics are general operations applied between a forecast and analysis xarray
    dataset. EWB metrics prioritize the use of any arbitrary sets of forecasts and
    analyses, so long as the spatiotemporal dimensions are the same.
    """

    # default to preserving lead_time in EWB metrics
    name: str
    preserve_dims: str = "lead_time"

    def __init__(
        self,
        forecast_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        target_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
    ):
        self.forecast_variable = forecast_variable
        self.target_variable = target_variable
        # Check if both variables are None - this is allowed
        if self.forecast_variable is None and self.target_variable is None:
            pass
        # If only one is None, raise an error
        elif self.forecast_variable is None or self.target_variable is None:
            raise ValueError(
                "Both forecast_variable and target_variable must be provided, "
                "or both must be None"
            )
        else:
            # catch if the user provides a DerivedVariable object/class instead of a
            # string or not using the .name attribute
            self.forecast_variable = evaluate.maybe_convert_variable_to_string(
                self.forecast_variable
            )
            self.target_variable = evaluate.maybe_convert_variable_to_string(
                self.target_variable
            )

    @classmethod
    @abstractmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the base metric.
        """
        pass

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs,
    ) -> xr.DataArray:
        """Compute the metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        return cls._compute_metric(
            forecast,
            target,
            **kwargs,
        )


class AppliedMetric(ABC):
    """An applied metric is a wrapper around a BaseMetric.

    An AppliedMetric is a wrapper around a BaseMetric that is intended for more complex
    rollups or aggregations. Typically, these metrics are used for one event
    type and are very specific.

    Temporal onset mean error, case duration mean error, and maximum temperature mean
    absolute error are all examples of applied metrics.

    Attributes:
        base_metric: The BaseMetric to wrap.
        _compute_applied_metric: An abstract method to compute the inputs to the base
        metric.
        compute_applied_metric: A method to compute the metric.
    """

    base_metric: type[BaseMetric]

    @classmethod
    @abstractmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> dict[str, xr.DataArray]:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the applied metric.
        """
        pass

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        # first, compute the inputs to the base metric, a dictionary of forecast and
        # target
        applied_result = cls._compute_applied_metric(
            forecast,
            target,
            **utils.filter_kwargs_for_callable(kwargs, cls._compute_applied_metric),
        )
        # then, compute the base metric with the inputs
        return cls.base_metric.compute_metric(**applied_result)


class ThresholdMetric(BaseMetric):
    """Base class for threshold-based metrics.

    This class provides common functionality for metrics that require
    forecast and target thresholds for binarization.
    """

    def __init__(
        self,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        preserve_dims: str = "lead_time",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forecast_threshold = forecast_threshold
        self.target_threshold = target_threshold
        self.preserve_dims = preserve_dims

    def __call__(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the classmethod with the configured parameters
        return self.__class__.compute_metric(forecast, target, **kwargs)


class MAE(BaseMetric):
    name = "mae"

    @classmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return mae(forecast, target, preserve_dims=preserve_dims)


class ME(BaseMetric):
    name = "me"

    @classmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return mean_error(forecast, target, preserve_dims=preserve_dims)


class RMSE(BaseMetric):
    name = "rmse"

    @classmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return rmse(forecast, target, preserve_dims=preserve_dims)


class SpatialDisplacement(BaseMetric):
    name = "spatial_displacement"

    def __init__(
        self,
        forecast_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        target_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        forecast_mask_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        target_mask_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
    ):
        self.forecast_variable = forecast_variable
        self.target_variable = target_variable
        self.forecast_mask_variable = forecast_mask_variable
        self.target_mask_variable = target_mask_variable

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.Dataset, target: xr.Dataset, **kwargs: Any
    ) -> Any:
        from scipy.ndimage import center_of_mass, label

        # Get the masked data for target and forecast
        target_masked = target[cls.target_variable].where(
            target[cls.target_mask_variable], 0
        )
        forecast_masked = forecast[cls.forecast_variable].where(
            forecast[cls.forecast_mask_variable], 0
        )

        # Initialize arrays to store results
        lead_times = forecast.lead_time.values
        valid_times = forecast.valid_time.values

        target_lat_com = np.full((len(valid_times),), np.nan)
        target_lon_com = np.full((len(valid_times),), np.nan)
        forecast_lat_com = np.full((len(lead_times), len(valid_times)), np.nan)
        forecast_lon_com = np.full((len(lead_times), len(valid_times)), np.nan)

        # Iterate over all lead_time and valid_time combinations
        for lt_idx, lead_time in enumerate(forecast.lead_time):
            for vt_idx, valid_time in enumerate(forecast.valid_time):
                # Extract 2D slice for this time combination
                target_slice = target_masked.sel(valid_time=valid_time)
                forecast_slice = forecast_masked.sel(
                    lead_time=lead_time, valid_time=valid_time
                )

                # Label connected components and find center of mass
                target_labels, _ = label(target_slice.values > 0)
                forecast_labels, _ = label(forecast_slice.values > 0)

                if target_labels.max() > 0:
                    target_com = center_of_mass(target_slice.values, target_labels, 1)
                    # Convert indices to actual coordinates
                    target_lat_com[vt_idx] = target_slice.latitude.values[
                        int(target_com[0])
                    ]
                    target_lon_com[vt_idx] = target_slice.longitude.values[
                        int(target_com[1])
                    ]

                if forecast_labels.max() > 0:
                    forecast_com = center_of_mass(
                        forecast_slice.values, forecast_labels, 1
                    )
                    # Convert indices to actual coordinates
                    forecast_lat_com[lt_idx, vt_idx] = forecast_slice.latitude.values[
                        int(forecast_com[0])
                    ]
                    forecast_lon_com[lt_idx, vt_idx] = forecast_slice.longitude.values[
                        int(forecast_com[1])
                    ]

        # Create properly structured datasets with lead_time and valid_time dimensions
        target_com = xr.Dataset(
            {
                "latitude": (["valid_time"], target_lat_com),
                "longitude": (["valid_time"], target_lon_com),
            },
            coords={"valid_time": valid_times},
        )

        forecast_com = xr.Dataset(
            {
                "latitude": (["lead_time", "valid_time"], forecast_lat_com),
                "longitude": (["lead_time", "valid_time"], forecast_lon_com),
            },
            coords={"lead_time": lead_times, "valid_time": valid_times},
        )
        return utils.calculate_haversine_distance(forecast_com, target_com)


class MaximumMAE(AppliedMetric):
    base_metric = MAE

    name = "maximum_mae"

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs,
    ) -> dict[str, xr.DataArray]:
        forecast = forecast.compute()
        target_spatial_mean = target.compute().mean(["latitude", "longitude"])
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        )
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        filtered_max_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= maximum_timestep - np.timedelta64(tolerance_range // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= maximum_timestep + np.timedelta64(tolerance_range // 2, "h")
            ),
            drop=True,
        ).max("valid_time")
        return {
            "forecast": filtered_max_forecast,
            "target": maximum_value,
            "preserve_dims": cls.base_metric.preserve_dims,
        }


class MinimumMAE(AppliedMetric):
    base_metric = MAE

    name = "minimum_mae"

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs: Any,
    ) -> Any:
        forecast = forecast.compute()
        target_spatial_mean = target.compute().mean(["latitude", "longitude"])
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
        filtered_min_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= minimum_timestep - np.timedelta64(tolerance_range // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= minimum_timestep + np.timedelta64(tolerance_range // 2, "h")
            ),
            drop=True,
        ).min("valid_time")
        return {
            "forecast": filtered_min_forecast,
            "target": minimum_value,
            "preserve_dims": cls.base_metric.preserve_dims,
        }


class MaxMinMAE(AppliedMetric):
    base_metric = MAE

    name = "max_min_mae"

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs: Any,
    ) -> Any:
        forecast = forecast.mean(
            [
                dim
                for dim in forecast.dims
                if dim not in ["valid_time", "lead_time", "time"]
            ]
        )
        target = target.mean(
            [
                dim
                for dim in target.dims
                if dim not in ["valid_time", "lead_time", "time"]
            ]
        )
        time_resolution_hours = utils.determine_temporal_resolution(target)
        max_min_target_value = (
            target.groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present,
                time_resolution_hours=time_resolution_hours,
            )
            .max()
        )
        max_min_target_datetime = target.where(
            target == max_min_target_value, drop=True
        ).valid_time

        # Handle the case where there are >1 resulting target values
        max_min_target_datetime = (
            utils.maybe_get_closest_timestamp_to_center_of_valid_times(
                max_min_target_datetime, target.valid_time
            )
        )
        max_min_target_value = target.sel(valid_time=max_min_target_datetime)
        subset_forecast = (
            forecast.where(
                (
                    forecast.valid_time
                    >= (
                        max_min_target_datetime
                        - np.timedelta64(tolerance_range // 2, "h")
                    )
                )
                & (
                    forecast.valid_time
                    <= (
                        max_min_target_datetime
                        + np.timedelta64(tolerance_range // 2, "h")
                    )
                ),
                drop=True,
            )
            .groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present_forecast,
                time_resolution_hours=utils.determine_temporal_resolution(forecast),
            )
            .min("dayofyear")
        )

        return {
            "forecast": subset_forecast,
            "target": max_min_target_value,
            "preserve_dims": cls.base_metric.preserve_dims,
        }


class OnsetME(AppliedMetric):
    base_metric = ME
    preserve_dims: str = "init_time"
    name = "onset_me"

    @staticmethod
    def onset(forecast: xr.DataArray, **kwargs) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            # get the forecast resolution hours from the kwargs, otherwise default to 6
            num_timesteps = 24 // kwargs.get("forecast_resolution_hours", 6)
            if num_timesteps is None:
                return xr.DataArray(np.datetime64("NaT", "ns"))
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                time_resolution_hours=utils.determine_temporal_resolution(forecast),
            )
            if min_daily_vals.size >= 2:  # Check if we have at least 2 values
                for i in range(min_daily_vals.size - 1):
                    # TODO: CHANGE LOGIC; define forecast heatwave onset
                    if min_daily_vals[i] >= 288.15 and min_daily_vals[i + 1] >= 288.15:
                        return xr.DataArray(
                            forecast.where(
                                forecast["valid_time"].dt.dayofyear
                                == min_daily_vals.dayofyear[i],
                                drop=True,
                            )
                            .valid_time[0]
                            .values
                        )
        return xr.DataArray(np.datetime64("NaT", "ns"))

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        target_time = target.valid_time[0] + np.timedelta64(48, "h")
        forecast = (
            forecast.mean(["latitude", "longitude"]).groupby("init_time").map(cls.onset)
        )
        return {
            "forecast": forecast,
            "target": target_time,
            "preserve_dims": cls.preserve_dims,
        }


class DurationME(AppliedMetric):
    base_metric = ME

    preserve_dims: str = "init_time"

    name = "duration_me"

    @staticmethod
    def duration(forecast: xr.DataArray, **kwargs) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            # get the forecast resolution hours from the kwargs, otherwise default to 6
            num_timesteps = 24 // kwargs.get("forecast_resolution_hours", 6)
            if num_timesteps is None:
                return xr.DataArray(np.datetime64("NaT", "ns"))
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                time_resolution_hours=utils.determine_temporal_resolution(forecast),
            )
            # need to determine logic for 2+ consecutive days to find the date
            # that the heatwave starts
            if min_daily_vals.size >= 2:  # Check if we have at least 2 values
                for i in range(min_daily_vals.size - 1):
                    if min_daily_vals[i] >= 288.15 and min_daily_vals[i + 1] >= 288.15:
                        consecutive_days = np.timedelta64(
                            2, "D"
                        )  # Start with 2 since we found first pair
                        for j in range(i + 2, min_daily_vals.size):
                            if min_daily_vals[j] >= 288.15:
                                consecutive_days += np.timedelta64(1, "D")
                            else:
                                break
                        return xr.DataArray(consecutive_days.astype("timedelta64[ns]"))
        return xr.DataArray(np.timedelta64("NaT", "ns"))

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(
                cls.duration,
                forecast_resolution_hours=forecast.attrs["forecast_resolution_hours"],
            )
        )
        return {
            "forecast": forecast,
            "target": target_duration,
            "preserve_dims": cls.preserve_dims,
        }


# TODO: complete lead time detection implementation
class LeadTimeDetection(AppliedMetric):
    base_metric = MAE
    name = "lead_time_detection"
    preserve_dims: str = "init_time"

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        return {
            "forecast": forecast,
            "target": target,
            "preserve_dims": cls.preserve_dims,
            "threshold": 1,
            "variable": "atmospheric_river_land_intersection",
            "comparison": "==",
            "spatial_aggregation": "any",
        }
