import abc
import logging
from typing import Any, Callable, Optional, Type

import numpy as np
import scores
import sparse
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc, derived, utils

logger = logging.getLogger(__name__)


# Global cache for transformed contingency managers
_GLOBAL_CONTINGENCY_CACHE = utils.ThreadSafeDict()  # type: ignore


def get_cached_transformed_manager(
    forecast: xr.DataArray,
    target: xr.DataArray,
    forecast_threshold: float = 0.5,
    target_threshold: float = 0.5,
    preserve_dims: str = "lead_time",
) -> scores.categorical.BasicContingencyManager:
    """Get cached transformed contingency manager, creating if needed.

    This function provides a global cache that can be used by any metric
    with the same thresholds and data, regardless of how the metrics are created.
    """
    # Create cache key from data content hash and parameters
    forecast_hash = hash(forecast.data.tobytes())
    target_hash = hash(target.data.tobytes())

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
    binary_contingency_manager = scores.categorical.BinaryContingencyManager(
        binary_forecast, binary_target
    )
    transformed = binary_contingency_manager.transform(preserve_dims=preserve_dims)

    # Cache the result
    _GLOBAL_CONTINGENCY_CACHE[cache_key] = transformed

    return transformed


def _reduce_duck_array(
    da: xr.DataArray, func: Callable, reduce_dims: list[str]
) -> xr.DataArray:
    """Reduce the duck array of the data.

    Some data will return as a sparse array, which can also be reduced but
    requires some additional logic.

    Args:
        da: The xarray dataarray to reduce.
        func: The function to reduce the data.
        reduce_dims: The dimensions to reduce.

    Returns:
        The reduced xarray dataarray.
    """
    if isinstance(da.data, sparse.COO):
        da = utils.stack_sparse_data_from_dims(da, reduce_dims)
        # Apply the reduce function to the data
        return da.reduce(func, dim="stacked")
    else:
        # Handles np.ndarray, dask.array, and other duck arrays
        return da.reduce(func, dim=reduce_dims)


def clear_contingency_cache():
    """Clear the global contingency manager cache."""
    global _GLOBAL_CONTINGENCY_CACHE
    _GLOBAL_CONTINGENCY_CACHE.clear()


class ComputeDocstringMetaclass(abc.ABCMeta):
    """A metaclass that maps the docstring from self._compute_metric() to
    self.compute_metric().

    The `BaseMetric` abstract base class requires users to override a function called
    `_compute_metric()`, while providing a standardized public interface to this method
    called `compute_metric()`. This metaclass automatically maps the docstring from
    `_compute_metric()` to `compute_metric()` so that the documentation a user provides
    for their implementation will automatically appear with the public interface without
    any additional effort.
    """

    def __new__(cls, name, bases, namespace):
        cls = super().__new__(cls, name, bases, namespace)
        # NOTE: the `compute_metric()` method will be defined in the ABC `BaseMetric`,
        # and we never expect the user re-implement it. So it won't be in the namespace
        # of the concrete metric classes - it will only be in the namespace of the ABC
        # `BaseMetric`, and will be available as an attribute of the concrete metric
        # classes.
        if "_compute_metric" in namespace and hasattr(cls, "compute_metric"):
            # Transfer the docstring from _compute_metric to compute_metric, if the
            # former exists.
            if cls._compute_metric.__doc__ is not None:
                # Create a new method for _this_ class, so we can avoid overwriting what
                # we set for the parent.
                _original_compute_metric = cls.compute_metric

                def _compute_metric_with_docstring(self, *args, **kwargs):
                    return _original_compute_metric(self, *args, **kwargs)

                _compute_metric_with_docstring.__doc__ = cls._compute_metric.__doc__
                cls.compute_metric = _compute_metric_with_docstring

        return cls


class BaseMetric(abc.ABC, metaclass=ComputeDocstringMetaclass):
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
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
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
            # Convert DerivedVariable object/class to string using .name
            self.forecast_variable = derived._maybe_convert_variable_to_string(
                self.forecast_variable
            )
            self.target_variable = derived._maybe_convert_variable_to_string(
                self.target_variable
            )

    @abc.abstractmethod
    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        All implementations must accept **kwargs to handle extra
        parameters gracefully, even if they don't use them.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional parameters. Common ones include preserve_dims
                (dimension(s) to preserve, defaults to "lead_time").

        Returns:
            The computed metric result.
        """
        pass

    def compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Public interface to compute the metric.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments to pass to the
                metric implementation.

        Returns:
            The computed metric result.
        """
        return self._compute_metric(forecast, target, **kwargs)


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

    def __call__(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs):
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the instance method with the configured parameters
        return self.compute_metric(forecast, target, **kwargs)


class CSI(ThresholdMetric):
    """Critical Success Index metric."""

    name = "critical_success_index"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        return transformed.critical_success_index()


class FAR(ThresholdMetric):
    """False Alarm Ratio metric."""

    name = "false_alarm_ratio"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        return transformed.false_alarm_ratio()


class TP(ThresholdMetric):
    """True Positive metric."""

    name = "true_positive"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        counts = transformed.get_counts()
        return counts["tp_count"] / counts["total_count"]


class FP(ThresholdMetric):
    """False Positive metric."""

    name = "false_positive"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        counts = transformed.get_counts()
        return counts["fp_count"] / counts["total_count"]


class TN(ThresholdMetric):
    """True Negative metric."""

    name = "true_negative"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        counts = transformed.get_counts()
        return counts["tn_count"] / counts["total_count"]


class FN(ThresholdMetric):
    """False Negative metric."""

    name = "false_negative"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        counts = transformed.get_counts()
        return counts["fn_count"] / counts["total_count"]


class Accuracy(ThresholdMetric):
    """Accuracy metric."""

    name = "accuracy"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        forecast_threshold = kwargs.get("forecast_threshold", 0.5)
        target_threshold = kwargs.get("target_threshold", 0.5)
        preserve_dims = kwargs.get("preserve_dims", "lead_time")

        transformed = get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        return transformed.accuracy()


def create_threshold_metrics(
    forecast_threshold: float = 0.5,
    target_threshold: float = 0.5,
    preserve_dims: str = "lead_time",
    metrics: Optional[list[str]] = None,
):
    """Create multiple threshold-based metrics with the specified thresholds.

    Args:
        forecast_threshold: Threshold for binarizing forecast data
        target_threshold: Threshold for binarizing target data
        preserve_dims: Dimensions to preserve during contingency table computation
        metrics: List of metric names to create (e.g., ['CSI', 'FAR', 'TP'])

    Returns:
        A list of metric objects with the specified thresholds
    """
    if metrics is None:
        metrics = ["CSI", "FAR", "Accuracy", "TP", "FP", "TN", "FN"]

    # Mapping of metric names to their classes (all threshold-based metrics)
    metric_classes: dict[str, Type[ThresholdMetric]] = {
        "CSI": CSI,
        "FAR": FAR,
        "Accuracy": Accuracy,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
    }

    result_metrics = []

    # Create metrics by instantiating their classes
    for metric_name in metrics:
        if metric_name not in metric_classes:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric_class = metric_classes[metric_name]
        metric = metric_class(
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        result_metrics.append(metric)

    return result_metrics


class MAE(BaseMetric):
    name = "mae"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Mean Absolute Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".

        Returns:
            The computed Mean Absolute Error result.
        """
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return scores.continuous.mae(forecast, target, preserve_dims=preserve_dims)


class ME(BaseMetric):
    """Mean Error (bias) metric."""

    name = "me"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Mean Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".

        Returns:
            The computed Mean Error result.
        """
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return scores.continuous.mean_error(
            forecast, target, preserve_dims=preserve_dims
        )


class RMSE(BaseMetric):
    """Root Mean Square Error metric."""

    name = "rmse"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the Root Mean Square Error.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".

        Returns:
            The computed Root Mean Square Error result.
        """
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return scores.continuous.rmse(forecast, target, preserve_dims=preserve_dims)


class Signal(BaseMetric):
    """Metric to detect signal presence in forecast data.

    Returns a boolean DataArray indicating whether threshold criteria are met
    for each (init_time, lead_time) combination. The metric is flexible for
    different signal detection criteria specified via threshold, comparison,
    and spatial_aggregation parameters.
    """

    name = "signal"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = kwargs.get("threshold", 0.5)
        self.comparison = kwargs.get("comparison", ">=")
        self.spatial_aggregation = kwargs.get("spatial_aggregation", "any")

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataset with init_time, lead_time, valid_time
            target: The target dataset (used for reference/validation)
            threshold: Threshold value for signal detection
            comparison: Comparison operator (">=", "<=", ">", "<", "==", "!=")
            spatial_aggregation: How to aggregate spatially ("any", "all", "mean")
            **kwargs: Additional arguments

        Returns:
            Boolean DataArray with dims [init_time, lead_time] indicating
            whether criteria are met for each init_time and lead_time pair.
        """
        if self.threshold is None:
            # Return False for all when no detection criteria specified
            dims = ["init_time", "lead_time"]
            coords = {
                "init_time": forecast.valid_time - forecast.lead_time,
                "lead_time": forecast.lead_time,
            }
            if "valid_time" in forecast.dims:
                dims.append("valid_time")
                coords["valid_time"] = forecast.valid_time
            return xr.DataArray(
                False,
                dims=dims,
                coords=coords,
                name="early_signal",
            )

        # Apply threshold comparison
        comparison_ops = {
            ">=": lambda x, t: x >= t,
            "<=": lambda x, t: x <= t,
            ">": lambda x, t: x > t,
            "<": lambda x, t: x < t,
            "==": lambda x, t: x == t,
            "!=": lambda x, t: x != t,
        }

        if self.comparison not in comparison_ops:
            raise ValueError(f"Comparison '{self.comparison}' not supported")

        # Create detection mask
        detection_mask = comparison_ops[self.comparison](forecast, self.threshold)

        # Apply spatial aggregation
        spatial_dims = [
            dim
            for dim in detection_mask.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]

        if spatial_dims:
            if self.spatial_aggregation == "any":
                detection_mask = detection_mask.any(spatial_dims)
            elif self.spatial_aggregation == "all":
                detection_mask = detection_mask.all(spatial_dims)
            elif self.spatial_aggregation == "mean":
                detection_mask = detection_mask.mean(spatial_dims) > 0.5
            else:
                raise ValueError(
                    f"Spatial aggregation '{self.spatial_aggregation}' not supported"
                )

        detection_mask.name = "early_signal"
        return detection_mask


class MaximumMAE(MAE):
    name = "MaximumMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> dict[str, xr.DataArray]:
        """Compute MaximumMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".
                tolerance_range (int): Time window (hours) around target's maximum value
                to search for forecast maximum. Defaults to 24 hours.

        Returns:
            MAE of the maximum values.
        """
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        tolerance_range = kwargs.get("tolerance_range", 24)
        target_spatial_mean = _reduce_duck_array(
            target, func=np.nanmean, reduce_dims=["latitude", "longitude"]
        )
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        ).compute()
        forecast_spatial_mean = _reduce_duck_array(
            forecast, func=np.nanmean, reduce_dims=["latitude", "longitude"]
        )
        filtered_max_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= maximum_timestep.data - np.timedelta64(tolerance_range // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= maximum_timestep.data + np.timedelta64(tolerance_range // 2, "h")
            ),
            drop=True,
        ).max("valid_time")
        return super()._compute_metric(
            forecast=filtered_max_forecast,
            target=maximum_value,
            preserve_dims=preserve_dims,
        )


class MinimumMAE(MAE):
    """MAE of the minimum value in a tolerance window.

    Computes the MAE between the forecast and target minimum
    values, where the forecast is filtered to a time window
    around the target's minimum.
    """

    name = "MinimumMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MinimumMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".
                tolerance_range (int): Time window (hours) around target's minimum
                value to search for forecast minimum. Defaults to 24 hours.

        Returns:
            MAE of the minimum values.
        """
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        tolerance_range = kwargs.get("tolerance_range", 24)
        target_spatial_mean = _reduce_duck_array(
            target, func=np.nanmean, reduce_dims=["latitude", "longitude"]
        )
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = _reduce_duck_array(
            forecast, func=np.nanmean, reduce_dims=["latitude", "longitude"]
        )
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
        filtered_min_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= minimum_timestep.data - np.timedelta64(tolerance_range // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= minimum_timestep.data + np.timedelta64(tolerance_range // 2, "h")
            ),
            drop=True,
        ).min("valid_time")
        return super()._compute_metric(
            forecast=filtered_min_forecast,
            target=minimum_value,
            preserve_dims=preserve_dims,
        )


class MaxMinMAE(MAE):
    """MAE of the maximum of daily minimum values.

    Computes the MAE between the warmest nighttime (daily minimum)
    temperature in the target and forecast, commonly used for
    heatwave evaluation.
    """

    name = "MaxMinMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MaxMinMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".
                tolerance_range (int): Time window (hours) around target's max-min
                value to search for forecast max-min. Defaults to 24 hours.

        Returns:
            MAE of the maximum daily minimum values.
        """
        reduce_dims = [
            dim
            for dim in forecast.dims
            if dim not in ["valid_time", "lead_time", "time"]
        ]
        forecast = _reduce_duck_array(
            forecast, func=np.nanmean, reduce_dims=reduce_dims
        )
        target = _reduce_duck_array(target, func=np.nanmean, reduce_dims=reduce_dims)

        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        tolerance_range = kwargs.get("tolerance_range", 24)
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
        subset_forecast = (
            forecast.where(
                (
                    forecast.valid_time
                    >= (
                        max_min_target_datetime.data
                        - np.timedelta64(tolerance_range // 2, "h")
                    )
                )
                & (
                    forecast.valid_time
                    <= (
                        max_min_target_datetime.data
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

        return super()._compute_metric(
            forecast=subset_forecast,
            target=max_min_target_value,
            preserve_dims=preserve_dims,
        )


class OnsetME(ME):
    """Mean error of heatwave onset time.

    Computes the mean error between forecast and observed timing
    of event onset (currently configured for heatwaves).
    """

    name = "OnsetME"

    def onset(self, forecast: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        """Identify onset time from forecast data.

        Args:
            forecast: The forecast DataArray.

        Returns:
            DataArray containing the onset datetime, or NaT if
            onset criteria not met.
        """
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

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute OnsetME.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "init_time".

        Returns:
            Mean error of onset timing.
        """
        preserve_dims = kwargs.get("preserve_dims", "init_time")

        target_time = target.valid_time[0] + np.timedelta64(48, "h")
        forecast = (
            _reduce_duck_array(
                forecast, func=np.nanmean, reduce_dims=["latitude", "longitude"]
            )
            .groupby("init_time")
            .map(self.onset)
        )
        return super()._compute_metric(
            forecast=forecast,
            target=target_time,
            preserve_dims=preserve_dims,
        )


class DurationME(ME):
    """Mean error of event duration.

    Computes the mean error between forecast and observed duration
    of an event (currently configured for heatwaves).
    """

    name = "DurationME"

    def duration(self, forecast: xr.DataArray, **kwargs: Any) -> xr.DataArray:
        """Calculate event duration from forecast data.

        Args:
            forecast: The forecast DataArray.

        Returns:
            DataArray containing the duration as timedelta, or
            NaT if duration criteria not met.
        """
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

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute DurationME.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "init_time".

        Returns:
            Mean error of event duration.
        """
        preserve_dims = kwargs.get("preserve_dims", "init_time")

        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            _reduce_duck_array(
                forecast, func=np.nanmean, reduce_dims=["latitude", "longitude"]
            )
            .groupby("init_time")
            .map(
                self.duration,
            )
        )
        return super()._compute_metric(
            forecast=forecast,
            target=target_duration,
            preserve_dims=preserve_dims,
        )


# TODO: fill landfall displacement out
class LandfallDisplacement(BaseMetric):
    """Spatial displacement error of landfall location.

    Note: Not yet implemented.
    """

    name = "LandfallDisplacement"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LandfallDisplacement is not implemented yet")


# TODO: complete landfall time mean error implementation
class LandfallTimeME(BaseMetric):
    """Mean error of landfall time.

    Note: Not yet implemented.
    """

    name = "LandfallTimeME"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LandfallTimeME is not implemented yet")


# TODO: complete landfall intensity mean absolute error implementation
class LandfallIntensityMAE(BaseMetric):
    """MAE of landfall intensity.

    Note: Not yet implemented.
    """

    name = "LandfallIntensityMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LandfallIntensityMAE is not implemented yet")


# TODO: complete lead time detection implementation
class LeadTimeDetection(BaseMetric):
    """Lead time detection metric.

    Note: Not yet implemented.
    """

    name = "LeadTimeDetection"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LeadTimeDetection is not implemented yet")


class SpatialDisplacement(BaseMetric):
    name = "spatial_displacement"
    preserve_dims: str = "lead_time"

    def __init__(
        self,
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        forecast_mask_variable: Optional[str | derived.DerivedVariable] = None,
        target_mask_variable: Optional[str | derived.DerivedVariable] = None,
    ):
        self.forecast_variable = forecast_variable
        self.target_variable = target_variable
        self.forecast_mask_variable = forecast_mask_variable
        self.target_mask_variable = target_mask_variable

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        def center_of_mass_ufunc(data):
            """ufunc tooling to calculate the center of mass of a 2D array, returning
            a tuple of the latitude and longitude indices, or np.nan tuple if no
            non-zero values are present.
            """
            if (data > 0).any():
                return ndimage.center_of_mass(data)
            else:
                return (np.nan, np.nan)

        target_lat_idx, target_lon_idx = xr.apply_ufunc(
            center_of_mass_ufunc,
            target,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="allowed",
        )

        # Process target coordinates
        target_lat_idx = np.round(target_lat_idx)
        target_lon_idx = np.round(target_lon_idx)
        target_lat_coords, target_lon_coords = utils.idx_to_coords(
            target_lat_idx,
            target_lon_idx,
            target.latitude.values,
            target.longitude.values,
        )
        target_coordinates = np.array([target_lat_coords, target_lon_coords])

        # Process forecast coordinates
        forecast_lat_idx, forecast_lon_idx = xr.apply_ufunc(
            center_of_mass_ufunc,
            forecast,
            input_core_dims=[["latitude", "longitude"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="allowed",
        )
        forecast_lat_idx = np.round(forecast_lat_idx)
        forecast_lon_idx = np.round(forecast_lon_idx)
        forecast_lat_coords, forecast_lon_coords = utils.idx_to_coords(
            forecast_lat_idx,
            forecast_lon_idx,
            forecast.latitude.values,
            forecast.longitude.values,
        )
        forecast_coordinates = np.array([forecast_lat_coords, forecast_lon_coords])

        # Calculate haversine distance
        distance = calc.haversine_distance(forecast_coordinates, target_coordinates)

        # Create DataArray with all dimensions
        result = xr.DataArray(
            distance,
            coords={"lead_time": forecast.lead_time, "valid_time": forecast.valid_time},
            dims=["lead_time", "valid_time"],
            name="spatial_displacement",
        )

        # Reduce over non-preserved dimensions (valid_time) by taking mean
        time_dims_to_reduce = [
            dim for dim in result.dims if dim not in self.preserve_dims
        ]
        if time_dims_to_reduce:
            result = result.mean(dim=time_dims_to_reduce)

        return result
