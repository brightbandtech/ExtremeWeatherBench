import abc
import logging
from typing import Any, Literal, Optional, Type

import numpy as np
import scores
import xarray as xr

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
    DataArray. EWB metrics prioritize the use of any arbitrary sets of forecasts and
    analyses, so long as the spatiotemporal dimensions are the same.
    """

    def __init__(
        self,
        name: str,
        preserve_dims: str = "lead_time",
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
    ):
        self.name = name
        self.preserve_dims = preserve_dims
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
        name: str,
        preserve_dims: str = "lead_time",
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.forecast_threshold = forecast_threshold
        self.target_threshold = target_threshold
        self.preserve_dims = preserve_dims

    def __call__(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs) -> Any:
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the instance method with the configured parameters
        return self.compute_metric(forecast, target, **kwargs)


class CSI(ThresholdMetric):
    """Critical Success Index metric."""

    def __init__(self, *args, **kwargs):
        super().__init__("critical_success_index", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("false_alarm_ratio", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("true_positive", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("false_positive", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("true_negative", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("false_negative", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("accuracy", *args, **kwargs)

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
            name=metric_name,
            forecast_threshold=forecast_threshold,
            target_threshold=target_threshold,
            preserve_dims=preserve_dims,
        )
        result_metrics.append(metric)

    return result_metrics


class MAE(BaseMetric):
    def __init__(self, name: str = "MAE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

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

    def __init__(self, name: str = "ME", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("rmse", *args, **kwargs)

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


class EarlySignal(BaseMetric):
    """Metric to identify the earliest signal detection in forecast data.

    This metric finds the first occurrence where a signal is detected based on
    threshold criteria and returns the corresponding init_time, lead_time, and
    valid_time information. The metric is designed to be flexible for different
    signal detection criteria that can be specified in applied metrics downstream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("early_signal", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        threshold: Optional[float] = None,
        variable: Optional[str] = None,
        comparison: str = ">=",
        spatial_aggregation: str = "any",
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataarray with init_time, lead_time, valid_time
            target: The target dataarray (used for reference/validation)
            threshold: Threshold value for signal detection
            variable: Variable name to analyze for signal detection
            comparison: Comparison operator (">=", "<=", ">", "<", "==", "!=")
            spatial_aggregation: How to aggregate spatially ("any", "all", "mean")
            **kwargs: Additional arguments

        Returns:
            DataArray containing earliest detection times with coordinates:
            - earliest_init_time: First init_time when signal was detected
            - earliest_lead_time: Corresponding lead_time
            - earliest_valid_time: Corresponding valid_time
            - detection_found: Boolean indicating if any detection occurred
        """
        if threshold is None or variable is None:
            # Return structure for when no detection criteria specified
            return xr.Dataset(
                {
                    "earliest_init_time": xr.DataArray(np.datetime64("NaT")),
                    "earliest_lead_time": xr.DataArray(np.timedelta64("NaT")),
                    "earliest_valid_time": xr.DataArray(np.datetime64("NaT")),
                    "detection_found": xr.DataArray(False),
                }
            )

        if variable not in forecast.data_vars:
            raise ValueError(f"Variable '{variable}' not found in forecast dataset")

        data = forecast[variable]

        # Apply threshold comparison
        comparison_ops = {
            ">=": lambda x, t: x >= t,
            "<=": lambda x, t: x <= t,
            ">": lambda x, t: x > t,
            "<": lambda x, t: x < t,
            "==": lambda x, t: x == t,
            "!=": lambda x, t: x != t,
        }

        if comparison not in comparison_ops:
            raise ValueError(f"Comparison '{comparison}' not supported")

        # Create detection mask
        detection_mask = comparison_ops[comparison](data, threshold)

        # Apply spatial aggregation
        spatial_dims = [
            dim
            for dim in detection_mask.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]

        if spatial_dims:
            if spatial_aggregation == "any":
                detection_mask = detection_mask.any(spatial_dims)
            elif spatial_aggregation == "all":
                detection_mask = detection_mask.all(spatial_dims)
            elif spatial_aggregation == "mean":
                detection_mask = detection_mask.mean(spatial_dims) > 0.5
            else:
                raise ValueError(
                    f"Spatial aggregation '{spatial_aggregation}' not supported"
                )

        # Find earliest detection for each init_time
        earliest_results = {}

        for init_t in forecast.init_time:
            init_mask = detection_mask.sel(init_time=init_t)

            # Find first occurrence along lead_time dimension
            if init_mask.any():
                # Get the first True index along lead_time
                first_detection_idx = init_mask.argmax("lead_time")
                earliest_lead = forecast.lead_time[first_detection_idx]
                earliest_valid = init_t.values + np.timedelta64(int(earliest_lead), "h")

                earliest_results[init_t.values] = {
                    "init_time": init_t.values,
                    "lead_time": earliest_lead.values,
                    "valid_time": earliest_valid,
                    "found": True,
                }
            else:
                earliest_results[init_t.values] = {
                    "init_time": init_t.values,
                    "lead_time": np.timedelta64("NaT"),
                    "valid_time": np.datetime64("NaT"),
                    "found": False,
                }

        # Convert to xarray Dataset
        init_times = list(earliest_results.keys())
        earliest_init_times = [r["init_time"] for r in earliest_results.values()]
        earliest_lead_times = [r["lead_time"] for r in earliest_results.values()]
        earliest_valid_times = [r["valid_time"] for r in earliest_results.values()]
        detection_found = [r["found"] for r in earliest_results.values()]

        result = xr.Dataset(
            {
                "earliest_init_time": xr.DataArray(
                    earliest_init_times,
                    coords={"init_time": init_times},
                    dims=["init_time"],
                ),
                "earliest_lead_time": xr.DataArray(
                    earliest_lead_times,
                    coords={"init_time": init_times},
                    dims=["init_time"],
                ),
                "earliest_valid_time": xr.DataArray(
                    earliest_valid_times,
                    coords={"init_time": init_times},
                    dims=["init_time"],
                ),
                "detection_found": xr.DataArray(
                    detection_found,
                    coords={"init_time": init_times},
                    dims=["init_time"],
                ),
            }
        )

        return result


class MaximumMAE(MAE):
    def __init__(self, *args, **kwargs):
        super().__init__("MaximumMAE", *args, **kwargs)

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
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
        )
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        ).compute()
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
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

    def __init__(self, *args, **kwargs):
        super().__init__("MinimumMAE", *args, **kwargs)

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
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
        )
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
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

    def __init__(self, *args, **kwargs):
        super().__init__(name="MaxMinMAE", *args, **kwargs)

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
        forecast = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=reduce_dims, skipna=True
        )
        target = utils.reduce_dataarray(
            target, method="mean", reduce_dims=reduce_dims, skipna=True
        )

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

    def __init__(self, *args, **kwargs):
        super().__init__("OnsetME", *args, **kwargs)

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
            utils.reduce_dataarray(
                forecast,
                method="mean",
                reduce_dims=["latitude", "longitude"],
                skipna=True,
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

    def __init__(self, *args, **kwargs):
        super().__init__("DurationME", *args, **kwargs)

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
            utils.reduce_dataarray(
                forecast,
                method="mean",
                reduce_dims=["latitude", "longitude"],
                skipna=True,
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


# TODO: complete spatial displacement implementation
class SpatialDisplacement(BaseMetric):
    """Spatial displacement error metric.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("LandfallDisplacement", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("SpatialDisplacement is not implemented yet")


def calculate_landfall_distance_km(
    landfall1: xr.DataArray, landfall2: xr.DataArray
) -> xr.DataArray:
    """Calculate the distance between two landfall points in km.
    Handles both scalar and multi-dimensional (with init_time) DataArrays.

    Args:
        landfall1: First landfall xarray DataArray
        landfall2: Second landfall xarray DataArray

    Returns:
        Distance in kilometers as xarray DataArray
    """
    if landfall1 is None or landfall2 is None:
        return xr.DataArray(np.nan)

    # Use xarray operations to handle multi-dimensional case
    distance_degrees = calc.haversine_distance(
        [landfall1.coords["latitude"], landfall1.coords["longitude"]],
        [landfall2.coords["latitude"], landfall2.coords["longitude"]],
        units="degrees",
    )

    # Convert from degrees to kilometers (using Earth's radius)
    distance_km = np.radians(distance_degrees) * 6371

    return distance_km


def calculate_landfall_time_difference_hours(
    landfall1: xr.DataArray, landfall2: xr.DataArray
) -> xr.DataArray:
    """Calculate the time difference between two landfall points in hours.

    Args:
        landfall1: First landfall xarray DataArray
        landfall2: Second landfall xarray DataArray

    Returns:
        Time difference in hours (landfall1 - landfall2) as xarray DataArray
        with init_time as the sole dimension
    """
    if landfall1 is None or landfall2 is None:
        return xr.DataArray(np.nan)

    # Get time values from landfall DataArrays (as coordinates)
    time1 = landfall1.coords["valid_time"]
    time2 = landfall2.coords["valid_time"]

    # Calculate time difference in hours
    time_diff = time1 - time2
    time_diff_hours = time_diff / np.timedelta64(1, "h")

    # Return with init_time as the only dimension
    if "init_time" in landfall1.dims:
        # Ensure the values are properly shaped for init_time dimension
        if time_diff_hours.dims == ():
            # Scalar case - broadcast to all init_times
            values = np.full(len(landfall1.init_time), float(time_diff_hours.values))
        else:
            # Already has the right dimensions
            values = time_diff_hours.values

        return xr.DataArray(
            values,
            dims=["init_time"],
            coords={"init_time": landfall1.coords["init_time"]},
        )
    else:
        # Scalar case - no init_time dimension
        return time_diff_hours


def _get_nan_result_for_forecast(forecast: xr.DataArray) -> xr.DataArray:
    """Create NaN result with appropriate dimensions based on forecast."""
    if "lead_time" in forecast.dims:
        # Calculate init_times from lead_time and valid_time
        init_times_calc = forecast.coords["valid_time"] - forecast.lead_time
        unique_init_times = np.unique(init_times_calc.values)

        return xr.DataArray(
            np.full(len(unique_init_times), np.nan),
            dims=["init_time"],
            coords={"init_time": unique_init_times},
        )
    elif "init_time" in forecast.coords:
        return xr.DataArray(
            np.full(len(forecast.init_time), np.nan),
            dims=["init_time"],
            coords={"init_time": forecast.init_time},
        )
    else:
        return xr.DataArray(np.nan)


def _get_landfall_data(
    forecast: xr.DataArray,
    target: xr.DataArray,
    approach: Literal["first", "next"],
) -> tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Get forecast and target landfall DataArrays based on approach.

    Uses calc.find_landfalls() for landfall detection. The DataArrays
    must have latitude, longitude, and valid_time as coordinates.

    Args:
        forecast: Forecast track DataArray with lat/lon/valid_time coords
        target: Target track DataArray with lat/lon/valid_time coords
        approach: Landfall detection approach ('first' or 'next')

    Returns:
        Tuple of (forecast_landfall, target_landfall) DataArrays
        with latitude, longitude, valid_time as coordinates
    """
    if approach == "first":
        # First landfall approach - simple
        forecast_landfall = calc.find_landfalls(forecast, return_all=False)
        target_landfall = calc.find_landfalls(target, return_all=False)
        return forecast_landfall, target_landfall

    elif approach == "next":
        # Next landfall approach - more complex
        target_landfalls = calc.find_landfalls(target, return_all=True)
        if target_landfalls is None:
            return None, None

        next_target_landfalls = calc.find_next_landfall_for_init_time(
            forecast, target_landfalls
        )
        if next_target_landfalls is None:
            return None, None

        forecast_landfalls = calc.find_landfalls(forecast, return_all=True)
        if forecast_landfalls is None:
            return None, next_target_landfalls

        # Match forecast landfalls to target for each init_time
        return forecast_landfalls, next_target_landfalls

    else:
        raise ValueError(f"Unknown approach: {approach}")


def _compute_landfall_metric_value(
    forecast_landfall: xr.DataArray,
    target_landfall: xr.DataArray,
    metric_type: Literal["displacement", "timing", "intensity"],
    intensity_var: str = "surface_wind_speed",
    approach: Literal["first", "next"] = "first",
) -> xr.DataArray:
    """Compute landfall metric value (displacement, timing, or intensity).

    Args:
        forecast_landfall: Forecast landfall DataArray
        target_landfall: Target landfall DataArray
        metric_type: Type of metric ('displacement', 'timing', 'intensity')
        intensity_var: Variable to use for intensity metrics
        approach: Landfall approach (affects handling for 'next' approach)

    Returns:
        Computed metric as xarray DataArray
    """
    if metric_type == "displacement":
        if approach == "next" and "init_time" in target_landfall.dims:
            # Handle next approach with per-init_time matching
            return _compute_next_displacement(forecast_landfall, target_landfall)
        else:
            return _compute_displacement(forecast_landfall, target_landfall)

    elif metric_type == "timing":
        if approach == "next" and "init_time" in target_landfall.dims:
            return _compute_next_timing(forecast_landfall, target_landfall)
        else:
            return calculate_landfall_time_difference_hours(
                forecast_landfall, target_landfall
            )

    elif metric_type == "intensity":
        if approach == "next" and "init_time" in target_landfall.dims:
            return _compute_next_intensity(
                forecast_landfall, target_landfall, intensity_var
            )
        else:
            return _compute_intensity(forecast_landfall, target_landfall, intensity_var)

    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def _compute_displacement(
    forecast_landfall: xr.DataArray, target_landfall: xr.DataArray
) -> xr.DataArray:
    """Compute displacement for first landfall approach."""
    if "init_time" in forecast_landfall.dims:
        # Vector case - compute distance for each init_time
        distances = []
        for i in range(len(forecast_landfall.coords["init_time"])):
            f_lat = forecast_landfall.coords["latitude"].isel(init_time=i).values
            f_lon = forecast_landfall.coords["longitude"].isel(init_time=i).values
            t_lat = target_landfall.coords["latitude"].values
            t_lon = target_landfall.coords["longitude"].values

            # Skip if any coordinates are NaN
            if np.isnan(f_lat) or np.isnan(f_lon) or np.isnan(t_lat) or np.isnan(t_lon):
                distances.append(np.nan)
            else:
                dist = calc.haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units="km"
                )
                # Ensure we append a scalar value
                distances.append(
                    float(dist.item()) if hasattr(dist, "item") else float(dist)
                )

        return xr.DataArray(
            distances,
            dims=["init_time"],
            coords={"init_time": forecast_landfall.coords["init_time"]},
        )
    else:
        # Scalar case
        return calculate_landfall_distance_km(forecast_landfall, target_landfall)


def _compute_intensity(
    forecast_landfall: xr.DataArray,
    target_landfall: xr.DataArray,
    intensity_var: str,
) -> xr.DataArray:
    """Compute intensity error for first landfall approach.

    Args:
        forecast_landfall: Forecast landfall DataArray with intensity values
        target_landfall: Target landfall DataArray with intensity values
        intensity_var: Expected variable name (for validation)

    Returns:
        Absolute error between forecast and target intensity
    """
    # The DataArray values are the intensity values
    # Verify the DataArray name matches expected if name is set
    if forecast_landfall.name and forecast_landfall.name != intensity_var:
        logger.warning(
            f"Forecast landfall variable '{forecast_landfall.name}' "
            f"does not match expected '{intensity_var}'"
        )
    if target_landfall.name and target_landfall.name != intensity_var:
        logger.warning(
            f"Target landfall variable '{target_landfall.name}' "
            f"does not match expected '{intensity_var}'"
        )

    # Calculate absolute error using DataArray values directly
    return np.abs(forecast_landfall - target_landfall)


def _compute_next_displacement(
    forecast_landfalls: xr.DataArray, target_landfalls: xr.DataArray
) -> xr.DataArray:
    """Compute displacement for next landfall approach.

    Matches each target landfall to the closest forecast landfall
    in time for the same init_time.

    Args:
        forecast_landfalls: Forecast landfall DataArray with landfall dim
        target_landfalls: Target landfall DataArray with init_time dim

    Returns:
        DataArray of distances in km for each init_time
    """
    results = []
    init_times_out = []

    for i, init_time in enumerate(target_landfalls.coords["init_time"]):
        target_landfall = target_landfalls.isel(init_time=i)
        target_time = target_landfall.coords["valid_time"].values

        # Find matching forecast landfall for this init_time
        if "init_time" in forecast_landfalls.dims:
            init_time_match = forecast_landfalls.coords["init_time"] == init_time
            if not init_time_match.any():
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            forecast_for_init = forecast_landfalls.where(init_time_match, drop=True)

            if len(forecast_for_init.coords["init_time"]) == 0:
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            # Find forecast landfall closest to target time
            time_diffs = np.abs(forecast_for_init.coords["valid_time"] - target_time)
            closest_idx = time_diffs.argmin()
            closest_forecast = forecast_for_init.isel(landfall=closest_idx)
        else:
            closest_forecast = forecast_landfalls

        # Calculate distance
        try:
            result = calculate_landfall_distance_km(closest_forecast, target_landfall)
            results.append(float(result.values))
        except Exception:
            results.append(np.nan)

        init_times_out.append(init_time.values)

    return xr.DataArray(
        results, dims=["init_time"], coords={"init_time": init_times_out}
    )


def _compute_next_timing(
    forecast_landfalls: xr.DataArray, target_landfalls: xr.DataArray
) -> xr.DataArray:
    """Compute timing error for next landfall approach.

    Matches each target landfall to the closest forecast landfall
    in time for the same init_time.

    Args:
        forecast_landfalls: Forecast landfall DataArray with landfall dim
        target_landfalls: Target landfall DataArray with init_time dim

    Returns:
        DataArray of time differences in hours for each init_time
    """
    results = []
    init_times_out = []

    for i, init_time in enumerate(target_landfalls.coords["init_time"]):
        target_landfall = target_landfalls.isel(init_time=i)
        target_time = target_landfall.coords["valid_time"].values

        # Find matching forecast landfall
        if "init_time" in forecast_landfalls.dims:
            init_time_match = forecast_landfalls.coords["init_time"] == init_time
            if not init_time_match.any():
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            forecast_for_init = forecast_landfalls.where(init_time_match, drop=True)

            if len(forecast_for_init.coords["init_time"]) == 0:
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            # Find forecast landfall closest to target time
            time_diffs = np.abs(forecast_for_init.coords["valid_time"] - target_time)
            closest_idx = time_diffs.argmin()
            closest_forecast = forecast_for_init.isel(landfall=closest_idx)
        else:
            closest_forecast = forecast_landfalls

        # Calculate time difference
        try:
            result = calculate_landfall_time_difference_hours(
                closest_forecast, target_landfall
            )
            results.append(float(result.values))
        except Exception:
            results.append(np.nan)

        init_times_out.append(init_time.values)

    return xr.DataArray(
        results, dims=["init_time"], coords={"init_time": init_times_out}
    )


def _compute_next_intensity(
    forecast_landfalls: xr.DataArray,
    target_landfalls: xr.DataArray,
    intensity_var: str,
) -> xr.DataArray:
    """Compute intensity error for next landfall approach.

    Matches each target landfall to the closest forecast landfall
    in time for the same init_time.

    Args:
        forecast_landfalls: Forecast landfall DataArray with landfall dim
        target_landfalls: Target landfall DataArray with init_time dim
        intensity_var: Expected variable name (for validation)

    Returns:
        DataArray of absolute intensity errors for each init_time
    """
    results = []
    init_times_out = []

    for i, init_time in enumerate(target_landfalls.coords["init_time"]):
        target_landfall = target_landfalls.isel(init_time=i)
        target_time = target_landfall.coords["valid_time"].values

        # Find matching forecast landfall
        if "init_time" in forecast_landfalls.dims:
            init_time_match = forecast_landfalls.coords["init_time"] == init_time
            if not init_time_match.any():
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            forecast_for_init = forecast_landfalls.where(init_time_match, drop=True)

            if len(forecast_for_init.coords["init_time"]) == 0:
                results.append(np.nan)
                init_times_out.append(init_time.values)
                continue

            # Find forecast landfall closest to target time
            time_diffs = np.abs(forecast_for_init.coords["valid_time"] - target_time)
            closest_idx = time_diffs.argmin()
            closest_forecast = forecast_for_init.isel(landfall=closest_idx)
        else:
            closest_forecast = forecast_landfalls

        # Calculate intensity error
        try:
            result = _compute_intensity(
                closest_forecast, target_landfall, intensity_var
            )
            results.append(float(result.values))
        except Exception:
            results.append(np.nan)

        init_times_out.append(init_time.values)

    return xr.DataArray(
        results, dims=["init_time"], coords={"init_time": init_times_out}
    )


def compute_landfall_metric(
    forecast: xr.DataArray,
    target: xr.DataArray,
    metric_type: Literal["displacement", "timing", "intensity"],
    approach: Literal["first", "next"] = "first",
    intensity_var: str = "surface_wind_speed",
) -> xr.DataArray:
    """Unified function to compute landfall metrics.

    Args:
        forecast: Forecast TC track DataArray with lat/lon/time coords
        target: Target TC track DataArray with lat/lon/time coords
        metric_type: Type of metric to compute
        approach: Landfall detection approach
        intensity_var: Variable to use for intensity metrics

    Returns:
        Computed metric as xarray DataArray
    """
    # Get landfall data based on approach
    forecast_landfall, target_landfall = _get_landfall_data(forecast, target, approach)

    # Handle case where no landfall is found
    if forecast_landfall is None or target_landfall is None:
        return _get_nan_result_for_forecast(forecast)

    # Compute the metric
    return _compute_landfall_metric_value(
        forecast_landfall,
        target_landfall,
        metric_type,
        intensity_var,
        approach,
    )


# Legacy functions - kept for backwards compatibility


def compute_first_landfall_metric(
    forecast: xr.DataArray,
    target: xr.DataArray,
    metric_type: Literal["displacement", "timing", "intensity"],
    intensity_var: str = "surface_wind_speed",
):
    """Compute metric using first landfall approach (classic).

    .. deprecated::
        Use compute_landfall_metric() or the LandfallDisplacement,
        LandfallTimeME, LandfallIntensityMAE classes instead.
    """
    logger.warning(
        "compute_first_landfall_metric is deprecated. Use "
        "compute_landfall_metric() or metric classes instead."
    )
    return compute_landfall_metric(
        forecast, target, metric_type, approach="first", intensity_var=intensity_var
    )


def compute_next_landfall_metric(
    forecast: xr.DataArray,
    target: xr.DataArray,
    metric_type: Literal["displacement", "timing", "intensity"],
    intensity_var: str = "surface_wind_speed",
):
    """Compute metric using next upcoming landfall approach.

    .. deprecated::
        Use compute_landfall_metric() or the LandfallDisplacement,
        LandfallTimeME, LandfallIntensityMAE classes instead.
    """
    logger.warning(
        "compute_next_landfall_metric is deprecated. Use "
        "compute_landfall_metric() or metric classes instead."
    )
    return compute_landfall_metric(
        forecast, target, metric_type, approach="next", intensity_var=intensity_var
    )


def compute_landfall_selector(
    approach: Literal["first", "next"],
    forecast: xr.DataArray,
    target: xr.DataArray,
    metric_type: Literal["displacement", "timing", "intensity"],
    intensity_var: str = "surface_wind_speed",
):
    """Select the appropriate landfall metric based on the approach.

    .. deprecated::
        Use compute_landfall_metric() or the LandfallDisplacement,
        LandfallTimeME, LandfallIntensityMAE classes instead.
    """
    logger.warning(
        "compute_landfall_selector is deprecated. Use "
        "compute_landfall_metric() or metric classes instead."
    )
    return compute_landfall_metric(
        forecast, target, metric_type, approach=approach, intensity_var=intensity_var
    )


def compute_displacement_metric(
    forecast_landfall: xr.DataArray, target_landfall: xr.DataArray
):
    """Compute displacement between forecast and target landfall.

    .. deprecated::
        Use _compute_displacement() directly or metric classes.
    """
    logger.warning(
        "compute_displacement_metric is deprecated. Use "
        "_compute_displacement() or metric classes instead."
    )
    return _compute_displacement(forecast_landfall, target_landfall)


def compute_timing_metric(
    forecast_landfall: xr.DataArray,
    target_landfall: xr.DataArray,
):
    """Compute timing difference between forecast and target landfall.

    .. deprecated::
        Use calculate_landfall_time_difference_hours() or metric classes.
    """
    logger.warning(
        "compute_timing_metric is deprecated. Use "
        "calculate_landfall_time_difference_hours() or classes instead."
    )
    return calculate_landfall_time_difference_hours(forecast_landfall, target_landfall)


def compute_intensity_metric(
    forecast_landfall: xr.DataArray,
    target_landfall: xr.DataArray,
    intensity_var: str,
):
    """Compute intensity difference between forecast and target landfall.

    .. deprecated::
        Use _compute_intensity() directly or metric classes.
    """
    logger.warning(
        "compute_intensity_metric is deprecated. Use "
        "_compute_intensity() or metric classes instead."
    )
    return _compute_intensity(forecast_landfall, target_landfall, intensity_var)


class LandfallDisplacement(SpatialDisplacement):
    """Landfall displacement metric with configurable landfall detection
    approaches.

    This metric computes the great circle distance between forecast and target
    landfall positions using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target data

    Parameters:
        approach (str): Landfall detection approach ('first', 'next')
        exclude_post_landfall (bool): Whether to exclude init_times after all landfalls
    """

    approach: Literal["first", "next"] = "first"
    preserve_dims: str = "init_time"

    def __init__(
        self,
        approach: Literal["first", "next"] = "first",
        exclude_post_landfall: bool = False,
    ):
        """Initialize the landfall displacement metric.

        Args:
            approach: Landfall detection approach ('first', 'next', 'all')
            exclude_post_landfall: Whether to exclude init_times after all landfalls
        """
        super().__init__()
        self.approach = approach
        self.exclude_post_landfall = exclude_post_landfall

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall displacement using the configured approach.

        Args:
            forecast: Forecast TC track DataArray with lat/lon/time coords
            target: Target/analysis TC track DataArray with lat/lon/time
            **kwargs: Additional arguments

        Returns:
            xarray.DataArray with landfall displacement distance in km
        """
        return compute_landfall_metric(
            forecast, target, metric_type="displacement", approach=self.approach
        )


class LandfallTimeME(ME):
    """Landfall timing metric with configurable landfall detection approaches.

    This metric computes the time difference between forecast and target landfall
    timing using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target data

    Parameters:
        approach: Landfall detection approach ('first', 'next')
    """

    approach: Literal["first", "next"] = "first"
    preserve_dims: str = "init_time"

    def __init__(self, approach: Literal["first", "next"] = "first"):
        """Initialize the landfall timing metric.

        Args:
            approach: Landfall detection approach ('first', 'next')
        """
        super().__init__()
        self.approach = approach

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute landfall timing error using the configured approach.

        Args:
            forecast: Forecast TC track DataArray with lat/lon/time coords
            target: Target/analysis TC track DataArray with lat/lon/time
            **kwargs: Additional arguments

        Returns:
            xarray.DataArray with landfall timing errors in hours
        """
        return compute_landfall_metric(
            forecast, target, metric_type="timing", approach=self.approach
        )


class LandfallIntensityMAE(MAE):
    """Landfall intensity metric with configurable landfall detection approaches.

    This metric computes the mean absolute error between forecast and target
    intensity at landfall using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target

    The intensity variable is determined by forecast_variable and
    target_variable. To evaluate multiple intensity variables (e.g.,
    surface_wind_speed and air_pressure_at_mean_sea_level), create
    separate metric instances for each variable.

    Parameters:
        approach: Landfall detection approach ('first', 'next')
        forecast_variable: Variable to use for forecast intensity
        target_variable: Variable to use for target intensity
    """

    name = "landfall_intensity_mae"
    approach: Literal["first", "next"] = "first"
    preserve_dims: str = "init_time"

    def __init__(
        self,
        approach: Literal["first", "next"] = "first",
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
    ):
        """Initialize the landfall intensity metric.

        Args:
            approach: Landfall detection approach ('first', 'next')
            forecast_variable: Variable for forecast intensity (optional)
            target_variable: Variable for target intensity (optional)
        """
        super().__init__(
            forecast_variable=forecast_variable, target_variable=target_variable
        )
        self.approach = approach

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute landfall intensity error using the configured approach.

        Args:
            forecast: Forecast TC track DataArray with lat/lon/time coords
            target: Target/analysis TC track DataArray with lat/lon/time
            **kwargs: Additional arguments

        Returns:
            xarray.DataArray with landfall intensity errors
        """
        # Use the DataArray name as the intensity variable
        intensity_var = forecast.name if forecast.name else "surface_wind_speed"

        return compute_landfall_metric(
            forecast,
            target,
            metric_type="intensity",
            approach=self.approach,
            intensity_var=intensity_var,
        )
