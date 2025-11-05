import abc
import logging
from typing import Any, Callable, Literal, Optional, Type

import numpy as np
import pandas as pd
import scores
import sparse
import xarray as xr

from extremeweatherbench import derived, utils

logger = logging.getLogger(__name__)


# Global cache for transformed contingency managers
_GLOBAL_CONTINGENCY_CACHE = utils.ThreadSafeDict()  # type: ignore


def get_cached_transformed_manager(
    forecast: xr.Dataset,
    target: xr.Dataset,
    forecast_threshold: float = 0.5,
    target_threshold: float = 0.5,
    preserve_dims: str = "lead_time",
) -> scores.categorical.BasicContingencyManager:
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

    def __call__(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the classmethod with the configured parameters
        return self.__class__.compute_metric(forecast, target, **kwargs)


class CSI(ThresholdMetric):
    """Critical Success Index metric."""

    name = "critical_success_index"

    def _compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
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
        forecast: xr.Dataset,
        target: xr.Dataset,
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
        forecast: xr.Dataset,
        target: xr.Dataset,
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
        forecast: xr.Dataset,
        target: xr.Dataset,
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
        forecast: xr.Dataset,
        target: xr.Dataset,
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


class EarlySignal(BaseMetric):
    """Metric to identify the earliest signal detection in forecast data.

    This metric finds the first occurrence where a signal is detected based on
    threshold criteria and returns the corresponding init_time, lead_time, and
    valid_time information. The metric is designed to be flexible for different
    signal detection criteria that can be specified in applied metrics downstream.
    """

    name = "early_signal"

    def _compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        threshold: Optional[float] = None,
        variable: Optional[str] = None,
        comparison: str = ">=",
        spatial_aggregation: str = "any",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataset with init_time, lead_time, valid_time
            target: The target dataset (used for reference/validation)
            threshold: Threshold value for signal detection
            variable: Variable name to analyze for signal detection
            comparison: Comparison operator (">=", "<=", ">", "<", "==", "!=")
            spatial_aggregation: How to aggregate spatially ("any", "all", "mean")
            **kwargs: Additional arguments

        Returns:
            Dataset containing earliest detection times with coordinates:
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
    """Compute the mean error between forecast and target onset times.

    This metric finds the first time when the criteria is met for a
    specified number of consecutive timesteps, then computes the mean
    error between forecast and target onset times.

    Args:
        climatology: The climatology dataset for the threshold criteria.
        min_consecutive_timesteps: Minimum number of consecutive timesteps
            that must meet the criteria to be considered onset. Default is 1.
        criteria_sign: Comparison operator (">", ">=", "<", "<=", "==").
            Default is ">=" for heatwave detection.

    Returns:
        The mean error (in hours) between forecast and target onset times.
    """

    def __init__(
        self,
        climatology: xr.DataArray,
        min_consecutive_timesteps: int = 1,
        criteria_sign: Literal[">", ">=", "<", "<=", "=="] = ">=",
        name: str = "onset_me",
        preserve_dims: str = "init_time",
    ):
        self.climatology = climatology
        self.min_consecutive_timesteps = min_consecutive_timesteps
        self.criteria_sign = criteria_sign
        self.name = name
        self.preserve_dims = preserve_dims

    def _find_onset_time(self, data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        """Find the first time where criteria is met for N consecutive steps.

        Args:
            data: Input data array with valid_time dimension
            mask: Boolean mask where condition is met

        Returns:
            DataArray containing the onset time (or NaT if not found)
        """
        # Get time dimension
        time_dim = "valid_time"
        if time_dim not in mask.dims:
            return xr.DataArray(np.datetime64("NaT", "ns"))

        # Convert mask to float for easier processing
        mask_float = mask.astype(float)

        # Check for consecutive timesteps
        n_times = mask.sizes[time_dim]
        if n_times < self.min_consecutive_timesteps:
            return xr.DataArray(np.datetime64("NaT", "ns"))

        # Find first occurrence of N consecutive True values
        for i in range(n_times - self.min_consecutive_timesteps + 1):
            window = mask_float.isel(
                {time_dim: slice(i, i + self.min_consecutive_timesteps)}
            )
            # Check if all values in window are True (== 1.0)
            # Mean over time_dim only - mask should already be spatially averaged
            window_mean = window.mean(dim=time_dim, skipna=True)
            # If there are other dims (e.g., lead_time), check all are 1.0
            if window_mean.ndim == 0:
                # Scalar case
                if float(window_mean) == 1.0:
                    onset_time = data[time_dim].isel({time_dim: i})
                    return onset_time
            else:
                # Array case - check if all values are 1.0
                if bool((window_mean == 1.0).all()):
                    onset_time = data[time_dim].isel({time_dim: i})
                    return onset_time

        # No onset found
        return xr.DataArray(np.datetime64("NaT", "ns"))

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> Any:
        """Compute spatially averaged onset time mean error.

        Args:
            forecast: Forecast dataset with dims (init_time, lead_time, valid_time)
            target: Target dataset with dims (valid_time)

        Returns:
            Mean error (hours) between forecast and target onset times
        """
        climatology_time = convert_day_yearofday_to_time(
            self.climatology, forecast.valid_time.dt.year.values[0]
        )
        spatial_dims = [
            dim
            for dim in forecast.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]

        forecast = _reduce_duck_array(
            forecast, func=np.nanmean, reduce_dims=spatial_dims
        )
        target = _reduce_duck_array(target, func=np.nanmean, reduce_dims=spatial_dims)

        if isinstance(target.data, sparse.COO):
            # If target is sparse, interp with stacked coordinates
            climatology_time = climatology_time.interp(
                latitude=target["stacked"]["latitude"],
                longitude=target["stacked"]["longitude"],
                method="nearest",
                kwargs={"fill_value": None},
            )
        else:
            # Otherwise, interp with target coordinates
            climatology_time = climatology_time.interp_like(
                target, method="nearest", kwargs={"fill_value": None}
            )

        # Create comparison masks
        forecast_mask = create_comparison_mask(
            forecast, climatology_time, self.criteria_sign
        )
        target_mask = create_comparison_mask(
            target, climatology_time, self.criteria_sign
        )

        # Spatially average the masks (mean over spatial dims)
        for dim in spatial_dims:
            if dim in forecast_mask.dims:
                forecast_mask = forecast_mask.mean(dim=dim, skipna=True)
            if dim in target_mask.dims:
                target_mask = target_mask.mean(dim=dim, skipna=True)

        # Threshold averaged mask (>=0.5 means majority of points meet criteria)
        forecast_mask = forecast_mask >= 0.5
        target_mask = target_mask >= 0.5

        # Check if init_time is a dimension or just a coordinate
        has_init_time_dim = "init_time" in forecast.dims

        # Find onset times for each init_time
        if has_init_time_dim:
            forecast_onset = forecast.groupby("init_time").map(
                lambda x: self._find_onset_time(
                    x, forecast_mask.sel(init_time=x.init_time.values[0])
                )
            )
        else:
            # For lead_time structure, group by init_time coordinate
            forecast_onset = forecast.groupby("init_time").map(
                lambda x: self._find_onset_time(x, forecast_mask)
            )

        target_onset = self._find_onset_time(target, target_mask)

        # Convert onset times to hours since a reference time
        # Use target onset as reference for easier interpretation
        if not pd.isna(target_onset.values):
            ref_time = pd.Timestamp(target_onset.values)
            # Convert forecast onset to hours difference
            forecast_onset_hours = xr.apply_ufunc(
                lambda t: (pd.Timestamp(t) - ref_time).total_seconds() / 3600
                if not pd.isna(t)
                else np.nan,
                forecast_onset,
                vectorize=True,
            )
            target_onset_hours = xr.DataArray(0.0)  # Reference is 0
        else:
            # If no target onset, use NaN
            forecast_onset_hours = xr.full_like(forecast_onset, np.nan, dtype=float)
            target_onset_hours = xr.DataArray(np.nan)

        return super()._compute_metric(
            forecast=forecast_onset_hours,
            target=target_onset_hours,
            preserve_dims=self.preserve_dims,
        )


class DurationME(ME):
    """Compute the duration of a case's event.
    This metric computes the mean error between the forecast and target durations.

    Args:
        climatology: The climatology dataset for the heatwave criteria.

    Returns:
        The mean error between the forecast and target heatwave durations.
    """

    def __init__(
        self,
        climatology: xr.DataArray,
        criteria_sign: Literal[">", ">=", "<", "<=", "=="] = ">=",
        name: str = "heatwave_duration_me",
        preserve_dims: str = "init_time",
    ):
        self.climatology = climatology
        self.criteria_sign = criteria_sign
        self.name = name
        self.preserve_dims = preserve_dims

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> Any:
        """Compute spatially averaged duration mean error.

        Args:
            forecast: Forecast dataset with dims (init_time, lead_time, valid_time)
            target: Target dataset with dims (valid_time)

        Returns:
            Mean error between forecast and target heatwave durations
        """
        climatology_time = convert_day_yearofday_to_time(
            self.climatology, forecast.valid_time.dt.year.values[0]
        )
        spatial_dims = [
            dim
            for dim in forecast.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]
        forecast = _reduce_duck_array(
            forecast, func=np.nanmean, reduce_dims=spatial_dims
        )
        target = _reduce_duck_array(target, func=np.nanmean, reduce_dims=spatial_dims)
        if isinstance(target.data, sparse.COO):
            # If target is sparse, interp with stacked coordinates
            climatology_time = climatology_time.interp(
                latitude=target["stacked"]["latitude"],
                longitude=target["stacked"]["longitude"],
                method="nearest",
                kwargs={"fill_value": None},
            )
        else:
            # Otherwise, interp with target coordinates
            climatology_time = climatology_time.interp_like(
                target, method="nearest", kwargs={"fill_value": None}
            )
        forecast_mask = create_comparison_mask(
            forecast, climatology_time, self.criteria_sign
        )

        # Calculate target duration (count of timesteps exceeding climatology)
        target_mask = create_comparison_mask(
            target, climatology_time, self.criteria_sign
        )

        # Track NaN locations in forecast data
        forecast_valid_mask = ~forecast.isnull()

        # Apply valid data mask (exclude NaN positions in forecast)
        forecast_mask_final = forecast_mask.where(forecast_valid_mask)
        try:
            target_mask_final = target_mask.where(forecast_valid_mask)
        # If sparse, will need to expand_dims first as transpose is not supported
        except AttributeError:
            print("target_mask is sparse")
            target_mask_final = target_mask.expand_dims(dim={"lead_time": 41}).where(
                forecast_valid_mask
            )

        # Sum to get durations (NaN values are excluded by default)
        forecast_duration = forecast_mask_final.groupby(self.preserve_dims).sum()
        target_duration = target_mask_final.groupby(self.preserve_dims).sum()

        # TODO: product of time resolution hours and duration
        return super()._compute_metric(
            forecast=forecast_duration,
            target=target_duration,
            preserve_dims=self.preserve_dims,
        )


def create_comparison_mask(
    data: xr.DataArray,
    criteria: xr.DataArray,
    sign: str = ">=",
) -> xr.DataArray:
    """Create comparison mask based on sign.

    Args:
        data: Input data array
        criteria: Criteria to compare against
        sign: Comparison operator (">", ">=", "<", "<=", "==")

    Returns:
        Boolean mask where condition is met
    """
    match sign:
        case ">=":
            return data >= criteria
        case ">":
            return data > criteria
        case "<=":
            return data <= criteria
        case "<":
            return data < criteria
        case "==":
            return data == criteria
        case _:
            raise ValueError(f"Unsupported sign: {sign}")


def convert_day_yearofday_to_time(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Convert dayofyear and hour coordinates in an xarray Dataset to a new time
    coordinate.

    Args:
        dataset: The input xarray dataset.
        year: The base year to use for the time coordinate.

    Returns:
        The dataset with a new time coordinate.
    """
    # Create a new time coordinate by combining dayofyear and hour
    time_dim = pd.date_range(
        start=f"{year}-01-01",
        periods=len(dataset["dayofyear"]) * len(dataset["hour"]),
        freq="6h",
    )
    dataset = dataset.stack(valid_time=("dayofyear", "hour")).drop(
        ["dayofyear", "hour"]
    )
    # Assign the new time coordinate to the dataset
    dataset = dataset.assign_coords(valid_time=time_dim)

    return dataset
