import abc
import logging
import operator
from typing import Any, Callable, Optional, Sequence, Type

import numpy as np
import scores
import xarray as xr

from extremeweatherbench import derived, utils

logger = logging.getLogger(__name__)


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
        # Store the original variables (str or DerivedVariable instances)
        # Do NOT convert to string to preserve output_variables info
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

    Can be used in two ways:
    1. As a base class for specific threshold metrics (CSI, FAR, etc.)
    2. As a composite metric to compute multiple threshold metrics
       efficiently by reusing the transformed contingency manager.

    Example of composite usage:
        composite = ThresholdMetric(
            metrics=[CSI, FAR, Accuracy],
            forecast_threshold=0.7,
            target_threshold=0.5
        )
        results = composite.compute_metric(forecast, target)
        # Returns: {"critical_success_index": ...,
        #           "false_alarm_ratio": ..., "accuracy": ...}
    """

    def __init__(
        self,
        name: str = "threshold_metrics",
        preserve_dims: str = "lead_time",
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        metrics: Optional[list[Type["ThresholdMetric"]]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            **kwargs,
        )
        self.forecast_threshold = forecast_threshold
        self.target_threshold = target_threshold
        self.preserve_dims = preserve_dims
        self.metrics = metrics or []

        # If metrics provided, instantiate them
        if self.metrics:
            self._metric_instances = [
                (
                    metric_cls(
                        forecast_threshold=self.forecast_threshold,
                        target_threshold=self.target_threshold,
                        preserve_dims=self.preserve_dims,
                    )
                    if isinstance(metric_cls, type)
                    else metric_cls
                )
                for metric_cls in self.metrics
            ]
        else:
            self._metric_instances = []

    def __call__(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs) -> Any:
        """Make instances callable using their configured thresholds."""
        # Use instance attributes as defaults, but allow override from kwargs
        kwargs.setdefault("forecast_threshold", self.forecast_threshold)
        kwargs.setdefault("target_threshold", self.target_threshold)
        kwargs.setdefault("preserve_dims", self.preserve_dims)

        # Call the instance method with the configured parameters
        return self.compute_metric(forecast, target, **kwargs)

    def transformed_contingency_manager(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        forecast_threshold: float,
        target_threshold: float,
        preserve_dims: str,
        op_func: Callable = operator.ge,
    ) -> scores.categorical.BasicContingencyManager:
        """Create and transform a contingency manager.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            forecast_threshold: Threshold for binarizing forecast.
            target_threshold: Threshold for binarizing target.
            preserve_dims: Dimension(s) to preserve during transform.

        Returns:
            Transformed contingency manager.
        """
        # Apply thresholds to binarize the data
        binary_forecast = (op_func(forecast, forecast_threshold)).astype(float)
        binary_target = (op_func(target, target_threshold)).astype(float)

        # Create and transform contingency manager
        binary_contingency_manager = scores.categorical.BinaryContingencyManager(
            binary_forecast, binary_target
        )
        transformed = binary_contingency_manager.transform(preserve_dims=preserve_dims)

        return transformed

    def maybe_expand_composite(self) -> Sequence["BaseMetric"]:
        """Expand composite metrics into individual metrics.

        Returns sub-metrics for composites, or [self] for regular.

        Returns:
            List of metrics to evaluate.
        """
        if self._metric_instances:
            return self._metric_instances
        return [self]

    def is_composite(self) -> bool:
        """Check if this is a composite metric.

        Returns:
            True if composite (has sub-metrics), False otherwise.
        """
        return bool(self._metric_instances)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute metric (not supported for ThresholdMetric base).

        ThresholdMetric must be subclassed (like CSI, FAR) or used
        as a composite with metrics list.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "ThresholdMetric._compute_metric must be implemented "
            "by subclasses (CSI, FAR, etc.) or use ThresholdMetric "
            "as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
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

        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=forecast_threshold,
                target_threshold=target_threshold,
                preserve_dims=preserve_dims,
            )
        return transformed.accuracy()


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


# TODO: fill landfall displacement out
class LandfallDisplacement(BaseMetric):
    """Spatial displacement error of landfall location.

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
        raise NotImplementedError("LandfallDisplacement is not implemented yet")


# TODO: complete landfall time mean error implementation
class LandfallTimeME(BaseMetric):
    """Mean error of landfall time.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("LandfallTimeME", *args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__("LandfallIntensityMAE", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LandfallIntensityMAE is not implemented yet")


# TODO: complete spatial displacement implementation
class SpatialDisplacement(BaseMetric):
    """Spatial displacement error metric.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("SpatialDisplacement", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("SpatialDisplacement is not implemented yet")


# TODO: complete lead time detection implementation
class LeadTimeDetection(BaseMetric):
    """Lead time detection metric.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("LeadTimeDetection", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("LeadTimeDetection is not implemented yet")


# TODO: complete regional hits and misses implementation
class RegionalHitsMisses(BaseMetric):
    """Regional hits and misses metric.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("RegionalHitsMisses", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("RegionalHitsMisses is not implemented yet")


# TODO: complete hits and misses implementation
class HitsMisses(BaseMetric):
    """Hits and misses metric.

    Note: Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__("HitsMisses", *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("HitsMisses is not implemented yet")
