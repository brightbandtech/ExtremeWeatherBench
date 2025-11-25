import abc
import logging
import operator
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union

import numpy as np
import scores
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc, derived, utils

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

    Args:
        name: The name of the metric.
        preserve_dims: The dimensions to preserve in the computation. Defaults to
        "lead_time".
        forecast_variable: The forecast variable to use in the computation.
        target_variable: The target variable to use in the computation.
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

        # If the forecast or target is sparse, densify it.
        # Ideally we would keep the sparse data structure, but Dask and sparse
        # do not play well together as of Nov 2025.
        forecast = utils.maybe_densify_dataarray(forecast)
        target = utils.maybe_densify_dataarray(target)
        return self._compute_metric(forecast, target, **kwargs)

    def maybe_expand_composite(self) -> Sequence["BaseMetric"]:
        """Expand composite metrics into individual metrics.

        Base implementation returns [self]. Override for composites.

        Returns:
            List containing just this metric.
        """
        return [self]

    def is_composite(self) -> bool:
        """Check if this is a composite metric.

        Base implementation returns False. Override for composites.

        Returns:
            False for base metrics.
        """
        return False

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs,
    ) -> dict:
        """Prepare kwargs for metric evaluation.

        Base implementation just returns kwargs as-is.
        Override for metrics that need special preparation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.
            **base_kwargs: Base kwargs to include in result.

        Returns:
            Dictionary of kwargs (unchanged for base metrics).
        """
        return base_kwargs.copy()


class CompositeMetric(BaseMetric):
    """Base class for composite metrics.

    This class provides common functionality for composite metrics.

    Args:
        name: The name of the metric.
        preserve_dims: The dimensions to preserve in the computation. Defaults to
        "lead_time".
        forecast_variable: The forecast variable to use in the computation.
        target_variable: The target variable to use in the computation.
        *args: Additional arguments to pass to the metric.
        **kwargs: Additional keyword arguments to pass to the metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_instances: list["BaseMetric"] = []

    def maybe_expand_composite(self) -> Sequence["BaseMetric"]:
        """Expand composite metrics into individual metrics.

        Returns:
            List containing just this metric.
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

    @abc.abstractmethod
    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Returns:
            Dictionary of kwargs (unchanged for composite metrics).
        """

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute metric (not supported for CompositeMetric base).

        CompositeMetric must be subclassed (like ThresholdMetric, LandfallMetric)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "CompositeMetric._compute_metric must be implemented "
            "by subclasses (ThresholdMetric, LandfallMetric) or use "
            "CompositeMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class ThresholdMetric(CompositeMetric):
    """Base class for threshold-based metrics.

    This class provides common functionality for metrics that require
    forecast and target thresholds for binarization.

    Args:
        name: The name of the metric. Defaults to "threshold_metrics".
        preserve_dims: The dimensions to preserve in the computation. Defaults to
        "lead_time".
        forecast_variable: The forecast variable to use in the computation.
        target_variable: The target variable to use in the computation.
        forecast_threshold: The threshold for binarizing the forecast. Defaults to 0.5.
        target_threshold: The threshold for binarizing the target. Defaults to 0.5.
        metrics: A list of metrics to use as a composite. Defaults to None.
        *args: Additional arguments to pass to the metric
        **kwargs: Additional keyword arguments to pass to the metric

    Can be used in two ways:
    1. As a base class for specific threshold metrics (CriticalSuccessIndex,
    FalseAlarmRatio, etc.)
    2. As a composite metric to compute multiple threshold metrics
       efficiently by reusing the transformed contingency manager.

    Example of composite usage:
        composite = ThresholdMetric(
            metrics=[CriticalSuccessIndex, FalseAlarmRatio, Accuracy],
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
        if self.metrics is not None:
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
        op_func: Union[
            Callable, Literal[">", ">=", "<", "<=", "==", "!="]
        ] = operator.ge,
    ) -> scores.categorical.BasicContingencyManager:
        """Create and transform a contingency manager.

        This method is used to create and transform a contingency manager from the
        scores module. The op_func is used to binarize the forecast and target data with
        either a string representation of the operator, e.g. ">=", or a callable
        function from the operator module, e.g. operator.ge.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            forecast_threshold: Threshold for binarizing forecast.
            target_threshold: Threshold for binarizing target.
            preserve_dims: Dimension(s) to preserve during transform.
            op_func: Function or string representation of the operator to apply to the
            forecast and target. Defaults to operator.ge (greater than or equal to).

        Returns:
            Transformed contingency manager.
        """
        # Apply thresholds to binarize the data
        op_func = utils.maybe_get_operator(op_func)
        binary_forecast = utils.maybe_densify_dataarray(
            op_func(forecast, forecast_threshold)
        ).astype(float)
        binary_target = utils.maybe_densify_dataarray(
            op_func(target, target_threshold)
        ).astype(float)

        # Create and transform contingency manager
        binary_contingency_manager = scores.categorical.BinaryContingencyManager(
            binary_forecast, binary_target
        )
        transformed = binary_contingency_manager.transform(preserve_dims=preserve_dims)

        return transformed

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Computes the transformed contingency manager once and adds
        it to kwargs for efficient composite evaluation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.
            **base_kwargs: Base kwargs to include in result.

        Returns:
            Dictionary of kwargs including transformed_manager.
        """
        kwargs = base_kwargs.copy()

        if self.is_composite() and len(self._metric_instances) > 1:
            kwargs["transformed_manager"] = self.transformed_contingency_manager(
                forecast=forecast_data,
                target=target_data,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
            kwargs["forecast_threshold"] = self.forecast_threshold
            kwargs["target_threshold"] = self.target_threshold
            kwargs["preserve_dims"] = self.preserve_dims

        return kwargs

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute metric (not supported for ThresholdMetric base).

        ThresholdMetric must be subclassed (like CriticalSuccessIndex, FalseAlarmRatio)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "ThresholdMetric._compute_metric must be implemented "
            "by subclasses (CriticalSuccessIndex, FalseAlarmRatio, etc.) or use "
            "ThresholdMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class CriticalSuccessIndex(ThresholdMetric):
    """Critical Success Index metric.

    The Critical Success Index is computed between the forecast and target using the
    preserve_dims dimensions.

    Args:
        name: The name of the metric. Defaults to "CriticalSuccessIndex".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The Critical Success Index between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "CriticalSuccessIndex", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.critical_success_index()


class FalseAlarmRatio(ThresholdMetric):
    """False Alarm Ratio metric.

    The False Alarm Ratio is computed between the forecast and target using the
    preserve_dims dimensions. Note that this is not the same as the False Alarm Rate.

    Args:
        name: The name of the metric. Defaults to "FalseAlarmRatio".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The False Alarm Ratio between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "FalseAlarmRatio", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.false_alarm_ratio()


class TruePositives(ThresholdMetric):
    """True Positive ratio.

    The True Positive is the number of times the forecast is a true positive (top right
    cell in the contingency table) divided by the total number of observations.

    Args:
        name: The name of the metric. Defaults to "TruePositives".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The True Positive ratio between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "TruePositives", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["tp_count"] / counts["total_count"]


class FalsePositives(ThresholdMetric):
    """False Positive ratio.

    The False Positive is the number of times the forecast is a false positive divided
    by the total number of observations.

    Args:
        name: The name of the metric. Defaults to "FalsePositives".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The False Positive ratio between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "FalsePositives", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["fp_count"] / counts["total_count"]


class TrueNegatives(ThresholdMetric):
    """True Negative ratio.

    The True Negative is the number of times the forecast is a true negative divided by
    the total number of observations.

    Args:
        name: The name of the metric. Defaults to "TrueNegatives".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The True Negative ratio between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "TrueNegatives", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["tn_count"] / counts["total_count"]


class FalseNegatives(ThresholdMetric):
    """False Negative ratio.

    The False Negative is the number of times the forecast is a false negative (top left
    cell in the contingency table) divided by the total number of observations.

    Args:
        name: The name of the metric. Defaults to "FalseNegatives".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The False Negative ratio between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "FalseNegatives", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        return counts["fn_count"] / counts["total_count"]


class Accuracy(ThresholdMetric):
    """Accuracy metric.

    The Accuracy is the number of times the forecast is correct (top right or bottom
    right cell in the contingency table) divided by the total number of observations, or
    (true positives + true negatives) / (total number of samples).

    Args:
        name: The name of the metric. Defaults to "Accuracy".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The Accuracy between the forecast and target as a DataArray.
    """

    def __init__(self, name: str = "Accuracy", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # Use pre-computed manager if provided, else compute
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        return transformed.accuracy()


class MeanSquaredError(BaseMetric):
    """Mean Squared Error metric.

    Args:
        name: The name of the metric. Defaults to "MeanSquaredError".
        interval_where_one: From scores: endpoints of the interval where the threshold
        weights are 1. Must be increasing. Infinite endpoints are permissible. By
        supplying a tuple of arrays, endpoints can vary with dimension.
        interval_where_positive: From scores:endpoints of the interval where the
        threshold weights are positive. Must be increasing. Infinite endpoints are only
        permissible when the corresponding interval_where_one endpoint is infinite. By
        supplying a tuple of arrays, endpoints can vary with dimension.
        weights: From scores: an array of weights to apply to the score (e.g., weighting
        a grid by latitude). If None, no weights are applied. If provided, the weights
        must be broadcastable to the data dimensions and must not contain negative or
        NaN values. If appropriate, users can choose to replace NaN values in weights
        by calling weights.fillna(0). The weighting approach follows
        xarray.computation.weighted.DataArrayWeighted. See the scores weighting tutorial
        for more information on how to use weights.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The Mean or threshold-weighted Squared Error between the forecast and target
        as a DataArray.
    """

    def __init__(
        self,
        name: str = "MeanSquaredError",
        interval_where_one: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        interval_where_positive: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        weights: Optional[xr.DataArray] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.interval_where_one = interval_where_one
        self.interval_where_positive = interval_where_positive
        self.weights = weights

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        if self.interval_where_one is not None:
            return scores.continuous.tw_squared_error(
                forecast,
                target,
                interval_where_one=self.interval_where_one,
                interval_where_positive=self.interval_where_positive,
                weights=self.weights,
                preserve_dims=self.preserve_dims,
            )
        return scores.continuous.mse(forecast, target, preserve_dims=self.preserve_dims)


class MeanAbsoluteError(BaseMetric):
    """Mean Absolute Error metric.


    Args:
        name: The name of the metric. Defaults to "MeanAbsoluteError".
        interval_where_one: From scores: endpoints of the interval where the threshold
        weights are 1. Must be increasing. Infinite endpoints are permissible. By
        supplying a tuple of arrays, endpoints can vary with dimension.
        interval_where_positive: From scores:endpoints of the interval where the
        threshold weights are positive. Must be increasing. Infinite endpoints are only
        permissible when the corresponding interval_where_one endpoint is infinite. By
        supplying a tuple of arrays, endpoints can vary with dimension.
        weights: From scores: an array of weights to apply to the score (e.g., weighting
        a grid by latitude). If None, no weights are applied. If provided, the weights
        must be broadcastable to the data dimensions and must not contain negative or
        NaN values. If appropriate, users can choose to replace NaN values in weights
        by calling weights.fillna(0). The weighting approach follows
        xarray.computation.weighted.DataArrayWeighted. See the scores weighting tutorial
        for more information on how to use weights.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The Mean or threshold-weighted Absolute Error between the forecast and target
        as a DataArray.
    """

    def __init__(
        self,
        name: str = "MeanAbsoluteError",
        interval_where_one: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        interval_where_positive: Optional[
            tuple[int | float | xr.DataArray, int | float | xr.DataArray]
        ] = None,
        weights: Optional[xr.DataArray] = None,
        *args,
        **kwargs,
    ):
        self.interval_where_one = interval_where_one
        self.interval_where_positive = interval_where_positive
        self.weights = weights
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
        if self.interval_where_one is not None:
            return scores.continuous.tw_absolute_error(
                forecast,
                target,
                interval_where_one=self.interval_where_one,
                interval_where_positive=self.interval_where_positive,
                weights=self.weights,
                preserve_dims=self.preserve_dims,
            )
        return scores.continuous.mae(forecast, target, preserve_dims=self.preserve_dims)


class MeanError(BaseMetric):
    """Mean Error (bias) metric.

    The mean error (or mean bias error) is computed between the forecast and target
    using the preserve_dims dimensions.

    Args:
        name: The name of the metric. Defaults to "MeanError".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The mean error between the forecast and target.
    """

    def __init__(self, name: str = "MeanError", *args, **kwargs):
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
        return scores.continuous.mean_error(
            forecast, target, preserve_dims=self.preserve_dims
        )


class RootMeanSquaredError(BaseMetric):
    """Root Mean Square Error metric.

    The Root Mean Square Error is computed between the forecast and target using the
    preserve_dims dimensions.

    Args:
        name: The name of the metric. Defaults to "RootMeanSquaredError".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.


    Args:
        name: The name of the metric. Defaults to "RootMeanSquaredError".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, name: str = "RootMeanSquaredError", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

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
        return scores.continuous.rmse(
            forecast, target, preserve_dims=self.preserve_dims
        )


class EarlySignal(BaseMetric):
    """Early Signal detection metric.

    This metric finds the first occurrence where a signal is detected based on
    threshold criteria and returns the corresponding init_time, lead_time, and
    valid_time information. The metric is designed to be flexible for different
    signal detection criteria that can be specified in applied metrics downstream.

    Args:
        name: The name of the metric.
        comparison_operator: The comparison operator to use for signal detection.
        threshold: The threshold value for signal detection.
        spatial_aggregation: The spatial aggregation method to use for signal detection.
            any: Return True if any gridpoint meets the criteria.
            all: Return True if all gridpoints meet the criteria.
            half: Return True if at least half of the gridpoints meet the criteria.
        **kwargs: Additional keyword arguments. Supported kwargs:
            preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".

    Returns:
        A DataArray with dims [init_time, lead_time] indicating
        whether criteria are met for each init_time and lead_time pair.
    """

    def __init__(
        self,
        name: str = "EarlySignal",
        comparison_operator: Union[
            Callable, Literal[">", ">=", "<", "<=", "==", "!="]
        ] = ">=",
        threshold: float = 0.5,
        spatial_aggregation: Literal["any", "all", "half"] = "any",
        **kwargs,
    ):
        # Extract threshold params before passing to super
        self.comparison_operator = utils.maybe_get_operator(comparison_operator)
        self.threshold = threshold
        self.spatial_aggregation = spatial_aggregation
        super().__init__(name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataarray with init_time, lead_time, valid_time
            target: The target dataarray (used for reference/validation)
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
                name=self.name,
            )
        # Create detection mask
        detection_mask = self.comparison_operator(forecast, self.threshold)

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
            elif self.spatial_aggregation == "half":
                detection_mask = operator.ge(detection_mask.mean(spatial_dims), 0.5)
            else:
                raise ValueError(
                    f"Spatial aggregation '{self.spatial_aggregation}' not supported"
                )

        detection_mask.name = self.name
        return detection_mask


class MaximumMeanAbsoluteError(MeanAbsoluteError):
    """Computes the mean absolute error between the forecast and target maximum values.

    The forecast is filtered to a time window around the target's maximum using
    tolerance_range_hours (in the event of variation between the timing between the
    target and forecast maximum values). The mean absolute error is computed between the
    filtered forecast and target maximum value.

    Args:
        tolerance_range_hours: The time window (hours) around the target's maximum
        value to search for forecast minimum. Defaults to 24 hours.
        name: The name of the metric. Defaults to "MaximumMeanAbsoluteError".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The mean absolute error between the forecast and target maximum values.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        name: str = "MaximumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        self.tolerance_range_hours = tolerance_range_hours
        self.reduce_spatial_dims = reduce_spatial_dims
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> dict[str, xr.DataArray]:
        """Compute MaximumMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. None currently supported in
            MaximumMeanAbsoluteError.
        Returns:
            MeanAbsoluteError of the maximum values.
        """
        # Enforced spatial reduction for MaximumMeanAbsoluteError
        reduce_spatial_dims = ["latitude", "longitude"]
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        ).compute()
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        filtered_max_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= maximum_timestep.data
                - np.timedelta64(self.tolerance_range_hours // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= maximum_timestep.data
                + np.timedelta64(self.tolerance_range_hours // 2, "h")
            ),
            drop=True,
        ).max("valid_time")
        return super()._compute_metric(
            forecast=filtered_max_forecast,
            target=maximum_value,
            preserve_dims=self.preserve_dims,
        )


class MinimumMeanAbsoluteError(MeanAbsoluteError):
    """Computes the mean absolute error between the forecast and target minimum values.

    The forecast is filtered to a time window around the target's minimum using
    tolerance_range_hours (in the event of variation between the timing between the
    target and forecast minimum values). The mean absolute error is computed between the
    filtered forecast and target minimum value.

    Args:
        tolerance_range_hours: The time window (hours) around the target's minimum
        value to search for forecast minimum. Defaults to 24 hours.
        name: The name of the metric. Defaults to "MinimumMeanAbsoluteError".
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The mean absolute error between the forecast and target minimum values.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        name: str = "MinimumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        self.tolerance_range_hours = tolerance_range_hours
        self.reduce_spatial_dims = reduce_spatial_dims
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MinimumMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. None currently supported in
            MinimumMeanAbsoluteError.
        Returns:
            MeanAbsoluteError of the minimum values.
        """
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
        filtered_min_forecast = forecast_spatial_mean.where(
            (
                forecast_spatial_mean.valid_time
                >= minimum_timestep.data
                - np.timedelta64(self.tolerance_range_hours // 2, "h")
            )
            & (
                forecast_spatial_mean.valid_time
                <= minimum_timestep.data
                + np.timedelta64(self.tolerance_range_hours // 2, "h")
            ),
            drop=True,
        ).min("valid_time")
        return super()._compute_metric(
            forecast=filtered_min_forecast,
            target=minimum_value,
            preserve_dims=self.preserve_dims,
        )


class MaximumLowestMeanAbsoluteError(MeanAbsoluteError):
    """Mean Absolute Error of the maximum of aggregated minimum values.

    Meant for heatwave evaluation by aggregating the minimum values over a day and then
    computing the MeanAbsoluteError between the warmest nighttime (daily minimum)
    temperature in the target and forecast.

    Args:
        name: The name of the metric. Defaults to "MaximumLowestMeanAbsoluteError"
        *args: Additional arguments
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        name: str = "MaximumLowestMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        self.tolerance_range_hours = tolerance_range_hours
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MaximumLowestMeanAbsoluteError.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments. Supported kwargs:
                preserve_dims (str): Dimension(s) to preserve. Defaults to "lead_time".
                tolerance_range (int): Time window (hours) around target's max-min
                value to search for forecast max-min. Defaults to 24 hours.

        Returns:
            MeanAbsoluteError of the highest aggregated minimum value.
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
                        - np.timedelta64(self.tolerance_range_hours // 2, "h")
                    )
                )
                & (
                    forecast.valid_time
                    <= (
                        max_min_target_datetime.data
                        + np.timedelta64(self.tolerance_range_hours // 2, "h")
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
            preserve_dims=self.preserve_dims,
        )


class DurationMeanError(MeanError):
    """Compute the duration of a case's event.
    This metric computes the mean error between the forecast and target durations.

    Args:
        threshold_criteria: The criteria for event detection. Can be either a DataArray
        of a climatology with dimensions (dayofyear, hour, latitude, longitude) or a
        float value representing a fixed threshold.
        op_func: Comparison operator or string (e.g., operator.ge for >=)
        name: Name of the metric
        preserve_dims: Dimensions to preserve during aggregation. Defaults to
        "init_time".
    """

    def __init__(
        self,
        threshold_criteria: xr.DataArray | float,
        op_func: Union[Callable, Literal[">", ">=", "<", "<=", "==", "!="]] = ">=",
        name: str = "duration_me",
        preserve_dims: str = "init_time",
    ):
        super().__init__(name=name, preserve_dims=preserve_dims)
        self.threshold_criteria = threshold_criteria
        self.op_func = utils.maybe_get_operator(op_func)

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
            Mean error between forecast and target event durations
        """
        spatial_dims = [
            dim
            for dim in forecast.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
        ]
        # Handle criteria - either climatology (xr.DataArray) or float threshold
        if isinstance(self.threshold_criteria, xr.DataArray):
            # Climatology case, convert from dayofyear/hour to valid_time
            self.threshold_criteria = utils.convert_day_yearofday_to_time(
                self.threshold_criteria, forecast.valid_time.dt.year.values[0]
            )

            # Interpolate climatology to target coordinates
            self.threshold_criteria = utils.interp_climatology_to_target(
                target, self.threshold_criteria
            )
        forecast = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=spatial_dims
        )
        target = utils.reduce_dataarray(target, method="mean", reduce_dims=spatial_dims)
        forecast = forecast.compute()
        target = target.compute()
        forecast_mask = self.op_func(forecast, self.threshold_criteria)
        target_mask = self.op_func(target, self.threshold_criteria)
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
        forecast_duration = forecast_mask_final.groupby(self.preserve_dims).sum(
            skipna=True
        )
        target_duration = target_mask_final.groupby(self.preserve_dims).sum(skipna=True)

        # TODO: product of time resolution hours and duration
        return super()._compute_metric(
            forecast=forecast_duration,
            target=target_duration,
            preserve_dims=self.preserve_dims,
        )


class LandfallMetric(CompositeMetric):
    """Base class for landfall metrics.

    Landfall metrics compute landfalls using the calc.find_landfalls function, which
    utilizes a land geometry and line segments based on track data to determine
    intersections.

    Can be used as a base class for custom landfall metrics, as a mixin with other
    metrics, or as a composite metric for multiple landfall metrics (which utilize
    identical landfalling locations).
    """

    def __init__(
        self,
        name: str = "landfall_metrics",
        preserve_dims: str = "init_time",
        approach: Literal["first", "next"] = "first",
        exclude_post_landfall: bool = False,
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        metrics: Optional[list[Type["LandfallMetric"]]] = None,
        *args,
        **kwargs,
    ):
        """Initialize LandfallMetric.

        Landfalls are detected using the calc.find_landfalls function, which utilizes a
        land geometry and line segments based on coordinates to determine intersections.

        Using approach, "first" will calculate the first detected landfall for an entire
        forecast, i.e. later landfalls in a multi-landfall event will not be considered.
        "next" will calculate the next landfall for each init_time.
        Using Ida as an example (case 220), "first" would only run calculations for the
        first landfall in Cuba, ignoring the later US landfall. "next" would run
        calculations for the first landfall in Cuba, then the next landfall in the US,
        etc. based on the init_time and when landfall occurred.

        Args:
            name: The name of the metric. Defaults to "landfall_metrics" for the base
            class
            preserve_dims: The dimensions to preserve. Defaults to "init_time"
            approach: The approach to use for landfall detection. Defaults to "first"
            exclude_post_landfall: Whether to exclude post-landfall data. Defaults to
            False
            forecast_variable: The forecast variable to use. Defaults to None
            target_variable: The target variable to use. Defaults to None
            metrics: A list of metrics to use as a composite. Defaults to None
            *args: Additional arguments to pass to the metric
            **kwargs: Additional keyword arguments to pass to the metric
        """
        super().__init__(
            name=name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            *args,
            **kwargs,
        )
        self.approach = approach
        self.exclude_post_landfall = exclude_post_landfall
        self.metrics = metrics or []

        # If metrics provided, instantiate them
        if self.metrics is not None:
            self._metric_instances = [
                (
                    metric_cls(
                        preserve_dims=self.preserve_dims,
                        forecast_variable=self.forecast_variable,
                        target_variable=self.target_variable,
                    )
                    if isinstance(metric_cls, type)
                    else metric_cls
                )
                for metric_cls in self.metrics
            ]
        else:
            self._metric_instances = []

    def __call__(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the metric.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments
        """
        return self.compute_metric(forecast, target, **kwargs)

    def compute_landfalls(self, forecast: xr.DataArray, target: xr.DataArray) -> Any:
        """Compute landfalls for a given forecast and target dataarray.

        This function computes the landfalls for a given forecast and target dataarray
        using calc.find_landfalls. Currently, this access pattern doesn't include
        passing land geometry in, but calc.find_landfalls will use NaturalEarth's 10m
        land
        geometry by default.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray

        Returns:
            Tuple of forecast and target landfalls, or None if no landfalls are found
        """
        # For "first" approach: get only first landfall
        # For "next" approach: get all target landfalls, then filter
        return_next_landfall = self.approach == "next"

        # Get first forecast landfall per init_time
        forecast_landfalls = calc.find_landfalls(
            forecast, return_next_landfall=return_next_landfall
        )

        if forecast_landfalls is None:
            return None, None

        # Get only first forecast landfall per init_time
        if "landfall" in forecast_landfalls.dims:
            forecast_landfalls = forecast_landfalls.isel(landfall=0)

        # Get all target landfalls
        target_landfalls_pre_init = calc.find_landfalls(
            target, return_next_landfall=return_next_landfall
        )

        if target_landfalls_pre_init is None:
            return None, None

        if return_next_landfall:
            # Find next target landfall for each init_time
            target_landfalls = calc.find_next_landfall_for_init_time(
                forecast_landfalls, target_landfalls_pre_init
            )
        else:
            if "landfall" in target_landfalls_pre_init.dims:
                target_landfalls = target_landfalls_pre_init.isel(landfall=0)
            else:
                target_landfalls = target_landfalls_pre_init
            forecast_landfalls = forecast_landfalls.where(
                forecast_landfalls.init_time < target_landfalls.valid_time.values,
                drop=True,
            )
        return forecast_landfalls, target_landfalls

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation

        Computes the landfalls once and adds them to kwargs to avoid recomputing when
        used as a composite metric.

        Args:
            forecast_data: The forecast DataArray
            target_data: The target DataArray
            **base_kwargs: Base kwargs to include in result

        Returns:
            Dictionary of kwargs including transformed_manager
        """
        kwargs = base_kwargs.copy()

        if self.is_composite() and len(self._metric_instances) > 1:
            kwargs["forecast_landfall"], kwargs["target_landfall"] = (
                self.compute_landfalls(forecast=forecast_data, target=target_data)
            )

        kwargs["preserve_dims"] = self.preserve_dims

        return kwargs

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute metric (not supported for LandfallMetric base)

        LandfallMetric must be subclassed (like LandfallDisplacement,
        LandfallTimeMeanError)
        or used as a composite with metrics list.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments
        """
        raise NotImplementedError(
            "LandfallMetric._compute_metric must be implemented "
            "by subclasses (LandfallDisplacement, LandfallTimeMeanError, etc.) or use "
            "LandfallMetric as a composite with metrics=[...] list. Composites are "
            "automatically expanded in the evaluation pipeline."
        )


class SpatialDisplacement(BaseMetric):
    """Spatial displacement error metric for atmospheric rivers and similar events.

    Computes the great circle distance between the center of mass of forecast
    and target spatial patterns.

    Args:
        name: The name of the metric. Defaults to "spatial_displacement".
        **kwargs: Additional keyword arguments.

    Returns:
        The spatial displacement between the forecast and target as a DataArray.
    """

    def __init__(
        self,
        name: str = "spatial_displacement",
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute spatial displacement.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments.

        Returns:
            The spatial displacement between the forecast and target as a DataArray.
        """

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


class LandfallDisplacement(LandfallMetric, BaseMetric):
    """Calculate the distance between forecast and target landfall positions.

    This metric computes the distance between the forecast and target
    landfall positions, defaulting to kilometers.

    Args:
        name: The name of the metric. Defaults to "landfall_displacement"
        approach: The approach to use for landfall detection. Defaults to "first"
        exclude_post_landfall: Whether to exclude post-landfall data. Defaults to False
    """

    def __init__(
        self,
        name: str = "landfall_displacement",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.units = kwargs.get("units", "km")

    @staticmethod
    def _calculate_distance(
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
        units: Literal["km", "kilometers", "deg", "degrees"] = "km",
    ) -> xr.DataArray:
        """Calculate the distance between two landfall points in kilometers or degrees.

        Handles both scalar and multi-dimensional (with init_time) DataArrays.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray
            target_landfall: Target landfall xarray DataArray
            units: The units to use for the distance. Defaults to "km"
        Returns:
            Distance in the specified units as xarray DataArray
        """
        if forecast_landfall is None or target_landfall is None:
            return xr.DataArray(np.nan)

        # Find common init_times between forecast and target
        init_times_1 = set(forecast_landfall.coords["init_time"].values)
        init_times_2 = set(target_landfall.coords["init_time"].values)
        common_init_times = sorted(init_times_1.intersection(init_times_2))

        if not common_init_times:
            return xr.DataArray(np.nan)

        # Compute distance for each common init_time
        distances = []
        for init_time in common_init_times:
            f_lat = forecast_landfall.coords["latitude"].sel(init_time=init_time).values
            f_lon = (
                forecast_landfall.coords["longitude"].sel(init_time=init_time).values
            )
            t_lat = target_landfall.coords["latitude"].sel(init_time=init_time).values
            t_lon = target_landfall.coords["longitude"].sel(init_time=init_time).values

            # Skip if any coordinates are NaN
            if (
                np.any(np.isnan(f_lat))
                or np.any(np.isnan(f_lon))
                or np.any(np.isnan(t_lat))
                or np.any(np.isnan(t_lon))
            ):
                distances.append(np.nan)
            else:
                dist = calc.haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units=units
                )
                # Ensure we append a scalar value
                distances.append(
                    float(dist.item()) if hasattr(dist, "item") else float(dist)
                )

        return xr.DataArray(
            distances,
            dims=["init_time"],
            coords={"init_time": common_init_times},
        )

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall displacement metric."""
        forecast_landfall, target_landfall = (
            kwargs.get("forecast_landfall", None),
            kwargs.get("target_landfall", None),
        )
        if forecast_landfall is None or target_landfall is None:
            forecast_landfall, target_landfall = self.compute_landfalls(
                forecast=forecast, target=target
            )
        if forecast_landfall is None or target_landfall is None:
            return xr.DataArray(np.nan)
        return self._calculate_distance(
            forecast_landfall,
            target_landfall,
            units=self.units,
        )


class LandfallTimeMeanError(LandfallMetric, MeanError):
    """Landfall time mean error.

    This metric computes the mean error between the forecast and target landfall times.
    A positive value indicates the forecast landfall time is later than the target
    landfall time, a negative value indicates the forecast landfall time is earlier than
    the target landfall time.

    Args:
        name: The name of the metric. Defaults to "landfall_time_me"
        approach: The approach to use for landfall detection. Defaults to "first"
        exclude_post_landfall: Whether to exclude post-landfall data. Defaults to False
    """

    def __init__(
        self,
        name: str = "landfall_time_me",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

    @staticmethod
    def _calculate_time_difference(
        forecast_landfall: xr.DataArray, target_landfall: xr.DataArray
    ) -> xr.DataArray:
        """Calculate the time difference between two landfall points in hours

        Args:
            forecast_landfall: Forecast landfall xarray DataArray
            target_landfall: Target landfall xarray DataArray

        Returns:
            Time difference in hours (forecast_landfall - target_landfall)
            as xarray DataArray with init_time dimension
        """
        if forecast_landfall is None or target_landfall is None:
            return xr.DataArray(np.nan)

        # Find common init_times between forecast and target
        init_times_1 = set(forecast_landfall.coords["init_time"].values)
        init_times_2 = set(target_landfall.coords["init_time"].values)
        common_init_times = sorted(init_times_1.intersection(init_times_2))

        if not common_init_times:
            return xr.DataArray(np.nan)

        # Calculate time difference for each common init_time
        time_diffs = []
        for init_time in common_init_times:
            time1 = forecast_landfall.coords["valid_time"].sel(init_time=init_time)
            time2 = target_landfall.coords["valid_time"].sel(init_time=init_time)

            # Calculate time difference in hours
            time_diff = (time1 - time2) / np.timedelta64(1, "h")
            time_diffs.append(float(time_diff.values))

        return xr.DataArray(
            time_diffs,
            dims=["init_time"],
            coords={"init_time": common_init_times},
        )

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall time metric."""
        forecast_landfall, target_landfall = (
            kwargs.get("forecast_landfall", None),
            kwargs.get("target_landfall", None),
        )

        if forecast_landfall is None or target_landfall is None:
            forecast_landfall, target_landfall = self.compute_landfalls(
                forecast=forecast, target=target
            )
        if forecast_landfall is None or target_landfall is None:
            return xr.DataArray(np.nan)
        return self._calculate_time_difference(forecast_landfall, target_landfall)


class LandfallIntensityMeanAbsoluteError(LandfallMetric, MeanAbsoluteError):
    """Compute the MeanAbsoluteError between forecast and target

    This metric computes the mean absolute error between forecast and target
    intensity at landfall.

    The intensity variable is determined by forecast_variable and
    target_variable. To evaluate multiple intensity variables (e.g.,
    surface_wind_speed and air_pressure_at_mean_sea_level), create
    separate metric instances for each variable.

    Args:
        name: The name of the metric. Defaults to "landfall_intensity_mae"
        approach: The approach to use for landfall detection. Defaults to "first"
        exclude_post_landfall: Whether to exclude post-landfall data. Defaults to False
        forecast_variable: Variable to use for forecast intensity
        target_variable: Variable to use for target intensity
    """

    def __init__(
        self,
        name: str = "landfall_intensity_mae",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)

    @staticmethod
    def _compute_absolute_error(
        forecast_landfall: xr.DataArray, target_landfall: xr.DataArray
    ) -> xr.DataArray:
        """Compute absolute error between landfall intensity values

        Args:
            forecast_landfall: Forecast intensity at landfall
            target_landfall: Target intensity at landfall

        Returns:
            Absolute error at landfall points
        """
        # Landfall points are already extracted, just compute absolute error
        return np.abs(forecast_landfall - target_landfall)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall intensity metric."""
        forecast_landfall, target_landfall = (
            kwargs.get("forecast_landfall", None),
            kwargs.get("target_landfall", None),
        )
        if forecast_landfall is None or target_landfall is None:
            forecast_landfall, target_landfall = self.compute_landfalls(
                forecast=forecast, target=target
            )
        if forecast_landfall is None or target_landfall is None:
            return xr.DataArray(np.nan)
        return self._compute_absolute_error(forecast_landfall, target_landfall)
