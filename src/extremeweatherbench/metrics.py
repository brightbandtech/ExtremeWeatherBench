import abc
import logging
import operator
from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import scores
import xarray as xr
from scipy import ndimage
import pandas as pd

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
        new_cls = super().__new__(cls, name, bases, namespace)
        # NOTE: the `compute_metric()` method will be defined in the ABC `BaseMetric`,
        # and we never expect the user re-implement it. So it won't be in the namespace
        # of the concrete metric classes - it will only be in the namespace of the ABC
        # `BaseMetric`, and will be available as an attribute of the concrete metric
        # classes.
        if (
            "_compute_metric" in namespace
            and hasattr(new_cls, "compute_metric")
            and new_cls._compute_metric.__doc__ is not None
        ):
            # Create a new method for _this_ class, so we can avoid overwriting what
            # we set for the parent.
            _original_compute_metric = new_cls.compute_metric

            def _compute_metric_with_docstring(self, *args, **kwargs):
                return _original_compute_metric(self, *args, **kwargs)

            _compute_metric_with_docstring.__doc__ = new_cls._compute_metric.__doc__
            new_cls.compute_metric = _compute_metric_with_docstring

        return new_cls


class BaseMetric(abc.ABC, metaclass=ComputeDocstringMetaclass):
    """Abstract base class defining the foundational interface for all metrics.

    Metrics are general operations applied between forecast and analysis xarray
    DataArrays. EWB metrics prioritize the use of any arbitrary sets of
    forecasts and analyses, so long as the spatiotemporal dimensions are the
    same.

    Public methods:
        compute_metric: Public interface to compute the metric
        maybe_expand_composite: Expand composite metrics into individual metrics
        is_composite: Check if this is a composite metric
        __repr__: String representation of the metric
        __eq__: Check equality with another metric

    Abstract methods:
        _compute_metric: Logic to compute the metric (must be implemented)
    """

    def __init__(
        self,
        name: str,
        preserve_dims: str = "lead_time",
        forecast_variable: str | derived.DerivedVariable | None = None,
        target_variable: str | derived.DerivedVariable | None = None,
    ):
        """Initialize the base metric.

        Args:
            name: The name of the metric.
            preserve_dims: The dimensions to preserve in the computation.
                Defaults to "lead_time".
            forecast_variable: The forecast variable to use in the
                computation.
            target_variable: The target variable to use in the computation.
        """
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
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for metric evaluation.

        Base implementation just returns kwargs as-is.
        Override for metrics that need special preparation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

        Returns:
            Dictionary of kwargs (unchanged for base metrics).
        """
        return base_kwargs.copy()


class CompositeMetric(BaseMetric):
    """Base class for composite metrics that can contain multiple sub-metrics.

    Extends BaseMetric to provide functionality for composite metrics that
    aggregate multiple individual metrics for efficient evaluation.

    Public methods:
        maybe_expand_composite: Expand into individual metrics (overrides base)
        is_composite: Check if has sub-metrics (overrides base)

    Abstract methods:
        maybe_prepare_composite_kwargs: Prepare kwargs for composite evaluation
        _compute_metric: Compute the metric (must be implemented by subclasses)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the composite metric.

        Args:
            *args: Positional arguments passed to BaseMetric.__init__
            **kwargs: Keyword arguments passed to BaseMetric.__init__
        """
        super().__init__(*args, **kwargs)
        self._metric_instances: list[BaseMetric] = []

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
    """Base class for threshold-based metrics with binary classification.

    Extends CompositeMetric to provide functionality for metrics that require
    forecast and target thresholds for binarization. Can be used as a base
    class for specific threshold metrics or as a composite metric.

    Public methods:
        transformed_contingency_manager: Create contingency manager
        maybe_prepare_composite_kwargs: Prepare kwargs (overrides parent)
        __call__: Make instances callable with configured thresholds

    Abstract methods:
        _compute_metric: Compute the metric (must be implemented by subclasses)

    Usage patterns:
        1. As a base class for specific metrics (CriticalSuccessIndex, etc.)
        2. As a composite metric to compute multiple threshold metrics
           efficiently by reusing the transformed contingency manager

    Example:
        composite = ThresholdMetric(
            metrics=[CriticalSuccessIndex, FalseAlarmRatio, Accuracy],
            forecast_threshold=0.7,
            target_threshold=0.5
        )
    """

    def __init__(
        self,
        name: str = "threshold_metrics",
        preserve_dims: str = "lead_time",
        forecast_variable: str | derived.DerivedVariable | None = None,
        target_variable: str | derived.DerivedVariable | None = None,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        metrics: list[type["ThresholdMetric"]] | None = None,
        **kwargs,
    ):
        """Initialize the threshold metric.

        Args:
            name: The name of the metric. Defaults to "threshold_metrics".
            preserve_dims: The dimensions to preserve in the computation.
                Defaults to "lead_time".
            forecast_variable: The forecast variable to use in the
                computation.
            target_variable: The target variable to use in the computation.
            forecast_threshold: The threshold for binarizing the forecast.
                Defaults to 0.5.
            target_threshold: The threshold for binarizing the target.
                Defaults to 0.5.
            metrics: A list of metrics to use as a composite. Defaults to
                None.
            **kwargs: Additional keyword arguments passed to parent.
        """
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
        op_func: Callable | Literal[">", ">=", "<", "<=", "==", "!="] = operator.ge,
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
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Computes the transformed contingency manager once and adds
        it to kwargs for efficient composite evaluation.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

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
    """Compute Critical Success Index (CSI) from binary classifications.

    Extends ThresholdMetric to compute CSI between forecast and target using
    the preserve_dims dimensions. CSI measures the fraction of correctly
    predicted events.
    """

    def __init__(self, name: str = "CriticalSuccessIndex", *args, **kwargs):
        """Initialize the Critical Success Index metric.

        Args:
            name: The name of the metric. Defaults to
                "CriticalSuccessIndex".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute False Alarm Ratio (FAR) from binary classifications.

    Extends ThresholdMetric to compute FAR between forecast and target using
    the preserve_dims dimensions. FAR measures the fraction of predicted
    events that did not occur. Note: FAR is not the same as False Alarm Rate.
    """

    def __init__(self, name: str = "FalseAlarmRatio", *args, **kwargs):
        """Initialize the False Alarm Ratio metric.

        Args:
            name: The name of the metric. Defaults to "FalseAlarmRatio".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute True Positive ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of true positives (correctly
    predicted events) to the total number of observations. Corresponds to the
    top right cell in the contingency table.
    """

    def __init__(self, name: str = "TruePositives", *args, **kwargs):
        """Initialize the True Positives metric.

        Args:
            name: The name of the metric. Defaults to "TruePositives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute False Positive ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of false positives
    (incorrectly predicted events) to the total number of observations.
    """

    def __init__(self, name: str = "FalsePositives", *args, **kwargs):
        """Initialize the False Positives metric.

        Args:
            name: The name of the metric. Defaults to "FalsePositives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute True Negative ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of true negatives (correctly
    predicted non-events) to the total number of observations.
    """

    def __init__(self, name: str = "TrueNegatives", *args, **kwargs):
        """Initialize the True Negatives metric.

        Args:
            name: The name of the metric. Defaults to "TrueNegatives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute False Negative ratio from binary classifications.

    Extends ThresholdMetric to compute the ratio of false negatives (missed
    events) to the total number of observations. Corresponds to the top left
    cell in the contingency table.
    """

    def __init__(self, name: str = "FalseNegatives", *args, **kwargs):
        """Initialize the False Negatives metric.

        Args:
            name: The name of the metric. Defaults to "FalseNegatives".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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
    """Compute classification accuracy from binary classifications.

    Extends ThresholdMetric to compute the ratio of correct predictions (true
    positives + true negatives) to the total number of observations. Measures
    overall correctness of the forecast.
    """

    def __init__(self, name: str = "Accuracy", *args, **kwargs):
        """Initialize the Accuracy metric.

        Args:
            name: The name of the metric. Defaults to "Accuracy".
            *args: Additional positional arguments passed to ThresholdMetric.
            **kwargs: Additional keyword arguments passed to ThresholdMetric.
        """
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


class ReceiverOperatingCharacteristic(ThresholdMetric):
    """Receiver Operating Characteristic metric."""

    def __init__(self, name: str = "ReceiverOperatingCharacteristic", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
        op_func = utils.maybe_get_operator(kwargs.get("op_func", operator.ge))

        # Binarize forecast and target using thresholds
        binary_forecast = utils.maybe_densify_dataarray(
            op_func(forecast, kwargs.get("forecast_threshold", self.forecast_threshold))
        ).astype(float)
        binary_target = utils.maybe_densify_dataarray(
            op_func(target, kwargs.get("target_threshold", self.target_threshold))
        ).astype(float)

        return scores.probability.roc_curve_data(
            binary_forecast,
            binary_target,
            thresholds="auto",
            preserve_dims=preserve_dims,
            weights=None,
        )


class ReceiverOperatingCharacteristicSkillScore(ReceiverOperatingCharacteristic):
    """Receiver Operating Characteristic Skill Score metric."""

    def __init__(
        self, name: str = "ReceiverOperatingCharacteristicSkillScore", *args, **kwargs
    ):
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        auc_reference: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        roc_curve_data = super()._compute_metric(forecast, target, **kwargs)
        return (roc_curve_data["AUC"] - auc_reference) / (1 - auc_reference)


class MeanSquaredError(BaseMetric):
    """Compute Mean Squared Error between forecast and target.

    Extends BaseMetric to calculate MSE with optional interval-based
    weighting and custom weights for spatial/temporal averaging.
    """

    def __init__(
        self,
        name: str = "MeanSquaredError",
        interval_where_one: tuple[
            int | float | xr.DataArray, int | float | xr.DataArray
        ]
        | None = None,
        interval_where_positive: tuple[
            int | float | xr.DataArray, int | float | xr.DataArray
        ]
        | None = None,
        weights: xr.DataArray | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the Mean Squared Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanSquaredError".
            interval_where_one: Endpoints of the interval where threshold
                weights are 1. Must be increasing. Infinite endpoints
                permissible.
            interval_where_positive: Endpoints of the interval where threshold
                weights are positive. Must be increasing.
            weights: Array of weights to apply to the score (e.g., latitude
                weighting). If None, no weights are applied.
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
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
    """Compute Mean Absolute Error between forecast and target.

    Extends BaseMetric to calculate MAE with optional interval-based
    weighting and custom weights for spatial/temporal averaging.
    """

    def __init__(
        self,
        name: str = "MeanAbsoluteError",
        interval_where_one: tuple[
            int | float | xr.DataArray, int | float | xr.DataArray
        ]
        | None = None,
        interval_where_positive: tuple[
            int | float | xr.DataArray, int | float | xr.DataArray
        ]
        | None = None,
        weights: xr.DataArray | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the Mean Absolute Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanAbsoluteError".
            interval_where_one: Endpoints of the interval where threshold
                weights are 1. Must be increasing. Infinite endpoints
                permissible.
            interval_where_positive: Endpoints of the interval where threshold
                weights are positive. Must be increasing.
            weights: Array of weights to apply to the score (e.g., latitude
                weighting). If None, no weights are applied.
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
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
    """Compute Mean Error (bias) between forecast and target.

    Extends BaseMetric to calculate mean error (bias) using the preserve_dims
    dimensions. Positive values indicate forecast exceeds target.
    """

    def __init__(self, name: str = "MeanError", *args, **kwargs):
        """Initialize the Mean Error metric.

        Args:
            name: The name of the metric. Defaults to "MeanError".
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
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

        Returns:
            The computed Mean Error result.
        """
        return scores.continuous.mean_error(
            forecast, target, preserve_dims=self.preserve_dims
        )


class RootMeanSquaredError(BaseMetric):
    """Compute Root Mean Squared Error between forecast and target.

    Extends BaseMetric to calculate RMSE using the preserve_dims dimensions.
    RMSE is the square root of the mean squared error.
    """

    def __init__(self, name: str = "RootMeanSquaredError", *args, **kwargs):
        """Initialize the Root Mean Squared Error metric.

        Args:
            name: The name of the metric. Defaults to "RootMeanSquaredError".
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
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

        Returns:
            The computed Root Mean Square Error result.
        """
        return scores.continuous.rmse(
            forecast, target, preserve_dims=self.preserve_dims
        )


class EarlySignal(BaseMetric):
    """Detect first occurrence of signal exceeding threshold criteria.

    Extends BaseMetric to find the earliest time when a signal is detected
    based on threshold criteria, returning init_time, lead_time, and
    valid_time information. Flexible for different signal detection criteria.

    For any case, the default behavior is to confirm any valid signal in space, then
    time. This behavior can be changed by setting spatial_aggregation and
    temporal_aggregation to "any", "all", or "half". Further, the order of aggregation
    can be changed by setting order to ["temporal", "spatial"] if desired.
    """

    def __init__(
        self,
        name: str = "EarlySignal",
        comparison_operator: Callable
        | Literal[">", ">=", "<", "<=", "==", "!="] = ">=",
        forecast_threshold: float = 0.5,
        overlap_target_threshold: float | None = None,
        spatial_aggregation: Literal["any", "all", "half"] = "any",
        temporal_aggregation: Literal["any", "all", "half"] = "any",
        aggregation_order: tuple[
            Literal["spatial", "temporal"], Literal["temporal", "spatial"]
        ] = ("spatial", "temporal"),
        **kwargs,
    ):
        """Initialize the Early Signal detection metric.

        Args:
            name: The name of the metric. Defaults to "EarlySignal".
            comparison_operator: The comparison operator for signal detection.
            forecast_threshold: The threshold value for signal detection.
            overlap_target_threshold: The target threshold for signal detection if
                target is meant to overlap. Defaults to None for no overlap applied.
            spatial_aggregation: Spatial aggregation method. Options: "any"
                (any gridpoint meets criteria), "all" (all gridpoints meet
                criteria), or "half" (at least half meet criteria).
            temporal_aggregation: Temporal aggregation method. Options: "any"
                (any valid time meets criteria), "all" (all valid times meet
                criteria), or "half" (at least half valid times meet criteria).
            aggregation_order: The order of spatial and temporal aggregation. Defaults
                to ("spatial", "temporal").
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name=name, **kwargs)
        self.comparison_operator = utils.maybe_get_operator(comparison_operator)
        self.forecast_threshold = forecast_threshold
        self.overlap_target_threshold = overlap_target_threshold
        self.spatial_aggregation = spatial_aggregation
        self.temporal_aggregation = temporal_aggregation
        self.aggregation_order = aggregation_order

    def _apply_aggregation(
        self,
        detection_mask: xr.DataArray,
        aggregation: Literal["any", "all", "half"],
        dims_to_reduce: list[str],
    ) -> xr.DataArray:
        """Apply aggregation to the detection mask.

        Args:
            detection_mask: The detection mask DataArray.
            aggregation: The aggregation method. Options: "any"
                (any gridpoint meets criteria), "all" (all gridpoints meet
                criteria), or "half" (at least half meet criteria).
            dims_to_reduce: The dimensions to reduce.
        Returns:
            The aggregated detection mask DataArray.
        """
        if aggregation == "any":
            # fillna(False): NaN means no data, not a True detection
            result = detection_mask.fillna(False).any(dims_to_reduce)
        elif aggregation == "all":
            # fillna(True): NaN means no data, don't penalize "all" condition
            result = detection_mask.fillna(True).all(dims_to_reduce)
        elif aggregation == "half":
            result = detection_mask.mean(dim=dims_to_reduce) >= 0.5
        else:
            raise ValueError(f"Aggregation '{aggregation}' not supported")
        return result

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute early signal detection.

        Args:
            forecast: The forecast dataarray with init_time, lead_time, valid_time.
            target: The target dataarray (used for reference/validation).

        Returns:
            Boolean DataArray with dims [init_time, lead_time] indicating
            whether criteria are met for each init_time and lead_time pair.
        """

        # Create detection masks with nans preserved
        forecast_detection_mask = self.comparison_operator(
            forecast, self.forecast_threshold
        ).where(~forecast.isnull())
        if self.overlap_target_threshold:
            target_detection_mask = self.comparison_operator(
                target, self.overlap_target_threshold
            ).where(~target.isnull())

            # Only assess forecast where target detects the event (value == 1);
            # locations where target is 0 or NaN are set to NaN (not dropped).
            # drop=True uses boolean dask indexing which is unsupported; NaN
            # values are handled correctly by _apply_aggregation downstream.
            forecast_detection_mask = forecast_detection_mask.where(
                target_detection_mask == 1
            )

        # Create lists of dimensions to reduce
        spatial_dims = [
            dim
            for dim in forecast_detection_mask.dims
            if dim not in ["init_time", "lead_time", "valid_time"]
            and dim not in self.preserve_dims
        ]
        temporal_dims = [
            dim
            for dim in forecast_detection_mask.dims
            if dim in ["init_time", "lead_time", "valid_time"]
            and dim not in self.preserve_dims
        ]

        # Apply aggregation in the order specified
        if self.aggregation_order[0] == "spatial":
            forecast_detection_mask = self._apply_aggregation(
                forecast_detection_mask, self.spatial_aggregation, spatial_dims
            )
            forecast_detection_mask = self._apply_aggregation(
                forecast_detection_mask,
                self.temporal_aggregation,
                temporal_dims,
            )
        elif self.aggregation_order[0] == "temporal":
            forecast_detection_mask = self._apply_aggregation(
                forecast_detection_mask,
                self.temporal_aggregation,
                temporal_dims,
            )
            forecast_detection_mask = self._apply_aggregation(
                forecast_detection_mask, self.spatial_aggregation, spatial_dims
            )

        forecast_detection_mask.name = self.name
        return forecast_detection_mask


class MaximumMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE between forecast and target maximum values.

    For each initialization, the forecast maximum is taken over that
    run's lead times inside tolerance_range_hours of the target's
    maximum. Useful for evaluating peak value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] | None = None,
        name: str = "MaximumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Maximum Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                maximum to search for forecast maximum. Defaults to 24.
            reduce_spatial_dims: Spatial dimensions to reduce. Defaults to
                ["latitude", "longitude"].
            name: The name of the metric. Defaults to
                "MaximumMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        if reduce_spatial_dims is None:
            reduce_spatial_dims = ["latitude", "longitude"]
        self.tolerance_range_hours = tolerance_range_hours
        self.reduce_spatial_dims = reduce_spatial_dims
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Compute MaximumMeanAbsoluteError.

        The forecast maximum is taken over each initialization's own lead times
        inside the tolerance window, so every value compared to the target is a
        genuine maximum over the forecast's diurnal cycle. Results are indexed
        by the lead time of the target's maximum relative to each
        initialization.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            MeanAbsoluteError of the maximum values.
        """
        reduce_spatial_dims = self.reduce_spatial_dims
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        if target_spatial_mean.valid_time.size == 0 or bool(
            target_spatial_mean.isnull().all()
        ):
            return utils._create_nan_dataarray(self.preserve_dims)
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        ).compute()
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=reduce_spatial_dims, skipna=True
        )
        filtered_max_forecast = utils.reduce_forecast_over_window_per_init(
            forecast_spatial_mean,
            center_time=maximum_timestep,
            tolerance_range_hours=self.tolerance_range_hours,
            method="max",
        )
        return super()._compute_metric(
            forecast=filtered_max_forecast,
            target=maximum_value,
            preserve_dims=self.preserve_dims,
        )


class MinimumMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE between forecast and target minimum values.

    For each initialization, the forecast minimum is taken over that
    run's lead times inside tolerance_range_hours of the target's
    minimum. Useful for evaluating minimum value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] | None = None,
        name: str = "MinimumMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Minimum Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                minimum to search for forecast minimum. Defaults to 24.
            reduce_spatial_dims: Spatial dimensions to reduce. Defaults to
                ["latitude", "longitude"].
            name: The name of the metric. Defaults to
                "MinimumMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        if reduce_spatial_dims is None:
            reduce_spatial_dims = ["latitude", "longitude"]
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

        As with MaximumMeanAbsoluteError, the forecast minimum is taken over
        each initialization's own lead times inside the tolerance window rather
        than across initializations at fixed lead time.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

        Returns:
            MeanAbsoluteError of the minimum values.
        """
        target_spatial_mean = utils.reduce_dataarray(
            target, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        if target_spatial_mean.valid_time.size == 0 or bool(
            target_spatial_mean.isnull().all()
        ):
            return utils._create_nan_dataarray(self.preserve_dims)
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = utils.reduce_dataarray(
            forecast, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
        )
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
        filtered_min_forecast = utils.reduce_forecast_over_window_per_init(
            forecast_spatial_mean,
            center_time=minimum_timestep,
            tolerance_range_hours=self.tolerance_range_hours,
            method="min",
        )
        return super()._compute_metric(
            forecast=filtered_min_forecast,
            target=minimum_value,
            preserve_dims=self.preserve_dims,
        )


class MaximumLowestMeanAbsoluteError(MeanAbsoluteError):
    """Compute MAE of the warmest daily minimum for heatwaves.

    The target contributes the warmest complete daily minimum. Each
    forecast initialization contributes its lowest value inside the
    tolerance window, scored only if that run covers a full day.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        name: str = "MaximumLowestMeanAbsoluteError",
        *args,
        **kwargs,
    ):
        """Initialize the Maximum Lowest Mean Absolute Error metric.

        Args:
            tolerance_range_hours: Time window (hours) around target's
                max-min value to search for forecast max-min. Defaults to 24.
            name: The name of the metric. Defaults to
                "MaximumLowestMeanAbsoluteError".
            *args: Additional positional arguments passed to
                MeanAbsoluteError.
            **kwargs: Additional keyword arguments passed to
                MeanAbsoluteError.
        """
        self.tolerance_range_hours = tolerance_range_hours
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute MaximumLowestMeanAbsoluteError.

        The target contributes the warmest of its daily minima. Each forecast
        initialization contributes the lowest value it predicts inside the
        tolerance window centered on that time, which spans one diurnal
        minimum, so both sides answer the same question. An initialization is
        only scored if it covers a full day inside the window.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

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
        # Carry the time the target's warmest minimum occurred into the output,
        # matching the other peak metrics.
        max_min_target_value = max_min_target_value.assign_coords(
            valid_time=np.asarray(max_min_target_datetime.values).reshape(-1)[0]
        )
        subset_forecast = utils.reduce_forecast_over_window_per_init(
            forecast,
            center_time=max_min_target_datetime,
            tolerance_range_hours=self.tolerance_range_hours,
            method="min",
            required_timesteps=utils.expected_timesteps_per_day(forecast),
        )
        return super()._compute_metric(
            forecast=subset_forecast,
            target=max_min_target_value,
            preserve_dims=self.preserve_dims,
        )


def _calculate_event_duration(
    mask: xr.DataArray,
    time_resolution_hours: float,
    preserve_dims: str = "init_time",
) -> xr.DataArray:
    """Count total consecutive-run duration in hours along valid_time.

    A run of N adjacent True timesteps each separated by exactly
    time_resolution_hours counts as (N - 1) * time_resolution_hours
    hours. An isolated True timestep contributes 0 hours.
    Discontinuous True values (gap > time_resolution_hours between
    any two True points) are not counted.

    NaN values in mask are treated as False.

    When preserve_dims is a regular array dimension (the simple
    init_time case), consecutive hours are summed over valid_time,
    returning total duration per preserve_dims slice.  When
    preserve_dims is only a coordinate (the lead_time case where
    init_time is a 2-D coord), a groupby aggregation is used
    instead so that total hours are accumulated per unique
    preserve_dims value.

    For the init_time case the predecessor lookup shifts both
    valid_time and lead_time by the same temporal distance
    (expected_gap) so that the predecessor cell belongs to the
    same forecast run.  The number of lead_time index steps is
    expected_gap / lead_time_step (e.g. 4 for a daily target
    on a 6 h lead_time grid).

    Args:
        mask: Boolean (or NaN-filled bool) DataArray with a
            valid_time dimension.
        time_resolution_hours: Expected temporal spacing in hours.
        preserve_dims: Dimension or coordinate to aggregate over.
            Defaults to "init_time".

    Returns:
        DataArray with valid_time reduced out, values in hours.
    """
    if mask.valid_time.size == 0:
        return utils._create_nan_dataarray(preserve_dims)
    expected_gap = np.timedelta64(int(time_resolution_hours), "h")
    mask = mask.fillna(False).astype(bool)
    vt = mask.valid_time.values
    gaps = np.zeros(vt.size, dtype=bool)
    gaps[1:] = (vt[1:] - vt[:-1]) == expected_gap
    is_expected_gap = xr.DataArray(gaps, dims=["valid_time"], coords={"valid_time": vt})
    # When init_time is only a coordinate (not a dim) and lead_time is a dim,
    # consecutive timesteps within one forecast run follow the diagonal of the
    # (valid_time, lead_time) grid.  Shifting along valid_time alone always
    # lands on NaN-filled cells (sparse outer-join grid), so shift both axes.
    # The number of lead_time index steps must equal expected_gap in time so
    # that the predecessor cell has the same init_time (vt - lead = const).
    # For daily targets aligned to a 6 h lead_time grid this is 24h/6h = 4.
    if "lead_time" in mask.dims and preserve_dims not in mask.dims:
        lt_step = np.unique(np.diff(utils._lead_time_as_timedelta(mask.lead_time)))
        n_lead_steps = (
            max(1, int(expected_gap / lt_step[0]))
            if len(lt_step) == 1 and lt_step[0] > np.timedelta64(0)
            else 1
        )
        mask_prev = mask.shift(
            {"valid_time": 1, "lead_time": n_lead_steps}, fill_value=False
        )
    else:
        mask_prev = mask.shift(valid_time=1, fill_value=False)
    consecutive_pairs = mask & mask_prev & is_expected_gap
    hours = consecutive_pairs * time_resolution_hours
    if preserve_dims in hours.dims:
        return hours.sum("valid_time")
    return hours.groupby(preserve_dims).sum()


class DurationMeanError(MeanError):
    """Compute mean error of event duration between forecast and target.

    Extends MeanError to compute the mean error between forecast and target
    event durations based on threshold criteria and spatial aggregation.
    """

    def __init__(
        self,
        threshold_criteria: xr.DataArray | float,
        reduce_spatial_dims: list[str] | None = None,
        op_func: Callable | Literal[">", ">=", "<", "<=", "==", "!="] = ">=",
        name: str = "DurationMeanError",
        preserve_dims: str = "init_time",
        product_time_resolution_hours: bool = False,
    ):
        """Initialize the Duration Mean Error metric.

        Args:
            threshold_criteria: Criteria for event detection. Either a
                DataArray of climatology with dimensions (dayofyear, hour,
                latitude, longitude) or a float fixed threshold.
            reduce_spatial_dims: Spatial dimensions to reduce prior to
                applying threshold criteria. Defaults to ["latitude",
                "longitude"].
            op_func: Comparison operator or string (e.g., operator.ge for
                >=).
            name: Name of the metric. Defaults to "DurationMeanError".
            preserve_dims: Dimensions to preserve during aggregation.
                Defaults to "init_time".
            product_time_resolution_hours: Whether to multiply duration by
                time resolution of forecast (in hours). Defaults to False.
        """
        if reduce_spatial_dims is None:
            reduce_spatial_dims = ["latitude", "longitude"]
        super().__init__(name=name, preserve_dims=preserve_dims)
        self.reduce_spatial_dims = reduce_spatial_dims
        self.threshold_criteria = threshold_criteria
        self.op_func = utils.maybe_get_operator(op_func)
        self.product_time_resolution_hours = product_time_resolution_hours

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> Any:
        """Compute spatially averaged duration mean error.

        Args:
            forecast: the forecast DataArray.
            target: the target DataArray.
            **kwargs: Accepts case_metadata (IndividualCase) to apply
                a ±48 h buffer around the case start/end before any
                reduction. If absent, the buffer step is skipped.

        Returns:
            The mean error between forecast and target event durations
            in hours.
        """
        # Apply ±48 h buffer around case start/end before any reduction.
        case_metadata = kwargs.get("case_metadata", None)
        if case_metadata is not None:
            buffer = np.timedelta64(48, "h")
            case_start = np.datetime64(case_metadata.start_date) - buffer
            case_end = np.datetime64(case_metadata.end_date) + buffer
            forecast = forecast.sel(valid_time=slice(case_start, case_end))
            target = target.sel(valid_time=slice(case_start, case_end))
        else:
            logger.debug(
                "case_metadata not in kwargs; skipping ±48 h valid_time buffer"
            )

        # Handle criteria - either climatology (xr.DataArray) or float threshold
        # Use local variable to avoid mutating self.threshold_criteria
        threshold_criteria = self.threshold_criteria

        # Need to get climatology into the correct format and interpolation for
        # comparison
        if isinstance(threshold_criteria, xr.DataArray):
            # Climatology case, convert from dayofyear/hour to valid_time.
            # Note that unintended behavior may occur if the case spans multiple years.
            days = np.intersect1d(
                forecast.valid_time.dt.dayofyear, threshold_criteria.dayofyear
            )
            if days.size:
                threshold_criteria = threshold_criteria.sel(dayofyear=days)
            threshold_criteria = utils.convert_day_yearofday_to_time(
                threshold_criteria, forecast.valid_time.dt.year.values[0]
            )

            # Interpolate climatology to target coordinates
            threshold_criteria = utils.interp_climatology_to_target(
                target, threshold_criteria
            )
        # Reduce spatial dimensions if specified (default is to reduce)
        if len(self.reduce_spatial_dims) > 0:
            target = utils.reduce_dataarray(
                target, method="mean", reduce_dims=self.reduce_spatial_dims, skipna=True
            )
            forecast = utils.reduce_dataarray(
                forecast,
                method="mean",
                reduce_dims=self.reduce_spatial_dims,
                skipna=True,
            )

            if isinstance(threshold_criteria, xr.DataArray):
                threshold_criteria = utils.reduce_dataarray(
                    threshold_criteria,
                    method="mean",
                    reduce_dims=self.reduce_spatial_dims,
                    skipna=True,
                )
        forecast_mask = self.op_func(forecast, threshold_criteria)
        target_mask = self.op_func(target, threshold_criteria)

        # Track NaN locations in forecast data
        forecast_valid_mask = ~forecast.isnull()

        # Apply valid data mask (exclude NaN positions in forecast)
        forecast_mask_final = forecast_mask.where(forecast_valid_mask)
        try:
            target_mask_final = target_mask.where(forecast_valid_mask)
        # If sparse, will need to expand_dims first as transpose is not supported
        except AttributeError:
            logger.info(
                "Target mask is sparse, expanding dimensions to handle unsupported "
                "transpose operation."
            )
            target_mask_final = target_mask.expand_dims(
                dim={"lead_time": target.lead_time.size}
            ).where(forecast_valid_mask)

        time_resolution_hours = utils.determine_temporal_resolution(forecast)
        if time_resolution_hours is None:
            logger.warning(
                "Could not determine temporal resolution; duration defaults to 0"
            )
            time_resolution_hours = 0
        # product_time_resolution_hours is deprecated; duration is now
        # always returned in hours by _calculate_event_duration
        forecast_duration = _calculate_event_duration(
            forecast_mask_final, time_resolution_hours, self.preserve_dims
        )
        target_duration = _calculate_event_duration(
            target_mask_final, time_resolution_hours, self.preserve_dims
        )

        return super()._compute_metric(
            forecast=forecast_duration,
            target=target_duration,
            preserve_dims=self.preserve_dims,
        )


class LandfallMetric(CompositeMetric):
    """Base class for tropical cyclone landfall metrics.

    Extends CompositeMetric to compute landfalls using calc.find_landfalls,
    which utilizes land geometry and line segments based on track data to
    determine intersections.

    Can be used as a base class for custom landfall metrics, as a mixin with
    other metrics, or as a composite metric for multiple landfall metrics.

    Public methods:
        maybe_prepare_composite_kwargs: Prepare kwargs for landfall composites
    """

    def __init__(
        self,
        name: str = "landfall_metrics",
        preserve_dims: str = "init_time",
        approach: Literal["first", "next"] = "first",
        exclude_post_landfall: bool = False,
        forecast_variable: str | derived.DerivedVariable | None = None,
        target_variable: str | derived.DerivedVariable | None = None,
        metrics: list[type["LandfallMetric"]] | None = None,
        min_target_separation_hours: float = 0.0,
        max_time_mismatch_hours: float | None = None,
        landfall_time_filter: tuple[str, float] | None = ("window", 24.0),
        *args,
        **kwargs,
    ):
        """Configure landfall detection and filtering.

        ``approach="first"`` evaluates the first observed
        landfall only; ``approach="next"`` matches each
        init_time to the closest future observed landfall.

        Args:
            name: Metric name.
            preserve_dims: Dims to keep in the result.
            approach: ``"first"`` or ``"next"``.
            exclude_post_landfall: Drop post-landfall data.
            forecast_variable: Forecast variable name.
            target_variable: Target variable name.
            metrics: Sub-metric classes for composite mode.
            min_target_separation_hours: Min hours between
                consecutive target landfalls. Filters rapid
                island crossings. Only for ``"next"``.
            max_time_mismatch_hours: Max hours between
                matched forecast/target landfall times.
                Only for ``"next"``.
            landfall_time_filter: How to filter paired
                forecast/target landfalls by timing.
                ``("window", hours)`` keeps pairs within
                a fixed ± window.
                ``("proportional", fraction)`` keeps pairs
                where timing error < fraction * lead time.
                ``None`` disables filtering.
        """
        super().__init__(
            *args,
            name=name,
            preserve_dims=preserve_dims,
            forecast_variable=forecast_variable,
            target_variable=target_variable,
            **kwargs,
        )
        self.approach = approach
        self.exclude_post_landfall = exclude_post_landfall
        self.min_target_separation_hours = min_target_separation_hours
        self.max_time_mismatch_hours = max_time_mismatch_hours
        self.landfall_time_filter = landfall_time_filter
        self.metrics = metrics or []
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

    def _nan_landfall_pair(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Return a (NaN, NaN) forecast/target landfall pair."""
        a = utils._create_nan_dataarray(self.preserve_dims)
        return (a, a.copy())

    def maybe_compute_landfalls(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Detect and pair forecast/target landfalls.

        Uses ``calc.find_landfalls`` (NaturalEarth 10m land
        geometry) then applies approach-specific matching and
        the configured time filters.

        Pre-computed landfalls can be passed via ``kwargs``
        keys ``forecast_landfall`` and ``target_landfall``
        to skip detection.

        Args:
            forecast: Forecast DataArray.
            target: Target DataArray.
            **kwargs: May include pre-computed landfalls.

        Returns:
            ``(forecast_landfall, target_landfall)`` tuple,
            or NaN arrays if no valid landfalls remain.
        """
        forecast_landfall, target_landfall = (
            kwargs.get("forecast_landfall", None),
            kwargs.get("target_landfall", None),
        )
        if forecast_landfall is not None and target_landfall is not None:
            return forecast_landfall, target_landfall

        # For "first" approach: get only first landfall
        # For "next" approach: get all target landfalls, then filter
        return_next_landfall = self.approach == "next"

        forecast_landfalls = calc.find_landfalls(forecast)

        if len(forecast_landfalls) == 0:
            return self._nan_landfall_pair()

        target_landfalls = calc.find_landfalls(target)

        if len(target_landfalls) == 0:
            return self._nan_landfall_pair()

        if return_next_landfall:
            max_lead = None
            track_starts = None
            if "lead_time" in forecast.dims:
                max_lead = forecast.lead_time.values.max()
                if not isinstance(max_lead, np.timedelta64):
                    max_lead = np.timedelta64(int(max_lead), "h")
                track_starts = calc.get_forecast_track_start_times(forecast)
            target_landfalls = calc.find_next_landfall_for_init_time(
                forecast_landfalls,
                target_landfalls,
                max_lead_time=max_lead,
                min_target_separation_hours=self.min_target_separation_hours,
                max_time_mismatch_hours=self.max_time_mismatch_hours,
                track_start_times=track_starts,
            )
            if len(target_landfalls) == 0:
                return self._nan_landfall_pair()
            # Reduce to first forecast landfall per init_time so
            # downstream metrics compare one-to-one instead of
            # averaging across multiple forecast landfalls.
            forecast_landfalls = calc.select_first_forecast_landfall_per_init(
                forecast_landfalls
            )
            common = utils.find_common_init_times(forecast_landfalls, target_landfalls)
            if len(common) == 0:
                return self._nan_landfall_pair()
            forecast_landfalls = forecast_landfalls.sel(init_time=common)
            target_landfalls = target_landfalls.sel(init_time=common)
        else:
            # "first" approach: compare to the earliest IBTrACS
            # landfall that the forecast can actually cover.
            # If the first landfall has no forecast data (e.g.
            # tracker misses a Cuba crossing), fall back to the
            # next landfall with coverage.
            reduced_forecast = calc.select_first_forecast_landfall_per_init(
                forecast_landfalls
            )
            track_starts = None
            if "lead_time" in forecast.dims:
                track_starts = calc.get_forecast_track_start_times(forecast)

            n_targets = target_landfalls.sizes.get("landfall", 1)
            selected_idx = None
            for lf_idx in range(n_targets):
                candidate_vt = (
                    target_landfalls.coords["valid_time"].isel(landfall=lf_idx).values
                )
                candidate_fc = reduced_forecast.where(
                    reduced_forecast.init_time < candidate_vt,
                    drop=True,
                )
                if len(candidate_fc) == 0:
                    continue
                if track_starts is not None:
                    candidate_fc = calc.filter_inits_by_track_start(
                        candidate_fc,
                        candidate_vt,
                        track_starts,
                    )
                if len(candidate_fc) > 0:
                    selected_idx = lf_idx
                    forecast_landfalls = candidate_fc
                    break

            if selected_idx is None:
                return self._nan_landfall_pair()

            # slice keeps the landfall dim (length 1) so
            # broadcast_first_target_to_init_times can
            # .isel(landfall=0) on it.
            selected_target = target_landfalls.isel(
                landfall=slice(selected_idx, selected_idx + 1)
            )
            target_landfalls = calc.broadcast_first_target_to_init_times(
                forecast_landfalls.init_time,
                selected_target,
            )

        if self.landfall_time_filter is not None:
            method, value = self.landfall_time_filter
            if method == "window":
                forecast_landfalls, target_landfalls = (
                    calc.filter_by_landfall_time_window(
                        forecast_landfalls,
                        target_landfalls,
                        window_hours=value,
                    )
                )
            elif method == "proportional":
                forecast_landfalls, target_landfalls = (
                    calc.filter_by_landfall_time_tolerance(
                        forecast_landfalls,
                        target_landfalls,
                        tolerance_fraction=value,
                    )
                )
            else:
                raise ValueError(
                    f'Unknown filter method {method!r}. Use "window" or "proportional".'
                )
            if len(forecast_landfalls) == 0:
                return self._nan_landfall_pair()

        return forecast_landfalls, target_landfalls

    def maybe_prepare_composite_kwargs(
        self,
        forecast_data: xr.DataArray,
        target_data: xr.DataArray,
        **base_kwargs: Any,
    ) -> dict:
        """Prepare kwargs for composite metric evaluation.

        Computes the landfalls once and adds them to kwargs to avoid recomputing when
        used as a composite metric.

        Args:
            forecast_data: The forecast DataArray.
            target_data: The target DataArray.

        Returns:
            Dictionary of kwargs including transformed_manager.
        """
        kwargs = base_kwargs.copy()

        if self.is_composite() and len(self._metric_instances) > 1:
            kwargs["forecast_landfall"], kwargs["target_landfall"] = (
                self.maybe_compute_landfalls(forecast=forecast_data, target=target_data)
            )

        kwargs["preserve_dims"] = self.preserve_dims

        return kwargs

    @staticmethod
    def _attach_landfall_metadata(
        result: xr.DataArray,
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
    ) -> xr.DataArray:
        """Attach forecast/target landfall coords to a result.

        Adds latitude, longitude, and valid_time for both the
        forecast and target landfalls as coordinates on the
        result DataArray. These become extra columns when
        converted to a DataFrame in the evaluation pipeline.
        """
        if "init_time" not in result.dims:
            return result
        coord_map = {}
        for prefix, lf in (
            ("forecast_landfall", forecast_landfall),
            ("target_landfall", target_landfall),
        ):
            for coord in ("latitude", "longitude", "valid_time"):
                if coord in lf.coords:
                    vals = lf.coords[coord].values
                    if vals.ndim == 0:
                        vals = np.full(
                            len(result.init_time),
                            vals,
                        )
                    coord_map[f"{prefix}_{coord}"] = (
                        "init_time",
                        vals,
                    )
        return result.assign_coords(coord_map)

    @staticmethod
    def _reduce_landfall_dim(
        result: xr.DataArray,
    ) -> xr.DataArray:
        """Mean over landfall dim if present."""
        if "landfall" in result.dims:
            return result.mean(dim="landfall")
        return result

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
    """Compute spatial displacement between forecast and target patterns.

    Extends BaseMetric to compute great circle distance between centers of
    mass of forecast and target spatial patterns. Useful for atmospheric
    rivers and similar spatial features.
    """

    def __init__(
        self,
        name: str = "spatial_displacement",
        **kwargs: Any,
    ):
        """Initialize the Spatial Displacement metric.

        Args:
            name: The name of the metric. Defaults to
                "spatial_displacement".
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute spatial displacement.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

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


class LandfallDisplacement(LandfallMetric):
    """Compute distance between forecast and target landfall positions.

    Extends LandfallMetric to calculate the spatial distance between forecast
    and target landfall positions, defaulting to kilometers.
    """

    def __init__(
        self,
        name: str = "landfall_displacement",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Displacement metric.

        Args:
            name: The name of the metric. Defaults to
                "landfall_displacement".
            *args: Additional positional arguments passed to LandfallMetric.
            **kwargs: Additional keyword arguments passed to LandfallMetric.
        """
        super().__init__(name, *args, **kwargs)
        self.units = kwargs.get("units", "km")

    def calculate_displacement(
        self,
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
        units: Literal["km", "kilometers", "deg", "degrees"] = "km",
    ) -> xr.DataArray:
        """Calculate the distance between two landfall points in km or degrees.

        Handles targets indexed by init_time (next approach) or by landfall
        (first approach). When the target has a landfall dim instead of
        init_time, the single target coordinates are reused for every
        forecast init_time.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray
            target_landfall: Target landfall xarray DataArray
            units: The units to use for the distance. Defaults to "km"
        Returns:
            Distance in the specified units as xarray DataArray
        """
        forecast_landfall, target_landfall = xr.align(
            forecast_landfall, target_landfall, join="inner"
        )
        if len(forecast_landfall) == 0:
            return xr.DataArray(np.nan)
        distances = calc.haversine_distance(
            [forecast_landfall.latitude, forecast_landfall.longitude],
            [target_landfall.latitude, target_landfall.longitude],
            units=units,
        )
        if not isinstance(distances, xr.DataArray):
            raise TypeError("haversine_distance returned a scalar; expected DataArray")
        result = distances.where(forecast_landfall.notnull())
        return self._reduce_landfall_dim(result)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall displacement metric."""
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall
        ) or not utils.is_valid_landfall(target_landfall):
            return utils._create_nan_dataarray(self.preserve_dims)
        result = self.calculate_displacement(
            forecast_landfall,
            target_landfall,
            units=self.units,
        )
        return self._attach_landfall_metadata(
            result,
            forecast_landfall,
            target_landfall,
        )


class LandfallTimeMeanError(LandfallMetric):
    """Compute mean error between forecast and target landfall times.

    Extends LandfallMetric to calculate timing difference. Positive values
    indicate forecast landfall is later than target; negative values indicate
    forecast landfall is earlier than target.
    """

    def __init__(
        self,
        name: str = "landfall_time_me",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Time Mean Error metric.

        Args:
            name: The name of the metric. Defaults to "landfall_time_me".
            *args: Additional positional arguments passed to LandfallMetric.
            **kwargs: Additional keyword arguments passed to LandfallMetric.
        """
        super().__init__(name, *args, **kwargs)

    def calculate_time_difference(
        self,
        forecast_landfall: xr.DataArray,
        target_landfall: xr.DataArray,
    ) -> xr.DataArray:
        """Calculate the time difference between two landfall points in hours.

        Handles targets indexed by init_time (next approach) or by landfall
        (first approach). When the target has a landfall dim instead of
        init_time, the single target valid_time is reused for every
        forecast init_time.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray.
            target_landfall: Target landfall xarray DataArray.

        Returns:
            Time difference in hours (forecast_landfall - target_landfall)
            as xarray DataArray with init_time dimension.
        """

        forecast_landfall, target_landfall = xr.align(
            forecast_landfall, target_landfall, join="inner"
        )
        if len(forecast_landfall) == 0:
            return xr.DataArray(np.nan)
        time_diffs = (
            forecast_landfall.valid_time - target_landfall.valid_time
        ) / np.timedelta64(1, "h")
        result = time_diffs.where(forecast_landfall.notnull())
        return self._reduce_landfall_dim(result)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall time metric."""
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall
        ) or not utils.is_valid_landfall(target_landfall):
            return utils._create_nan_dataarray(self.preserve_dims)
        result = self.calculate_time_difference(
            forecast_landfall,
            target_landfall,
        )
        return self._attach_landfall_metadata(
            result,
            forecast_landfall,
            target_landfall,
        )


class LandfallIntensityMeanAbsoluteError(LandfallMetric, MeanAbsoluteError):
    """Compute MAE of forecast and target intensity at landfall.

    Extends both LandfallMetric and MeanAbsoluteError to calculate mean
    absolute error between forecast and target intensity at landfall time.

    The intensity variable is determined by forecast_variable and
    target_variable. For multiple intensity variables, create separate metric
    instances for each variable.
    """

    def __init__(
        self,
        name: str = "landfall_intensity_mae",
        *args,
        **kwargs,
    ):
        """Initialize the Landfall Intensity Mean Absolute Error metric.

        Args:
            name: The name of the metric. Defaults to
                "landfall_intensity_mae".
            *args: Additional positional arguments passed to parent classes.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """Compute the landfall intensity metric."""
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall
        ) or not utils.is_valid_landfall(target_landfall):
            return utils._create_nan_dataarray(self.preserve_dims)

        result = self._reduce_landfall_dim(
            np.abs(forecast_landfall - target_landfall).where(
                forecast_landfall.notnull()
            )
        )
        return self._attach_landfall_metadata(
            result,
            forecast_landfall,
            target_landfall,
        )

class SPEDS(BaseMetric):
    def __init__(
        self, 
        alpha=0.33, 
        beta=0.33, 
        gamma=0.33, 
        window_size=3,
        name="SPEDS",
        *args,
        **kwargs
    ):
        """ 
        Spatial Probabilistic Extreme Dynamics Score (SPEDS).
        """
        super().__init__(name=name, *args, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size

    def compute_spatial_gradients(self, data: np.ndarray) -> np.ndarray:
        """Computes smooth spatial gradients using SciPy safely, handling NaNs."""
        has_nans = np.isnan(data).any()
        
        if has_nans:
            # Fill NaNs with 0 to prevent NaN propagation through the Sobel filter
            clean_data = np.nan_to_num(data, nan=0.0)
        else:
            clean_data = data

        # ndimage.sobel applies the 3x3 kernels and handles padding automatically.
        gx = ndimage.sobel(clean_data, axis=-1) # Gradient along X (longitude)
        gy = ndimage.sobel(clean_data, axis=-2) # Gradient along Y (latitude)
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(gx**2 + gy**2 + 1e-8)
        
        if has_nans:
            # Restore NaNs where the original data was missing
            grad_mag = np.where(np.isnan(data), np.nan, grad_mag)
            
        return grad_mag

    def _compute_tail_risk(self, f_data: np.ndarray, t_data: np.ndarray) -> np.ndarray:
        """Calculates the MSE of the neighborhood-smoothed fields using SciPy safely."""
        filter_size = [1] * (t_data.ndim - 2) + [self.window_size, self.window_size]
        
        # Safely fill NaNs so uniform_filter doesn't propagate them
        has_nans = np.isnan(t_data).any() or np.isnan(f_data).any()
        if has_nans:
            t_data_clean = np.nan_to_num(t_data, nan=0.0)
            f_data_clean = np.nan_to_num(f_data, nan=0.0)
        else:
            t_data_clean = t_data
            f_data_clean = f_data
            
        smooth_targets = ndimage.uniform_filter(t_data_clean, size=filter_size, mode='constant', cval=0.0)
        smooth_forecasts = ndimage.uniform_filter(f_data_clean, size=filter_size, mode='constant', cval=0.0)

        mse = (smooth_forecasts - smooth_targets) ** 2
        
        if has_nans:
            # Restore NaNs where EITHER target or forecast had missing data
            mask = np.isnan(t_data) | np.isnan(f_data)
            mse = np.where(mask, np.nan, mse)
            
        return mse

    def _compute_gradient_fidelity(self, f_data: np.ndarray, t_data: np.ndarray) -> np.ndarray:
        """Calculates the MSE of the spatial gradients (Sobel)."""
        target_grad = self.compute_spatial_gradients(t_data)
        forecast_grad = self.compute_spatial_gradients(f_data)
        return (forecast_grad - target_grad) ** 2

    def _compute_sharpness_penalty_xr(self, forecast: xr.DataArray, active_ensemble_dim: str):
        """Calculates GMD-based CRPS spread from the ensemble forecast as an Xarray DataArray."""
        if active_ensemble_dim is not None and active_ensemble_dim in forecast.dims:
            num_ensemble = forecast[active_ensemble_dim].size
            
            rank = forecast.rank(dim=active_ensemble_dim)
            gini_mean_difference = (
                2 * (((2 * rank - num_ensemble - 1) * forecast).mean(dim=active_ensemble_dim, skipna=True))
                / (num_ensemble - 1)
            )
            return 0.5 * gini_mean_difference
        return None

    def _compute_metric(self, forecast: xr.DataArray, target: xr.DataArray, selected_ensemble=None, **kwargs) -> xr.DataArray:
        # ---------------------------------------------------------
        # 1. Ensemble Selection & Version Splitting
        # ---------------------------------------------------------
        if forecast.name != target.name:
            raise ValueError(f"VARIABLE MISMATCH ERROR! Forecast is '{forecast.name}' but Target is '{target.name}'.")

        possible_ensemble_dims = ["ensemble", "realization", "member", "ens"]
        active_ensemble_dim = next((x for x in possible_ensemble_dims if x in forecast.dims), None)
        
        if active_ensemble_dim:
            num_ensemble = forecast[active_ensemble_dim].size
            if selected_ensemble is not None:
                if num_ensemble < selected_ensemble:
                    raise ValueError(f"Not enough ensemble members. Found {num_ensemble}, needed {selected_ensemble}.")
                forecast = forecast.sel({active_ensemble_dim: range(1, selected_ensemble + 1)})
            
            ensemble_spread_xr = self._compute_sharpness_penalty_xr(forecast, active_ensemble_dim)
            forecast_mean = forecast.mean(dim=active_ensemble_dim, skipna=True)
        else:
            ensemble_spread_xr = None
            forecast_mean = forecast

        # ---------------------------------------------------------
        # 2. Xarray Native Broadcasting & Alignment
        # ---------------------------------------------------------
        spatial_dims = [d for d in ["latitude", "longitude"] if d in forecast_mean.dims]
        other_dims = [d for d in forecast_mean.dims if d not in spatial_dims]

        baseline_scalar = forecast_mean.mean(dim=spatial_dims, skipna=True)
        baseline_mean = baseline_scalar.broadcast_like(forecast_mean)
        
        spatial_max = forecast_mean.max(dim=spatial_dims, skipna=True)
        spatial_min = forecast_mean.min(dim=spatial_dims, skipna=True)
        baseline_sharpness_scalar = 0.25 * (spatial_max - spatial_min)
        baseline_sharpness_xr = baseline_sharpness_scalar.broadcast_like(forecast_mean)
        
        forecast_mean, target = xr.broadcast(forecast_mean, target)
        forecast_mean = forecast_mean.transpose(*other_dims, *spatial_dims)
        target = target.transpose(*other_dims, *spatial_dims)

        if ensemble_spread_xr is not None:
            ensemble_spread_xr, _ = xr.broadcast(ensemble_spread_xr, target)
            ensemble_spread_xr = ensemble_spread_xr.transpose(*other_dims, *spatial_dims)
            
        baseline_sharpness_xr, _ = xr.broadcast(baseline_sharpness_xr, target)
        baseline_sharpness_xr = baseline_sharpness_xr.transpose(*other_dims, *spatial_dims)

        # ---------------------------------------------------------
        # 3. Compute Components in NumPy (Model & Baseline)
        # ---------------------------------------------------------
        # Extract underlying NumPy arrays directly
        f_data = forecast_mean.values
        t_data = target.values

        if ensemble_spread_xr is not None:
            sharpness_map = ensemble_spread_xr.values
        else:
            sharpness_map = np.zeros_like(t_data)

        # A. Model Loss Calculation
        tail_error_map = self._compute_tail_risk(f_data, t_data)
        grad_error_map = self._compute_gradient_fidelity(f_data, t_data)
        
        # B. Baseline Loss Calculation
        baseline_expanded = baseline_scalar.broadcast_like(forecast_mean)
        b_data = baseline_expanded.values

        baseline_tail_map = self._compute_tail_risk(b_data, t_data)
        baseline_sharpness_map = baseline_sharpness_xr.values
        baseline_grad_map = self._compute_gradient_fidelity(b_data, t_data)
        
        baseline_loss_map = (self.alpha * baseline_tail_map) + (self.beta * baseline_sharpness_map) + (self.gamma * baseline_grad_map)

        # ---------------------------------------------------------
        # 4. Map Back to Xarray, Average Spatially, and Calculate Final Score
        # ---------------------------------------------------------
        # Pass the numpy arrays directly back into Xarray
        tail_xr = target.copy(data=tail_error_map)
        grad_xr = target.copy(data=grad_error_map)
        sharp_xr = target.copy(data=sharpness_map)
        
        b_tail_xr = target.copy(data=baseline_tail_map)
        b_grad_xr = target.copy(data=baseline_grad_map)
        b_sharp_xr = target.copy(data=baseline_sharpness_map)

        tail_mean_xr = tail_xr.mean(dim=spatial_dims, skipna=True)
        grad_mean_xr = grad_xr.mean(dim=spatial_dims, skipna=True)
        sharp_mean_xr = sharp_xr.mean(dim=spatial_dims, skipna=True)
        
        b_tail_mean_xr = b_tail_xr.mean(dim=spatial_dims, skipna=True)
        b_grad_mean_xr = b_grad_xr.mean(dim=spatial_dims, skipna=True)
        b_sharp_mean_xr = b_sharp_xr.mean(dim=spatial_dims, skipna=True)

        eps = 1e-8 
        tail_score_xr = tail_mean_xr / (b_tail_mean_xr + eps)
        grad_score_xr = grad_mean_xr / (b_grad_mean_xr + eps)
        sharp_score_xr = sharp_mean_xr / (b_sharp_mean_xr + eps)
        
        combined_score_xr = (self.alpha * tail_score_xr) + \
                            (self.gamma * grad_score_xr) + \
                            (self.beta  * sharp_score_xr)

        if hasattr(self, "preserve_dims") and self.preserve_dims is not None:
            dims_to_reduce = [dim for dim in combined_score_xr.dims if dim not in self.preserve_dims]
            if dims_to_reduce:
                tail_score_xr = tail_score_xr.mean(dim=dims_to_reduce, skipna=True)
                grad_score_xr = grad_score_xr.mean(dim=dims_to_reduce, skipna=True)
                sharp_score_xr = sharp_score_xr.mean(dim=dims_to_reduce, skipna=True)
                combined_score_xr = combined_score_xr.mean(dim=dims_to_reduce, skipna=True)
                
        components = ["tail_risk", "gradient_fidelity", "sharpness", "total_speds"]
        final_score_xr = xr.concat(
            [tail_score_xr, grad_score_xr, sharp_score_xr, combined_score_xr], 
            dim=pd.Index(components, name="speds_component")
        )

        return final_score_xr

class NLL(BaseMetric):
    """Compute Negative Log-Likelihood (NLL) assuming a Gaussian ensemble distribution."""

    def __init__(
        self,
        name: str = "NLL",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        selected_ensemble: Optional[int] = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        print("selected_ensemble nll:", selected_ensemble)
        # 1. Identify the ensemble dimension in the forecast dataset
        possible_ensemble_dims = ["ensemble", "realization", "member", "ens"]
        active_ensemble_dim = next((dim for dim in possible_ensemble_dims if dim in forecast.dims), None)
        
        if not active_ensemble_dim:
            raise ValueError(f"No ensemble dimension found in forecast. Checked: {possible_ensemble_dims}")

        # 2. Handle optional ensemble subset selection
        if selected_ensemble is not None:
            available_members = forecast.sizes[active_ensemble_dim]
            if selected_ensemble > available_members:
                raise ValueError(f"Selected ensemble members ({selected_ensemble}) exceed available ({available_members})")
            forecast = forecast.isel({active_ensemble_dim: slice(0, selected_ensemble)})

        n_ensemble = forecast.sizes[active_ensemble_dim]
        if n_ensemble < 2:
            raise ValueError(f"NLL requires at least 2 ensemble members to compute variance. Found: {n_ensemble}")

        # 3. Compute the Ensemble Mean (mu) and Standard Deviation (sigma)
        mu = forecast.mean(dim=active_ensemble_dim, skipna=True)
        sigma = forecast.std(dim=active_ensemble_dim, skipna=True)

        # Add a small epsilon for numerical stability to prevent division by zero or log(0)
        eps = 1e-6
        sigma = xr.where(sigma < eps, eps, sigma)

        # 4. Apply the Gaussian Negative Log-Likelihood (NLL) formula
        # NLL = 0.5 * log(2 * pi * sigma^2) + (target - mu)^2 / (2 * sigma^2)
        variance = sigma ** 2
        pi = np.pi
        
        nll_val = 0.5 * np.log(2 * pi * variance) + ((target - mu) ** 2) / (2 * variance)

        # 5. Spatiotemporal Aggregation (Preserve lead_time / preserve_dims)
        dims_to_reduce = [
            dim for dim in nll_val.dims if dim not in self.preserve_dims
        ]
        
        if dims_to_reduce:
            nll_val = nll_val.mean(dim=dims_to_reduce, skipna=True)

        return nll_val

class CRPS(BaseMetric):
    """Compute Continuous Ranked Probability Score (CRPS) for ensemble forecasts."""

    def __init__(
        self,
        name: str = "CRPS",
        *args,
        **kwargs,
    ):
 
        super().__init__(name=name, *args, **kwargs)  

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        selected_ensemble: Optional[int]=None,
        **kwargs: Any,
    ) -> Any:
        print("selected_ensemble crps:", selected_ensemble)
        """Compute CRPS using Zamo & Naveau (2018) PWM method adapted for DataArray."""

        # Define common ensemble dimension names typically used in meteorological datasets
        possible_ensemble_dims = ["ensemble", "realization", "member", "ens"]
        
        # Automatically check if any of the possible ensemble dimensions exist in the forecast data
        active_ensemble_dim = next((dim for dim in possible_ensemble_dims if dim in forecast.dims), None)
        
        # Check if the dataset has enough ensemble members to calculate probabilities
        n_ensemble = forecast.sizes[active_ensemble_dim]
        if n_ensemble < 2:
            raise ValueError(f"CRPS requires at least 2 ensemble members. Found: {n_ensemble}")
        else:
            forecast = forecast.sel({active_ensemble_dim: range(1, n_ensemble + 1)})

        # 1. Calculate the Skill part: E|X - Y|
        # This is the Mean Absolute Error between each ensemble member and the ground truth.
        skill = abs(target - forecast).mean(dim=active_ensemble_dim, skipna=True)

        # 2. Calculate the Spread part: 0.5 * E|X - X'|
        # Instead of comparing every pair (which is slow, O(N^2)), we use an efficient
        # mathematical trick based on ranking (O(N log N)) from Zamo & Naveau (2018).
        rank = _rank_da(forecast, active_ensemble_dim)
        spread = (
            2
            * (((2 * rank - n_ensemble - 1) * forecast).mean(dim=active_ensemble_dim, skipna=True))
            / (n_ensemble - 1)
        )
        spread_term = 0.5 * spread

        # 3. Calculate Pointwise CRPS
        crps_val = skill - spread_term

        # 4. Spatiotemporal Aggregation
        # Identify all dimensions that are NOT in `self.preserve_dims` (which is usually 'lead_time').
        # This will automatically collect 'latitude', 'longitude', and 'valid_time'.
        dims_to_reduce = [
            dim for dim in crps_val.dims if dim not in self.preserve_dims
        ]
        
        # Take the mean over the spatial and temporal dimensions to return a single 
        # aggregated scalar value per case and lead time.
        if dims_to_reduce:
            crps_val = crps_val.mean(dim=dims_to_reduce, skipna=True)

        return crps_val

class SpatialFSS(BaseMetric):
    def __init__(self, threshold: Union[float, xr.DataArray], window_size: int= 5, name: str="SpatialFSS",  *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.threshold = threshold
        self.window_size = window_size


    def _compute_metric(
        self, forecast: xr.DataArray, 
        target: xr.DataArray, 
        selected_ensemble: Optional[int]=None, 
        **kwargs: Any
        )-> xr.DataArray:
        print("selected_ensemble fss:", selected_ensemble)
        possible_ensemble_dims = ["ensemble", "realization", "member", "ens"]

        active_ensemble_dim = next((x for x in possible_ensemble_dims if x in forecast.dims), None)

        if active_ensemble_dim:
            num_ensemble = forecast[active_ensemble_dim].size
            if num_ensemble<selected_ensemble:
                raise ValueError(f"No ensemble dimension found in forecast. Checked: {possible_ensemble_dims}")
            else:
                forecast = forecast.sel({active_ensemble_dim:range(1, selected_ensemble+1)})
                print(forecast[active_ensemble_dim].size)
                target_frac = (target>self.threshold).astype(float)

                forecast_frac = (forecast>self.threshold).astype(float)
                forecast_prob = forecast_frac.mean(dim=active_ensemble_dim, skipna=True)

                # Apply the rolling window to avoid the double panelty by pixel to pixel matching
                rolling_kwargs={
                    "latitude":self.window_size,
                    "longitude":self.window_size,
                    "center": True
                    }

                target_frac = target_frac.rolling(**rolling_kwargs).mean(skipna=True)
                forecast_frac = forecast_prob.rolling(**rolling_kwargs).mean(skipna=True)

                # Calculate the mse and reference mse, where reference mse is the maximume mse
                mse = (target_frac-forecast_frac)**2
                mse_ref = (target_frac**2) + (forecast_frac**2)

                
                mse_mean = mse.mean(dim=["latitude", "longitude"], skipna=True)
                mse_ref_mean = mse_ref.mean(dim=["latitude", "longitude"], skipna=True)

                fss = 1 - (mse_mean/(mse_ref_mean + 1e-8))

                # Reduce the dimensions that we do not want
                dims_to_reduce = [dim for dim in fss.dims if dim not in self.preserve_dims]

                if dims_to_reduce:
                    fss = fss.mean(dim=dims_to_reduce)

                return fss

def _get_n_ensemble(da: xr.DataArray, ensemble_dim: str) -> int:
    if ensemble_dim not in da.dims:
        raise ValueError(f"{ensemble_dim=} not found in {da.dims=}")
    return da.sizes[ensemble_dim]

def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(x)
    x = np.swapaxes(x, axis, -1)
    j = np.argsort(x, axis=-1)
    ordinal_ranks = np.broadcast_to(
        np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
    )
    ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
    np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
    return np.swapaxes(ordered_ranks, axis, -1)

def _rank_da(da: xr.DataArray, dim: str) -> xr.DataArray:
    ranked_vals = _rankdata(da.values, axis=da.dims.index(dim))
    return da.copy(data=ranked_vals)

class _HeatWaveBase(BaseMetric):
    """
    Shared logic for the two heat wave scores below.
 
    A grid point counts as "in a heat wave" when its anomaly (value minus
    climatology) exceeds `threshold_offset` for at least `min_consecutive_days`
    consecutive days (WMO / HWDI style). Every step inside a qualifying run is
    flagged, for both the forecast members and the actual (target) data.
 
    Assumes one init_time per case. If the forecast has both `lead_time` and
    `valid_time` dimensions, it is first collapsed onto a dense valid_time axis
    so "consecutive days" is measured along real time; scores are then reported
    per lead_time.
    """
 
    ENSEMBLE_DIM_CANDIDATES = ("ensemble", "realization", "member", "ens")
 
    def __init__(
        self,
        climatology: xr.DataArray,
        name: str = "HeatWaveMetric",
        threshold_offset: float = 5.0,     # degC / K above climatology (WMO: 5.0)
        min_consecutive_days: int = 5,     # WMO: 5 days
        time_dim: str = "valid_time",
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.climatology = climatology
        self.threshold_offset = threshold_offset
        self.min_consecutive_days = min_consecutive_days
        self.time_dim = time_dim
        self.debug = debug
 
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _find_ensemble_dim(self, forecast: xr.DataArray) -> str:
        for d in self.ENSEMBLE_DIM_CANDIDATES:
            if d in forecast.dims:
                return d
        raise ValueError(
            f"No ensemble dimension found in forecast. Checked: {self.ENSEMBLE_DIM_CANDIDATES}"
        )
 
    def _align_climatology(self, forecast: xr.DataArray) -> xr.DataArray:
        """Climatology on the forecast's exact time/lat/lon labels (no silent inner-join shrinkage)."""
        clim = self.climatology.sel({self.time_dim: forecast[self.time_dim]}, method="nearest")
 
        lat_q = forecast["latitude"].values
        lon_q = forecast["longitude"].values
        clim_lon = clim["longitude"].values
 
        # Reconcile longitude conventions (0..360 vs -180..180)
        if clim_lon.min() >= 0 and lon_q.min() < 0:
            lon_q = np.mod(lon_q, 360.0)
        elif clim_lon.min() < 0 and lon_q.max() > 180:
            lon_q = ((lon_q + 180.0) % 360.0) - 180.0
 
        lat_idx = clim.indexes["latitude"].get_indexer(lat_q, method="nearest")
        lon_idx = clim.indexes["longitude"].get_indexer(lon_q, method="nearest")
        clim = clim.isel(latitude=lat_idx, longitude=lon_idx)
 
        # Relabel with the forecast's own coordinates so arithmetic aligns 1:1
        return clim.assign_coords(
            latitude=forecast["latitude"].values,
            longitude=forecast["longitude"].values,
        )
 
    def _match_grid(self, target: xr.DataArray, forecast: xr.DataArray) -> xr.DataArray:
        """Put the target on the forecast's lat/lon labels if they differ."""
        same = np.array_equal(target["latitude"].values, forecast["latitude"].values) and \
               np.array_equal(target["longitude"].values, forecast["longitude"].values)
        if not same:
            target = target.reindex(
                latitude=forecast["latitude"].values,
                longitude=forecast["longitude"].values,
                method="nearest",
                tolerance=0.5,
            )
        return target
 
    def _steps_required(self, times: xr.DataArray) -> int:
        """Convert min_consecutive_days into number of timesteps (e.g. 5 days @ 12h = 10)."""
        deltas = np.diff(times.values).astype("timedelta64[s]").astype(float)
        step_hours = float(np.median(deltas)) / 3600.0
        return int(round(self.min_consecutive_days * 24.0 / step_hours))
 
    def _consecutive_exceedance_mask(self, anomaly: xr.DataArray, n_steps: int) -> xr.DataArray:
        """True for every step that belongs to a run of >= n_steps consecutive exceedances."""
        above = (anomaly > self.threshold_offset).astype(float).where(anomaly.notnull())
 
        # True at the LAST step of each qualifying run
        run_end = above.rolling({self.time_dim: n_steps}, min_periods=n_steps).sum() >= n_steps
 
        # Spread that flag backwards over the whole run (look ahead n_steps - 1 steps)
        in_run = (
            run_end.astype(float)
            .isel({self.time_dim: slice(None, None, -1)})
            .rolling({self.time_dim: n_steps}, min_periods=1).max()
            .isel({self.time_dim: slice(None, None, -1)})
        )
        return in_run > 0
 
    def _prepare(self, forecast, target, selected_ensemble):
        """Returns (forecast_mask, actual_mask, valid, pair_valid, ens_dim)."""
        ens_dim = self._find_ensemble_dim(forecast)
        if selected_ensemble is not None:
            if forecast[ens_dim].size < selected_ensemble:
                raise ValueError(
                    f"Requested {selected_ensemble} ensemble members, but forecast "
                    f"only has {forecast[ens_dim].size}."
                )
            forecast = forecast.isel({ens_dim: slice(0, selected_ensemble)})
 
        if self.time_dim not in forecast.dims:
            raise ValueError(f"'{self.time_dim}' not in forecast dims {forecast.dims}.")
 
        clim = self._align_climatology(forecast)
        target = self._match_grid(target, forecast)
 
        # Make the forecast a dense series along valid_time
        if "lead_time" in forecast.dims:
            other_dims = [d for d in forecast.dims if d not in ("lead_time", self.time_dim)]
            pair_valid = forecast.notnull().any(dim=other_dims)   # (lead_time, valid_time)
            forecast_dense = forecast.mean(dim="lead_time", skipna=True)  # single init_time
        else:
            pair_valid = None
            forecast_dense = forecast
 
        forecast_anomaly = forecast_dense - clim
        actual_anomaly = target - clim
 
        n_steps = self._steps_required(target[self.time_dim])
        n_times = forecast_anomaly.sizes[self.time_dim]
        if n_steps > n_times:
            raise ValueError(
                f"Need {n_steps} consecutive steps ({self.min_consecutive_days} days) "
                f"but the case only has {n_times} valid times; no heat wave can ever be detected."
            )
 
        forecast_mask = self._consecutive_exceedance_mask(forecast_anomaly, n_steps)
        actual_mask = self._consecutive_exceedance_mask(actual_anomaly, n_steps)
 
        # Points where both forecast and actual data exist (excludes missing / NaN areas)
        valid = actual_anomaly.notnull() & forecast_anomaly.notnull()
 
        if self.debug:
            print("forecast dims:", dict(forecast.sizes))
            print("target dims  :", dict(target.sizes))
            print("steps required:", n_steps, "| valid times available:", n_times)
            print("frac anomaly > thr -> forecast:",
                  float((forecast_anomaly > self.threshold_offset).mean()),
                  "| actual:", float((actual_anomaly > self.threshold_offset).mean()))
            print("heat-wave points -> forecast:", int(forecast_mask.sum()),
                  "| actual:", int(actual_mask.sum()))
 
        return forecast_mask, actual_mask, valid, pair_valid, ens_dim
 
    def _rate(self, flags: xr.DataArray, pair_valid, ens_dim: str) -> xr.DataArray:
        """Average 0/1/NaN flags over space/time, then over members."""
        if pair_valid is not None:
            flags = flags.where(pair_valid)   # keep each valid_time under its own lead_time
 
        per_member = flags.mean(dim=[self.time_dim, "latitude", "longitude"], skipna=True)
        score = per_member.mean(dim=ens_dim, skipna=True)
 
        preserve = getattr(self, "preserve_dims", [])
        if isinstance(preserve, str):
            preserve = [preserve]
        dims_to_reduce = [d for d in score.dims if d not in preserve]
        if dims_to_reduce:
            score = score.mean(dim=dims_to_reduce, skipna=True)
        return score
 
 
class EnsembleHeatWaveDetectionRate(_HeatWaveBase):
    """
    HIT RATE (higher is better, 0..1).
    Where the actual data IS in a heat wave, what fraction of points did the
    ensemble members also flag as a heat wave (averaged over members)?
    NaN means there was no actual heat wave to detect.
    """
 
    def __init__(self, climatology, name: str = "EnsembleHeatWaveDetectionRate", **kwargs):
        super().__init__(climatology=climatology, name=name, **kwargs)
 
    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        selected_ensemble: Optional[int] = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        forecast_mask, actual_mask, valid, pair_valid, ens_dim = self._prepare(
            forecast, target, selected_ensemble
        )
        print("selected_ensemble rate:", selected_ensemble)
        event = actual_mask & valid
        hits = forecast_mask.astype(float).where(event)
        return self._rate(hits, pair_valid, ens_dim)
 
 
class EnsembleHeatWaveFalseAlarmRate(_HeatWaveBase):
    """
    FALSE ALARM RATE (lower is better, 0..1).
    Where the actual data is NOT in a heat wave, what fraction of points did the
    ensemble members still flag as a heat wave (averaged over members)?
    NaN means every valid point was an actual heat wave, so there were no
    non-event points to check.
 
    Read together with EnsembleHeatWaveDetectionRate: a good model has a HIGH
    hit rate AND a LOW false alarm rate. A model that flags heat waves
    everywhere gets hit rate 1.0 but false alarm rate 1.0.
    """
 
    def __init__(self, climatology, name: str = "EnsembleHeatWaveFalseAlarmRate", **kwargs):
        super().__init__(climatology=climatology, name=name, **kwargs)
 
    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        selected_ensemble: Optional[int] = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        forecast_mask, actual_mask, valid, pair_valid, ens_dim = self._prepare(
            forecast, target, selected_ensemble
        )
        print("selected_ensemble false alarm:", selected_ensemble)
        non_event = (~actual_mask) & valid
        false_alarms = forecast_mask.astype(float).where(non_event)
        return self._rate(false_alarms, pair_valid, ens_dim)

class CoverOneSD(BaseMetric):
    """Compute the CoverOneSD metric (percentage of data within 1 standard deviation of the ensemble mean)."""

    def __init__(self, name: str = "CoverOneSD", *args, **kwargs):
        """Initialize the CoverOneSD metric.

        Args:
            name: The name of the metric. Defaults to "CoverOneSD".
            *args: Additional positional arguments passed to BaseMetric.
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
        super().__init__(name, *args, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        selected_ensemble: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Compute the percentage coverage within one standard deviation of the ensemble mean.

        Args:
            forecast: The forecast DataArray containing ensemble members.
            target: The target DataArray.
            selected_ensemble: Optional number of ensemble members to select.

        Returns:
            The computed percentage coverage result.
        """
        print("selected_ensemble cover1sd:", selected_ensemble)
        percentage_cov = None
        # Define common ensemble dimension names typically used in meteorological datasets
        possible_ensemble_dims = ["ensemble", "realization", "member", "ens"]
        
        # Automatically check if any of the possible ensemble dimensions exist in the forecast data
        active_ensemble_dim = next((dim for dim in possible_ensemble_dims if dim in forecast.dims), None)
        
        if active_ensemble_dim:
            num_ensemble = forecast.sizes[active_ensemble_dim]
            
            if selected_ensemble is not None and num_ensemble < selected_ensemble:
                raise ValueError(
                    f"Selected ensemble members {selected_ensemble} should be less than "
                    f"or equal to available {num_ensemble} members."
                )
            elif selected_ensemble is not None:
                # Select the subset of ensemble members using the dynamically detected dimension name
                forecast = forecast.sel({active_ensemble_dim: range(1, selected_ensemble + 1)})
            
            # Compute the ensemble mean and standard deviation
            forecast_mean = forecast.mean(dim=active_ensemble_dim, skipna=True)
            forecast_std = forecast.std(dim=active_ensemble_dim, skipna=True)
            
            # Calculate coverage: target within 1 standard deviation of the ensemble mean
            cover1sd = abs(target - forecast_mean) <= forecast_std
            spatial_dims = [dim for dim in ["valid_time", "latitude", "longitude"] if dim in cover1sd.dims]
            
            percentage_cov = cover1sd.mean(dim=spatial_dims, skipna=True) * 100
        else:
            # If no ensemble dimension is found, raise an error
            raise ValueError("No ensemble dimension found in the forecast data.")

        return percentage_cov