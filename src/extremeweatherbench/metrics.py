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
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
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
        forecast_variable: Optional[str | derived.DerivedVariable] = None,
        target_variable: Optional[str | derived.DerivedVariable] = None,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        metrics: Optional[list[Type["ThresholdMetric"]]] = None,
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


class MeanSquaredError(BaseMetric):
    """Compute Mean Squared Error between forecast and target.

    Extends BaseMetric to calculate MSE with optional interval-based
    weighting and custom weights for spatial/temporal averaging.
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
        """Initialize the Early Signal detection metric.

        Args:
            name: The name of the metric. Defaults to "EarlySignal".
            comparison_operator: The comparison operator for signal detection.
            threshold: The threshold value for signal detection.
            spatial_aggregation: Spatial aggregation method. Options: "any"
                (any gridpoint meets criteria), "all" (all gridpoints meet
                criteria), or "half" (at least half meet criteria).
            **kwargs: Additional keyword arguments passed to BaseMetric.
        """
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
            forecast: The forecast dataarray with init_time, lead_time, valid_time.
            target: The target dataarray (used for reference/validation).

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
    """Compute MAE between forecast and target maximum values.

    Extends MeanAbsoluteError to filter forecast to a time window around the
    target's maximum using tolerance_range_hours. Useful for evaluating peak
    value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
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

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.

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
    """Compute MAE between forecast and target minimum values.

    Extends MeanAbsoluteError to filter forecast to a time window around the
    target's minimum using tolerance_range_hours. Useful for evaluating
    minimum value timing and magnitude.
    """

    def __init__(
        self,
        tolerance_range_hours: int = 24,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
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
    """Compute MAE of maximum aggregated minimum values for heatwaves.

    Extends MeanAbsoluteError for heatwave evaluation by aggregating daily
    minimum values and computing MAE between the warmest nighttime (daily
    minimum) temperature in target and forecast.
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
    """Compute mean error of event duration between forecast and target.

    Extends MeanError to compute the mean error between forecast and target
    event durations based on threshold criteria and spatial aggregation.
    """

    def __init__(
        self,
        threshold_criteria: xr.DataArray | float,
        reduce_spatial_dims: list[str] = ["latitude", "longitude"],
        op_func: Union[Callable, Literal[">", ">=", "<", "<=", "==", "!="]] = ">=",
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

        Returns:
            The mean error between forecast and target event durations.
        """
        # Handle criteria - either climatology (xr.DataArray) or float threshold
        # Use local variable to avoid mutating self.threshold_criteria
        threshold_criteria = self.threshold_criteria

        # Need to get climatology into the correct format and interpolation for
        # comparison
        if isinstance(threshold_criteria, xr.DataArray):
            # Climatology case, convert from dayofyear/hour to valid_time.
            # Note that unintended behavior may occur if the case spans multiple years.
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

        # Sum to get durations (NaN values are excluded by default)
        forecast_duration = forecast_mask_final.groupby(self.preserve_dims).sum(
            skipna=True
        )
        target_duration = target_mask_final.groupby(self.preserve_dims).sum(skipna=True)

        if self.product_time_resolution_hours:
            time_resolution_hours = utils.determine_temporal_resolution(forecast)
            forecast_duration = forecast_duration * time_resolution_hours
            target_duration = target_duration * time_resolution_hours

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
                class.
            preserve_dims: The dimensions to preserve. Defaults to "init_time".
            approach: The approach to use for landfall detection. Defaults to "first".
            exclude_post_landfall: Whether to exclude post-landfall data. Defaults to
                False.
            forecast_variable: The forecast variable to use. Defaults to None.
            target_variable: The target variable to use. Defaults to None.
            metrics: A list of metrics to use as a composite. Defaults to None.
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

    def maybe_compute_landfalls(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute landfalls for a given forecast and target dataarray.

        This function computes the landfalls for a given forecast and target dataarray
        using calc.find_landfalls. Currently, this access pattern doesn't include
        passing land geometry in, but calc.find_landfalls will use NaturalEarth's 10m
        land geometry by default.

        Args:
            forecast: The forecast DataArray
            target: The target DataArray
            **kwargs: Additional keyword arguments, may include pre-computed
                forecast_landfall and target_landfall

        Returns:
            Tuple of (forecast_landfall, target_landfall). If no landfalls are found,
            returns NaN DataArrays with init_time dimension.
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

        # Get first forecast landfall per init_time
        forecast_landfalls = calc.find_landfalls(
            forecast, return_next_landfall=return_next_landfall
        )

        # If no forecast landfalls, return NaN DataArrays for both forecast and target
        if forecast_landfalls is None:
            nan_landfalls = utils._create_nan_dataarray(self.preserve_dims)
            return (nan_landfalls, nan_landfalls.copy())

        # Get only first forecast landfall per init_time
        if "landfall" in forecast_landfalls.dims:
            forecast_landfalls = forecast_landfalls.isel(landfall=0)

        # Get all target landfalls
        target_landfalls_pre_init = calc.find_landfalls(
            target, return_next_landfall=return_next_landfall
        )

        # If no target landfalls, return NaN DataArrays for both forecast and target
        if target_landfalls_pre_init is None:
            nan_landfalls = utils._create_nan_dataarray(self.preserve_dims)
            return (nan_landfalls, nan_landfalls.copy())

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
        """Calculate the distance between two landfall points in kilometers or degrees.

        Handles both scalar and multi-dimensional (with init_time) DataArrays.

        Args:
            forecast_landfall: Forecast landfall xarray DataArray
            target_landfall: Target landfall xarray DataArray
            units: The units to use for the distance. Defaults to "km"
        Returns:
            Distance in the specified units as xarray DataArray
        """
        # Find common init_times between forecast and target
        common_init_times = utils.find_common_init_times(
            forecast_landfall, target_landfall
        )

        # If no common init_times, return NaN DataArray
        if not common_init_times:
            return utils._create_nan_dataarray(self.preserve_dims)

        # Compute distance for each common init_time
        distances = []
        for init_time in common_init_times:
            f_lat = forecast_landfall.sel(init_time=init_time).coords["latitude"].values
            f_lon = (
                forecast_landfall.sel(init_time=init_time).coords["longitude"].values
            )
            t_lat = target_landfall.sel(init_time=init_time).coords["latitude"].values
            t_lon = target_landfall.sel(init_time=init_time).coords["longitude"].values

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
        forecast_landfall, target_landfall = self.maybe_compute_landfalls(
            forecast, target, **kwargs
        )
        if not utils.is_valid_landfall(
            forecast_landfall
        ) or not utils.is_valid_landfall(target_landfall):
            return utils._create_nan_dataarray(self.preserve_dims)
        return self.calculate_displacement(
            forecast_landfall,
            target_landfall,
            units=self.units,
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

        Args:
            forecast_landfall: Forecast landfall xarray DataArray.
            target_landfall: Target landfall xarray DataArray.

        Returns:
            Time difference in hours (forecast_landfall - target_landfall)
            as xarray DataArray with init_time dimension.
        """
        # Find common init_times between forecast and target
        common_init_times = utils.find_common_init_times(
            forecast_landfall, target_landfall
        )

        # If no common init_times, return NaN DataArray
        if not common_init_times:
            return utils._create_nan_dataarray(self.preserve_dims)

        # Calculate time difference for each common init_time
        time_diffs = []
        for init_time in common_init_times:
            time1 = forecast_landfall.sel(init_time=init_time).coords["valid_time"]
            time2 = target_landfall.sel(init_time=init_time).coords["valid_time"]

            # Calculate time difference in hours
            time_diff = (time1 - time2) / np.timedelta64(1, "h")
            time_diffs.append(float(time_diff.values))

        return xr.DataArray(
            time_diffs,
            dims=self.preserve_dims,
            coords={self.preserve_dims: common_init_times},
        )

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
        return self.calculate_time_difference(forecast_landfall, target_landfall)


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

        # The complexity of the landfall outputs makes it easier just to use np.abs here
        return np.abs(forecast_landfall - target_landfall)
