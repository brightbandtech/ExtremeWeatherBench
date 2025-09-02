import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import derived, evaluate, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global cache for transformed contingency managers
_GLOBAL_CONTINGENCY_CACHE: Dict[
    Tuple[int, int, float, float, str], cat.BasicContingencyManager
] = {}


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
    try:
        forecast_hash = hash(forecast.to_array().values.tobytes())
        target_hash = hash(target.to_array().values.tobytes())
    except (TypeError, AttributeError):
        # Fallback to object id if hashing fails
        forecast_hash = id(forecast)
        target_hash = id(target)

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


def tp(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract true positive count from transformed contingency manager."""
    counts = transformed_manager.get_counts()
    return counts["tp_count"] / counts["total_count"]


def fp(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract false positive count from transformed contingency manager."""
    counts = transformed_manager.get_counts()
    return counts["fp_count"] / counts["total_count"]


def tn(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract true negative count from transformed contingency manager."""
    counts = transformed_manager.get_counts()
    return counts["tn_count"] / counts["total_count"]


def fn(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract false negative count from transformed contingency manager."""
    counts = transformed_manager.get_counts()
    return counts["fn_count"] / counts["total_count"]


def csi_function(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract CSI from transformed contingency manager."""
    return transformed_manager.critical_success_index()


def far_function(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract FAR from transformed contingency manager."""
    return transformed_manager.false_alarm_ratio()


def accuracy_function(transformed_manager: cat.BasicContingencyManager) -> xr.DataArray:
    """Extract accuracy from transformed contingency manager."""
    return transformed_manager.accuracy()


# Factory functions for commonly used metrics with thresholds
def CSI(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a CSI metric.
    Uses global caching for better performance."""

    class CachedCSI:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"CSI_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedCSI()


def FAR(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a FAR metric.
    Uses global caching for better performance."""

    class CachedFAR:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"FAR_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedFAR()


# Factory functions for contingency table components
def TP(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a True Positive metric.
    Uses global caching for better performance."""

    class CachedTP:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"TP_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedTP()


def FP(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a False Positive metric.
    Uses global caching for better performance."""

    class CachedFP:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"FP_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedFP()


def TN(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a True Negative metric.
    Uses global caching for better performance."""

    class CachedTN:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"TN_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedTN()


def FN(forecast_threshold: float = 0.5, target_threshold: float = 0.5, **kwargs):
    """Create a False Negative metric.
    Uses global caching for better performance."""

    class CachedFN:
        def __init__(self):
            self.forecast_threshold = forecast_threshold
            self.target_threshold = target_threshold
            self.preserve_dims = kwargs.get("preserve_dims", "lead_time")

        @property
        def name(self) -> str:
            return f"FN_fcst{self.forecast_threshold}_tgt{self.target_threshold}"

        @classmethod
        def _compute_metric(
            cls,
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
            return counts["fn_count"] / counts["total_count"]

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            # Override preserve_dims from kwargs if provided
            preserve_dims = kwargs.get("preserve_dims", self.preserve_dims)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "preserve_dims"}

            return self._compute_metric(
                forecast,
                target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=preserve_dims,
                **kwargs_filtered,
            )

    return CachedFN()


class CachedContingencyManager:
    """A container that caches the transformed contingency manager for reuse.

    This allows multiple metrics to share the same transformed contingency
    manager when they use identical thresholds, avoiding redundant computations.
    """

    def __init__(
        self,
        forecast_threshold: float = 0.5,
        target_threshold: float = 0.5,
        preserve_dims: str = "lead_time",
    ):
        self.forecast_threshold = forecast_threshold
        self.target_threshold = target_threshold
        self.preserve_dims = preserve_dims
        self._transformed_cache = None
        self._cache_key = None

    def get_transformed(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        preserve_dims: Optional[str] = None,
    ) -> cat.BasicContingencyManager:
        """Get the transformed contingency manager, using global cache."""
        preserve_dims = preserve_dims or self.preserve_dims

        # Use global cache function
        return get_cached_transformed_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=self.forecast_threshold,
            target_threshold=self.target_threshold,
            preserve_dims=preserve_dims,
        )


def create_threshold_metrics(
    forecast_threshold: float = 0.5,
    target_threshold: float = 0.5,
    preserve_dims: str = "lead_time",
    functions: Optional[List[Callable]] = None,
    instances: Optional[List[Callable]] = None,
):
    """Create multiple metrics that share the same cached transformed manager.

    Args:
        forecast_threshold: Threshold for binarizing forecast data
        target_threshold: Threshold for binarizing target data
        preserve_dims: Dimensions to preserve during contingency table computation
        functions: List of functions to compute (e.g., [csi_function, far_function])
        instances: List of instance functions (e.g., [tp, fp, tn, fn])

    Returns:
        A list of metric objects that share the same cached transformation
    """
    if functions is None:
        functions = [csi_function, far_function, accuracy_function]
    if instances is None:
        instances = [tp, fp, tn, fn]

    # Create shared cache manager
    cache_manager = CachedContingencyManager(
        forecast_threshold=forecast_threshold,
        target_threshold=target_threshold,
        preserve_dims=preserve_dims,
    )

    metrics = []

    # Create function-based metrics
    for func in functions:
        metric = create_cached_function_metric(cache_manager, func)
        metrics.append(metric)

    # Create instance-based metrics
    for inst in instances:
        metric = create_cached_instance_metric(cache_manager, inst)
        metrics.append(metric)

    return metrics


def create_cached_function_metric(cache_manager: CachedContingencyManager, func):
    """Create a metric that uses a cached transformed manager."""

    class CachedFunctionMetric:
        def __init__(self):
            self.cache_manager = cache_manager
            self.func = func
            self._custom_name = (
                f"{func.__name__}_fcst{cache_manager.forecast_threshold}"
                f"_tgt{cache_manager.target_threshold}"
            )

        @property
        def name(self) -> str:
            return self._custom_name

        @classmethod
        def _compute_metric(
            cls,
            forecast: xr.Dataset,
            target: xr.Dataset,
            cache_manager: Optional[CachedContingencyManager] = None,
            func: Optional[Callable] = None,
            **kwargs: Any,
        ) -> Any:
            if cache_manager is None or func is None:
                raise ValueError("cache_manager and func must be provided")
            preserve_dims = kwargs.get("preserve_dims", cache_manager.preserve_dims)
            transformed = cache_manager.get_transformed(forecast, target, preserve_dims)
            return func(transformed)

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            return self._compute_metric(
                forecast,
                target,
                cache_manager=self.cache_manager,
                func=self.func,
                **kwargs,
            )

    return CachedFunctionMetric()


def create_cached_instance_metric(cache_manager: CachedContingencyManager, inst):
    """Create an instance metric that uses a cached transformed manager."""

    class CachedInstanceMetric:
        def __init__(self):
            self.cache_manager = cache_manager
            self.inst = inst
            self._custom_name = (
                f"{inst.__name__}_fcst{cache_manager.forecast_threshold}"
                f"_tgt{cache_manager.target_threshold}"
            )

        @property
        def name(self) -> str:
            return self._custom_name

        @classmethod
        def _compute_metric(
            cls,
            forecast: xr.Dataset,
            target: xr.Dataset,
            cache_manager: Optional[CachedContingencyManager] = None,
            inst: Optional[Callable] = None,
            **kwargs: Any,
        ) -> Any:
            if cache_manager is None or inst is None:
                raise ValueError("cache_manager and inst must be provided")
            preserve_dims = kwargs.get("preserve_dims", cache_manager.preserve_dims)
            transformed = cache_manager.get_transformed(forecast, target, preserve_dims)
            return inst(transformed)

        def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
            return self._compute_metric(
                forecast,
                target,
                cache_manager=self.cache_manager,
                inst=self.inst,
                **kwargs,
            )

    return CachedInstanceMetric()


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
    variables: Optional[list[str | Type["derived.DerivedVariable"]]] = None

    def __init__(
        self,
        forecast_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        target_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
    ):
        self.forecast_variable = forecast_variable
        self.target_variable = target_variable
        if self.forecast_variable and not self.target_variable:
            raise ValueError(
                "Target variable must be provided if forecast variable is provided"
            )
        if self.target_variable and not self.forecast_variable:
            raise ValueError(
                "Forecast variable must be provided if target variable is provided"
            )
        else:
            self.forecast_variable = evaluate._normalize_variable(
                self.forecast_variable
            )
            self.target_variable = evaluate._normalize_variable(self.target_variable)

    @classmethod
    @abstractmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        """Compute the metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        pass

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs,
    ):
        return cls._compute_metric(
            forecast,
            target,
            **utils.filter_kwargs_for_callable(kwargs, cls._compute_metric),
        )


class AppliedMetric(ABC):
    """An applied metric is a derivative of a BaseMetric.

    It is a wrapper around one or more BaseMetrics that is intended for more
    complex rollups or aggregations. Typically, these metrics are used for one
    event type and are very specific. Temporal onset mean error, case duration
    mean error, and maximum temperature mean absolute error, are all examples
    of applied metrics.

    Attributes:
        base_metrics: A list of BaseMetrics to compute.
        compute_applied_metric: A required method to compute the metric.
    """

    base_metric: type[BaseMetric]

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # first, compute the inputs to the base metric, a dictionary of forecast and
        # target
        applied_result = cls._compute_applied_metric(
            forecast,
            target,
            **utils.filter_kwargs_for_callable(kwargs, cls._compute_applied_metric),
        )
        # then, compute the base metric with the inputs
        return cls.base_metric.compute_metric(**applied_result)

    @classmethod
    @abstractmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        """Compute the applied metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the applied metric.
        """
        pass


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


# TODO: base metric for identifying signal and complete implementation
class EarlySignal(BaseMetric):
    """Metric to identify the earliest signal detection in forecast data.

    This metric finds the first occurrence where a signal is detected based on
    threshold criteria and returns the corresponding init_time, lead_time, and
    valid_time information. The metric is designed to be flexible for different
    signal detection criteria that can be specified in applied metrics downstream.
    """

    name = "early_signal"

    @classmethod
    def _compute_metric(
        cls,
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
    name = "maximum_mae"
    base_metric = MAE

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
        maximum_timestep = target_spatial_mean.idxmax("valid_time").values
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)
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
    name = "minimum_mae"
    base_metric = MAE

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
        minimum_timestep = target_spatial_mean.idxmin("valid_time").values
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
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
    name = "max_min_mae"
    base_metric = MAE

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs: Any,
    ) -> Any:
        forecast = forecast.compute().mean(["latitude", "longitude"])
        target = target.compute().mean(["latitude", "longitude"])
        max_min_target_value = (
            target.groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=utils.determine_timesteps_per_day_resolution(target),
            )
            .max()
        )
        max_min_target_datetime = target.where(
            target == max_min_target_value, drop=True
        ).valid_time.values
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
                num_timesteps=utils.determine_timesteps_per_day_resolution(forecast),
            )
            .min("dayofyear")
        )

        return {
            "forecast": subset_forecast,
            "target": max_min_target_value,
            "preserve_dims": cls.base_metric.preserve_dims,
        }


class OnsetME(AppliedMetric):
    name = "onset_me"
    base_metric = ME

    preserve_dims: str = "init_time"

    @staticmethod
    def onset(forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                num_timesteps=utils.determine_timesteps_per_day_resolution(forecast),
            )
            if len(min_daily_vals) >= 2:  # Check if we have at least 2 values
                for i in range(len(min_daily_vals) - 1):
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
    name = "duration_me"
    base_metric = ME
    preserve_dims: str = "init_time"

    @staticmethod
    def duration(forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=4,
            )
            # need to determine logic for 2+ consecutive days to find the date
            # that the heatwave starts
            if len(min_daily_vals) >= 2:  # Check if we have at least 2 values
                for i in range(len(min_daily_vals) - 1):
                    if min_daily_vals[i] >= 288.15 and min_daily_vals[i + 1] >= 288.15:
                        consecutive_days = np.timedelta64(
                            2, "D"
                        )  # Start with 2 since we found first pair
                        for j in range(i + 2, len(min_daily_vals)):
                            if min_daily_vals[j] >= 288.15:
                                consecutive_days += np.timedelta64(1, "D")
                            else:
                                break
                        return xr.DataArray(consecutive_days.astype("timedelta64[ns]"))
        return xr.DataArray(np.timedelta64("NaT", "ns"))

    @classmethod
    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.duration)
        )
        return {
            "forecast": forecast,
            "target": target_duration,
            "preserve_dims": self.preserve_dims,
        }


class LandfallDetection(AppliedMetric):
    preserve_dims: str = "init_time"

    name = "landfall_detection"
    base_metric = EarlySignal

    @classmethod
    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        return {
            "forecast": forecast,
            "target": target,
            "preserve_dims": self.preserve_dims,
            "threshold": 1,
            "variable": "atmospheric_river_land_intersection",
            "comparison": "==",
            "spatial_aggregation": "any",
        }
