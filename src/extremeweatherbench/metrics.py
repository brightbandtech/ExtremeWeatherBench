import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import utils

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
    preserve_dims: str = "lead_time"

    @property
    def name(self) -> str:
        return self.__class__.__name__

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

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def base_metric(self) -> type[BaseMetric]:
        pass

    def compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> Any:
        # TODO: build a spatial dim/time dim separator to allow for spatial and temporal
        # metrics to be computed separately

        # Filter kwargs for each method
        applied_metric_kwargs = utils.filter_kwargs_for_callable(
            kwargs, self._compute_applied_metric
        )
        base_metric_kwargs = utils.filter_kwargs_for_callable(
            kwargs, self.base_metric._compute_metric
        )

        # Compute the applied metric first
        applied_result = self._compute_applied_metric(
            forecast, target, **applied_metric_kwargs
        )

        # Then compute the base metric with the applied result
        return self.base_metric._compute_metric(**applied_result, **base_metric_kwargs)

    @abstractmethod
    def _compute_applied_metric(
        self,
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
    @classmethod
    def _compute_metric(
        cls, forecast: xr.Dataset, target: xr.Dataset, **kwargs: Any
    ) -> Any:
        # Dummy implementation for early signal
        raise NotImplementedError("EarlySignal is not implemented yet")


class MaximumMAE(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self,
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
            "preserve_dims": self.base_metric().preserve_dims,
        }


class MinimumMAE(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self,
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
            "preserve_dims": self.base_metric.preserve_dims,
        }


class MaxMinMAE(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self,
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
            "preserve_dims": self.base_metric().preserve_dims,
        }


class OnsetME(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return ME

    preserve_dims: str = "init_time"

    def onset(self, forecast: xr.DataArray) -> xr.DataArray:
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

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        target_time = target.valid_time[0] + np.timedelta64(48, "h")
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.onset)
        )
        return {
            "forecast": forecast,
            "target": target_time,
            "preserve_dims": self.preserve_dims,
        }


class DurationME(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return ME

    preserve_dims: str = "init_time"

    def duration(self, forecast: xr.DataArray) -> xr.DataArray:
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


# TODO: fill landfall displacement out
class LandfallDisplacement(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for landfall displacement
        raise NotImplementedError("LandfallDisplacement is not implemented yet")


# TODO: complete landfall time mean error implementation
class LandfallTimeME(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return ME

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for landfall time mean error
        raise NotImplementedError("LandfallTimeME is not implemented yet")


# TODO: complete landfall intensity mean absolute error implementation
class LandfallIntensityMAE(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for landfall intensity mean absolute error
        raise NotImplementedError("LandfallIntensityMAE is not implemented yet")


# TODO: complete spatial displacement implementation
class SpatialDisplacement(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for spatial displacement
        raise NotImplementedError("SpatialDisplacement is not implemented yet")


# TODO: complete lead time detection implementation
class LeadTimeDetection(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for lead time detection
        raise NotImplementedError("LeadTimeDetection is not implemented yet")
