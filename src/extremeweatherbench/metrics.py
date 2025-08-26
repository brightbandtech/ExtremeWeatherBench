import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import utils
from extremeweatherbench.events.tropical_cyclone import (
    compute_landfall_metric,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class BinaryContingencyTable(BaseMetric):
    @classmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return cat.BinaryContingencyManager(
            forecast, target, preserve_dims=preserve_dims
        )


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


class LandfallDisplacement(BaseMetric):
    """Landfall displacement metric with configurable landfall detection
    approaches.

    This metric computes the great circle distance between forecast and target
    landfall positions using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target data
    - 'all': Considers all landfalls, filtering by init_time and aggregating

    Parameters:
        approach (str): Landfall detection approach ('first', 'next', 'all')
        aggregation (str): How to aggregate multiple landfalls for 'all'
                          approach ('mean', 'median', 'min', 'max')
        exclude_post_landfall (bool): Whether to exclude init_times after all landfalls
    """

    preserve_dims: str = "init_time"

    def __init__(
        self, approach="first", aggregation="mean", exclude_post_landfall=False
    ):
        """Initialize the landfall displacement metric.

        Args:
            approach: Landfall detection approach ('first', 'next', 'all')
            aggregation: Aggregation method for multiple landfalls ('mean', 'median',
            'min', 'max')
            exclude_post_landfall: Whether to exclude init_times after all landfalls
        """
        super().__init__()
        self.approach = approach
        self.aggregation = aggregation
        self.exclude_post_landfall = exclude_post_landfall

        # Validate parameters
        valid_approaches = ["first", "next", "all"]
        if approach not in valid_approaches:
            raise ValueError(
                f"approach must be one of {valid_approaches}, got {approach}"
            )

        valid_aggregations = ["mean", "median", "min", "max"]
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got {aggregation}"
            )

        # Override compute_metric to inject instance configuration
        def _instance_compute_metric(forecast, target, **kwargs):
            kwargs.update(
                {
                    "approach": self.approach,
                    "aggregation": self.aggregation,
                }
            )
            return self.__class__._compute_metric(forecast, target, **kwargs)

        self.compute_metric = _instance_compute_metric

    @classmethod
    def _compute_metric(cls, forecast, target, **kwargs: Any) -> Any:
        """Compute the landfall displacement using the configured approach.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments (including approach, aggregation)

        Returns:
            xarray.DataArray with landfall displacement distance in km
        """
        approach = kwargs.get("approach", "first")
        aggregation = kwargs.get("aggregation", "mean")

        return compute_landfall_metric(
            forecast,
            target,
            approach=approach,
            metric_type="displacement",
            aggregation=aggregation,
        )


class LandfallTimeME(BaseMetric):
    """Landfall timing metric with configurable landfall detection approaches.

    This metric computes the time difference between forecast and target landfall
    timing using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target data
    - 'all': Considers all landfalls, filtering by init_time and aggregating

    Parameters:
        approach (str): Landfall detection approach ('first', 'next', 'all')
        aggregation (str): How to aggregate multiple landfalls for 'all' approach
                          ('mean', 'median', 'min', 'max', 'abs_mean')
        units (str): Time units for the error ('hours', 'days')
    """

    preserve_dims: str = "init_time"

    def __init__(self, approach="first", aggregation="mean", units="hours"):
        """Initialize the landfall timing metric.

        Args:
            approach: Landfall detection approach ('first', 'next', 'all')
            aggregation: Aggregation method for multiple landfalls
            units: Time units for the error ('hours', 'days')
        """
        super().__init__()
        self.approach = approach
        self.aggregation = aggregation
        self.units = units

        # Validate parameters
        valid_approaches = ["first", "next", "all"]
        if approach not in valid_approaches:
            raise ValueError(
                f"approach must be one of {valid_approaches}, got {approach}"
            )

        valid_aggregations = ["mean", "median", "min", "max", "abs_mean"]
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got {aggregation}"
            )

        valid_units = ["hours", "days"]
        if units not in valid_units:
            raise ValueError(f"units must be one of {valid_units}, got {units}")

        # Override compute_metric to inject instance configuration
        def _instance_compute_metric(forecast, target, **kwargs):
            kwargs.update(
                {
                    "approach": self.approach,
                    "aggregation": self.aggregation,
                    "units": self.units,
                }
            )
            return self.__class__._compute_metric(forecast, target, **kwargs)

        self.compute_metric = _instance_compute_metric

    @classmethod
    def _compute_metric(cls, forecast, target, **kwargs: Any) -> Any:
        """Compute landfall timing error using the configured approach.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments (including approach, aggregation, units)

        Returns:
            xarray.DataArray with landfall timing errors in specified units
        """
        approach = kwargs.get("approach", "first")
        aggregation = kwargs.get("aggregation", "mean")
        units = kwargs.get("units", "hours")

        return compute_landfall_metric(
            forecast,
            target,
            approach=approach,
            metric_type="timing",
            aggregation=aggregation,
            units=units,
        )


class LandfallIntensityMAE(BaseMetric):
    """Landfall intensity metric with configurable landfall detection approaches.

    This metric computes the mean absolute error between forecast and target intensity
    at landfall using different approaches:

    - 'first': Uses the first landfall point for each track
    - 'next': For each init_time, finds the next upcoming landfall in target data
    - 'all': Considers all landfalls, filtering by init_time and aggregating

    Parameters:
        approach (str): Landfall detection approach ('first', 'next', 'all')
        aggregation (str): How to aggregate multiple landfalls for 'all'
                          approach ('mean', 'median', 'min', 'max')
        intensity_var (str): Variable to use for intensity ('surface_wind_speed',
        'minimum_central_pressure')
    """

    preserve_dims: str = "init_time"

    def __init__(
        self, approach="first", aggregation="mean", intensity_var="surface_wind_speed"
    ):
        """Initialize the landfall intensity metric.

        Args:
            approach: Landfall detection approach ('first', 'next', 'all')
            aggregation: Aggregation method for multiple landfalls
            intensity_var: Variable to use for intensity
        """
        super().__init__()
        self.approach = approach
        self.aggregation = aggregation
        self.intensity_var = intensity_var

        # Validate parameters
        valid_approaches = ["first", "next", "all"]
        if approach not in valid_approaches:
            raise ValueError(
                f"approach must be one of {valid_approaches}, got {approach}"
            )

        valid_aggregations = ["mean", "median", "min", "max"]
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got {aggregation}"
            )

        # Override compute_metric to inject instance configuration
        def _instance_compute_metric(forecast, target, **kwargs):
            kwargs.update(
                {
                    "approach": self.approach,
                    "aggregation": self.aggregation,
                    "intensity_var": self.intensity_var,
                }
            )
            return self.__class__._compute_metric(forecast, target, **kwargs)

        self.compute_metric = _instance_compute_metric

    @classmethod
    def _compute_metric(cls, forecast, target, **kwargs: Any) -> Any:
        """
        Compute landfall intensity error using the configured approach.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments (including approach, aggregation,
            intensity_var)

        Returns:
            xarray.DataArray with landfall intensity errors
        """
        approach = kwargs.get("approach", "first")
        aggregation = kwargs.get("aggregation", "mean")
        intensity_var = kwargs.get("intensity_var", "surface_wind_speed")

        return compute_landfall_metric(
            forecast,
            target,
            approach=approach,
            metric_type="intensity",
            aggregation=aggregation,
            intensity_var=intensity_var,
        )


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


# TODO: complete false alarm ratio implementation
class FAR(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return BinaryContingencyTable

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for False Alarm Rate
        raise NotImplementedError("FAR is not implemented yet")


# TODO: complete CSI implementation
class CSI(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return BinaryContingencyTable

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for Critical Success Index
        raise NotImplementedError("CSI is not implemented yet")


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


# TODO: complete regional hits and misses implementation
class RegionalHitsMisses(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return BinaryContingencyTable

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for regional hits and misses
        raise NotImplementedError("RegionalHitsMisses is not implemented yet")


# TODO: complete hits and misses implementation
class HitsMisses(AppliedMetric):
    @property
    def base_metric(self) -> type[BaseMetric]:
        return BinaryContingencyTable

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for hits and misses
        raise NotImplementedError("HitsMisses is not implemented yet")
