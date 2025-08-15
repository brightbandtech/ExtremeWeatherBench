import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import utils

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
        return self.base_metric._compute_metric(
            **self._compute_applied_metric(
                forecast,
                target,
                **utils.filter_kwargs_for_callable(
                    kwargs, self._compute_applied_metric
                ),
            ),
            **utils.filter_kwargs_for_callable(
                kwargs, self.base_metric._compute_metric
            ),
        )

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
