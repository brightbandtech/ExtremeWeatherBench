import abc
import logging
from typing import Any

import numpy as np
import scores
import xarray as xr

from extremeweatherbench import utils

logger = logging.getLogger(__name__)


class BaseMetric(abc.ABC):
    """A BaseMetric class is an abstract class that defines the foundational interface
    for all metrics.

    Metrics are general operations applied between a forecast and analysis xarray
    dataset. EWB metrics prioritize the use of any arbitrary sets of forecasts and
    analyses, so long as the spatiotemporal dimensions are the same.
    """

    @property
    def name(self) -> str:
        """The name of the metric.

        Defaults to the class name if not explicitly set.
        """
        return getattr(
            self, "_name", self.__class__.__dict__.get("name", self.__class__.__name__)
        )

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the metric."""
        self._name = value

    @abc.abstractmethod
    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",  # default to preserving lead_time in metrics
    ) -> Any:
        """Compute the metric.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve when computing
                the metric. Defaults to "lead_time".

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

        Filters kwargs to match _compute_metric signature before
        calling the implementation.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            **kwargs: Additional keyword arguments to pass to the
                metric implementation.

        Returns:
            The computed metric result.
        """
        unique_kwargs = utils.filter_kwargs_for_callable(kwargs, self._compute_metric)
        return self._compute_metric(forecast, target, **unique_kwargs)


class MAE(BaseMetric):
    """Mean Absolute Error metric."""

    name = "MAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",
    ) -> Any:
        return scores.continuous.mae(forecast, target, preserve_dims=preserve_dims)


class ME(BaseMetric):
    """Mean Error (bias) metric."""

    name = "ME"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",
    ) -> Any:
        return scores.continuous.mean_error(
            forecast, target, preserve_dims=preserve_dims
        )


class RMSE(BaseMetric):
    """Root Mean Square Error metric."""

    name = "RMSE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",
    ) -> Any:
        return scores.continuous.rmse(forecast, target, preserve_dims=preserve_dims)


# TODO: base metric for identifying signal and complete implementation
class EarlySignal(BaseMetric):
    name = "EarlySignal"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",
    ) -> Any:
        # Dummy implementation for early signal
        raise NotImplementedError("EarlySignal is not implemented yet")


class MaximumMAE(BaseMetric):
    """MAE of the maximum value in a tolerance window.

    Computes the MAE between the forecast and target maximum
    values, where the forecast is filtered to a time window
    around the target's maximum.
    """

    name = "MaximumMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "lead_time",
        tolerance_range: int = 24,
    ) -> Any:
        """Compute MaximumMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve. Defaults to
                "lead_time".
            tolerance_range: Time window (hours) around target
                maximum to search for forecast maximum. Defaults
                to 24 hours.

        Returns:
            MAE of the maximum values.
        """
        forecast = forecast.compute()
        target_spatial_mean = target.compute().mean(["latitude", "longitude"])
        maximum_timestep = target_spatial_mean.idxmax("valid_time")
        maximum_value = target_spatial_mean.sel(valid_time=maximum_timestep)

        # Handle the case where there are >1 resulting target values
        maximum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            maximum_timestep, target.valid_time
        )
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
        return MAE().compute_metric(
            forecast=filtered_max_forecast,
            target=maximum_value,
            preserve_dims=preserve_dims,
        )


class MinimumMAE(BaseMetric):
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
        preserve_dims: str = "lead_time",
        tolerance_range: int = 24,
    ) -> Any:
        """Compute MinimumMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve. Defaults to
                "lead_time".
            tolerance_range: Time window (hours) around target
                minimum to search for forecast minimum. Defaults
                to 24 hours.

        Returns:
            MAE of the minimum values.
        """
        forecast = forecast.compute()
        target_spatial_mean = target.compute().mean(["latitude", "longitude"])
        minimum_timestep = target_spatial_mean.idxmin("valid_time")
        minimum_value = target_spatial_mean.sel(valid_time=minimum_timestep)
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        # Handle the case where there are >1 resulting target values
        minimum_timestep = utils.maybe_get_closest_timestamp_to_center_of_valid_times(
            minimum_timestep, target.valid_time
        )
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
        return MAE().compute_metric(
            forecast=filtered_min_forecast,
            target=minimum_value,
            preserve_dims=preserve_dims,
        )


class MaxMinMAE(BaseMetric):
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
        preserve_dims: str = "lead_time",
        tolerance_range: int = 24,
    ) -> Any:
        """Compute MaxMinMAE.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve. Defaults to
                "lead_time".
            tolerance_range: Time window (hours) around target
                max-min to search for forecast max-min. Defaults
                to 24 hours.

        Returns:
            MAE of the maximum daily minimum values.
        """
        forecast = forecast.compute().mean(["latitude", "longitude"])
        target = target.compute().mean(["latitude", "longitude"])
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
                time_resolution_hours=utils.determine_temporal_resolution(forecast),
            )
            .min("dayofyear")
        )

        return MAE().compute_metric(
            forecast=subset_forecast,
            target=max_min_target_value,
            preserve_dims=preserve_dims,
        )


class OnsetME(BaseMetric):
    """Mean error of heatwave onset time.

    Computes the mean error between forecast and observed timing
    of event onset (currently configured for heatwaves).
    """

    name = "OnsetME"

    def onset(self, forecast: xr.DataArray) -> xr.DataArray:
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
        preserve_dims: str = "init_time",
    ) -> Any:
        """Compute OnsetME.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve. Defaults to
                "init_time".

        Returns:
            Mean error of onset timing.
        """
        target_time = target.valid_time[0] + np.timedelta64(48, "h")
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.onset)
        )
        return ME().compute_metric(
            forecast=forecast,
            target=target_time,
            preserve_dims=preserve_dims,
        )


class DurationME(BaseMetric):
    """Mean error of event duration.

    Computes the mean error between forecast and observed duration
    of an event (currently configured for heatwaves).
    """

    name = "DurationME"

    def duration(self, forecast: xr.DataArray) -> xr.DataArray:
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
        preserve_dims: str = "init_time",
    ) -> Any:
        """Compute DurationME.

        Args:
            forecast: The forecast DataArray.
            target: The target DataArray.
            preserve_dims: Dimension(s) to preserve. Defaults to
                "init_time".

        Returns:
            Mean error of event duration.
        """
        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.duration)
        )
        return ME().compute_metric(
            forecast=forecast,
            target=target_duration,
            preserve_dims=preserve_dims,
        )


# TODO: fill landfall displacement out
class LandfallDisplacement(BaseMetric):
    """Spatial displacement error of landfall location.

    Note: Not yet implemented.
    """

    name = "LandfallDisplacement"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for landfall displacement
        raise NotImplementedError("LandfallDisplacement is not implemented yet")


# TODO: complete landfall time mean error implementation
class LandfallTimeME(BaseMetric):
    """Mean error of landfall time.

    Note: Not yet implemented.
    """

    name = "LandfallTimeME"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for landfall time mean error
        raise NotImplementedError("LandfallTimeME is not implemented yet")


# TODO: complete landfall intensity mean absolute error implementation
class LandfallIntensityMAE(BaseMetric):
    """MAE of landfall intensity.

    Note: Not yet implemented.
    """

    name = "LandfallIntensityMAE"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for landfall intensity mean absolute error
        raise NotImplementedError("LandfallIntensityMAE is not implemented yet")


# TODO: complete spatial displacement implementation
class SpatialDisplacement(BaseMetric):
    """Spatial displacement error metric.

    Note: Not yet implemented.
    """

    name = "SpatialDisplacement"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for spatial displacement
        raise NotImplementedError("SpatialDisplacement is not implemented yet")


# TODO: complete false alarm ratio implementation
class FAR(BaseMetric):
    """False Alarm Ratio metric.

    Note: Not yet implemented.
    """

    name = "FAR"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for False Alarm Rate
        raise NotImplementedError("FAR is not implemented yet")


# TODO: complete CSI implementation
class CSI(BaseMetric):
    """Critical Success Index metric.

    Note: Not yet implemented.
    """

    name = "CSI"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for Critical Success Index
        raise NotImplementedError("CSI is not implemented yet")


# TODO: complete lead time detection implementation
class LeadTimeDetection(BaseMetric):
    """Lead time detection metric.

    Note: Not yet implemented.
    """

    name = "LeadTimeDetection"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for lead time detection
        raise NotImplementedError("LeadTimeDetection is not implemented yet")


# TODO: complete regional hits and misses implementation
class RegionalHitsMisses(BaseMetric):
    """Regional hits and misses metric.

    Note: Not yet implemented.
    """

    name = "RegionalHitsMisses"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for regional hits and misses
        raise NotImplementedError("RegionalHitsMisses is not implemented yet")


# TODO: complete hits and misses implementation
class HitsMisses(BaseMetric):
    """Hits and misses metric.

    Note: Not yet implemented.
    """

    name = "HitsMisses"

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        preserve_dims: str = "init_time",
    ) -> Any:
        # Dummy implementation for hits and misses
        raise NotImplementedError("HitsMisses is not implemented yet")
