import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import calc, utils

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
    preserve_dims: str = "init_time"

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
    """
    Computes the haversine distance between forecast and analysis landfall points.

    This metric finds the landfall points for both forecast and target tropical
    cyclone tracks and computes the spatial displacement in kilometers using
    the great circle distance formula.
    """

    preserve_dims: str = "init_time"

    @classmethod
    def _compute_metric(cls, forecast, target, **kwargs: Any) -> Any:
        """
        Compute landfall displacement between forecast and target TC tracks.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments

        Returns:
            xarray.DataArray with landfall displacement distances in km
        """
        from extremeweatherbench.events.tropical_cyclone import find_landfall_xarray

        # Convert DataArrays to Datasets if needed
        if isinstance(forecast, xr.DataArray):
            forecast = forecast.to_dataset()
        if isinstance(target, xr.DataArray):
            target = target.to_dataset()

        # Find landfall points using pure xarray
        forecast_landfall = find_landfall_xarray(forecast)
        target_landfall = find_landfall_xarray(target)

        if forecast_landfall is None or target_landfall is None:
            # Handle case where no landfall is found
            # We need to determine what init_times exist in the original forecast data

            # Check if forecast has lead_time dimension (the original structure)
            if "lead_time" in forecast.dims:
                # Calculate init_times from lead_time and valid_time
                init_times_calc = forecast.valid_time - forecast.lead_time
                # Get unique init_times
                unique_init_times = np.unique(init_times_calc.values)

                # Create NaN result with init_time dimension
                nan_distances = xr.DataArray(
                    np.full(len(unique_init_times), np.nan),
                    dims=["init_time"],
                    coords={"init_time": unique_init_times},
                )

                return nan_distances
            else:
                # Scalar case - return NaN scalar
                return xr.DataArray(np.nan)

        # Check if we have init_time dimension in forecast
        if "init_time" in forecast_landfall.dims:
            # Vector case - compute distance for each init_time
            distances = []
            for i in range(len(forecast_landfall.init_time)):
                f_lat = forecast_landfall.latitude.isel(init_time=i).values
                f_lon = forecast_landfall.longitude.isel(init_time=i).values
                t_lat = target_landfall.latitude.values  # Target is scalar
                t_lon = target_landfall.longitude.values

                # Skip if any coordinates are NaN
                if (
                    np.isnan(f_lat)
                    or np.isnan(f_lon)
                    or np.isnan(t_lat)
                    or np.isnan(t_lon)
                ):
                    distances.append(np.nan)
                else:
                    dist = calc.calculate_haversine_distance(
                        [f_lat, f_lon], [t_lat, t_lon], units="km"
                    )
                    distances.append(dist)

            distance_result = xr.DataArray(
                distances,
                dims=["init_time"],
                coords={"init_time": forecast_landfall.init_time},
            )
        else:
            # Scalar case
            f_lat = forecast_landfall.latitude.values
            f_lon = forecast_landfall.longitude.values
            t_lat = target_landfall.latitude.values
            t_lon = target_landfall.longitude.values

            if np.isnan(f_lat) or np.isnan(f_lon) or np.isnan(t_lat) or np.isnan(t_lon):
                distance_km = np.nan
            else:
                distance_km = calc.calculate_haversine_distance(
                    [f_lat, f_lon], [t_lat, t_lon], units="km"
                )

            distance_result = xr.DataArray(distance_km)

        return distance_result


class LandfallTimeME(AppliedMetric):
    """
    Computes the mean error in landfall timing between forecast and analysis tracks.

    This metric finds the landfall points for both forecast and target tropical
    cyclone tracks and computes the time difference in hours.
    """

    @property
    def base_metric(self) -> type[BaseMetric]:
        return ME

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """
        Extract landfall times from forecast and target TC tracks.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments

        Returns:
            Dict containing forecast and target landfall times for ME calculation
        """
        import pandas as pd

        from extremeweatherbench.events.tropical_cyclone import (
            calculate_landfall_time_difference_hours_xarray,
            find_landfall_xarray,
        )

        # Find landfall points using pure xarray
        forecast_landfall = find_landfall_xarray(forecast)
        target_landfall = find_landfall_xarray(target)

        if forecast_landfall is None or target_landfall is None:
            # Return NaN if either track doesn't make landfall
            return {
                "forecast": xr.DataArray(pd.NaT),
                "target": xr.DataArray(pd.NaT),
                "preserve_dims": self.base_metric.preserve_dims,
            }

        # Calculate time difference directly
        time_diff_hours = calculate_landfall_time_difference_hours_xarray(
            forecast_landfall, target_landfall
        )

        result = {
            "forecast": time_diff_hours,
            "target": xr.zeros_like(time_diff_hours),  # Target as reference (0 error)
        }

        # Only specify preserve_dims if we have dimensions to preserve
        if hasattr(time_diff_hours, "dims") and time_diff_hours.dims:
            preserve_dims = [dim for dim in time_diff_hours.dims if dim == "init_time"]
            if preserve_dims:  # Only add if there are dims to preserve
                result["preserve_dims"] = preserve_dims

        return result


class LandfallIntensityMAE(AppliedMetric):
    """
    Computes the mean absolute error in landfall intensity between forecast and
    analysis tracks.

    This metric finds the landfall points for both forecast and target tropical
    cyclone tracks and computes the intensity error using wind speed.
    """

    @property
    def base_metric(self) -> type[BaseMetric]:
        return MAE

    def _compute_applied_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        """
        Extract landfall intensities from forecast and target TC tracks.

        Args:
            forecast: Forecast TC track dataset
            target: Target/analysis TC track dataset
            **kwargs: Additional arguments

        Returns:
            Dict containing forecast and target landfall intensities for MAE calculation
        """
        from extremeweatherbench.events.tropical_cyclone import find_landfall_xarray

        # Find landfall points using pure xarray
        forecast_landfall = find_landfall_xarray(forecast)
        target_landfall = find_landfall_xarray(target)

        if forecast_landfall is None or target_landfall is None:
            # Return NaN if either track doesn't make landfall
            return {
                "forecast": xr.DataArray(np.nan),
                "target": xr.DataArray(np.nan),
                "preserve_dims": self.base_metric.preserve_dims,
            }

        # Extract wind speeds at landfall
        forecast_intensity = forecast_landfall.surface_wind_speed
        target_intensity = target_landfall.surface_wind_speed

        result = {
            "forecast": forecast_intensity,
            "target": target_intensity,
        }

        # Only specify preserve_dims if we have dimensions to preserve
        if hasattr(forecast_intensity, "dims") and forecast_intensity.dims:
            preserve_dims = [
                dim for dim in forecast_intensity.dims if dim == "init_time"
            ]
            if preserve_dims:  # Only add if there are dims to preserve
                result["preserve_dims"] = preserve_dims

        return result


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
