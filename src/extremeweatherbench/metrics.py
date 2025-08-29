import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
import scores.categorical as cat  # type: ignore[import-untyped]
import xarray as xr
from scores.continuous import mae, mean_error, rmse  # type: ignore[import-untyped]

from extremeweatherbench import calc, derived, utils
from extremeweatherbench.events import tropical_cyclone

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
    name: str
    preserve_dims: str = "lead_time"

    def __init__(
        self,
        forecast_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
        target_variable: Optional[str | Type["derived.DerivedVariable"]] = None,
    ):
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
        else:
            # catch if the user provides a DerivedVariable object/class instead of a
            # string or not using the .name attribute
            if not isinstance(self.forecast_variable, str):
                self.forecast_variable = self.forecast_variable.name
            if not isinstance(self.target_variable, str):
                self.target_variable = self.target_variable.name

    @classmethod
    @abstractmethod
    def _compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the base metric.
        """
        pass

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs,
    ) -> xr.DataArray:
        """Compute the metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        return cls._compute_metric(
            forecast,
            target,
            **kwargs,
        )


class AppliedMetric(ABC):
    """An applied metric is a wrapper around a BaseMetric.

    It is a wrapper around a BaseMetric that is intended for more complex
    rollups or aggregations. Typically, these metrics are used for one event
    type and are very specific.

    Temporal onset mean error, case duration mean
    error, and maximum temperature mean absolute error, are all examples of
    applied metrics.

    Attributes:
        base_metric: The BaseMetric to wrap.
        _compute_applied_metric: An abstract method to compute the inputs to the base
        compute_applied_metric: A method to compute the metric.
    """

    base_metric: type[BaseMetric]

    @classmethod
    @abstractmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Logic to compute, roll up, or otherwise transform the inputs for the base
        metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the applied metric.
        """
        pass

    @classmethod
    def compute_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs: Any,
    ) -> xr.DataArray:
        # first, compute the inputs to the base metric, a dictionary of forecast and
        # target
        applied_result = cls._compute_applied_metric(
            forecast,
            target,
            **utils.filter_kwargs_for_callable(kwargs, cls._compute_applied_metric),
        )
        # then, compute the base metric with the inputs
        return cls.base_metric.compute_metric(**applied_result)


class BinaryContingencyTable(BaseMetric):
    """A binary contingency table metric.

    This metric computes a binary contingency table from a forecast and target
    dataset.
    """

    name = "binary_contingency_table"

    def _compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs: Any,
    ) -> Any:
        preserve_dims = kwargs.get("preserve_dims", "lead_time")
        return cat.BinaryContingencyManager(
            forecast, target, preserve_dims=preserve_dims
        )


class MAE(BaseMetric):
    """Mean absolute error.

    This metric computes the mean absolute error between a forecast and target
    dataset.
    """

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


class MaximumMAE(AppliedMetric):
    base_metric = MAE

    name = "maximum_mae"

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs,
    ) -> tuple[xr.DataArray, xr.DataArray]:
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
    base_metric = MAE

    name = "minimum_mae"

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
    base_metric = MAE

    name = "max_min_mae"

    @classmethod
    def _compute_applied_metric(
        cls,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
        **kwargs: Any,
    ) -> Any:
        num_timesteps = utils.determine_timesteps_per_day_resolution(forecast)
        if num_timesteps is None:
            return {
                "forecast": xr.DataArray(np.nan),
                "target": xr.DataArray(np.nan),
                "preserve_dims": None,
            }

        max_min_target_value = (
            target.groupby("valid_time.dayofyear")
            .map(
                utils.min_if_all_timesteps_present,
                num_timesteps=num_timesteps,
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
                num_timesteps=num_timesteps,
            )
            .min("dayofyear")
        )

        return {
            "forecast": subset_forecast,
            "target": max_min_target_value,
            "preserve_dims": cls.base_metric().preserve_dims,
        }


class OnsetME(AppliedMetric):
    base_metric = ME
    preserve_dims: str = "init_time"
    name = "onset_me"

    @staticmethod
    def onset(forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            num_timesteps = utils.determine_timesteps_per_day_resolution(forecast)
            if num_timesteps is None:
                return xr.DataArray(np.datetime64("NaT", "ns"))
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                num_timesteps=num_timesteps,
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
    base_metric = ME

    preserve_dims: str = "init_time"

    name = "duration_me"

    @staticmethod
    def duration(forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            num_timesteps = utils.determine_timesteps_per_day_resolution(forecast)
            if num_timesteps is None:
                return xr.DataArray(np.datetime64("NaT", "ns"))
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                utils.min_if_all_timesteps_present,
                num_timesteps=num_timesteps,
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
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(cls.duration)
        )
        return {
            "forecast": forecast,
            "target": target_duration,
            "preserve_dims": cls.preserve_dims,
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

    name = "landfall_displacement"

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
    def _compute_metric(self, forecast, target, **kwargs: Any) -> Any:
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

        return tropical_cyclone.compute_landfall_metric(
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

    name = "landfall_time_me"

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
        """Compute landfall time error using the configured approach.

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

        return tropical_cyclone.compute_landfall_metric(
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

    name = "landfall_intensity_mae"

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

        return tropical_cyclone.compute_landfall_metric(
            forecast,
            target,
            approach=approach,
            metric_type="intensity",
            aggregation=aggregation,
            intensity_var=intensity_var,
        )


# TODO: complete false alarm ratio implementation
class FAR(AppliedMetric):
    base_metric = BinaryContingencyTable

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for False Alarm Rate
        raise NotImplementedError("FAR is not implemented yet")


# TODO: complete CSI implementation
class CSI(AppliedMetric):
    base_metric = BinaryContingencyTable

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for Critical Success Index
        raise NotImplementedError("CSI is not implemented yet")


# TODO: complete lead time detection implementation
class LeadTimeDetection(AppliedMetric):
    base_metric = MAE

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for lead time detection
        raise NotImplementedError("LeadTimeDetection is not implemented yet")


# TODO: complete regional hits and misses implementation
class RegionalHitsMisses(AppliedMetric):
    base_metric = BinaryContingencyTable

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for regional hits and misses
        raise NotImplementedError("RegionalHitsMisses is not implemented yet")


# TODO: complete hits and misses implementation
class HitsMisses(AppliedMetric):
    base_metric = BinaryContingencyTable

    @classmethod
    def _compute_applied_metric(
        cls, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> Any:
        # Dummy implementation for hits and misses
        raise NotImplementedError("HitsMisses is not implemented yet")


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
    def _compute_metric(
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

        haversine_distance = calc.haversine_distance(
            forecast_com.latitude,
            forecast_com.longitude,
            target_com.latitude,
            target_com.longitude,
            units="km",
        )

        return haversine_distance
