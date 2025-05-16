"""Metrics, parent and base classes for use during ExtremeWeatherBench case studies / analyses."""

import pandas as pd
import xarray as xr
from scores.continuous import rmse
import scores.categorical as cat
import logging
from extremeweatherbench import utils
import abc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Metric(abc.ABC):
    """A base class defining the interface for ExtremeWeatherBench metrics."""

    @abc.abstractmethod
    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        """Evaluate a specific metric given a forecast and observation dataset."""

    @property
    def name(self) -> str:
        """Return the class name without parentheses."""
        return self.__class__.__name__


class BinaryMetric(abc.ABC):
    """A base class defining the interface for ExtremeWeatherBench binary metrics."""

    @abc.abstractmethod
    def compute(self, binary_operator: cat.BinaryContingencyManager):
        """Evaluate a specific binary metric given a forecast and observation dataset."""


class CategoricalMetric(Metric):
    """A base class defining the interface for ExtremeWeatherBench categorical metrics."""

    def __init__(self, forecast_threshold: float, observation_threshold: float):
        self.forecast_threshold = forecast_threshold
        self.observation_threshold = observation_threshold

    @abc.abstractmethod
    def compute_threshold_outputs(
        self,
        forecast: xr.DataArray,
        observation: xr.DataArray,
    ):
        """Return a threshold(s) for the forecast and observation datasets."""


class BinaryContingencyTable(CategoricalMetric):
    """A binary contingency table for a categorical forecast evaluated against observations."""

    def compute_threshold_outputs(
        self,
        forecast: xr.DataArray,
        observation: xr.DataArray,
    ):
        """Return a binary contingency table for the forecast and observation datasets."""
        # Create boolean masks for the forecast and observation
        self.binary_forecast = forecast.where(forecast >= self.forecast_threshold)
        self.binary_observation = observation.where(
            observation >= self.observation_threshold
        )
        self.contingency_manager = cat.BinaryContingencyManager(
            self.binary_forecast, self.binary_observation
        )

    def compute_all_contingency_tables(self):
        """Return all contingency table metrics for the forecast and observation datasets."""

        # Compute the contingency table
        return self.binary_forecast, self.binary_observation


class Accuracy(BinaryMetric):
    """Accuracy of a binary forecast evaluated against observations."""

    def compute(self, binary_operator: cat.BinaryContingencyManager):
        """Return the accuracy of the binary forecast evaluated against observations."""
        return binary_operator.accuracy()


class FalseAlarmRate(BinaryMetric):
    """False alarm rate of a binary forecast evaluated against observations."""

    def compute(self, binary_operator: cat.BinaryContingencyManager):
        """Return the false alarm rate of the binary forecast evaluated against observations."""
        return binary_operator.false_alarm_rate()


class CriticalSuccessIndex(BinaryMetric):
    """Critical success index of a binary forecast evaluated against observations."""

    def compute(self, binary_operator: cat.BinaryContingencyManager):
        """Return the critical success index of the binary forecast evaluated against observations."""
        return binary_operator.critical_success_index()


class FalseAlarmRatio(BinaryMetric):
    """False alarm ratio of a binary forecast evaluated against observations."""

    def compute(self, binary_operator: cat.BinaryContingencyManager):
        """Return the false alarm ratio of the binary forecast evaluated against observations."""
        return binary_operator.false_alarm_ratio()


class ProbabilityOfDetection(CategoricalMetric):
    """Probability of detection of a categorical forecast evaluated against observations."""

    def compute_threshold_outputs(
        self,
        forecast: xr.DataArray,
        observation: xr.DataArray,
    ):
        """Return a probability of detection for the forecast and observation datasets."""
        probability_of_detection = cat.probability_of_detection(
            fcst=forecast, obs=observation, preserve_dims="time"
        )
        return probability_of_detection


class IntersectionOverUnion(CategoricalMetric):
    """Intersection over union of a categorical forecast evaluated against observations."""

    def compute_threshold_outputs(
        self,
        forecast: xr.DataArray,
        observation: xr.DataArray,
    ):
        # Create boolean masks for the forecast and observation
        self.forecast = forecast.where(forecast >= self.forecast_threshold)
        self.observation = observation.where(observation >= self.observation_threshold)
        return self.forecast, self.observation

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        forecast, observation = self.compute_threshold_or_output(forecast, observation)
        intersection = forecast * observation
        union = forecast + observation
        return intersection.sum() / union.sum()


class RegionalRMSE(Metric):
    """Root mean squared error of a regional forecast evaluated against observations."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        rmse_values = []
        for init_time in forecast.init_time:
            init_forecast, subset_observation = utils.temporal_align_dataarrays(
                forecast,
                observation,
                pd.Timestamp(init_time.values).to_pydatetime(),
            )
            output_rmse = rmse(init_forecast, subset_observation, preserve_dims="time")
            rmse_values.append(output_rmse)
        rmse_dataset = xr.concat(rmse_values, dim="time")
        grouped_fhour_rmse_dataset = rmse_dataset.groupby("lead_time").mean()
        return grouped_fhour_rmse_dataset


class MaximumMAE(Metric):
    """Mean absolute error of forecasted maximum values.

    Attributes:

    time_deviation_tolerance: amount of time in hours to allow for forecast deviation from the observed maximum
    temperature timestamp.
    """

    def __init__(self, time_deviation_tolerance: int = 48):
        self.time_deviation_tolerance = time_deviation_tolerance

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        max_mae_values = []
        observation_spatial_mean = observation.mean(["latitude", "longitude"])
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        observation_spatial_mean = utils.align_observations_temporal_resolution(
            forecast_spatial_mean, observation_spatial_mean
        )
        for init_time in forecast_spatial_mean.init_time:
            max_datetime = observation_spatial_mean.idxmax("time").values
            max_value = observation_spatial_mean.sel(time=max_datetime).values
            init_forecast_spatial_mean, _ = utils.temporal_align_dataarrays(
                forecast_spatial_mean,
                observation_spatial_mean,
                pd.Timestamp(init_time.values).to_pydatetime(),
            )

            if max_datetime in init_forecast_spatial_mean.time.values:
                # Subset to +-48 hours centered on the maximum temperature timestamp
                filtered_forecast = utils.center_forecast_on_time(
                    init_forecast_spatial_mean,
                    time=pd.Timestamp(max_datetime),
                    hours=self.time_deviation_tolerance,
                )
                lead_time = filtered_forecast.where(
                    filtered_forecast.time == max_datetime, drop=True
                ).lead_time
                max_mae_dataarray = xr.DataArray(
                    data=[abs(filtered_forecast.max().values - max_value)],
                    dims=["lead_time"],
                    coords={"lead_time": lead_time.values},
                )
                max_mae_values.append(max_mae_dataarray)

        max_mae_full_da = utils.process_dataarray_for_output(max_mae_values)
        return max_mae_full_da


class MaxOfMinTempMAE(Metric):
    """Mean absolute error of forecasted highest minimum temperature values.

    Attributes:

    time_deviation_tolerance: amount of time in hours to allow for forecast deviation from the observed maximum
    temperature timestamp.
    """

    time_deviation_tolerance: int = 48

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        max_min_mae_values = []
        observation_spatial_mean = observation.mean(["latitude", "longitude"])
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        # Verify observation_spatial_mean's time resolution matches forecast_spatial_mean
        observation_spatial_mean = utils.align_observations_temporal_resolution(
            forecast_spatial_mean, observation_spatial_mean
        )
        observation_spatial_mean = utils.truncate_incomplete_days(
            observation_spatial_mean
        )
        max_min_timestamp = utils.return_max_min_timestamp(observation_spatial_mean)
        max_min_value = observation_spatial_mean.sel(time=max_min_timestamp).values

        for init_time in forecast_spatial_mean.init_time:
            init_forecast_spatial_mean, _ = utils.temporal_align_dataarrays(
                forecast_spatial_mean,
                observation_spatial_mean,
                pd.Timestamp(init_time.values).to_pydatetime(),
            )
            if max_min_timestamp in init_forecast_spatial_mean.time.values:
                filtered_forecast = utils.truncate_incomplete_days(
                    init_forecast_spatial_mean
                )
                filtered_forecast = utils.center_forecast_on_time(
                    filtered_forecast,
                    time=pd.Timestamp(max_min_timestamp),
                    hours=self.time_deviation_tolerance,
                )
                # Ensure that the forecast has a full day of data for each day
                # after centering on the max of min timestamp
                filtered_forecast = utils.truncate_incomplete_days(filtered_forecast)
                lead_time = filtered_forecast.where(
                    filtered_forecast.time == max_min_timestamp, drop=True
                ).lead_time
                filtered_forecast_max_min = filtered_forecast.where(
                    filtered_forecast
                    == filtered_forecast.groupby("time.dayofyear").min().max(),
                    drop=True,
                )
                # TODO: add temporal displacement error, which is
                # filtered_forecast_max_min.time.values[0] - max_min_timestamp
                if max_min_timestamp in filtered_forecast.time.values:
                    max_min_mae_dataarray = xr.DataArray(
                        data=abs(filtered_forecast_max_min - max_min_value),
                        dims=["lead_time"],
                        coords={"lead_time": lead_time.values},
                        attrs={
                            "description": (
                                "Mean absolute error of forecasted highest minimum temperature values,"
                                "where lead_time is the time from initialization until the highest minimum"
                                "observed temperature."
                            )
                        },
                    )
                    max_min_mae_values.append(max_min_mae_dataarray)
        max_min_mae_full_da = utils.process_dataarray_for_output(max_min_mae_values)
        return max_min_mae_full_da


class OnsetME(Metric):
    """Mean error of the onset of an event, in hours."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        raise NotImplementedError("Onset mean error not yet implemented.")


class DurationME(Metric):
    """Mean error in the duration of an event, in hours."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        raise NotImplementedError("Duration mean error not yet implemented.")
