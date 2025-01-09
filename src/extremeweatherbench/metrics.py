import dataclasses

import numpy as np
import xarray as xr
from scores.continuous import rmse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: get permissions to upload this to bb bucket
T2M_85TH_PERCENTILE_CLIMATOLOGY_PATH = (
    "/home/taylor/data/era5_2m_temperature_85th_by_hour_dayofyear.zarr"
)


@dataclasses.dataclass
class Metric:
    """A base class defining the interface for ExtremeWeatherBench metrics."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """Evaluate a specific metric given a forecast and observation dataset."""
        raise NotImplementedError

    def name(self) -> str:
        """Return the class name without parentheses."""
        return self.__class__.__name__

    def align_datasets(
        self, forecast: xr.Dataset, observation: xr.Dataset, init_time: np.datetime64
    ):
        """Align the forecast and observation datasets."""
        try:
            forecast = forecast.sel(init_time=init_time)
        except ValueError:
            init_time_duplicate_length = len(
                forecast.where(forecast.init_time == init_time, drop=True).init_time
            )
            if init_time_duplicate_length > 1:
                logger.warning(
                    "init time %s has more than %d forecast associated with it, taking first only",
                    init_time.values,
                    init_time_duplicate_length,
                )
            forecast = forecast.sel(init_time=init_time.values).isel(init_time=0)
        time = init_time.values + np.array(
            forecast["lead_time"], dtype="timedelta64[h]"
        )
        forecast = forecast.assign_coords(time=("lead_time", time))
        forecast = forecast.swap_dims({"lead_time": "time"})
        forecast, observation = xr.align(forecast, observation, join="inner")
        return forecast, observation


@dataclasses.dataclass
class DurationME(Metric):
    """Mean error in the duration of an event, in hours.

    Attributes:
        threshold: A numerical value for defining whether an event is occurring.
        threshold_tolerance: A numerical tolerance value for defining whether an
            event is occurring.
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        print(forecast)
        print(observation)


@dataclasses.dataclass
class RegionalRMSE(Metric):
    """Root mean squared error of a regional forecast evalauted against observations."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        rmse_values = []
        for init_time in forecast.init_time:
            init_forecast, subset_observation = self.align_datasets(
                forecast, observation, init_time
            )
            output_rmse = rmse(init_forecast, subset_observation, preserve_dims="time")
            rmse_values.append(output_rmse)
        rmse_dataset = xr.concat(rmse_values, dim="time")
        grouped_fhour_rmse_dataset = rmse_dataset.groupby("lead_time").mean()
        return grouped_fhour_rmse_dataset


@dataclasses.dataclass
class MaximumMAE(Metric):
    """Mean absolute error of forecasted maximum values."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        maximummae_values = []
        for init_time in forecast.init_time:
            logger.info("Computing MaximumMAE for model run %s", init_time.values)
            init_forecast, subset_observation = self.align_datasets(
                forecast, observation, init_time
            )
            observation_max = self.subset_max(subset_observation)
            forecast_max = self.subset_max(init_forecast)
            output_maximummae = abs(forecast_max - observation_max)
            maximummae_values.append(output_maximummae)
        maximummae_dataset = xr.concat(maximummae_values, dim="time")
        breakpoint()
        grouped_fhour_maximummae_dataset = maximummae_dataset.groupby(
            "lead_time"
        ).mean()
        return grouped_fhour_maximummae_dataset

    def subset_max(self, dataset: xr.Dataset):
        """Subset the input dataset to only include the maximum values."""
        subset_dataset = dataset.mean(["latitude", "longitude"]).max()
        return subset_dataset


@dataclasses.dataclass
class MaxMinMAE(Metric):
    """Mean absolute error of the forecasted highest minimum value, rolled up by a
    predefined time interval (e.g. daily).

    Attributes:
        # NOTE(daniel): May work better in the long run to define a custom TimeInterval
        # class, say as tuple[datetime.datetime, datetime.datetime].
        time_interval: A string defining the time interval to roll up the metric.
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        print(forecast)
        print(observation)
        return None


@dataclasses.dataclass
class OnsetME(Metric):
    """Mean error of the onset of an event, in hours.

    Attributes:
        endpoint_extension_criteria: The number of hours beyond the event window
            to potentially include in an analysis.
    """

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        print(forecast)
        print(observation)
        return None
