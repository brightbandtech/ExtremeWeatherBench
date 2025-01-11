import dataclasses

import numpy as np
import pandas as pd
import xarray as xr
from scores.continuous import rmse
import logging
from extremeweatherbench import utils
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T2M_85TH_PERCENTILE_CLIMATOLOGY_PATH = "gs://brightband-scratch/taylor/climatology/era5_2m_temperature_85th_rolling_by_hour_dayofyear.zarr"


@dataclasses.dataclass
class Metric:
    """A base class defining the interface for ExtremeWeatherBench metrics."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """Evaluate a specific metric given a forecast and observation dataset."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the class name without parentheses."""
        return self.__class__.__name__

    def align_datasets(
        self,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        init_time: datetime.datetime,
    ):
        """Align the forecast and observation datasets."""
        try:
            forecast = forecast.sel(init_time=init_time)
        # handles duplicate initialization times. please try to avoid this situation
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
                forecast, observation, pd.Timestamp(init_time.values).to_pydatetime()
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
        max_mae_values = []
        observation_spatial_mean = observation.mean(["latitude", "longitude"])
        observation_spatial_mean = observation_spatial_mean.where(
            observation_spatial_mean.time.dt.hour % 6 == 0, drop=True
        )
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        for init_time in forecast_spatial_mean.init_time:
            for var in observation_spatial_mean.data_vars:
                if var != "air_temperature":
                    logger.warning("MaximumMAE only supports air_temperature")
                else:
                    max_date = observation_spatial_mean[var].idxmax("time").values
                    max_value = observation_spatial_mean[var].sel(time=max_date).values
                    init_forecast_spatial_mean, _ = self.align_datasets(
                        forecast_spatial_mean,
                        observation_spatial_mean,
                        pd.Timestamp(init_time.values).to_pydatetime(),
                    )

                    if max_date in init_forecast_spatial_mean.time.values:
                        lead_time = init_forecast_spatial_mean.where(
                            init_forecast_spatial_mean.time == max_date, drop=True
                        ).lead_time
                        max_mae_dataarray = xr.DataArray(
                            data=[
                                abs(
                                    init_forecast_spatial_mean.max()[
                                        "air_temperature"
                                    ].values
                                    - max_value
                                )
                            ],
                            dims=["lead_time"],
                            coords={"lead_time": lead_time.values},
                        )
                        max_mae_values.append(max_mae_dataarray)
        # Reverse the lead time so that the minimum lead time is first
        max_mae_dataarray = max_mae_dataarray.isel(lead_time=slice(None, None, -1))
        max_mae_dataarray = utils.expand_lead_times_to_6_hourly(max_mae_dataarray)

        return max_mae_dataarray


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
