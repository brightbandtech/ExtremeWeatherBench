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


@dataclasses.dataclass
class Metric:
    """A base class defining the interface for ExtremeWeatherBench metrics."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        """Evaluate a specific metric given a forecast and observation dataset."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the class name without parentheses."""
        return self.__class__.__name__

    def align_datasets(
        self,
        forecast: xr.DataArray,
        observation: xr.DataArray,
        init_time_datetime: datetime.datetime,
    ):
        """Align the forecast and observation datasets."""
        try:
            forecast = forecast.sel(init_time=init_time_datetime)
        # handles duplicate initialization times. please try to avoid this situation
        except ValueError:
            init_time_duplicate_length = len(
                forecast.where(
                    forecast.init_time == init_time_datetime, drop=True
                ).init_time
            )
            if init_time_duplicate_length > 1:
                logger.warning(
                    "init time %s has more than %d forecast associated with it, taking first only",
                    init_time_datetime,
                    init_time_duplicate_length,
                )
            forecast = forecast.sel(init_time=init_time_datetime).isel(init_time=0)
        time = np.array(
            [
                init_time_datetime + pd.Timedelta(hours=int(t))
                for t in forecast["lead_time"]
            ]
        )
        forecast = forecast.assign_coords(time=("lead_time", time))
        forecast = forecast.swap_dims({"lead_time": "time"})
        forecast, observation = xr.align(forecast, observation, join="inner")
        return forecast, observation


@dataclasses.dataclass
class RegionalRMSE(Metric):
    """Root mean squared error of a regional forecast evalauted against observations."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
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

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        max_mae_values = []
        observation_spatial_mean = observation.mean(["latitude", "longitude"])
        observation_spatial_mean = observation_spatial_mean.where(
            observation_spatial_mean.time.dt.hour % 6 == 0, drop=True
        )
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        for init_time in forecast_spatial_mean.init_time:
            if forecast.name == "air_temperature":
                max_date = observation_spatial_mean.idxmax("time").values
                max_value = observation_spatial_mean.sel(time=max_date).values
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
                        data=[abs(init_forecast_spatial_mean.max().values - max_value)],
                        dims=["lead_time"],
                        coords={"lead_time": lead_time.values},
                    )
                    max_mae_values.append(max_mae_dataarray)
        max_mae_full_da = utils.process_dataarray_for_output(max_mae_values)
        return max_mae_full_da


@dataclasses.dataclass
class MaxOfMinTempMAE(Metric):
    """Mean absolute error of forecasted highest minimum temperature values."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        max_min_mae_values = []
        observation_spatial_mean = observation.mean(["latitude", "longitude"])
        observation_spatial_mean = observation_spatial_mean.where(
            observation_spatial_mean.time.dt.hour % 6 == 0, drop=True
        )
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        for init_time in forecast_spatial_mean.init_time:
            if forecast_spatial_mean.name == "air_temperature":
                # Keep only times at 00, 06, 12, and 18Z
                # Group by dayofyear and check if each day has all 4 synoptic times
                valid_days = (
                    observation_spatial_mean.groupby("time.dayofyear").count("time")
                    == 4
                )
                # Only keep days that have all 4 synoptic times
                observation_spatial_mean = observation_spatial_mean.where(
                    observation_spatial_mean.time.dt.dayofyear.isin(
                        valid_days.where(valid_days).dropna(dim="dayofyear").dayofyear
                    ),
                    drop=True,
                )
                max_min_timestamp = observation_spatial_mean.where(
                    (
                        observation_spatial_mean
                        == observation_spatial_mean.groupby("time.dayofyear")
                        .min()
                        .max()
                    ),
                    drop=True,
                ).time
                max_min_value = observation_spatial_mean.sel(
                    time=max_min_timestamp
                ).values
                init_forecast_spatial_mean, _ = self.align_datasets(
                    forecast_spatial_mean,
                    observation_spatial_mean,
                    pd.Timestamp(init_time.values).to_pydatetime(),
                )
                if max_min_timestamp.values in init_forecast_spatial_mean.time.values:
                    lead_time = init_forecast_spatial_mean.where(
                        init_forecast_spatial_mean.time == max_min_timestamp, drop=True
                    ).lead_time
                    max_min_mae_dataarray = xr.DataArray(
                        data=abs(
                            init_forecast_spatial_mean.max().values - max_min_value
                        ),
                        dims=["lead_time"],
                        coords={"lead_time": lead_time.values},
                    )
                    max_min_mae_values.append(max_min_mae_dataarray)
            else:
                raise KeyError("Only air_temperature forecasts are supported.")
        max_min_mae_full_da = utils.process_dataarray_for_output(max_min_mae_values)
        return max_min_mae_full_da


@dataclasses.dataclass
class OnsetME(Metric):
    """Mean error of the onset of an event, in hours."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        raise NotImplementedError("Onset mean error not yet implemented.")


@dataclasses.dataclass
class DurationME(Metric):
    """Mean error in the duration of an event, in hours."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        raise NotImplementedError("Duration mean error not yet implemented.")
