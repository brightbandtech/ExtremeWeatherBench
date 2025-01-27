import dataclasses

import numpy as np
import pandas as pd
import xarray as xr
from scores.continuous import rmse
import logging
from extremeweatherbench import utils

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

    def _align_observations_temporal_resolution(
        self, forecast: xr.DataArray, observation: xr.DataArray
    ) -> xr.DataArray:
        """Align the temporal resolution of the forecast and observation dataarrays,
        in case the observations are at a higher temporal resolution than forecast data.
        Metrics which need a singular timestep from gridded obs will fail if the forecasts
        are not aligned with the observation timestamps (e.g. a 03z minimum temp in observations
        when the forecast only has 00z and 06z timesteps).
        Args:
            forecast: The forecast dataarray to align.
            observation: The observation dataarray to align.
        Returns:
            The aligned observation dataarray.
        """
        obs_time_delta = pd.to_timedelta(np.diff(observation.time).mean())
        forecast_time_delta = pd.to_timedelta(
            np.diff(forecast.lead_time).mean(), unit="h"
        )

        if forecast_time_delta != obs_time_delta:
            if forecast_time_delta > obs_time_delta:
                # Resample observations to match forecast resolution
                observation = observation.resample(time=forecast_time_delta).first()
            else:
                logger.warning(
                    "Observation time resolution (%s) is coarser than forecast time resolution (%s)",
                    obs_time_delta,
                    forecast_time_delta,
                )

        return observation


@dataclasses.dataclass
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


@dataclasses.dataclass
class MaximumMAE(Metric):
    """Mean absolute error of forecasted maximum values."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        if forecast.name == "air_temperature":
            max_mae_values = []
            observation_spatial_mean = observation.mean(["latitude", "longitude"])
            forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
            observation_spatial_mean = self._align_observations_temporal_resolution(
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
                        hours=48,
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
        else:
            raise NotImplementedError(
                "Only air_temperature is currently supported for MaximumMAE."
            )
        max_mae_full_da = utils.process_dataarray_for_output(max_mae_values)
        return max_mae_full_da


@dataclasses.dataclass
class MaxOfMinTempMAE(Metric):
    """Mean absolute error of forecasted highest minimum temperature values."""

    def compute(self, forecast: xr.DataArray, observation: xr.DataArray):
        if forecast.name == "air_temperature":
            max_min_mae_values = []
            observation_spatial_mean = observation.mean(["latitude", "longitude"])
            forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
            # Verify observation_spatial_mean's time resolution matches forecast_spatial_mean
            observation_spatial_mean = self._align_observations_temporal_resolution(
                forecast_spatial_mean, observation_spatial_mean
            )
            observation_spatial_mean = self._truncate_incomplete_days(
                observation_spatial_mean
            )
            max_min_timestamp = self._return_max_min_timestamp(observation_spatial_mean)
            max_min_value = observation_spatial_mean.sel(time=max_min_timestamp).values

            for init_time in forecast_spatial_mean.init_time:
                init_forecast_spatial_mean, _ = utils.temporal_align_dataarrays(
                    forecast_spatial_mean,
                    observation_spatial_mean,
                    pd.Timestamp(init_time.values).to_pydatetime(),
                )
                if max_min_timestamp in init_forecast_spatial_mean.time.values:
                    filtered_forecast = self._truncate_incomplete_days(
                        init_forecast_spatial_mean
                    )
                    filtered_forecast = utils.center_forecast_on_time(
                        filtered_forecast,
                        time=pd.Timestamp(max_min_timestamp),
                        hours=48,
                    )
                    # Ensure that the forecast has a full day of data for each day
                    # after centering on the max of min timestamp
                    filtered_forecast = self._truncate_incomplete_days(
                        filtered_forecast
                    )
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
        else:
            raise NotImplementedError(
                "Only air_temperature is currently supported for MaxOfMinTempMAE."
            )

        max_min_mae_full_da = utils.process_dataarray_for_output(max_min_mae_values)
        return max_min_mae_full_da

    def _truncate_incomplete_days(self, da: xr.DataArray) -> xr.DataArray:
        """Truncate a dataarray to only include full days of data."""
        # Group by dayofyear and check if each day has a complete times
        # Count how many unique hours exist per day in the data
        hours_per_day = len(np.unique(da.time.dt.hour.values))
        valid_days = da.groupby("time.dayofyear").count("time") == hours_per_day
        # Only keep days that have a full set of timestamps
        da = da.where(
            da.time.dt.dayofyear.isin(
                valid_days.where(valid_days).dropna(dim="dayofyear").dayofyear
            ),
            drop=True,
        )
        return da

    def _return_max_min_timestamp(self, da: xr.DataArray) -> pd.Timestamp:
        """Return the timestamp of the maximum minimum temperature in a DataArray."""
        return pd.Timestamp(
            da.where(
                da == da.groupby("time.dayofyear").min().max(),
                drop=True,
            ).time.values[0]
        )


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
