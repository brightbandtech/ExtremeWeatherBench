import dataclasses

import numpy as np
import xarray as xr
from extremeweatherbench import utils
from scores.continuous import rmse

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

    def to_string(self) -> str:
        """Return a string representation of the metric."""
        raise NotImplementedError


@dataclasses.dataclass
class DurationME(Metric):
    """Mean error in the duration of an event, in hours.

    Attributes:
        threshold: A numerical value for defining whether an event is occurring.
        threshold_tolerance: A numerical tolerance value for defining whether an
            event is occurring.
    """

    # NOTE(daniel): We probably need to define a field to which these thresholds
    # are applied, right?

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        print(forecast)
        print(observation)

    # @property
    # def type(self) -> str:
    #     return "duration_me"
    # NOTE(daniel): Why wouldn't we just make this just the to_string method?
    # In fact, this can be in the base Metric class and we can just define another
    # default, private attribute like "_event_type" that can be over-ridden by
    # every specialized class to return here.


@dataclasses.dataclass
class RegionalRMSE(Metric):
    """Root mean squared error of a regional forecast evalauted against observations."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        rmse_values = []
        for init_time in forecast.init_time:
            forecast_values = forecast.sel(init_time=init_time).to_dataarray()
            fhours = forecast_values.time - forecast_values.init_time
            forecast_values = forecast_values.assign_coords(fhours=fhours)
            observation_values = observation.to_dataarray()
            output_rmse = rmse(observation_values, forecast_values)
            rmse_values.append(output_rmse.compute())
        rmse_dataarray = xr.DataArray(
            rmse_values, coords=[forecast_values.fhours], dims=["fhours"], name="rmse"
        )
        return rmse_dataarray


@dataclasses.dataclass
class MaximumMAE(Metric):
    """Mean absolute error of forecasted maximum values."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        rmse_values = []
        for init_time in forecast.init_time:
            forecast_values = forecast.sel(init_time=init_time).to_dataarray()
            fhours = forecast_values.time - forecast_values.init_time
            forecast_values = forecast_values.assign_coords(fhours=fhours)
            observation_values = observation.to_dataarray()
            output_rmse = rmse(observation_values, forecast_values)
            rmse_values.append(output_rmse.compute())
        breakpoint()
        rmse_dataarray = xr.DataArray(
            rmse_values, coords=[forecast.fhours], dims=["fhours"], name="rmse"
        )
        return rmse_dataarray


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
