import dataclasses

import numpy as np
import xarray as xr


@dataclasses.dataclass
class Metric:
    """A base class defining the interface for ExtremeWeatherBench metrics."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        """Evaluate a specific metric given a forecast and observation dataset."""
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
    threshold: float
    threshold_tolerance: float

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        raise NotImplementedError

    # @property
    # def type(self) -> str:
    #     return "duration_me"
    # NOTE(daniel): Why wouldn't we just make this just the to_string method?
    # In fact, this can be in the base Metric class and we can just define another
    # default, private attribute like "_event_type" that can be over-ridden by
    # every specialized class to return here.
    def __str__(self) -> str:
        return "duration_me"


@dataclasses.dataclass
class RegionalRMSE(Metric):
    """Root mean squared error of a regional forecast evalauted against observations."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        raise NotImplementedError

    @property
    def type(self) -> str:
        return "regional_rmse"


@dataclasses.dataclass
class MaximumMAE(Metric):
    """Mean absolute error of forecasted maximum values."""

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        # print(forecast)
        # print(observation)

        max_t2_times = (
            merged_df.reset_index()
            .groupby("init_time")
            .apply(lambda x: x.loc[x["t2"].idxmax()])
        )
        max_t2_times["model"] = "PanguWeather"
        max_t2_times = max_t2_times[
            max_t2_times.index
            < era5_dataset.case_analysis_ds["time"][
                era5_dataset.case_analysis_ds["2m_temperature"]
                .mean(["latitude", "longitude"])
                .argmax()
                .values
            ].values
        ]
        max_t2_times["time_error"] = abs(
            max_t2_times["time"]
            - era5_dataset.case_analysis_ds["time"][
                era5_dataset.case_analysis_ds["2m_temperature"]
                .mean(["latitude", "longitude"])
                .argmax()
                .values
            ].values
        ) / np.timedelta64(1, "h")
        max_t2_times["t2_mae"] = abs(
            max_t2_times["t2"]
            - era5_dataset.case_analysis_ds["2m_temperature"]
            .mean(["latitude", "longitude"])
            .max()
            .values
        )
        merged_pivot = max_t2_times.pivot(
            index="model", columns="init_time", values="t2_mae"
        )
        raise NotImplementedError

    @property
    def type(self) -> str:
        return "maximum_mae"


@dataclasses.dataclass
class MaxMinMAE(Metric):
    """Mean absolute error of the forecasted highest minimum value, rolled up by a
    predefined time interval (e.g. daily).

    Attributes:
        # NOTE(daniel): May work better in the long run to define a custom TimeInterval
        # class, say as tuple[datetime.datetime, datetime.datetime].
        time_interval: A string defining the time interval to roll up the metric.
    """

    time_interval: str

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        raise NotImplementedError

    @property
    def type(self) -> str:
        return "minmax_mae"


@dataclasses.dataclass
class OnsetME(Metric):
    """Mean error of the onset of an event, in hours.

    Attributes:
        endpoint_extension_criteria: The number of hours beyond the event window
            to potentially include in an analysis.
    """

    endpoint_extension_criteria: float

    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        raise NotImplementedError

    @property
    def type(self) -> str:
        return "maximum_mae"
