from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import refactor_scripts as rs
import scores.categorical as cat
import xarray as xr
from scores.continuous import mae, mean_error, rmse

#: Storage/access options for gridded target datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point target dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"

IBTRACS_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"  # noqa: E501


# TODO: assign LSRs to a 0.25 degree grid
class PracticallyPerfectHindcast(rs.DerivedVariable):
    """A derived variable that computes the practically perfect hindcast."""

    name = "practically_perfect_hindcast"
    input_variables = ["report_type"]

    def build(self, data: xr.Dataset) -> xr.DataArray:
        """Process the practically perfect hindcast."""
        self._check_data_for_variables(data)
        pph = rs.practically_perfect_hindcast(
            data[self.input_variables], report_type=["tor", "hail"]
        )
        return pph["practically_perfect"]


class CravenSignificantSevereParameter(rs.DerivedVariable):
    """A derived variable that computes the Craven significant severe parameter."""

    name = "craven_significant_severe_parameter"
    input_variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "2m_relative_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "geopotential",
    ]

    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the Craven significant severe parameter."""
        # cbss_ds = calc.craven_brooks_sig_svr(data[variables],variable_mapping={'pressure':'level', 'dewpoint':
        # 'dewpoint_temperature','temperature':'air_temperature'})
        test_da = data[self.input_variables[0]] * 2
        return test_da


class BinaryContingencyTable(rs.BaseMetric):
    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        preserve_dims: str = "lead_time",
    ):
        return cat.BinaryContingencyManager(
            forecast, target, preserve_dims=preserve_dims
        )


class MAE(rs.BaseMetric):
    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        preserve_dims: str = "lead_time",
    ):
        return mae(forecast, target, preserve_dims=preserve_dims)


class ME(rs.BaseMetric):
    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        preserve_dims: str = "lead_time",
    ):
        return mean_error(forecast, target, preserve_dims=preserve_dims)


class RMSE(rs.BaseMetric):
    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        preserve_dims: str = "lead_time",
    ):
        return rmse(forecast, target, preserve_dims=preserve_dims)


class MaximumMAE(rs.AppliedMetric):
    base_metrics = [MAE]

    def compute_applied_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        tolerance_range: int = 24,
    ):
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
        return self.base_metrics[0]().compute_metric(
            filtered_max_forecast, maximum_value
        )


class MaxMinMAE(rs.AppliedMetric):
    base_metrics = [MAE]

    def compute_applied_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        tolerance_range: int = 24,
    ):
        forecast = forecast.compute().mean(["latitude", "longitude"])
        target = target.compute().mean(["latitude", "longitude"])
        max_min_target_dayofyear = (
            target.groupby("valid_time.dayofyear")
            .map(
                rs.min_if_all_timesteps_present,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=4,
            )
            .idxmax("dayofyear")
            .values
        )
        max_min_target_datetime = target.where(
            target.valid_time.dt.dayofyear == max_min_target_dayofyear, drop=True
        ).idxmin("valid_time")
        max_min_target_value = target.sel(valid_time=max_min_target_datetime)
        subset_forecast = (
            forecast.where(
                (
                    forecast.valid_time
                    >= max_min_target_datetime
                    - np.timedelta64(tolerance_range // 2, "h")
                )
                & (
                    forecast.valid_time
                    <= max_min_target_datetime
                    + np.timedelta64(tolerance_range // 2, "h")
                ),
                drop=True,
            )
            .groupby("valid_time.dayofyear")
            .map(
                rs.min_if_all_timesteps_present_forecast,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=2,
            )
            .min("dayofyear")
        )
        return self.base_metrics[0]().compute_metric(
            subset_forecast, max_min_target_value
        )


class OnsetME(rs.AppliedMetric):
    base_metrics = [ME]

    def onset(self, forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                rs.min_if_all_timesteps_present,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=4,
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
                    else:
                        return xr.DataArray(np.datetime64("NaT", "ns"))
            else:
                return xr.DataArray(np.datetime64("NaT", "ns"))
        else:
            return xr.DataArray(np.datetime64("NaT", "ns"))

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        target_time = target.valid_time[0] + np.timedelta64(48, "h")
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.onset)
        )
        return self.base_metrics[0]().compute_metric(
            forecast, target_time, preserve_dims="init_time"
        )


class DurationME(rs.AppliedMetric):
    base_metrics = [ME]

    def duration(self, forecast: xr.DataArray) -> xr.DataArray:
        if (forecast.valid_time.max() - forecast.valid_time.min()).values.astype(
            "timedelta64[h]"
        ) >= 48:
            min_daily_vals = forecast.groupby("valid_time.dayofyear").map(
                rs.min_if_all_timesteps_present,
                # TODO: calculate num timesteps per day dynamically
                num_timesteps=4,
            )
            # need to determine logic for 2+ consecutive days to find the date that the heatwave starts
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
                    else:
                        return xr.DataArray(np.timedelta64("NaT", "ns"))
            else:
                return xr.DataArray(np.timedelta64("NaT", "ns"))
        else:
            return xr.DataArray(np.timedelta64("NaT", "ns"))

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for duration mean error
        target_duration = target.valid_time[-1] - target.valid_time[0]
        forecast = (
            forecast.mean(["latitude", "longitude"])
            .groupby("init_time")
            .map(self.duration)
        )
        return self.base_metrics[0]().compute_metric(
            forecast, target_duration, preserve_dims="init_time"
        )


class CSI(rs.AppliedMetric):
    base_metrics = [BinaryContingencyTable]

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for Critical Success Index
        return self.base_metrics[0]().compute_metric(forecast, target)


class LeadTimeDetection(rs.AppliedMetric):
    base_metrics = [MAE]

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for lead time detection
        return self.base_metrics[0]().compute_metric(forecast, target)


class RegionalHitsMisses(rs.AppliedMetric):
    base_metrics = [BinaryContingencyTable]

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for regional hits and misses
        return self.base_metrics[0]().compute_metric(forecast, target)


class HitsMisses(rs.AppliedMetric):
    base_metrics = [BinaryContingencyTable]

    def __init__(
        self, variables: list[str | rs.DerivedVariable], threshold: float = 0.5
    ):
        super().__init__(variables)
        self.threshold = threshold

    def compute_applied_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for hits and misses
        return self.base_metrics[0]().compute_metric(forecast, target)


class ERA5(rs.TargetBase):
    """
    Target class for ERA5 gridded data.

    The easiest approach to using this class
    is to use the ARCO ERA5 dataset provided by Google for a source. Otherwise, either a
    different zarr source or modifying the open_data_from_source method to open the data
    using another method is required.
    """

    source: str = ARCO_ERA5_FULL_URI

    def open_data_from_source(
        self,
        target_storage_options: Optional[dict] = None,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
    ) -> rs.IncomingDataInput:
        data = xr.open_zarr(
            self.source,
            storage_options=target_storage_options,
            chunks=chunks,
        )
        return data

    def subset_data_to_case(
        self,
        data: rs.IncomingDataInput,
        case_operator: rs.CaseOperator,
    ) -> rs.IncomingDataInput:
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise ValueError(f"Expected xarray Dataset or DataArray, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_data = data.sel(
            valid_time=slice(case_operator.case.start_date, case_operator.case.end_date)
        )

        # check that the variables are in the target data
        target_variables = [
            v for v in case_operator.target_variables if isinstance(v, str)
        ]
        if target_variables and any(
            var not in subset_time_data.data_vars for var in target_variables
        ):
            raise ValueError(f"Variables {target_variables} not found in target data")
        # subset the variables
        elif target_variables:
            subset_time_variable_data = subset_time_data[target_variables]
        else:
            raise ValueError(
                "Variables not defined for ERA5. Please list at least one variable to select."
            )
        # mask the data to the case location
        fully_subset_data = case_operator.case.location.mask(
            subset_time_variable_data, drop=True
        )

        return fully_subset_data

    def _custom_convert_to_dataset(self, data: rs.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        return data


class GHCN(rs.TargetBase):
    """
    Target class for GHCN tabular data.

    Data is processed using polars to maintain the lazy loading
    paradigm in open_data_from_source and to separate the subsetting
    into subset_data_to_case.
    """

    source: str = DEFAULT_GHCN_URI

    def open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> rs.IncomingDataInput:
        target_data: pl.LazyFrame = pl.scan_parquet(
            self.source, storage_options=target_storage_options
        )

        return target_data

    def subset_data_to_case(
        self,
        target_data: rs.IncomingDataInput,
        case: rs.CaseOperator,
    ) -> rs.IncomingDataInput:
        # Create filter expressions for LazyFrame
        time_min = case.case.start_date - pd.Timedelta(days=2)
        time_max = case.case.end_date + pd.Timedelta(days=2)

        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Apply filters using proper polars expressions
        subset_target_data = target_data.filter(
            (pl.col("time") >= time_min)
            & (pl.col("time") <= time_max)
            & (pl.col("latitude") >= case.case.location.latitude_min)
            & (pl.col("latitude") <= case.case.location.latitude_max)
            & (pl.col("longitude") >= case.case.location.longitude_min)
            & (pl.col("longitude") <= case.case.location.longitude_max)
        )

        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        target_variables = [v for v in case.target_variables if isinstance(v, str)]
        if target_variables is None:
            all_variables = ["time", "latitude", "longitude"]
        else:
            all_variables = target_variables + ["time", "latitude", "longitude"]

        # check that the variables are in the target data
        schema_fields = [field for field in subset_target_data.collect_schema()]
        if target_variables and any(var not in schema_fields for var in all_variables):
            raise ValueError(f"Variables {all_variables} not found in target data")

        # subset the variables
        if target_variables:
            subset_target_data = subset_target_data.select(all_variables)

        return subset_target_data

    def _custom_convert_to_dataset(self, data: rs.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_pandas()
            data = data.set_index(["time", "latitude", "longitude"])
            # GHCN data can have duplicate values right now, dropping here if it occurs
            try:
                data = data.to_xarray()
            except ValueError as e:
                if "non-unique" in str(e):
                    pass
                data = data.drop_duplicates().to_xarray()
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")


class LSR(rs.TargetBase):
    """
    Target class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods.
    """

    source: str = LSR_URI

    def open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> rs.IncomingDataInput:
        # force LSR to use anon token to prevent google reauth issues for users
        target_data = pd.read_parquet(
            self.source, storage_options=target_storage_options
        )

        return target_data

    def subset_data_to_case(
        self,
        target_data: rs.IncomingDataInput,
        case: rs.CaseOperator,
    ) -> rs.IncomingDataInput:
        if not isinstance(target_data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(target_data)}")

        # latitude, longitude are strings by default, convert to float
        target_data["lat"] = target_data["lat"].astype(float)
        target_data["lon"] = target_data["lon"].astype(float)
        target_data["time"] = pd.to_datetime(target_data["time"])

        filters = (
            (target_data["time"] >= case.case.start_date)
            & (target_data["time"] <= case.case.end_date)
            & (target_data["lat"] >= case.case.location.latitude_min)
            & (target_data["lat"] <= case.case.location.latitude_max)
            & (
                target_data["lon"]
                >= rs.convert_longitude_to_180(case.case.location.longitude_min)
            )
            & (
                target_data["lon"]
                <= rs.convert_longitude_to_180(case.case.location.longitude_max)
            )
        )

        subset_target_data = target_data.loc[filters]

        subset_target_data = subset_target_data.rename(
            columns={"lat": "latitude", "lon": "longitude", "time": "valid_time"}
        )

        return subset_target_data

    def _custom_convert_to_dataset(self, data: rs.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pd.DataFrame):
            data = data.set_index(["valid_time", "latitude", "longitude"])
            data = xr.Dataset.from_dataframe(
                data[~data.index.duplicated(keep="first")], sparse=True
            )
            return data
        else:
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")


class IBTrACS(rs.TargetBase):
    """
    Target class for IBTrACS data.
    """

    source: str = IBTRACS_URI

    def open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> rs.IncomingDataInput:
        # not using storage_options in this case due to NetCDF4Backend not supporting them
        target_data: pl.LazyFrame = pl.scan_csv(
            self.source, storage_options=target_storage_options
        )
        return target_data

    def subset_data_to_case(
        self,
        target_data: rs.IncomingDataInput,
        case: rs.CaseOperator,
    ) -> rs.IncomingDataInput:
        # Create filter expressions for LazyFrame
        year = case.case.start_date.year

        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Apply filters using proper polars expressions
        subset_target_data = target_data.filter(
            (pl.col("NAME") == case.case.title.upper())
        )

        all_variables = [
            "SEASON",
            "NUMBER",
            "NAME",
            "ISO_TIME",
            "LAT",
            "LON",
            "WMO_WIND",
            "USA_WIND",
            "WMO_PRES",
            "USA_PRES",
        ]
        # Get the season (year) from the case start date, cast as string as polars is interpreting the schema as strings
        season = str(year)

        # First filter by name to get the storm data
        subset_target_data = target_data.filter(
            (pl.col("NAME") == case.case.title.upper())
        )

        # Create a subquery to find all storm numbers in the same season
        matching_numbers = (
            subset_target_data.filter(pl.col("SEASON") == season)
            .select("NUMBER")
            .unique()
        )

        # Apply the filter to get all data for storms with the same number in the same season
        # This maintains the lazy evaluation
        subset_target_data = target_data.join(
            matching_numbers, on="NUMBER", how="inner"
        ).filter(
            (pl.col("NAME") == case.case.title.upper()) & (pl.col("SEASON") == season)
        )

        # check that the variables are in the target data
        schema_fields = [field for field in subset_target_data.collect_schema()]
        target_variables = [v for v in case.target_variables if isinstance(v, str)]
        if target_variables and any(var not in schema_fields for var in all_variables):
            raise ValueError(f"Variables {all_variables} not found in target data")

        # subset the variables
        if target_variables:
            subset_target_data = subset_target_data.select(all_variables)

        return subset_target_data

    def _custom_convert_to_dataset(self, data: rs.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_pandas()
            data = data.set_index(["ISO_TIME"])
            try:
                data = data.to_xarray()
            except ValueError as e:
                if "non-unique" in str(e):
                    pass
                data = data.drop_duplicates().to_xarray()
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")


class KerchunkForecast(rs.Forecast):
    """
    Forecast class for kerchunked forecast data.
    """

    def __init__(self, forecast_source: str):
        super().__init__(forecast_source)

    def open_data_from_source(
        self,
        forecast_storage_options: Optional[dict] = None,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
    ) -> rs.IncomingDataInput:
        return rs.open_kerchunk_reference(
            self.forecast_source,
            storage_options=forecast_storage_options,
            chunks=chunks,
        )


class ZarrForecast(rs.Forecast):
    """
    Forecast class for zarr forecast data.
    """

    def __init__(self, forecast_source: str):
        super().__init__(forecast_source)

    def open_data_from_source(
        self,
        forecast_storage_options: Optional[dict] = None,
        chunks: dict = {"valid_time": 48, "latitude": 721, "longitude": 1440},
    ) -> rs.IncomingDataInput:
        return xr.open_zarr(
            self.forecast_source, storage_options=forecast_storage_options
        )


class HeatWave(rs.EventType):
    event_type = "heat_wave"
    metric_evaluation_objects: list[rs.MetricEvaluationObject]


class SevereConvection(rs.EventType):
    event_type = "severe_convection"
    forecast_variables = [
        CravenSignificantSevereParameter,
    ]
    target_variables = [
        PracticallyPerfectHindcast,
    ]
    metrics = [CSI, LeadTimeDetection, RegionalHitsMisses, HitsMisses]
    targets = [LSR]


class AtmosphericRiver(rs.EventType):
    event_type = "atmospheric_river"
    forecast_variables = []
    target_variables = []
    metrics = [CSI, LeadTimeDetection]
    targets = [ERA5]
