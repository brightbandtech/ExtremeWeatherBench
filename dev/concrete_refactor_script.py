from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import refactor_scripts as rs
import scores.categorical as cat
import xarray as xr

#: Storage/access options for gridded observation datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point observation dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"

IBTRACS_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"  # noqa: E501


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
    def compute(self, forecast: xr.Dataset, target: xr.Dataset, **kwargs):
        return cat.BinaryContingencyManager(forecast, target, **kwargs)


class MaximumMAE(rs.AppliedMetric):
    base_metric = [rs.MAE]

    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
    ):
        maximum_timestep = (
            target.mean(["latitude", "longitude"]).idxmax("valid_time").values
        )
        maximum_value = (
            target.mean(["latitude", "longitude"])
            .sel(valid_time=maximum_timestep)
            .values
        )
        forecast_spatial_mean = forecast.mean(["latitude", "longitude"])
        filtered_max_forecast = (
            forecast_spatial_mean.mean(["latitude", "longitude"])
            .where(
                (
                    forecast_spatial_mean.valid_time
                    >= maximum_timestep - np.timedelta64(48, "h")
                )
                & (
                    forecast_spatial_mean.valid_time
                    <= maximum_timestep + np.timedelta64(48, "h")
                ),
                drop=True,
            )
            .max("valid_time")
        )
        return self.base_metric().compute(filtered_max_forecast, maximum_value)


class MaxMinMAE(rs.AppliedMetric):
    base_metric = rs.MAE

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for finding both max and min values
        return self.metric().compute(forecast, target)


class OnsetME(rs.AppliedMetric):
    base_metric = rs.ME

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for onset mean error
        return self.metric().compute(forecast, target)


class DurationME(rs.AppliedMetric):
    base_metric = rs.MAE

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for duration mean error
        return self.metric().compute(forecast, target)


class CSI(rs.AppliedMetric):
    base_metric = BinaryContingencyTable

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for Critical Success Index
        return self.metric().compute(forecast, target)


class LeadTimeDetection(rs.AppliedMetric):
    base_metric = rs.MAE

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for lead time detection
        return self.metric().compute(forecast, target)


class RegionalHitsMisses(rs.AppliedMetric):
    base_metric = BinaryContingencyTable

    def __init__(self, variables: list[str | rs.DerivedVariable]):
        super().__init__(variables)

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for regional hits and misses
        return self.metric().compute(forecast, target)


class HitsMisses(rs.AppliedMetric):
    base_metric = BinaryContingencyTable

    def __init__(
        self, variables: list[str | rs.DerivedVariable], threshold: float = 0.5
    ):
        super().__init__(variables)
        self.threshold = threshold

    def compute_metric(self, forecast: xr.Dataset, target: xr.Dataset):
        # Dummy implementation for hits and misses
        return self.metric().compute(forecast, target)


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
        case: rs.CaseOperator,
    ) -> rs.IncomingDataInput:
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise ValueError(f"Expected xarray Dataset or DataArray, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_data = data.sel(
            time=slice(case.case.start_date, case.case.end_date)
        )

        # check that the variables are in the target data
        target_variables = [v for v in case.target_variables if isinstance(v, str)]
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
        # # calling chunk here to avoid loading subset_data into memory
        chunks = {"time": 48, "latitude": 721, "longitude": 1440}
        subset_time_variable_data = subset_time_variable_data.chunk(chunks)
        # mask the data to the case location
        fully_subset_data = case.case.location.mask(
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
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
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
    observation_variables = [
        rs.PracticallyPerfectHindcast,
    ]
    metrics = [CSI, LeadTimeDetection, RegionalHitsMisses, HitsMisses]
    observations = [LSR]


class AtmosphericRiver(rs.EventType):
    event_type = "atmospheric_river"
    forecast_variables = []
    observation_variables = []
    metrics = [CSI, LeadTimeDetection]
    observations = [ERA5]
