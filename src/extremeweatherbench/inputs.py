import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import derived, utils

if TYPE_CHECKING:
    from extremeweatherbench import case, metrics

#: Storage/access options for gridded target datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point target dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"

PPH_URI = "gs://extremeweatherbench/datasets/practically_perfect_hindcast_20200104_20250430.zarr"

IBTRACS_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"  # noqa: E501

IBTrACS_metadata_variable_mapping = {
    "ISO_TIME": "valid_time",
    "NAME": "tc_name",
    "LAT": "latitude",
    "LON": "longitude",
    "USA_WIND": "surface_wind_speed",
    "USA_PRES": "pressure_at_mean_sea_level",
}


@dataclasses.dataclass
class InputBase(ABC):
    """
    An abstract base dataclass for target and forecast data.

    Attributes:
        source: The source of the data, which can be a local path or a remote URL/URI.
        variables: A list of variables to select from the data.
        variable_mapping: A dictionary of variable names to map to the data.
        storage_options: Storage/access options for the data.
        preprocess: A function to preprocess the data.
    """

    source: str
    variables: list[Union[str, "derived.DerivedVariable"]]
    variable_mapping: dict
    storage_options: dict
    preprocess: Callable = utils._default_preprocess

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def open_and_maybe_preprocess_data_from_source(
        self,
    ) -> utils.IncomingDataInput:
        data = self._open_data_from_source()
        data = self.preprocess(data)
        return data

    @abstractmethod
    def _open_data_from_source(self) -> utils.IncomingDataInput:
        """Open the target data from the source, opting to avoid loading the entire dataset into memory if possible.

        Returns:
            The target data with a type determined by the user.
        """

    @abstractmethod
    def subset_data_to_case(
        self,
        data: utils.IncomingDataInput,
        case: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        """Subset the target data to the case information provided in CaseOperator.

        Time information, spatial bounds, and variables are captured in the case metadata
        where this method is used to subset.

        Args:
            data: The target data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
            pandas dataframe, or numpy array.
            case: The case operator to subset the data to; includes time information, spatial bounds, and variables.

        Returns:
            The target data with the variables subset to the case metadata.
        """

    def maybe_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        """Convert the target data to an xarray dataset if it is not already.

        This method handles the common conversion cases automatically. Override this method
        only if you need custom conversion logic beyond the standard cases.

        Args:
            data: The target data to convert.

        Returns:
            The target data as an xarray dataset.
        """
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            return data.to_dataset()
        else:
            # For other data types, try to use a custom conversion method if available
            return self._custom_convert_to_dataset(data)

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        """Hook method for custom conversion logic. Override this method in subclasses
        if you need custom conversion behavior for non-xarray data types.

        By default, this raises a NotImplementedError to encourage explicit handling
        of custom data types.

        Args:
            data: The target data to convert.

        Returns:
            The target data as an xarray dataset.
        """
        raise NotImplementedError(
            f"Conversion from {type(data)} to xarray.Dataset not implemented. "
            f"Override _custom_convert_to_dataset in your TargetBase subclass."
        )

    def add_source_to_dataset_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Add the name of the dataset to the dataset attributes."""
        ds.attrs["source"] = self.name
        return ds


@dataclasses.dataclass
class ForecastBase(InputBase):
    """A class defining the interface for ExtremeWeatherBench forecast data."""

    def subset_data_to_case(
        self,
        data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_indices = utils.derive_indices_from_init_time_and_lead_time(
            data,
            case_operator.case.start_date,
            case_operator.case.end_date,
        )

        subset_time_data = data.sel(
            init_time=data.init_time[np.unique(subset_time_indices[0])]
        )
        subset_time_data = utils.convert_init_time_to_valid_time(subset_time_data)

        try:
            subset_time_data = subset_time_data[case_operator.forecast.variables]
        except KeyError:
            raise KeyError(
                f"Variables {case_operator.forecast.variables} not found in forecast data"
            )
        fully_subset_data = case_operator.case.location.mask(
            subset_time_data, drop=True
        )
        return fully_subset_data


@dataclasses.dataclass
class EvaluationObject:
    """A class to store the evaluation object for a metric.

    A EvaluationObject is a metric evaluation object for all cases in an event.
    The evaluation is a set of all metrics, target variables, and forecast variables.

    Multiple EvaluationObjects can be used to evaluate a single event type. This is useful for
    evaluating distinct Targets or metrics with unique variables to evaluate.

    Attributes:
        event_type: The event type to evaluate.
        metric: A list of BaseMetric objects.
        target: A TargetBase object.
        forecast: A ForecastBase object.
    """

    event_type: str
    metric: list["metrics.BaseMetric"]
    target: "TargetBase"
    forecast: "ForecastBase"


@dataclasses.dataclass
class KerchunkForecast(ForecastBase):
    """
    Forecast class for kerchunked forecast data.
    """

    chunks: dict = dataclasses.field(
        default_factory=lambda: {"time": 48, "latitude": 721, "longitude": 1440}
    )

    def _open_data_from_source(self) -> utils.IncomingDataInput:
        return open_kerchunk_reference(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
        )


@dataclasses.dataclass
class ZarrForecast(ForecastBase):
    """
    Forecast class for zarr forecast data.
    """

    chunks: dict = dataclasses.field(
        default_factory=lambda: {"time": 48, "latitude": 721, "longitude": 1440}
    )

    def _open_data_from_source(self) -> utils.IncomingDataInput:
        return xr.open_zarr(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
        )


@dataclasses.dataclass
class TargetBase(InputBase):
    """An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """


@dataclasses.dataclass
class ERA5(TargetBase):
    """Target class for ERA5 gridded data.

    The easiest approach to using this class is to use the ARCO ERA5 dataset provided by
    Google for a source. Otherwise, either a different zarr source or modifying the
    open_data_from_source method to open the data using another method is required.
    """

    def _open_data_from_source(
        self,
        target_storage_options: Optional[dict] = None,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
    ) -> utils.IncomingDataInput:
        data = xr.open_zarr(
            self.source,
            storage_options=target_storage_options,
            chunks=chunks,
        )
        return data

    def subset_data_to_case(
        self,
        data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        return zarr_target_subsetter(data, case_operator)


@dataclasses.dataclass
class GHCN(TargetBase):
    """Target class for GHCN tabular data.

    Data is processed using polars to maintain the lazy loading
    paradigm in open_data_from_source and to separate the subsetting
    into subset_data_to_case.
    """

    def _open_data_from_source(
        self,
        target_storage_options: Optional[dict] = None,
    ) -> utils.IncomingDataInput:
        target_data: pl.LazyFrame = pl.scan_parquet(
            self.source, storage_options=target_storage_options
        )

        return target_data

    def subset_data_to_case(
        self,
        target_data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        # Create filter expressions for LazyFrame
        time_min = case_operator.case.start_date - pd.Timedelta(days=2)
        time_max = case_operator.case.end_date + pd.Timedelta(days=2)

        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Apply filters using proper polars expressions
        subset_target_data = target_data.filter(
            (pl.col("time") >= time_min)
            & (pl.col("time") <= time_max)
            & (pl.col("latitude") >= case_operator.case.location.latitude_min)
            & (pl.col("latitude") <= case_operator.case.location.latitude_max)
            & (pl.col("longitude") >= case_operator.case.location.longitude_min)
            & (pl.col("longitude") <= case_operator.case.location.longitude_max)
        )

        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        target_variables = [
            v for v in case_operator.target.variables if isinstance(v, str)
        ]
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

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
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


@dataclasses.dataclass
class LSR(TargetBase):
    """Target class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods for US data. Australia data should be
    00 UTC to 00 UTC.
    """

    def _open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> utils.IncomingDataInput:
        # force LSR to use anon token to prevent google reauth issues for users
        target_data = pd.read_parquet(
            self.source, storage_options=target_storage_options
        )

        return target_data

    def subset_data_to_case(
        self,
        target_data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(target_data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(target_data)}")

        # latitude, longitude are strings by default, convert to float
        target_data["latitude"] = target_data["latitude"].astype(float)
        target_data["longitude"] = target_data["longitude"].astype(float)
        target_data["valid_time"] = pd.to_datetime(target_data["valid_time"])

        filters = (
            (target_data["valid_time"] >= case_operator.case.start_date)
            & (target_data["valid_time"] <= case_operator.case.end_date)
            & (target_data["latitude"] >= case_operator.case.location.latitude_min)
            & (target_data["latitude"] <= case_operator.case.location.latitude_max)
            & (
                target_data["longitude"]
                >= utils.convert_longitude_to_180(
                    case_operator.case.location.longitude_min
                )
            )
            & (
                target_data["longitude"]
                <= utils.convert_longitude_to_180(
                    case_operator.case.location.longitude_max
                )
            )
        )

        subset_target_data = target_data.loc[filters]

        return subset_target_data

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        # Map report_type column to numeric values
        if "report_type" in data.columns:
            report_type_mapping = {"wind": 1, "hail": 2, "tor": 3}
            data["report_type"] = data["report_type"].map(report_type_mapping)

        if isinstance(data, pd.DataFrame):
            # Normalize these times for the LSR data
            # Western hemisphere reports get bucketed to 12Z on the date they fall between 12Z-12Z
            # Eastern hemisphere reports get bucketed to 00Z on the date they occur

            # First, let's figure out which hemisphere each report is in
            western_hemisphere_mask = data["longitude"] < 0
            eastern_hemisphere_mask = data["longitude"] >= 0

            # For western hemisphere: if report is between today 12Z and tomorrow 12Z, assign to today 12Z
            if western_hemisphere_mask.any():
                western_data = data[western_hemisphere_mask].copy()
                # Get the date portion and create 12Z times
                report_dates = western_data["valid_time"].dt.date
                twelve_z_times = pd.to_datetime(report_dates) + pd.Timedelta(hours=12)
                next_day_twelve_z = twelve_z_times + pd.Timedelta(days=1)

                # Check if report falls in the 12Z to 12Z+1day window
                in_window_mask = (western_data["valid_time"] >= twelve_z_times) & (
                    western_data["valid_time"] < next_day_twelve_z
                )
                # For reports that don't fall in today's 12Z window, try yesterday's window
                yesterday_twelve_z = twelve_z_times - pd.Timedelta(days=1)
                in_yesterday_window = (
                    western_data["valid_time"] >= yesterday_twelve_z
                ) & (western_data["valid_time"] < twelve_z_times)

                # Assign 12Z times
                western_data.loc[in_window_mask, "valid_time"] = twelve_z_times[
                    in_window_mask
                ]
                western_data.loc[in_yesterday_window, "valid_time"] = (
                    yesterday_twelve_z[in_yesterday_window]
                )

                data.loc[western_hemisphere_mask] = western_data

            # For eastern hemisphere: assign to 00Z of the same date
            if eastern_hemisphere_mask.any():
                eastern_data = data[eastern_hemisphere_mask].copy()
                # Get the date portion and create 00Z times
                report_dates = eastern_data["valid_time"].dt.date
                zero_z_times = pd.to_datetime(report_dates)
                eastern_data["valid_time"] = zero_z_times

                data.loc[eastern_hemisphere_mask] = eastern_data

            data = data.set_index(["valid_time", "latitude", "longitude"])
            data = xr.Dataset.from_dataframe(
                data[~data.index.duplicated(keep="first")], sparse=True
            )
            data.attrs["report_type_mapping"] = report_type_mapping
            return data
        else:
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")


@dataclasses.dataclass
class PPH(TargetBase):
    """Target class for practically perfect hindcast data."""

    def _open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> utils.IncomingDataInput:
        return xr.open_zarr(self.source, storage_options=target_storage_options)

    def subset_data_to_case(
        self,
        target_data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        return zarr_target_subsetter(target_data, case_operator)

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        return data


@dataclasses.dataclass
class IBTrACS(TargetBase):
    """Target class for IBTrACS data."""

    def _open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> utils.IncomingDataInput:
        # not using storage_options in this case due to NetCDF4Backend not supporting them
        target_data: pl.LazyFrame = pl.scan_csv(
            self.source,
            storage_options=target_storage_options,
            skip_rows_after_header=1,
        )
        return target_data

    def subset_data_to_case(
        self,
        target_data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Get the season (year) from the case start date, cast as string as polars is interpreting the schema as strings
        year = case_operator.case.start_date.year
        season = str(year)

        # Create a subquery to find all storm numbers in the same season
        matching_numbers = (
            target_data.filter(pl.col("SEASON") == season).select("NUMBER").unique()
        )

        # Apply the filter to get all data for storms with the same number in the same season
        # This maintains the lazy evaluation
        subset_target_data = target_data.join(
            matching_numbers, on="NUMBER", how="inner"
        ).filter(
            (pl.col("NAME") == case_operator.case.title.upper())
            & (pl.col("SEASON") == season)
        )

        all_variables = IBTrACS_metadata_variable_mapping.values()
        # check that the variables are in the target data
        schema_fields = [field for field in subset_target_data.collect_schema()]
        target_variables = [
            v for v in case_operator.target.variables if isinstance(v, str)
        ]
        if target_variables and any(var not in schema_fields for var in all_variables):
            raise ValueError(f"Variables {all_variables} not found in target data")

        # subset the variables
        subset_target_data = subset_target_data.select(all_variables)

        return subset_target_data

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
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


def open_kerchunk_reference(
    forecast_dir: str,
    storage_options: dict = {"remote_protocol": "s3", "remote_options": {"anon": True}},
    chunks: Union[dict, str] = "auto",
) -> xr.Dataset:
    """Open a dataset from a kerchunked reference file in parquet or json format.
    This has been built primarily for the CIRA MLWP S3 bucket's data (https://registry.opendata.aws/aiwp/),
    but can work with other data in the future. Currently only supports CIRA data unless
    schema is identical to the CIRA schema.

    Args:
        file: The path to the kerchunked reference file.
        remote_protocol: The remote protocol to use.

    Returns:
        The opened dataset.
    """
    if forecast_dir.endswith(".parq") or forecast_dir.endswith(".parquet"):
        kerchunk_ds = xr.open_dataset(
            forecast_dir,
            engine="kerchunk",
            storage_options=storage_options,
            chunks=chunks,
        )
    elif forecast_dir.endswith(".json"):
        storage_options["fo"] = forecast_dir
        kerchunk_ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "storage_options": storage_options,
                "consolidated": False,
            },
            chunks=chunks,
        )
    else:
        raise TypeError(
            "Unknown kerchunk file type found in forecast path, only json and parquet are supported."
        )
    return kerchunk_ds


def open_lazy_target_data(target_base: "TargetBase") -> utils.IncomingDataInput:
    """Open the target data from the target URI."""
    return target_base._open_data_from_source()


def zarr_target_subsetter(
    data: xr.Dataset,
    case_operator: "case.CaseOperator",
    time_variable: str = "valid_time",
) -> xr.Dataset:
    """Subset a zarr dataset to a case operator."""
    # subset time first to avoid OOM masking issues
    subset_time_data = data.sel(
        {
            time_variable: slice(
                case_operator.case.start_date, case_operator.case.end_date
            )
        }
    )

    # check that the variables are in the target data
    target_variables = [v for v in case_operator.target.variables if isinstance(v, str)]
    if target_variables and any(
        var not in subset_time_data.data_vars for var in target_variables
    ):
        raise ValueError(f"Variables {target_variables} not found in target data")
    # subset the variables if they exist in the target data
    elif target_variables:
        subset_time_variable_data = subset_time_data[target_variables]
    else:
        raise ValueError(
            "Variables not defined. Please list at least one target variable to select."
        )
    # mask the data to the case location
    fully_subset_data = case_operator.case.location.mask(
        subset_time_variable_data, drop=True
    )

    return fully_subset_data
