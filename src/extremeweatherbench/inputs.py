from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import derived, utils

if TYPE_CHECKING:
    from extremeweatherbench import case

#: Storage/access options for gridded target datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point target dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"

IBTRACS_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"  # noqa: E501


class InputBase(ABC):
    """
    An abstract base class for target and forecast data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict = {},
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
    ):
        self.source = source
        self.variables = variables
        self.variable_mapping = variable_mapping
        self.storage_options = storage_options
        self.preprocess = preprocess

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
        """
        Open the target data from the source, opting to avoid loading the entire dataset into memory if possible.

        Returns:
            The target data with a type determined by the user.
        """

    @abstractmethod
    def subset_data_to_case(
        self,
        data: utils.IncomingDataInput,
        case: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        """
        Subset the target data to the case information provided in CaseOperator.

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
        """
        Convert the target data to an xarray dataset if it is not already.

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
        """
        Hook method for custom conversion logic. Override this method in subclasses
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


class ForecastBase(InputBase):
    """A base class defining the interface for ExtremeWeatherBench forecast data.

    A Forecast is data that acts as the "forecast" for a case.

    Attributes:
        forecast_source: The source of the forecast data, which can be a local path or a remote URL/URI.
    """

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
            subset_time_data = subset_time_data[case_operator.forecast_config.variables]
        except KeyError:
            raise KeyError(
                f"Variables {case_operator.forecast_config.variables} not found in forecast data"
            )
        fully_subset_data = case_operator.case.location.mask(
            subset_time_data, drop=True
        )
        return fully_subset_data


class KerchunkForecast(ForecastBase):
    """
    Forecast class for kerchunked forecast data.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict[str, str],
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
    ):
        super().__init__(
            source, variables, variable_mapping, storage_options, preprocess
        )
        self.chunks = chunks

    def _open_data_from_source(self) -> utils.IncomingDataInput:
        return open_kerchunk_reference(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
        )


class ZarrForecast(ForecastBase):
    """
    Forecast class for zarr forecast data.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict[str, str],
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
        chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
    ):
        super().__init__(
            source, variables, variable_mapping, storage_options, preprocess
        )
        self.chunks = chunks

    def _open_data_from_source(self) -> utils.IncomingDataInput:
        return xr.open_zarr(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
        )


class TargetBase(InputBase):
    """
    An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """


class ERA5(TargetBase):
    """
    Target class for ERA5 gridded data.

    The easiest approach to using this class
    is to use the ARCO ERA5 dataset provided by Google for a source. Otherwise, either a
    different zarr source or modifying the open_data_from_source method to open the data
    using another method is required.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict[str, str],
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
    ):
        super().__init__(
            source, variables, variable_mapping, storage_options, preprocess
        )

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
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise ValueError(f"Expected xarray Dataset or DataArray, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_data = data.sel(
            valid_time=slice(case_operator.case.start_date, case_operator.case.end_date)
        )

        # check that the variables are in the target data
        target_variables = [
            v for v in case_operator.target_config.variables if isinstance(v, str)
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

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        return data


class GHCN(TargetBase):
    """
    Target class for GHCN tabular data.

    Data is processed using polars to maintain the lazy loading
    paradigm in open_data_from_source and to separate the subsetting
    into subset_data_to_case.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict[str, str],
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
    ):
        super().__init__(
            source, variables, variable_mapping, storage_options, preprocess
        )

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
            v for v in case_operator.target_variables if isinstance(v, str)
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


class LSR(TargetBase):
    """
    Target class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be
    12 UTC to the next day at 12 UTC to match SPC methods.
    """

    def __init__(
        self,
        source: str,
        variables: list[Union[str, "derived.DerivedVariable"]],
        variable_mapping: dict[str, str],
        storage_options: Optional[dict] = None,
        preprocess: Callable = utils._default_preprocess,
    ):
        super().__init__(
            source, variables, variable_mapping, storage_options, preprocess
        )

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
        target_data["lat"] = target_data["lat"].astype(float)
        target_data["lon"] = target_data["lon"].astype(float)
        target_data["time"] = pd.to_datetime(target_data["time"])

        filters = (
            (target_data["time"] >= case_operator.case.start_date)
            & (target_data["time"] <= case_operator.case.end_date)
            & (target_data["lat"] >= case_operator.case.location.latitude_min)
            & (target_data["lat"] <= case_operator.case.location.latitude_max)
            & (
                target_data["lon"]
                >= utils.convert_longitude_to_180(
                    case_operator.case.location.longitude_min
                )
            )
            & (
                target_data["lon"]
                <= utils.convert_longitude_to_180(
                    case_operator.case.location.longitude_max
                )
            )
        )

        subset_target_data = target_data.loc[filters]

        subset_target_data = subset_target_data.rename(
            columns={"lat": "latitude", "lon": "longitude", "time": "valid_time"}
        )

        return subset_target_data

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pd.DataFrame):
            data = data.set_index(["valid_time", "latitude", "longitude"])
            data = xr.Dataset.from_dataframe(
                data[~data.index.duplicated(keep="first")], sparse=True
            )
            return data
        else:
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")


class IBTrACS(TargetBase):
    """
    Target class for IBTrACS data.
    """

    def _open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> utils.IncomingDataInput:
        # not using storage_options in this case due to NetCDF4Backend not supporting them
        target_data: pl.LazyFrame = pl.scan_csv(
            self.source, storage_options=target_storage_options
        )
        return target_data

    def subset_data_to_case(
        self,
        target_data: utils.IncomingDataInput,
        case_operator: "case.CaseOperator",
    ) -> utils.IncomingDataInput:
        # Create filter expressions for LazyFrame
        year = case_operator.case.start_date.year

        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Apply filters using proper polars expressions
        subset_target_data = target_data.filter(
            (pl.col("NAME") == case_operator.case.title.upper())
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
            (pl.col("NAME") == case_operator.case.title.upper())
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
            (pl.col("NAME") == case_operator.case.title.upper())
            & (pl.col("SEASON") == season)
        )

        # check that the variables are in the target data
        schema_fields = [field for field in subset_target_data.collect_schema()]
        target_variables = [
            v for v in case_operator.target_variables if isinstance(v, str)
        ]
        if target_variables and any(var not in schema_fields for var in all_variables):
            raise ValueError(f"Variables {all_variables} not found in target data")

        # subset the variables
        if target_variables:
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
    if "parq" in forecast_dir or "parquet" in forecast_dir:
        kerchunk_ds = xr.open_dataset(
            forecast_dir,
            engine="kerchunk",
            storage_options=storage_options,
            chunks=chunks,
        )
    elif "json" in forecast_dir:
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
