import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import cases, derived, utils

if TYPE_CHECKING:
    from extremeweatherbench import metrics

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
    "WMO_WIND": "wmo_surface_wind_speed",
    "WMO_PRES": "wmo_air_pressure_at_mean_sea_level",
    "USA_WIND": "usa_surface_wind_speed",
    "USA_PRES": "usa_air_pressure_at_mean_sea_level",
    "NEUMANN_WIND": "neumann_surface_wind_speed",
    "NEUMANN_PRES": "neumann_air_pressure_at_mean_sea_level",
    "TOKYO_WIND": "tokyo_surface_wind_speed",
    "TOKYO_PRES": "tokyo_air_pressure_at_mean_sea_level",
    "CMA_WIND": "cma_surface_wind_speed",
    "CMA_PRES": "cma_air_pressure_at_mean_sea_level",
    "HKO_WIND": "hko_surface_wind_speed",
    "KMA_WIND": "kma_surface_wind_speed",
    "KMA_PRES": "kma_air_pressure_at_mean_sea_level",
    "NEWDELHI_WIND": "newdelhi_surface_wind_speed",
    "NEWDELHI_PRES": "newdelhi_air_pressure_at_mean_sea_level",
    "REUNION_WIND": "reunion_surface_wind_speed",
    "REUNION_PRES": "reunion_air_pressure_at_mean_sea_level",
    "BOM_WIND": "bom_surface_wind_speed",
    "BOM_PRES": "bom_air_pressure_at_mean_sea_level",
    "NADI_WIND": "nadi_surface_wind_speed",
    "NADI_PRES": "nadi_air_pressure_at_mean_sea_level",
    "WELLINGTON_WIND": "wellington_surface_wind_speed",
    "WELLINGTON_PRES": "wellington_air_pressure_at_mean_sea_level",
    "DS824_WIND": "ds824_surface_wind_speed",
    "DS824_PRES": "ds824_air_pressure_at_mean_sea_level",
    "MLC_WIND": "mlc_surface_wind_speed",
    "MLC_PRES": "mlc_air_pressure_at_mean_sea_level",
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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        """Subset the target data to the case information provided in CaseOperator.

        Time information, spatial bounds, and variables are captured in the case metadata
        where this method is used to subset.

        Args:
            data: The target data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
            pandas dataframe, or numpy array.
            case_operator: The case operator to subset the data to; includes time information, spatial bounds, and
            variables.

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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_indices = utils.derive_indices_from_init_time_and_lead_time(
            data,
            case_operator.case_metadata.start_date,
            case_operator.case_metadata.end_date,
        )

        subset_time_data = data.sel(
            init_time=data.init_time[np.unique(subset_time_indices[0])]
        )
        subset_time_data = utils.convert_init_time_to_valid_time(subset_time_data)

        # use the list of required variables from the derived variables in the eval to add to the list of variables
        expected_and_maybe_derived_variables = (
            derived.maybe_pull_required_variables_from_derived_input(
                case_operator.forecast.variables
            )
        )
        try:
            subset_time_data = subset_time_data[expected_and_maybe_derived_variables]
        except KeyError:
            raise KeyError(
                f"One of the variables {expected_and_maybe_derived_variables} not found in forecast data"
            )
        fully_subset_data = case_operator.case_metadata.location.mask(
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
            decode_timedelta=True,
        )


@dataclasses.dataclass
class TargetBase(InputBase):
    """An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Align the forecast data to the target data.

        This method is used to align the forecast data to the target data (not vice versa).
        Implementation is useful for non-gridded targets that need to be aligned to the forecast
        data.

        Args:
            forecast_data: The forecast data to align.
            target_data: The target data to align to.

        Returns:
            A tuple of the aligned target data and forecast data. Defaults to passing through
            the target and forecast data.
        """
        return target_data, forecast_data


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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        return zarr_target_subsetter(data, case_operator)

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Align forecast data to ERA5 target data.

        This method handles alignment between forecast data and ERA5 target data by:
        1. Aligning time dimensions (handles valid_time vs time naming)
        2. Handling spatial alignment (regridding if needed)

        Args:
            forecast_data: Forecast dataset with valid_time, latitude, longitude
            target_data: ERA5 target dataset with time, latitude, longitude

        Returns:
            Tuple of (aligned_target_data, aligned_forecast_data)
        """
        # Handle time dimension naming differences
        target_time_dim = "time" if "time" in target_data.dims else "valid_time"
        forecast_time_dim = (
            "valid_time" if "valid_time" in forecast_data.dims else "time"
        )

        # Rename target time dimension to match forecast if needed
        aligned_target = target_data.copy()
        if target_time_dim != forecast_time_dim:
            aligned_target = aligned_target.rename({target_time_dim: forecast_time_dim})

        # Align time dimensions - find overlapping times
        time_aligned_target, time_aligned_forecast = xr.align(
            aligned_target,
            forecast_data,
            join="inner",
            exclude=["latitude", "longitude"],
        )

        # Spatial alignment - check if regridding is needed
        target_lats = time_aligned_target.latitude.values
        target_lons = time_aligned_target.longitude.values
        forecast_lats = time_aligned_forecast.latitude.values
        forecast_lons = time_aligned_forecast.longitude.values

        # Check if spatial grids are identical (within tolerance)
        lats_match = len(target_lats) == len(forecast_lats) and np.allclose(
            target_lats, forecast_lats, rtol=1e-5
        )
        lons_match = len(target_lons) == len(forecast_lons) and np.allclose(
            target_lons, forecast_lons, rtol=1e-5
        )

        if not (lats_match and lons_match):
            # Regrid forecast to target grid using nearest neighbor interpolation
            time_aligned_forecast = time_aligned_forecast.interp(
                latitude=target_lats, longitude=target_lons, method="nearest"
            )

        return time_aligned_target, time_aligned_forecast


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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        # Create filter expressions for LazyFrame
        time_min = case_operator.case_metadata.start_date - pd.Timedelta(days=2)
        time_max = case_operator.case_metadata.end_date + pd.Timedelta(days=2)

        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Apply filters using proper polars expressions
        subset_target_data = target_data.filter(
            (pl.col("valid_time") >= time_min)
            & (pl.col("valid_time") <= time_max)
            & (
                pl.col("latitude")
                >= case_operator.case_metadata.location.geopandas.total_bounds[1]
            )
            & (
                pl.col("latitude")
                <= case_operator.case_metadata.location.geopandas.total_bounds[3]
            )
            & (
                pl.col("longitude")
                >= case_operator.case_metadata.location.geopandas.total_bounds[0]
            )
            & (
                pl.col("longitude")
                <= case_operator.case_metadata.location.geopandas.total_bounds[2]
            )
        )
        # convert to Kelvin
        subset_target_data = subset_target_data.with_columns(
            pl.col("surface_air_temperature").add(273.15)
        )
        # Add time, latitude, and longitude to the variables, polars doesn't do indexes
        target_variables = [
            v for v in case_operator.target.variables if isinstance(v, str)
        ]
        if target_variables is None:
            all_variables = ["valid_time", "latitude", "longitude"]
        else:
            all_variables = target_variables + ["valid_time", "latitude", "longitude"]

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
            data = data.set_index(["valid_time", "latitude", "longitude"])
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

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        return align_point_obs_target_to_forecast(target_data, forecast_data)


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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(target_data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(target_data)}")

        # latitude, longitude are strings by default, convert to float
        target_data["latitude"] = target_data["latitude"].astype(float)
        target_data["longitude"] = target_data["longitude"].astype(float)
        target_data["valid_time"] = pd.to_datetime(target_data["valid_time"])

        # filters to apply to the target data including datetimes and location bounds
        filters = (
            (target_data["valid_time"] >= case_operator.case_metadata.start_date)
            & (target_data["valid_time"] <= case_operator.case_metadata.end_date)
            & (
                target_data["latitude"]
                >= case_operator.case_metadata.location.latitude_min
            )
            & (
                target_data["latitude"]
                <= case_operator.case_metadata.location.latitude_max
            )
            & (
                target_data["longitude"]
                >= utils.convert_longitude_to_180(
                    case_operator.case_metadata.location.longitude_min
                )
            )
            & (
                target_data["longitude"]
                <= utils.convert_longitude_to_180(
                    case_operator.case_metadata.location.longitude_max
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

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        return align_point_obs_target_to_forecast(target_data, forecast_data)


# TODO: get PPH connector working properly
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
        case_operator: "cases.CaseOperator",
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
        case_operator: "cases.CaseOperator",
    ) -> utils.IncomingDataInput:
        if not isinstance(target_data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(target_data)}")

        # Get the season (year) from the case start date, cast as string as polars is interpreting the schema as strings
        season = case_operator.case_metadata.start_date.year
        if case_operator.case_metadata.start_date.month > 11:
            season += 1

        # Create a subquery to find all storm numbers in the same season
        matching_numbers = (
            target_data.filter(pl.col("SEASON").cast(pl.Int64) == season)
            .select("NUMBER")
            .unique()
        )

        # Apply the filter to get all data for storms with the same number in the same season
        # This maintains the lazy evaluation
        subset_target_data = target_data.join(
            matching_numbers, on="NUMBER", how="inner"
        ).filter(
            (pl.col("tc_name") == case_operator.case_metadata.title.upper())
            & (pl.col("SEASON").cast(pl.Int64) == season)
        )

        all_variables = IBTrACS_metadata_variable_mapping.values()
        # subset the variables
        subset_target_data = subset_target_data.select(all_variables)

        schema = subset_target_data.collect_schema()
        # Convert pressure and surface wind columns to float, replacing " " with null
        # Get column names that contain "pressure" or "wind"
        pressure_cols = [col for col in schema if "pressure" in col.lower()]
        wind_cols = [col for col in schema if "wind" in col.lower()]

        # Apply transformations to convert " " to null and cast to float
        subset_target_data = subset_target_data.with_columns(
            [
                pl.when(pl.col(col) == " ")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Float64, strict=False)
                .alias(col)
                for col in pressure_cols + wind_cols
            ]
        )

        # Drop rows where ALL columns are null (equivalent to pandas dropna(how="all"))
        subset_target_data = subset_target_data.filter(
            ~pl.all_horizontal(pl.all().is_null())
        )

        # Create unified pressure and wind columns by preferring USA and WMO data
        # For surface wind speed
        wind_columns = [col for col in schema if "surface_wind_speed" in col]
        wind_priority = ["usa_surface_wind_speed", "wmo_surface_wind_speed"] + [
            col
            for col in wind_columns
            if col not in ["usa_surface_wind_speed", "wmo_surface_wind_speed"]
        ]

        # For pressure at mean sea level
        pressure_columns = [
            col for col in schema if "air_pressure_at_mean_sea_level" in col
        ]
        pressure_priority = [
            "usa_air_pressure_at_mean_sea_level",
            "wmo_air_pressure_at_mean_sea_level",
        ] + [
            col
            for col in pressure_columns
            if col
            not in [
                "usa_air_pressure_at_mean_sea_level",
                "wmo_air_pressure_at_mean_sea_level",
            ]
        ]

        # Create unified columns using coalesce (equivalent to pandas bfill)
        subset_target_data = subset_target_data.with_columns(
            [
                pl.coalesce(wind_priority).alias("surface_wind_speed"),
                pl.coalesce(pressure_priority).alias("air_pressure_at_mean_sea_level"),
            ]
        )

        # Select only the columns to keep
        columns_to_keep = [
            "valid_time",
            "tc_name",
            "latitude",
            "longitude",
            "surface_wind_speed",
            "air_pressure_at_mean_sea_level",
        ]

        subset_target_data = subset_target_data.select(columns_to_keep)

        # Drop rows where wind speed OR pressure are null (equivalent to pandas dropna with how="any")
        subset_target_data = subset_target_data.filter(
            pl.col("surface_wind_speed").is_not_null()
            & pl.col("air_pressure_at_mean_sea_level").is_not_null()
        )

        return subset_target_data

    def _custom_convert_to_dataset(self, data: utils.IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_pandas()

            # IBTrACS data is in -180 to 180, convert to 0 to 360
            data["longitude"] = utils.convert_longitude_to_360(data["longitude"])

            # Due to missing data in the IBTrACS dataset, polars doesn't convert the valid_time to a datetime by default
            data["valid_time"] = pd.to_datetime(data["valid_time"])
            data = data.set_index(["valid_time", "latitude", "longitude"])

            try:
                data = xr.Dataset.from_dataframe(data, sparse=True)
            except ValueError as e:
                if "non-unique" in str(e):
                    pass
                data = xr.Dataset.from_dataframe(data.drop_duplicates(), sparse=True)
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


def zarr_target_subsetter(
    data: xr.Dataset,
    case_operator: "cases.CaseOperator",
    time_variable: str = "valid_time",
) -> xr.Dataset:
    """Subset a zarr dataset to a case operator."""
    # subset time first to avoid OOM masking issues
    subset_time_data = data.sel(
        {
            time_variable: slice(
                case_operator.case_metadata.start_date,
                case_operator.case_metadata.end_date,
            )
        }
    )

    target_and_maybe_derived_variables = (
        derived.maybe_pull_required_variables_from_derived_input(
            case_operator.target.variables
        )
    )
    # check that the variables are in the target data
    if target_and_maybe_derived_variables and any(
        var not in subset_time_data.data_vars
        for var in target_and_maybe_derived_variables
    ):
        raise ValueError(
            f"Variables {target_and_maybe_derived_variables} not found in target data"
        )
    # subset the variables if they exist in the target data
    elif target_and_maybe_derived_variables:
        subset_time_variable_data = subset_time_data[target_and_maybe_derived_variables]
    else:
        raise ValueError(
            "Variables not defined. Please list at least one target variable to select."
        )
    # mask the data to the case location
    fully_subset_data = case_operator.case_metadata.location.mask(
        subset_time_variable_data, drop=True
    )

    return fully_subset_data


def align_point_obs_target_to_forecast(
    target_data: xr.Dataset,
    forecast_data: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    lons = xr.DataArray(target_data["longitude"].values, dims="location")
    lats = xr.DataArray(target_data["latitude"].values, dims="location")

    time_aligned_target_data, time_aligned_forecast_data = xr.align(
        target_data, forecast_data, exclude=["latitude", "longitude"]
    )

    time_aligned_forecast_data = time_aligned_forecast_data.interp(
        latitude=lats, longitude=lons, method="nearest"
    )
    time_aligned_forecast_data = time_aligned_forecast_data.set_index(
        location=("latitude", "longitude")
    ).unstack("location")
    return time_aligned_target_data, time_aligned_forecast_data
