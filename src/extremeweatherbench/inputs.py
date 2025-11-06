import abc
import dataclasses
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    TypeAlias,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import cases, derived, sources, utils

if TYPE_CHECKING:
    from extremeweatherbench import metrics

logger = logging.getLogger(__name__)

#: Storage/access options for gridded target datasets.
ARCO_ERA5_FULL_URI = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

#: Storage/access options for default point target dataset.
DEFAULT_GHCN_URI = "gs://extremeweatherbench/datasets/ghcnh_all_2020_2024.parq"

#: Storage/access options for local storm report (LSR) tabular data.
LSR_URI = "gs://extremeweatherbench/datasets/combined_canada_australia_us_lsr_01012020_09272025.parq"  # noqa: E501

PPH_URI = (
    "gs://extremeweatherbench/datasets/"
    "practically_perfect_hindcast_20200104_20250927.zarr"
)

IBTRACS_URI = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)


# ERA5 metadata variable mapping
ERA5_metadata_variable_mapping = {
    "time": "valid_time",
    "2m_temperature": "surface_air_temperature",
    "2m_dewpoint_temperature": "surface_dewpoint_temperature",
    "temperature": "air_temperature",
    "dewpoint": "dewpoint_temperature",
    "2m_relative_humidity": "surface_relative_humidity",
    "2m_specific_humidity": "surface_specific_humidity",
    "10m_u_component_of_wind": "surface_eastward_wind",
    "10m_v_component_of_wind": "surface_northward_wind",
    "u_component_of_wind": "eastward_wind",
    "v_component_of_wind": "northward_wind",
    "specific_humidity": "specific_humidity",
    "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
}

# CIRA MLWP forecasts metadata variable mapping
CIRA_metadata_variable_mapping = {
    "time": "valid_time",
    "t2": "surface_air_temperature",
    "t": "air_temperature",
    "q": "specific_humidity",
    "u": "eastward_wind",
    "v": "northward_wind",
    "p": "air_pressure",
    "z": "geopotential_height",
    "r": "relative_humidity",
    "u10": "surface_eastward_wind",
    "v10": "surface_northward_wind",
    "u100": "100m_eastward_wind",
    "v100": "100m_northward_wind",
}

# HRES forecast (weatherbench2)metadata variable mapping
HRES_metadata_variable_mapping = {
    "2m_temperature": "surface_air_temperature",
    "2m_dewpoint_temperature": "surface_dewpoint_temperature",
    "temperature": "air_temperature",
    "dewpoint": "dewpoint_temperature",
    "10m_u_component_of_wind": "surface_eastward_wind",
    "10m_v_component_of_wind": "surface_northward_wind",
    "u_component_of_wind": "eastward_wind",
    "v_component_of_wind": "northward_wind",
    "prediction_timedelta": "lead_time",
    "time": "init_time",
    "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
    "10m_wind_speed": "surface_wind_speed",
}

IBTrACS_metadata_variable_mapping = {
    "SID": "storm_id",
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

IncomingDataInput: TypeAlias = xr.Dataset | xr.DataArray | pl.LazyFrame | pd.DataFrame


def _default_preprocess(input_data: IncomingDataInput) -> IncomingDataInput:
    """Default forecast preprocess function that does nothing."""
    return input_data


@dataclasses.dataclass
class InputBase(abc.ABC):
    """An abstract base dataclass for target and forecast data.

    Attributes:
        source: The source of the data, which can be a local path or a remote URL/URI.
        name: The name of the input data source.
        variables: A list of variables to select from the data.
        variable_mapping: A dictionary of variable names to map to the data.
        storage_options: Storage/access options for the data.
        preprocess: A function to preprocess the data.
    """

    source: str
    name: str
    variables: list[Union[str, "derived.DerivedVariable"]] = dataclasses.field(
        default_factory=list
    )
    variable_mapping: dict = dataclasses.field(default_factory=dict)
    storage_options: Optional[dict] = None
    preprocess: Callable = _default_preprocess

    def open_and_maybe_preprocess_data_from_source(
        self,
    ) -> IncomingDataInput:
        data = self._open_data_from_source()
        data = self.preprocess(data)
        return data

    def set_name(self, name: str) -> None:
        """Set the name of the input data source.

        Args:
            name: The new name to assign to this input data source.
        """
        self.name = name

    @abc.abstractmethod
    def _open_data_from_source(self) -> IncomingDataInput:
        """Open the input data from the source, opting to avoid loading the entire
        dataset into memory if possible.

        Returns:
            The input data with a type determined by the user.
        """

    @abc.abstractmethod
    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        """Subset the input data to the case information provided in IndividualCase.

        Time information, spatial bounds, and variables are captured in the case
        metadata
        where this method is used to subset.

        Args:
            data: The input data to subset, which should be a xarray dataset,
                xarray dataarray, polars lazyframe, pandas dataframe, or numpy
                array.
            case_metadata: The case metadata to subset the data to; includes time
                information, spatial bounds, and variables.

        Returns:
            The input data with the variables subset to the case metadata.
        """

    def maybe_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """Convert the input data to an xarray dataset if it is not already.

        This method handles the common conversion cases automatically. Override
        this method
        only if you need custom conversion logic beyond the standard cases.

        Args:
            data: The input data to convert.

        Returns:
            The input data as an xarray dataset.
        """
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            return data.to_dataset()
        else:
            # For other data types, try to use a custom conversion method if available
            return self._custom_convert_to_dataset(data)

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        """Hook method for custom conversion logic. Override this method in subclasses
        if you need custom conversion behavior for non-xarray data types.

        By default, this raises a NotImplementedError to encourage explicit handling
        of custom data types.

        Args:
            data: The input data to convert.

        Returns:
            The input data as an xarray dataset.
        """
        raise NotImplementedError(
            f"Conversion from {type(data)} to xarray.Dataset not implemented. "
            f"Override _custom_convert_to_dataset in your InputBase subclass."
        )

    def add_source_to_dataset_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Add the name of the source to the dataset attributes."""
        ds.attrs["source"] = self.name
        return ds

    def maybe_map_variable_names(self, data: IncomingDataInput) -> IncomingDataInput:
        """Map the variable names to the data, if required.

        Args:
            data: The incoming data in the form of an object that has a rename
                method for data variables/columns.

        Returns:
            A dataset with mapped variable names, if any exist, else the original data.
        """
        # Some inputs may not have variables defined, in which case we return
        # the data unmodified
        if not self.variables and not self.variable_mapping:
            return data

        variable_mapping = self.variable_mapping

        if isinstance(data, xr.DataArray):
            return data.rename(variable_mapping[data.name])
        elif isinstance(data, xr.Dataset):
            old_name_obj = list(data.variables.keys())
        elif isinstance(data, pl.LazyFrame):
            old_name_obj = list(data.collect_schema().names())
        elif isinstance(data, pd.DataFrame):
            old_name_obj = list(data.columns)
        else:
            raise ValueError(f"Data type {type(data)} not supported")

        if not variable_mapping:
            return data

        output_dict = {
            old_name: new_name
            for old_name, new_name in variable_mapping.items()
            if old_name in old_name_obj
        }

        return (
            data.rename(output_dict)
            if not isinstance(data, pd.DataFrame)
            else data.rename(columns=output_dict)
        )


@dataclasses.dataclass
class ForecastBase(InputBase):
    """A class defining the interface for ExtremeWeatherBench forecast data."""

    chunks: Optional[Union[dict, str]] = "auto"

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        drop = kwargs.get("drop", False)
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")
        # Drop duplicate init_time values
        if len(np.unique(data.init_time)) != len(data.init_time):
            _, index = np.unique(data.init_time, return_index=True)
            data = data.isel(init_time=index)

        # subset time first to avoid OOM masking issues
        subset_time_indices = utils.derive_indices_from_init_time_and_lead_time(
            data,
            case_metadata.start_date,
            case_metadata.end_date,
        )

        # If there are no valid times, return an empty dataset
        if len(subset_time_indices[0]) == 0:
            return xr.Dataset(coords={"valid_time": []})

        # Use only valid init_time indices, but keep all lead_times
        unique_init_indices = np.unique(subset_time_indices[0])
        subset_time_data = data.sel(init_time=data.init_time[unique_init_indices])

        # Create a mask indicating which (init_time, lead_time) combinations
        # result in valid_times within the case date range
        valid_combinations_mask = np.zeros(
            (len(subset_time_data.init_time), len(subset_time_data.lead_time)),
            dtype=bool,
        )

        # Map the valid indices back to the subset data coordinates
        for i, j in zip(subset_time_indices[0], subset_time_indices[1]):
            # Find the position of this init_time in the subset data
            init_pos = np.where(unique_init_indices == i)[0]
            if len(init_pos) > 0:
                valid_combinations_mask[init_pos[0], j] = True

        # Add the mask as a coordinate so downstream code can use it
        subset_time_data = subset_time_data.assign_coords(
            valid_time_mask=(["init_time", "lead_time"], valid_combinations_mask)
        )

        spatiotemporally_subset_data = case_metadata.location.mask(
            subset_time_data, drop=drop
        )

        # convert from init_time/lead_time to init_time/valid_time
        spatiotemporally_subset_data = utils.convert_init_time_to_valid_time(
            spatiotemporally_subset_data
        )

        # Now filter to only include valid_times within the case date range
        # This eliminates the actual time steps that fall outside the range
        time_filtered_data = spatiotemporally_subset_data.sel(
            valid_time=slice(case_metadata.start_date, case_metadata.end_date)
        )

        return time_filtered_data


@dataclasses.dataclass
class EvaluationObject:
    """A class to store the evaluation object for a forecast and target pairing.

    A EvaluationObject is an evaluation object which contains a forecast, target,
    and metrics to evaluate. The evaluation is a set of all metrics, target variables,
    and forecast variables.

    Multiple EvaluationObjects can be used to evaluate a single event type.
    This is useful for
    evaluating distinct targets or metrics with unique variables to evaluate.

    Attributes:
        event_type: The event type to evaluate.
        metric_list: A list of BaseMetric objects.
        target: A TargetBase object.
        forecast: A ForecastBase object.
    """

    event_type: str
    metric_list: list[
        Union[
            Callable[..., Any],
            "metrics.BaseMetric",
        ]
    ]
    target: "TargetBase"
    forecast: "ForecastBase"


@dataclasses.dataclass
class KerchunkForecast(ForecastBase):
    """Forecast class for kerchunked forecast data."""

    chunks: Optional[Union[dict, str]] = "auto"
    storage_options: dict = dataclasses.field(default_factory=dict)

    def _open_data_from_source(self) -> IncomingDataInput:
        return open_kerchunk_reference(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks or "auto",
        )


@dataclasses.dataclass
class ZarrForecast(ForecastBase):
    """Forecast class for zarr forecast data."""

    chunks: Optional[Union[dict, str]] = "auto"

    def _open_data_from_source(self) -> IncomingDataInput:
        return xr.open_zarr(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
            decode_timedelta=True,
        )


@dataclasses.dataclass
class TargetBase(InputBase):
    """An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded
    dataset, a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be
    in the same coordinate system for evaluation.
    """

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Align the forecast data to the target data.

        This method is used to align the forecast data to the target data (not
        vice versa). Implementation is key for non-gridded targets that have dims
        unlike the forecast data.

        Args:
            forecast_data: The forecast data to align.
            target_data: The target data to align to.

        Returns:
            A tuple of the aligned forecast data and target data. Defaults to
            passing through
            the forecast and target data.
        """
        return forecast_data, target_data


@dataclasses.dataclass
class ERA5(TargetBase):
    """Target class for ERA5 gridded data, ideally using the ARCO ERA5 dataset provided
    by Google. Otherwise, either a different zarr source for ERA5.
    """

    name: str = "ERA5"
    chunks: Optional[Union[dict, str]] = None
    source: str = ARCO_ERA5_FULL_URI
    variable_mapping: dict = dataclasses.field(
        default_factory=lambda: ERA5_metadata_variable_mapping.copy()
    )

    def _open_data_from_source(self) -> IncomingDataInput:
        data = xr.open_zarr(
            self.source,
            storage_options=self.storage_options,
            chunks=self.chunks,
        )
        return data

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        drop = kwargs.get("drop", False)
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")
        return zarr_target_subsetter(data, case_metadata, drop=drop)

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
            Tuple of (aligned_forecast_data, aligned_target_data)
        """
        aligned_forecast_data, aligned_target_data = align_forecast_to_target(
            forecast_data, target_data
        )
        return aligned_forecast_data, aligned_target_data


@dataclasses.dataclass
class GHCN(TargetBase):
    """Target class for GHCN tabular data.

    Data is processed using polars to maintain the lazy loading paradigm in
    open_data_from_source and to separate the subsetting into subset_data_to_case.
    """

    name: str = "GHCN"
    source: str = DEFAULT_GHCN_URI

    def _open_data_from_source(self) -> IncomingDataInput:
        target_data: pl.LazyFrame = pl.scan_parquet(
            self.source, storage_options=self.storage_options
        )

        return target_data

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        if not isinstance(data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(data)}")

        # Create filter expressions for LazyFrame
        time_min = case_metadata.start_date - pd.Timedelta(days=2)
        time_max = case_metadata.end_date + pd.Timedelta(days=2)
        case_location = case_metadata.location.as_geopandas()
        # Apply filters using proper polars expressions
        subset_target_data = data.filter(
            (pl.col("valid_time") >= time_min)
            & (pl.col("valid_time") <= time_max)
            & (pl.col("latitude") >= case_location.total_bounds[1])
            & (pl.col("latitude") <= case_location.total_bounds[3])
            & (pl.col("longitude") >= case_location.total_bounds[0])
            & (pl.col("longitude") <= case_location.total_bounds[2])
        ).sort("valid_time")
        return subset_target_data

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pl.LazyFrame):
            # convert to Kelvin, GHCN data is in Celsius by default
            if "surface_air_temperature" in data.collect_schema().names():
                data = data.with_columns(pl.col("surface_air_temperature").add(273.15))
            data = data.collect(engine="streaming").to_pandas()
            data["longitude"] = utils.convert_longitude_to_360(data["longitude"])

            data = data.set_index(["valid_time", "latitude", "longitude"])
            # GHCN data can have duplicate values right now, dropping here if it occurs
            try:
                data = xr.Dataset.from_dataframe(
                    data[~data.index.duplicated(keep="first")], sparse=True
                )
            except Exception as e:
                logger.warning(
                    "Error converting GHCN data to xarray: %s, returning empty Dataset",
                    e,
                )
                return xr.Dataset()
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        return align_forecast_to_target(forecast_data, target_data)


@dataclasses.dataclass
class LSR(TargetBase):
    """Target class for local storm report (LSR) tabular data.

    run_pipeline() returns a dataset with LSRs and practically perfect hindcast gridded
    probability data. IndividualCase date ranges for LSRs should ideally be 12 UTC to
    the next day at 12 UTC to match SPC methods for US data. Australia data should be 00
    UTC to 00 UTC.
    """

    name: str = "local_storm_reports"
    source: str = LSR_URI

    def _open_data_from_source(self) -> IncomingDataInput:
        # force LSR to use anon token to prevent google reauth issues for users
        target_data = pd.read_parquet(self.source, storage_options=self.storage_options)

        return target_data

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(data)}")

        data = data.copy()

        # latitude, longitude are strings by default, convert to float
        data["latitude"] = data["latitude"].astype(float)
        data["longitude"] = data["longitude"].astype(float)
        data["valid_time"] = pd.to_datetime(data["valid_time"])

        # filters to apply to the target data including datetimes and location bounds
        bounds = case_metadata.location.as_geopandas().total_bounds
        filters = (
            (data["valid_time"] >= case_metadata.start_date)
            & (data["valid_time"] <= case_metadata.end_date)
            & (data["latitude"] >= bounds[1])
            & (data["latitude"] <= bounds[3])
            & (data["longitude"] >= utils.convert_longitude_to_180(bounds[0]))
            & (data["longitude"] <= utils.convert_longitude_to_180(bounds[2]))
        )
        subset_target_data = data.loc[filters]

        return subset_target_data

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data is not a pandas DataFrame: {type(data)}")

        # Map report_type column to numeric values
        if "report_type" in data.columns:
            report_type_mapping = {"wind": 1, "hail": 2, "tor": 3}
            data["report_type"] = data["report_type"].map(report_type_mapping)

        # Normalize these times for the LSR data
        # Western hemisphere reports get bucketed to 12Z on the date they fall
        # between 12Z-12Z
        # Eastern hemisphere reports get bucketed to 00Z on the date they occur

        # First, let's figure out which hemisphere each report is in
        western_hemisphere_mask = data["longitude"] < 0
        eastern_hemisphere_mask = data["longitude"] >= 0

        # For western hemisphere: if report is between today 12Z and tomorrow
        # 12Z, assign to today 12Z
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
            in_yesterday_window = (western_data["valid_time"] >= yesterday_twelve_z) & (
                western_data["valid_time"] < twelve_z_times
            )

            # Assign 12Z times
            western_data.loc[in_window_mask, "valid_time"] = twelve_z_times[
                in_window_mask
            ]
            western_data.loc[in_yesterday_window, "valid_time"] = yesterday_twelve_z[
                in_yesterday_window
            ]

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

    # TODO: keep forecasts on original grid for LSRs
    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        return align_forecast_to_target(forecast_data, target_data)


# TODO: get PPH connector working properly
@dataclasses.dataclass
class PPH(TargetBase):
    """Target class for practically perfect hindcast data."""

    name: str = "practically_perfect_hindcast"
    source: str = PPH_URI
    variable_mapping: dict = dataclasses.field(
        default_factory=lambda: IBTrACS_metadata_variable_mapping.copy()
    )

    def _open_data_from_source(
        self,
    ) -> IncomingDataInput:
        return xr.open_zarr(self.source, storage_options=self.storage_options)

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        drop = kwargs.get("drop", False)
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")
        return zarr_target_subsetter(data, case_metadata, drop=drop)

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        if isinstance(data, xr.Dataset):
            return data
        else:
            raise ValueError(f"Data is not an xarray Dataset: {type(data)}")

    def maybe_align_forecast_to_target(
        self,
        forecast_data: xr.Dataset,
        target_data: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        return align_forecast_to_target(forecast_data, target_data)


@dataclasses.dataclass
class IBTrACS(TargetBase):
    """Target class for IBTrACS data."""

    name: str = "IBTrACS"
    source: str = IBTRACS_URI

    def _open_data_from_source(self) -> IncomingDataInput:
        # not using storage_options in this case due to NetCDF4Backend not
        # supporting them
        target_data: pl.LazyFrame = pl.scan_csv(
            self.source,
            storage_options=self.storage_options,
            skip_rows_after_header=1,
        )
        return target_data

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_metadata: "cases.IndividualCase",
        **kwargs,
    ) -> IncomingDataInput:
        # Note: drop parameter not applicable for polars LazyFrame data
        if not isinstance(data, pl.LazyFrame):
            raise ValueError(f"Expected polars LazyFrame, got {type(data)}")

        # Get the season (year) from the case start date, cast as string as
        # polars is interpreting the schema as strings
        season = case_metadata.start_date.year
        if case_metadata.start_date.month > 11:
            season += 1

        # Create a subquery to find all storm numbers in the same season
        matching_numbers = (
            data.filter(pl.col("SEASON").cast(pl.Int64) == season)
            .select("NUMBER")
            .unique()
        )

        possible_names = utils.extract_tc_names(case_metadata.title)

        # Apply the filter to get all data for storms with the same number in
        # the same season, matching any of the possible names
        # This maintains the lazy evaluation
        name_filter = pl.col("tc_name").is_in(possible_names)
        subset_target_data = data.join(
            matching_numbers, on="NUMBER", how="inner"
        ).filter(name_filter & (pl.col("SEASON").cast(pl.Int64) == season))

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

        # Drop rows where wind speed OR pressure are null (equivalent to pandas
        # dropna with how="any")
        subset_target_data = subset_target_data.filter(
            pl.col("surface_wind_speed").is_not_null()
            & pl.col("air_pressure_at_mean_sea_level").is_not_null()
        )
        self._current_case_id = case_metadata.case_id_number

        return subset_target_data

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
        if isinstance(data, pl.LazyFrame):
            data = data.collect(engine="streaming").to_pandas()

            # IBTrACS data is in -180 to 180, convert to 0 to 360
            data["longitude"] = data["longitude"].apply(utils.convert_longitude_to_360)

            # Due to missing data in the IBTrACS dataset, polars doesn't convert
            # the valid_time to a datetime by default
            data["valid_time"] = pd.to_datetime(data["valid_time"])
            data = data.set_index(["valid_time", "latitude", "longitude"])

            try:
                data = xr.Dataset.from_dataframe(data, sparse=True)
            except ValueError as e:
                if "non-unique" in str(e):
                    # Drop duplicates from the pandas DataFrame before converting
                    data_df = data.drop_duplicates()
                    data = xr.Dataset.from_dataframe(data_df, sparse=True)
                else:
                    raise
            return data
        else:
            raise ValueError(f"Data is not a polars LazyFrame: {type(data)}")


def open_kerchunk_reference(
    forecast_dir: str,
    storage_options: dict = {"remote_protocol": "s3", "remote_options": {"anon": True}},
    chunks: Union[dict, str] = "auto",
) -> xr.Dataset:
    """Open a dataset from a kerchunked reference file in parquet or json format.
    This has been built primarily for the CIRA MLWP S3 bucket's data
    (https://registry.opendata.aws/aiwp/),
    but can work with other data in the future. Currently only supports CIRA data unless
    schema is identical to the CIRA schema.

    Args:
        forecast_dir: The path to the kerchunked reference file.
        storage_options: The storage options to use.
        chunks: The chunks to use; defaults to "auto".

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
            "Unknown kerchunk file type found in forecast path, only json and "
            "parquet are supported."
        )
    return kerchunk_ds


def zarr_target_subsetter(
    data: xr.Dataset,
    case_metadata: "cases.IndividualCase",
    time_variable: str = "valid_time",
    drop: bool = False,
) -> xr.Dataset:
    """Subset a zarr dataset to a case operator.

    Args:
        data: The dataset to subset.
        case_metadata: The case metadata to subset the dataset to.
        time_variable: The time variable to use; defaults to "valid_time".

    Returns:
        The subset dataset.
    """
    # Determine the actual time variable in the dataset
    if time_variable not in data.dims:
        if "time" in data.dims:
            time_variable = "time"
        elif "valid_time" in data.dims:
            time_variable = "valid_time"
        else:
            raise ValueError(
                f"No suitable time dimension found in dataset. Available "
                f"dimensions: {list(data.dims)}"
            )

    # subset time first to avoid OOM masking issues
    subset_time_data = data.sel(
        {
            time_variable: slice(
                case_metadata.start_date,
                case_metadata.end_date,
            )
        }
    )
    # mask the data to the case location
    fully_subset_data = case_metadata.location.mask(subset_time_data, drop=drop)
    # chunk the data if it doesn't have chunks, e.g. ARCO ERA5
    if not fully_subset_data.chunks:
        fully_subset_data = fully_subset_data.chunk()
    return fully_subset_data


def align_forecast_to_target(
    forecast_data: xr.Dataset,
    target_data: xr.Dataset,
    # TODO: provide passthrough for other methods
    method: str = "nearest",
) -> tuple[xr.Dataset, xr.Dataset]:
    # Find spatial dimensions that exist in both datasets
    intersection_dims = [
        dim
        for dim in forecast_data.dims
        if dim in target_data.dims
        and dim not in ["time", "valid_time", "lead_time", "init_time"]
    ]

    spatial_dims = {str(dim): target_data[dim] for dim in intersection_dims}
    # Align time dimensions - find overlapping times
    time_aligned_target, time_aligned_forecast = xr.align(
        target_data,
        forecast_data,
        join="inner",
        exclude=spatial_dims.keys(),
    )

    # Regrid forecast to target grid using nearest neighbor interpolation
    # extrapolate in the case of targets slightly outside the forecast domain
    if spatial_dims:
        interp_method: Literal["nearest", "linear"] = (
            "nearest" if method == "nearest" else "linear"
        )

        interp_kwargs = cast(dict[str, Any], {"method": interp_method})
        interp_kwargs.update(spatial_dims)

        time_space_aligned_forecast = time_aligned_forecast.interp(**interp_kwargs)
    else:
        time_space_aligned_forecast = time_aligned_forecast

    return time_space_aligned_forecast, time_aligned_target


def maybe_subset_variables(
    data: IncomingDataInput,
    variables: list[Union[str, "derived.DerivedVariable"]],
    source_module: Optional["sources.base.Source"] = None,
) -> IncomingDataInput:
    """Subset the variables from the data, if required.

    If the variables list includes derived variables, extracts their required
    and optional variables for subsetting.

        Args:
        data: The dataset to subset (xr.Dataset, xr.DataArray, pl.LazyFrame,
            or pd.DataFrame).
        variables: List of variable names and/or derived variable classes.
        source_module: Optional pre-created source module. If None, creates one.

    Returns:
        The data subset to only the specified variables.
    """
    # If there are no variables, return the data unaltered
    if len(variables) == 0:
        return data

    expected_and_maybe_derived_variables = (
        derived.maybe_include_variables_from_derived_input(variables)
    )

    # Use provided source module or get one
    if source_module is None:
        source_module = sources.get_backend_module(type(data))
    data = source_module.safely_pull_variables(
        data,
        expected_and_maybe_derived_variables,
    )
    return data


def check_for_missing_data(
    data: IncomingDataInput,
    case_metadata: "cases.IndividualCase",
    source_module: Optional["sources.base.Source"] = None,
) -> bool:
    """Check if the data has missing data in the given date range."""
    # Use provided source module or get one
    if source_module is None:
        source_module = sources.get_backend_module(type(data))

    # First check if the data has valid times in the given date range
    if not source_module.check_for_valid_times(
        data, case_metadata.start_date, case_metadata.end_date
    ):
        return False
    # Then check if the data has spatial data for the given location
    elif not source_module.check_for_spatial_data(data, case_metadata.location):
        return False
    else:
        return True
