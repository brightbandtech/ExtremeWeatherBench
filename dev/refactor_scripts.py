from __future__ import annotations

import dataclasses
import datetime
import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, TypeAlias, Union

import dacite
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from extremeweatherbench import regions, utils

# from extremeweatherbench import derived, metrics, targets  # type: ignore
# from extremeweatherbench.case import CaseOperator
# from extremeweatherbench.forecasts import Forecast
# from extremeweatherbench.regions import Region

IncomingDataInput: TypeAlias = xr.Dataset | xr.DataArray | pl.LazyFrame | pd.DataFrame


class Forecast(ABC):
    """A base class defining the interface for ExtremeWeatherBench forecast data.

    A Forecast is data that acts as the "forecast" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset.
    """

    def __init__(self, forecast_source: str):
        self.forecast_source = forecast_source

    @abstractmethod
    def open_data_from_source(
        self, forecast_storage_options: Optional[dict] = None
    ) -> IncomingDataInput:
        """Open the forecast data from the source.

        Args:
            forecast_storage_options: Optional storage options for the forecast source if the source is a remote URL.

        Returns:
            The forecast data with a type determined by the user.
        """

    def forecast_preprocess(
        self,
        forecast_ds: xr.Dataset,
        forecast_preprocess_function: Callable = utils._default_preprocess,
    ) -> xr.Dataset:
        forecast_ds = forecast_preprocess_function(forecast_ds)
        return forecast_ds

    def run_pipeline(
        self,
        case_operator: CaseOperator,
        **kwargs,
    ):
        forecast_ds = self.open_data_from_source(
            forecast_storage_options=case_operator.forecast_storage_options
        )
        forecast_ds = self.forecast_preprocess(
            forecast_ds,
            forecast_preprocess_function=kwargs.get(
                "forecast_preprocess_function", utils._default_preprocess
            ),
        )
        forecast_ds = _maybe_rename_and_subset_forecast_dataset(
            forecast_ds,
            case_operator=case_operator,
        )
        forecast_ds = _maybe_convert_dataset_lead_time_to_int(forecast_ds)
        return forecast_ds


class DerivedVariable(ABC):
    """An abstract base class defining the interface for ExtremeWeatherBench derived variables.

    A DerivedVariable is any variable that requires extra computation than what is provided in analysis
    or forecast data. Some examples include the practically perfect hindcast, MLCAPE, IVT, or atmospheric river masks.

    Attributes:
        name: The name that is used for applications of derived variables. Defaults to the class name.
        input_variables: A list of variables that are used to build the variable.
        build: A method that builds the variable from the input variables. Build is used specifically to distinguish
        from the compute method in xarray, which eagerly processes the data and loads into memory; build is used to
        lazily process the data and return a dataset that can be used later to compute the variable.
        _check_data_for_variables: A method that checks that the data has the variables required to build the variable,
        using input_variables.
        derive_variable: An abstractmethod that defines the computation to derive the variable from input_variables.
    """

    @property
    def name(self) -> str:
        """A name for the derived variable. Defaults to the class name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """A list of variables that are used to compute the variable.

        Each derived variable is a product of one or more variables in an incoming dataset.
        The input variables are the names of the variables in the incoming dataset.

        """
        pass

    @abstractmethod
    def derive_variable(self, data: xr.Dataset) -> xr.DataArray:
        """Derive the variable from the input variables.

        The output of the derivation must be a single variable output returned as
        a DataArray.

        Args:
            data: The dataset to derive the variable from.

        Returns:
            A DataArray with the derived variable.
        """
        pass

    def build(self, data: xr.Dataset) -> xr.DataArray:
        """Build the derived variable from the input variables.

        This method is used to build the derived variable from the input variables.
        It checks that the data has the variables required to build the variable,
        and then derives the variable from the input variables.

        Args:
            data: The dataset to build the derived variable from.

        Returns:
            A DataArray with the derived variable.
        """
        self._check_data_for_variables(data)
        return self.derive_variable(data)

    def _check_data_for_variables(self, data: xr.Dataset):
        """Check that the data has the variables required to build the variable, based on assigned input variables."""
        for v in self.input_variables:
            if v not in data.data_vars:
                raise ValueError(f"Input variable {v} not found in data")


class BaseMetric(ABC):
    """A BaseMetric class is an abstract class that defines the foundational interface for all metrics.

    Metrics are general operations applied between a forecast and analysis
    xarray dataset. EWB metrics prioritize the use of any arbitrary sets of forecasts
    and analyses, so long as the spatiotemporal dimensions are the same.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def compute(self, forecast: xr.Dataset, observation: xr.Dataset):
        pass


class AppliedMetric(ABC):
    """An applied metric is a derivative of a BaseMetric.

    It is a wrapper around one or more BaseMetrics that is intended for more complex rollups or aggregations.
    Typically, these metrics are used for one event type and are very specific. Temporal onset mean error,
    case duration mean error, and maximum temperature mean absolute error, are all examples of applied metrics.

    Attributes:
        base_metrics: A list of BaseMetrics to compute.
        compute_metric: A required method to compute the metric.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def base_metrics(self) -> list[BaseMetric]:
        pass

    def compute_metric(
        self,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        forecast_variables: list[str],
        observation_variables: list[str],
        **kwargs,
    ):
        self.base_metrics().compute(forecast, observation)


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single or individual case.

    An IndividualCase defines the relevant metadata for a single case study for a
    given extreme weather event; it is designed to be easily instantiable through a
    simple YAML-based configuration file.

    Attributes:
        case_id_number: A unique numerical identifier for the event.
        start_date: The start date of the case, for use in subsetting data for analysis.
        end_date: The end date of the case, for use in subsetting data for analysis.
        location: A Location dataclass representing the location of a case.
        event_type: A string representing the type of extreme weather event.
    """

    case_id_number: int
    title: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    location: "regions.Region"
    event_type: str


@dataclasses.dataclass
class IndividualCaseCollection:
    """A collection of IndividualCases."""

    cases: list[IndividualCase]


@dataclasses.dataclass
class CaseOperator:
    """A class which stores the graph to process an individual case.

    This class is used to store the graph to process an individual case. The purpose of
    this class is to be a one-stop-shop for the evaluation of a single case. Multiple
    CaseOperators can be run in parallel to evaluate multiple cases, or run through the
    ExtremeWeatherBench.run() method to evaluate all cases in an evaluation in serial.

    Attributes:
        case: IndividualCase metadata
        metric: A metric that is intended to be evaluated for the case
        target: A target to evaluate against the forecast
        forecast: The incoming forecast data
        target_variables: Names of the variables present in the target data relevant to the evaluation
        forecast_variables: Names of the variables present in the forecast data relevant to the evaluation
    """

    case: IndividualCase
    metric: "BaseMetric"
    target: "TargetBase"
    forecast: "Forecast"
    target_variables: list[Union[str, "DerivedVariable"]]
    forecast_variables: list[str | "DerivedVariable"]
    target_storage_options: dict
    forecast_storage_options: dict
    target_variable_mapping: dict
    forecast_variable_mapping: dict


class TargetBase(ABC):
    """
    An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """

    source: str

    @abstractmethod
    def open_data_from_source(
        self, target_storage_options: Optional[dict] = None
    ) -> IncomingDataInput:
        """
        Open the target data from the source, opting to avoid loading the entire dataset into memory if possible.

        Args:
            source: The source of the observation data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.

        Returns:
            The target data with a type determined by the user.
        """

    @abstractmethod
    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case: CaseOperator,
    ) -> IncomingDataInput:
        """
        Subset the target data to the case information provided in CaseOperator.

        Time information, spatial bounds, and variables are captured in the case metadata
        where this method is used to subset.

        Args:
            data: The observation data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
            pandas dataframe, or numpy array.
            case: The case operator to subset the data to; includes time information, spatial bounds, and variables.

        Returns:
            The target data with the variables subset to the case metadata.
        """

    def maybe_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
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

    def _custom_convert_to_dataset(self, data: IncomingDataInput) -> xr.Dataset:
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

    def run_pipeline(
        self,
        case_operator: CaseOperator,
        target_storage_options: Optional[dict] = None,
        target_variable_mapping: dict = {},
    ) -> xr.Dataset:
        """
        Shared method for running the target pipeline.

        Args:
            source: The source of the target data, which can be a local path or a remote URL.
            storage_options: Optional storage options for the source if the source is a remote URL.
            target_variables: The variables to include in the target. Some target objects may not have variables, or
            only have a singular variable; thus, this is optional.
            target_variable_mapping: A dictionary of variable names to map to the target data.
            **kwargs: Additional keyword arguments to pass in as needed.

        Returns:
            The target data with a type determined by the user.
        """

        # Open data and process through pipeline steps
        data = (
            # opens data from user-defined source
            self.open_data_from_source(
                target_storage_options=target_storage_options,
            )
            # maps variable names to the target data if not already using EWB naming conventions
            .pipe(
                maybe_map_variable_names,
                variable_mapping=target_variable_mapping,
            )
            # subsets the target data using the caseoperator metadata
            .pipe(
                self.subset_data_to_case,
                case_operator=case_operator,
            )
            # converts the target data to an xarray dataset if it is not already
            .pipe(self.maybe_convert_to_dataset)
            # derives variables from the target data if derived variables are defined
            .pipe(maybe_derive_variables, variables=case_operator.target_variables)
        )
        return data


@dataclasses.dataclass
class MetricEvaluationObject:
    """A class to store the evaluation object for a metric."""

    metric: BaseMetric
    target: TargetBase
    forecast: Forecast
    target_variables: list[str | DerivedVariable]
    forecast_variables: list[str | DerivedVariable]
    target_storage_options: dict
    forecast_storage_options: dict
    target_variable_mapping: dict
    forecast_variable_mapping: dict


class EventType(ABC):
    """A base class defining the interface for ExtremeWeatherBench event types.

    An Event in ExtremeWeatherBench defines a specific weather event type, such as a heat wave,
    severe convective weather, or atmospheric rivers. These events encapsulate a set of cases and
    derived behavior for evaluating those cases. These cases will share common metrics, observations,
    and variables while each having unique dates and locations.

    Attributes:
        case_metadata: A dictionary with case metadata; EWB uses a YAML file to define the cases upstream.
        metric_evaluation_objects: A list of MetricEvaluationObject objects.
    """

    def __init__(
        self,
        case_metadata: dict[str, Any],
        metric_evaluation_objects: list[MetricEvaluationObject],
    ):
        """Initialize the EventType.

        Args:
            case_metadata: A dictionary with case metadata; EWB uses a YAML file to define the cases upstream.
            metric_evaluation_objects: A list of MetricEvaluationObject objects.
        """
        self.case_metadata = case_metadata
        self.metric_evaluation_objects = metric_evaluation_objects

    @property
    @abstractmethod
    def event_type(self) -> str:
        pass

    def _build_base_case_metadata_collection(self) -> IndividualCaseCollection:
        """Build a list of IndividualCases from the case_metadata."""

    def build_case_operators(
        self,
    ) -> list["CaseOperator"]:
        """Build a CaseOperator from the event type.

        Args:
            forecast_source: The forecast source to use for the case operators.

        Returns:
            A list of CaseOperator objects.
        """
        case_metadata_collection = dacite.from_dict(
            data_class=IndividualCaseCollection,
            data=self.case_metadata,
            config=dacite.Config(
                type_hooks={regions.Region: regions.map_to_create_region},
            ),
        )
        case_metadata_collection = [
            c for c in case_metadata_collection.cases if c.event_type == self.event_type
        ]

        case_operators = []
        for single_case, metric_evaluation_object in itertools.product(
            case_metadata_collection, self.metric_evaluation_objects
        ):
            case_operators.append(
                CaseOperator(
                    case=single_case,
                    metric=metric_evaluation_object.metric,
                    target=metric_evaluation_object.target,
                    forecast=metric_evaluation_object.forecast,
                    target_variables=metric_evaluation_object.target_variables,
                    forecast_variables=metric_evaluation_object.forecast_variables,
                    target_storage_options=metric_evaluation_object.target_storage_options,
                    forecast_storage_options=metric_evaluation_object.forecast_storage_options,
                    target_variable_mapping=metric_evaluation_object.target_variable_mapping,
                    forecast_variable_mapping=metric_evaluation_object.forecast_variable_mapping,
                )
            )
        return case_operators


def maybe_map_variable_names(
    data: IncomingDataInput, variable_mapping: Optional[dict] = None
) -> IncomingDataInput:
    """Map the variable names to the observation data, if required.

    Args:
        data: The incoming data in the form of an object that has a rename method for data variables/columns.
        variable_mapping: The mapping of variable names to the incoming data, with the format {incoming_name: new_name}.

    Returns:
        A dataset with mapped variable names, if any exist, else the original data.
    """
    if variable_mapping is None:
        return data
    # Filter the mapping to only include variables that exist in the dataset
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        subset_variable_mapping = {
            v: k for v, k in variable_mapping.items() if v in data.data_vars
        }
    elif isinstance(data, (pl.LazyFrame, pl.DataFrame, pd.DataFrame)):
        subset_variable_mapping = {
            v: k for v, k in variable_mapping.items() if v in data.columns
        }
    else:
        raise ValueError(
            f"Data is not a dataset, data array, lazy frame, dataframe, or pandas dataframe: {type(data)}"
        )
    if subset_variable_mapping:
        data = data.rename(subset_variable_mapping)
    return data


def maybe_derive_variables(
    ds: xr.Dataset, variables: list[str | DerivedVariable]
) -> xr.Dataset:
    """Derive variables from the data if any exist in a list of variables.

    Derived variables must maintain the same spatial dimensions as the original dataset.

    Args:
        ds: The dataset, ideally already subset in case of in memory operations in the derived variables.
        case: The case to derive the variables for.
        variables: The potential variables to derive as a list of strings or DerivedVariable objects.

    Returns:
        A dataset with derived variables, if any exist, else the original dataset.
    """
    derived_variables = {}

    derived_variables = [v for v in variables if not isinstance(v, str)]
    if derived_variables:
        for v in derived_variables:
            derived_variable = v()
            derived_data = derived_variable.build(data=ds)
            ds[derived_variable.name] = derived_data

    return ds


def open_and_preprocess_forecast_dataset(
    forecast_dir: str,
    forecast_variables: list[str | DerivedVariable],
    forecast_variable_mapping: dict[str, list[str | DerivedVariable]],
    forecast_preprocess: Callable = utils._default_preprocess,
    forecast_storage_options: dict = {
        "remote_protocol": "s3",
        "remote_options": {"anon": True},
    },
    forecast_chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
) -> xr.Dataset:
    """Open the forecast dataset specified for evaluation.

    If a URI is provided (e.g. s3://bucket/path/to/forecast), the filesystem
    will be inferred from the provided source (in this case, s3). Otherwise,
    the filesystem will assumed to be local.

    Preprocessing examples:
        A typical preprocess function handles metadata changes:

        def _preprocess_cira_forecast_dataset(
            ds: xr.Dataset
        ) -> xr.Dataset:
            ds = ds.rename({"time": "lead_time"})
            return ds

        The preprocess function is applied before variable renaming occurs, so it should
        reference the original variable names in the forecast dataset, not the standardized
        names defined in the ForecastSchemaConfig.

    Args:
        eval_config: The evaluation configuration.
        forecast_schema_config: The forecast schema configuration.
        preprocess: A function that preprocesses the forecast dataset.

    Returns:
        The opened forecast dataset.
    """
    if "zarr" in forecast_dir:
        forecast_ds = xr.open_zarr(forecast_dir, chunks=forecast_chunks)
    elif "parq" in forecast_dir or "json" in forecast_dir or "parquet" in forecast_dir:
        forecast_ds = open_kerchunk_reference(
            forecast_dir,
            storage_options=forecast_storage_options,
            chunks=forecast_chunks,
        )
    else:
        raise TypeError(
            "Unknown file type found in forecast path, only json, parquet, and zarr are supported."
        )
    forecast_ds = forecast_preprocess(forecast_ds)
    forecast_ds = _maybe_rename_and_subset_forecast_dataset(
        forecast_ds, forecast_variable_mapping
    )
    return forecast_ds


def open_kerchunk_reference(
    forecast_dir: str,
    storage_options: dict = {"remote_protocol": "s3", "remote_options": {"anon": True}},
    chunks: dict = {"time": 48, "latitude": 721, "longitude": 1440},
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
            forecast_dir, engine="kerchunk", storage_options=storage_options
        )
        kerchunk_ds = kerchunk_ds.compute()
    elif "json" in forecast_dir:
        storage_options["fo"] = forecast_dir
        kerchunk_ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "storage_options": storage_options,
                "consolidated": False,
            },
        )
    else:
        raise TypeError(
            "Unknown kerchunk file type found in forecast path, only json and parquet are supported."
        )
    return kerchunk_ds


def _maybe_rename_and_subset_forecast_dataset(
    data: xr.Dataset, case_operator: CaseOperator
) -> xr.Dataset:
    """Subset and rename a dataset to the correct names expected by the evaluation routines.

    Args:
        data: The incoming data in the form of an object that has a rename method for data variables/columns.
        variable_mapping: The mapping of variable names to the incoming data, with the format {incoming_name: new_name}.

    Returns:
        A dataset with mapped variable names, if any exist, else the original data.
    """
    # Mapping here is used to rename the incoming data variables to the correct
    # names expected by the evaluation routines.

    # subset time first to avoid OOM masking issues
    subset_time_indices = utils.derive_indices_from_init_time_and_lead_time(
        data,
        case_operator.case.start_date,
        case_operator.case.end_date,
    )
    subset_time_data = data.isel(init_time=np.unique(subset_time_indices[0]))
    # Filter the mapping to only include variables that are in the forecast dataset, else
    # an error will be raised.
    subset_time_data = maybe_map_variable_names(
        subset_time_data, case_operator.forecast_variable_mapping
    )
    try:
        subset_time_data = subset_time_data[case_operator.forecast_variables]
    except KeyError:
        raise KeyError(
            f"Variables {case_operator.forecast_variables} not found in forecast data"
        )
    fully_subset_data = case_operator.case.location.mask(subset_time_data, drop=True)
    return fully_subset_data


def _maybe_convert_dataset_lead_time_to_int(dataset: xr.Dataset) -> xr.Dataset:
    """Convert types of variables in an xarray Dataset based on the schema,
    ensuring that, for example, the variable representing lead_time is of type int.

    Args:
        dataset: The input xarray Dataset that uses the schema's variable names.

    Returns:
        An xarray Dataset with adjusted types.
    """

    lead_time = (
        dataset["lead_time"] if "lead_time" in dataset.coords else dataset["time"]
    )
    if lead_time.dtype == np.dtype("timedelta64[ns]"):
        # Convert timedelta64[ns] to hours and cast to int
        dataset["lead_time"] = (lead_time / np.timedelta64(1, "h")).astype(int)
    elif lead_time.dtype == np.dtype("int64"):
        # Already an int, do nothing
        pass
    else:
        temporal_resolution_hours = np.squeeze(
            np.unique(np.diff(dataset["time"].values)) / np.timedelta64(1, "h")
        )
        dataset["time"] = np.arange(
            0,
            dataset["time"].shape[0] * temporal_resolution_hours,
            temporal_resolution_hours,
        )
        dataset = dataset.rename({"time": "lead_time"})
    return dataset

    # def _build_forecast(forecast_dir: str, case_operator: CaseOperator, **kwargs):
    forecast_ds = open_and_preprocess_forecast_dataset(
        forecast_dir,
        forecast_variables=case_operator.forecast_variables,
        forecast_variable_mapping=kwargs.get("forecast_variable_mapping", {}),
        forecast_storage_options=kwargs.get(
            "forecast_storage_options",
            {"remote_protocol": "s3", "remote_options": {"anon": True}},
        ),
        forecast_chunks=kwargs.get(
            "forecast_chunks", {"time": 48, "latitude": 721, "longitude": 1440}
        ),
    )
    time_indices = utils.derive_indices_from_init_time_and_lead_time(
        forecast_ds, case_operator.case.start_date, case_operator.case.end_date
    )
    forecast_ds_time_subset = forecast_ds.isel(init_time=np.unique(time_indices))
    forecast_ds = case_operator.case.location.mask(forecast_ds_time_subset, drop=True)
    return forecast_ds


class ExtremeWeatherBench:
    def __init__(
        self,
        events: list[EventType],
        forecast: Forecast,
    ):
        self.events = events
        self.forecast = forecast

    @property
    def case_operators(self) -> list[CaseOperator]:
        case_operators = []
        for event in self.events:
            case_operators.extend(event.build_case_operators(self.forecast))
        return case_operators

    def run(
        self,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Runs the workflow in the order of the event operators and cases inside the event operators."""
        # In run(), set up a method to store the opened targets and forecasts to avoid reloading each case operator

        # Collect all case operators from all events
        for event in self.events:
            case_operators = event.build_case_operators()
            for case_operator in case_operators:
                observation_ds, forecast_ds = self.build_datasets(
                    case_operator, **kwargs
                )
                return observation_ds, forecast_ds

    def build_datasets(self, case_operator: CaseOperator, **kwargs):
        observation_ds = case_operator.target().run_pipeline(
            case_operator=case_operator,
            target_storage_options=case_operator.target_storage_options,
            target_variable_mapping=case_operator.target_variable_mapping,
        )
        forecast_ds = case_operator.forecast.run_pipeline(
            case_operator=case_operator,
            forecast_storage_options=case_operator.forecast_storage_options,
            forecast_variable_mapping=case_operator.forecast_variable_mapping,
            forecast_preprocess_function=kwargs.get(
                "forecast_preprocess_function", utils._default_preprocess
            ),
        )
        return observation_ds, forecast_ds

    # def build_forecast(self, case_operator: CaseOperator, forecast_dir: str, **kwargs):
    #     pre_derived_forecast_ds = _build_forecast(forecast_dir, case_operator, **kwargs)
    #     derived_forecast_ds = maybe_derive_variables(
    #         pre_derived_forecast_ds, variables=case_operator.forecast_variables
    #     )
    #     return derived_forecast_ds
