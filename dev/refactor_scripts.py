from __future__ import annotations

import dataclasses
import datetime
import inspect
import itertools
import logging
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, TypeAlias

import dacite
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import regions, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# from extremeweatherbench import derived, metrics, targets  # type: ignore
# from extremeweatherbench.case import CaseOperator
# from extremeweatherbench.forecasts import Forecast
# from extremeweatherbench.regions import Region

IncomingDataInput: TypeAlias = xr.Dataset | xr.DataArray | pl.LazyFrame | pd.DataFrame


def catch_exceptions(func: Callable) -> Callable:
    """Catch exceptions and log them."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}, returning nan")
            return xr.DataArray(np.nan)

    return wrapper


def _default_preprocess(input_data: IncomingDataInput) -> IncomingDataInput:
    """Default forecast preprocess function that does nothing."""
    return input_data


@dataclasses.dataclass
class TargetConfig:
    target: TargetBase
    source: str | Path
    variables: list[str | DerivedVariable]
    variable_mapping: dict
    storage_options: dict
    preprocess: Callable = _default_preprocess


@dataclasses.dataclass
class ForecastConfig:
    forecast: ForecastBase
    source: str | Path
    variables: list[str | DerivedVariable]
    variable_mapping: dict
    storage_options: dict
    preprocess: Callable = _default_preprocess


def _filter_kwargs_for_callable(kwargs: dict, callable_obj: Callable) -> dict:
    """Filter kwargs to only include arguments that the callable can accept.

    This method uses introspection to determine which arguments the callable
    can accept and filters kwargs accordingly.

    Args:
        kwargs: The full kwargs dictionary to filter
        callable_obj: The callable (function, method, etc.) to check against

    Returns:
        A filtered dictionary containing only the kwargs that the callable can accept
    """
    # Get the signature of the callable
    sig = inspect.signature(callable_obj)

    # Get the parameter names that the callable accepts
    # Handle different types of callables (functions, methods, etc.)
    if hasattr(callable_obj, "__self__") and callable_obj.__self__ is not None:
        # This is a bound method, skip 'self'
        accepted_params = list(sig.parameters.keys())[1:]
    else:
        # This is a function or unbound method, include all parameters
        accepted_params = list(sig.parameters.keys())

    # Filter kwargs to only include accepted parameters
    filtered_kwargs = {}
    for param_name in accepted_params:
        if param_name in kwargs:
            filtered_kwargs[param_name] = kwargs[param_name]

    return filtered_kwargs


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
        variables: list[str | DerivedVariable],
        variable_mapping: dict = {},
        storage_options: Optional[dict] = None,
        preprocess: Callable = _default_preprocess,
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
    ) -> IncomingDataInput:
        data = self._open_data_from_source()
        data = self.preprocess(data)
        return data

    @abstractmethod
    def _open_data_from_source(self) -> IncomingDataInput:
        """
        Open the target data from the source, opting to avoid loading the entire dataset into memory if possible.

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
            data: The target data to subset, which should be a xarray dataset, xarray dataarray, polars lazyframe,
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

    def add_source_to_dataset_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Add the name of the dataset to the dataset attributes."""
        ds.attrs["source"] = self.name
        return ds


class TargetBase(InputBase):
    """
    An abstract base class for target data.

    A TargetBase is data that acts as the "truth" for a case. It can be a gridded dataset,
    a point observation dataset, or any other reference dataset. Targets in EWB
    are not required to be the same variable as the forecast dataset, but they must be in the
    same coordinate system for evaluation.
    """


class ForecastBase(InputBase):
    """A base class defining the interface for ExtremeWeatherBench forecast data.

    A Forecast is data that acts as the "forecast" for a case.

    Attributes:
        forecast_source: The source of the forecast data, which can be a local path or a remote URL/URI.
    """

    def subset_data_to_case(
        self,
        data: IncomingDataInput,
        case_operator: CaseOperator,
    ) -> IncomingDataInput:
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected xarray Dataset, got {type(data)}")

        # subset time first to avoid OOM masking issues
        subset_time_indices = utils.derive_indices_from_init_time_and_lead_time(
            data,
            case_operator.case.start_date,
            case_operator.case.end_date,
        )

        subset_time_data = data.isel(init_time=np.unique(subset_time_indices[0]))
        subset_time_data = lead_time_init_time_to_valid_time(subset_time_data)

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

    # default to preserving lead_time in EWB metrics
    preserve_dims: str = "lead_time"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs,
    ):
        """Compute the metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        pass

    def compute_metric(
        self,
        forecast: xr.Dataset,
        target: xr.Dataset,
        **kwargs,
    ):
        return self._compute_metric(
            forecast,
            target,
            **_filter_kwargs_for_callable(kwargs, self._compute_metric),
        )


class AppliedMetric(ABC):
    """An applied metric is a derivative of a BaseMetric.

    It is a wrapper around one or more BaseMetrics that is intended for more complex rollups or aggregations.
    Typically, these metrics are used for one event type and are very specific. Temporal onset mean error,
    case duration mean error, and maximum temperature mean absolute error, are all examples of applied metrics.

    Attributes:
        base_metrics: A list of BaseMetrics to compute.
        compute_applied_metric: A required method to compute the metric.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def base_metric(self) -> BaseMetric:
        pass

    def compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ):
        return self.base_metric()._compute_metric(
            **self._compute_applied_metric(
                forecast,
                target,
                **_filter_kwargs_for_callable(kwargs, self._compute_applied_metric),
            ),
            **_filter_kwargs_for_callable(kwargs, self.base_metric()._compute_metric),
        )

    @abstractmethod
    def _compute_applied_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ):
        """Compute the applied metric.

        Args:
            forecast: The forecast dataset.
            target: The target dataset.
            kwargs: Additional keyword arguments to pass to the applied metric.
        """
        pass


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
        target_config: A TargetConfig object
        forecast_config: A ForecastConfig object
    """

    case: IndividualCase
    metric: "BaseMetric"
    target_config: "TargetConfig"
    forecast_config: "ForecastConfig"


@dataclasses.dataclass
class MetricEvaluationObject:
    """A class to store the evaluation object for a metric.

    A MetricEvaluationObject is a metric evaluation object for all cases in an event.
    The evaluation is a set of all metrics, target variables, and forecast variables.

    Multiple MEO's can be used to evaluate a single event type. This is useful for
    evaluating distinct Targets or metrics with unique variables to evaluate.

    Attributes:
        event_type: The event type to evaluate.
        metric: A list of BaseMetric objects.
        target_config: A TargetConfig object.
        forecast_config: A ForecastConfig object.
    """

    event_type: str
    metric: list[BaseMetric]
    target_config: TargetConfig
    forecast_config: ForecastConfig


class ExtremeWeatherBench:
    def __init__(
        self,
        cases: dict[str, list],
        metrics: list[MetricEvaluationObject],
    ):
        self.cases = cases
        self.metrics = metrics

    @property
    def case_operators(self) -> list[CaseOperator]:
        return build_case_operators(self.cases, self.metrics)

    def run(
        self,
        cache_dir: Optional[str | Path] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the workflow in the order of the event operators and cases inside the event operators."""
        self.cache_dir = cache_dir

        # instantiate the cache directory if caching and build it if it does not exist
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                self.cache_dir = Path(self.cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        run_results = []
        with logging_redirect_tqdm():
            for case_operator in tqdm(self.case_operators):
                run_results.append(self.compute_case_operator(case_operator, **kwargs))

                # store the results of each case operator if caching
                if self.cache_dir:
                    pd.concat(run_results).to_pickle(
                        self.cache_dir / "case_results.pkl"
                    )
        return pd.concat(run_results, ignore_index=True)

    def compute_case_operator(self, case_operator: CaseOperator, **kwargs):
        target_ds, forecast_ds = self._build_datasets(case_operator, **kwargs)

        # align the target and forecast datasets to ensure they have the same valid_time dimension
        target_ds, forecast_ds = xr.align(target_ds, forecast_ds)

        # compute and cache the datasets if requested
        if kwargs.get("pre_compute", False):
            target_ds, forecast_ds = self._compute_and_maybe_cache(
                target_ds, forecast_ds
            )

        logger.info(f"datasets built for case {case_operator.case.case_id_number}")
        results = []
        for variables, metric in itertools.product(
            zip(
                case_operator.forecast_config.variables,
                case_operator.target_config.variables,
            ),
            case_operator.metric,
        ):
            results.append(
                self._evaluate_metric_and_return_df(
                    target_ds=target_ds,
                    forecast_ds=forecast_ds,
                    forecast_variable=variables[0],
                    target_variable=variables[1],
                    metric=metric,
                    target_name=case_operator.target_config.target.name,
                    case_id_number=case_operator.case.case_id_number,
                    event_type=case_operator.case.event_type,
                    **kwargs,
                )
            )

            # cache the results of each metric if caching
            if self.cache_dir:
                results.to_pickle(self.cache_dir / "results.pkl")

        return pd.concat(results, ignore_index=True)

    def _compute_and_maybe_cache(self, *datasets: xr.Dataset):
        """Compute and cache the datasets if caching."""
        logger.info("computing datasets")
        computed_datasets = (dataset.compute() for dataset in datasets)
        if self.cache_dir:
            raise NotImplementedError("Caching is not implemented yet")
            # (computed_dataset.to_netcdf(self.cache_dir) for computed_dataset in computed_datasets)
        return computed_datasets

    def _evaluate_metric_and_return_df(
        self,
        forecast_ds: xr.Dataset,
        target_ds: xr.Dataset,
        forecast_variable: str | DerivedVariable,
        target_variable: str | DerivedVariable,
        metric: BaseMetric,
        case_id_number: int,
        event_type: str,
        **kwargs,
    ):
        metric = metric()
        logger.info(f"computing metric {metric.name}")
        metric_result = metric.compute_metric(
            forecast_ds[forecast_variable],
            target_ds[target_variable],
            **kwargs,
        )

        # Convert to DataFrame and add metadata
        df = metric_result.to_dataframe(name="value").reset_index()
        df["target_variable"] = target_variable
        df["metric"] = metric.name
        df["target_source"] = target_ds.attrs["source"]
        df["forecast_source"] = forecast_ds.attrs["source"]
        df["case_id_number"] = case_id_number
        df["event_type"] = event_type
        return df

    def _build_datasets(self, case_operator: CaseOperator, **kwargs):
        """Build the target and forecast datasets for a case operator.

        This method will process through all stages of the pipeline for the target and forecast datasets,
        including preprocessing, variable renaming, and subsetting.
        """
        target_input = case_operator.target_config.target(
            source=case_operator.target_config.source,
            variables=case_operator.target_config.variables,
            variable_mapping=case_operator.target_config.variable_mapping,
            storage_options=case_operator.target_config.storage_options,
            preprocess=case_operator.target_config.preprocess,
        )

        forecast_input = case_operator.forecast_config.forecast(
            source=case_operator.forecast_config.source,
            variables=case_operator.forecast_config.variables,
            variable_mapping=case_operator.forecast_config.variable_mapping,
            storage_options=case_operator.forecast_config.storage_options,
            preprocess=case_operator.forecast_config.preprocess,
        )

        logger.info("running target pipeline")
        target_ds = run_pipeline(
            input_data=target_input,
            case_operator=case_operator,
        )

        logger.info("running forecast pipeline")
        forecast_ds = run_pipeline(
            input_data=forecast_input,
            case_operator=case_operator,
        )
        return target_ds, forecast_ds


def lead_time_init_time_to_valid_time(forecast: xr.Dataset) -> xr.DataArray:
    """Convert init_time and lead_time to valid_time.

    Args:
        forecast: The forecast dataset.

    Returns:
        The forecast dataset with valid_time dimension.
    """
    if "lead_time" not in forecast.dims or "init_time" not in forecast.dims:
        raise ValueError(
            "lead_time and init_time must be dimensions of the forecast dataset"
        )

    lead_time_grid, init_time_grid = np.meshgrid(forecast.lead_time, forecast.init_time)
    valid_times = (
        init_time_grid.flatten()
        + pd.to_timedelta(lead_time_grid.flatten(), unit="h").to_numpy()
    )
    return valid_times


def align_target_and_forecast_time_dimensions(
    target_ds: xr.Dataset, forecast_ds: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """Align the time dimensions of the target and forecast datasets.

    Args:
        target_ds: The target dataset.
        forecast_ds: The forecast dataset.

    Returns:
        The aligned target and forecast datasets.
    """
    valid_time = xr.DataArray(
        forecast_ds.init_time, coords={"init_time": forecast_ds.init_time}
    ) + xr.DataArray(forecast_ds.lead_time, coords={"lead_time": forecast_ds.lead_time})
    trimmed_valid_time = valid_time.where(valid_time.isin(target_ds.time.values))

    return trimmed_valid_time


def min_if_all_timesteps_present(
    da: xr.DataArray,
    num_timesteps: int,
) -> xr.DataArray:
    """Return the minimum value of a DataArray if all timesteps of a day are present.

    Args:
        da: The input DataArray.

    Returns:
        The minimum value of the DataArray if all timesteps are present, otherwise the original DataArray.
    """
    if len(da.values) == num_timesteps:
        return da.min()
    else:
        return xr.DataArray(np.nan)


def min_if_all_timesteps_present_forecast(
    da: xr.DataArray, num_timesteps
) -> xr.DataArray:
    """Return the minimum value of a DataArray if all timesteps of a day are present given a dataset with lead_time and
    valid_time dimensions.

    Args:
        da: The input DataArray.

    Returns:
        The minimum value of the DataArray if all timesteps are present, otherwise the original DataArray.
    """
    if len(da.valid_time) == num_timesteps:
        return da.min("valid_time")
    else:
        # Return an array with the same lead_time dimension but filled with NaNs
        return xr.DataArray(
            np.full(len(da.lead_time), np.nan),
            coords={"lead_time": da.lead_time},
            dims=["lead_time"],
        )


def build_case_operators(
    cases_dict: dict[str, list],
    metric_evaluation_objects: list[MetricEvaluationObject],
) -> list["CaseOperator"]:
    """Build a CaseOperator from the case metadata and metric evaluation objects.

    Args:
        cases_dict: The case metadata to use for the case operators.
        metric_evaluation_objects: The metric evaluation objects to use for the case operators.

    Returns:
        A list of CaseOperator objects.
    """
    case_metadata_collection = dacite.from_dict(
        data_class=IndividualCaseCollection,
        data=cases_dict,
        config=dacite.Config(
            type_hooks={regions.Region: regions.map_to_create_region},
        ),
    )

    # build list of case operators based on information provided in case dict and
    case_operators = []
    for single_case, metric_evaluation_object in itertools.product(
        case_metadata_collection.cases, metric_evaluation_objects
    ):
        # checks if case matches the event type provided in metric eval object
        if single_case.event_type in metric_evaluation_object.event_type:
            case_operators.append(
                CaseOperator(
                    case=single_case,
                    metric=metric_evaluation_object.metric,
                    target_config=metric_evaluation_object.target_config,
                    forecast_config=metric_evaluation_object.forecast_config,
                )
            )
    return case_operators


def maybe_map_variable_names(
    data: IncomingDataInput, variable_mapping: Optional[dict] = None
) -> IncomingDataInput:
    """Map the variable names to the target data, if required.

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
            v: k for v, k in variable_mapping.items() if v in data.keys()
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


def _maybe_convert_dataset_lead_time_to_timedelta(
    dataset: xr.Dataset,
    lead_time_variable: str = "lead_time",
    start_hour: int = 0,
) -> xr.Dataset:
    """Convert types of variables in an xarray Dataset based on the schema,
    ensuring that, for example, the variable representing lead_time is of type int.

    Args:
        dataset: The input xarray Dataset that uses the schema's variable names.
        lead_time_variable: The variable name of the lead time.
        start_hour: The start hour of the lead time.

    Returns:
        An xarray Dataset with adjusted types.
    """

    lead_time = dataset[lead_time_variable]
    if lead_time.dtype == np.dtype("timedelta64[ns]"):
        # already a timedelta, do nothing
        dataset[lead_time_variable] = (lead_time / np.timedelta64(1, "h")).astype(int)
    elif lead_time.dtype == np.dtype("int64"):
        # convert int to timedelta, assuming hours
        # setting dtype to ns prevents xarray warnings
        dataset[lead_time_variable] = (lead_time * np.timedelta64(1, "h")).astype(
            np.dtype("timedelta64[ns]")
        )
    else:
        temporal_resolution_hours = np.squeeze(
            np.unique(np.diff(lead_time.values)) / np.timedelta64(1, "h")
        ).astype(np.dtype("timedelta64[ns]"))
        dataset[lead_time_variable] = np.arange(
            start_hour,
            lead_time.shape[0] * temporal_resolution_hours,
            temporal_resolution_hours,
        )
    return dataset


def run_pipeline(
    input_data: InputBase,
    case_operator: CaseOperator,
    **kwargs,
) -> xr.Dataset:
    """
    Shared method for running the target pipeline.

    Args:
        input_data: The input data to run the pipeline on.
        case_operator: The case operator to run the pipeline on.
        **kwargs: Additional keyword arguments to pass in as needed.

    Returns:
        The target data with a type determined by the user.
    """

    # Open data and process through pipeline steps
    data = (
        # opens data from user-defined source
        input_data.open_and_maybe_preprocess_data_from_source()
        # maps variable names to the target data if not already using EWB naming conventions
        .pipe(
            maybe_map_variable_names,
            variable_mapping=input_data.variable_mapping,
        )
        # subsets the target data using the caseoperator metadata
        .pipe(
            input_data.subset_data_to_case,
            case_operator=case_operator,
        )
        # converts the target data to an xarray dataset if it is not already
        .pipe(input_data.maybe_convert_to_dataset)
        # derives variables from the target data if derived variables are defined
        .pipe(maybe_derive_variables, variables=input_data.variables)
        .pipe(input_data.add_source_to_dataset_attrs)
    )
    return data


def determine_timesteps_per_day_resolution(
    ds: xr.Dataset | xr.DataArray,
) -> int:
    """Determine the number of timesteps per day for a dataset.

    Args:
        ds: The input dataset with a valid_time dimension or coordinate.

    Returns:
        The number of timesteps per day.
    """
    num_timesteps = 24 // np.unique(np.diff(ds.valid_time)).astype(
        "timedelta64[h]"
    ).astype(int)
    if len(num_timesteps) > 1:
        raise ValueError(
            "The number of timesteps per day is not consistent in the dataset."
        )
    return num_timesteps[0]
