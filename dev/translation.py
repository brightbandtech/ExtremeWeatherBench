# This file is used to define the translation layers for event types (dummy version).
# The overall flow of the translation layer is:
# 1. Define the event type operator with observations, metrics, and forecasts
# 1a. Each of the three have "instructions" that define the translations needed for the graph
# 2. Instantiate the translation layer
# 3. Generate the graph from the information in the event type operator
# 4. Run the translation layer, outputting an object defined by the event type operator to be fed into metrics
# 5. ???
# 6. Profit!
# The translation layer is a pipelining passthrough class for data to be defined as needed for an event.
#
# It is used to define an acylic graph of transformations, derivations, or other data manipulations
# that are needed for an event type. Some event types will need multiple layers of translation, and
# some will need no translation at all.
# The intent of using this translation layer is to enable a flexible and user-friendly API for new
# event types, in the face of significantly variable observation data.

import dataclasses
import typing
from abc import ABC, abstractmethod
from collections import OrderedDict

import xarray as xr
import yaml


class DerivedVariable(ABC):
    """
    A derived variable is a variable that is derived from other variables.
    """

    def __init__(self, name: str, variables: typing.List[str]):
        self.name = name
        self.variables = variables

    @property
    def name(self) -> str:
        """
        Get the name of the derived variable.
        """
        return self.name

    @abstractmethod
    def generate(self, data: xr.Dataset) -> xr.Dataset:
        """
        Generate a derived variable from the data.

        Args:
            data: The data to generate the derived variable from.

        Returns:
            The derived variable.
        """
        # do something with data + self.variables
        pass


class Observation(ABC):
    """
    An observation is a class that defines the observations that are needed for an event type.
    Observations are loosely defined objects that can come in as tabular or gridded data and are
    eventually output as an xarray dataset. Some observations require minimal processing, e.g.
    ERA5 2m temperature, whereas others require more coaxing, such as IBTrACS data or local storm
    reports.

    Observations for an event type are defined in the events.yaml file.
    """

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

    @abstractmethod
    def generate(self) -> xr.Dataset:
        """
        Generate the observation from the path.
        """
        pass


class Metric:
    """
    A metric is a function that is used to evaluate the performance of a model (Exists in EWB already,
    this is a placeholder for now)
    """

    def __init__(
        self,
        name: str,
        variables: typing.List[str | DerivedVariable],
        observations: typing.List[Observation],
        threshold: typing.Optional[float] = None,
    ):
        self.name = name
        self.variables = variables

        if threshold is not None:
            self.threshold = threshold

    @abstractmethod
    def compute_metric(
        self, observations: xr.DataArray, forecasts: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the metric from the observations.
        """
        pass


@dataclasses.dataclass
class EventTypeOperator:
    """
    The EventTypeOperator is a class that defines the operations that are needed for an event type.

    The EventTypeOperator acts as the "recipe book" to develop an acyclic graph based on. Specifically,
    the metrics define the variables, whether existing or derived, that are needed from the observations and
    forecasts. The Observations are a list of one or more observation sources that will be ingested for the
    operations defined in the metrics, which can be anything from a gridded dataset to a tabular format
    with time series of specific variables and point locations.

    Attributes:
        variables: A list of variables (strings or DerivedVariables) that are used for the event type.
        metrics: A list of metrics that are used to evaluate the performance of the event type.
        observations: A list of observations that are used to generate the event type.
        An event type can have multiple sources of observations, such as ERA5 and GHCN, or ERA5 and IBTrACS.
    """

    variables: typing.List[str | DerivedVariable]
    metrics: typing.List[Metric]
    observations: typing.List[Observation]

    @abstractmethod
    def output(self) -> xr.Dataset:
        """
        Define the output schema for the event type.
        """
        pass


class TranslationLayer:
    """
    The translation layer is a pipelining passthrough class for data to be defined as needed for an event.

    It is used to define and execute an acylic graph of transformations, derivations, or other data manipulations
    that are needed for an event type. Some event types will need multiple layers of translation, and
    some will need no translation at all.

    The translation layer is instantiated with an event type operator and a function that generates
    the graph of the event type operator. The graph is then executed to translate the data based on the
    event type operator.

    Attributes:
        event: The event type operator.
        graph_function: A function that generates the graph; build_graph is supplied as what should be the
        composed function.
    """

    def __init__(
        self,
        event: EventTypeOperator,
        forecasts: xr.DataTree,
    ):
        self.event = event
        self.forecasts = forecasts

    def compose_graph(self) -> OrderedDict:
        """
        Generate the graph of the translation layer.
        """
        graph = OrderedDict()
        for observation in self.event.observations:
            graph[observation.name] = observation

        for metric in self.event.metrics:
            graph[metric.name] = metric.generate()
        return graph

    def execute_graph(self) -> OrderedDict:
        """
        Execute the graph of the translation layer.
        """
        pass


# Example workflow

events_yaml = yaml.safe_load(open("events.yaml"))


# Create a class for each observation type; each needs its own processing logic using generate()
class IBTrACS(Observation):
    def __init__(self, path: str):
        super().__init__(path, "IBTrACS")

    def generate(self) -> xr.Dataset:
        # do things to generate the observation
        pass


class ERA5(Observation):
    def __init__(self, path: str):
        super().__init__(path, "ERA5")

    def generate(self) -> xr.Dataset:
        # do things to generate the observation
        pass


# Create a class for each metric type;
class SpatialLandfallError(Metric):
    def __init__(
        self,
        name: str,
        variables: typing.List[str | DerivedVariable],
        observations: typing.List[Observation],
    ):
        super().__init__(name, variables)

    def generate(self) -> xr.Dataset:
        # do things to generate the metric
        pass


class LandfallIntensityMAE(Metric):
    def __init__(
        self,
        name: str,
        variables: typing.List[str | DerivedVariable],
        threshold: typing.Optional[float] = None,
    ):
        super().__init__(name, variables, threshold)

    def generate(self) -> xr.Dataset:
        # do things to generate the metric
        pass


class TropicalCyclone(EventTypeOperator):
    def __init__(
        self,
        metrics: typing.List[Metric],
        observations: typing.List[Observation],
        forecasts: xr.DataTree,  # note that this is a datatree, not a dataset
    ):
        super().__init__(
            metrics=metrics,
            observations=observations,
            forecasts=forecasts,
        )
        self.dataset_schema = {"time": [], "location": []}

    def output(self) -> xr.Dataset:
        """
        Returns an xr.Dataset with dimensions 'time' and 'location'.
        """
        ds = xr.Dataset(coords=self.dataset_schema)
        return ds


dummy_metrics = [
    SpatialLandfallError(name="spatial_landfall_error", variables=["IBTrACS"]),
    LandfallIntensityMAE(name="landfall_intensity_mae", variables=["IBTrACS"]),
]
dummy_observations = [IBTrACS(path="/path/to/IBTrACS.parq")]
dummy_forecasts = xr.Dataset()

dummy_event = TropicalCyclone(
    metrics=dummy_metrics, observations=dummy_observations, forecasts=dummy_forecasts
)

translation_layer = TranslationLayer(
    event=dummy_event,
    forecasts=dummy_forecasts,
)
graph = translation_layer.compose_graph()
