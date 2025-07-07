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
# that are needed for an event type.
# The intent of using this translation layer is to enable a flexible and user-friendly API for new
# event types, in the face of significantly variable observation data.


import typing
from abc import ABC, abstractmethod
from collections import OrderedDict

import xarray as xr


class DerivedVariable(ABC):
    """
    A derived variable is a variable that is derived from other variables.
    """

    def __init__(self, name: str, variables: typing.List[str]):
        self.name = name
        self.variables = variables

    def generate(self) -> xr.Dataset:
        """
        Generate the derived variable from the variables.
        """
        pass


class Observation(ABC):
    """
    An observation is a class that defines the observations that are needed for an event type.

    Observations are loosely defined objects that can come in as tabular or gridded data and are
    eventually output as an xarray dataset. Some observations require minimal processing, e.g.
    ERA5 2m temperature, whereas others require more coaxing, such as IBTrACS data or local storm
    reports. Observations for an event type are defined in the events.yaml file. Child classes
    will define the function to ingest and transform the data as needed through the generate() method.
    """

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

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
    ):
        self.name = name
        self.variables = variables

    @abstractmethod
    def compute_metric(
        self, observations: xr.DataArray, forecasts: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute the metric from the observations.
        """
        pass


class EventTypeOperator(ABC):
    """
    The EventTypeOperator is a class that defines the operations that are needed for an event type.

    The EventTypeOperator acts as the "recipe book" to develop an acyclic graph based on. Specifically,
    the metrics define the variables, whether existing or derived, that are needed from the observations and
    forecasts. The Observations are a list of one or more observation sources that will be ingested for the
    operations defined in the metrics, which can be anything from a gridded dataset to a tabular format
    with time series of specific variables and point locations.

    Attributes:
        observations: A list of observations that are used to generate the event type.
        metrics: A list of metrics that are used to evaluate the performance of the event type.


        An event type can have multiple sources of observations, such as ERA5 and GHCN, or ERA5 and IBTrACS.

        forecasts: A forecast dataset that is used to evaluate the performance of the event type.
    """

    def __init__(
        self,
        observations: typing.List[Observation],
        variables: typing.List[str | DerivedVariable],
        metrics: typing.List[Metric],
    ):
        self.observations = observations
        self.variables = variables
        self.metrics = metrics


class TranslationLayer:
    """
    The TranslationLayer is a class that defines the translation layer for an event type.
    """

    def __init__(self, event: EventTypeOperator):
        self.event = event

    def generate_graph(self) -> OrderedDict:
        """
        Generate the graph of the translation layer using the event type operator.
        """
        graph = OrderedDict()

        # prep functions to create derived variables if there are any in the dataset
        graph["variables"] = ...

        # prep functions to convert observations to xr.Datasets following desired schema
        graph["observations"] = ...

        # prep functions to compute metrics from observations and forecasts
        graph["metrics"] = ...

        return graph
