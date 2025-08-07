import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

from extremeweatherbench import utils

if TYPE_CHECKING:
    from extremeweatherbench import derived, inputs, metrics


@dataclasses.dataclass
class TargetConfig:
    target: "inputs.TargetBase"
    source: str | Path
    variables: list[Union[str, "derived.DerivedVariable"]]
    variable_mapping: dict
    storage_options: dict
    preprocess: Callable = utils._default_preprocess


@dataclasses.dataclass
class ForecastConfig:
    forecast: "inputs.ForecastBase"
    source: Union[str, Path]
    variables: list[Union[str, "derived.DerivedVariable"]]
    variable_mapping: dict
    storage_options: dict
    preprocess: Callable = utils._default_preprocess


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
    metric: list["metrics.BaseMetric"]
    target_config: TargetConfig
    forecast_config: ForecastConfig
