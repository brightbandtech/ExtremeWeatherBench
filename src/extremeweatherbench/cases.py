"""Classes for defining individual units of case studies for analysis.

Some code similarly structured to WeatherBenchX (Rasp et al.).
"""

import dataclasses
import datetime
import importlib.resources
import itertools
import logging
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import dacite
import yaml  # type: ignore[import]

from extremeweatherbench import regions

if TYPE_CHECKING:
    from extremeweatherbench import inputs, metrics

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single case study.

    Attributes:
        case_id_number: Unique numerical identifier for the event.
        title: Title of the case study.
        start_date: Start date for subsetting data for analysis.
        end_date: End date for subsetting data for analysis.
        location: Region object representing the case location.
        event_type: String representing the type of extreme weather event.
    """

    case_id_number: int
    title: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    location: "regions.Region"
    event_type: str


@dataclasses.dataclass
class CaseOperator:
    """Operator dataclass for an evaluation of a single evaluation object.

    Attributes:
        case_metadata: IndividualCase metadata for this operator.
        metric_list: List of metrics to evaluate for this case.
        target: TargetBase object for ground truth data.
        forecast: ForecastBase object for forecast data.
    """

    case_metadata: IndividualCase
    metric_list: Sequence["metrics.BaseMetric"]
    target: "inputs.TargetBase"
    forecast: "inputs.ForecastBase"


def _read_incoming_yaml(input_pth: str | pathlib.Path) -> Any:
    """Read a case metadata yaml file with yaml.safe_load."""
    with open(input_pth, "rb") as f:
        return yaml.safe_load(f)


def build_case_operators(
    case_list: list[IndividualCase],
    evaluation_objects: list["inputs.EvaluationObject"],
) -> list[CaseOperator]:
    """Build a CaseOperator from the case metadata and metric evaluation objects.

    Args:
        case_list: List of IndividualCase objects defining cases to process.
        evaluation_objects: The evaluation objects to apply to the case operators.

    Returns:
        A list of CaseOperator objects.
    """
    # build list of case operators based on information provided in case dict and
    case_operators = []
    for single_case, evaluation_object in itertools.product(
        case_list, evaluation_objects
    ):
        # checks if case matches the event type provided in metric eval object
        if single_case.event_type in evaluation_object.event_type:
            case_operators.append(
                CaseOperator(
                    case_metadata=single_case,
                    metric_list=evaluation_object.metric_list,
                    target=evaluation_object.target,
                    forecast=evaluation_object.forecast,
                )
            )
    return case_operators


def load_individual_cases_from_dict(
    cases: list[dict[str, Any]] | list[IndividualCase],
) -> list[IndividualCase]:
    """Convert case metadata dicts to IndividualCase objects.

    IndividualCase objects in the input are passed through unchanged.

    Args:
        cases: A list of cases as either dicts or IndividualCase objects.

    Returns:
        A list of IndividualCase objects.
    """
    config = dacite.Config(type_hooks={regions.Region: regions.map_to_create_region})
    return [
        case
        if isinstance(case, IndividualCase)
        else dacite.from_dict(data_class=IndividualCase, data=case, config=config)
        for case in cases
    ]


def load_individual_cases_from_yaml(
    yaml_file: str | pathlib.Path,
) -> list[IndividualCase]:
    """Load IndividualCase metadata from your own yaml file.

    The file must be a list of cases in the same format as the bundled
    events.yaml; dacite raises if a case does not match IndividualCase.

    Example yaml file:

    ```yaml
    - case_id_number: 1
      title: Event 1
      start_date: 2021-01-01 00:00:00
      end_date: 2021-01-03 00:00:00
      location:
        type: bounded_region
        parameters:
          latitude_min: 10.0
          latitude_max: 55.6
          longitude_min: 265.0
          longitude_max: 283.3
      event_type: tropical_cyclone
    ```

    Args:
        yaml_file: A path to a yaml file containing the case metadata.

    Returns:
        A list of IndividualCase objects.
    """
    return load_individual_cases_from_dict(_read_incoming_yaml(yaml_file))


def load_ewb_cases() -> list[IndividualCase]:
    """Load the cases bundled with EWB (data/events.yaml).

    ``load_cases`` is an alias for this function.
    """
    events_yaml = importlib.resources.files("extremeweatherbench.data") / "events.yaml"
    with importlib.resources.as_file(events_yaml) as path:
        return load_individual_cases_from_yaml(path)


load_cases = load_ewb_cases
