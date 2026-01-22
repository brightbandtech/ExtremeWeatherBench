"""Classes for defining individual units of case studies for analysis.

Some code similarly structured to WeatherBenchX (Rasp et al.).
"""

import dataclasses
import datetime
import importlib
import itertools
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Sequence, Union

import dacite
import yaml  # type: ignore[import]

from extremeweatherbench import regions

if TYPE_CHECKING:
    from extremeweatherbench import inputs, metrics

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single extreme weather case study.

    Defines relevant metadata for a single case study of an extreme weather
    event. Designed for easy instantiation through simple YAML configuration
    files.

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
    """Operator storing the processing graph for an individual case.

    Serves as a one-stop-shop for evaluating a single case. Multiple
    CaseOperators can run in parallel for multiple cases, or serially through
    ExtremeWeatherBench.run().

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


def build_case_operators(
    case_list: list[IndividualCase],
    evaluation_objects: list["inputs.EvaluationObject"],
) -> list[CaseOperator]:
    """Build a CaseOperator from the case metadata and metric evaluation objects.

    Args:
        cases: The case metadata to use for the case operators as a dictionary of cases
            or a list of IndividualCases.
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


def load_individual_cases(
    cases: Union[list[dict[str, Any]], list[IndividualCase]],
) -> list[IndividualCase]:
    """Load IndividualCase metadata from a dictionary.

    Will pass through existing IndividualCase objects and convert dictionaries to IndividualCase objects.

    Args:
        cases: A dictionary of cases based on the IndividualCase dataclass.

    Returns:
        A list of IndividualCase objects.
    """

    # Iterate through the cases and convert dictionaries to IndividualCase objects if
    # they are not already IndividualCase objects
    case_list = [
        case
        if isinstance(case, IndividualCase)
        else dacite.from_dict(
            data_class=IndividualCase,
            data=case,
            config=dacite.Config(
                type_hooks={regions.Region: regions.map_to_create_region},
            ),
        )
        for case in cases
    ]

    return case_list


def load_individual_cases_from_yaml(
    yaml_file: Union[str, pathlib.Path],
) -> list[IndividualCase]:
    """Load IndividualCase metadata directly from a yaml file.

    This function is a wrapper around load_individual_cases that reads the yaml file
    directly. It is useful for loading cases from a yaml file that is not part of the
    ExtremeWeatherBench data package. Note that the yaml file must be in the same format
    as described in the ExtremeWeatherBench documentation; errors will be raised within
    the dacite.from_dict call if the yaml file otherwise.

    Example of a yaml file:

    ```yaml
    cases:
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
    yaml_event_case = read_incoming_yaml(yaml_file)
    return load_individual_cases(yaml_event_case)


def load_ewb_events_yaml_into_case_list() -> list[IndividualCase]:
    """Loads the EWB events yaml file into a list of IndividualCase objects."""
    import extremeweatherbench.data

    events_yaml_file = importlib.resources.files(extremeweatherbench.data).joinpath(
        "events.yaml"
    )
    with importlib.resources.as_file(events_yaml_file) as file:
        yaml_event_case = read_incoming_yaml(file)

    return load_individual_cases(yaml_event_case)


def read_incoming_yaml(input_pth: Union[str, pathlib.Path]):
    """Read events yaml from data into a dictionary.

    This function is a wrapper around yaml.safe_load that reads the yaml file directly.
    It is useful for reading yaml files other than the EWB events.yaml file.

    Args:
        input_pth: A path to a yaml file containing the case metadata.

    Returns:
        A dictionary of case metadata.
    """
    input_pth = pathlib.Path(input_pth)
    with open(input_pth, "rb") as f:
        yaml_event_case = yaml.safe_load(f)
    return yaml_event_case
