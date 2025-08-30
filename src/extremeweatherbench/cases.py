"""Classes for defining individual units of case studies for analysis.

Some code similarly structured to WeatherBenchX (Rasp et al.).
"""

import dataclasses
import datetime
import itertools
import logging
from typing import TYPE_CHECKING, Callable, Union

import dacite

from extremeweatherbench import regions

if TYPE_CHECKING:
    from extremeweatherbench import inputs, metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        case_metadata: IndividualCase metadata
        metric_list: A list of metrics that are intended to be evaluated for the case
        target_config: A TargetConfig object
        forecast_config: A ForecastConfig object
    """

    case_metadata: IndividualCase
    metric_list: list[Union[Callable, "metrics.BaseMetric", "metrics.AppliedMetric"]]
    target: "inputs.TargetBase"
    forecast: "inputs.ForecastBase"


def build_case_operators(
    cases_dict: dict[str, list],
    metric_evaluation_objects: list["inputs.EvaluationObject"],
) -> list[CaseOperator]:
    """Build a CaseOperator from the case metadata and metric evaluation objects.

    Args:
        cases_dict: The case metadata to use for the case operators.
        metric_evaluation_objects: The metric evaluation objects to use for the
            case operators.

    Returns:
        A list of CaseOperator objects.
    """
    if isinstance(cases_dict["cases"], list):
        case_metadata_collection = dacite.from_dict(
            data_class=IndividualCaseCollection,
            data=cases_dict,
            config=dacite.Config(
                type_hooks={regions.Region: regions.map_to_create_region},
            ),
        )
    else:
        raise TypeError("cases_dict['cases'] must be a list of cases")

    # build list of case operators based on information provided in case dict and
    case_operators = []
    for single_case, metric_evaluation_object in itertools.product(
        case_metadata_collection.cases, metric_evaluation_objects
    ):
        # checks if case matches the event type provided in metric eval object
        if single_case.event_type in metric_evaluation_object.event_type:
            case_operators.append(
                CaseOperator(
                    case_metadata=single_case,
                    metric_list=metric_evaluation_object.metric_list,
                    target=metric_evaluation_object.target,
                    forecast=metric_evaluation_object.forecast,
                )
            )
    return case_operators


def load_individual_cases(cases: dict[str, list]):
    """Load IndividualCase metadata from a dictionary.

    Args:
        cases: A dictionary of cases based on the IndividualCase dataclass.

    Returns:
        A list of IndividualCase objects.
    """
    case_metadata_collection = dacite.from_dict(
        data_class=IndividualCaseCollection,
        data=cases,
        config=dacite.Config(
            type_hooks={regions.Region: regions.map_to_create_region},
        ),
    )

    return case_metadata_collection
