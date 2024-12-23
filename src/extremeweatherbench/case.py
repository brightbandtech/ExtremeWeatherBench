"""Utiltiies for defining individual units of case studies for analysis."""

import dataclasses
import datetime
from extremeweatherbench.utils import Location


@dataclasses.dataclass
class IndividualCase:
    """Container for metadata defining a single or individual case.

    An IndividualCase defines the relevant metadata for a single case study for a
    given extreme weather event; it is designed to be easily instantiable through a
    simple YAML-based configuration file.

    Attributes:
        id: A unique numerical identifier for the event.
        start_date: A datetime.date object representing the start date of the case, for
            use in subsetting data for analysis.
        end_date: A datetime.date object representing the end date of the case, for use
            in subsetting data for analysis.
        location: A Location object representing the latitude and longitude of the event
            center or  focus.
        bounding_box_km: int: The side length of a bounding box centered on location, in
            kilometers.
    """

    id: int
    start_date: datetime.date
    end_date: datetime.date
    location: Location
    bounding_box_km: int
