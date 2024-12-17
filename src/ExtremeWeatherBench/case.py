"""Contains the IndividualCase class for metadata of a single case."""
import dataclasses
from collections import namedtuple
import datetime

Location = namedtuple('Location', ['latitude', 'longitude'])

@dataclasses.dataclass
class IndividualCase:
    """
    IndividualCase holds the metadata for a single case
    based on the events.yaml metadata.
    Attributes:
        id: the numerical identifier for the event
        start_date: datetime.date: the start date of the case
        end_date: datetime.date: the end date of the case
        location: Location: the latitude and longitude of the center of the location
        bounding_box_km: int: the side length of the square in kilometers
    """

    id: int
    start_date: datetime.date
    end_date: datetime.date
    location: Location
    bounding_box_km: int