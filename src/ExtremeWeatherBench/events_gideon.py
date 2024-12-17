import dataclasses

@dataclasses.dataclass
class Location:
    latitude: float
    longitude: float

    def __post_init__(self):
        # TODO: validate latitude and longitude
        pass


class Event:
    def __init__(self):
        pass



@dataclasses.dataclass
class HeatWaveConfig:
    id: int
    title: str
    start_date: str
    end_date: str
    location: Location
    bounding_box_km: int

    def build(self) -> Event:
        Event()


@dataclasses.dataclass
class FreezeConfig:
    id: int
    title: str
    start_date: str
    end_date: str
    location: Location
    bounding_box_km: int


@dataclasses.dataclass
class EventConfig:
    events: list[HeatWaveConfig | FreezeConfig]
