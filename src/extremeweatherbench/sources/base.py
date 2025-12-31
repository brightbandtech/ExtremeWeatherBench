import datetime
from typing import Any, Protocol, runtime_checkable

from extremeweatherbench import regions


@runtime_checkable
class Source(Protocol):
    """A protocol for input sources."""

    def safely_pull_variables(
        self,
        data: Any,
        variables: list[str],
    ) -> Any: ...

    def check_for_valid_times(
        self, data: Any, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> bool: ...

    def check_for_spatial_data(self, data: Any, location: "regions.Region") -> bool: ...

    def safe_concat(self, data_objects: list[Any]) -> Any: ...

    def ensure_output_schema(self, data: Any, **metadata) -> Any: ...
