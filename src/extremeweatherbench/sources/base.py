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
        alternative_variables: dict[str, list[str]],
        optional_variables: list[str],
    ) -> Any:
        """Extract variables from data with alternative and optional support.

        This function handles variable extraction from data, supporting both
        required and optional variables. Alternative variables can replace
        required ones when the required variable is missing.

        Args:
            data: The data to extract variables from.
            variables: List of required variable names to extract. These must
                be present in the data unless replaced by alternatives.
            alternative_variables: Dictionary mapping required variable names
                to lists of alternative variables that can replace them. Keys
                are required variable names, values are lists of alternative
                variable names that must all be present together.
            optional_variables: List of optional variable names to extract.
                These are only included if present in the data.

        Returns:
            Data containing only the identified variables.
        """
        pass

    def check_for_valid_times(
        self, data: Any, start_date: datetime.datetime, end_date: datetime.datetime
    ) -> bool:
        """Check if the data has valid times in the given date range.

        Args:
            data: The data to check for valid times.
            start_date: The start date of the time range to check.
            end_date: The end date of the time range to check.

        Returns:
            True if the data has any times within the specified range,
            False otherwise.
        """
        pass

    def check_for_spatial_data(self, data: Any, location: "regions.Region") -> bool:
        """Check if the data has spatial data for the given location.

        Args:
            data: The data to check for spatial data.
            location: The region to check for spatial overlap.

        Returns:
            True if the data has any data within the specified region,
            False otherwise.
        """
        pass
