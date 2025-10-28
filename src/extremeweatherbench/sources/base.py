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
        optional_variables: list[str],
        optional_variables_mapping: dict[str, list[str]],
    ) -> Any:
        """This function handles variable extraction from a Pandas DataFrame, supporting
        both required and optional variables. Optional variables can replace required
        ones based on the provided mapping, allowing for flexible data processing.

        Args:
            data: The data to extract variables from.
            variables: List of required variable names to extract. These must be
                present in the data unless replaced by optional variables.
            optional_variables: List of optional variable names to extract. These
                are only included if present in the data.
            optional_variables_mapping: Dictionary mapping optional variable names
                to the required variables they can replace. Keys are optional
                variable names, values can be a single string or list of strings
                representing the required variables to replace.

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
