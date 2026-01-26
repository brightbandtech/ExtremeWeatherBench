import datetime
from typing import Any, Protocol, runtime_checkable

import extremeweatherbench.regions as regions


@runtime_checkable
class Source(Protocol):
    """Protocol defining the interface for input data sources.

    This protocol specifies the methods that input source implementations must
    provide for variable extraction, temporal validation, and spatial data
    checking.

    Required methods:
        safely_pull_variables: Extract specified variables from data
        check_for_valid_times: Check if data has valid times in date range
        check_for_spatial_data: Check if data has spatial coverage for region
    """

    def safely_pull_variables(
        self,
        data: Any,
        variables: list[str],
    ) -> Any:
        """This function handles variable extraction from a Pandas DataFrame, supporting
        both required and optional variables. Optional variables can replace required
        ones based on the provided mapping, allowing for flexible data processing.

        Args:
            data: The data to extract variables from.
            variables: List of required variable names to extract. These must be
                present in the data.

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
