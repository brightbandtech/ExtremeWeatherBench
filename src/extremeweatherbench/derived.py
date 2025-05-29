"""
This module contains functions for calculating derived variables for event types.
"""

import abc
import logging

import numpy as np
import xarray as xr

np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DerivedVariable(abc.ABC):
    """A class to hold a derived variable for an event type.

    Derived variables are not variables that exist in the original dataset, but can be
    calculated from the original variables. Certain event types may require derived
    outputs such as the Craven-Brooks Significant Severe parameter, a variable that is
    calculated based on temperature, dewpoint, pressure coordinates, and bulk
    0-6km wind shear.

    Attributes:
        name: The name of the derived variable.

    Methods:
        calculate: Calculate the derived variable for the given data.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """Return the class name without parentheses."""
        return self.__class__.__name__

    @abc.abstractmethod
    def calculate(self, data: xr.Dataset | xr.DataArray) -> xr.DataArray:
        """Calculate the derived variable for the given data.

        Args:
            data: The data to calculate the derived variable for.
        Returns:
            The derived variable as a DataArray.
        """
