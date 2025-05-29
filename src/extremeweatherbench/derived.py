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
    Attributes:
        name: The name of the derived variable.
        description: The description of the derived variable.
        formula: The formula for the derived variable.
    """

    def __init__(self, name: str, description: str, formula: str):
        self.name = name
        self.description = description
        self.formula = formula

    @property
    def name(self) -> str:
        """The name of the derived variable."""
        return self._name

    @abc.abstractmethod
    def calculate(self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Calculate the derived variable for the given data.
        Args:
            data: The data to calculate the derived variable for.
        Returns:
            The derived variable.
        """
