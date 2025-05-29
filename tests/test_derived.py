"""Tests for the derived variables module."""

import pytest
import xarray as xr
import numpy as np
from extremeweatherbench.derived import DerivedVariable
from extremeweatherbench.case import IndividualCase
import datetime


class TestDerivedVariable(DerivedVariable):
    """A concrete implementation of DerivedVariable for testing."""

    def calculate(self, data: xr.Dataset | xr.DataArray) -> xr.DataArray:
        """Calculate a simple derived variable for testing."""
        if isinstance(data, xr.Dataset):
            return data["var1"] * 2
        return data * 2


def test_derived_variable_initialization():
    """Test that DerivedVariable uses class name as the name property."""
    var = TestDerivedVariable()
    assert var.name == "TestDerivedVariable"


def test_derived_variable_abstract_method():
    """Test that DerivedVariable cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DerivedVariable()


def test_derived_variable_calculate():
    """Test that the calculate method works with both Dataset and DataArray inputs."""
    var = TestDerivedVariable()

    # Test with DataArray
    data_array = xr.DataArray(np.array([1, 2, 3]), name="var1")
    result = var.calculate(data_array)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values, np.array([2, 4, 6]))

    # Test with Dataset
    dataset = xr.Dataset({"var1": data_array})
    result = var.calculate(dataset)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values, np.array([2, 4, 6]))


def test_calculate_derived_variables():
    """Test the _calculate_derived_variables method in IndividualCase."""
    # Create a test case
    case = IndividualCase(
        id=1,
        title="Test Case",
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 1, 2),
        location=None,  # Not needed for this test
        bounding_box_degrees=1.0,
        event_type="test",
        data_vars=[TestDerivedVariable()],
    )

    # Create test data
    data = xr.Dataset({"var1": xr.DataArray(np.array([1, 2, 3]), name="var1")})

    # Calculate derived variables
    result = case._calculate_derived_variables(data)

    # Check that the derived variable was added
    assert "TestDerivedVariable" in result
    np.testing.assert_array_equal(
        result["TestDerivedVariable"].values, np.array([2, 4, 6])
    )


def test_calculate_derived_variables_no_derived():
    """Test _calculate_derived_variables with no derived variables."""
    # Create a test case with regular variables
    case = IndividualCase(
        id=1,
        title="Test Case",
        start_date=datetime.datetime(2023, 1, 1),
        end_date=datetime.datetime(2023, 1, 2),
        location=None,  # Not needed for this test
        bounding_box_degrees=1.0,
        event_type="test",
        data_vars=["var1", "var2"],
    )

    # Create test data
    data = xr.Dataset(
        {
            "var1": xr.DataArray(np.array([1, 2, 3]), name="var1"),
            "var2": xr.DataArray(np.array([4, 5, 6]), name="var2"),
        }
    )

    # Calculate derived variables
    result = case._calculate_derived_variables(data)

    # Check that the dataset is unchanged
    assert set(result.data_vars) == {"var1", "var2"}
    np.testing.assert_array_equal(result["var1"].values, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result["var2"].values, np.array([4, 5, 6]))
