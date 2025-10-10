"""Tests for extremeweatherbench.sources modules."""

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench.sources.pandas_dataframe import (
    check_for_spatial_data_pandas_dataframe,
    check_for_valid_times_pandas_dataframe,
    safely_pull_variables_pandas_dataframe,
)
from extremeweatherbench.sources.polars_lazyframe import (
    check_for_spatial_data_polars_lazyframe,
    check_for_valid_times_polars_lazyframe,
    safely_pull_variables_polars_lazyframe,
)
from extremeweatherbench.sources.xarray_dataarray import (
    check_for_spatial_data_xr_dataarray,
    check_for_valid_times_xr_dataarray,
    safely_pull_variables_xr_dataarray,
)
from extremeweatherbench.sources.xarray_dataset import (
    check_for_spatial_data_xr_dataset,
    check_for_valid_times_xr_dataset,
    safely_pull_variables_xr_dataset,
)


class TestPandasDataFrameModule:
    """Tests for pandas_dataframe module."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = {
            "valid_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "temperature": np.random.randn(10),
            "pressure": np.random.randn(10),
            "humidity": np.random.randn(10),
            "wind_speed": np.random.randn(10),
        }
        return pd.DataFrame(data)

    def test_safely_pull_variables_success(self, sample_dataframe):
        """Test successful variable extraction."""
        variables = ["temperature", "pressure"]
        optional_variables = []
        optional_variables_mapping = {}

        result = safely_pull_variables_pandas_dataframe(
            sample_dataframe, variables, optional_variables, optional_variables_mapping
        )

        assert list(result.columns) == variables
        assert len(result) == len(sample_dataframe)

    def test_safely_pull_variables_with_optional(self, sample_dataframe):
        """Test variable extraction with optional variables."""
        variables = ["temperature"]
        optional_variables = ["humidity", "nonexistent"]
        optional_variables_mapping = {}

        result = safely_pull_variables_pandas_dataframe(
            sample_dataframe, variables, optional_variables, optional_variables_mapping
        )

        expected_columns = ["humidity", "temperature"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping(self, sample_dataframe):
        """Test variable extraction with optional variable mapping."""
        variables = ["temp", "press"]
        optional_variables = ["temperature", "pressure"]
        optional_variables_mapping = {
            "temperature": "temp",
            "pressure": "press",
        }

        result = safely_pull_variables_pandas_dataframe(
            sample_dataframe, variables, optional_variables, optional_variables_mapping
        )

        expected_columns = ["temperature", "pressure"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping_list(self, sample_dataframe):
        """Test variable extraction with optional variable mapping as list."""
        variables = ["temp", "press"]
        optional_variables = ["wind_speed"]
        optional_variables_mapping = {
            "wind_speed": ["temp", "press"],
        }

        result = safely_pull_variables_pandas_dataframe(
            sample_dataframe, variables, optional_variables, optional_variables_mapping
        )

        expected_columns = ["wind_speed"]
        assert list(result.columns) == expected_columns

    def test_safely_pull_variables_missing_required(self, sample_dataframe):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]
        optional_variables = []
        optional_variables_mapping = {}

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_pandas_dataframe(
                sample_dataframe,
                variables,
                optional_variables,
                optional_variables_mapping,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = check_for_valid_times_pandas_dataframe(df, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_one_date_missing(self):
        """Test check_for_valid_times when one date is not in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 10)  # Outside range

        result = check_for_valid_times_pandas_dataframe(df, start_date, end_date)
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self):
        """Test check_for_valid_times when neither date is in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = check_for_valid_times_pandas_dataframe(df, start_date, end_date)
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when DataFrame has latitude and longitude columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataFrame with spatial data
        data = {
            "latitude": [40.0, 41.0, 42.0, 43.0],
            "longitude": [-74.0, -73.0, -72.0, -71.0],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pd.DataFrame(data)
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_pandas_dataframe(df, region)
        assert result is True

    def test_check_for_spatial_data_with_lat_lon(self):
        """Test check_for_spatial_data when DataFrame has lat and lon columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataFrame with spatial data using 'lat' and 'lon'
        data = {
            "lat": [40.0, 41.0, 42.0, 43.0],
            "lon": [-74.0, -73.0, -72.0, -71.0],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pd.DataFrame(data)
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_pandas_dataframe(df, region)
        assert result is True

    def test_check_for_spatial_data_no_spatial_columns(self):
        """Test check_for_spatial_data when DataFrame has no spatial columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataFrame without spatial data
        data = {
            "temperature": [20.0, 21.0, 22.0, 23.0],
            "pressure": [1013.0, 1014.0, 1015.0, 1016.0],
        }
        df = pd.DataFrame(data)
        
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_pandas_dataframe(df, region)
        assert result is False

    def test_check_for_spatial_data_no_overlap(self):
        """Test check_for_spatial_data when data is outside region bounds."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataFrame with spatial data outside region
        data = {
            "latitude": [50.0, 51.0, 52.0, 53.0],  # Outside region
            "longitude": [-80.0, -79.0, -78.0, -77.0],  # Outside region
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pd.DataFrame(data)
        
        # Create region that doesn't overlap with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_pandas_dataframe(df, region)
        assert result is False


class TestPolarsLazyFrameModule:
    """Tests for polars_lazyframe module."""

    @pytest.fixture
    def sample_lazyframe(self):
        """Create a sample LazyFrame for testing."""
        data = {
            "valid_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "temperature": np.random.randn(10),
            "pressure": np.random.randn(10),
            "humidity": np.random.randn(10),
            "wind_speed": np.random.randn(10),
        }
        df = pl.DataFrame(data)
        return df.lazy()

    def test_safely_pull_variables_success(self, sample_lazyframe):
        """Test successful variable extraction."""
        variables = ["temperature", "pressure"]
        optional_variables = []
        optional_variables_mapping = {}

        result = safely_pull_variables_polars_lazyframe(
            sample_lazyframe, variables, optional_variables, optional_variables_mapping
        )

        assert sorted(result.columns) == sorted(variables)

    def test_safely_pull_variables_with_optional(self, sample_lazyframe):
        """Test variable extraction with optional variables."""
        variables = ["temperature"]
        optional_variables = ["humidity", "nonexistent"]
        optional_variables_mapping = {}

        result = safely_pull_variables_polars_lazyframe(
            sample_lazyframe, variables, optional_variables, optional_variables_mapping
        )

        expected_columns = ["humidity", "temperature"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping(self, sample_lazyframe):
        """Test variable extraction with optional variable mapping."""
        variables = ["temp", "press"]
        optional_variables = ["temperature", "pressure"]
        optional_variables_mapping = {
            "temperature": "temp",
            "pressure": "press",
        }

        result = safely_pull_variables_polars_lazyframe(
            sample_lazyframe, variables, optional_variables, optional_variables_mapping
        )

        expected_columns = ["temperature", "pressure"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_missing_required(self, sample_lazyframe):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]
        optional_variables = []
        optional_variables_mapping = {}

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_polars_lazyframe(
                sample_lazyframe,
                variables,
                optional_variables,
                optional_variables_mapping,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = check_for_valid_times_polars_lazyframe(lf, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_one_date_missing(self):
        """Test check_for_valid_times when one date is not in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 10)  # Outside range

        result = check_for_valid_times_polars_lazyframe(lf, start_date, end_date)
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self):
        """Test check_for_valid_times when neither date is in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = check_for_valid_times_polars_lazyframe(lf, start_date, end_date)
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when LazyFrame has latitude and longitude columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create LazyFrame with spatial data
        data = {
            "latitude": [40.0, 41.0, 42.0, 43.0],
            "longitude": [-74.0, -73.0, -72.0, -71.0],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_polars_lazyframe(lf, region)
        assert result is True

    def test_check_for_spatial_data_with_lat_lon(self):
        """Test check_for_spatial_data when LazyFrame has lat and lon columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create LazyFrame with spatial data using 'lat' and 'lon'
        data = {
            "lat": [40.0, 41.0, 42.0, 43.0],
            "lon": [-74.0, -73.0, -72.0, -71.0],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_polars_lazyframe(lf, region)
        assert result is True

    def test_check_for_spatial_data_no_spatial_columns(self):
        """Test check_for_spatial_data when LazyFrame has no spatial columns."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create LazyFrame without spatial data
        data = {
            "temperature": [20.0, 21.0, 22.0, 23.0],
            "pressure": [1013.0, 1014.0, 1015.0, 1016.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()
        
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_polars_lazyframe(lf, region)
        assert result is False

    def test_check_for_spatial_data_no_overlap(self):
        """Test check_for_spatial_data when data is outside region bounds."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create LazyFrame with spatial data outside region
        data = {
            "latitude": [50.0, 51.0, 52.0, 53.0],  # Outside region
            "longitude": [-80.0, -79.0, -78.0, -77.0],  # Outside region
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()
        
        # Create region that doesn't overlap with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_polars_lazyframe(lf, region)
        assert result is False


class TestXarrayDataArrayModule:
    """Tests for xarray_dataarray module."""

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample DataArray for testing."""
        time = pd.date_range("2021-01-01", periods=10, freq="1D")
        data = np.random.randn(10, 5, 5)
        da = xr.DataArray(
            data,
            dims=["valid_time", "lat", "lon"],
            coords={
                "valid_time": time,
                "lat": range(5),
                "lon": range(5),
            },
            name="temperature",
        )
        return da

    def test_safely_pull_variables_success(self, sample_dataarray):
        """Test successful variable extraction."""
        variables = ["temperature"]
        optional_variables = []
        optional_variables_mapping = {}

        result = safely_pull_variables_xr_dataarray(
            sample_dataarray, variables, optional_variables, optional_variables_mapping
        )

        assert result.name == "temperature"
        assert result.equals(sample_dataarray)

    def test_safely_pull_variables_with_optional(self, sample_dataarray):
        """Test variable extraction with optional variables."""
        variables = ["pressure"]
        optional_variables = ["temperature"]
        optional_variables_mapping = {}

        result = safely_pull_variables_xr_dataarray(
            sample_dataarray, variables, optional_variables, optional_variables_mapping
        )

        assert result.name == "temperature"
        assert result.equals(sample_dataarray)

    def test_safely_pull_variables_missing_required(self, sample_dataarray):
        """Test error when required variables are missing."""
        variables = ["pressure"]
        optional_variables = []
        optional_variables_mapping = {}

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_xr_dataarray(
                sample_dataarray,
                variables,
                optional_variables,
                optional_variables_mapping,
            )

    def test_safely_pull_variables_unnamed_dataarray(self):
        """Test with unnamed DataArray."""
        time = pd.date_range("2021-01-01", periods=5, freq="1D")
        data = np.random.randn(5, 3, 3)
        da = xr.DataArray(
            data,
            dims=["valid_time", "lat", "lon"],
            coords={
                "valid_time": time,
                "lat": range(3),
                "lon": range(3),
            },
        )

        variables = ["unnamed"]
        optional_variables = []
        optional_variables_mapping = {}

        result = safely_pull_variables_xr_dataarray(
            da, variables, optional_variables, optional_variables_mapping
        )

        assert result.equals(da)

    def test_check_for_valid_times_both_dates_in_dataset(self, sample_dataarray):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = check_for_valid_times_xr_dataarray(
            sample_dataarray, start_date, end_date
        )
        assert result is True

    def test_check_for_valid_times_one_date_missing(self, sample_dataarray):
        """Test check_for_valid_times when one date is not in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 15)  # Outside range

        result = check_for_valid_times_xr_dataarray(
            sample_dataarray, start_date, end_date
        )
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self, sample_dataarray):
        """Test check_for_valid_times when neither date is in dataset."""
        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = check_for_valid_times_xr_dataarray(
            sample_dataarray, start_date, end_date
        )
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when DataArray has latitude and longitude dimensions."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataArray with spatial data
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        da = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [40.0, 41.0, 42.0, 43.0],
                "longitude": [-74.0, -73.0, -72.0, -71.0],
            },
            name="temperature",
        )
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataarray(da, region)
        assert result is True

    def test_check_for_spatial_data_with_lat_lon(self):
        """Test check_for_spatial_data when DataArray has lat and lon dimensions."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataArray with spatial data using 'lat' and 'lon'
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        da = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={
                "lat": [40.0, 41.0, 42.0, 43.0],
                "lon": [-74.0, -73.0, -72.0, -71.0],
            },
            name="temperature",
        )
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataarray(da, region)
        assert result is True

    def test_check_for_spatial_data_no_spatial_dimensions(self):
        """Test check_for_spatial_data when DataArray has no spatial dimensions."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataArray without spatial data
        data = np.random.randn(10)  # 1D array
        da = xr.DataArray(
            data,
            dims=["time"],
            coords={"time": pd.date_range("2021-01-01", periods=10, freq="1D")},
            name="temperature",
        )
        
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataarray(da, region)
        assert result is False

    def test_check_for_spatial_data_no_overlap(self):
        """Test check_for_spatial_data when data is outside region bounds."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create DataArray with spatial data outside region
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        da = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={
                "latitude": [50.0, 51.0, 52.0, 53.0],  # Outside region
                "longitude": [-80.0, -79.0, -78.0, -77.0],  # Outside region
            },
            name="temperature",
        )
        
        # Create region that doesn't overlap with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataarray(da, region)
        assert result is False


class TestXarrayDatasetModule:
    """Tests for xarray_dataset module."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample Dataset for testing."""
        time = pd.date_range("2021-01-01", periods=10, freq="1D")
        data_shape = (10, 5, 5)
        ds = xr.Dataset(
            {
                "temperature": (
                    ["valid_time", "lat", "lon"],
                    np.random.randn(*data_shape),
                ),
                "pressure": (
                    ["valid_time", "lat", "lon"],
                    np.random.randn(*data_shape),
                ),
                "humidity": (
                    ["valid_time", "lat", "lon"],
                    np.random.randn(*data_shape),
                ),
                "wind_speed": (
                    ["valid_time", "lat", "lon"],
                    np.random.randn(*data_shape),
                ),
            },
            coords={
                "valid_time": time,
                "lat": range(5),
                "lon": range(5),
            },
        )
        return ds

    def test_safely_pull_variables_success(self, sample_dataset):
        """Test successful variable extraction."""
        variables = ["temperature", "pressure"]
        optional_variables = []
        optional_variables_mapping = {}

        result = safely_pull_variables_xr_dataset(
            sample_dataset, variables, optional_variables, optional_variables_mapping
        )

        assert sorted(result.data_vars) == sorted(variables)
        assert result["temperature"].equals(sample_dataset["temperature"])
        assert result["pressure"].equals(sample_dataset["pressure"])

    def test_safely_pull_variables_with_optional(self, sample_dataset):
        """Test variable extraction with optional variables."""
        variables = ["temperature"]
        optional_variables = ["humidity", "nonexistent"]
        optional_variables_mapping = {}

        result = safely_pull_variables_xr_dataset(
            sample_dataset, variables, optional_variables, optional_variables_mapping
        )

        expected_vars = ["humidity", "temperature"]
        assert sorted(result.data_vars) == sorted(expected_vars)

    def test_safely_pull_variables_with_mapping(self, sample_dataset):
        """Test variable extraction with optional variable mapping."""
        variables = ["temp", "press"]
        optional_variables = ["temperature", "pressure"]
        optional_variables_mapping = {
            "temperature": "temp",
            "pressure": "press",
        }

        result = safely_pull_variables_xr_dataset(
            sample_dataset, variables, optional_variables, optional_variables_mapping
        )

        expected_vars = ["temperature", "pressure"]
        assert sorted(result.data_vars) == sorted(expected_vars)

    def test_safely_pull_variables_with_mapping_list(self, sample_dataset):
        """Test variable extraction with optional variable mapping as list."""
        variables = ["temp", "press"]
        optional_variables = ["wind_speed"]
        optional_variables_mapping = {
            "wind_speed": ["temp", "press"],
        }

        result = safely_pull_variables_xr_dataset(
            sample_dataset, variables, optional_variables, optional_variables_mapping
        )

        expected_vars = ["wind_speed"]
        assert list(result.data_vars) == expected_vars

    def test_safely_pull_variables_missing_required(self, sample_dataset):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]
        optional_variables = []
        optional_variables_mapping = {}

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_xr_dataset(
                sample_dataset,
                variables,
                optional_variables,
                optional_variables_mapping,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self, sample_dataset):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = check_for_valid_times_xr_dataset(sample_dataset, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_one_date_missing(self, sample_dataset):
        """Test check_for_valid_times when one date is not in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 15)  # Outside range

        result = check_for_valid_times_xr_dataset(sample_dataset, start_date, end_date)
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self, sample_dataset):
        """Test check_for_valid_times when neither date is in dataset."""
        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = check_for_valid_times_xr_dataset(sample_dataset, start_date, end_date)
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when Dataset has latitude and longitude coordinates."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create Dataset with spatial data
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        ds = xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={
                "latitude": [40.0, 41.0, 42.0, 43.0],
                "longitude": [-74.0, -73.0, -72.0, -71.0],
            },
        )
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataset(ds, region)
        assert result is True

    def test_check_for_spatial_data_with_lat_lon(self):
        """Test check_for_spatial_data when Dataset has lat and lon coordinates."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create Dataset with spatial data using 'lat' and 'lon'
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        ds = xr.Dataset(
            {"temperature": (["lat", "lon"], data)},
            coords={
                "lat": [40.0, 41.0, 42.0, 43.0],
                "lon": [-74.0, -73.0, -72.0, -71.0],
            },
        )
        
        # Create region that overlaps with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataset(ds, region)
        assert result is True

    def test_check_for_spatial_data_no_spatial_coordinates(self):
        """Test check_for_spatial_data when Dataset has no spatial coordinates."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create Dataset without spatial data
        data = np.random.randn(10)  # 1D array
        ds = xr.Dataset(
            {"temperature": (["time"], data)},
            coords={"time": pd.date_range("2021-01-01", periods=10, freq="1D")},
        )
        
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataset(ds, region)
        assert result is False

    def test_check_for_spatial_data_no_overlap(self):
        """Test check_for_spatial_data when data is outside region bounds."""
        from extremeweatherbench.regions import BoundingBoxRegion
        
        # Create Dataset with spatial data outside region
        data = np.random.randn(4, 4)  # 4x4 spatial grid
        ds = xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={
                "latitude": [50.0, 51.0, 52.0, 53.0],  # Outside region
                "longitude": [-80.0, -79.0, -78.0, -77.0],  # Outside region
            },
        )
        
        # Create region that doesn't overlap with data
        region = BoundingBoxRegion(
            latitude_min=39.5, latitude_max=43.5,
            longitude_min=-74.5, longitude_max=-70.5
        )
        
        result = check_for_spatial_data_xr_dataset(ds, region)
        assert result is False
