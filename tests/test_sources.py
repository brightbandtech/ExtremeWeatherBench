"""Tests for extremeweatherbench.sources modules."""

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench.sources import (
    pandas_dataframe,
    polars_lazyframe,
    xarray_dataarray,
    xarray_dataset,
)


class TestPandasDataFrameModule:
    """Tests for pandas_dataframe module."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        data = {
            "valid_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "lead_time": pd.timedelta_range("0 hours", periods=10, freq="1h"),
            "init_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "latitude": np.random.uniform(30, 50, 10),
            "longitude": np.random.uniform(-120, -70, 10),
            "temperature": np.random.randn(10),
            "pressure": np.random.randn(10),
            "humidity": np.random.randn(10),
            "wind_speed": np.random.randn(10),
        }
        return pd.DataFrame(data)

    def test_safely_pull_variables_success(self, sample_dataframe):
        """Test successful variable extraction."""
        variables = ["temperature", "pressure"]

        result = pandas_dataframe.safely_pull_variables(sample_dataframe, variables)

        # Only the requested variables are returned
        assert sorted(result.columns) == sorted(variables)
        assert len(result) == len(sample_dataframe)

    def test_safely_pull_variables_with_optional(self, sample_dataframe):
        """Test variable extraction with multiple variables."""
        variables = ["temperature", "humidity"]

        result = pandas_dataframe.safely_pull_variables(sample_dataframe, variables)

        # Only the requested variables are returned
        expected_columns = ["humidity", "temperature"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping(self, sample_dataframe):
        """Test variable extraction with multiple variables."""
        variables = ["temperature", "pressure"]

        result = pandas_dataframe.safely_pull_variables(sample_dataframe, variables)

        # Only the requested variables are returned
        expected_columns = ["temperature", "pressure"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping_list(self, sample_dataframe):
        """Test variable extraction with single variable."""
        variables = ["wind_speed"]

        result = pandas_dataframe.safely_pull_variables(sample_dataframe, variables)

        # Only the requested variable is returned
        expected_columns = ["wind_speed"]
        assert sorted(result.columns) == sorted(expected_columns)

    def test_safely_pull_variables_missing_required(self, sample_dataframe):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]

        with pytest.raises(KeyError, match="Required variables.*not found"):
            pandas_dataframe.safely_pull_variables(
                sample_dataframe,
                variables,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = pandas_dataframe.check_for_valid_times(df, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_one_date_missing(self):
        """Test check_for_valid_times when one date is not in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 10)  # Outside range

        result = pandas_dataframe.check_for_valid_times(df, start_date, end_date)
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self):
        """Test check_for_valid_times when neither date is in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pd.DataFrame({"valid_time": dates, "value": range(len(dates))})

        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = pandas_dataframe.check_for_valid_times(df, start_date, end_date)
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when DataFrame has latitude and longitude
        columns."""
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = pandas_dataframe.check_for_spatial_data(df, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = pandas_dataframe.check_for_spatial_data(df, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = pandas_dataframe.check_for_spatial_data(df, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = pandas_dataframe.check_for_spatial_data(df, region)
        assert result is False


class TestPolarsLazyFrameModule:
    """Tests for polars_lazyframe module."""

    @pytest.fixture
    def sample_lazyframe(self):
        """Create a sample LazyFrame for testing."""
        data = {
            "valid_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "lead_time": pd.timedelta_range("0 hours", periods=10, freq="1h"),
            "init_time": pd.date_range("2021-01-01", periods=10, freq="1D"),
            "latitude": np.random.uniform(30, 50, 10),
            "longitude": np.random.uniform(-120, -70, 10),
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

        result = polars_lazyframe.safely_pull_variables(sample_lazyframe, variables)

        # Only the requested variables are returned
        assert sorted(result.collect_schema().names()) == sorted(variables)

    def test_safely_pull_variables_with_optional(self, sample_lazyframe):
        """Test variable extraction with multiple variables."""
        variables = ["temperature", "humidity"]

        result = polars_lazyframe.safely_pull_variables(sample_lazyframe, variables)

        # Only the requested variables are returned
        expected_columns = ["humidity", "temperature"]
        assert sorted(result.collect_schema().names()) == sorted(expected_columns)

    def test_safely_pull_variables_with_mapping(self, sample_lazyframe):
        """Test variable extraction with multiple variables."""
        variables = ["temperature", "pressure"]

        result = polars_lazyframe.safely_pull_variables(sample_lazyframe, variables)

        # Only the requested variables are returned
        expected_columns = ["temperature", "pressure"]
        assert sorted(result.collect_schema().names()) == sorted(expected_columns)

    def test_safely_pull_variables_missing_required(self, sample_lazyframe):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]

        with pytest.raises(KeyError, match="Required variables.*not found"):
            polars_lazyframe.safely_pull_variables(
                sample_lazyframe,
                variables,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_one_date_missing(self):
        """Test check_for_valid_times when one date is not in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 10)  # Outside range

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self):
        """Test check_for_valid_times when neither date is in dataset."""
        dates = pd.date_range("2021-01-01", "2021-01-05", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is False

    def test_check_for_valid_times_with_datetime_column(self):
        """Ensure datetime columns work with pl.lit() comparisons."""
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="1D")
        df = pl.DataFrame({"valid_time": dates, "value": range(len(dates))})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_with_string_column(self):
        """Test that string datetime columns are converted properly."""
        date_strings = [
            "2021-01-01T00:00:00",
            "2021-01-02T00:00:00",
            "2021-01-03T00:00:00",
            "2021-01-04T00:00:00",
            "2021-01-05T00:00:00",
        ]
        df = pl.DataFrame({"valid_time": date_strings, "value": range(5)})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 2)
        end_date = datetime.datetime(2021, 1, 4)

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_with_date_type(self):
        """Test with polars Date type columns."""
        dates = pl.date_range(
            datetime.date(2021, 1, 1),
            datetime.date(2021, 1, 10),
            interval="1d",
            eager=True,
        )
        df = pl.DataFrame({"valid_time": dates, "value": range(10)})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is True

    def test_check_for_valid_times_no_time_column(self):
        """Test when no time column exists in the LazyFrame."""
        df = pl.DataFrame({"value": range(10), "id": range(10)})
        lf = df.lazy()

        start_date = datetime.datetime(2021, 1, 1)
        end_date = datetime.datetime(2021, 1, 10)

        result = polars_lazyframe.check_for_valid_times(lf, start_date, end_date)
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when LazyFrame has latitude and longitude
        columns."""
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
        assert result is False

    def test_check_for_spatial_data_partial_overlap(self):
        """Test when only some data points fall within the region."""
        from extremeweatherbench.regions import BoundingBoxRegion

        # Create LazyFrame with data partially in and out of region
        data = {
            "latitude": [38.0, 40.0, 45.0, 50.0],
            "longitude": [-75.0, -73.0, -71.0, -80.0],
            "temperature": [20.0, 21.0, 22.0, 23.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()

        # Create region that overlaps with some points
        region = BoundingBoxRegion(
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
        assert result is True

    def test_check_for_spatial_data_edge_case(self):
        """Test with data points exactly on region boundaries."""
        from extremeweatherbench.regions import BoundingBoxRegion

        # Create LazyFrame with data on boundaries
        data = {
            "latitude": [39.5, 43.5],
            "longitude": [-74.5, -70.5],
            "temperature": [20.0, 21.0],
        }
        df = pl.DataFrame(data)
        lf = df.lazy()

        # Create region with exact boundaries
        region = BoundingBoxRegion(
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = polars_lazyframe.check_for_spatial_data(lf, region)
        assert result is True


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

        result = xarray_dataarray.safely_pull_variables(sample_dataarray, variables)

        assert result.name == "temperature"
        assert result.equals(sample_dataarray)

    def test_safely_pull_variables_with_optional(self, sample_dataarray):
        """Test variable extraction with optional variables."""
        variables = ["temperature"]

        result = xarray_dataarray.safely_pull_variables(sample_dataarray, variables)

        assert result.name == "temperature"
        assert result.equals(sample_dataarray)

    def test_safely_pull_variables_missing_required(self, sample_dataarray):
        """Test error when required variables are missing."""
        variables = ["pressure"]

        with pytest.raises(KeyError, match="Required variables.*not found"):
            xarray_dataarray.safely_pull_variables(
                sample_dataarray,
                variables,
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

        result = xarray_dataarray.safely_pull_variables(da, variables)

        assert result.equals(da)

    def test_check_for_valid_times_both_dates_in_dataset(self, sample_dataarray):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = xarray_dataarray.check_for_valid_times(
            sample_dataarray, start_date, end_date
        )
        assert result is True

    def test_check_for_valid_times_one_date_missing(self, sample_dataarray):
        """Test check_for_valid_times when one date is not in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 15)  # Outside range

        result = xarray_dataarray.check_for_valid_times(
            sample_dataarray, start_date, end_date
        )
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self, sample_dataarray):
        """Test check_for_valid_times when neither date is in dataset."""
        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = xarray_dataarray.check_for_valid_times(
            sample_dataarray, start_date, end_date
        )
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when DataArray has latitude and longitude
        sdimensions."""
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataarray.check_for_spatial_data(da, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataarray.check_for_spatial_data(da, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataarray.check_for_spatial_data(da, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataarray.check_for_spatial_data(da, region)
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

        result = xarray_dataset.safely_pull_variables(sample_dataset, variables)

        assert sorted(result.data_vars) == sorted(variables)
        assert result["temperature"].equals(sample_dataset["temperature"])
        assert result["pressure"].equals(sample_dataset["pressure"])

    def test_safely_pull_variables_with_optional(self, sample_dataset):
        """Test variable extraction with optional variables."""
        variables = ["temperature", "humidity"]

        result = xarray_dataset.safely_pull_variables(sample_dataset, variables)

        expected_vars = ["humidity", "temperature"]
        assert sorted(result.data_vars) == sorted(expected_vars)

    def test_safely_pull_variables_with_mapping(self, sample_dataset):
        """Test variable extraction with variable mapping."""
        variables = ["temperature", "pressure"]

        result = xarray_dataset.safely_pull_variables(sample_dataset, variables)

        expected_vars = ["temperature", "pressure"]
        assert sorted(result.data_vars) == sorted(expected_vars)

    def test_safely_pull_variables_with_mapping_list(self, sample_dataset):
        """Test variable extraction with variable mapping as list."""
        variables = ["wind_speed"]

        result = xarray_dataset.safely_pull_variables(sample_dataset, variables)

        expected_vars = ["wind_speed"]
        assert list(result.data_vars) == expected_vars

    def test_safely_pull_variables_missing_required(self, sample_dataset):
        """Test error when required variables are missing."""
        variables = ["temperature", "nonexistent"]

        with pytest.raises(KeyError, match="Required variables.*not found"):
            xarray_dataset.safely_pull_variables(
                sample_dataset,
                variables,
            )

    def test_check_for_valid_times_both_dates_in_dataset(self, sample_dataset):
        """Test check_for_valid_times when both start and end dates are in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 7)

        result = xarray_dataset.check_for_valid_times(
            sample_dataset, start_date, end_date
        )
        assert result is True

    def test_check_for_valid_times_one_date_missing(self, sample_dataset):
        """Test check_for_valid_times when one date is not in dataset."""
        start_date = datetime.datetime(2021, 1, 3)
        end_date = datetime.datetime(2021, 1, 15)  # Outside range

        result = xarray_dataset.check_for_valid_times(
            sample_dataset, start_date, end_date
        )
        assert result is True  # Should still return True if any data in range

    def test_check_for_valid_times_neither_date_in_dataset(self, sample_dataset):
        """Test check_for_valid_times when neither date is in dataset."""
        start_date = datetime.datetime(2021, 2, 1)  # Outside range
        end_date = datetime.datetime(2021, 2, 10)  # Outside range

        result = xarray_dataset.check_for_valid_times(
            sample_dataset, start_date, end_date
        )
        assert result is False

    def test_check_for_spatial_data_with_latitude_longitude(self):
        """Test check_for_spatial_data when Dataset has latitude and longitude
        coordinates."""
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataset.check_for_spatial_data(ds, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataset.check_for_spatial_data(ds, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataset.check_for_spatial_data(ds, region)
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
            latitude_min=39.5,
            latitude_max=43.5,
            longitude_min=-74.5,
            longitude_max=-70.5,
        )

        result = xarray_dataset.check_for_spatial_data(ds, region)
        assert result is False
