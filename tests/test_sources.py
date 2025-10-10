"""Tests for the sources module functions."""

import pandas as pd
import polars as pl
import pytest
import xarray as xr


class TestSafelyPullVariablesXrDataset:
    """Test safely_pull_variables_xr_dataset function."""

    def test_required_variables_only(self, sample_era5_dataset):
        """Test pulling only required variables from xarray Dataset."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        result = safely_pull_variables_xr_dataset(
            sample_era5_dataset,
            variables=["2m_temperature"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "mean_sea_level_pressure" not in result.data_vars

    def test_with_optional_variables(self, sample_era5_dataset):
        """Test pulling optional variables from xarray Dataset."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        result = safely_pull_variables_xr_dataset(
            sample_era5_dataset,
            variables=["2m_temperature"],
            optional_variables=["mean_sea_level_pressure"],
            optional_variables_mapping={},
        )

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "mean_sea_level_pressure" in result.data_vars

    def test_optional_replaces_required(self, sample_era5_dataset):
        """Test optional variable replacing required variable."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        # Add a dewpoint variable to replace temperature
        sample_era5_dataset = sample_era5_dataset.assign(
            dewpoint_temperature=sample_era5_dataset["2m_temperature"] - 5
        )

        result = safely_pull_variables_xr_dataset(
            sample_era5_dataset,
            variables=["2m_temperature"],
            optional_variables=["dewpoint_temperature"],
            optional_variables_mapping={"dewpoint_temperature": ["2m_temperature"]},
        )

        assert isinstance(result, xr.Dataset)
        assert "dewpoint_temperature" in result.data_vars
        assert "2m_temperature" not in result.data_vars

    def test_optional_replaces_multiple_required(self, sample_era5_dataset):
        """Test optional variable replacing multiple required variables."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        # Add a combined variable
        sample_era5_dataset = sample_era5_dataset.assign(
            combined_var=sample_era5_dataset["2m_temperature"]
        )

        result = safely_pull_variables_xr_dataset(
            sample_era5_dataset,
            variables=["2m_temperature", "mean_sea_level_pressure"],
            optional_variables=["combined_var"],
            optional_variables_mapping={
                "combined_var": ["2m_temperature", "mean_sea_level_pressure"]
            },
        )

        assert isinstance(result, xr.Dataset)
        assert "combined_var" in result.data_vars
        assert "2m_temperature" not in result.data_vars
        assert "mean_sea_level_pressure" not in result.data_vars

    def test_missing_required_variable_raises_error(self, sample_era5_dataset):
        """Test error when required variable is missing."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_xr_dataset(
                sample_era5_dataset,
                variables=["nonexistent_variable"],
                optional_variables=[],
                optional_variables_mapping={},
            )

    def test_multiple_variables(self, sample_era5_dataset):
        """Test pulling multiple required variables."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataset

        result = safely_pull_variables_xr_dataset(
            sample_era5_dataset,
            variables=["2m_temperature", "mean_sea_level_pressure"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "mean_sea_level_pressure" in result.data_vars


class TestSafelyPullVariablesXrDataArray:
    """Test safely_pull_variables_xr_dataarray function."""

    def test_matching_name(self, sample_gridded_obs_dataarray):
        """Test pulling variables from DataArray with matching name."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataarray

        result = safely_pull_variables_xr_dataarray(
            sample_gridded_obs_dataarray,
            variables=["2m_temperature"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, xr.DataArray)
        assert result.name == "2m_temperature"

    def test_matching_optional_name(self, sample_gridded_obs_dataarray):
        """Test pulling variables when DataArray matches optional variable."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataarray

        result = safely_pull_variables_xr_dataarray(
            sample_gridded_obs_dataarray,
            variables=["some_other_var"],
            optional_variables=["2m_temperature"],
            optional_variables_mapping={},
        )

        assert isinstance(result, xr.DataArray)
        assert result.name == "2m_temperature"

    def test_no_match_raises_error(self, sample_gridded_obs_dataarray):
        """Test error when DataArray name doesn't match requested variable."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataarray

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_xr_dataarray(
                sample_gridded_obs_dataarray,
                variables=["nonexistent_variable"],
                optional_variables=[],
                optional_variables_mapping={},
            )

    def test_unnamed_dataarray(self):
        """Test handling of unnamed DataArray."""
        from extremeweatherbench.sources import safely_pull_variables_xr_dataarray

        # Create an unnamed DataArray
        da = xr.DataArray([1, 2, 3], dims=["x"])

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_xr_dataarray(
                da,
                variables=["test_var"],
                optional_variables=[],
                optional_variables_mapping={},
            )


class TestSafelyPullVariablesPandasDataFrame:
    """Test safely_pull_variables_pandas_dataframe function."""

    def test_required_variables_only(self):
        """Test pulling only required variables from Pandas DataFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_pandas_dataframe,
        )

        df = pd.DataFrame(
            {
                "temp": [1, 2, 3],
                "pressure": [4, 5, 6],
                "humidity": [7, 8, 9],
            }
        )

        result = safely_pull_variables_pandas_dataframe(
            df,
            variables=["temp", "pressure"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, pd.DataFrame)
        assert "temp" in result.columns
        assert "pressure" in result.columns
        assert "humidity" not in result.columns

    def test_with_optional_variables(self):
        """Test pulling optional variables from Pandas DataFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_pandas_dataframe,
        )

        df = pd.DataFrame(
            {
                "temp": [1, 2, 3],
                "dewpoint": [4, 5, 6],
            }
        )

        result = safely_pull_variables_pandas_dataframe(
            df,
            variables=["temp"],
            optional_variables=["dewpoint"],
            optional_variables_mapping={},
        )

        assert isinstance(result, pd.DataFrame)
        assert "temp" in result.columns
        assert "dewpoint" in result.columns

    def test_optional_replaces_required(self):
        """Test optional variable replacing required in Pandas DataFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_pandas_dataframe,
        )

        df = pd.DataFrame(
            {
                "temp": [1, 2, 3],
                "dewpoint": [4, 5, 6],
            }
        )

        result = safely_pull_variables_pandas_dataframe(
            df,
            variables=["temp", "humidity"],
            optional_variables=["dewpoint"],
            optional_variables_mapping={"dewpoint": ["temp", "humidity"]},
        )

        assert isinstance(result, pd.DataFrame)
        assert "dewpoint" in result.columns
        assert "temp" not in result.columns

    def test_missing_required_variable_raises_error(self):
        """Test error when required variable is missing from DataFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_pandas_dataframe,
        )

        df = pd.DataFrame({"temp": [1, 2, 3]})

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_pandas_dataframe(
                df,
                variables=["nonexistent"],
                optional_variables=[],
                optional_variables_mapping={},
            )

    def test_single_string_in_mapping(self):
        """Test optional mapping with single string instead of list."""
        from extremeweatherbench.sources import (
            safely_pull_variables_pandas_dataframe,
        )

        df = pd.DataFrame(
            {
                "dewpoint": [4, 5, 6],
            }
        )

        # Test with string instead of list in mapping
        result = safely_pull_variables_pandas_dataframe(
            df,
            variables=["temp"],
            optional_variables=["dewpoint"],
            optional_variables_mapping={"dewpoint": "temp"},
        )

        assert isinstance(result, pd.DataFrame)
        assert "dewpoint" in result.columns


class TestSafelyPullVariablesPolarsLazyFrame:
    """Test safely_pull_variables_polars_lazyframe function."""

    def test_required_variables_only(self, sample_ghcn_dataframe):
        """Test pulling only required variables from Polars LazyFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_polars_lazyframe,
        )

        lazy_df = sample_ghcn_dataframe.lazy()

        result = safely_pull_variables_polars_lazyframe(
            lazy_df,
            variables=["surface_air_temperature"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "surface_air_temperature" in collected.columns
        assert "station_id" not in collected.columns

    def test_with_optional_variables(self, sample_ghcn_dataframe):
        """Test pulling optional variables from Polars LazyFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_polars_lazyframe,
        )

        lazy_df = sample_ghcn_dataframe.lazy()

        result = safely_pull_variables_polars_lazyframe(
            lazy_df,
            variables=["surface_air_temperature"],
            optional_variables=["latitude"],
            optional_variables_mapping={},
        )

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "surface_air_temperature" in collected.columns
        assert "latitude" in collected.columns

    def test_optional_replaces_required(self, sample_ghcn_dataframe):
        """Test optional variable replacing required in Polars LazyFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_polars_lazyframe,
        )

        # Add a dewpoint column
        df_with_dewpoint = sample_ghcn_dataframe.with_columns(
            pl.col("surface_air_temperature").alias("dewpoint_temperature")
        )
        lazy_df = df_with_dewpoint.lazy()

        result = safely_pull_variables_polars_lazyframe(
            lazy_df,
            variables=["surface_air_temperature", "humidity"],
            optional_variables=["dewpoint_temperature"],
            optional_variables_mapping={
                "dewpoint_temperature": ["surface_air_temperature", "humidity"]
            },
        )

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "dewpoint_temperature" in collected.columns
        assert "surface_air_temperature" not in collected.columns

    def test_missing_required_variable_raises_error(self, sample_ghcn_dataframe):
        """Test error when required variable is missing from LazyFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_polars_lazyframe,
        )

        lazy_df = sample_ghcn_dataframe.lazy()

        with pytest.raises(KeyError, match="Required variables.*not found"):
            safely_pull_variables_polars_lazyframe(
                lazy_df,
                variables=["nonexistent_variable"],
                optional_variables=[],
                optional_variables_mapping={},
            )

    def test_multiple_variables(self, sample_ghcn_dataframe):
        """Test pulling multiple variables from Polars LazyFrame."""
        from extremeweatherbench.sources import (
            safely_pull_variables_polars_lazyframe,
        )

        lazy_df = sample_ghcn_dataframe.lazy()

        result = safely_pull_variables_polars_lazyframe(
            lazy_df,
            variables=["surface_air_temperature", "latitude"],
            optional_variables=[],
            optional_variables_mapping={},
        )

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "surface_air_temperature" in collected.columns
        assert "latitude" in collected.columns
        assert "longitude" not in collected.columns
