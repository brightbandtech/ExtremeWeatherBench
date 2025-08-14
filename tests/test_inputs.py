"""Comprehensive tests for extremeweatherbench.inputs module."""

from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench import inputs


class TestInputBase:
    """Test the abstract InputBase class."""

    def test_input_base_is_abstract(self):
        """Test that InputBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            inputs.InputBase(
                source="test",
                variables=["test"],
                variable_mapping={},
                storage_options={},
            )

    def test_input_base_name_property(self):
        """Test that name property returns class name."""

        # Create a concrete subclass for testing
        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )
        assert test_input.name == "TestInput"

    def test_maybe_convert_to_dataset_with_dataset(self, sample_era5_dataset):
        """Test maybe_convert_to_dataset with xarray Dataset input."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )
        result = test_input.maybe_convert_to_dataset(sample_era5_dataset)
        assert isinstance(result, xr.Dataset)
        assert result is sample_era5_dataset

    def test_maybe_convert_to_dataset_with_dataarray(
        self, sample_gridded_obs_dataarray
    ):
        """Test maybe_convert_to_dataset with xarray DataArray input."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )
        result = test_input.maybe_convert_to_dataset(sample_gridded_obs_dataarray)
        assert isinstance(result, xr.Dataset)

    def test_maybe_convert_to_dataset_custom_conversion(self):
        """Test maybe_convert_to_dataset with custom data type."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )

        with pytest.raises(NotImplementedError):
            test_input.maybe_convert_to_dataset("invalid_data_type")

    def test_add_source_to_dataset_attrs(self, sample_era5_dataset):
        """Test adding source to dataset attributes."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )

        result = test_input.add_source_to_dataset_attrs(sample_era5_dataset)
        assert result.attrs["source"] == "TestInput"


class TestMaybeMapVariableNames:
    """Test the maybe_map_variable_names method across different data types."""

    @pytest.fixture
    def test_input_base(self):
        """Create a concrete test class for InputBase."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        return TestInput

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_xarray_dataset_with_mapping(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test variable mapping with xarray Dataset."""
        mock_derived.return_value = ["original_temp", "original_pressure"]

        # Create a dataset with original variable names
        test_data = sample_era5_dataset.copy()
        test_data = test_data.rename(
            {
                "2m_temperature": "original_temp",
                "mean_sea_level_pressure": "original_pressure",
            }
        )

        # Test input with variable mapping
        test_input = test_input_base(
            source="test",
            variables=["mapped_temp", "mapped_pressure"],
            variable_mapping={
                "original_temp": "mapped_temp",
                "original_pressure": "mapped_pressure",
            },
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(test_data)

        # Check that variables were renamed according to mapping
        assert "mapped_temp" in result.data_vars
        assert "mapped_pressure" in result.data_vars
        assert "original_temp" not in result.data_vars
        assert "original_pressure" not in result.data_vars

        mock_derived.assert_called_once_with(mock.ANY)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_xarray_dataset_no_mapping(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test variable selection without mapping with xarray Dataset."""
        mock_derived.return_value = ["2m_temperature"]

        test_input = test_input_base(
            source="test",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)

        # Check that only specified variable is included, no renaming
        assert "2m_temperature" in result.data_vars
        assert "mean_sea_level_pressure" not in result.data_vars

        mock_derived.assert_called_once_with([])

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_xarray_dataarray(
        self, mock_derived, test_input_base, sample_gridded_obs_dataarray
    ):
        """Test variable mapping with xarray DataArray."""
        mock_derived.return_value = ["2m_temperature"]

        test_input = test_input_base(
            source="test",
            variables=["temp"],
            variable_mapping={"2m_temperature": "temp"},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_gridded_obs_dataarray)

        # DataArray should be renamed
        assert isinstance(result, xr.Dataset)
        assert "temp" in result.data_vars
        assert "2m_temperature" not in result.data_vars

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_polars_lazyframe(
        self, mock_derived, test_input_base, sample_ghcn_dataframe
    ):
        """Test variable mapping with polars LazyFrame."""
        import polars as pl

        mock_derived.return_value = ["surface_air_temperature", "latitude"]

        test_input = test_input_base(
            source="test",
            variables=["temp"],
            variable_mapping={"surface_air_temperature": "temp"},
            storage_options={},
        )

        lazy_data = sample_ghcn_dataframe.lazy()
        result = test_input.maybe_map_variable_names(lazy_data)

        # Check that LazyFrame columns were renamed
        assert isinstance(result, pl.LazyFrame)
        schema = result.collect_schema()
        assert "temp" in schema.names()
        assert "surface_air_temperature" not in schema.names()
        assert "latitude" in schema.names()

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_pandas_dataframe(
        self, mock_derived, test_input_base, sample_lsr_dataframe
    ):
        """Test variable mapping with pandas DataFrame."""
        mock_derived.return_value = ["report_type", "magnitude"]

        test_input = test_input_base(
            source="test",
            variables=["event_type"],
            variable_mapping={"report_type": "event_type"},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_lsr_dataframe)

        # Check that DataFrame columns were renamed
        assert isinstance(result, pd.DataFrame)
        assert "event_type" in result.columns
        assert "report_type" not in result.columns
        assert "magnitude" in result.columns

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_partial_mapping(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test variable mapping when only some variables in mapping exist in data."""
        mock_derived.return_value = ["2m_temperature", "nonexistent_var"]

        test_input = test_input_base(
            source="test",
            variables=["temp"],
            variable_mapping={
                "2m_temperature": "temp",
                "nonexistent_var": "missing_var",
            },
            storage_options={},
        )

        # This should raise an error since nonexistent_var is not in data
        with pytest.raises(KeyError):
            test_input.maybe_map_variable_names(sample_era5_dataset)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_no_variables_defined(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test error when no variables are defined."""
        mock_derived.return_value = []

        test_input = test_input_base(
            source="test", variables=[], variable_mapping={}, storage_options={}
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)
        xr.testing.assert_identical(result, sample_era5_dataset)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_unsupported_data_type(
        self, mock_derived, test_input_base
    ):
        """Test error with unsupported data type."""
        mock_derived.return_value = ["test_var"]

        test_input = test_input_base(
            source="test",
            variables=["test_var"],
            variable_mapping={},
            storage_options={},
        )

        # Create mock data that doesn't have .variables attribute
        class MockData:
            def __init__(self):
                self.variables = ["test_var"]

            def __getitem__(self, key):
                return self

        mock_data = MockData()

        with pytest.raises(ValueError, match="Data type .* not supported"):
            test_input.maybe_map_variable_names(mock_data)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_empty_variable_mapping(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test when variable_mapping is empty dict."""
        mock_derived.return_value = ["2m_temperature"]

        test_input = test_input_base(
            source="test",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)

        # Should return subset data without any renaming
        assert "2m_temperature" in result.data_vars
        assert "mean_sea_level_pressure" not in result.data_vars

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_maybe_map_variable_names_with_derived_variables(
        self, mock_derived, test_input_base, sample_era5_dataset
    ):
        """Test variable mapping with derived variables involved."""
        # Mock derived variables that require multiple base variables
        mock_derived.return_value = [
            "2m_temperature",
            "mean_sea_level_pressure",
            "extra_derived_var",
        ]

        # Add the extra variable that would be required by derived variable
        test_data = sample_era5_dataset.copy()
        test_data["extra_derived_var"] = test_data["2m_temperature"] * 2

        test_input = test_input_base(
            source="test",
            variables=["temp"],
            variable_mapping={
                "2m_temperature": "temp",
                "mean_sea_level_pressure": "pressure",
            },
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(test_data)

        # Should include derived variable requirements and apply mapping
        assert "temp" in result.data_vars
        assert "pressure" in result.data_vars
        assert "extra_derived_var" in result.data_vars
        assert "2m_temperature" not in result.data_vars
        assert "mean_sea_level_pressure" not in result.data_vars


class TestForecastBase:
    """Test the ForecastBase class."""

    def test_forecast_base_subset_data_to_case_invalid_input(self):
        """Test subset_data_to_case with invalid input type."""
        forecast = inputs.ZarrForecast(
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Expected xarray Dataset"):
            forecast.subset_data_to_case("invalid_data", Mock())

    @patch("extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time")
    @patch("extremeweatherbench.utils.convert_init_time_to_valid_time")
    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_forecast_base_subset_data_to_case(
        self, mock_derived, mock_convert, mock_derive, sample_forecast_dataset
    ):
        """Test subset_data_to_case with valid input."""
        # Setup mocks
        mock_derive.return_value = np.array([[0, 1], [0, 1]])
        mock_convert.return_value = sample_forecast_dataset
        mock_derived.return_value = ["surface_air_temperature"]

        # Create mock case operator
        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case.case_metadata.location.mask.return_value = sample_forecast_dataset
        mock_case.forecast.variables = ["surface_air_temperature"]

        forecast = inputs.ZarrForecast(
            source="test.zarr",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = forecast.subset_data_to_case(sample_forecast_dataset, mock_case)
        assert isinstance(result, xr.Dataset)


class TestTargetBase:
    """Test the TargetBase class."""

    def test_target_base_maybe_align_forecast_to_target_default(
        self, sample_forecast_dataset, sample_era5_dataset
    ):
        """Test default implementation of maybe_align_forecast_to_target."""

        class TestTarget(inputs.TargetBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        target = TestTarget(
            source="test", variables=["test"], variable_mapping={}, storage_options={}
        )

        forecast_result, target_result = target.maybe_align_forecast_to_target(
            sample_forecast_dataset, sample_era5_dataset
        )

        # Default implementation should return inputs unchanged
        assert forecast_result is sample_forecast_dataset
        assert target_result is sample_era5_dataset


class TestEvaluationObject:
    """Test the EvaluationObject dataclass."""

    def test_evaluation_object_creation(self):
        """Test creating an EvaluationObject."""
        mock_metric = Mock()
        mock_target = Mock()
        mock_forecast = Mock()

        eval_obj = inputs.EvaluationObject(
            event_type="test_event",
            metric=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        assert eval_obj.event_type == "test_event"
        assert eval_obj.metric == [mock_metric]
        assert eval_obj.target == mock_target
        assert eval_obj.forecast == mock_forecast


class TestZarrForecast:
    """Test the ZarrForecast class."""

    @patch("xarray.open_zarr")
    def test_zarr_forecast_open_data_from_source(
        self, mock_open_zarr, sample_forecast_dataset
    ):
        """Test opening data from zarr source."""
        mock_open_zarr.return_value = sample_forecast_dataset

        forecast = inputs.ZarrForecast(
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = forecast._open_data_from_source()

        mock_open_zarr.assert_called_once_with(
            "test.zarr",
            storage_options={},
            chunks=None,
            decode_timedelta=True,
        )
        assert result == sample_forecast_dataset

    def test_zarr_forecast_custom_chunks(self):
        """Test ZarrForecast with custom chunks."""
        custom_chunks = {"time": 24, "latitude": 361, "longitude": 720}

        forecast = inputs.ZarrForecast(
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
            chunks=custom_chunks,
        )

        assert forecast.chunks == custom_chunks


class TestKerchunkForecast:
    """Test the KerchunkForecast class."""

    @patch("extremeweatherbench.inputs.open_kerchunk_reference")
    def test_kerchunk_forecast_open_data_from_source(
        self, mock_open_kerchunk, sample_forecast_dataset
    ):
        """Test opening data from kerchunk source."""
        mock_open_kerchunk.return_value = sample_forecast_dataset

        forecast = inputs.KerchunkForecast(
            source="test.parq",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = forecast._open_data_from_source()

        mock_open_kerchunk.assert_called_once_with(
            "test.parq",
            storage_options={},
            chunks="auto",
        )
        assert result == sample_forecast_dataset


class TestERA5:
    """Test the ERA5 target class."""

    @patch("xarray.open_zarr")
    def test_era5_open_data_from_source(self, mock_open_zarr, sample_era5_dataset):
        """Test opening ERA5 data from source."""
        mock_open_zarr.return_value = sample_era5_dataset

        era5 = inputs.ERA5(
            source="gs://test-bucket/era5.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
            chunks={"time": 48, "latitude": "auto", "longitude": "auto"},
        )

        result = era5._open_data_from_source()

        mock_open_zarr.assert_called_once_with(
            "gs://test-bucket/era5.zarr",
            storage_options={},
            chunks={"time": 48, "latitude": "auto", "longitude": "auto"},
        )
        xr.testing.assert_identical(result, sample_era5_dataset)

    @patch("extremeweatherbench.inputs.zarr_target_subsetter")
    def test_era5_subset_data_to_case(self, mock_subsetter, sample_era5_dataset):
        """Test ERA5 subset_data_to_case."""
        mock_subsetter.return_value = sample_era5_dataset
        mock_case = Mock()

        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = era5.subset_data_to_case(sample_era5_dataset, mock_case)

        mock_subsetter.assert_called_once_with(sample_era5_dataset, mock_case)
        assert result == sample_era5_dataset

    def test_era5_maybe_align_forecast_to_target_same_grid(
        self, sample_era5_dataset, sample_forecast_with_valid_time
    ):
        """Test ERA5 alignment when grids are identical."""
        # Make forecast and target have same spatial grid
        target_data = sample_era5_dataset.copy()
        forecast_data = sample_forecast_with_valid_time.copy()

        # Ensure same spatial coordinates
        forecast_data = forecast_data.interp(
            latitude=target_data.latitude, longitude=target_data.longitude
        )

        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        aligned_forecast, aligned_target = era5.maybe_align_forecast_to_target(
            forecast_data, target_data
        )

        # Should align in time but not regrid spatially
        assert isinstance(aligned_target, xr.Dataset)
        assert isinstance(aligned_forecast, xr.Dataset)

        # Check that time dimensions are aligned
        assert "valid_time" in aligned_target.dims
        assert "valid_time" in aligned_forecast.dims

    def test_era5_maybe_align_forecast_to_target_different_grid(
        self, sample_era5_dataset
    ):
        """Test ERA5 alignment when grids are different."""
        # Create forecast with different spatial grid
        forecast_data = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.randn(20, 45, 90),
                ),
            },
            coords={
                "valid_time": pd.date_range("2021-06-20", periods=20, freq="6h"),
                "latitude": np.linspace(-90, 90, 45),  # Different resolution
                "longitude": np.linspace(0, 359, 90),  # Different resolution
            },
        )

        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        aligned_forecast, aligned_target = era5.maybe_align_forecast_to_target(
            forecast_data, sample_era5_dataset
        )

        # Should regrid forecast to target grid
        assert np.array_equal(
            aligned_target.latitude.values, aligned_forecast.latitude.values
        )
        assert np.array_equal(
            aligned_target.longitude.values, aligned_forecast.longitude.values
        )

    def test_era5_maybe_align_forecast_to_target_time_dimension_mismatch(self):
        """Test ERA5 alignment when time dimensions have different names."""
        # Target with 'time' dimension
        target_data = xr.Dataset(
            {
                "2m_temperature": (
                    ["time", "latitude", "longitude"],
                    np.random.randn(30, 91, 180),
                ),
            },
            coords={
                "time": pd.date_range("2021-06-20", periods=30, freq="6h"),
                "latitude": np.linspace(-90, 90, 91),
                "longitude": np.linspace(0, 359, 180),
            },
        )

        # Forecast with 'valid_time' dimension
        forecast_data = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.randn(25, 91, 180),
                ),
            },
            coords={
                "valid_time": pd.date_range("2021-06-21", periods=25, freq="6h"),
                "latitude": np.linspace(-90, 90, 91),
                "longitude": np.linspace(0, 359, 180),
            },
        )

        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        aligned_forecast, aligned_target = era5.maybe_align_forecast_to_target(
            forecast_data, target_data
        )

        # Both should have valid_time dimension after alignment
        assert "valid_time" in aligned_target.dims
        assert "valid_time" in aligned_forecast.dims
        assert "time" not in aligned_target.dims


class TestGHCN:
    """Test the GHCN target class."""

    @patch("polars.scan_parquet")
    def test_ghcn_open_data_from_source(self, mock_scan_parquet, sample_ghcn_dataframe):
        """Test opening GHCN data from source."""
        mock_scan_parquet.return_value = sample_ghcn_dataframe.lazy()

        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = ghcn._open_data_from_source()

        mock_scan_parquet.assert_called_once_with("test.parquet", storage_options={})
        assert isinstance(result, pl.LazyFrame)

    def test_ghcn_subset_data_to_case(self, sample_ghcn_dataframe):
        """Test GHCN subset_data_to_case."""
        # Create mock case operator
        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case.case_metadata.location.geopandas.total_bounds = [-120, 30, -90, 50]
        mock_case.target.variables = ["surface_air_temperature"]

        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = ghcn.subset_data_to_case(sample_ghcn_dataframe.lazy(), mock_case)

        assert isinstance(result, pl.LazyFrame)

    def test_ghcn_subset_data_to_case_invalid_input(self):
        """Test GHCN subset_data_to_case with invalid input."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Expected polars LazyFrame"):
            ghcn.subset_data_to_case("invalid_data", Mock())

    def test_ghcn_custom_convert_to_dataset(self, sample_ghcn_dataframe):
        """Test GHCN custom conversion to dataset."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = ghcn._custom_convert_to_dataset(sample_ghcn_dataframe.lazy())

        assert isinstance(result, xr.Dataset)
        assert "valid_time" in result.dims
        assert "latitude" in result.dims
        assert "longitude" in result.dims

    def test_ghcn_custom_convert_to_dataset_invalid_input(self):
        """Test GHCN custom conversion with invalid input."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Data is not a polars LazyFrame"):
            ghcn._custom_convert_to_dataset("invalid_data")

    @patch("extremeweatherbench.inputs.align_forecast_to_point_obs_target")
    def test_ghcn_maybe_align_forecast_to_target(
        self, mock_align, sample_forecast_dataset, sample_era5_dataset
    ):
        """Test GHCN alignment method."""
        mock_align.return_value = (sample_forecast_dataset, sample_era5_dataset)

        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = ghcn.maybe_align_forecast_to_target(
            sample_forecast_dataset, sample_era5_dataset
        )

        mock_align.assert_called_once_with(sample_forecast_dataset, sample_era5_dataset)
        assert result == (sample_forecast_dataset, sample_era5_dataset)


class TestLSR:
    """Test the LSR (Local Storm Report) target class."""

    @patch("pandas.read_parquet")
    def test_lsr_open_data_from_source(self, mock_read_parquet, sample_lsr_dataframe):
        """Test opening LSR data from source."""
        mock_read_parquet.return_value = sample_lsr_dataframe

        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        result = lsr._open_data_from_source()

        mock_read_parquet.assert_called_once_with("test.parquet", storage_options={})
        assert result.equals(sample_lsr_dataframe)

    def test_lsr_subset_data_to_case(self, sample_lsr_dataframe):
        """Test LSR subset_data_to_case."""
        # Create mock case operator
        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-21")
        mock_case.case_metadata.location.latitude_min = 30
        mock_case.case_metadata.location.latitude_max = 50
        mock_case.case_metadata.location.longitude_min = -110
        mock_case.case_metadata.location.longitude_max = -90

        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        result = lsr.subset_data_to_case(sample_lsr_dataframe, mock_case)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_lsr_dataframe)

    def test_lsr_subset_data_to_case_invalid_input(self):
        """Test LSR subset_data_to_case with invalid input."""
        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            lsr.subset_data_to_case("invalid_data", Mock())

    def test_lsr_custom_convert_to_dataset(self, sample_lsr_dataframe):
        """Test LSR custom conversion to dataset."""
        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        result = lsr._custom_convert_to_dataset(sample_lsr_dataframe)

        assert isinstance(result, xr.Dataset)

    def test_lsr_custom_convert_to_dataset_invalid_input(self):
        """Test LSR custom conversion with invalid input."""
        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Data is not a pandas DataFrame"):
            lsr._custom_convert_to_dataset("invalid_data")

    @patch("extremeweatherbench.inputs.align_forecast_to_point_obs_target")
    def test_lsr_maybe_align_forecast_to_target(
        self, mock_align, sample_forecast_dataset, sample_era5_dataset
    ):
        """Test LSR alignment method."""
        mock_align.return_value = (sample_forecast_dataset, sample_era5_dataset)

        lsr = inputs.LSR(
            source="test.parquet",
            variables=["report"],
            variable_mapping={},
            storage_options={},
        )

        result = lsr.maybe_align_forecast_to_target(
            sample_forecast_dataset, sample_era5_dataset
        )

        mock_align.assert_called_once_with(sample_forecast_dataset, sample_era5_dataset)
        assert result == (sample_forecast_dataset, sample_era5_dataset)


class TestPPH:
    """Test the PPH (Practically Perfect Hindcast) target class."""

    @patch("xarray.open_zarr")
    def test_pph_open_data_from_source(self, mock_open_zarr, sample_era5_dataset):
        """Test opening PPH data from source."""
        mock_open_zarr.return_value = sample_era5_dataset

        pph = inputs.PPH(
            source="test.zarr",
            variables=["precipitation"],
            variable_mapping={},
            storage_options={},
        )

        result = pph._open_data_from_source()

        mock_open_zarr.assert_called_once_with("test.zarr", storage_options={})
        assert result == sample_era5_dataset

    @patch("extremeweatherbench.inputs.zarr_target_subsetter")
    def test_pph_subset_data_to_case(self, mock_subsetter, sample_era5_dataset):
        """Test PPH subset_data_to_case."""
        mock_subsetter.return_value = sample_era5_dataset
        mock_case = Mock()

        pph = inputs.PPH(
            source="test.zarr",
            variables=["precipitation"],
            variable_mapping={},
            storage_options={},
        )

        result = pph.subset_data_to_case(sample_era5_dataset, mock_case)

        mock_subsetter.assert_called_once_with(sample_era5_dataset, mock_case)
        assert result == sample_era5_dataset

    def test_pph_custom_convert_to_dataset(self, sample_era5_dataset):
        """Test PPH custom conversion (should return data unchanged)."""
        pph = inputs.PPH(
            source="test.zarr",
            variables=["precipitation"],
            variable_mapping={},
            storage_options={},
        )

        result = pph._custom_convert_to_dataset(sample_era5_dataset)

        assert result is sample_era5_dataset


class TestIBTrACS:
    """Test the IBTrACS target class."""

    @patch("polars.scan_csv")
    def test_ibtracs_open_data_from_source(
        self, mock_scan_csv, sample_ibtracs_dataframe
    ):
        """Test opening IBTrACS data from source."""
        mock_scan_csv.return_value = sample_ibtracs_dataframe.lazy()

        ibtracs = inputs.IBTrACS(
            source="test.csv",
            variables=["surface_wind_speed"],
            variable_mapping={},
            storage_options={},
        )

        result = ibtracs._open_data_from_source()

        mock_scan_csv.assert_called_once_with(
            "test.csv",
            storage_options={},
            skip_rows_after_header=1,
        )
        assert isinstance(result, pl.LazyFrame)

    def test_ibtracs_custom_convert_to_dataset(self, sample_ibtracs_dataframe):
        """Test IBTrACS custom conversion to dataset."""
        ibtracs = inputs.IBTrACS(
            source="test.csv",
            variables=["surface_wind_speed"],
            variable_mapping={},
            storage_options={},
        )

        result = ibtracs._custom_convert_to_dataset(sample_ibtracs_dataframe.lazy())

        assert isinstance(result, xr.Dataset)

    def test_ibtracs_custom_convert_to_dataset_invalid_input(self):
        """Test IBTrACS custom conversion with invalid input."""
        ibtracs = inputs.IBTrACS(
            source="test.csv",
            variables=["surface_wind_speed"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Data is not a polars LazyFrame"):
            ibtracs._custom_convert_to_dataset("invalid_data")


class TestStandaloneFunctions:
    """Test standalone functions in inputs module."""

    @patch("xarray.open_dataset")
    def test_open_kerchunk_reference_parquet(
        self, mock_open_dataset, sample_forecast_dataset
    ):
        """Test opening kerchunk reference from parquet file."""
        mock_open_dataset.return_value = sample_forecast_dataset

        storage_options = {"remote_protocol": "s3", "remote_options": {"anon": True}}
        chunks = {"time": 24}

        result = inputs.open_kerchunk_reference(
            "test.parq", storage_options=storage_options, chunks=chunks
        )

        mock_open_dataset.assert_called_once_with(
            "test.parq",
            engine="kerchunk",
            storage_options=storage_options,
            chunks=chunks,
        )
        assert result == sample_forecast_dataset

    @patch("xarray.open_dataset")
    def test_open_kerchunk_reference_json(
        self, mock_open_dataset, sample_forecast_dataset
    ):
        """Test opening kerchunk reference from JSON file."""
        mock_open_dataset.return_value = sample_forecast_dataset

        storage_options = {"remote_protocol": "s3", "remote_options": {"anon": True}}

        result = inputs.open_kerchunk_reference(
            "test.json", storage_options=storage_options
        )

        expected_storage_options = storage_options.copy()
        expected_storage_options["fo"] = "test.json"

        mock_open_dataset.assert_called_once_with(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "storage_options": expected_storage_options,
                "consolidated": False,
            },
            chunks="auto",
        )
        assert result == sample_forecast_dataset

    def test_open_kerchunk_reference_unsupported_format(self):
        """Test opening kerchunk reference with unsupported file format."""
        with pytest.raises(TypeError, match="Unknown kerchunk file type"):
            inputs.open_kerchunk_reference("test.txt")

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_zarr_target_subsetter(self, mock_derived, sample_era5_dataset):
        """Test zarr_target_subsetter function."""
        mock_derived.return_value = ["2m_temperature"]

        # Create mock case operator
        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case.case_metadata.location.mask.return_value = sample_era5_dataset
        mock_case.target.variables = ["2m_temperature"]

        result = inputs.zarr_target_subsetter(sample_era5_dataset, mock_case)

        mock_derived.assert_called_once_with(["2m_temperature"])
        mock_case.case_metadata.location.mask.assert_called_once()
        assert isinstance(result, xr.Dataset)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_zarr_target_subsetter_missing_variables(
        self, mock_derived, sample_era5_dataset
    ):
        """Test zarr_target_subsetter with missing variables."""
        mock_derived.return_value = ["nonexistent_variable"]

        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case.target.variables = ["nonexistent_variable"]

        with pytest.raises(ValueError, match="Variables .* not found in target data"):
            inputs.zarr_target_subsetter(sample_era5_dataset, mock_case)

    @patch(
        "extremeweatherbench.derived.maybe_pull_required_variables_from_derived_input"
    )
    def test_zarr_target_subsetter_no_variables(
        self, mock_derived, sample_era5_dataset
    ):
        """Test zarr_target_subsetter with no variables defined."""
        mock_derived.return_value = None

        mock_case = Mock()
        mock_case.case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case.case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case.target.variables = None

        with pytest.raises(ValueError, match="Variables not defined"):
            inputs.zarr_target_subsetter(sample_era5_dataset, mock_case)

    def test_align_forecast_to_point_obs_target(self):
        """Test align_forecast_to_point_obs_target function."""
        # Create simple test data with overlapping times
        target_times = pd.date_range("2021-06-20", periods=3, freq="6h")
        forecast_times = pd.date_range("2021-06-20", periods=5, freq="6h")

        # Create target dataset with location stacked properly
        target_ds = xr.Dataset(
            {
                "temperature": (["valid_time", "location"], np.random.randn(3, 2)),
                "longitude": (["location"], [-100, -101]),
                "latitude": (["location"], [40, 41]),
            },
            coords={"valid_time": target_times, "location": ["A", "B"]},
        )

        # Create forecast dataset
        forecast_ds = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.randn(5, 91, 180),
                ),
            },
            coords={
                "valid_time": forecast_times,
                "latitude": np.linspace(-90, 90, 91),
                "longitude": np.linspace(0, 359, 180),
            },
        )

        aligned_forecast, aligned_target = inputs.align_forecast_to_point_obs_target(
            forecast_ds, target_ds
        )

        assert isinstance(aligned_target, xr.Dataset)
        assert isinstance(aligned_forecast, xr.Dataset)

        # Check that both datasets have the same time dimension
        assert len(aligned_target.valid_time) == len(aligned_forecast.valid_time)
        assert len(aligned_target.valid_time) > 0  # Should have overlapping times

        # Check that forecast has been interpolated to observation locations
        assert "latitude" in aligned_forecast.dims
        assert "longitude" in aligned_forecast.dims

    def test_align_forecast_to_point_obs_target_time_alignment(self):
        """Test time alignment in align_forecast_to_point_obs_target."""
        # Create simple target and forecast datasets with overlapping times
        target_times = pd.date_range("2021-06-20", periods=5, freq="6h")
        forecast_times = pd.date_range("2021-06-20 06:00", periods=3, freq="6h")

        target_ds = xr.Dataset(
            {
                "temperature": (["valid_time", "location"], np.random.randn(5, 2)),
                "longitude": (["location"], [-100, -101]),
                "latitude": (["location"], [40, 41]),
            },
            coords={"valid_time": target_times, "location": ["A", "B"]},
        )

        forecast_ds = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.randn(3, 91, 180),
                ),
            },
            coords={
                "valid_time": forecast_times,
                "latitude": np.linspace(-90, 90, 91),
                "longitude": np.linspace(0, 359, 180),
            },
        )

        aligned_forecast, aligned_target = inputs.align_forecast_to_point_obs_target(
            forecast_ds, target_ds
        )

        # Should have overlapping times only
        assert len(aligned_target.valid_time) == len(aligned_forecast.valid_time)
        assert len(aligned_target.valid_time) <= min(
            len(target_times), len(forecast_times)
        )


class TestConstants:
    """Test module constants."""

    def test_arco_era5_uri(self):
        """Test ARCO ERA5 URI constant."""
        assert inputs.ARCO_ERA5_FULL_URI.startswith("gs://")
        assert "era5" in inputs.ARCO_ERA5_FULL_URI.lower()

    def test_ghcn_uri(self):
        """Test GHCN URI constant."""
        assert inputs.DEFAULT_GHCN_URI.startswith("gs://")
        assert "ghcn" in inputs.DEFAULT_GHCN_URI.lower()

    def test_lsr_uri(self):
        """Test LSR URI constant."""
        assert inputs.LSR_URI.startswith("gs://")
        assert "lsr" in inputs.LSR_URI.lower()

    def test_pph_uri(self):
        """Test PPH URI constant."""
        assert inputs.PPH_URI.startswith("gs://")

    def test_ibtracs_uri(self):
        """Test IBTrACS URI constant."""
        assert inputs.IBTRACS_URI.startswith("https://")
        assert "ibtracs" in inputs.IBTRACS_URI.lower()

    def test_ibtracs_variable_mapping(self):
        """Test IBTrACS variable mapping."""
        mapping = inputs.IBTrACS_metadata_variable_mapping

        assert isinstance(mapping, dict)
        assert "ISO_TIME" in mapping
        assert "LAT" in mapping
        assert "LON" in mapping
        assert mapping["ISO_TIME"] == "valid_time"
        assert mapping["LAT"] == "latitude"
        assert mapping["LON"] == "longitude"


@pytest.mark.integration
class TestInputsIntegration:
    """Integration tests for inputs module."""

    def test_era5_full_workflow_with_zarr(self, temp_zarr_file):
        """Test complete ERA5 workflow with zarr file."""
        era5 = inputs.ERA5(
            source=temp_zarr_file,
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test opening data
        data = era5._open_data_from_source()
        assert isinstance(data, xr.Dataset)
        assert "2m_temperature" in data.data_vars

        # Test conversion
        dataset = era5.maybe_convert_to_dataset(data)
        assert isinstance(dataset, xr.Dataset)

    def test_ghcn_full_workflow_with_parquet(self, temp_parquet_file):
        """Test complete GHCN workflow with parquet file."""
        ghcn = inputs.GHCN(
            source=temp_parquet_file,
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test opening data
        data = ghcn._open_data_from_source()
        assert isinstance(data, pl.LazyFrame)

        # Test conversion
        dataset = ghcn._custom_convert_to_dataset(data)
        assert isinstance(dataset, xr.Dataset)

    def test_era5_alignment_comprehensive(
        self, sample_era5_dataset, sample_forecast_with_valid_time
    ):
        """Test comprehensive ERA5 alignment scenarios."""
        era5 = inputs.ERA5(
            source="test.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Test with matching spatial grids but different time ranges
        target_subset = sample_era5_dataset.sel(time=slice("2021-06-20", "2021-06-21"))
        forecast_subset = sample_forecast_with_valid_time.sel(
            valid_time=slice("2021-06-20 12:00", "2021-06-21 12:00")
        )

        aligned_forecast, aligned_target = era5.maybe_align_forecast_to_target(
            forecast_subset, target_subset
        )

        # Should find overlapping times
        assert len(aligned_target.valid_time) > 0
        assert len(aligned_forecast.valid_time) > 0
        assert len(aligned_target.valid_time) == len(aligned_forecast.valid_time)
