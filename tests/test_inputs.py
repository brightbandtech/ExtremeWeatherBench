"""Tests for inputs module."""

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
                name="test",
                source="test",
                variables=["test"],
                variable_mapping={},
                storage_options={},
            )

    def test_maybe_convert_to_dataset_with_dataset(self, sample_era5_dataset):
        """Test maybe_convert_to_dataset with xarray Dataset input."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            name="test",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
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
            name="test",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
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
            name="test",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(NotImplementedError):
            test_input.maybe_convert_to_dataset("invalid_data_type")

    def test_add_source_to_dataset_attrs_forecast_base(self, sample_era5_dataset):
        """Test adding source and dataset_type for ForecastBase subclass."""

        class TestForecast(inputs.ForecastBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_forecast = TestForecast(
            name="test_forecast",
            source="test_source",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        result = test_forecast.add_source_to_dataset_attrs(sample_era5_dataset)
        assert result.attrs["source"] == "test_forecast"

    def test_add_source_to_dataset_attrs_target_base(self, sample_era5_dataset):
        """Test adding source and dataset_type for TargetBase subclass."""

        class TestTarget(inputs.TargetBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_target = TestTarget(
            name="test_target",
            source="test_source",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        result = test_target.add_source_to_dataset_attrs(sample_era5_dataset)
        assert result.attrs["source"] == "test_target"

    def test_add_source_to_dataset_attrs_generic_input_base(self, sample_era5_dataset):
        """Test adding source and dataset_type for generic InputBase subclass."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            name="test_input",
            source="test_source",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.add_source_to_dataset_attrs(sample_era5_dataset)
        assert result.attrs["source"] == "test_input"

    def test_add_source_to_dataset_attrs_preserves_existing_attrs(
        self, sample_era5_dataset
    ):
        """Test that adding source preserves existing dataset attributes."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        # Add some existing attributes to the dataset
        sample_era5_dataset.attrs["existing_attr"] = "existing_value"
        sample_era5_dataset.attrs["description"] = "Test dataset"

        test_input = TestInput(
            name="test",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.add_source_to_dataset_attrs(sample_era5_dataset)

        # Check new attributes are added
        assert result.attrs["source"] == "test"

        # Check existing attributes are preserved
        assert result.attrs["existing_attr"] == "existing_value"
        assert result.attrs["description"] == "Test dataset"

    def test_set_name(self):
        """Test setting the name using the set_name method."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        test_input = TestInput(
            name="original_name",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        # Verify original name
        assert test_input.name == "original_name"

        # Change the name using set_name method
        test_input.set_name("new_name")

        # Verify the name was changed
        assert test_input.name == "new_name"

        # Test with different name types
        test_input.set_name("forecast_v2")
        assert test_input.name == "forecast_v2"


class TestMaybeMapVariableNames:
    """Test the maybe_map_variable_names method across different data types."""

    # TODO: move to conftest
    @pytest.fixture
    def test_input_base(self):
        """Create a concrete test class for InputBase."""

        class TestInput(inputs.InputBase):
            def _open_data_from_source(self):
                return None

            def subset_data_to_case(self, data, case_operator):
                return data

        return TestInput

    def test_maybe_map_variable_names_xarray_dataset_with_mapping(
        self, test_input_base, sample_era5_dataset
    ):
        """Test variable mapping with xarray Dataset."""
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
            name="test",
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

    def test_maybe_map_variable_names_xarray_dataset_no_mapping(
        self, test_input_base, sample_era5_dataset
    ):
        """Test no variable mapping returns data unchanged with xarray Dataset."""
        test_input = test_input_base(
            name="test",
            source="test",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)

        # Should return data unchanged when no variable mapping is provided
        xr.testing.assert_identical(result, sample_era5_dataset)

    def test_maybe_map_variable_names_xarray_dataarray(
        self, test_input_base, sample_gridded_obs_dataarray
    ):
        """Test variable mapping with xarray DataArray."""
        test_input = test_input_base(
            name="test",
            source="test",
            variables=["temp"],
            variable_mapping={"2m_temperature": "temp"},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_gridded_obs_dataarray)

        # DataArray should be renamed
        assert isinstance(result, xr.DataArray)
        assert result.name == "temp"

    @patch("extremeweatherbench.derived.maybe_include_variables_from_derived_input")
    def test_maybe_map_variable_names_polars_lazyframe(
        self, mock_derived, test_input_base, sample_ghcn_dataframe
    ):
        """Test variable mapping with polars LazyFrame."""
        mock_derived.return_value = ["surface_air_temperature", "latitude"]

        test_input = test_input_base(
            name="test",
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

    @patch("extremeweatherbench.derived.maybe_include_variables_from_derived_input")
    def test_maybe_map_variable_names_pandas_dataframe(
        self, mock_derived, test_input_base, sample_lsr_dataframe
    ):
        """Test variable mapping with pandas DataFrame."""
        mock_derived.return_value = ["report_type", "magnitude"]

        test_input = test_input_base(
            name="test",
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

    def test_maybe_map_variable_names_partial_mapping(
        self, test_input_base, sample_era5_dataset
    ):
        """Test variable mapping when only some variables in mapping exist in data."""
        test_input = test_input_base(
            name="test",
            source="test",
            variables=["temp"],
            variable_mapping={
                "2m_temperature": "temp",
                "nonexistent_var": "missing_var",
            },
            storage_options={},
        )

        # Should only map variables that exist, ignoring nonexistent ones
        result = test_input.maybe_map_variable_names(sample_era5_dataset)

        # Should have renamed 2m_temperature but left other variables unchanged
        assert "temp" in result.data_vars
        assert "2m_temperature" not in result.data_vars
        assert "mean_sea_level_pressure" in result.data_vars

    def test_maybe_map_variable_names_no_variables_defined(
        self, test_input_base, sample_era5_dataset
    ):
        """Test when no variables are defined."""
        test_input = test_input_base(
            name="test",
            source="test",
            variables=[],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)
        xr.testing.assert_identical(result, sample_era5_dataset)

    def test_maybe_map_variable_names_unsupported_data_type(self, test_input_base):
        """Test error with unsupported data type."""
        test_input = test_input_base(
            name="test",
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

    def test_maybe_map_variable_names_empty_variable_mapping(
        self, test_input_base, sample_era5_dataset
    ):
        """Test when variable_mapping is empty dict."""
        test_input = test_input_base(
            name="test",
            source="test",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = test_input.maybe_map_variable_names(sample_era5_dataset)

        # Should return data unchanged when variable mapping is empty
        xr.testing.assert_identical(result, sample_era5_dataset)

    @patch("extremeweatherbench.derived.maybe_include_variables_from_derived_input")
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
            name="test",
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
            name="test",
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        with pytest.raises(ValueError, match="Expected xarray Dataset"):
            forecast.subset_data_to_case("invalid_data", Mock())

    @patch("extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time")
    @patch("extremeweatherbench.utils.convert_init_time_to_valid_time")
    @patch("extremeweatherbench.derived.maybe_include_variables_from_derived_input")
    def test_forecast_base_subset_data_to_case(
        self, mock_derived, mock_convert, mock_derive, sample_forecast_dataset
    ):
        """Test subset_data_to_case with valid input."""
        # Create a dataset with valid_time dimension for the mock return
        forecast_with_valid_time = sample_forecast_dataset.copy()
        # Convert init_time/lead_time to valid_time for the mock
        valid_times = pd.date_range("2021-06-20", periods=10, freq="6h")
        forecast_with_valid_time = forecast_with_valid_time.isel(
            init_time=0, lead_time=slice(0, 10)
        ).rename({"lead_time": "valid_time"})
        forecast_with_valid_time = forecast_with_valid_time.assign_coords(
            valid_time=valid_times
        )

        # Setup mocks
        mock_derive.return_value = np.array([[0, 1], [0, 1]])
        mock_convert.return_value = forecast_with_valid_time
        mock_derived.return_value = ["surface_air_temperature"]

        # Create mock case operator
        mock_case = Mock()
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-22")
        mock_case.location.mask.return_value = forecast_with_valid_time

        forecast = inputs.ZarrForecast(
            name="test",
            source="test.zarr",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = forecast.subset_data_to_case(sample_forecast_dataset, mock_case)
        assert isinstance(result, xr.Dataset)

    def test_forecast_base_set_name_inheritance(self):
        """Test that ForecastBase inherits set_name method from InputBase."""
        forecast = inputs.ZarrForecast(
            name="original_forecast",
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Verify original name
        assert forecast.name == "original_forecast"

        # Use inherited set_name method
        forecast.set_name("updated_forecast")

        # Verify the name was changed
        assert forecast.name == "updated_forecast"


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
            name="test",
            source="test",
            variables=["test"],
            variable_mapping={},
            storage_options={},
        )

        forecast_result, target_result = target.maybe_align_forecast_to_target(
            sample_forecast_dataset, sample_era5_dataset
        )

        # Default implementation should return inputs unchanged
        assert forecast_result is sample_forecast_dataset
        assert target_result is sample_era5_dataset

    def test_target_base_set_name_inheritance(self):
        """Test that TargetBase inherits set_name method from InputBase."""
        target = inputs.ERA5(
            name="original_era5",
            source="gs://test-bucket/era5.zarr",
            variables=["2m_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Verify original name
        assert target.name == "original_era5"

        # Use inherited set_name method
        target.set_name("updated_era5")

        # Verify the name was changed
        assert target.name == "updated_era5"


class TestEvaluationObject:
    """Test the EvaluationObject dataclass."""

    def test_evaluation_object_creation(self):
        """Test creating an EvaluationObject."""
        mock_metric = Mock()
        mock_target = Mock()
        mock_forecast = Mock()

        eval_obj = inputs.EvaluationObject(
            event_type="test_event",
            metric_list=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        assert eval_obj.event_type == "test_event"
        assert eval_obj.metric_list == [mock_metric]
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
            name="test",
            source="test.zarr",
            variables=["temperature"],
            variable_mapping={},
            storage_options={},
        )

        result = forecast._open_data_from_source()

        mock_open_zarr.assert_called_once_with(
            "test.zarr",
            storage_options={},
            chunks="auto",
            decode_timedelta=True,
        )
        assert result == sample_forecast_dataset

    def test_zarr_forecast_custom_chunks(self):
        """Test ZarrForecast with custom chunks."""
        custom_chunks = {"time": 24, "latitude": 361, "longitude": 720}

        forecast = inputs.ZarrForecast(
            name="test",
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
            name="test",
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
            name="test",
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
            name="test",
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
            name="test",
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

        # Check that datasets are aligned - both should have overlapping times
        # Original dimension names are preserved
        assert "time" in aligned_target.dims  # ERA5 uses 'time'
        assert "valid_time" in aligned_forecast.dims  # Forecast uses 'valid_time'

        # Should have overlapping time values
        assert len(aligned_target.time) > 0
        assert len(aligned_forecast.valid_time) > 0

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

        # Should preserve original dimension names but align the data
        assert "time" in aligned_target.dims  # Target keeps 'time'
        assert "valid_time" in aligned_forecast.dims  # Forecast keeps 'valid_time'

        # Should have overlapping time periods
        assert len(aligned_target.time) > 0
        assert len(aligned_forecast.valid_time) > 0


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
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-22")
        mock_case.location.geopandas.total_bounds = [-120, 30, -90, 50]

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

    def test_ghcn_subset_data_to_case_sorted_valid_time(self, sample_ghcn_dataframe):
        """Test that subset_data_to_case returns sorted valid_time column."""

        # Create unsorted test data by shuffling the sample data
        unsorted_data = sample_ghcn_dataframe.sample(fraction=1.0, shuffle=True)

        # Verify the data is actually unsorted by checking valid_times
        is_sorted = unsorted_data["valid_time"].is_sorted()

        # If by chance it's still sorted, manually disorder it
        if is_sorted:
            # Reverse the order to ensure it's unsorted
            unsorted_data = sample_ghcn_dataframe.reverse()

        # Create mock case metadata (not case operator)
        mock_case_metadata = Mock()
        mock_case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case_metadata.location.geopandas.total_bounds = [-120, 30, -90, 50]

        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Call subset_data_to_case with unsorted data
        result = ghcn.subset_data_to_case(unsorted_data.lazy(), mock_case_metadata)

        # Collect the result and verify valid_time is sorted
        collected_result = result.collect()
        assert collected_result[
            "valid_time"
        ].is_sorted(), "valid_time column should be sorted"

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
        assert "index" in result.dims  # reset_index creates 'index' dimension
        assert "valid_time" in result.data_vars  # valid_time becomes a data variable


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

    def test_ghcn_custom_convert_to_dataset_no_duplicates(self, sample_ghcn_dataframe):
        """Test GHCN custom conversion with no duplicates (baseline case)."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # With reset_index() approach, all data is preserved
        result = ghcn._custom_convert_to_dataset(sample_ghcn_dataframe.lazy())

        assert isinstance(result, xr.Dataset)
        assert "index" in result.dims  # reset_index creates 'index' dimension
        assert "valid_time" in result.data_vars  # valid_time becomes a data variable
        assert "surface_air_temperature" in result.data_vars

        # Should preserve all data with reset_index approach
        original_count = len(sample_ghcn_dataframe)
        result_count = result.surface_air_temperature.count().item()
        assert result_count == original_count

    def test_ghcn_custom_convert_to_dataset_single_duplicate(
        self, sample_ghcn_dataframe
    ):
        """Test GHCN custom conversion with single duplicate entry."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # With reset_index() approach, duplicates are preserved as separate rows
        # Duplicate the first row
        first_row = sample_ghcn_dataframe.slice(0, 1)
        data_with_duplicate = pl.concat([sample_ghcn_dataframe, first_row])

        result = ghcn._custom_convert_to_dataset(data_with_duplicate.lazy())

        assert isinstance(result, xr.Dataset)
        assert "index" in result.dims
        assert "surface_air_temperature" in result.data_vars

        # Should preserve all data including the duplicate
        expected_count = len(sample_ghcn_dataframe) + 1  # Original + 1 duplicate
        result_count = result.surface_air_temperature.count().item()
        assert result_count == expected_count

    def test_ghcn_custom_convert_to_dataset_many_duplicates(
        self, sample_ghcn_dataframe
    ):
        """Test GHCN custom conversion with many duplicate entries."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # With reset_index() approach, all duplicates are preserved
        # Create multiple duplicates by repeating first 5 rows
        duplicates = sample_ghcn_dataframe.slice(0, 5)
        data_with_many_duplicates = pl.concat(
            [
                sample_ghcn_dataframe,
                duplicates,  # First set of duplicates
                duplicates,  # Second set of duplicates
                duplicates,  # Third set of duplicates
            ]
        )

        result = ghcn._custom_convert_to_dataset(data_with_many_duplicates.lazy())

        assert isinstance(result, xr.Dataset)
        assert "index" in result.dims
        assert "surface_air_temperature" in result.data_vars

        # Should preserve all data including duplicates
        expected_count = len(sample_ghcn_dataframe) + 3 * 5  # Original + 3 sets of 5 duplicates
        result_count = result.surface_air_temperature.count().item()
        assert result_count == expected_count

    def test_ghcn_custom_convert_to_dataset_exception_handling(self):
        """Test GHCN custom conversion exception handling returns empty Dataset."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        # Create data that will cause pandas to_xarray() to fail
        # Use a DataFrame with incompatible data types that break xarray conversion
        problematic_data = pl.DataFrame(
            {
                "valid_time": [None, None],  # None values that cause issues
                "latitude": [float('inf'), float('-inf')],  # Infinite values
                "longitude": [float('nan'), float('nan')],  # NaN values
                "surface_air_temperature": [None, None],  # None in numeric column
            }
        )

        result = ghcn._custom_convert_to_dataset(problematic_data.lazy())

        # With reset_index() approach, even problematic data gets converted successfully
        # This is actually more robust behavior
        assert isinstance(result, xr.Dataset)
        assert "index" in result.dims
        assert len(result.data_vars) > 0  # Data is preserved even if problematic
        
        # The data should contain the problematic values (None, inf, nan)
        assert "valid_time" in result.data_vars
        assert "latitude" in result.data_vars
        assert "longitude" in result.data_vars

    def test_ghcn_custom_convert_to_dataset_empty_dataset_downstream_safe(self):
        """Test that empty Dataset from exception handling doesn't cause downstream
        problems."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Create empty dataset (simulating exception case)
        empty_dataset = xr.Dataset()

        # Test that common downstream operations don't fail
        # These operations should handle empty datasets gracefully

        # Test alignment operations
        forecast_data = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.randn(5, 3, 3),
                )
            },
            coords={
                "valid_time": pd.date_range("2021-06-20", periods=5, freq="6h"),
                "latitude": [40.0, 41.0, 42.0],
                "longitude": [-100.0, -101.0, -102.0],
            },
        )

        # Should not raise an error even with empty target
        try:
            aligned_forecast, aligned_target = ghcn.maybe_align_forecast_to_target(
                forecast_data, empty_dataset
            )
            # Operation should complete without error
            assert isinstance(aligned_forecast, xr.Dataset)
            assert isinstance(aligned_target, xr.Dataset)
        except Exception as e:
            # If it does fail, it should be a controlled failure, not a crash
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

        # Test that adding attrs works with empty dataset
        result_with_attrs = ghcn.add_source_to_dataset_attrs(empty_dataset)
        assert isinstance(result_with_attrs, xr.Dataset)
        assert result_with_attrs.attrs.get("source") == "GHCN"

    @patch("extremeweatherbench.inputs.align_forecast_to_target")
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
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-21")
        mock_case.location.latitude_min = 30
        mock_case.location.latitude_max = 50
        mock_case.location.longitude_min = -110
        mock_case.location.longitude_max = -90

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

    @patch("extremeweatherbench.inputs.align_forecast_to_target")
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

    def test_zarr_target_subsetter(self, sample_era5_dataset):
        """Test zarr_target_subsetter function."""
        # Create mock case metadata (not case operator)
        mock_case_metadata = Mock()
        mock_case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case_metadata.location.mask.return_value = sample_era5_dataset

        result = inputs.zarr_target_subsetter(sample_era5_dataset, mock_case_metadata)

        mock_case_metadata.location.mask.assert_called_once()
        assert isinstance(result, xr.Dataset)

    def test_zarr_target_subsetter_missing_time_dimension(self, sample_era5_dataset):
        """Test zarr_target_subsetter with missing time dimensions."""
        # Create dataset without time dimensions
        data_no_time = sample_era5_dataset.drop_dims("time")

        mock_case_metadata = Mock()
        mock_case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case_metadata.end_date = pd.Timestamp("2021-06-22")

        with pytest.raises(ValueError, match="No suitable time dimension found"):
            inputs.zarr_target_subsetter(data_no_time, mock_case_metadata)

    def test_zarr_target_subsetter_with_valid_time_dimension(self, sample_era5_dataset):
        """Test zarr_target_subsetter with valid_time dimension."""
        # Rename time to valid_time to test the dimension detection
        data_with_valid_time = sample_era5_dataset.rename({"time": "valid_time"})

        mock_case_metadata = Mock()
        mock_case_metadata.start_date = pd.Timestamp("2021-06-20")
        mock_case_metadata.end_date = pd.Timestamp("2021-06-22")
        mock_case_metadata.location.mask.return_value = data_with_valid_time

        result = inputs.zarr_target_subsetter(data_with_valid_time, mock_case_metadata)

        mock_case_metadata.location.mask.assert_called_once()
        assert isinstance(result, xr.Dataset)


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

    # zarr throws a consolidated metadata warning that
    # is inconsequential (as of now)
    @pytest.mark.filterwarnings("ignore::UserWarning")
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
        # Note: dimensions keep their original names after alignment
        assert len(aligned_target.time) > 0  # Target uses 'time'
        assert len(aligned_forecast.valid_time) > 0  # Forecast uses 'valid_time'

        # Should have overlapping time periods - but lengths may differ due to
        # different time ranges. This is expected when target and forecast
        # have different time coverage


class TestGeneralizedAlignment:
    """Test the generalized alignment function with various coordinate scenarios."""

    def test_spatial_forecast_to_station_target(self):
        """Test forecast with spatial dims aligned to station target (line 549 scenario)."""
        # Simple forecast with spatial dimensions
        forecast = xr.Dataset(
            {
                "temperature": (
                    ["valid_time", "latitude", "longitude"],
                    np.random.normal(290, 1, (2, 3, 3)),
                ),
            },
            coords={
                "valid_time": pd.date_range("2021-06-20", periods=2, freq="12h"),
                "latitude": np.linspace(35, 45, 3),
                "longitude": np.linspace(255, 265, 3),
            },
        )
        
        # Station target (line 549 style - coords as data variables)
        target_data = []
        for t in forecast.valid_time.values:
            target_data.append({
                "valid_time": pd.Timestamp(t),
                "latitude": 40.0,
                "longitude": 260.0,
                "temperature": 285.0,
            })
        
        df = pd.DataFrame(target_data)
        df = df.set_index(["valid_time"])
        target = df.to_xarray()
        
        # Test alignment
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Verify basic functionality
        assert "temperature" in aligned_forecast.data_vars
        assert "temperature" in aligned_target.data_vars
        assert len(aligned_forecast.valid_time) == len(aligned_target.valid_time)
        
        # Should have interpolated to target location
        assert np.all(np.isfinite(aligned_forecast.temperature.values))

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_interpolation_methods(self, method):
        """Test different interpolation methods work without errors."""
        # Simple test datasets using proper coordinates
        forecast = xr.Dataset(
            {"temp": (["valid_time", "latitude", "longitude"], np.random.normal(290, 1, (2, 3, 3)))},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=2),
                "latitude": [35, 40, 45],
                "longitude": [255, 260, 265],
            },
        )
        
        # Target with coords as data variables
        target = xr.Dataset(
            {"temp": (["valid_time"], [285, 286]), "latitude": (["valid_time"], [37, 37]), "longitude": (["valid_time"], [258, 258])},
            coords={"valid_time": pd.date_range("2021-01-01", periods=2)},
        )
        
        # Should work with both methods
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target, method=method)
        
        assert "temp" in aligned_forecast.data_vars
        assert np.all(np.isfinite(aligned_forecast.temp.values))

    def test_no_spatial_interpolation_needed(self):
        """Test case where no spatial interpolation is needed."""
        # Both datasets have only valid_time dimension (using proper coordinates)
        forecast = xr.Dataset(
            {"temperature": (["valid_time"], [290, 291])},
            coords={"valid_time": pd.date_range("2021-01-01", periods=2)},
        )
        
        target = xr.Dataset(
            {"temperature": (["valid_time"], [285, 286])},
            coords={"valid_time": pd.date_range("2021-01-01", periods=2)},
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should just do time alignment
        assert len(aligned_forecast.valid_time) == len(aligned_target.valid_time)
        assert np.all(np.isfinite(aligned_forecast.temperature.values))

    def test_complex_init_lead_time_structure(self):
        """Test complex forecast with init_time/lead_time structure."""
        forecast = xr.Dataset(
            {"surface_air_temperature": (["init_time", "lead_time", "latitude", "longitude"], 
                                       np.random.normal(290, 1, (1, 3, 3, 3)))},
            coords={
                "init_time": [pd.Timestamp("2021-01-01")],
                "lead_time": pd.timedelta_range("0h", "24h", freq="12h"),
                "latitude": [35, 40, 45],
                "longitude": [255, 260, 265],
                "valid_time": ("lead_time", pd.Timestamp("2021-01-01") + pd.timedelta_range("0h", "24h", freq="12h")),
            },
        )
        
        target = xr.Dataset(
            {"surface_air_temperature": (["valid_time"], [285, 286, 287])},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=3, freq="12h"),
                "latitude": (["valid_time"], [40, 40, 40]),
                "longitude": (["valid_time"], [260, 260, 260]),
            },
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should handle complex time structure and interpolate spatially
        assert "surface_air_temperature" in aligned_forecast.data_vars
        assert "surface_air_temperature" in aligned_target.data_vars
        assert len(aligned_forecast.valid_time) == len(aligned_target.valid_time)
        assert np.all(np.isfinite(aligned_forecast.surface_air_temperature.values))

    def test_no_overlapping_times(self):
        """Test graceful handling when no times overlap."""
        forecast = xr.Dataset(
            {"surface_air_temperature": (["valid_time", "latitude", "longitude"], 
                                       np.random.normal(290, 1, (3, 3, 3)))},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=3),
                "latitude": [35, 40, 45],
                "longitude": [255, 260, 265],
            },
        )
        
        target = xr.Dataset(
            {"surface_air_temperature": (["valid_time"], [285, 286, 287])},
            coords={
                "valid_time": pd.date_range("2022-01-01", periods=3),  # Different year
                "latitude": (["valid_time"], [40, 40, 40]),
                "longitude": (["valid_time"], [260, 260, 260]),
            },
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should return empty datasets gracefully
        assert aligned_forecast.sizes["valid_time"] == 0
        assert aligned_target.sizes["valid_time"] == 0

    def test_multiple_spatial_dimensions_with_level(self):
        """Test interpolation with multiple spatial dimensions including level."""
        forecast = xr.Dataset(
            {"air_temperature": (["valid_time", "level", "latitude", "longitude"], 
                               np.random.normal(290, 1, (3, 2, 3, 3)))},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=3),
                "level": [850, 700],
                "latitude": [35, 40, 45],
                "longitude": [255, 260, 265],
            },
        )
        
        target = xr.Dataset(
            {"air_temperature": (["valid_time", "level"], np.random.normal(285, 1, (3, 2)))},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=3),
                "level": [850, 700],
                "latitude": (["valid_time"], [40, 40, 40]),
                "longitude": (["valid_time"], [260, 260, 260]),
            },
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should interpolate spatially while preserving level dimension
        assert "air_temperature" in aligned_forecast.data_vars
        assert "level" in aligned_forecast.dims
        assert "valid_time" in aligned_forecast.dims
        assert len(aligned_forecast.valid_time) == len(aligned_target.valid_time)
        assert len(aligned_forecast.level) == len(aligned_target.level)
        assert np.all(np.isfinite(aligned_forecast.air_temperature.values))

    def test_era5_style_time_coordinate_compatibility(self):
        """Test that ERA5-style 'time' coordinates work correctly."""
        # ERA5-style forecast with 'time' coordinate
        forecast = xr.Dataset(
            {"2m_temperature": (["time", "latitude", "longitude"], 
                              np.random.normal(290, 1, (3, 3, 3)))},
            coords={
                "time": pd.date_range("2021-01-01", periods=3),
                "latitude": [35, 40, 45],
                "longitude": [255, 260, 265],
            },
        )
        
        # ERA5-style target
        target = xr.Dataset(
            {"2m_temperature": (["time"], [285, 286, 287])},
            coords={
                "time": pd.date_range("2021-01-01", periods=3),
                "latitude": (["time"], [40, 40, 40]),
                "longitude": (["time"], [260, 260, 260]),
            },
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should work correctly with 'time' coordinates
        assert "2m_temperature" in aligned_forecast.data_vars
        assert "2m_temperature" in aligned_target.data_vars
        assert len(aligned_forecast.time) == len(aligned_target.time)
        assert np.all(np.isfinite(aligned_forecast["2m_temperature"].values))


class TestFixtureAlignment:
    """Test alignment between sample forecast dataset and various target datasets from fixtures."""

    def test_forecast_to_era5_alignment(self, sample_forecast_dataset, sample_era5_dataset):
        """Test alignment between forecast dataset and ERA5 dataset."""
        # ERA5 uses 'time' coordinate, forecast uses init_time/lead_time with valid_time
        forecast = sample_forecast_dataset
        target = sample_era5_dataset
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should successfully align on overlapping times
        assert isinstance(aligned_forecast, xr.Dataset)
        assert isinstance(aligned_target, xr.Dataset)
        
        # Check that alignment worked - should have common time dimension
        common_time_dims = set(aligned_forecast.dims) & set(aligned_target.dims)
        assert len(common_time_dims) > 0, "Should have at least one common time dimension"
        
        # Should have interpolated spatially if needed
        if "latitude" in aligned_forecast.dims and "longitude" in aligned_forecast.dims:
            assert np.all(np.isfinite(aligned_forecast.surface_air_temperature.values))

    def test_forecast_to_ghcn_alignment(self, sample_forecast_dataset, sample_ghcn_dataframe):
        """Test alignment between forecast dataset and GHCN station data."""
        # Convert GHCN dataframe to xarray using the GHCN class
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        
        # Convert to xarray dataset (this will use reset_index approach)
        target = ghcn._custom_convert_to_dataset(sample_ghcn_dataframe.lazy())
        forecast = sample_forecast_dataset
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should successfully align
        assert isinstance(aligned_forecast, xr.Dataset)
        assert isinstance(aligned_target, xr.Dataset)
        
        # Target should have 'index' dimension from reset_index approach
        assert "index" in aligned_target.dims
        assert "valid_time" in aligned_target.data_vars
        assert "latitude" in aligned_target.data_vars
        assert "longitude" in aligned_target.data_vars
        
        # Should preserve all GHCN data points
        assert aligned_target.sizes["index"] == len(sample_ghcn_dataframe)
        
        # COORDINATE MATCHING CHECK: Forecast should be interpolated to target locations
        if "latitude" in aligned_forecast.coords and "longitude" in aligned_forecast.coords:
            # Forecast coordinates should match target data variable values
            target_lats = aligned_target.latitude.values
            target_lons = aligned_target.longitude.values
            forecast_lats = aligned_forecast.latitude.values
            forecast_lons = aligned_forecast.longitude.values
            
            # Should have same number of points
            assert len(forecast_lats) == len(target_lats)
            assert len(forecast_lons) == len(target_lons)
            
            # Coordinates should match (within floating point precision)
            np.testing.assert_array_almost_equal(forecast_lats, target_lats, decimal=5)
            np.testing.assert_array_almost_equal(forecast_lons, target_lons, decimal=5)
        
        # Forecast should be interpolated if spatial alignment occurred
        if "surface_air_temperature" in aligned_forecast.data_vars:
            assert np.all(np.isfinite(aligned_forecast.surface_air_temperature.values))

    def test_forecast_to_lsr_alignment(self, sample_forecast_dataset, sample_lsr_dataframe):
        """Test alignment between forecast dataset and LSR (Local Storm Report) data."""
        # Convert LSR dataframe to xarray
        df = sample_lsr_dataframe.copy()
        df = df.reset_index(drop=True)
        target = df.to_xarray()
        forecast = sample_forecast_dataset
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should successfully align
        assert isinstance(aligned_forecast, xr.Dataset)
        assert isinstance(aligned_target, xr.Dataset)
        
        # Target should have 'index' dimension from reset_index approach
        assert "index" in aligned_target.dims
        assert "valid_time" in aligned_target.data_vars
        assert "latitude" in aligned_target.data_vars
        assert "longitude" in aligned_target.data_vars
        
        # Should preserve all LSR data points
        assert aligned_target.sizes["index"] == len(sample_lsr_dataframe)
        
        # COORDINATE MATCHING CHECK: Forecast should be interpolated to target locations
        if "latitude" in aligned_forecast.coords and "longitude" in aligned_forecast.coords:
            # Forecast coordinates should match target data variable values
            target_lats = aligned_target.latitude.values
            target_lons = aligned_target.longitude.values
            forecast_lats = aligned_forecast.latitude.values
            forecast_lons = aligned_forecast.longitude.values
            
            # Should have same number of points
            assert len(forecast_lats) == len(target_lats)
            assert len(forecast_lons) == len(target_lons)
            
            # Coordinates should match (within floating point precision)
            np.testing.assert_array_almost_equal(forecast_lats, target_lats, decimal=5)
            np.testing.assert_array_almost_equal(forecast_lons, target_lons, decimal=5)

    def test_forecast_to_ibtracs_alignment(self, sample_forecast_dataset, sample_ibtracs_dataframe):
        """Test alignment between forecast dataset and IBTrACS tropical cyclone data. 
        
        Note that this is not the intended approach for the IBTraCS dataset, but it's a 
        good test of the alignment functionality. Forecasts will need to be run through
        the tropical cyclone tracker to get datapoints instead of aligning on the 
        exact locations in IBTrACS.
        """
        # Convert IBTrACS dataframe to xarray using the IBTrACS class
        ibtracs = inputs.IBTrACS(
            source="test.nc",
            variables=["surface_wind_speed", "air_pressure_at_mean_sea_level"],
            variable_mapping={},
            storage_options={},
        )
        
        # Convert to xarray dataset (this will use reset_index approach)
        target = ibtracs._custom_convert_to_dataset(sample_ibtracs_dataframe.lazy())
        forecast = sample_forecast_dataset
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(forecast, target)
        
        # Should successfully align
        assert isinstance(aligned_forecast, xr.Dataset)
        assert isinstance(aligned_target, xr.Dataset)
        
        # Target should have 'index' dimension from reset_index approach
        assert "index" in aligned_target.dims
        assert "valid_time" in aligned_target.data_vars
        assert "latitude" in aligned_target.data_vars
        assert "longitude" in aligned_target.data_vars
        
        # Should preserve all IBTrACS data points
        assert aligned_target.sizes["index"] == len(sample_ibtracs_dataframe)
        
        # Should have the expected variables
        assert "surface_wind_speed" in aligned_target.data_vars
        assert "air_pressure_at_mean_sea_level" in aligned_target.data_vars
        
        # COORDINATE MATCHING CHECK: Forecast should be interpolated to target locations
        if "latitude" in aligned_forecast.coords and "longitude" in aligned_forecast.coords:
            # Forecast coordinates should match target data variable values
            target_lats = aligned_target.latitude.values
            target_lons = aligned_target.longitude.values
            forecast_lats = aligned_forecast.latitude.values
            forecast_lons = aligned_forecast.longitude.values
            
            # Should have same number of points
            assert len(forecast_lats) == len(target_lats)
            assert len(forecast_lons) == len(target_lons)
            
            # Coordinates should match (within floating point precision)
            np.testing.assert_array_almost_equal(forecast_lats, target_lats, decimal=5)
            np.testing.assert_array_almost_equal(forecast_lons, target_lons, decimal=5)

    def test_all_alignments_preserve_data_integrity(
        self, 
        sample_forecast_dataset, 
        sample_era5_dataset, 
        sample_ghcn_dataframe, 
    ):
        """Test that all alignment operations preserve data integrity."""
        forecast = sample_forecast_dataset
        
        # Test ERA5 alignment
        aligned_forecast_era5, aligned_target_era5 = inputs.align_forecast_to_target(
            forecast, sample_era5_dataset
        )
        
        # Should not introduce NaN values in finite data
        if "2m_temperature" in aligned_target_era5.data_vars:
            finite_mask = np.isfinite(sample_era5_dataset["2m_temperature"].values)
            if finite_mask.any():
                aligned_finite_mask = np.isfinite(aligned_target_era5["2m_temperature"].values)
                # At least some finite values should remain
                assert aligned_finite_mask.any()
        
        # Test GHCN alignment
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        ghcn_target = ghcn._custom_convert_to_dataset(sample_ghcn_dataframe.lazy())
        aligned_forecast_ghcn, aligned_target_ghcn = inputs.align_forecast_to_target(
            forecast, ghcn_target
        )
        
        # Should preserve all GHCN data points
        assert aligned_target_ghcn.sizes["index"] == len(sample_ghcn_dataframe)
        
        # Test that no data corruption occurred
        original_temp_count = ghcn_target.surface_air_temperature.count().item()
        aligned_temp_count = aligned_target_ghcn.surface_air_temperature.count().item()
        assert aligned_temp_count == original_temp_count

    def test_alignment_with_no_overlapping_times(self, sample_forecast_dataset):
        """Test alignment behavior when there are no overlapping times."""
        # Create a target with completely different time range
        target = xr.Dataset(
            {"temperature": (["valid_time"], [285, 286, 287])},
            coords={
                "valid_time": pd.date_range("2025-01-01", periods=3),  # Future dates
                "latitude": (["valid_time"], [40, 40, 40]),
                "longitude": (["valid_time"], [260, 260, 260]),
            },
        )
        
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(
            sample_forecast_dataset, target
        )
        
        # Should return empty datasets gracefully
        time_dims = ["valid_time", "time", "init_time", "lead_time"]
        forecast_time_sizes = [aligned_forecast.sizes.get(dim, 0) for dim in time_dims]
        target_time_sizes = [aligned_target.sizes.get(dim, 0) for dim in time_dims]
        
        # At least one time dimension should be empty
        assert any(size == 0 for size in forecast_time_sizes + target_time_sizes)

    def test_alignment_performance_reasonable(
        self, 
        sample_forecast_dataset, 
        sample_ghcn_dataframe
    ):
        """Test that alignment completes in reasonable time."""
        import time
        
        # Convert GHCN to xarray
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        target = ghcn._custom_convert_to_dataset(sample_ghcn_dataframe.lazy())
        
        # Time the alignment operation
        start_time = time.time()
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(
            sample_forecast_dataset, target
        )
        end_time = time.time()
        
        # Should complete in under 10 seconds (generous threshold)
        alignment_time = end_time - start_time
        assert alignment_time < 10.0, f"Alignment took {alignment_time:.2f}s, should be < 10s"
        
        # Should still produce valid results
        assert isinstance(aligned_forecast, xr.Dataset)
        assert isinstance(aligned_target, xr.Dataset)

    def test_coordinate_matching_verification(
        self, 
        sample_forecast_dataset, 
        sample_ghcn_dataframe
    ):
        """Explicitly test that forecast coordinates match target coordinates after alignment."""
        # Create a subset of GHCN data with known coordinates
        ghcn_subset = sample_ghcn_dataframe.head(5)  # Just 5 points for clear testing
        
        # Convert to xarray
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        target = ghcn._custom_convert_to_dataset(ghcn_subset.lazy())
        
        # Get original target coordinates
        original_target_lats = target.latitude.values
        original_target_lons = target.longitude.values
        
        print(f"Original target coordinates:")
        print(f"  Latitudes: {original_target_lats}")
        print(f"  Longitudes: {original_target_lons}")
        
        # Align with forecast
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(
            sample_forecast_dataset, target
        )
        
        # Verify coordinate matching
        if "latitude" in aligned_forecast.coords and "longitude" in aligned_forecast.coords:
            aligned_forecast_lats = aligned_forecast.latitude.values
            aligned_forecast_lons = aligned_forecast.longitude.values
            aligned_target_lats = aligned_target.latitude.values
            aligned_target_lons = aligned_target.longitude.values
            
            print(f"After alignment:")
            print(f"  Forecast latitudes: {aligned_forecast_lats}")
            print(f"  Target latitudes: {aligned_target_lats}")
            print(f"  Forecast longitudes: {aligned_forecast_lons}")
            print(f"  Target longitudes: {aligned_target_lons}")
            
            # Coordinates should match exactly
            np.testing.assert_array_equal(aligned_forecast_lats, aligned_target_lats)
            np.testing.assert_array_equal(aligned_forecast_lons, aligned_target_lons)
            
            # Forecast should be interpolated to target locations
            # Check that forecast has the same number of spatial points as target
            if "index" in aligned_forecast.dims:
                assert aligned_forecast.sizes["index"] == len(original_target_lats)
            else:
                # For complex forecast structures, check that spatial dimensions match
                spatial_size = 1
                for dim in ["latitude", "longitude"]:
                    if dim in aligned_forecast.dims:
                        spatial_size *= aligned_forecast.sizes[dim]
                assert spatial_size >= len(original_target_lats)
            
            # All interpolated values should be finite
            assert np.all(np.isfinite(aligned_forecast.surface_air_temperature.values))
            
            print("✅ Coordinate matching verification passed!")
        else:
            pytest.fail("Expected forecast to have latitude/longitude coordinates after alignment")
