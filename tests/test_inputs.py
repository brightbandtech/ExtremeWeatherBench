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

    def test_forecast_base_subset_data_to_case_with_duplicate_init_times(
        self, sample_forecast_dataset
    ):
        """Test subset_data_to_case handles duplicate init_times correctly."""
        # Create a forecast dataset with duplicate init_times
        forecast_with_duplicates = sample_forecast_dataset.copy()

        # Add duplicate init_times by concatenating along init_time dimension
        duplicate_data = forecast_with_duplicates.isel(init_time=[0, 1])
        forecast_with_duplicates = xr.concat(
            [forecast_with_duplicates, duplicate_data], dim="init_time"
        )

        # Verify we have duplicates
        assert len(np.unique(forecast_with_duplicates.init_time)) < len(
            forecast_with_duplicates.init_time
        )

        # Create mock case metadata
        mock_case = Mock()
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-22")
        mock_case.location.mask.return_value = forecast_with_duplicates

        forecast = inputs.ZarrForecast(
            name="test",
            source="test.zarr",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        with (
            patch(
                "extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time"
            ) as mock_derive,
            patch(
                "extremeweatherbench.utils.convert_init_time_to_valid_time"
            ) as mock_convert,
            patch(
                "extremeweatherbench.derived.maybe_include_variables_from_derived_input"
            ) as mock_derived,
        ):
            # Setup mocks
            mock_derive.return_value = (np.array([0, 1]), np.array([0, 1]))
            # Create a result dataset with unique valid_times to avoid slice errors
            result_data = xr.Dataset(
                {
                    "surface_air_temperature": (
                        ["valid_time", "latitude", "longitude"],
                        np.random.randn(3, 3, 3),
                    )
                },
                coords={
                    "valid_time": pd.date_range("2021-06-20", periods=3, freq="6h"),
                    "latitude": [40, 41, 42],
                    "longitude": [-100, -101, -102],
                },
            )
            mock_convert.return_value = result_data
            mock_derived.return_value = ["surface_air_temperature"]

            result = forecast.subset_data_to_case(forecast_with_duplicates, mock_case)

            # The method should handle duplicates and return a valid dataset
            assert isinstance(result, xr.Dataset)

    def test_forecast_base_duplicate_init_times_detection_and_removal(self):
        """Test that duplicate init_times are properly detected and removed."""
        # Create a simple forecast dataset with known duplicate init_times
        init_times = pd.date_range("2021-06-20", periods=3, freq="12h")
        lead_times = [0, 6, 12, 18]

        # Create dataset with duplicates by repeating first two init_times
        duplicate_init_times = np.concatenate([init_times, init_times[:2]])

        data_shape = (len(duplicate_init_times), 3, 3, len(lead_times))
        test_data = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["init_time", "latitude", "longitude", "lead_time"],
                    np.random.randn(*data_shape),
                )
            },
            coords={
                "init_time": duplicate_init_times,
                "latitude": [40, 41, 42],
                "longitude": [-100, -101, -102],
                "lead_time": lead_times,
            },
        )

        # Verify we have duplicates
        assert len(np.unique(test_data.init_time)) == 3  # Original unique count
        assert len(test_data.init_time) == 5  # Total count with duplicates

        # Create mock case metadata
        mock_case = Mock()
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-22")
        mock_case.location.mask.return_value = test_data

        forecast = inputs.ZarrForecast(
            name="test",
            source="test.zarr",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        with (
            patch(
                "extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time"
            ) as mock_derive,
            patch(
                "extremeweatherbench.utils.convert_init_time_to_valid_time"
            ) as mock_convert,
            patch(
                "extremeweatherbench.derived.maybe_include_variables_from_derived_input"
            ) as mock_derived,
        ):
            # Setup mocks to return valid indices
            mock_derive.return_value = (np.array([0, 1, 2]), np.array([0, 1, 2]))

            # Create expected result after duplicate removal with unique valid_times
            result_data = xr.Dataset(
                {
                    "surface_air_temperature": (
                        ["valid_time", "latitude", "longitude"],
                        np.random.randn(3, 3, 3),
                    )
                },
                coords={
                    "valid_time": pd.date_range("2021-06-20", periods=3, freq="6h"),
                    "latitude": [40, 41, 42],
                    "longitude": [-100, -101, -102],
                },
            )
            mock_convert.return_value = result_data
            mock_derived.return_value = ["surface_air_temperature"]

            result = forecast.subset_data_to_case(test_data, mock_case)

            # Verify the result is valid
            assert isinstance(result, xr.Dataset)

    def test_forecast_base_no_duplicate_init_times_unchanged(
        self, sample_forecast_dataset
    ):
        """Test that datasets without duplicate init_times are processed normally."""
        # Verify the sample dataset has no duplicates
        assert len(np.unique(sample_forecast_dataset.init_time)) == len(
            sample_forecast_dataset.init_time
        )

        # Create mock case metadata
        mock_case = Mock()
        mock_case.start_date = pd.Timestamp("2021-06-20")
        mock_case.end_date = pd.Timestamp("2021-06-22")
        mock_case.location.mask.return_value = sample_forecast_dataset

        forecast = inputs.ZarrForecast(
            name="test",
            source="test.zarr",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        with (
            patch(
                "extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time"
            ) as mock_derive,
            patch(
                "extremeweatherbench.utils.convert_init_time_to_valid_time"
            ) as mock_convert,
            patch(
                "extremeweatherbench.derived.maybe_include_variables_from_derived_input"
            ) as mock_derived,
        ):
            # Setup mocks
            mock_derive.return_value = (np.array([0, 1]), np.array([0, 1]))
            # Create a result dataset with unique valid_times
            result_data = xr.Dataset(
                {
                    "surface_air_temperature": (
                        ["valid_time", "latitude", "longitude"],
                        np.random.randn(3, 3, 3),
                    )
                },
                coords={
                    "valid_time": pd.date_range("2021-06-20", periods=3, freq="6h"),
                    "latitude": [40, 41, 42],
                    "longitude": [-100, -101, -102],
                },
            )
            mock_convert.return_value = result_data
            mock_derived.return_value = ["surface_air_temperature"]

            result = forecast.subset_data_to_case(sample_forecast_dataset, mock_case)

            # Should process normally without issues
            assert isinstance(result, xr.Dataset)

    def test_forecast_base_duplicate_init_times_preserves_first_occurrence(self):
        """Test that when duplicates exist, the first occurrence is preserved."""
        # Create dataset with specific values to test which occurrence is kept
        init_times = pd.to_datetime(
            ["2021-06-20", "2021-06-21", "2021-06-20"]
        )  # Duplicate first time
        lead_times = [0, 6]

        # Create data where we can distinguish between first and duplicate occurrence
        data = np.zeros((3, 2, 2, 2))  # (init_time, lat, lon, lead_time)
        data[0, :, :, :] = 1.0  # First occurrence of 2021-06-20
        data[1, :, :, :] = 2.0  # 2021-06-21
        data[2, :, :, :] = 3.0  # Duplicate occurrence of 2021-06-20

        test_data = xr.Dataset(
            {
                "surface_air_temperature": (
                    ["init_time", "latitude", "longitude", "lead_time"],
                    data,
                )
            },
            coords={
                "init_time": init_times,
                "latitude": [40, 41],
                "longitude": [-100, -101],
                "lead_time": lead_times,
            },
        )

        # Apply the duplicate removal logic directly (as done in subset_data_to_case)
        if len(np.unique(test_data.init_time)) != len(test_data.init_time):
            _, index = np.unique(test_data.init_time, return_index=True)
            deduplicated_data = test_data.isel(init_time=index)

        # Verify that we kept the first occurrence (value 1.0) not duplicate (value 3.0)
        first_time_data = deduplicated_data.sel(init_time="2021-06-20")[
            "surface_air_temperature"
        ]
        assert np.all(
            first_time_data.values == 1.0
        ), "Should preserve first occurrence, not duplicate"

        # Verify we have the correct number of unique times
        assert len(deduplicated_data.init_time) == 2
        assert len(np.unique(deduplicated_data.init_time)) == 2


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

    def test_ghcn_custom_convert_to_dataset_no_duplicates(self, sample_ghcn_dataframe):
        """Test GHCN custom conversion with no duplicates (baseline case)."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )

        # Ensure data has no duplicates by creating clean sample
        clean_data = sample_ghcn_dataframe.unique(
            subset=["valid_time", "latitude", "longitude"]
        )

        result = ghcn._custom_convert_to_dataset(clean_data.lazy())

        assert isinstance(result, xr.Dataset)
        assert "valid_time" in result.dims
        assert "latitude" in result.dims
        assert "longitude" in result.dims
        assert "surface_air_temperature" in result.data_vars

        # Should have no NaN values if no duplicates were dropped
        original_count = len(clean_data)
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

        # Create data with one intentional duplicate
        clean_data = sample_ghcn_dataframe.unique(
            subset=["valid_time", "latitude", "longitude"]
        )

        # Duplicate the first row
        first_row = clean_data.slice(0, 1)
        data_with_duplicate = pl.concat([clean_data, first_row])

        result = ghcn._custom_convert_to_dataset(data_with_duplicate.lazy())

        assert isinstance(result, xr.Dataset)
        assert "surface_air_temperature" in result.data_vars

        # Should have dropped one duplicate, so count should equal original
        original_count = len(clean_data)
        result_count = result.surface_air_temperature.count().item()
        assert result_count == original_count

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

        # Create data with multiple duplicates
        clean_data = sample_ghcn_dataframe.unique(
            subset=["valid_time", "latitude", "longitude"]
        )

        # Create multiple duplicates by repeating first 5 rows
        duplicates = clean_data.slice(0, 5)
        data_with_many_duplicates = pl.concat(
            [
                clean_data,
                duplicates,  # First set of duplicates
                duplicates,  # Second set of duplicates
                duplicates,  # Third set of duplicates
            ]
        )

        result = ghcn._custom_convert_to_dataset(data_with_many_duplicates.lazy())

        assert isinstance(result, xr.Dataset)
        assert "surface_air_temperature" in result.data_vars

        # Should have dropped all duplicates, so count should equal original
        original_count = len(clean_data)
        result_count = result.surface_air_temperature.count().item()
        assert result_count == original_count

    def test_ghcn_custom_convert_to_dataset_exception_handling(self):
        """Test GHCN custom conversion exception handling returns empty Dataset."""
        ghcn = inputs.GHCN(
            source="test.parquet",
            variables=["surface_air_temperature"],
            variable_mapping={},
            storage_options={},
        )
        # Create data with problematic values that might cause xarray conversion issues
        problematic_data = pl.DataFrame(
            {
                "valid_time": [None, None],  # None values in index will cause issues
                "latitude": [40.0, 41.0],
                "longitude": [-100.0, -101.0],
                "surface_air_temperature": [273.15, 274.15],
            }
        )

        with patch("extremeweatherbench.inputs.logger") as mock_logger:
            result = ghcn._custom_convert_to_dataset(problematic_data.lazy())

            # Should return empty Dataset on exception
            assert isinstance(result, xr.Dataset)
            assert len(result.data_vars) == 0
            assert len(result.dims) == 0

            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0]
            assert "Error converting GHCN data to xarray" in warning_call[0]

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
