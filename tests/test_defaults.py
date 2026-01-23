"""Tests for defaults module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from extremeweatherbench import defaults, inputs, metrics


class TestGetCIRAIcechunk:
    """Tests for get_cira_icechunk function."""

    def test_invalid_model_name_raises_value_error(self):
        """Test that an invalid model name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            defaults.get_cira_icechunk(model_name="INVALID_MODEL")

        assert "INVALID_MODEL" in str(exc_info.value)
        assert "CIRA_MODEL_NAMES" in str(exc_info.value)

    def test_empty_model_name_raises_value_error(self):
        """Test that an empty model name raises ValueError."""
        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="")

    def test_none_model_name_raises_error(self):
        """Test that None as model name raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            defaults.get_cira_icechunk(model_name=None)  # type: ignore

    def test_case_sensitive_model_name(self):
        """Test that model name matching is case-sensitive."""
        # Lowercase version of a valid model name should fail
        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="four_v200_gfs")

        # Mixed case should fail
        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="Four_V200_GFS")

    def test_partial_model_name_raises_value_error(self):
        """Test that partial model names are rejected."""
        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="FOUR")

        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="GFS")

    def test_model_name_with_extra_chars_raises_value_error(self):
        """Test that model names with extra characters are rejected."""
        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name="FOUR_v200_GFS_extra")

        with pytest.raises(ValueError):
            defaults.get_cira_icechunk(model_name=" FOUR_v200_GFS")

    def test_error_message_lists_valid_model_names(self):
        """Test that the error message includes the list of valid model names."""
        with pytest.raises(ValueError) as exc_info:
            defaults.get_cira_icechunk(model_name="BAD_MODEL")

        error_msg = str(exc_info.value)
        # Check that at least some valid model names are shown in the error
        assert "FOUR_v200_GFS" in error_msg or "CIRA_MODEL_NAMES" in error_msg

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_valid_model_name_four_v200_gfs(
        self, mock_forecast, mock_open, mock_storage
    ):
        """Test that FOUR_v200_GFS is a valid model name."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        result = defaults.get_cira_icechunk(model_name="FOUR_v200_GFS")

        assert result is not None
        mock_storage.assert_called_once()
        mock_open.assert_called_once()

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_valid_model_name_auro_v100_gfs(
        self, mock_forecast, mock_open, mock_storage
    ):
        """Test that AURO_v100_GFS is a valid model name."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        result = defaults.get_cira_icechunk(model_name="AURO_v100_GFS")

        assert result is not None

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_all_cira_model_names_are_valid(
        self, mock_forecast, mock_open, mock_storage
    ):
        """Test that all model names in CIRA_MODEL_NAMES are accepted."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        for model_name in defaults.CIRA_MODEL_NAMES:
            result = defaults.get_cira_icechunk(model_name=model_name)
            assert result is not None, f"Model {model_name} should be valid"

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_custom_name_parameter(self, mock_forecast, mock_open, mock_storage):
        """Test that a custom name parameter is passed to XarrayForecast."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        defaults.get_cira_icechunk(model_name="FOUR_v200_GFS", name="CustomName")

        # Check that XarrayForecast was called with the custom name
        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["name"] == "CustomName"

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_default_name_uses_model_name(self, mock_forecast, mock_open, mock_storage):
        """Test that name defaults to model_name when not provided."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        defaults.get_cira_icechunk(model_name="FOUR_v200_GFS")

        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["name"] == "FOUR_v200_GFS"

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_empty_variables_list(self, mock_forecast, mock_open, mock_storage):
        """Test that empty variables list is valid."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        result = defaults.get_cira_icechunk(model_name="FOUR_v200_GFS", variables=[])

        assert result is not None
        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["variables"] == []

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_custom_variables_list(self, mock_forecast, mock_open, mock_storage):
        """Test that a custom variables list is passed through."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        variables = ["surface_air_temperature", "air_pressure"]
        defaults.get_cira_icechunk(model_name="FOUR_v200_GFS", variables=variables)

        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["variables"] == variables

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_custom_preprocess_function(self, mock_forecast, mock_open, mock_storage):
        """Test that a custom preprocess function is passed through."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        def custom_preprocess(ds: xr.Dataset) -> xr.Dataset:
            return ds

        defaults.get_cira_icechunk(
            model_name="FOUR_v200_GFS", preprocess=custom_preprocess
        )

        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["preprocess"] == custom_preprocess

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_returns_xarray_forecast_object(
        self, mock_forecast, mock_open, mock_storage
    ):
        """Test that the function returns an XarrayForecast object."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        expected_forecast = MagicMock()
        mock_forecast.return_value = expected_forecast

        result = defaults.get_cira_icechunk(model_name="FOUR_v200_GFS")

        assert result is expected_forecast

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_gcs_storage_configuration(self, mock_forecast, mock_open, mock_storage):
        """Test that GCS storage is configured with correct parameters."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        defaults.get_cira_icechunk(model_name="FOUR_v200_GFS")

        mock_storage.assert_called_once_with(
            bucket="extremeweatherbench", prefix="cira-icechunk", anonymous=True
        )

    @patch("extremeweatherbench.defaults.icechunk.gcs_storage")
    @patch("extremeweatherbench.defaults.inputs.open_icechunk_dataset_from_datatree")
    @patch("extremeweatherbench.defaults.inputs.XarrayForecast")
    def test_uses_cira_variable_mapping(self, mock_forecast, mock_open, mock_storage):
        """Test that CIRA metadata variable mapping is used."""
        mock_storage.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        mock_forecast.return_value = MagicMock()

        defaults.get_cira_icechunk(model_name="FOUR_v200_GFS")

        call_kwargs = mock_forecast.call_args[1]
        assert call_kwargs["variable_mapping"] == inputs.CIRA_metadata_variable_mapping


class TestCiraModelNames:
    """Tests for CIRA_MODEL_NAMES constant."""

    def test_cira_model_names_is_list(self):
        """Test that CIRA_MODEL_NAMES is a list."""
        assert isinstance(defaults.CIRA_MODEL_NAMES, list)

    def test_cira_model_names_not_empty(self):
        """Test that CIRA_MODEL_NAMES is not empty."""
        assert len(defaults.CIRA_MODEL_NAMES) > 0

    def test_cira_model_names_contains_expected_models(self):
        """Test that CIRA_MODEL_NAMES contains expected model names."""
        expected_models = [
            "FOUR_v200_GFS",
            "FOUR_v200_IFS",
            "AURO_v100_GFS",
            "AURO_v100_IFS",
            "PANG_v100_GFS",
            "PANG_v100_IFS",
            "GRAP_v100_GFS",
            "GRAP_v100_IFS",
        ]
        for model in expected_models:
            assert model in defaults.CIRA_MODEL_NAMES

    def test_cira_model_names_all_strings(self):
        """Test that all entries in CIRA_MODEL_NAMES are strings."""
        for model in defaults.CIRA_MODEL_NAMES:
            assert isinstance(model, str)

    def test_cira_model_names_no_duplicates(self):
        """Test that CIRA_MODEL_NAMES has no duplicate entries."""
        assert len(defaults.CIRA_MODEL_NAMES) == len(set(defaults.CIRA_MODEL_NAMES))


class TestDefaults:
    """Test the defaults module."""

    def test_preprocess_cira_forecast_dataset(self):
        """Test the _preprocess_cira_forecast_dataset function."""

        # Create a mock dataset with 'time' coordinate matching expected output size
        # The function creates lead_time with 41 values (0 to 240 by 6)
        time_data = np.array([i for i in range(0, 241, 6)], dtype="timedelta64[h]")
        temp_data = np.random.random(len(time_data))
        mock_ds = xr.Dataset(
            {"temperature": (["time"], temp_data)}, coords={"time": time_data}
        )

        result = defaults._preprocess_cira_forecast_dataset(mock_ds)

        # Check that 'time' was renamed to 'lead_time'
        assert "lead_time" in result.coords
        assert "time" not in result.coords

        # Check that lead_time has the expected values (0 to 240 by 6)
        expected_lead_times = np.array(
            [i for i in range(0, 241, 6)], dtype="timedelta64[h]"
        ).astype("timedelta64[ns]")
        np.testing.assert_array_equal(result["lead_time"].values, expected_lead_times)

    def test_get_brightband_evaluation_objects_returns_list(self):
        """Test that get_brightband_evaluation_objects returns a list."""
        result = defaults.get_brightband_evaluation_objects()
        assert isinstance(result, list)

    def test_get_brightband_evaluation_objects_list_not_empty(self):
        """Test that get_brightband_evaluation_objects returns a non-empty list."""
        result = defaults.get_brightband_evaluation_objects()
        assert len(result) > 0

    def test_get_brightband_evaluation_objects_contains_evaluation_objects(self):
        """Test that the returned list contains EvaluationObject instances."""
        result = defaults.get_brightband_evaluation_objects()

        # Check that all items in the list are EvaluationObject instances
        for item in result:
            assert isinstance(item, inputs.EvaluationObject)

    def test_get_brightband_evaluation_objects_has_expected_event_types(self):
        """Test that evaluation objects have expected event types."""
        result = defaults.get_brightband_evaluation_objects()

        event_types = [obj.event_type for obj in result]

        # Should have heat_wave and freeze event types
        assert "heat_wave" in event_types
        assert "freeze" in event_types

        # Count occurrences (should have multiple of each type)
        heat_wave_count = event_types.count("heat_wave")
        freeze_count = event_types.count("freeze")

        assert heat_wave_count >= 1
        assert freeze_count >= 1

    def test_get_brightband_evaluation_objects_has_metrics(self):
        """Test that evaluation objects have metric lists."""
        result = defaults.get_brightband_evaluation_objects()

        for obj in result:
            assert hasattr(obj, "metric_list")
            assert isinstance(obj.metric_list, list)
            assert len(obj.metric_list) > 0

    def test_get_brightband_evaluation_objects_has_targets_and_forecasts(self):
        """Test that evaluation objects have targets and forecasts."""
        result = defaults.get_brightband_evaluation_objects()

        for obj in result:
            assert hasattr(obj, "target")
            assert hasattr(obj, "forecast")
            assert obj.target is not None
            assert obj.forecast is not None

    def test_get_brightband_evaluation_objects_imports_metrics_successfully(self):
        """Test that the function successfully imports and uses metrics module."""
        # This test verifies that the function can import metrics without error
        # and that the returned objects have metric classes from the metrics module
        result = defaults.get_brightband_evaluation_objects()

        # Verify that the function returns a list (basic functionality)
        assert isinstance(result, list)
        assert len(result) > 0

        for obj in result:
            assert len(obj.metric_list) > 0
            # Check that at least one metric is from the metrics module
            for metric in obj.metric_list:
                # The metric should be an instance from the metrics module
                assert hasattr(metrics, metric.__class__.__name__)

    def test_target_objects_exist(self):
        """Test that target objects are properly defined."""
        # Test ERA5 targets
        assert hasattr(defaults, "era5_heatwave_target")
        assert hasattr(defaults, "era5_freeze_target")
        assert isinstance(defaults.era5_heatwave_target, inputs.ERA5)
        assert isinstance(defaults.era5_freeze_target, inputs.ERA5)

        # Test GHCN targets
        assert hasattr(defaults, "ghcn_heatwave_target")
        assert hasattr(defaults, "ghcn_freeze_target")
        assert isinstance(defaults.ghcn_heatwave_target, inputs.GHCN)
        assert isinstance(defaults.ghcn_freeze_target, inputs.GHCN)

    def test_forecast_objects_exist(self):
        """Test that forecast objects are properly defined."""
        assert hasattr(defaults, "cira_fcnv2_heatwave_forecast")
        assert hasattr(defaults, "cira_fcnv2_freeze_forecast")
        assert hasattr(defaults, "cira_fcnv2_tropical_cyclone_forecast")
        assert hasattr(defaults, "cira_fcnv2_atmospheric_river_forecast")
        assert hasattr(defaults, "cira_fcnv2_severe_convection_forecast")
        assert isinstance(defaults.cira_fcnv2_heatwave_forecast, inputs.XarrayForecast)
        assert isinstance(defaults.cira_fcnv2_freeze_forecast, inputs.XarrayForecast)
        assert isinstance(
            defaults.cira_fcnv2_tropical_cyclone_forecast, inputs.XarrayForecast
        )
        assert isinstance(
            defaults.cira_fcnv2_atmospheric_river_forecast, inputs.XarrayForecast
        )
        assert isinstance(
            defaults.cira_fcnv2_severe_convection_forecast, inputs.XarrayForecast
        )

    def test_era5_heatwave_target_configuration(self):
        """Test ERA5 heatwave target configuration."""
        target = defaults.era5_heatwave_target

        assert target.variables == ["surface_air_temperature"]
        assert "2m_temperature" in target.variable_mapping
        assert target.variable_mapping["2m_temperature"] == "surface_air_temperature"
        assert "time" in target.variable_mapping
        assert target.variable_mapping["time"] == "valid_time"

    def test_era5_freeze_target_configuration(self):
        """Test ERA5 freeze target configuration."""
        target = defaults.era5_freeze_target

        expected_variables = ["surface_air_temperature"]
        assert target.variables == expected_variables

        expected_mapping = {
            "2m_temperature": "surface_air_temperature",
            "time": "valid_time",
        }
        for key, value in expected_mapping.items():
            assert target.variable_mapping[key] == value

    def test_get_brightband_evaluation_objects_no_exceptions(self):
        """Test that get_brightband_evaluation_objects runs without exceptions."""
        try:
            result = defaults.get_brightband_evaluation_objects()
            # Basic validation that it returns something reasonable
            assert isinstance(result, list)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"get_brightband_evaluation_objects raised an exception: {e}")


class TestCiraFcnv2PreprocessFunctions:
    """Tests that each cira_fcnv2 forecast has the correct preprocessing function."""

    def test_heatwave_forecast_has_default_preprocess(self):
        """Test that cira_fcnv2_heatwave_forecast uses default preprocess."""
        forecast = defaults.cira_fcnv2_heatwave_forecast
        assert forecast.preprocess == inputs._default_preprocess

    def test_freeze_forecast_has_default_preprocess(self):
        """Test that cira_fcnv2_freeze_forecast uses default preprocess."""
        forecast = defaults.cira_fcnv2_freeze_forecast
        assert forecast.preprocess == inputs._default_preprocess

    def test_tropical_cyclone_forecast_has_tc_preprocess(self):
        """Test that cira_fcnv2_tropical_cyclone_forecast uses TC preprocess."""
        forecast = defaults.cira_fcnv2_tropical_cyclone_forecast
        assert forecast.preprocess == defaults._preprocess_cira_tc_forecast_dataset

    def test_atmospheric_river_forecast_has_ar_preprocess(self):
        """Test that cira_fcnv2_atmospheric_river_forecast uses AR preprocess."""
        forecast = defaults.cira_fcnv2_atmospheric_river_forecast
        assert forecast.preprocess == defaults._preprocess_cira_ar_forecast_dataset

    def test_severe_convection_forecast_has_severe_preprocess(self):
        """Test that cira_fcnv2_severe_convection_forecast uses severe preprocess."""
        forecast = defaults.cira_fcnv2_severe_convection_forecast
        assert forecast.preprocess == defaults._preprocess_severe_cira_forecast_dataset

    def test_all_forecasts_have_preprocess_attribute(self):
        """Test that all cira_fcnv2 forecasts have a preprocess attribute set."""
        forecasts = [
            defaults.cira_fcnv2_heatwave_forecast,
            defaults.cira_fcnv2_freeze_forecast,
            defaults.cira_fcnv2_tropical_cyclone_forecast,
            defaults.cira_fcnv2_atmospheric_river_forecast,
            defaults.cira_fcnv2_severe_convection_forecast,
        ]
        for forecast in forecasts:
            assert hasattr(forecast, "preprocess")
            assert forecast.preprocess is not None
            assert callable(forecast.preprocess)

    def test_preprocess_functions_are_distinct_where_expected(self):
        """Test that different event types use different preprocess functions."""
        # TC, AR, and severe should have distinct preprocess functions
        tc_preprocess = defaults.cira_fcnv2_tropical_cyclone_forecast.preprocess
        ar_preprocess = defaults.cira_fcnv2_atmospheric_river_forecast.preprocess
        severe_preprocess = defaults.cira_fcnv2_severe_convection_forecast.preprocess

        assert tc_preprocess != ar_preprocess
        assert tc_preprocess != severe_preprocess
        # Note: AR and severe could be the same or different depending on impl

    def test_heatwave_and_freeze_use_same_preprocess(self):
        """Test that heatwave and freeze forecasts use the same preprocess."""
        heatwave_preprocess = defaults.cira_fcnv2_heatwave_forecast.preprocess
        freeze_preprocess = defaults.cira_fcnv2_freeze_forecast.preprocess
        assert heatwave_preprocess == freeze_preprocess
