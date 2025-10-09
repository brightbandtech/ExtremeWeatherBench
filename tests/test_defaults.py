"""Tests for defaults module."""

import numpy as np
import pytest
import xarray as xr

from extremeweatherbench import defaults, inputs, metrics


class TestDefaults:
    """Test the defaults module."""

    def test_output_columns_exists(self):
        """Test that OUTPUT_COLUMNS is defined and contains expected columns."""
        expected_columns = [
            "value",
            "lead_time",
            "init_time",
            "target_variable",
            "metric",
            "forecast_source",
            "target_source",
            "case_id_number",
            "event_type",
        ]
        assert hasattr(defaults, "OUTPUT_COLUMNS")
        assert defaults.OUTPUT_COLUMNS == expected_columns

    def test_preprocess_bb_cira_forecast_dataset(self):
        """Test the _preprocess_bb_cira_forecast_dataset function."""

        # Create a mock dataset with 'time' coordinate matching expected output size
        # The function creates lead_time with 41 values (0 to 240 by 6)
        time_data = np.array([i for i in range(0, 241, 6)], dtype="timedelta64[h]")
        temp_data = np.random.random(len(time_data))
        mock_ds = xr.Dataset(
            {"temperature": (["time"], temp_data)}, coords={"time": time_data}
        )

        result = defaults._preprocess_bb_cira_forecast_dataset(mock_ds)

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
                # The metric should be a class from the metrics module
                assert hasattr(metrics, metric.__name__)

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
        assert hasattr(defaults, "cira_heatwave_forecast")
        assert hasattr(defaults, "cira_freeze_forecast")
        assert isinstance(defaults.cira_heatwave_forecast, inputs.KerchunkForecast)
        assert isinstance(defaults.cira_freeze_forecast, inputs.KerchunkForecast)

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

        expected_variables = [
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
        ]
        assert target.variables == expected_variables

        expected_mapping = {
            "2m_temperature": "surface_air_temperature",
            "10m_u_component_of_wind": "surface_eastward_wind",
            "10m_v_component_of_wind": "surface_northward_wind",
            "time": "valid_time",
        }
        for key, value in expected_mapping.items():
            assert target.variable_mapping[key] == value

    def test_cira_forecasts_have_preprocess_function(self):
        """Test that CIRA forecasts have the preprocess function set."""
        assert defaults.cira_heatwave_forecast.preprocess is not None
        assert defaults.cira_freeze_forecast.preprocess is not None

        # Test that the preprocess function is the expected one
        assert (
            defaults.cira_heatwave_forecast.preprocess
            == defaults._preprocess_bb_cira_forecast_dataset
        )
        assert (
            defaults.cira_freeze_forecast.preprocess
            == defaults._preprocess_bb_cira_forecast_dataset
        )

    def test_get_brightband_evaluation_objects_no_exceptions(self):
        """Test that get_brightband_evaluation_objects runs without exceptions."""
        try:
            result = defaults.get_brightband_evaluation_objects()
            # Basic validation that it returns something reasonable
            assert isinstance(result, list)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(f"get_brightband_evaluation_objects raised an exception: {e}")
