"""Tests for the extremeweatherbench package __init__.py API."""

import types


class TestModuleImports:
    """Test that submodules are importable and are actual modules."""

    def test_calc_is_module(self):
        """Test that calc is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import calc

        assert isinstance(calc, types.ModuleType)

    def test_utils_is_module(self):
        """Test that utils is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import utils

        assert isinstance(utils, types.ModuleType)

    def test_metrics_is_module(self):
        """Test that metrics is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import metrics

        assert isinstance(metrics, types.ModuleType)

    def test_regions_is_module(self):
        """Test that regions is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import regions

        assert isinstance(regions, types.ModuleType)

    def test_derived_is_module(self):
        """Test that derived is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import derived

        assert isinstance(derived, types.ModuleType)

    def test_defaults_is_module(self):
        """Test that defaults is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import defaults

        assert isinstance(defaults, types.ModuleType)

    def test_cases_is_module(self):
        """Test that cases is an actual module, not a SimpleNamespace."""
        from extremeweatherbench import cases

        assert isinstance(cases, types.ModuleType)


class TestModuleAccessPatterns:
    """Test both import patterns work identically."""

    def test_ewb_dot_notation_equals_direct_import_calc(self):
        """Test ewb.calc is the same object as direct import."""
        import extremeweatherbench as ewb
        from extremeweatherbench import calc

        assert ewb.calc is calc

    def test_ewb_dot_notation_equals_direct_import_metrics(self):
        """Test ewb.metrics is the same object as direct import."""
        import extremeweatherbench as ewb
        from extremeweatherbench import metrics

        assert ewb.metrics is metrics

    def test_ewb_dot_notation_equals_direct_import_utils(self):
        """Test ewb.utils is the same object as direct import."""
        import extremeweatherbench as ewb
        from extremeweatherbench import utils

        assert ewb.utils is utils


class TestModuleLevelConstants:
    """Test that module-level constants are accessible."""

    def test_calc_g0_accessible(self):
        """Test that calc.g0 constant is accessible."""
        from extremeweatherbench import calc

        assert hasattr(calc, "g0")
        assert calc.g0 == 9.80665

    def test_calc_epsilon_accessible(self):
        """Test that calc.epsilon constant is accessible."""
        from extremeweatherbench import calc

        assert hasattr(calc, "epsilon")
        assert isinstance(calc.epsilon, float)


class TestPrivateFunctionAccess:
    """Test that private functions are accessible for testing purposes."""

    def test_calc_private_functions_accessible(self):
        """Test that private functions in calc are accessible."""
        from extremeweatherbench import calc

        assert hasattr(calc, "_is_true_landfall")
        assert hasattr(calc, "_detect_landfalls_wrapper")
        assert hasattr(calc, "_mask_init_time_boundaries")
        assert hasattr(calc, "_interpolate_and_format_landfalls")

    def test_utils_private_functions_accessible(self):
        """Test that private functions in utils are accessible."""
        from extremeweatherbench import utils

        assert hasattr(utils, "_create_nan_dataarray")
        assert hasattr(utils, "_cache_maybe_densify_helper")

    def test_derived_private_functions_accessible(self):
        """Test that private functions in derived are accessible."""
        from extremeweatherbench import derived

        assert hasattr(derived, "_maybe_convert_variable_to_string")

    def test_defaults_private_functions_accessible(self):
        """Test that private functions in defaults are accessible."""
        from extremeweatherbench import defaults

        assert hasattr(defaults, "_preprocess_cira_forecast_dataset")

    def test_regions_private_functions_accessible(self):
        """Test that private functions in regions are accessible."""
        from extremeweatherbench import regions

        assert hasattr(regions, "_adjust_bounds_to_dataset_convention")


class TestPublicFunctionAccess:
    """Test that all public functions are accessible via module."""

    def test_calc_public_functions(self):
        """Test public functions in calc are accessible."""
        from extremeweatherbench import calc

        assert hasattr(calc, "find_landfalls")
        assert hasattr(calc, "nantrapezoid")
        assert hasattr(calc, "dewpoint_from_specific_humidity")
        assert hasattr(calc, "find_land_intersection")
        assert hasattr(calc, "haversine_distance")

    def test_utils_public_functions(self):
        """Test public functions in utils are accessible."""
        from extremeweatherbench import utils

        assert hasattr(utils, "reduce_dataarray")
        assert hasattr(utils, "stack_dataarray_from_dims")
        assert hasattr(utils, "convert_longitude_to_360")


class TestTopLevelImports:
    """Test that top-level imports work for commonly used items."""

    def test_top_level_metric_imports(self):
        """Test that metrics can be imported at top level."""
        from extremeweatherbench import (
            MeanAbsoluteError,
            MeanError,
            MeanSquaredError,
            RootMeanSquaredError,
        )

        assert MeanAbsoluteError is not None
        assert MeanError is not None
        assert MeanSquaredError is not None
        assert RootMeanSquaredError is not None

    def test_top_level_input_imports(self):
        """Test that input classes can be imported at top level."""
        from extremeweatherbench import ERA5, GHCN, IBTrACS, ZarrForecast

        assert ERA5 is not None
        assert GHCN is not None
        assert IBTrACS is not None
        assert ZarrForecast is not None

    def test_top_level_region_imports(self):
        """Test that region classes can be imported at top level."""
        from extremeweatherbench import BoundingBoxRegion, CenteredRegion, Region

        assert Region is not None
        assert BoundingBoxRegion is not None
        assert CenteredRegion is not None

    def test_top_level_case_imports(self):
        """Test that case classes can be imported at top level."""
        from extremeweatherbench import CaseOperator, IndividualCase

        assert IndividualCase is not None
        assert CaseOperator is not None

    def test_evaluation_alias(self):
        """Test that evaluation alias works."""
        from extremeweatherbench import ExtremeWeatherBench, evaluation

        assert evaluation is ExtremeWeatherBench

    def test_load_cases_alias(self):
        """Test that load_cases alias works."""
        from extremeweatherbench import (
            load_cases,
            load_ewb_events_yaml_into_case_list,
        )

        assert load_cases is load_ewb_events_yaml_into_case_list


class TestNamespaceSubmodules:
    """Test the convenience namespace submodules."""

    def test_targets_namespace(self):
        """Test targets SimpleNamespace contains expected items."""
        from extremeweatherbench import targets

        assert isinstance(targets, types.SimpleNamespace)
        assert hasattr(targets, "ERA5")
        assert hasattr(targets, "GHCN")
        assert hasattr(targets, "IBTrACS")
        assert hasattr(targets, "TargetBase")

    def test_forecasts_namespace(self):
        """Test forecasts SimpleNamespace contains expected items."""
        from extremeweatherbench import forecasts

        assert isinstance(forecasts, types.SimpleNamespace)
        assert hasattr(forecasts, "ZarrForecast")
        assert hasattr(forecasts, "KerchunkForecast")
        assert hasattr(forecasts, "ForecastBase")


class TestMockPatching:
    """Test that mock.patch.object works with module imports."""

    def test_mock_patch_object_on_calc(self):
        """Test that mock.patch.object works on calc module."""
        from unittest import mock

        from extremeweatherbench import calc

        with mock.patch.object(calc, "haversine_distance") as mock_func:
            mock_func.return_value = 42.0
            result = calc.haversine_distance([0, 0], [1, 1])
            assert result == 42.0
            mock_func.assert_called_once()

    def test_mock_patch_string_on_calc(self):
        """Test that mock.patch with string path works on calc module."""
        from unittest import mock

        with mock.patch("extremeweatherbench.calc.haversine_distance") as mock_func:
            mock_func.return_value = 100.0
            from extremeweatherbench import calc

            result = calc.haversine_distance([0, 0], [1, 1])
            assert result == 100.0
