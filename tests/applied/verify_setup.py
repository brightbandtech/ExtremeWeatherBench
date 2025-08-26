#!/usr/bin/env python3
"""
Setup verification script that tests imports and basic functionality
without requiring external data access.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        from extremeweatherbench import derived, evaluate, inputs, metrics, utils

        logger.info("✓ Core ExtremeWeatherBench modules imported successfully")

        # Test specific classes/functions
        _ = inputs.ERA5
        _ = inputs.ZarrForecast
        _ = inputs.GHCN
        _ = inputs.IBTrACS
        _ = evaluate.ExtremeWeatherBench
        _ = metrics.MAE
        _ = derived.TropicalCycloneTrackVariables  # Use available derived variable
        _ = utils.load_events_yaml

        logger.info("✓ All required classes and functions accessible")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during import test: {e}")
        return False


def test_events_yaml():
    """Test that events.yaml can be loaded."""
    logger.info("Testing events.yaml loading...")

    try:
        from extremeweatherbench import utils

        events = utils.load_events_yaml()

        if not events or "cases" not in events:
            logger.error("✗ Events YAML loaded but has no cases")
            return False

        num_cases = len(events["cases"])
        logger.info(f"✓ Events YAML loaded successfully with {num_cases} cases")

        # Check for different event types
        event_types = set(case.get("event_type") for case in events["cases"])
        logger.info(f"✓ Found event types: {sorted(event_types)}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to load events.yaml: {e}")
        return False


def test_data_sources_config():
    """Test that data source URIs are accessible as constants."""
    logger.info("Testing data source configuration...")

    try:
        from extremeweatherbench import inputs

        # Test that URI constants exist
        _ = inputs.ARCO_ERA5_FULL_URI
        _ = inputs.DEFAULT_GHCN_URI
        _ = inputs.IBTRACS_URI
        _ = inputs.IBTrACS_metadata_variable_mapping

        logger.info("✓ All data source URIs and mappings accessible")
        return True

    except AttributeError as e:
        logger.error(f"✗ Missing data source configuration: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error in data source test: {e}")
        return False


def test_script_files():
    """Test that all miniaturized scripts exist."""
    logger.info("Testing miniaturized script files...")

    script_dir = Path(__file__).parent
    expected_scripts = [
        "mini_applied_ar.py",
        "mini_applied_heatwave.py",
        "mini_applied_tc.py",
        "mini_applied_all.py",
    ]

    missing_scripts = []
    for script in expected_scripts:
        script_path = script_dir / script
        if not script_path.exists():
            missing_scripts.append(script)
        elif not script_path.is_file():
            missing_scripts.append(f"{script} (not a file)")
        else:
            logger.info(f"✓ Found {script}")

    if missing_scripts:
        logger.error(f"✗ Missing scripts: {missing_scripts}")
        return False

    logger.info("✓ All miniaturized scripts found")
    return True


def main():
    """Run all verification tests."""
    logger.info("=" * 50)
    logger.info("ExtremeWeatherBench Setup Verification")
    logger.info("=" * 50)
    logger.info(f"Script location: {Path(__file__).absolute()}")

    tests = [
        ("Import Test", test_imports),
        ("Events YAML Test", test_events_yaml),
        ("Data Sources Config Test", test_data_sources_config),
        ("Script Files Test", test_script_files),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        logger.info("-" * 30)
        logger.info(f"Running {test_name}")

        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")

    # Summary
    logger.info("=" * 50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        logger.info("✓ All verification tests passed!")
        logger.info("Your setup is ready for miniaturized testing.")
        logger.info(
            "Note: Data access tests still require Google Cloud authentication."
        )
        sys.exit(0)
    else:
        logger.error(f"✗ {total_tests - passed_tests} verification tests failed")
        logger.error("Please fix the issues above before running miniaturized tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
