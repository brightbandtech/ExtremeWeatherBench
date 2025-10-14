#!/usr/bin/env python3
"""
Combined miniaturized script to test all event types with minimal data.
Runs atmospheric rivers, heat waves, and tropical cyclones sequentially.
"""

import logging
import sys
import time
from pathlib import Path

from extremeweatherbench import derived, evaluate, inputs, metrics, utils

# Configure logging to suppress verbose output
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_atmospheric_rivers():
    """Test atmospheric river evaluation with minimal data."""
    logger.info("=" * 50)
    logger.info("Testing Atmospheric Rivers")
    logger.info("=" * 50)

    start_time = time.time()

    # Load and filter cases
    case_yaml = utils.load_events_yaml()
    ar_cases = [
        case
        for case in case_yaml["cases"]
        if case["start_date"].year < 2023 and case["event_type"] == "atmospheric_river"
    ]
    test_yaml = {"cases": ar_cases[:1]}

    if not test_yaml["cases"]:
        logger.warning("No atmospheric river cases found")
        return False

    logger.info(f"Testing with {len(test_yaml['cases'])} atmospheric river case(s)")

    # Set up data sources
    # Note: Using basic variables since IntegratedVaporTransport is not yet implemented
    era5_target = inputs.ERA5(
        source=inputs.ARCO_ERA5_FULL_URI,
        variables=[
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
            "surface_air_temperature",
        ],
        variable_mapping={
            "specific_humidity": "specific_humidity",
            "u_component_of_wind": "eastward_wind",
            "v_component_of_wind": "northward_wind",
            "2m_temperature": "surface_air_temperature",
            "time": "valid_time",
        },
        storage_options={"remote_options": {"anon": True}},
        chunks={"time": 24, "latitude": 180, "longitude": 360},
    )

    hres_forecast = inputs.ZarrForecast(
        source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        variables=[
            "surface_air_temperature",
            "surface_eastward_wind",
            "surface_northward_wind",
            "eastward_wind",
            "northward_wind",
        ],
        variable_mapping={
            "2m_temperature": "surface_air_temperature",
            "10m_u_component_of_wind": "surface_eastward_wind",
            "10m_v_component_of_wind": "surface_northward_wind",
            "u_component_of_wind": "eastward_wind",
            "v_component_of_wind": "northward_wind",
            "prediction_timedelta": "lead_time",
            "time": "init_time",
            "lead_time": "prediction_timedelta",
            "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
            "10m_wind_speed": "surface_wind_speed",
        },
        storage_options={"remote_options": {"anon": True}},
        chunks={"prediction_timedelta": 10, "latitude": 180, "longitude": 360},
    )

    ar_metric_list = [
        inputs.EvaluationObject(
            event_type="atmospheric_river",
            metric=[metrics.MAE],
            target=era5_target,
            forecast=hres_forecast,
        ),
    ]

    try:
        test_ewb = evaluate.ExtremeWeatherBench(
            cases=test_yaml,
            metrics=ar_metric_list,
        )

        outputs = test_ewb.run(
            tolerance_range=24,
            pre_compute=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Atmospheric rivers test completed in {elapsed:.1f}s")
        logger.info(f"Generated {len(outputs)} result rows")
        return True

    except Exception as e:
        logger.error(f"✗ Atmospheric rivers test failed: {e}")
        return False


def test_heat_waves():
    """Test heat wave evaluation with minimal data."""
    logger.info("=" * 50)
    logger.info("Testing Heat Waves")
    logger.info("=" * 50)

    start_time = time.time()

    # Load and filter cases
    case_yaml = utils.load_events_yaml()
    heatwave_cases = [
        case for case in case_yaml["cases"] if case["event_type"] == "heat_wave"
    ]
    test_yaml = {"cases": heatwave_cases[:1]}

    if not test_yaml["cases"]:
        logger.warning("No heat wave cases found")
        return False

    logger.info(f"Testing with {len(test_yaml['cases'])} heat wave case(s)")

    # Set up data sources
    ghcn_target = inputs.GHCN(
        source=inputs.DEFAULT_GHCN_URI,
        variables=["surface_air_temperature"],
        variable_mapping={
            "surface_air_temperature": "surface_air_temperature",
            "time": "valid_time",
        },
        storage_options={},
    )

    hres_forecast = inputs.ZarrForecast(
        source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        variables=["surface_air_temperature"],
        variable_mapping={
            "2m_temperature": "surface_air_temperature",
            "prediction_timedelta": "lead_time",
            "time": "init_time",
        },
        storage_options={"remote_options": {"anon": True}},
        chunks={"prediction_timedelta": 10, "latitude": 180, "longitude": 360},
    )

    heatwave_metric_list = [
        inputs.EvaluationObject(
            event_type="heat_wave",
            metric=[metrics.MaximumMAE, metrics.RMSE],
            target=ghcn_target,
            forecast=hres_forecast,
        ),
    ]

    try:
        test_ewb = evaluate.ExtremeWeatherBench(
            cases=test_yaml,
            metrics=heatwave_metric_list,
        )

        outputs = test_ewb.run(
            tolerance_range=24,
            pre_compute=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Heat waves test completed in {elapsed:.1f}s")
        logger.info(f"Generated {len(outputs)} result rows")
        return True

    except Exception as e:
        logger.error(f"✗ Heat waves test failed: {e}")
        return False


def test_tropical_cyclones():
    """Test tropical cyclone evaluation with minimal data."""
    logger.info("=" * 50)
    logger.info("Testing Tropical Cyclones")
    logger.info("=" * 50)

    start_time = time.time()

    # Load and filter cases
    case_yaml = utils.load_events_yaml()
    tc_cases = [
        case for case in case_yaml["cases"] if case["event_type"] == "tropical_cyclone"
    ]

    if tc_cases:
        test_yaml = {"cases": tc_cases[:1]}
    elif len(case_yaml["cases"]) > 200:
        # Fallback to case 200 if no TC cases found
        test_yaml = {"cases": [case_yaml["cases"][200]]}
    else:
        test_yaml = {"cases": case_yaml["cases"][-1:]}

    logger.info(f"Testing with {len(test_yaml['cases'])} tropical cyclone case(s)")

    # Set up data sources
    ibtracs_target = inputs.IBTrACS(
        source=inputs.IBTRACS_URI,
        variable_mapping=inputs.IBTrACS_metadata_variable_mapping,
    )

    hres_forecast = inputs.ZarrForecast(
        source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        variables=[derived.TropicalCycloneTrackVariables],
        variable_mapping={
            "2m_temperature": "surface_air_temperature",
            "10m_u_component_of_wind": "surface_eastward_wind",
            "10m_v_component_of_wind": "surface_northward_wind",
            "u_component_of_wind": "eastward_wind",
            "v_component_of_wind": "northward_wind",
            "prediction_timedelta": "lead_time",
            "time": "init_time",
            "lead_time": "prediction_timedelta",
            "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
            "10m_wind_speed": "surface_wind_speed",
        },
        storage_options={"remote_options": {"anon": True}},
        chunks={"prediction_timedelta": 10, "latitude": 180, "longitude": 360},
    )

    tc_metric_list = [
        inputs.EvaluationObject(
            event_type="tropical_cyclone",
            metric=[metrics.LandfallTimeME, metrics.LandfallDisplacement],
            target=ibtracs_target,
            forecast=hres_forecast,
        ),
    ]

    try:
        test_ewb = evaluate.ExtremeWeatherBench(
            cases=test_yaml,
            metrics=tc_metric_list,
        )

        outputs = test_ewb.run(
            tolerance_range=24,
            pre_compute=False,
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Tropical cyclones test completed in {elapsed:.1f}s")
        logger.info(f"Generated {len(outputs)} result rows")
        return True

    except Exception as e:
        logger.error(f"✗ Tropical cyclones test failed: {e}")
        return False


def main():
    """Run all miniaturized tests."""
    logger.info("Starting miniaturized ExtremeWeatherBench tests")
    logger.info(f"Script location: {Path(__file__).absolute()}")

    total_start = time.time()
    tests_passed = 0
    total_tests = 3

    # Run each test type
    if test_atmospheric_rivers():
        tests_passed += 1

    if test_heat_waves():
        tests_passed += 1

    if test_tropical_cyclones():
        tests_passed += 1

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("=" * 50)
    logger.info("MINIATURIZED TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    if tests_passed == total_tests:
        logger.info("✓ All miniaturized tests passed!")
        sys.exit(0)
    else:
        logger.error(f"✗ {total_tests - tests_passed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
