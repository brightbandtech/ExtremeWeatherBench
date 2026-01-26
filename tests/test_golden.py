"""Tests which use the full end-to-end EWB workflow.

These tests are likely incompatible with Github Actions and will be used on a VM
or other virtual environment. These are intended to be fairly lightweight marquee
examples of each event type and core metrics. If the values deviate from expected
for a release, it will be flagged as a failure."""


# Load case data from the default events.yaml

import pathlib

import pytest

from extremeweatherbench import cases, defaults, derived, evaluate, inputs, metrics


@pytest.fixture(scope="module")
def reference_data_dir():
    """Path to reference data directory."""
    path = pathlib.Path(__file__).parent / "data"
    if not path.exists():
        pytest.skip(
            "Reference data not found. Run 'uv run data/generate_cape_reference_data.py' first."
        )
    return path


@pytest.fixture(scope="module")
def golden_tests_event_data(reference_data_dir):
    """Load golden tests event data."""
    ref_file = reference_data_dir / "golden_tests.yaml"
    if not ref_file.exists():
        pytest.skip(f"Golden tests event data not found: {ref_file}")

    return cases.load_individual_cases_from_yaml(ref_file)


@pytest.mark.integration
class TestGoldenTests:
    """Golden tests."""

    def test_heatwaves(self, golden_tests_event_data):
        """Heatwave tests."""
        # Define heatwave objects
        era5_heatwave_target = inputs.ERA5()
        ghcn_heatwave_target = inputs.GHCN()

        heatwave_metrics = [
            metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            metrics.DurationMeanError(
                threshold_criteria=defaults.get_climatology(quantile=0.85)
            ),
            metrics.MaximumLowestMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ]

        hres_heatwave_forecast = inputs.ZarrForecast(
            name="hres_heatwave_forecast",
            source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
            variables=["surface_air_temperature"],
            variable_mapping=inputs.HRES_metadata_variable_mapping,
        )

        heatwave_evaluation_objects = [
            inputs.EvaluationObject(
                event_type="heat_wave",
                metric_list=heatwave_metrics,
                target=era5_heatwave_target,
                forecast=hres_heatwave_forecast,
            ),
            inputs.EvaluationObject(
                event_type="heat_wave",
                metric_list=heatwave_metrics,
                target=ghcn_heatwave_target,
                forecast=hres_heatwave_forecast,
            ),
        ]
        # Run the evaluation
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=golden_tests_event_data,
            evaluation_objects=heatwave_evaluation_objects,
        )

        outputs = ewb.run(
            parallel_config={
                "backend": "loky",
                "n_jobs": len(heatwave_evaluation_objects) * len(heatwave_metrics),
            },
        )
        outputs.to_csv("golden_tests_heatwave_results.csv")

    def test_freezes(self, golden_tests_event_data):
        """Freeze tests."""
        era5_freeze_target = inputs.ERA5()
        ghcn_freeze_target = inputs.GHCN()
        hres_freeze_forecast = inputs.ZarrForecast(
            name="hres_freeze_forecast",
            source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
            variables=["surface_air_temperature"],
            variable_mapping=inputs.HRES_metadata_variable_mapping,
        )
        freeze_metrics = [
            metrics.MinimumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            metrics.DurationMeanError(
                threshold_criteria=defaults.get_climatology(quantile=0.15)
            ),
        ]
        freeze_evaluation_objects = [
            inputs.EvaluationObject(
                event_type="freeze",
                metric_list=freeze_metrics,
                target=era5_freeze_target,
                forecast=hres_freeze_forecast,
            ),
            inputs.EvaluationObject(
                event_type="freeze",
                metric_list=freeze_metrics,
                target=ghcn_freeze_target,
                forecast=hres_freeze_forecast,
            ),
        ]
        # Run the evaluation
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=golden_tests_event_data,
            evaluation_objects=freeze_evaluation_objects,
        )
        outputs = ewb.run(
            parallel_config={
                "backend": "loky",
                "n_jobs": len(freeze_evaluation_objects) * len(freeze_metrics),
            },
        )
        outputs.to_csv("golden_tests_freeze_results.csv")

    def test_severe_convection(self, golden_tests_event_data):
        """Severe convection tests."""
        lsr_severe_convection_target = inputs.LSR()
        pph_severe_convection_target = inputs.PPH()
        hres_severe_convection_forecast = inputs.ZarrForecast(
            name="hres_severe_convection_forecast",
            source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
            variables=[derived.CravenBrooksSignificantSevere()],
            variable_mapping=inputs.HRES_metadata_variable_mapping,
        )
        severe_convection_metrics = [
            metrics.ThresholdMetric(
                metrics=[metrics.CriticalSuccessIndex, metrics.FalseAlarmRatio],
                forecast_threshold=15000,
                target_threshold=0.3,
            ),
            metrics.EarlySignal(threshold=15000),
        ]
        severe_convection_evaluation_objects = [
            inputs.EvaluationObject(
                event_type="severe_convection",
                metric_list=severe_convection_metrics,
                target=lsr_severe_convection_target,
                forecast=hres_severe_convection_forecast,
            ),
            inputs.EvaluationObject(
                event_type="severe_convection",
                metric_list=severe_convection_metrics,
                target=pph_severe_convection_target,
                forecast=hres_severe_convection_forecast,
            ),
        ]
        # Run the evaluation
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=golden_tests_event_data,
            evaluation_objects=severe_convection_evaluation_objects,
        )
        outputs = ewb.run(
            parallel_config={
                "backend": "loky",
                "n_jobs": len(severe_convection_evaluation_objects)
                * len(severe_convection_metrics),
            },
        )
        outputs.to_csv("golden_tests_severe_convection_results.csv")

    def test_atmospheric_river(self, golden_tests_event_data):
        """Atmospheric river tests."""
        era5_atmospheric_river_target = inputs.ERA5()
        hres_atmospheric_river_forecast = inputs.ZarrForecast(
            name="hres_atmospheric_river_forecast",
            source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
            variables=[
                derived.AtmosphericRiverVariables(
                    output_variables=[
                        "atmospheric_river_mask",
                        "integrated_vapor_transport",
                        "atmospheric_river_land_intersection",
                    ]
                )
            ],
            variable_mapping=inputs.HRES_metadata_variable_mapping,
        )
        atmospheric_river_metrics = [
            metrics.CriticalSuccessIndex(),
            metrics.EarlySignal(),
            metrics.SpatialDisplacement(),
        ]
        atmospheric_river_evaluation_objects = [
            inputs.EvaluationObject(
                event_type="atmospheric_river",
                metric_list=atmospheric_river_metrics,
                target=era5_atmospheric_river_target,
                forecast=hres_atmospheric_river_forecast,
            ),
        ]
        # Run the evaluation
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=golden_tests_event_data,
            evaluation_objects=atmospheric_river_evaluation_objects,
        )
        outputs = ewb.run(
            parallel_config={
                "backend": "loky",
                "n_jobs": len(atmospheric_river_evaluation_objects)
                * len(atmospheric_river_metrics),
            },
        )
        outputs.to_csv("golden_tests_atmospheric_river_results.csv")

    def test_tropical_cyclone(self, golden_tests_event_data):
        """Tropical cyclone tests."""
        ibtracs_tropical_cyclone_target = inputs.IBTrACS()
        hres_tropical_cyclone_forecast = inputs.ZarrForecast(
            name="hres_tropical_cyclone_forecast",
            source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
            variables=[derived.TropicalCycloneTrackVariables()],
            variable_mapping=inputs.HRES_metadata_variable_mapping,
        )
        tropical_cyclone_metrics = [
            metrics.LandfallMetric(
                metrics=[
                    metrics.LandfallIntensityMeanAbsoluteError,
                    metrics.LandfallTimeMeanError,
                    metrics.LandfallDisplacement,
                ],
                approach="next",
                forecast_variable="air_pressure_at_mean_sea_level",
                target_variable="air_pressure_at_mean_sea_level",
            ),
        ]
        tropical_cyclone_evaluation_objects = [
            inputs.EvaluationObject(
                event_type="tropical_cyclone",
                metric_list=tropical_cyclone_metrics,
                target=ibtracs_tropical_cyclone_target,
                forecast=hres_tropical_cyclone_forecast,
            ),
        ]
        # Run the evaluation
        ewb = evaluate.ExtremeWeatherBench(
            case_metadata=golden_tests_event_data,
            evaluation_objects=tropical_cyclone_evaluation_objects,
        )
        outputs = ewb.run(
            parallel_config={
                "backend": "loky",
                "n_jobs": len(tropical_cyclone_evaluation_objects)
                * len(tropical_cyclone_metrics),
            },
        )
        outputs.to_csv("golden_tests_tropical_cyclone_results.csv")


if __name__ == "__main__":
    pytest.main([__file__])
