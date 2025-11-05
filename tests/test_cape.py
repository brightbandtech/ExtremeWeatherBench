"""Unit tests for CAPE/CIN implementations.

These tests verify correctness of the pure Python and Numba implementations
against MetPy reference data without requiring MetPy as a runtime dependency.

Reference data is generated once using data/generate_cape_reference_data.py which has
MetPy as a dependency. These tests only require the generated reference files.
"""

import pathlib

import numpy as np
import pytest

from extremeweatherbench._cape import (
    compute_ml_cape_cin_batched,
    compute_ml_cape_cin_from_profile,
)


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
def era5_reference(reference_data_dir):
    """Load ERA5 reference data."""
    ref_file = reference_data_dir / "era5_reference.npz"
    if not ref_file.exists():
        pytest.skip(f"ERA5 reference file not found: {ref_file}")

    with np.load(ref_file) as data:
        return {
            "pressure": data["pressure"],
            "temperature": data["temperature"],
            "dewpoint": data["dewpoint"],
            "geopotential": data["geopotential"],
            "cape_reference": data["cape_reference"],
            "cin_reference": data["cin_reference"],
        }


@pytest.fixture(scope="module")
def pathological_profiles(reference_data_dir):
    """Load pathological test profiles."""
    ref_file = reference_data_dir / "pathological_profiles.npz"
    if not ref_file.exists():
        pytest.skip(f"Pathological profiles not found: {ref_file}")

    with np.load(ref_file, allow_pickle=True) as data:
        profiles = {}
        names = data["names"]
        for i, name in enumerate(names):
            profiles[name] = {
                "pressure": data["pressure"][i],
                "temperature": data["temperature"][i],
                "dewpoint": data["dewpoint"][i],
                "geopotential": data["geopotential"][i],
                "cape_reference": data["cape_reference"][i],
                "cin_reference": data["cin_reference"][i],
                "description": str(data["descriptions"][i])
                if "descriptions" in data
                else "",
            }
        return profiles


def test_batch_processing(era5_reference):
    """Test that batch processing matches single-profile results."""
    n_profiles = min(100, len(era5_reference["pressure"]))

    # Single profile results
    cape_single = []
    cin_single = []

    for i in range(n_profiles):
        p = era5_reference["pressure"][i]
        t = era5_reference["temperature"][i]
        td = era5_reference["dewpoint"][i]
        z = era5_reference["geopotential"][i]

        cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)
        cape_single.append(cape)
        cin_single.append(cin)

    # Batch results
    p_batch = era5_reference["pressure"][:n_profiles]
    t_batch = era5_reference["temperature"][:n_profiles]
    td_batch = era5_reference["dewpoint"][:n_profiles]
    z_batch = era5_reference["geopotential"][:n_profiles]

    # Ensure contiguous
    p_batch = np.ascontiguousarray(p_batch, dtype=np.float64)
    t_batch = np.ascontiguousarray(t_batch, dtype=np.float64)
    td_batch = np.ascontiguousarray(td_batch, dtype=np.float64)
    z_batch = np.ascontiguousarray(z_batch, dtype=np.float64)

    cape_batch, cin_batch = compute_ml_cape_cin_batched(
        p_batch, t_batch, td_batch, z_batch
    )

    # Should match exactly
    cape_single = np.array(cape_single)
    cin_single = np.array(cin_single)

    # Test closeness using a tolerance of 1e-5 ~ 0.0001%
    np.testing.assert_allclose(
        cape_batch,
        cape_single,
        rtol=1e-5,
        err_msg="Batch CAPE differs from single-profile results",
    )
    np.testing.assert_allclose(
        cin_batch,
        cin_single,
        rtol=1e-5,
        err_msg="Batch CIN differs from single-profile results",
    )


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_all_nans(self):
        """Test handling of all-NaN input."""
        p = np.full(13, np.nan)
        t = np.full(13, np.nan)
        td = np.full(13, np.nan)
        z = np.full(13, np.nan)

        cape, _ = compute_ml_cape_cin_from_profile(p, t, td, z)

        # Should return NaN or 0, not crash
        assert np.isnan(cape) or cape == 0.0

    def test_minimum_levels(self):
        """Test with minimum number of levels (5)."""
        p = np.array([1000, 850, 700, 500, 300], dtype=np.float64)
        t = np.array([288, 280, 270, 250, 230], dtype=np.float64)
        td = t - 5
        z = 29.3 * 288 * np.log(1000 / p)

        # Should not crash
        cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)

        assert cape >= 0, "CAPE should be non-negative"
        assert cin <= 0, "CIN should be non-positive"

    def test_many_levels(self):
        """Test with many levels (100)."""
        n_levels = 100
        p = np.linspace(1000, 50, n_levels)
        t = 288.0 - (1000 - p) * 0.068
        td = t - 5
        z = 29.3 * 288 * np.log(1000 / p)

        # Should not crash and should be reasonably fast
        cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)

        assert cape >= 0, "CAPE should be non-negative"
        assert cin <= 0, "CIN should be non-positive"

    def test_supersaturated_levels(self):
        """Test with supersaturated levels (dewpoint > temperature)."""
        p = np.array(
            [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
            dtype=np.float64,
        )
        t = np.array(
            [288, 284, 280, 270, 264, 256, 243, 230, 225, 220, 215, 210, 205],
            dtype=np.float64,
        )
        td = t + 1  # Supersaturated!
        z = 29.3 * 288 * np.log(1000 / p)

        # Should handle gracefully (cap at saturation)
        cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)

        # Should not crash or produce invalid results
        assert not np.isnan(cape), "CAPE should not be NaN"
        assert not np.isnan(cin), "CIN should not be NaN"

    def test_inverted_temperature_profile(self):
        """Test with inverted temperature profile (temperature increases with height)."""
        p = np.array(
            [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
            dtype=np.float64,
        )
        t = np.array(
            [270, 275, 280, 285, 280, 270, 250, 235, 230, 225, 220, 215, 210],
            dtype=np.float64,
        )
        td = t - 10
        z = (
            np.array(
                [
                    0,
                    700,
                    1500,
                    3000,
                    4200,
                    5600,
                    7200,
                    9200,
                    10400,
                    11800,
                    13500,
                    16000,
                    20500,
                ],
                dtype=np.float64,
            )
            * 9.81
        )

        # Should handle inversion layers
        cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)

        # Probably no CAPE with strong inversion
        assert cape >= 0, "CAPE should be non-negative"


class TestKnownProfile:
    """Regression tests for a known profile that was pre-computed."""

    @pytest.fixture(scope="class")
    def known_results(self):
        """Known good results for specific profiles."""
        # This is the same data as in the example from MetPy's documentation at
        # https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.mixed_layer_cape_cin.html#mixed-layer-cape-cin
        p = np.array(
            [
                1008.0,
                1000.0,
                950.0,
                900.0,
                850.0,
                800.0,
                750.0,
                700.0,
                650.0,
                600.0,
                550.0,
                500.0,
                450.0,
                400.0,
                350.0,
                300.0,
                250.0,
                200.0,
                175.0,
                150.0,
                125.0,
                100.0,
                80.0,
                70.0,
                60.0,
                50.0,
                40.0,
                30.0,
                25.0,
                20.0,
            ],
            dtype=np.float64,
        )
        # temperature
        t = np.array(
            [
                29.3,
                28.1,
                25.5,
                20.9,
                18.4,
                15.9,
                13.1,
                10.1,
                6.7,
                3.1,
                -0.5,
                -4.5,
                -9.0,
                -14.8,
                -21.5,
                -29.7,
                -40.0,
                -52.4,
                -59.2,
                -66.5,
                -74.1,
                -78.5,
                -76.0,
                -71.6,
                -66.7,
                -61.3,
                -56.3,
                -51.7,
                -50.7,
                -47.5,
            ],
            dtype=np.float64,
        )
        td = np.array(
            [
                26.5,
                23.2,
                16.1,
                6.4,
                15.3,
                10.9,
                8.8,
                7.9,
                0.6,
                -16.6,
                -9.2,
                -9.9,
                -14.6,
                -32.8,
                -51.2,
                -32.7,
                -42.6,
                -58.9,
                -69.4,
                -71.4,
                -75.5,
                -78.8,
                -79.3,
                -72.1,
                -73.0,
                -64.2,
                -70.5,
                -75.7,
                -51.2,
                -56.4,
            ],
            dtype=np.float64,
        )
        return {
            "profile_0": {
                "pressure": p,
                "temperature": 273.15 + t,
                "dewpoint": 273.15 + td,
                # Crudely estimate geopotential height from pressure
                "geopotential": 29.3 * 273 * np.log(1000 / p),
                "mixed_layer_depth": 50,
                "expected_cape_range": (200, 800),
                "expected_cin_range": (-20, 20),
            }
        }

    def test_known_profile_results(self, known_results):
        """Test that known profiles produce results in expected ranges."""
        for name, profile in known_results.items():
            depth = profile["mixed_layer_depth"]
            cape, cin = compute_ml_cape_cin_from_profile(
                profile["pressure"],
                profile["temperature"],
                profile["dewpoint"],
                profile["geopotential"],
                depth,
            )

            cape_min, cape_max = profile["expected_cape_range"]
            cin_min, cin_max = profile["expected_cin_range"]

            assert cape_min <= cape <= cape_max, (
                f"{name}: CAPE {cape:.2f} outside expected range [{cape_min}, {cape_max}]"
            )
            assert cin_min <= cin <= cin_max, (
                f"{name}: CIN {cin:.2f} outside expected range [{cin_min}, {cin_max}]"
            )


class TestPerformance:
    """Performance smoke tests to capture regressions from dependency drift."""

    def test_single_profile_performance(self, era5_reference):
        """Test that single profile computation is reasonably fast."""
        import time

        p = era5_reference["pressure"][0]
        t = era5_reference["temperature"][0]
        td = era5_reference["dewpoint"][0]
        z = era5_reference["geopotential"][0]

        # Warm up
        for _ in range(10):
            _ = compute_ml_cape_cin_from_profile(p, t, td, z)

        # Time it
        start = time.perf_counter()
        n_iter = 1000
        for _ in range(n_iter):
            _ = compute_ml_cape_cin_from_profile(p, t, td, z)
        elapsed = time.perf_counter() - start

        time_per_call = (elapsed / n_iter) * 1e6  # microseconds

        # Should be faster than 100 μs per profile for Numba V2
        assert time_per_call < 100, (
            f"Single profile computation too slow: {time_per_call:.2f} μs"
        )

    @pytest.mark.flaky(reruns=3)
    def test_batch_performance(self, era5_reference):
        """Test that batch processing is reasonably fast."""
        import time

        n_profiles = 100
        p_batch = np.ascontiguousarray(
            era5_reference["pressure"][:n_profiles], dtype=np.float64
        )
        t_batch = np.ascontiguousarray(
            era5_reference["temperature"][:n_profiles], dtype=np.float64
        )
        td_batch = np.ascontiguousarray(
            era5_reference["dewpoint"][:n_profiles], dtype=np.float64
        )
        z_batch = np.ascontiguousarray(
            era5_reference["geopotential"][:n_profiles], dtype=np.float64
        )

        # Warm up
        _ = compute_ml_cape_cin_batched(p_batch, t_batch, td_batch, z_batch)

        # Time it
        start = time.perf_counter()
        _ = compute_ml_cape_cin_batched(p_batch, t_batch, td_batch, z_batch)
        elapsed = time.perf_counter() - start

        time_per_profile = (elapsed / n_profiles) * 1e6  # microseconds

        # Batch should be much faster than 100 μs per profile
        assert time_per_profile < 20, (
            f"Batch processing too slow: {time_per_profile:.2f} μs per profile"
        )


class TestReferenceMetPy:
    """Tests that our implementation produces similar results to the MetPy reference implementation."""

    def test_single_profile_vs_reference(self, era5_reference):
        """Test pure Python against reference for first profile."""
        idx = 5
        p = era5_reference["pressure"][idx]
        t = era5_reference["temperature"][idx]
        td = era5_reference["dewpoint"][idx]
        z = era5_reference["geopotential"][idx]

        cape, _ = compute_ml_cape_cin_from_profile(p, t, td, z)

        cape_ref = era5_reference["cape_reference"][idx]

        # Allow 10% relative error or 50 J/kg absolute error
        cape_tol = max(abs(cape_ref) * 0.10, 50.0)

        assert abs(cape - cape_ref) < cape_tol, (
            f"CAPE error too large: {cape:.2f} vs {cape_ref:.2f}"
        )

    def test_overall_accuracy(self, era5_reference):
        """Test pure Python overall accuracy metrics."""
        n_profiles = min(100, len(era5_reference["pressure"]))

        cape_computed = []
        cin_computed = []
        cape_ref = []
        cin_ref = []

        for i in range(n_profiles):
            p = era5_reference["pressure"][i]
            t = era5_reference["temperature"][i]
            td = era5_reference["dewpoint"][i]
            z = era5_reference["geopotential"][i]

            cape, cin = compute_ml_cape_cin_from_profile(p, t, td, z)

            cape_computed.append(cape)
            cin_computed.append(cin)
            cape_ref.append(era5_reference["cape_reference"][i])
            cin_ref.append(era5_reference["cin_reference"][i])

        cape_computed = np.array(cape_computed)
        cape_ref = np.array(cape_ref)

        # CAPE correlation should be > 0.95
        cape_corr = np.corrcoef(cape_computed, cape_ref)[0, 1]
        assert cape_corr > 0.95, f"CAPE correlation too low: {cape_corr:.4f}"

        # Mean absolute error should be within 150 J/kg
        cape_mae = np.mean(np.abs(cape_computed - cape_ref))
        assert cape_mae < 150.0, f"CAPE MAE too large: {cape_mae:.2f} J/kg"


@pytest.mark.parametrize(
    "profile_name, threshold, sign",
    [
        pytest.param("stable_no_convection", 10.0, "lt", id="stable_no_convection"),
        pytest.param("strong_surface_based", 100.0, "gt", id="strong_surface_based"),
        # TODO(darothen): Clean up the weak cape profile and add it back in
        # pytest.param("elevated_with_cin", 50.0, "gt", id="elevated_with_cin"),
        pytest.param("weak_cape", 100.0, "lt", id="weak_cape"),
        pytest.param("isothermal_layer", 10.0, "lt", id="isothermal_layer"),
    ],
)
def test_pathological(pathological_profiles, profile_name, threshold, sign):
    """Test pathological, pre-computed CAPE profiles."""
    if profile_name not in pathological_profiles:
        pytest.skip(f"{profile_name} profile not in reference data")

    profile = pathological_profiles[profile_name]
    cape, _ = compute_ml_cape_cin_from_profile(
        profile["pressure"],
        profile["temperature"],
        profile["dewpoint"],
        profile["geopotential"],
    )

    if sign == "lt":
        assert cape < threshold, (
            f"CAPE {cape:.2f} is not less than threshold {threshold:.2f}"
        )
    elif sign == "gt":
        assert cape > threshold, (
            f"CAPE {cape:.2f} is not greater than threshold {threshold:.2f}"
        )
    else:
        raise ValueError(f"Invalid sign: {sign}")
