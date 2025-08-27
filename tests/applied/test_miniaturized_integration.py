"""
Integration test module for miniaturized scripts.
This module provides pytest-compatible tests for the miniaturized scripts.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_script(script_path: Path, timeout: int = 300) -> tuple[int, str, str]:
    """Run a Python script and return exit code, stdout, stderr."""
    try:
        # Find the repo root (where pyproject.toml is)
        repo_root = Path(__file__).parent.parent.parent
        while (
            not (repo_root / "pyproject.toml").exists()
            and repo_root != repo_root.parent
        ):
            repo_root = repo_root.parent

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),  # Run from repo root
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Script timed out after {timeout} seconds"


class TestMiniaturizedScripts:
    """Test class for miniaturized scripts that can be run with pytest."""

    def test_verify_setup(self):
        """Test the setup verification script (no external deps)."""
        # Instead of running as subprocess, let's import and run the functions directly
        # This is more reliable for testing
        import importlib.util

        # Import the verify_setup module
        script_path = Path(__file__).parent / "verify_setup.py"
        spec = importlib.util.spec_from_file_location("verify_setup", script_path)
        verify_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify_setup)

        # Run each verification test
        assert verify_setup.test_imports(), "Import test failed"
        assert verify_setup.test_events_yaml(), "Events YAML test failed"
        assert verify_setup.test_data_sources_config(), (
            "Data sources config test failed"
        )
        assert verify_setup.test_script_files(), "Script files test failed"

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        and not Path.home()
        .joinpath(".config/gcloud/application_default_credentials.json")
        .exists(),
        reason="Google Cloud authentication not available",
    )
    def test_mini_heatwave(self):
        """Test the miniaturized heatwave script (requires GCP auth)."""
        script_path = Path(__file__).parent / "mini_applied_heatwave.py"
        exit_code, stdout, stderr = run_script(
            script_path, timeout=600
        )  # 10 min timeout

        assert exit_code == 0, (
            f"Heatwave test failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
        assert "Heatwave evaluation completed successfully" in stdout
        assert "Sample results:" in stdout

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        and not Path.home()
        .joinpath(".config/gcloud/application_default_credentials.json")
        .exists(),
        reason="Google Cloud authentication not available",
    )
    def test_mini_atmospheric_river(self):
        """Test the miniaturized atmospheric river script (requires GCP auth)."""
        script_path = Path(__file__).parent / "mini_applied_ar.py"
        exit_code, stdout, stderr = run_script(script_path, timeout=600)

        assert exit_code == 0, f"AR test failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        assert "Testing with" in stdout
        assert "atmospheric river case(s)" in stdout

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        and not Path.home()
        .joinpath(".config/gcloud/application_default_credentials.json")
        .exists(),
        reason="Google Cloud authentication not available",
    )
    def test_mini_tropical_cyclone(self):
        """Test the miniaturized tropical cyclone script (requires GCP auth)."""
        script_path = Path(__file__).parent / "mini_applied_tc.py"
        exit_code, stdout, stderr = run_script(
            script_path, timeout=900
        )  # 15 min timeout

        assert exit_code == 0, f"TC test failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        assert "Testing with" in stdout
        assert "tropical cyclone case(s)" in stdout


class TestMiniaturizedScriptsOffline:
    """Test class for parts of miniaturized scripts that don't require external data."""

    def test_script_imports(self):
        """Test that all miniaturized scripts can import successfully."""
        scripts_dir = Path(__file__).parent
        script_files = [
            "mini_applied_ar.py",
            "mini_applied_heatwave.py",
            "mini_applied_tc.py",
            "mini_applied_all.py",
            "verify_setup.py",
        ]

        for script_file in script_files:
            script_path = scripts_dir / script_file
            assert script_path.exists(), f"Script {script_file} not found"

            # Test that the script can be imported/compiled
            try:
                with open(script_path) as f:
                    code = f.read()
                compile(code, str(script_path), "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {script_file}: {e}")

    def test_configuration_validity(self):
        """Test that the script configurations are valid."""
        # This is covered by verify_setup.py, but we include it here for completeness
        from extremeweatherbench import inputs, metrics, utils

        # Test that required classes exist
        assert hasattr(inputs, "ERA5")
        assert hasattr(inputs, "GHCN")
        assert hasattr(inputs, "ZarrForecast")
        assert hasattr(inputs, "IBTrACS")
        assert hasattr(metrics, "MAE")
        assert hasattr(metrics, "RMSE")

        # Test that events can be loaded
        events = utils.load_events_yaml()
        assert "cases" in events
        assert len(events["cases"]) > 0

        # Test that we have the expected event types
        event_types = {case.get("event_type") for case in events["cases"]}
        expected_types = {"heat_wave", "atmospheric_river", "tropical_cyclone"}
        assert expected_types.issubset(event_types)


# Apply pytest marks to test classes - directly apply decorators
TestMiniaturizedScripts = pytest.mark.integration(
    pytest.mark.requires_auth(TestMiniaturizedScripts)
)
TestMiniaturizedScriptsOffline = pytest.mark.offline(TestMiniaturizedScriptsOffline)
