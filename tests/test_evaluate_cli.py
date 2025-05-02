"""Unit tests for the evaluate.py CLI interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from extremeweatherbench import evaluate


@pytest.fixture
def runner():
    """Fixture for Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_config_dir():
    """Fixture that creates a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_yaml_config():
    """Fixture that returns the path to the sample YAML config file."""
    return Path(__file__).parent / "data" / "sample_config.yaml"


@pytest.fixture
def sample_json_config(temp_config_dir):
    """Fixture that creates a sample JSON config file."""
    config_path = temp_config_dir / "config.json"
    # Load the YAML config and convert it to JSON
    with open(Path(__file__).parent / "data" / "sample_config.yaml") as f:
        config_data = yaml.safe_load(f)
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path


def test_cli_with_yaml_config(runner, sample_yaml_config):
    """Test CLI with YAML config file."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate.cli_runner, ["--config-file", str(sample_yaml_config)]
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_json_config(runner, sample_json_config):
    """Test CLI with JSON config file."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate.cli_runner, ["--config-file", str(sample_json_config)]
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_invalid_config_format(runner, temp_config_dir):
    """Test CLI with invalid config file format."""
    invalid_config = temp_config_dir / "config.txt"
    invalid_config.write_text("invalid config")

    result = runner.invoke(evaluate.cli_runner, ["--config-file", str(invalid_config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


def test_cli_with_default_flag(runner):
    """Test CLI with --default flag."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(evaluate.cli_runner, ["--default"])
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_individual_options(runner, temp_config_dir):
    """Test CLI with individual command line options."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate.cli_runner,
            [
                "--event-types",
                "HeatWave",
                "--output-dir",
                str(temp_config_dir / "outputs"),
                "--forecast-dir",
                str(temp_config_dir / "forecasts"),
                "--cache-dir",
                str(temp_config_dir / "cache"),
                "--gridded-obs-path",
                "gs://test-gridded-obs",
                "--point-obs-path",
                "gs://test-point-obs",
                "--remote-protocol",
                "s3",
                "--init-forecast-hour",
                "0",
                "--temporal-resolution-hours",
                "6",
                "--output-timesteps",
                "41",
                "--forecast-schema-surface-air-temperature",
                "t2m",
                "--forecast-schema-surface-eastward-wind",
                "u10",
                "--forecast-schema-surface-northward-wind",
                "v10",
                "--forecast-schema-air-temperature",
                "t",
                "--forecast-schema-eastward-wind",
                "u",
                "--forecast-schema-northward-wind",
                "v",
                "--forecast-schema-air-pressure-at-mean-sea-level",
                "msl",
                "--forecast-schema-lead-time",
                "time",
                "--forecast-schema-init-time",
                "init_time",
                "--forecast-schema-fhour",
                "fhour",
                "--forecast-schema-level",
                "level",
                "--forecast-schema-latitude",
                "latitude",
                "--forecast-schema-longitude",
                "longitude",
                "--point-schema-air-pressure-at-mean-sea-level",
                "air_pressure_at_mean_sea_level",
                "--point-schema-surface-air-pressure",
                "surface_air_pressure",
                "--point-schema-surface-wind-speed",
                "surface_wind_speed",
                "--point-schema-surface-wind-from-direction",
                "surface_wind_from_direction",
                "--point-schema-surface-air-temperature",
                "surface_air_temperature",
                "--point-schema-surface-dew-point-temperature",
                "surface_dew_point",
                "--point-schema-surface-relative-humidity",
                "surface_relative_humidity",
                "--point-schema-accumulated-1-hour-precipitation",
                "accumulated_1_hour_precipitation",
                "--point-schema-time",
                "time",
                "--point-schema-latitude",
                "latitude",
                "--point-schema-longitude",
                "longitude",
                "--point-schema-elevation",
                "elevation",
                "--point-schema-station-id",
                "station",
                "--point-schema-station-long-name",
                "name",
                "--point-schema-case-id",
                "id",
                "--point-schema-metadata-vars",
                "station",
                "--point-schema-metadata-vars",
                "id",
                "--point-schema-metadata-vars",
                "latitude",
                "--point-schema-metadata-vars",
                "longitude",
                "--point-schema-metadata-vars",
                "time",
            ],
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_missing_required_options(runner):
    """Test CLI with missing required options."""
    result = runner.invoke(evaluate.cli_runner, [])
    assert result.exit_code != 0

    # Triggers a failed forecast load as there isn't a valid forecast path
    assert isinstance(result.exception, TypeError)


def test_cli_with_invalid_event_type(runner):
    """Test CLI with invalid event type."""
    result = runner.invoke(
        evaluate.cli_runner,
        ["--event-types", "InvalidEvent"],
    )
    assert result.exit_code != 0

    # Throws a KeyError as the event type is not valid trying to pop from the event_type_map
    assert isinstance(result.exception, KeyError)


def test_cli_with_invalid_paths(runner):
    """Test CLI with invalid paths."""
    result = runner.invoke(
        evaluate.cli_runner,
        [
            "--event-types",
            "HeatWave",
            "--output-dir",
            "/nonexistent/path",
            "--forecast-dir",
            "/nonexistent/path",
            "--cache-dir",
            "/nonexistent/path",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, TypeError)


def test_cli_with_invalid_numeric_values(runner):
    """Test CLI with invalid numeric values."""
    result = runner.invoke(
        evaluate.cli_runner,
        [
            "--event-types",
            "HeatWave",
            "--init-forecast-hour",
            "invalid",
            "--temporal-resolution-hours",
            "invalid",
            "--output-timesteps",
            "invalid",
        ],
    )
    assert result.exit_code != 0
    assert "Error" in result.output


def test_cli_with_override_options(runner, sample_yaml_config):
    """Test CLI with conflicting options (config file and individual options)."""
    result = runner.invoke(
        evaluate.cli_runner,
        [
            "--config-file",
            str(sample_yaml_config),
            "--event-types",
            "Freeze",
        ],
    )
    assert result.exit_code != 0

    # Run should error out because there's a missing point obs file being used in the sample config
    assert isinstance(result.exception, FileNotFoundError)
