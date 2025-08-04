"""Unit tests for the evaluate.py CLI interface."""

from unittest.mock import patch

from extremeweatherbench import evaluate_cli
from extremeweatherbench.events import HeatWave
from extremeweatherbench.utils import load_events_yaml


def test_cli_with_yaml_config(runner, sample_yaml_config):
    """Test CLI with YAML config file."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", str(sample_yaml_config)]
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_invalid_config_format(runner, temp_config_dir):
    """Test CLI with invalid config file format."""
    invalid_config = temp_config_dir / "config.txt"
    invalid_config.write_text("invalid config")

    result = runner.invoke(
        evaluate_cli.cli_runner, ["--config-file", str(invalid_config)]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


def test_cli_with_default_flag(runner):
    """Test CLI with --default flag."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(evaluate_cli.cli_runner, ["--default"])
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_individual_options(runner, temp_config_dir):
    """Test CLI with individual command line options."""
    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--event-types",
                "heat_wave",
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
            ],
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()


def test_cli_with_missing_required_options(runner):
    """Test CLI with missing required options."""
    result = runner.invoke(evaluate_cli.cli_runner, [])

    # Returns the help message
    assert result.exit_code == 0


def test_cli_with_invalid_event_type(runner):
    """Test CLI with invalid event type."""
    result = runner.invoke(
        evaluate_cli.cli_runner,
        ["--event-types", "InvalidEvent"],
    )
    assert result.exit_code != 0

    # Throws a KeyError as the event type is not valid trying to pop from the event_type_map
    assert isinstance(result.exception, KeyError)


def test_cli_with_invalid_paths(runner):
    """Test CLI with invalid paths."""
    result = runner.invoke(
        evaluate_cli.cli_runner,
        [
            "--event-types",
            "heat_wave",
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
        evaluate_cli.cli_runner,
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


def test_event_type_constructor_no_case_ids(runner, temp_config_dir):
    """Test that when no case_ids are specified, all cases of the event_type are returned."""
    config_path = temp_config_dir / "config.yaml"
    config_content = """
    event_types:
      - !event_types
        event_type: heat_wave
    """
    config_path.write_text(config_content)

    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", str(config_path)]
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        # Verify that all heat_wave cases were included
        config = mock_evaluate.call_args[1]["eval_config"]
        heat_wave_events = [e for e in config.event_types if isinstance(e, HeatWave)]
        assert len(heat_wave_events) == 1
        # Get the number of heat_wave cases from the events.yaml
        yaml_event_case = load_events_yaml()
        expected_cases = len(
            [c for c in yaml_event_case["cases"] if c["event_type"] == "heat_wave"]
        )
        assert len(heat_wave_events[0].cases) == expected_cases


def test_event_type_constructor_with_case_ids(runner, temp_config_dir):
    """Test that when case_ids are specified, only those cases are returned."""
    config_path = temp_config_dir / "config.yaml"
    config_content = """
    event_types:
      - !event_types
        event_type: heat_wave
        case_ids: [1, 2]
    """
    config_path.write_text(config_content)

    with patch("extremeweatherbench.evaluate.evaluate") as mock_evaluate:
        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", str(config_path)]
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        # Verify that only the specified cases were included
        config = mock_evaluate.call_args[1]["eval_config"]
        heat_wave_events = [e for e in config.event_types if isinstance(e, HeatWave)]
        assert len(heat_wave_events) == 1
        assert len(heat_wave_events[0].cases) == 2


def test_event_type_constructor_mismatched_case_ids(runner, temp_config_dir):
    """Test that when case_ids don't match the event_type, a ValueError is raised."""
    config_path = temp_config_dir / "config.yaml"
    config_content = """
    event_types:
      - !event_types
        event_type: heat_wave
        case_ids:
          - 1
          - 2
          - 35  # Case 35 is not a heat_wave
    """
    config_path.write_text(config_content)

    result = runner.invoke(evaluate_cli.cli_runner, ["--config-file", str(config_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "doesn't match specified event_type" in str(result.exception)
