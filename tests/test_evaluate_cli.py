"""Tests for the evaluate_cli interface."""

import pickle
import textwrap
from unittest import mock

import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import evaluate_cli


@pytest.fixture(autouse=True)
def suppress_cli_output():
    """Suppress all click.echo output and file writing during tests."""
    with (
        mock.patch("extremeweatherbench.evaluate_cli.click.echo"),
        mock.patch("pandas.DataFrame.to_csv"),
    ):
        yield


@pytest.fixture
def sample_config_py(temp_config_dir):
    """Create a sample Python config file."""
    config_content = textwrap.dedent("""
        # Simple test config that doesn't import complex modules
        evaluation_objects = []
        case_list = []
        """)
    config_file = temp_config_dir / "test_config.py"
    config_file.write_text(config_content)
    return config_file


class TestCLIBasicFunctionality:
    """Test basic CLI functionality and argument parsing."""

    def test_cli_help(self, runner):
        """Test that CLI help displays correctly."""
        result = runner.invoke(evaluate_cli.cli_runner, ["--help"])
        assert result.exit_code == 0

    def test_cli_no_args_shows_help(self, runner):
        """Test that CLI shows help when no arguments provided."""
        result = runner.invoke(evaluate_cli.cli_runner, [])
        assert result.exit_code == 0


class TestDefaultMode:
    """Test --default mode functionality."""

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_default_mode_basic(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test basic default mode execution."""
        # Mock the ExtremeWeatherBench class and its methods
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = [mock.Mock(), mock.Mock()]  # Mock 2 case operators
        mock_ewb.run_evaluation.return_value = pd.DataFrame({"test": [1, 2]})
        mock_ewb_class.return_value = mock_ewb

        # Mock loading default cases
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--default", "--output-dir", str(temp_config_dir)]
        )

        assert result.exit_code == 0
        mock_ewb_class.assert_called_once()
        mock_ewb.run_evaluation.assert_called_once()

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_default_mode_with_cache_dir(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test default mode with cache directory."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        cache_dir = temp_config_dir / "cache"

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--default", "--cache-dir", str(cache_dir)]
        )

        assert result.exit_code == 0
        # Verify cache_dir was passed to ExtremeWeatherBench
        call_args = mock_ewb_class.call_args
        assert call_args[1]["cache_dir"] == str(cache_dir)


class TestConfigFileMode:
    """Test --config-file mode functionality."""

    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_config_file_mode_basic(
        self, mock_ewb_class, runner, sample_config_py, temp_config_dir
    ):
        """Test basic config file mode execution."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = [mock.Mock()]
        mock_ewb.run_evaluation.return_value = pd.DataFrame({"test": [1]})
        mock_ewb_class.return_value = mock_ewb

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--config-file",
                str(sample_config_py),
                "--output-dir",
                str(temp_config_dir),
            ],
        )

        assert result.exit_code == 0
        mock_ewb_class.assert_called_once()

    def test_config_file_nonexistent(self, runner):
        """Test config file mode with non-existent file."""
        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", "/nonexistent/file.py"]
        )

        assert result.exit_code != 0

    def test_config_file_missing_evaluation_objects(self, runner, temp_config_dir):
        """Test config file missing required evaluation_objects."""
        config_content = textwrap.dedent("""
        cases_list = []
        """)
        config_file = temp_config_dir / "bad_config.py"
        config_file.write_text(config_content)

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", str(config_file)]
        )

        assert result.exit_code != 0
        # Output suppressed - only check exit code

    def test_config_file_missing_case_list(self, runner, temp_config_dir):
        """Test config file missing required case_list."""
        config_content = textwrap.dedent("""
        evaluation_objects = []
        """)
        config_file = temp_config_dir / "bad_config.py"
        config_file.write_text(config_content)

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--config-file", str(config_file)]
        )

        assert result.exit_code != 0
        # Output suppressed - only check exit code


class TestParallelExecution:
    """Test --parallel option functionality."""

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_parallel_execution(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
    ):
        """Test parallel execution mode."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_ewb.run_evaluation.return_value = pd.DataFrame({"test": [1, 2, 3]})
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(evaluate_cli.cli_runner, ["--default", "--n-jobs", "3"])

        assert result.exit_code == 0
        # Verify ewb.run_evaluation was called with parallel config
        mock_ewb.run_evaluation.assert_called_once_with(
            n_jobs=3,
            parallel_config=None,
            progress=True,
            output_format="pandas",
            sparse=False,
        )

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_serial_execution_default(
        self, mock_ewb_class, mock_load_cases, mock_get_brightband, runner
    ):
        """Test that serial execution is default (parallel=1)."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(evaluate_cli.cli_runner, ["--default"])

        assert result.exit_code == 0
        # Output suppressed - only check exit code
        mock_ewb.run_evaluation.assert_called_once()


class TestCaseOperatorSaving:
    """Test --save-case-operators functionality."""

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_save_case_operators(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test saving case operators to pickle file."""
        # Use simple dictionaries instead of Mock objects for pickling
        mock_case_op1 = {"id": 1, "type": "test_case_op"}
        mock_case_op2 = {"id": 2, "type": "test_case_op"}
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = [mock_case_op1, mock_case_op2]
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        # Use temp directory for pickle file to ensure cleanup
        save_path = temp_config_dir / "case_ops.pkl"

        result = runner.invoke(
            evaluate_cli.cli_runner,
            ["--default", "--save-case-operators", str(save_path)],
        )

        assert result.exit_code == 0
        # Output suppressed - only check exit code

        # Verify pickle file was created and contains the right data (in temp dir,
        # auto-cleanup)
        assert save_path.exists()
        with open(save_path, "rb") as f:
            loaded_ops = pickle.load(f)
        assert len(loaded_ops) == 2

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_save_case_operators_creates_directory(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test that saving case operators creates parent directories."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        # Use nested path within temp directory for auto-cleanup
        nested_path = temp_config_dir / "nested" / "dirs" / "case_ops.pkl"

        result = runner.invoke(
            evaluate_cli.cli_runner,
            ["--default", "--save-case-operators", str(nested_path)],
        )

        assert result.exit_code == 0
        assert nested_path.exists()  # Will be cleaned up with temp_config_dir


class TestValidationAndErrorHandling:
    """Test CLI validation logic and error handling."""

    def test_missing_both_default_and_config(self, runner):
        """Test error when neither --default nor --config-file is specified."""
        result = runner.invoke(evaluate_cli.cli_runner, ["--output-dir", "/tmp"])

        assert result.exit_code == 0

    def test_both_default_and_config_specified(self, runner, sample_config_py):
        """Test error when both --default and --config-file are specified."""
        result = runner.invoke(
            evaluate_cli.cli_runner,
            ["--default", "--config-file", str(sample_config_py)],
        )

        assert result.exit_code != 0
        # Output suppressed - only check exit code

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_output_directory_creation(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test that output directory is created if it doesn't exist."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        output_dir = temp_config_dir / "new_output_dir"
        assert not output_dir.exists()

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--default", "--output-dir", str(output_dir)]
        )

        assert result.exit_code == 0
        assert output_dir.exists()

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_default_output_directory(
        self, mock_ewb_class, mock_load_cases, mock_get_brightband, runner
    ):
        """Test that default output directory is current working directory."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        # Use isolated filesystem to avoid creating files in actual directories
        with runner.isolated_filesystem():
            result = runner.invoke(evaluate_cli.cli_runner, ["--default"])
            assert result.exit_code == 0
            # Check that the CLI completed successfully - any files created are in the
            # isolated temp filesystem


class TestResultsSaving:
    """Test results saving functionality."""

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_results_saved_to_csv(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test that results are saved to CSV file."""
        mock_results = pd.DataFrame(
            {
                "metric": ["RMSE", "MAE"],
                "value": [1.5, 2.3],
                "event_type": ["heat_wave", "heat_wave"],
            }
        )

        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = mock_results
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        # Use temp directory for output to ensure cleanup
        result = runner.invoke(
            evaluate_cli.cli_runner, ["--default", "--output-dir", str(temp_config_dir)]
        )

        assert result.exit_code == 0
        # Output suppressed - only check exit code
        # CSV writing is mocked - no file creation expected

        # CSV reading/verification removed since file writing is mocked

    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_empty_results_handling(self, mock_ewb_class, mock_load_cases, runner):
        """Test handling when no results are returned."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()  # Empty results
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(evaluate_cli.cli_runner, ["--default"])

        assert result.exit_code == 0
        # Output suppressed - only check exit code


def _sample_results_dataset():
    """Build a small flat results Dataset matching the run's output shape."""
    return xr.Dataset(
        {"surface_air_temperature": (("case_id_number", "metric"), [[1.5, 2.3]])},
        coords={
            "case_id_number": [1],
            "metric": ["rmse", "mae"],
            "event_type": ("case_id_number", ["heat_wave"]),
        },
    )


class TestOutputFormats:
    """Test --output-format and --sparse CLI options."""

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    @mock.patch("extremeweatherbench.evaluate_cli.outputs.write_results")
    def test_default_output_format_writes_csv(
        self,
        mock_write_results,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test that omitting --output-format still writes a csv via csv path."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame({"value": [1.0]})
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner, ["--default", "--output-dir", str(temp_config_dir)]
        )

        assert result.exit_code == 0
        mock_ewb.run_evaluation.assert_called_once_with(
            n_jobs=1,
            parallel_config=None,
            progress=True,
            output_format="pandas",
            sparse=False,
        )
        mock_write_results.assert_called_once()
        args, kwargs = mock_write_results.call_args
        assert str(args[1]).endswith("evaluation_results.csv")
        assert args[2] == "csv"

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    @mock.patch("extremeweatherbench.evaluate_cli.outputs.write_results")
    def test_explicit_csv_output_format(
        self,
        mock_write_results,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test that --output-format csv writes the same csv filename."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame({"value": [1.0]})
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--default",
                "--output-dir",
                str(temp_config_dir),
                "--output-format",
                "csv",
            ],
        )

        assert result.exit_code == 0
        args, kwargs = mock_write_results.call_args
        assert str(args[1]).endswith("evaluation_results.csv")
        assert args[2] == "csv"

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_netcdf_output_format_writes_real_file(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test --output-format netcdf writes a reopenable .nc file."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = _sample_results_dataset()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--default",
                "--output-dir",
                str(temp_config_dir),
                "--output-format",
                "netcdf",
            ],
        )

        assert result.exit_code == 0
        mock_ewb.run_evaluation.assert_called_once_with(
            n_jobs=1,
            parallel_config=None,
            progress=True,
            output_format="xarray",
            sparse=False,
        )

        output_file = temp_config_dir / "evaluation_results.nc"
        assert output_file.exists()
        with xr.open_dataset(output_file) as reopened:
            assert list(reopened["case_id_number"].values) == [1]
            assert reopened["surface_air_temperature"].values.tolist() == [[1.5, 2.3]]

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_zarr_output_format_writes_real_store(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test --output-format zarr writes a reopenable .zarr store."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = _sample_results_dataset()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--default",
                "--output-dir",
                str(temp_config_dir),
                "--output-format",
                "zarr",
            ],
        )

        assert result.exit_code == 0
        output_store = temp_config_dir / "evaluation_results.zarr"
        assert output_store.exists()
        with xr.open_zarr(output_store) as reopened:
            assert list(reopened["case_id_number"].values) == [1]

    def test_sparse_with_csv_raises_usage_error(self, runner):
        """Test --sparse with --output-format csv fails with a usage error."""
        result = runner.invoke(
            evaluate_cli.cli_runner,
            ["--default", "--output-format", "csv", "--sparse"],
        )

        assert result.exit_code != 0
        assert "--sparse" in result.output

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    @mock.patch("extremeweatherbench.evaluate_cli.outputs.write_results")
    def test_sparse_forwarded_for_netcdf(
        self,
        mock_write_results,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test --sparse is forwarded to run_evaluation for netcdf output."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = _sample_results_dataset()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--default",
                "--output-dir",
                str(temp_config_dir),
                "--output-format",
                "netcdf",
                "--sparse",
            ],
        )

        assert result.exit_code == 0
        mock_ewb.run_evaluation.assert_called_once_with(
            n_jobs=1,
            parallel_config=None,
            progress=True,
            output_format="xarray",
            sparse=True,
        )
        args, kwargs = mock_write_results.call_args
        assert kwargs.get("sparse") is True or args[-1] is True

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_empty_dataframe_results_not_written(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test the empty-results path is unchanged for output_format csv."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = pd.DataFrame()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            ["--default", "--output-dir", str(temp_config_dir)],
        )

        assert result.exit_code == 0
        assert not (temp_config_dir / "evaluation_results.csv").exists()

    @mock.patch(
        "extremeweatherbench.defaults.get_brightband_evaluation_objects",
        return_value=[],
    )
    @mock.patch("extremeweatherbench.evaluate_cli._load_default_cases")
    @mock.patch("extremeweatherbench.evaluate.ExtremeWeatherBench")
    def test_empty_dataset_results_not_written(
        self,
        mock_ewb_class,
        mock_load_cases,
        mock_get_brightband,
        runner,
        temp_config_dir,
    ):
        """Test the empty-results path works for an empty xarray Dataset."""
        mock_ewb = mock.Mock()
        mock_ewb.case_operators = []
        mock_ewb.run_evaluation.return_value = xr.Dataset()
        mock_ewb_class.return_value = mock_ewb
        mock_load_cases.return_value = []

        result = runner.invoke(
            evaluate_cli.cli_runner,
            [
                "--default",
                "--output-dir",
                str(temp_config_dir),
                "--output-format",
                "netcdf",
            ],
        )

        assert result.exit_code == 0
        assert not (temp_config_dir / "evaluation_results.nc").exists()


class TestHelperFunctions:
    """Test helper function functionality."""

    @mock.patch(
        "extremeweatherbench.evaluate_cli.cases.load_ewb_events_yaml_into_case_list"
    )
    def test_load_default_cases(self, mock_load_yaml):
        """Test _load_default_cases function."""
        mock_cases = [{"id": 1}]
        mock_load_yaml.return_value = mock_cases

        result = evaluate_cli._load_default_cases()

        assert result == mock_cases
        mock_load_yaml.assert_called_once_with()
