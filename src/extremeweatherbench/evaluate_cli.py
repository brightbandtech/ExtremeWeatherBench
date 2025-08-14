import importlib.util
import os
import pickle
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from extremeweatherbench import defaults
from extremeweatherbench.evaluate import ExtremeWeatherBench, compute_case_operator


@click.command(no_args_is_help=True)
@click.option(
    "--default",
    is_flag=True,
    help="Use default Brightband evaluation objects with current directory as output",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to a config.py file containing evaluation objects",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Directory for analysis outputs (default: current directory)",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Optional directory for caching intermediate data",
)
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=1,
    help="Number of parallel jobs using joblib (default: 1 for serial execution)",
)
@click.option(
    "--save-case-operators",
    type=click.Path(),
    help="Save CaseOperator objects to a pickle file at this path",
)
def cli_runner(
    default: bool,
    config_file: Optional[str],
    output_dir: Optional[str],
    cache_dir: Optional[str],
    parallel: int,
    save_case_operators: Optional[str],
):
    """ExtremeWeatherBench command line interface.

    This CLI supports two main modes:

    1. Default mode (--default): Uses the predefined Brightband evaluation objects for
       comprehensive weather event evaluation including heat waves, freeze events,
       severe convection, atmospheric rivers, and tropical cyclones.

    2. Custom mode (--config-file): Uses a Python config file containing custom
       evaluation objects defined by the user.

    The CLI can run evaluations in serial or parallel using joblib, and optionally
    save CaseOperator objects for later use or inspection.

    Examples:
        # Use default evaluation objects
        $ ewb --default

        # Use custom config file with parallel execution
        $ ewb --config-file my_config.py --parallel 4

        # Save case operators to pickle file
        $ ewb --default --save-case-operators case_ops.pkl

        # Use custom output and cache directories
        $ ewb --default --output-dir ./results --cache-dir ./cache
    """
    # Set default output directory to current working directory
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate that either default or config_file is provided
    if not default and not config_file:
        raise click.UsageError("Either --default or --config-file must be specified")

    if default and config_file:
        raise click.UsageError("Cannot specify both --default and --config-file")

    # Load evaluation objects
    if default:
        click.echo("Using default Brightband evaluation objects...")
        evaluation_objects = defaults.BRIGHTBAND_EVALUATION_OBJECTS
        cases_dict = _load_default_cases()
    else:
        assert config_file is not None  # for mypy
        click.echo(f"Loading evaluation objects from {config_file}...")
        evaluation_objects, cases_dict = _load_config_file(config_file)

    # Initialize ExtremeWeatherBench
    ewb = ExtremeWeatherBench(
        cases=cases_dict,
        metrics=evaluation_objects,
        cache_dir=cache_dir if cache_dir else None,
    )

    # Get case operators
    case_operators = ewb.case_operators
    click.echo(f"Found {len(case_operators)} case operators to evaluate")

    # Save case operators if requested
    if save_case_operators:
        save_path = Path(save_case_operators)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(case_operators, f)
        click.echo(f"Case operators saved to {save_case_operators}")

    # Run evaluation
    if parallel > 1:
        click.echo(f"Running evaluation with {parallel} parallel jobs...")
        results = _run_parallel_evaluation(case_operators, parallel)
    else:
        click.echo("Running evaluation in serial...")
        results = ewb.run()

    # Save results
    output_file = output_path / "evaluation_results.csv"
    if isinstance(results, pd.DataFrame) and not results.empty:
        results.to_csv(output_file, index=False)
        click.echo(f"Results saved to {output_file}")
        click.echo(f"Evaluated {len(results)} cases")
    else:
        click.echo("No results to save")


def _load_default_cases() -> dict:
    """Load default case data for default evaluation objects."""
    from extremeweatherbench.utils import load_events_yaml

    return load_events_yaml()


def _load_config_file(config_path: str) -> tuple:
    """Load evaluation objects and cases from a Python config file.

    The config file should define:
    - evaluation_objects: List of EvaluationObject instances
    - cases_dict: Dictionary containing case data
    """
    config_path_obj = Path(config_path)

    # Load the config module
    spec = importlib.util.spec_from_file_location("config", str(config_path_obj))
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load config file: {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Extract required attributes
    if not hasattr(config_module, "evaluation_objects"):
        raise click.ClickException("Config file must define 'evaluation_objects' list")

    if not hasattr(config_module, "cases_dict"):
        raise click.ClickException("Config file must define 'cases_dict' dictionary")

    return config_module.evaluation_objects, config_module.cases_dict


def _run_parallel_evaluation(case_operators, n_jobs: int) -> pd.DataFrame:
    """Run case operators in parallel using joblib."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_case_operator)(case_op) for case_op in case_operators
    )

    # Filter out None results and concatenate
    valid_results = [r for r in results if r is not None]
    if valid_results:
        return pd.concat(valid_results, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    cli_runner()
