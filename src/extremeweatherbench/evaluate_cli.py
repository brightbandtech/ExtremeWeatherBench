import importlib.util
import os
import pathlib
import pickle
from typing import Optional

import click
import pandas as pd

from extremeweatherbench import defaults, evaluate


@click.command()
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
@click.option(
    "--precompute",
    is_flag=True,
    help=(
        "Pre-compute datasets to avoid recomputing them for each metric (faster but "
        "uses more memory)"
    ),
)
def cli_runner(
    default: bool,
    config_file: Optional[str],
    output_dir: Optional[str],
    cache_dir: Optional[str],
    parallel: int,
    save_case_operators: Optional[str],
    precompute: bool,
):
    """ExtremeWeatherBench command line interface.

    This CLI supports two main modes:

    1. Default mode (--default): Uses the predefined Brightband evaluation objects for
       comprehensive weather event evaluation including heat waves, freeze events,
       [severe convection, atmospheric rivers, and tropical cyclones] (bracketed events
       are not yet implemented).

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

        # Use precompute for faster execution (higher memory usage)
        $ ewb --default --precompute
    """
    # Store original output_dir value before setting default
    original_output_dir = output_dir

    # Set default output directory to current working directory
    if output_dir is None:
        output_dir = os.getcwd()

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate that either default or config_file is provided
    if not default and not config_file:
        ctx = click.get_current_context()
        # Check if any non-default arguments were provided
        args_provided = (
            original_output_dir is not None
            or cache_dir is not None
            or parallel != 1
            or save_case_operators is not None
            or precompute
        )

        if not args_provided:
            # No arguments provided, show help and exit 0
            click.echo(ctx.get_help())
            ctx.exit(0)
        else:
            # Some arguments provided but missing required flags, show error
            raise click.UsageError(
                "Either --default or --config-file must be specified"
            )

    if default and config_file:
        raise click.UsageError("Cannot specify both --default and --config-file")

    # Load evaluation objects
    if default:
        click.echo("Using default Brightband evaluation objects...")
        evaluation_objects = defaults.get_brightband_evaluation_objects()
        cases_dict = _load_default_cases()
    else:
        assert config_file is not None  # for mypy
        click.echo(f"Loading evaluation objects from {config_file}...")
        evaluation_objects, cases_dict = _load_config_file(config_file)

    # Initialize ExtremeWeatherBench
    ewb = evaluate.ExtremeWeatherBench(
        case_metadata=cases_dict,
        evaluation_objects=evaluation_objects,
        cache_dir=cache_dir if cache_dir else None,
    )

    # Get case operators
    case_operators = ewb.case_operators
    click.echo(f"Found {len(case_operators)} case operators to evaluate")

    # Save case operators if requested
    if save_case_operators:
        save_path = pathlib.Path(save_case_operators)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(case_operators, f)
        click.echo(f"Case operators saved to {save_case_operators}")

    # Run evaluation
    if parallel > 1:
        click.echo(f"Running evaluation with {parallel} parallel jobs...")
        results = evaluate._run_parallel(
            case_operators, parallel, pre_compute=precompute
        )
    else:
        click.echo("Running evaluation in serial...")
        results = ewb.run(pre_compute=precompute)

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
    from extremeweatherbench.cases import load_ewb_events_yaml_into_case_collection

    return load_ewb_events_yaml_into_case_collection()


def _load_config_file(config_path: str) -> tuple:
    """Load evaluation objects and cases from a Python config file.

    The config file should define:
    - evaluation_objects: List of EvaluationObject instances
    - cases_dict: Dictionary containing case data
    """
    config_path_obj = pathlib.Path(config_path)

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


if __name__ == "__main__":
    cli_runner()
