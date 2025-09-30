import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from extremeweatherbench import defaults, inputs
from extremeweatherbench.evaluate import ExtremeWeatherBench, _run_parallel, _run_serial


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
@click.option(
    "--forecast-path",
    type=str,
    help="Path to forecast data file (zarr or kerchunk format, supports local paths and remote URLs)",
)
@click.option(
    "--variable-mapping",
    type=str,
    help="JSON string of variable mapping for forecast data",
)
@click.option(
    "--storage-options",
    type=str,
    help="JSON string of storage options for remote forecast data access",
)
@click.option(
    "--preprocess-function",
    type=str,
    help="Python module path and function name for preprocessing (e.g., 'mymodule.preprocess_func')",
)
def cli_runner(
    default: bool,
    config_file: Optional[str],
    output_dir: Optional[str],
    cache_dir: Optional[str],
    parallel: int,
    save_case_operators: Optional[str],
    precompute: bool,
    forecast_path: Optional[str],
    variable_mapping: Optional[str],
    storage_options: Optional[str],
    preprocess_function: Optional[str],
):
    """ExtremeWeatherBench command line interface.

    This CLI supports multiple modes:

    1. Default mode (--default): Uses the predefined Brightband evaluation objects for
       comprehensive weather event evaluation including heat waves and freeze events
       with a FourCastNetv2 forecast dataset.

    2. Default mode with custom forecast (--default --forecast-path --variable-mapping):
       Uses default evaluation objects but with a custom forecast dataset.

    3. Custom mode (--config-file): Uses a Python config file containing custom
       evaluation objects defined by the user.

    The CLI can run evaluations in serial or parallel using joblib, and optionally
    save CaseOperator objects for later use or inspection.

    Examples:
        # Use default evaluation objects
        $ ewb --default

        # Use default evaluation with custom forecast
        $ ewb --default --forecast-path /path/to/forecast.zarr --variable-mapping '{"temp":"surface_air_temperature"}'
        
        # Use default evaluation with remote forecast and storage options
        $ ewb --default --forecast-path gs://bucket/forecast.parq --variable-mapping '{"t2":"surface_air_temperature"}' --storage-options '{"remote_protocol":"s3","remote_options":{"anon":true}}'
        
        # Use default evaluation with custom preprocess function
        $ ewb --default --forecast-path /path/to/forecast.zarr --variable-mapping '{"temp":"surface_air_temperature"}' --preprocess-function mymodule.preprocess_func

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

    output_path = Path(output_dir)
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

    # Validate forecast options when using --default
    if default:
        if forecast_path and variable_mapping:
            click.echo("Using default evaluation objects with custom forecast...")
            evaluation_objects = _create_evaluation_objects_with_custom_forecast(
                forecast_path, variable_mapping, storage_options, preprocess_function
            )
        elif forecast_path or variable_mapping or storage_options:
            # Some forecast options provided but not all required ones
            raise click.UsageError(
                "When using --default with custom forecast, both --forecast-path "
                "and --variable-mapping must be provided"
            )
        else:
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
        evaluation_objects=evaluation_objects,
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
        results = ewb.run(n_jobs=parallel, pre_compute=precompute)
    else:
        click.echo("Running evaluation in serial...")
        results = ewb.run(n_jobs=1, pre_compute=precompute)

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


def _load_preprocess_function(function_str: str):
    """Load a preprocess function from a module.function string.
    
    Args:
        function_str: String in format 'module.path.function_name'
        
    Returns:
        The loaded function
        
    Raises:
        click.ClickException: If function cannot be loaded
    """
    try:
        module_path, function_name = function_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        return func
    except (ValueError, ImportError, AttributeError) as e:
        raise click.ClickException(
            f"Could not load preprocess function '{function_str}': {e}"
        )


def _validate_forecast_path(forecast_path: str) -> None:
    """Validate forecast path (local file or remote URL).
    
    Args:
        forecast_path: Path to validate
        
    Raises:
        click.ClickException: If path is invalid
    """
    # Check if it's a remote URL
    if forecast_path.startswith(('http://', 'https://', 'gs://', 's3://', 'gcs://')):
        # For remote URLs, we can't validate existence, just accept them
        return
    
    # For local paths, check if they exist
    if not Path(forecast_path).exists():
        raise click.ClickException(f"Local file does not exist: {forecast_path}")


def _create_evaluation_objects_with_custom_forecast(
    forecast_path: str, 
    variable_mapping_str: str,
    storage_options_str: Optional[str] = None,
    preprocess_function_str: Optional[str] = None
) -> list:
    """Create evaluation objects using custom forecast data.
    
    Args:
        forecast_path: Path to the forecast data file (local or remote)
        variable_mapping_str: JSON string containing variable mapping
        storage_options_str: Optional JSON string containing storage options
        preprocess_function_str: Optional module.function string for preprocessing
        
    Returns:
        List of evaluation objects with custom forecast
    """
    # Validate the forecast path
    _validate_forecast_path(forecast_path)
    
    try:
        variable_mapping = json.loads(variable_mapping_str)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in --variable-mapping: {e}")
    
    # Parse storage options
    storage_options = {}
    if storage_options_str:
        try:
            storage_options = json.loads(storage_options_str)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in --storage-options: {e}")
    
    # Load preprocess function if specified
    preprocess_func = None
    if preprocess_function_str:
        preprocess_func = _load_preprocess_function(preprocess_function_str)
    
    # Infer forecast type from file extension
    # For URLs, get the path part after the last '/'
    if forecast_path.startswith(('http://', 'https://', 'gs://', 's3://', 'gcs://')):
        # Extract filename from URL
        filename = forecast_path.split('/')[-1]
        extension = Path(filename).suffix.lower()
    else:
        # Local path
        extension = Path(forecast_path).suffix.lower()
    
    if extension == ".zarr" or forecast_path.endswith(".zarr"):
        forecast_class = inputs.ZarrForecast
    elif extension in [".parq", ".parquet", ".json"]:
        forecast_class = inputs.KerchunkForecast
    else:
        raise click.ClickException(
            f"Cannot infer forecast type from extension '{extension}'. "
            "Supported extensions: .zarr, .parq, .parquet, .json"
        )
    
    # Get all unique variables from default evaluation objects
    all_variables = set()
    for eval_obj in defaults.BRIGHTBAND_EVALUATION_OBJECTS:
        all_variables.update(eval_obj.forecast.variables)
    
    # Create custom forecast object
    forecast_kwargs = {
        "source": forecast_path,
        "variables": list(all_variables),
        "variable_mapping": variable_mapping,
        "storage_options": storage_options,
    }
    
    # Add preprocess function if provided
    if preprocess_func is not None:
        forecast_kwargs["preprocess"] = preprocess_func
    
    custom_forecast = forecast_class(**forecast_kwargs)
    
    # Create new evaluation objects with custom forecast
    evaluation_objects = []
    for eval_obj in defaults.BRIGHTBAND_EVALUATION_OBJECTS:
        new_eval_obj = inputs.EvaluationObject(
            event_type=eval_obj.event_type,
            metric_list=eval_obj.metric_list,
            target=eval_obj.target,
            forecast=custom_forecast,
        )
        evaluation_objects.append(new_eval_obj)
    
    return evaluation_objects


if __name__ == "__main__":
    cli_runner()
