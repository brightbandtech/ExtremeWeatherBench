from extremeweatherbench import config, events, evaluate
import dacite
import click
from pathlib import Path
import yaml
import json


@click.command()
@click.option(
    "--default",
    is_flag=True,
    help="Use default values for all configurations and use current directory as output",
)
@click.option(
    "--test",
    is_flag=True,
    help="Run a test of the CLI using lightweight data",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to a YAML or JSON configuration file",
)
# Config class options
@click.option(
    "--event-types",
    multiple=True,
    help="List of event types to evaluate (e.g. HeatWave,Freeze)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=str(config.DEFAULT_OUTPUT_DIR),
    help="Directory for analysis outputs",
)
@click.option(
    "--forecast-dir",
    type=click.Path(),
    default=str(config.DEFAULT_FORECAST_DIR),
    help="Directory containing forecast data",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=str(config.DEFAULT_CACHE_DIR),
    help="Directory for caching intermediate data",
)
@click.option(
    "--gridded-obs-path",
    default=config.ARCO_ERA5_FULL_URI,
    help="URI/path to gridded observation dataset",
)
@click.option(
    "--point-obs-path",
    default=config.ISD_POINT_OBS_URI,
    help="URI/path to point observation dataset",
)
@click.option(
    "--remote-protocol",
    default="s3",
    help="Storage protocol for forecast data",
)
@click.option(
    "--init-forecast-hour",
    type=int,
    default=0,
    help="First forecast hour to include",
)
@click.option(
    "--temporal-resolution-hours",
    type=int,
    default=6,
    help="Resolution of forecast data in hours",
)
@click.option(
    "--output-timesteps",
    type=int,
    default=41,
    help="Number of timesteps to include",
)
def cli_runner(default, config_file, test, **kwargs):
    """ExtremeWeatherBench evaluation command line interface.

    Accepts either a config file path or individual configuration options.

    Example command with config file:
    python -m extremeweatherbench.evaluate --config-file config.yaml

    Example command with individual options:
    python -m extremeweatherbench.evaluate \
    --event-types HeatWave Freeze \
    --output-dir ./outputs \
    --forecast-dir ./forecasts \
    """
    if config_file:
        # Load config from file
        if config_file.endswith(".yaml") or config_file.endswith(".yml"):
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
        elif config_file.endswith(".json"):
            with open(config_file) as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON")

        # Convert string paths to Path objects
        for path_field in ["output_dir", "forecast_dir", "cache_dir"]:
            if path_field in config_dict:
                config_dict[path_field] = Path(config_dict[path_field])

        eval_config = dacite.from_dict(data_class=config.Config, data=config_dict)
        forecast_schema_config = dacite.from_dict(
            data_class=config.ForecastSchemaConfig,
            data=config_dict.get("forecast_schema_config", {}),
        )
        point_obs_schema_config = dacite.from_dict(
            data_class=config.PointObservationSchemaConfig,
            data=config_dict.get("point_obs_schema_config", {}),
        )
    elif default:
        event_list = [events.HeatWave]
        eval_config = config.Config(
            event_types=event_list,
            forecast_dir="gs://extremeweatherbench/FOUR_v200_GFS.parq",
        )
        # Run evaluation
        results = evaluate.evaluate(
            eval_config=eval_config,
        )
    elif test:
        event_list = [events.HeatWave]
        eval_config = config.Config(
            event_types=event_list,
            forecast_dir="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
        )
        # Run evaluation
        results = evaluate.evaluate(
            eval_config=eval_config,
        )
        eval_config.output_dir = Path("outputs/test")
        # Save results
        output_path = Path(eval_config.output_dir) / "evaluation_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path)
        click.echo(f"Results saved to {output_path}")
        return
    else:
        # Convert event type strings to actual event classes
        event_type_map = {
            "HeatWave": events.HeatWave,
            "Freeze": events.Freeze,
            # Add other event types as they become available
        }
        event_types = [event_type_map[et] for et in kwargs.pop("event_types")]
        # Extract schema configs
        forecast_schema_dict = {
            k.replace("forecast_schema_", ""): v
            for k, v in kwargs.items()
            if k.startswith("forecast_schema_")
        }
        point_obs_schema_dict = {
            k.replace("point_schema_", ""): v
            for k, v in kwargs.items()
            if k.startswith("point_schema_")
        }

        # Remove schema options from kwargs
        for k in list(kwargs.keys()):
            if k.startswith(("forecast_schema_", "point_schema_")):
                kwargs.pop(k)

        forecast_schema_config = config.ForecastSchemaConfig(**forecast_schema_dict)
        point_obs_schema_config = config.PointObservationSchemaConfig(
            **point_obs_schema_dict
        )
        # Create config objects
        eval_config = config.Config(
            event_types=event_types,
            **{k: v for k, v in kwargs.items() if hasattr(config.Config, k)},
        )
        eval_config.forecast_schema_config = forecast_schema_config
        eval_config.point_obs_schema_config = point_obs_schema_config

    # Run evaluation
    results = evaluate.evaluate(
        eval_config=eval_config,
    )

    # Save results
    output_path = Path(eval_config.output_dir) / "evaluation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    click.echo(f"Results saved to {output_path}")


if __name__ == "__main__":
    cli_runner()
