import click


@click.command(no_args_is_help=True)
@click.option(
    "--default",
    is_flag=True,
    help="Use default values for all configurations and use current directory as output",
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
    help="List of event types to evaluate (e.g. heatwave, freeze)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Directory for analysis outputs",
)
@click.option(
    "--forecast-dir",
    type=click.Path(),
    help="Directory containing forecast data",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Directory for caching intermediate data",
)
@click.option(
    "--gridded-obs-path",
    help="URI/path to gridded observation dataset",
)
@click.option(
    "--point-obs-path",
    help="URI/path to point observation dataset",
)
@click.option(
    "--remote-protocol",
    help="Storage protocol for forecast data (where the forecast data is stored on a cloud service or locally)",
)
@click.option(
    "--init-forecast-hour",
    type=int,
    help="First forecast hour to include",
)
@click.option(
    "--temporal-resolution-hours",
    type=int,
    help="Resolution of forecast data in hours",
)
@click.option(
    "--output-timesteps",
    type=int,
    help="Number of timesteps to include",
)
def cli_runner(default, config_file, **kwargs):
    """ExtremeWeatherBench command line interface.

    Accepts either a default flag (--default), a config file path (--config-file), or individual configuration options.

    If input forecast or point observation variables are not the ewb default metadata (likely the case for most users),
    the variable names must be specified in the config file (e.g. surface_temperature is "2_meter_temperature"
    in your forecast dataset).

    Individual flags will override config file values.

    The default flag uses a prebuilt virtualizarr parquet of CIRA MLWP data for FourCastNet, ERA5 for gridded
    observations, and GHCN hourly data for point observations from the Brightband EWB GCS bucket for HeatWave
    and Freeze events.

    Examples:
        $ ewb --config-file config.yaml
        $ ewb --event-types HeatWave Freeze --output-dir ./outputs --forecast-dir ./forecasts
    """
    pass


if __name__ == "__main__":
    cli_runner()
