import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import click
import dacite
import yaml

from extremeweatherbench import config, evaluate, events, regions, utils

EVENT_TYPE_MAP = {
    "heat_wave": events.HeatWave,
    "freeze": events.Freeze,
}


def event_type_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode):
    # Extract fields from the mapping node
    fields: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_scalar(key_node)
        # Handle different types of value nodes
        if isinstance(value_node, yaml.nodes.ScalarNode):
            value = loader.construct_scalar(value_node)
        elif isinstance(value_node, yaml.nodes.SequenceNode):
            if len(value_node.value) == 1:
                value = value_node.value[0].value
            elif len(value_node.value) > 1:
                value = loader.construct_sequence(value_node)  # type: ignore
        elif isinstance(value_node, yaml.nodes.MappingNode):
            if len(value_node.value) == 1:
                value = value_node.value[0].value
            elif len(value_node.value) > 1:
                value = loader.construct_mapping(value_node)  # type: ignore
        else:
            raise ValueError(f"Unexpected node type: {type(value_node)}")
        fields[key] = value
    yaml_event_case = utils.load_events_yaml()
    event_type = fields.pop("event_type")
    if "case_ids" in fields:
        case_ids = fields.pop("case_ids")
        if isinstance(case_ids, int):
            case_ids = [case_ids]
        elif isinstance(case_ids, list):
            pass
        elif isinstance(case_ids, str):
            case_ids = [int(case_ids)]
        else:
            breakpoint()
            raise ValueError("case_ids must be an integer or list")

        # Validate that all case_ids match the event_type
        for case_id in case_ids:
            single_case = yaml_event_case["cases"][case_id - 1]
            if single_case["event_type"] != event_type:
                raise ValueError(
                    (
                        f"Case ID {case_id} has event_type {single_case['event_type']} which doesn't match "
                        f"specified event_type {event_type}"
                    )
                )

        cases = []
        for i in case_ids:
            case_data = yaml_event_case["cases"][i - 1].copy()
            # Convert location dict to Region object
            case_data["location"] = regions.map_to_create_region(case_data["location"])
            cases.append(cases.IndividualCase(**case_data))
        input_event_dict = {"cases": cases, "event_type": event_type}
    else:
        input_event_dict = {"cases": yaml_event_case["cases"], "event_type": event_type}
    output = dacite.from_dict(
        data_class=EVENT_TYPE_MAP.get(event_type, None),
        data=input_event_dict,
        config=dacite.Config(type_hooks={regions.Region: regions.map_to_create_region}),
    )
    return output


yaml.SafeLoader.add_constructor(
    "!event_types",
    event_type_constructor,
)


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

        # Convert strings from local directories into Path objects
        dacite_config = dacite.Config(
            type_hooks={utils.PathOrStr: utils.maybe_convert_to_path}
        )

        eval_config = dacite.from_dict(
            data_class=config.Config, data=config_dict, config=dacite_config
        )
        forecast_schema_config = dacite.from_dict(
            data_class=config.ForecastSchemaConfig,
            data=config_dict.get("forecast_schema_config", {}),
        )
        point_obs_schema_config = dacite.from_dict(
            data_class=config.PointObservationSchemaConfig,
            data=config_dict.get("point_obs_schema_config", {}),
        )
        eval_config.forecast_schema_config = forecast_schema_config
        eval_config.point_obs_schema_config = point_obs_schema_config
    elif default:
        event_list = [events.HeatWave, events.Freeze]
        eval_config = config.Config(
            event_types=event_list,
            forecast_dir="gs://extremeweatherbench/FOUR_v200_GFS.parq",
        )
    else:
        # Check if event_types is specified
        if "event_types" not in kwargs:
            raise ValueError(
                "--event-types must be specified when not using --config-file or --default"
            )
        user_specified_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if "forecast_dir" in user_specified_kwargs:
            user_specified_kwargs["forecast_dir"] = utils.maybe_convert_to_path(
                user_specified_kwargs["forecast_dir"]
            )
        if "cache_dir" in user_specified_kwargs:
            user_specified_kwargs["cache_dir"] = utils.maybe_convert_to_path(
                user_specified_kwargs["cache_dir"]
            )
        # Convert event types to event classes
        event_types = [EVENT_TYPE_MAP[et] for et in kwargs.pop("event_types")]

        # Create config objects
        eval_config = config.Config(
            event_types=event_types,
        )
        eval_config = replace(
            eval_config,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
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
