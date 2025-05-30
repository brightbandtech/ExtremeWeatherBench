# %% [markdown]
# # ExtremeWeatherBench Quickstart
#
# Use this notebook as an example for a basic runthrough of ExtremeWeatherBench cases.

# %%
import logging
import sys

from extremeweatherbench import config, evaluate, events

# Suppress annoying logging messages
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)


# %%
# Set the event type(s) to evaluate
event_list = [events.HeatWave]

# Create a configuration to use in the evaluation
# Feel free to try others including HRES at
# gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr
heatwave_configuration = config.Config(
    event_types=event_list,
    forecast_dir="gs://extremeweatherbench/FOUR_v200_GFS.parq",
)

# Create a schema configuration to align the forecast data with observation data present in EWB
# The defaults in ForecastSchemaConfig all work for the FOUR_v200_GFS model above except for the
# surface_air_temperature variable which is t2 in the forecast data.
default_forecast_config = config.ForecastSchemaConfig(surface_air_temperature="t2")

# %%
# Set logging to info to see the progress of the evaluation
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %%
# Run the evaluation given the heatwave configuration and forecast schema configuration
cases = evaluate.evaluate(eval_config=heatwave_configuration)
