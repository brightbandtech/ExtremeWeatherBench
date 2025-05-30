# %% [markdown]
# # Rename Forecast Variables
#
# What if you have a zarr forecast ready to be analyzed, but named the required variables differently than what is
# expected by EWB?
#
#


# In this case, we have a zarr forecast with the following variables and their corresponding names in the dataset:
# - 2 meter temperature: **t2m**
# - 10 meter winds: **u10**, **v10**
# - mean sea level pressure: **msl**
#
#

# EWB expects the following variable names in accordance with CF Conventions:
# - **surface_air_temperature**
# - **surface_eastward_wind**
# - **surface_northward_wind**
# - **air_pressure_at_mean_sea_level**
#
#

# Here's how to address this:

# %% [markdown]
# ## Command Line Approach
#
# ```
# ewb rename-forecast-variables \
#     --forecast-dir gs://extremeweatherbench/FOUR_v200_GFS.parq \
#     --forecast-schema-config surface_air_temperature=t2 \
#     --forecast-schema-config surface_eastward_wind=u10 \
#     --forecast-schema-config surface_northward_wind=v10 \
#     --forecast-schema-config air_pressure_at_mean_sea_level=msl
# ```

# %% [markdown]
# ## Command Line Correction
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
cases = evaluate.evaluate(
    eval_config=heatwave_configuration, forecast_schema_config=default_forecast_config
)
