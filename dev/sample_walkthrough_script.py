"""
Driving script for a sample EWB evaluation run.
In this script, we will:
1. Define the configuration for the run, including the event type, forecast data, and evaluation metrics.
2. Trigger run
3. Get outputs
"""

import joblib

from extremeweatherbench import evaluate, events

# By default, defined events are in ewb.events. It is possible to extend within the script.
# event types are highly opinionated. All Observations, Metrics, and Cases are defaulted here.
# the most simple run can be:
events = events.EventOperator([events.HeatWave])

# the two sources of forecasts are zarr stores and virtual datasets that can reference
# zarrs, netcdfs, gribs, etc. (see virtualizarr)
# this simplifies inputs, and we can provide sample scripts to give users to convert their
# data to virtual datasets. Or, we can direct them to the virtualizarr docs.
forecast_url = "gs://extremeweatherbench/virtualizarr_store/fcn_v3.parq"

# I think we need to simplify the config approach, here's what I would like for mapping variables,
# for example:

# Users might not know exactly what variables are part of an event type, they can check doing this:
# using EventOperator makes things easy here but another paradigm could be
# events = [events.Heatwave]; event_variables = [n.event_variables for n in events]
# using EventOperator allows for an easy output:
event_variables = events.evaluation_variables

# where event_variables can be a dictionary or class, where users can double check...
# I think providing a table of names used in EWB in the user guide will be helpful for everyone.
print(event_variables)

# Then, if the variables in their forecast do not match the same names, they can provide a simple mapping.
# The pattern is the variable in the event's name : name in the forecast using event_variables
event_variable_mapping = {event_variables.surface_temperature: "2m_temperature"}

# Highly opinionated output of pandas dataframe with evaluation metrics will save if provided a path.
# if no path is provided, it will output in the script in case users want to chain together
# further analyses of metrics:
output_path = "/path/to/output/folder/"

# Cache directory can be specified if users want to have checkpoints in case of failures; defaults to None:
cache_path = "path/to/cache/folder/"

# Plotting module will be used if a plot directory is specified:
plot_dir = "path/to/plot/folder/"
ewb = evaluate.ExtremeWeatherBench(
    events=events,
    forecast_source=forecast_url,
    forecast_variable_mapping=event_variable_mapping,
    output_dir=output_path,
    cache_dir=cache_path,
    plot_dir=plot_dir,
)

# Two options for users to execute an analysis
# One is ExtremeWeatherBench.run(), which will execute in the kernel the script is run as-is:
ewb.run()

# Or ExtremeWeatherBench.create_task_list(), which will return a list of tasks that users can
# run however they see fit parallel (joblib, dask, etc)
ewb_tasks = ewb.create_task_list()

joblib.Parallel(n_jobs=-1)(joblib.delayed(task)() for task in ewb_tasks)

# If a variable is missing from the dataset that the event requires, it will report a
# warning in the logs and return nans for the metrics that require it,
# until there are no variables that will work which will then trigger a break and associated error.

# Some things to consider, not implemented for v1:
# 1. What happens if a user has data with a different resolution than 0.25 and wants
# to use events that require ERA5? For now, we can require them pre-emptively interpolate
