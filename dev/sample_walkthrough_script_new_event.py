"""
Driving script for a sample EWB evaluation run.
In this script, we will:
1. Define the configuration for the run, including the event type, forecast data, and evaluation metrics.
2. Trigger run
3. Get outputs
"""

import joblib
import dacite
from extremeweatherbench import evaluate, events, metrics, observations

# If a user wants to create their own event type, they'll need:
# 1. A yaml file or equivalent dictionary in the script to capture case information
# 2. Existing or new Defined variables or DerivedVariable child classes to be defined
# 3. Existing or new Observation child classes
# 4. Existing or new Metric child classes
# 5. A custom event type to wrap all of the above


# Scenario: A user wants to create an event type for Flooding, where they use SEEPS as a metric,
# PRISM precipitation data as the observation source, and accumulated precip as the variable.

## Note: EventContainer needs to be updated and renamed, probably just to Event
## Event should then be an ABC with abstract methods to codify requirements above

# Case data can be either a dictionary or yaml file. It will be parsed using dacite
# either way but a yaml uses yaml.safe_load().

dummy_yaml_data = {
    case_id_number: ...,
    title: ...,
    start_date: ...,
    end_date: ...,
    location: {
        latitude_min: ...,
        latitude_max: ...,
        longitude_min: ...,
        longitude_max: ...,
    },
    event_type: "flood",
}

# Variables can be a list of variable strings and/or DerivedVariables, 
# which require one or more variables to be calculated. 
# It is possible to have different observation and forecast variables.
observation_variables = ['precipitable_water', 'accumulated_precipitation']
forecast_variables = ['precipitable_water', 'accumulated_precipitation']

# Map the observation and forecast variables to each other for metrics.
metric_mapping = {obs_var: forecast_var for obs_var, forecast_var in zip(observation_variables, forecast_variables)}

# Here's code for what would be an Observation child class that loads in PRISM data:
class PRISMObs(Observation):
    def __init__():
        super().__init__(case: case.IndividualCase)
    
    def _open_data_from_source(self, source: str, storage_options: Optional[dict] = None):
        # this will download the data based on the case information
        pass

    def _subset_data_to_case(self, 
                             data: observations.ObservationDataInput,
                             variables: list[str | DerivedVariable],
                             ) -> observations.ObservationDataInput:
        # this will subset the data to the case
        pass

    def _maybe_convert_to_dataset(self, data: observations.ObservationDataInput) -> xr.Dataset:
        # this will convert the data to an xarray dataset if it's not already
        # and return the dataset.
        pass

# Here's an example of what setting up a new metric would look like for SEEPS.
## Note: I would like to have a climatology option here but I'd consider it out of scope for V1.

class SEEPS(metrics.Metric):
    def __init__():
        super().__init__(case: case.IndividualCase,
                         observations: list[Observation] | Observation,
                         observation_variables: list[str | DerivedVariable] | DerivedVariable | str,
                         forecast_variables: list[str | DerivedVariable] | DerivedVariable | str,
                         climatology: Optional[xr.Dataset] = None,
                         )
    
    def calculate_metric(self,
                          ) -> Hashable[str, float]:
        # this will calculate the metric based on observations and variables within both obs and forecasts
        pass



# This is the Event child class that will orchestrate the pieces made above
# when run by evaluate.ExtremeWeatherBench
class Flood(EventContainer):
    def __init__():
        super().__init__(cases: list[case.IndividualCase], 
                         event_type: str,
                         # these don't exist yet but I think they should be here
                         observations: list[Observation] | Observation,
                         variables: list[str | DerivedVariable] | DerivedVariable | str,
                         metrics: list[Metric] | Metric,
                         )

# It seems unintuitive to me to use the Flood class (or an event class) to wrap the cases,
# just to reference the same class again which instantiates the cases.
# This might be worth revisiting if it's an easy fix when coding it up.
dummy_cases = dacite.from_dict(data_class=Flood,
                               data=dummy_yaml_data)
    
cases = [dummy_cases]

flood = Flood(cases=cases,
              event_type='flood',
              observations=[PRISMObs()],
              variables=variables,
              metrics=SEEPS(),
              )

events = events.EventOperator([flood])
event_variables = events.evaluation_variables


event_variable_mapping = {event_variables.accumulated_precipitation: "apcp",
                          event_variables.precipitable_water: "pt"}

output_path = "/path/to/output/folder/"


cache_path = "path/to/cache/folder/"


plot_dir = "path/to/plot/folder/"

forecast_url = "gs://extremeweatherbench/virtualizarr_store/fcn_v3.parq"

ewb = evaluate.ExtremeWeatherBench(
    events=events,
    forecast_source=forecast_url,
    forecast_variable_mapping=event_variable_mapping,
    output_dir=output_path,
    cache_dir=cache_path,
    plot_dir=plot_dir,
)

ewb.run()

print(ewb.results)