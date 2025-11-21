# Extreme Weather Bench (EWB)

**EWB is currently in limited pre-release. Bugs are likely to occur for now.**

**v0.2 leading to v1.0 to be published alongside EWB preprint.**

[Read our blog post here](https://www.brightband.com/blog/extreme-weather-bench)

As AI weather models are growing in popularity, we need a standardized set of community driven tests that evaluate the models across a wide variety of high-impact hazards. Extreme Weather Bench (EWB) builds on the successful work of WeatherBench and introduces a set of high-impact weather events, spanning across multiple spatial and temporal scales and different parts of the weather spectrum. We provide data to use for testing, standard metrics for evaluation by forecasters worldwide for each of the phenomena, as well as impact-based metrics. EWB is a community system and will be adding additional phenomena, test cases and metrics in collaboration with the worldwide weather and forecast verification community.

# Events
EWB has cases broken down by multiple event types within `src/extremeweatherbench/data/events.yaml` between 2020 and 2024. EWB case studies are documented [here](docs/events/AllCaseStudies.md).  

## Available:
| Event Type | Number of Cases |
| ---------- | --------------- | 
| 🌇 Heat Waves | 46 |
| 🧊 Freezes | 14 |

# Events in Development:
| Event Type | Number of Cases |
| ---------- | --------------- | 
| 🌀 Tropical Cyclones | 107 |
| ☔️ Atmospheric Rivers | 56 |
| 🌪️ Severe Convection | 83 | 


# EWB paper and talks

* AMS 2025 talk: https://ams.confex.com/ams/105ANNUAL/meetingapp.cgi/Paper/451220
* EWB paper is in preparation and will be submitted in late 2025

# How do I suggest new data, metrics, or otherwise get involved?

We welcome your involvement!  The success of a benchmark suite rests on community involvement and feedback. There are several ways to get involved:

* Get involved in community discussion using the discussion board
* Submit new code requests using the issues
* Send us email at hello@brightband.com 

# Installing EWB

Currently, the easiest way to install EWB is using the ```pip``` command:

```shell
$ pip install git+https://github.com/brightbandtech/ExtremeWeatherBench.git
```

It is highly recommend to use [uv](https://docs.astral.sh/uv/) if possible:

```shell
$ git clone https://github.com/brightbandtech/ExtremeWeatherBench.git
$ cd ExtremeWeatherBench
$ uv sync
```
# How to Run EWB

Running EWB on sample data (included) is straightforward. 

## Using command line initialization:

```shell
$ ewb --default
```
**Note**: this will run every event type, case, target source, and metric for the individual event type as they become available (currently heat waves and freezes) for GFS initialized FourCastNetv2. It is expected a full evaluation will take some time, even on a large VM.
## Using Jupyter Notebook or a Script:
 
```python
from extremeweatherbench import inputs, metrics, evaluate, utils

# Select model
model = 'FOUR_v200_GFS'

# Set up path to directory of file - zarr or kerchunk/virtualizarr json/parquet
forecast_dir = f'gs://extremeweatherbench/{model}.parq'

# Define a forecast object; in this case, a KerchunkForecast
fcnv2_forecast = inputs.KerchunkForecast(
    source=forecast_dir, # source path
    variables=["surface_air_temperature"], # variables to use in the evaluation
    variable_mapping=inputs.CIRA_metadata_variable_mapping, # mapping to use for variables in forecast dataset to EWB variable names
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}}, # storage options for access
)

# Load in ERA5; source defaults to the ARCO ERA5 dataset from Google and variable mapping is provided by default as well
era5_heatwave_target = inputs.ERA5(
    variables=["surface_air_temperature"], # variable to use in the evaluation
    storage_options={"remote_options": {"anon": True}}, # storage options for access
    chunks=None, # define chunks for the ERA5 data
)

# EvaluationObjects are used to evaluate a single forecast source against a single target source with a defined event type. Event types are declared with each case. One or more metrics can be evaluated with each EvaluationObject.
heatwave_evaluation_list = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=era5_heatwave_target,
        forecast=fcnv2_forecast,
    ),
]
# Load in the EWB default list of event cases
cases = utils.load_events_yaml()

# Create the evaluation class, with cases and evaluation objects declared
ewb_instance = evaluate.ExtremeWeatherBench(
    case_metadata=cases,
    evaluation_objects=heatwave_evaluation_list,
)

# Execute a parallel run and return the evaluation results as a pandas DataFrame
heatwave_outputs = ewb_instance.run(
    n_jobs=16, # use 16 processes
    pre_compute=True, # load case data into memory before metrics are computed. Useful with smaller evaluation datasets with many metrics
)

# Save the results
outputs.to_csv('heatwave_evaluation_results.csv')
```
