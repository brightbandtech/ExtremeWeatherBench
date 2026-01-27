# Extreme Weather Bench (EWB)

[![Documentation Status](https://readthedocs.org/projects/extremeweatherbench/badge/?version=latest)](https://extremeweatherbench.readthedocs.io/en/latest/?badge=latest)

[Read our blog post here](https://www.brightband.com/blog/extreme-weather-bench)

As AI weather models are growing in popularity, we need a standardized set of community driven tests that evaluate the models across a wide variety of high-impact hazards. Extreme Weather Bench (EWB) builds on the successful work of WeatherBench and introduces a set of high-impact weather events, spanning across multiple spatial and temporal scales and different parts of the weather spectrum. We provide data to use for testing, standard metrics for evaluation by forecasters worldwide for each of the phenomena, as well as impact-based metrics. EWB is a community system and will be adding additional phenomena, test cases and metrics in collaboration with the worldwide weather and forecast verification community.

# Events
EWB has cases broken down by multiple event types within `src/extremeweatherbench/data/events.yaml` between 2020 and 2024. EWB case studies are documented [here](docs/events/AllCaseStudies.md).

## Available: 

| Event Type | Number of Cases |
| ---------- | --------------- | 
| 🌇 Heat Waves | 46 |
| 🧊 Freezes | 14 |
| 🌀 Tropical Cyclones | 106 |
| ☔️ Atmospheric Rivers | 56 |
| 🌪️ Severe Convection | 115 | 
| **Total Cases** | 337 |

# EWB paper and talks

* AMS 2025 talk: [Amy](https://ams.confex.com/ams/105ANNUAL/meetingapp.cgi/Paper/451220)
* AMS 2026 talks: [Amy](https://ams.confex.com/ams/106ANNUAL/meetingapp.cgi/Paper/477140), [Taylor](https://ams.confex.com/ams/106ANNUAL/meetingapp.cgi/Paper/477141)
* EWB paper is in preparation and will be submitted in late 2025

# How do I suggest new data, metrics, or otherwise get involved?

We welcome your involvement!  The success of a benchmark suite rests on community involvement and feedback. There are several ways to get involved:

* Get involved in community discussion using the discussion board
* Submit new code requests using the issues
* Send us email at hello@brightband.com 

# Installing EWB

Currently, the easiest way to install EWB is using ```pip``` or ```uv```:

```shell
$ pip install extremeweatherbench

# Or, add to an existing uv virtual environment
$ uv add extremeweatherbench
```

If you'd like to install the most recent updates to EWB:

```shell
$ pip install git+https://github.com/brightbandtech/ExtremeWeatherBench.git 
```

For extra installation options:

```shell
# For running the data prep modules:
$ pip install "extremeweatherbench[data-prep]"
$ uv add "extremeweatherbench[data-prep]"
```

# How to Run EWB

Running EWB on sample data (included) is straightforward. 

## Using Jupyter Notebook or a Script:
 
```python
import extremeweatherbench as ewb

# Load in a forecast; here, we load in GFS initialized FCNv2 from the CIRA MLWP archive with a default variable built-in for convenience
fcnv2_heatwave_forecast = ewb.defaults.cira_fcnv2_heatwave_forecast

# Load in ERA5 with another default convenience variable 
era5_heatwave_target = ewb.defaults.era5_heatwave_target

# EvaluationObjects are used to evaluate a single forecast source against a single target source with a defined event type. Event types are declared with each case. One or more metrics can be evaluated with each EvaluationObject.
heatwave_evaluation_list = [
    ewb.inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            ewb.metrics.MaximumMeanAbsoluteError(),
            ewb.metrics.RootMeanSquaredError(),
            ewb.metrics.MaximumLowestMeanAbsoluteError(),
        ],
        target=era5_heatwave_target,
        forecast=fcnv2_heatwave_forecast,
    ),
]
# Load in the EWB default list of event cases
case_metadata = ewb.cases.load_ewb_events_yaml_into_case_list()

# Create the evaluation class, with cases and evaluation objects declared
ewb_instance = ewb.evaluation(
    case_metadata=case_metadata,
    evaluation_objects=heatwave_evaluation_list,
)

# Execute a parallel run and return the evaluation results as a pandas DataFrame
heatwave_outputs = ewb_instance.run_evaluation(
    parallel_config={'n_jobs':16} # Uses 16 jobs with the loky backend as default
)

# Save the results
heatwave_outputs.to_csv('heatwave_evaluation_results.csv')
```

## Using command line initialization:

```shell
$ ewb --default
```
**Note**: this will run every event type, case, target source, and metric for the individual event type as they become available for GFS initialized FourCastNetv2. It is expected a full evaluation will take some time, even on a large VM.
