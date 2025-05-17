# Extreme Weather Bench (EWB)

[![Documentation Status](https://readthedocs.org/projects/extremeweatherbench/badge/?version=latest)](https://extremeweatherbench.readthedocs.io/en/latest/?badge=latest)


[Read our blog post here](https://www.brightband.com/blog/extreme-weather-bench)

As AI weather models are growing in popularity, we need a standardized set of community driven tests that evaluate the models across a wide variety of high-impact hazards. Extreme Weather Bench (EWB) builds on the successful work of WeatherBench and introduces a set of high-impact weather events, spanning across multiple spatial and temporal scales and different parts of the weather spectrum. We provide data to use for testing, standard metrics for evaluation by forecasters worldwide for each of the phenomena, as well as impact-based metrics. EWB is a community system and will be adding additional phenomena, test cases and metrics in collaboration with the worldwide weather and forecast verification community.

# EWB paper and talks

* AMS 2025 talk (recording will go live shortly after AMS): https://ams.confex.com/ams/105ANNUAL/meetingapp.cgi/Paper/451220
* EWB paper is in preparation and will be submitted by early Spring 2025

# How do I suggest new data, metrics, or otherwise get involved?

Extreme Weather Bench welcomes your involvement!  The success of a benchmark suite rests on community involvement and feedback. There are several ways to get involved:

* Get involved in community discussion using the discussion board
* Submit new code requests using the issues
* Send us email at hello@brightband.com 

# Installing EWB

Currently, the easiest way to install EWB is using the ```pip``` command:

```
pip install git+https://github.com/brightbandtech/ExtremeWeatherBench.git
```

# How to Run EWB

Running EWB on sample data (included) is straightforward. 
```python
from extremeweatherbench import config, events, evaluate
import pickle 

# Select model
model = 'FOUR_v200_GFS'

# Set up path to directory of file - zarr, json, or parquet
forecast_dir = f'gs://extremeweatherbench/{model}.parq'

# Choose the event types you want to include
event_list = [events.HeatWave,
              events.Freeze]

# Set up configuration object that includes events and the forecast directory
heatwave_configuration = config.Config(
    event_types=event_list,
    forecast_dir=forecast_dir,
    )

# Use ForecastSchemaConfig to map forecast variable names to CF convention-based names used in EWB
# the sample forecast kerchunk references to the CIRA MLWP archive are the default configuration
default_forecast_config = config.ForecastSchemaConfig()

# Run the evaluate script which outputs a dict of event results with associated metrics and variables
cases = evaluate.evaluate(eval_config=heatwave_configuration, forecast_schema_config=default_forecast_config)

# Save the results to a pickle file
with open(f'cases_{model}.pkl', 'wb') as f:
    pickle.dump(cases, f)
```
# EWB case studies and categories
EWB case studies are fully documented [here](docs/AllCaseStudies.md).  
