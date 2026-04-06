# Accessing a CIRA Forecast

We have a dedicated virtual reference [icechunk](https://icechunk.io/) store for CIRA data **up to May 26th, 2025** available at `gs://extremeweatherbench/cira-icechunk`. Compared to using parquet virtual references, we have seen a speed improvements of around 2x with ~25% more memory usage.

## Accessing a CIRA Model from the store

```python
import icechunk
from extremeweatherbench import inputs

storage = icechunk.gcs_storage(
    bucket="extremeweatherbench", prefix="cira-icechunk", anonymous=True
)
group_list = inputs.list_groups_in_icechunk_datatree(storage)
```

`group_list` is list of each group within the DataTree. Note that the list order will change, do not code a fixed numerical index in based on this output.

```['/',
 '/GRAP_v100_IFS',
 '/FOUR_v200_GFS',
 '/PANG_v100_IFS',
 '/PANG_v100_GFS',
 '/AURO_v100_IFS',
 '/FOUR_v200_IFS',
 '/GRAP_v100_GFS',
 '/AURO_v100_GFS']
 ```

## Loading the data as an XarrayObject

```python

# Helper function to access the virtual dataset
fcnv2 = inputs.get_cira_icechunk(model_name='FOUR_v200_IFS')
```

`fcnv2` is a `ForecastBase` object ready to be used within EWB's evaluation framework. 

> **Detailed Explanation**: `inputs.get_cira_icechunk` is syntactic sugar for this: 
```python
import icechunk

storage = icechunk.gcs_storage(
    bucket="extremeweatherbench", prefix="cira-icechunk", anonymous=True
)

fcnv2_icechunk_ds = inputs.open_icechunk_dataset_from_datatree(
    storage=storage, 
    group="FOUR_v200_IFS", 
    authorize_virtual_chunk_access=inputs.CIRA_CREDENTIALS
    )

fcnv2 = inputs.XarrayForecast(
    ds=fcnv2_icechunk_ds,
    variable_mapping=inputs.CIRA_metadata_variable_mapping
    )
```

Which is a three step process of accessing the icechunk storage, loading the dataset from the datatree/zarr group format, and finally applying that `Dataset` in a `ForecastBase` object.

## Set up metrics and target for evaluation

```python
metrics_list = [

    # Assign the forecast and target variable based on EWB's variable naming
    metrics.MaximumMeanAbsoluteError(
        forecast_variable='surface_air_temperature', 
        target_variable='surface_air_temperature'
        )
    
    # Arbitrary thresholds to check CSI on the temperature; how did the models do
    # spatially for the upper echelons of heat?
    metrics.CriticalSuccessIndex(
        forecast_variable='surface_air_temperature', 
        target_variable='surface_air_temperature',
        forecast_threshold=310,
        target_threshold=310
        )
        
]

# Load in GHCNh target
ghcn_target = inputs.GHCN()
```

## Load in case metadata

```python

# Use EWB's cases and subset to the first two heat waves
case_vals = cases.load_ewb_events_yaml_into_case_list()
case_vals = [case for case in case_vals if case.case_id_number in [1, 2]]
```

From here, all we need to do is plug in the event type, metric list, target, and forecast
to an `EvaluationObject` and run EWB's evaluation engine:

```python

evaluation_object = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=ghcn_target,
        forecast=fcnv2,
    ),
]

ewb = evaluate.ExtremeWeatherBench(
    case_metadata=case_vals,
    evaluation_objects=evaluation_object
    )

# Set up parallel configuration for the run to pass into joblib
parallel_config = {
    'backend': 'loky',
    'n_jobs': 4,
    }

output = ewb.run_evaluation(parallel_config=parallel_config)
```
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1w6poub9_6kYAX5UhDxwDD1FT4XPoTGS0/view?usp=sharing)

## Complete Example

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench import inputs, metrics, evaluate
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2021 Pacific NW Heat Dome — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9001,
    title="2021 Pacific NW Heat Dome (demo)",
    start_date=datetime.datetime(2021, 6, 27),
    end_date=datetime.datetime(2021, 6, 30),
    location=BoundingBoxRegion.create_region(
        latitude_min=46.0,
        latitude_max=52.0,
        longitude_min=236.0,
        longitude_max=242.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]

fcnv2 = inputs.get_cira_icechunk(model_name="FOUR_v200_IFS")
ghcn_target = inputs.GHCN()

metrics_list = [
    metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    metrics.MaximumMeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    metrics.CriticalSuccessIndex(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
        forecast_threshold=310,
        target_threshold=310,
    ),
]

evaluation_object = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=ghcn_target,
        forecast=fcnv2,
    ),
]

ewb_runner = evaluate.ExtremeWeatherBench(
    case_metadata=cases,
    evaluation_objects=evaluation_object,
)

output = ewb_runner.run_evaluation()
print(output)
```
