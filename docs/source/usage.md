# Using ExtremeWeatherBench

## Quickstart 

There are two main ways to use ExtremeWeatherBench, by script or by command line.

To run the Brightband-based evaluation on an existing AIWP model (FCN v2), which 
includes the default 290 cases for heat waves, freezes, severe convective days, 
tropical cyclones, and atmospheric rivers:

```bash
ewb --default
```

or:

```python
from extremeweatherbench import evaluate, defaults, utils

eval_objects = defaults.BRIGHTBAND_EVALUATION_OBJECTS

cases = utils.load_events_yaml()
ewb = ExtremeWeatherBench(cases=cases, 
evaluation_objects=eval_objects)

outputs = ewb.run()

outputs.to_csv('your_outputs.csv')
```

## Running an Evaluation for a Single Event Type

ExtremeWeatherBench has default event types and cases for heat waves, freezes, severe convection, tropical cyclones, and atmospheric rivers.

To run an evaluation, there are three components required: a forecast, a target, and an evaluation object.

```python
from extremeweatherbench import inputs
```
There are two built-in classes to set up a forecast: ZarrForecast and KerchunkForecast. Here is an example of a ZarrForecast, using Weatherbench2's HRES zarr store:

```python
hres_forecast = inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "prediction_timedelta": "lead_time",
        "time": "init_time",
    },
    storage_options={"remote_options": {"anon": True}},
)
```

A forecast needs a `source`, which is a link to the zarr store in this case. It also needs `variables` defined, which are based on CF Conventions (a list of all variables in EWB coming soon). Each forecast will likely have different names for their variables, so a `variable_mapping` dictionary is also essential to process the variables, as well as the coordinates and dimensions. EWB uses lead_time, init_time, and valid_time as time coordinates. The HRES data is mapped from `prediction_timedelta` to `lead_time`, as an example. Finally, `storage_options` define access patterns for the data if needed. These are passed to the opening function, e.g. `xarray.open_zarr`.

Next, a target dataset must be defined as well to evaluate against. For this evaluation, we'll use ERA5:

```python
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)
```

Similarly to forecasts, we need to define the `source`, which here is the ARCO ERA5 provided by Google. Also, the `variables`, `variable_mapping`, and `storage_options` reappear as well for the `inputs.ERA5` class. Both forecasts and targets, if relevant, have an optional `chunks` parameter which defaults to what should be the most efficient value - usually `None` or `'auto'`, but can be changed as seen above.

We then set up an `EvaluationObject` list:

```python
from extremeweatherbench import metrics

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
        forecast=hres_forecast,
    ),
]
```
Which includes the event_type of interest (as defined in the case dictionary or YAML file used), the list of metrics to run, one target, and one forecast.
There can be multiple `EvaluationObjects` which are used for an evaluation run.

Plugging these all in:

```python
from extremeweatherbench import utils, evaluate
case_yaml = utils.load_events_yaml()


ewb_instance = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=heatwave_evaluation_list,
)

outputs = ewb_instance.run(
    # tolerance range is the number of hours before and after the timestamp a
    # validating occurrence is checked in the forecasts
    tolerance_range=48,
    # pre-compute the datasets to avoid recomputing them for each metric
    pre_compute=True,
)

outputs.to_csv('your_file_name.csv')
```

Where the EWB default events YAML file is loaded in using a built-in utility helper function, then applied to an instance of `evaluate.ExtremeWeatherBench` along with the `EvaluationObject` list. Finally, we run the evaluation with the `.run()` method, where in this case keyword arguments are included for some metrics (`tolerance_range`) and the option to `pre_compute`, or load the data into memory, after subsetting and prior to metric calculation. This option is useful when there are many metrics that utilize the same dataset, especially when evaluating existing variables in the data.

The outputs are returned as a pandas DataFrame and can be manipulated inline or afterwards, such as here by saving the evaluation results to a csv.