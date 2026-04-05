# Multi Model Evaluation

Comparing multiple AI weather prediction (AIWP) models against a shared
target is one of the most common EWB workflows. Create one
`EvaluationObject` per model, all sharing the same target and metric
list, then pass them together to a single `ewb.evaluation` call. The
results `DataFrame` carries a `forecast_source` column that labels each
row by model, making comparison straightforward. See
[Usage](../usage.md) for the single-model baseline workflow.

## Example — Comparing four CIRA MLWP models on heatwaves

The CIRA MLWP icechunk store contains eight models, each in its own
zarr group. `ewb.inputs.get_cira_icechunk` is the convenience wrapper
that returns an `XarrayForecast` for any model name.

```python
import extremeweatherbench as ewb
from extremeweatherbench import inputs

model_names = [
    "FOUR_v200_IFS",
    "FOUR_v200_GFS",
    "GRAP_v100_IFS",
    "AURO_v100_IFS",
]

# One EvaluationObject per model; target and metrics are shared
target = ewb.ERA5(variables=["surface_air_temperature"])

metrics_list = [
    ewb.metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    ewb.metrics.MaximumMeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    ewb.metrics.RootMeanSquaredError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=target,
        forecast=inputs.get_cira_icechunk(model_name=name),
    )
    for name in model_names
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run()
outputs.to_csv("multi_model_heatwave.csv", index=False)
```

## Comparing models from different sources

Mix `ZarrForecast`, `XarrayForecast`, and CIRA models in the same run.
Each must have its own `name` to appear distinctly in the output:

```python
import extremeweatherbench as ewb
from extremeweatherbench import inputs

hres = ewb.ZarrForecast(
    source=(
        "gs://weatherbench2/datasets/hres/"
        "2016-2022-0012-1440x721.zarr"
    ),
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

fcnv2_ifs = inputs.get_cira_icechunk("FOUR_v200_IFS")
pangu_ifs = inputs.get_cira_icechunk("PANG_v100_IFS")

target = ewb.ERA5(variables=["surface_air_temperature"])

metrics_list = [
    ewb.metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=metrics_list,
        target=target,
        forecast=model,
    )
    for model in [hres, fcnv2_ifs, pangu_ifs]
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run()
```

## Filtering and plotting results

The output `DataFrame` has a `forecast_source` column that matches each
model's `name`. Use it to pivot or group results for comparison:

```python
import pandas as pd

outputs = pd.read_csv("multi_model_heatwave.csv")

# Mean MAE per model and lead time
mae = outputs[outputs["metric"] == "MeanAbsoluteError"]
pivot = (
    mae.groupby(["forecast_source", "lead_time"])["value"]
    .mean()
    .unstack("forecast_source")
)
print(pivot)
```

## Parallel execution

For large model comparisons, enable parallel execution by passing a
`parallel_config` dictionary to `runner.run()`. The configuration is
forwarded to `joblib.Parallel`:

```python
parallel_config = {
    "backend": "loky",
    "n_jobs": 4,
}
outputs = runner.run(parallel_config=parallel_config)
```

> **Detailed Explanation**: Each `(case, EvaluationObject)` pair becomes
> a `CaseOperator` that EWB can execute in parallel. With four models and
> 337 cases you have up to 1 348 operators. Memory scales with `n_jobs`
> since each worker holds its own copy of the forecast slice for the
> active case. Set `n_jobs` to the number of CPU cores available and
> adjust downward if you run out of memory. On a cloud VM with a fast
> network link, `n_jobs=8` is a good starting point for comparing four
> to eight models.

## Subsetting to a geographic region

Use `RegionSubsetter` to restrict which cases from the full list are
included in the run, for example to focus on North American events only:

```python
import extremeweatherbench as ewb

subsetter = ewb.RegionSubsetter(
    latitude_min=15.0,
    latitude_max=75.0,
    longitude_min=230.0,
    longitude_max=310.0,
)

runner = ewb.evaluation(
    case_metadata=ewb.load_cases(),
    evaluation_objects=eval_objects,
    region_subsetter=subsetter,
)
outputs = runner.run()
```
