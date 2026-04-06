# One Forecast, Two Targets

A single forecast can be evaluated against multiple targets in one run
by creating one `EvaluationObject` per target. This pattern is useful
when you want to compare gridded reanalysis skill against point
observation skill side-by-side, or when different event types share the
same forecast but require distinct ground-truth sources. The full
evaluation pipeline is described in [Usage](../usage.md).

## Why two targets?

ERA5 and GHCN answer different questions about the same forecast:

- **ERA5** (gridded reanalysis) measures how well the model captures the
  spatial temperature field over a region.
- **GHCN** (station observations) measures how well the model matches
  actual thermometer readings at weather stations.

Running both in a single pass is more efficient than separate runs
because EWB opens the forecast data once and caches it across case
operators.

## Example — Heat wave skill against ERA5 and GHCN

```python
import extremeweatherbench as ewb

forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

# Gridded ERA5 target
era5_target = ewb.ERA5(variables=["surface_air_temperature"])

# Point observation GHCN target
ghcn_target = ewb.GHCN()

shared_metrics = [
    ewb.metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    ewb.metrics.MaximumMeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=era5_target,
        forecast=forecast,
    ),
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=ghcn_target,
        forecast=forecast,
    ),
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run_evaluation()
```

The output `DataFrame` has a `target_source` column that distinguishes
rows from each target (`"ERA5"` vs `"GHCN"`), making it straightforward
to compare them:

```python
era5_results  = outputs[outputs["target_source"] == "ERA5"]
ghcn_results  = outputs[outputs["target_source"] == "GHCN"]
```

> **Detailed Explanation**: Each `EvaluationObject` expands into one
> `CaseOperator` per case. With two `EvaluationObjects` and 337 cases
> you get 674 operators; they share the forecast source, so IO is not
> doubled. However, GHCN and ERA5 alignment happens independently for
> each target — GHCN uses nearest-neighbour interpolation to match
> station locations, while ERA5 uses spatial regridding to the forecast
> grid. The `target_source` column is set from the `name` attribute on
> the target object.

## Example — Mixed event types with distinct targets

You can combine different event types and different targets in a single
run. Here, heat wave skill is evaluated against ERA5, while freeze skill
is evaluated against GHCN:

```python
import extremeweatherbench as ewb

forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=ewb.ERA5(variables=["surface_air_temperature"]),
        forecast=forecast,
    ),
    ewb.EvaluationObject(
        event_type="freeze",
        metric_list=[
            ewb.metrics.MinimumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=ewb.GHCN(),
        forecast=forecast,
    ),
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run_evaluation()
```

EWB matches each case's `event_type` field against the
`event_type` on each `EvaluationObject`, so heat wave cases run only
against the ERA5 target, and freeze cases run only against GHCN.

## Metrics that differ by target

Some metrics are more appropriate for gridded targets (spatial
displacement, RMSE over a grid) and others for point observations (MAE
at station locations). Supply different `metric_list` values to each
`EvaluationObject` accordingly:

```python
eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=ewb.ERA5(variables=["surface_air_temperature"]),
        forecast=forecast,
    ),
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=ewb.GHCN(),
        forecast=forecast,
    ),
]
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brightbandtech/extremeweatherbench/blob/main/notebooks/one_forecast_two_targets.ipynb)

## Complete Example

HRES evaluated against ERA5 and GHCN simultaneously for all heat wave cases.

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2022 India Heat Wave — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9009,
    title="2022 India Heat Wave (demo)",
    start_date=datetime.datetime(2022, 4, 28),
    end_date=datetime.datetime(2022, 5, 1),
    location=BoundingBoxRegion.create_region(
        latitude_min=24.0,
        latitude_max=30.0,
        longitude_min=76.0,
        longitude_max=82.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]

forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

era5_target = ewb.ERA5(
    variables=["surface_air_temperature"]
)
ghcn_target = ewb.GHCN()

shared_metrics = [
    ewb.metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
    ewb.metrics.MaximumMeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=era5_target,
        forecast=forecast,
    ),
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=ghcn_target,
        forecast=forecast,
    ),
]

runner = ewb.evaluation(
    case_metadata=cases,
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()

mae = outputs[outputs["metric"] == "MeanAbsoluteError"]
for source in ["ERA5", "GHCN"]:
    mean_mae = mae[
        mae["target_source"] == source
    ]["value"].mean()
    print(f"{source:6s} mean MAE: {mean_mae:.4f} K")
```
