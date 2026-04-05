# Single Case

Running EWB across all 337 default cases is the standard workflow, but
sometimes you only need to evaluate one specific event — to debug a
result, inspect a forecast failure, or prototype a new metric. This
recipe shows two approaches: filtering from the default case list, and
defining a custom `IndividualCase` from scratch. See
[Usage](../usage.md) for the full multi-case evaluation workflow.

## Approach 1 — Filter from the default case list

Load all EWB cases, then keep only the one you care about using its
`case_id_number` or any other field on `IndividualCase`:

```python
import extremeweatherbench as ewb

all_cases = ewb.load_cases()

# Keep only case 42 (inspect the full list to find your case)
single_case = [c for c in all_cases if c.case_id_number == 42]

target = ewb.ERA5(variables=["surface_air_temperature"])
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
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=target,
        forecast=forecast,
    ),
]

runner = ewb.evaluation(
    case_metadata=single_case,
    evaluation_objects=eval_objects,
)
outputs = runner.run()
print(outputs)
```

> **Detailed Explanation**: `ewb.load_cases()` returns a list of
> `IndividualCase` objects. Each object has `case_id_number`, `title`,
> `start_date`, `end_date`, `location`, and `event_type` attributes.
> Passing a single-element list to `ewb.evaluation` is identical to
> running all cases — the evaluation engine is a loop over that list.

## Approach 2 — Define a case from scratch

Use `IndividualCase` directly when you want to evaluate an event that
is not in the EWB default set, or when you want full control over the
spatial bounds and date range:

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# June 2021 Pacific Northwest heat dome
pnw_heat_dome = IndividualCase(
    case_id_number=9999,
    title="2021 Pacific Northwest Heat Dome",
    start_date=datetime.datetime(2021, 6, 26),
    end_date=datetime.datetime(2021, 7, 2),
    location=BoundingBoxRegion.create_region(
        latitude_min=42.0,
        latitude_max=52.0,
        longitude_min=234.0,  # 126 °W in 0–360
        longitude_max=247.0,  # 113 °W in 0–360
    ),
    event_type="heat_wave",
)

target = ewb.ERA5(variables=["surface_air_temperature"])
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
            ewb.metrics.MeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
        ],
        target=target,
        forecast=forecast,
    ),
]

runner = ewb.evaluation(
    case_metadata=[pnw_heat_dome],
    evaluation_objects=eval_objects,
)
outputs = runner.run()
```

> **Detailed Explanation**: `BoundingBoxRegion.create_region` accepts
> `latitude_min`, `latitude_max`, `longitude_min`, and `longitude_max`.
> Longitudes must be in the 0–360 convention to match EWB's internal
> coordinate system. You can convert from −180–180 with
> `ewb.convert_longitude_to_360`. The `event_type` field must match the
> `event_type` on at least one `EvaluationObject`; otherwise, the case
> is skipped by the pipeline.

## Loading cases from a custom YAML file

If you maintain your own YAML of case definitions (same schema as EWB's
`events.yaml`), load them directly:

```python
import extremeweatherbench as ewb

my_cases = ewb.load_individual_cases_from_yaml("path/to/my_cases.yaml")
single_case = [c for c in my_cases if c.case_id_number == 1]

runner = ewb.evaluation(
    case_metadata=single_case,
    evaluation_objects=eval_objects,
)
```

The YAML schema for a single case entry:

```yaml
- case_id_number: 1
  title: My Custom Event
  start_date: 2022-08-10 00:00:00
  end_date: 2022-08-14 00:00:00
  location:
    type: bounded_region
    parameters:
      latitude_min: 30.0
      latitude_max: 45.0
      longitude_min: 260.0
      longitude_max: 280.0
  event_type: heat_wave
```

## Complete Example

A custom case (2021 Pacific Northwest heat dome) evaluated against ERA5 with
no filtering of the default case list required.

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# June 2021 Pacific Northwest heat dome
pnw_heat_dome = IndividualCase(
    case_id_number=9999,
    title="2021 Pacific Northwest Heat Dome",
    start_date=datetime.datetime(2021, 6, 26),
    end_date=datetime.datetime(2021, 7, 2),
    location=BoundingBoxRegion.create_region(
        latitude_min=42.0,
        latitude_max=52.0,
        longitude_min=234.0,  # 126 °W in 0–360
        longitude_max=247.0,  # 113 °W in 0–360
    ),
    event_type="heat_wave",
)

forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

target = ewb.ERA5(variables=["surface_air_temperature"])

eval_objects = [
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
        target=target,
        forecast=forecast,
    ),
]

runner = ewb.evaluation(
    case_metadata=[pnw_heat_dome],
    evaluation_objects=eval_objects,
)
outputs = runner.run()
print(outputs)
```
