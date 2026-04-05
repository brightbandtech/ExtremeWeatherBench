# Evaluate Recent Tropical Cyclones

EWB ships 67 tropical cyclone (TC) cases covering 2020–2024. The default
TC evaluation uses IBTrACS best-track data as the target and a suite of
landfall metrics to assess position, timing, and intensity errors. This
recipe shows you how to run the full TC evaluation, customise the metrics
and models, and interpret the results. See the
[Tropical Cyclones](../events/TropicalCyclones.md) event page for the
full case list.

## Background: TC evaluation in EWB

EWB's TC pipeline has two layers:

1. **Track detection** — `TropicalCycloneTrackVariables` runs a
   TempestExtremes-style algorithm on the forecast to identify the TC
   centre at each time step and lead time.
2. **Landfall metrics** — `LandfallDisplacement`, `LandfallTimeMeanError`,
   and `LandfallIntensityMeanAbsoluteError` compare where and when the
   forecast track crosses coastlines against the IBTrACS best track.

> **Detailed Explanation**: IBTrACS provides 6-hourly best-track
> positions and intensities for all tropical cyclones globally. EWB
> reads the CSV directly from NCEI, preprocesses it using
> `_ibtracs_preprocess` (which unifies wind speed agencies, converts
> knots to m/s, and converts hPa to Pa), then filters to the storm
> matching the case `title`. Track detection on the forecast side
> requires `air_pressure_at_mean_sea_level`, `geopotential_thickness`,
> `surface_eastward_wind`, and `surface_northward_wind`. These are
> fetched from the forecast dataset using the `TropicalCycloneTrackVariables`
> `variables` class attribute.

## Example — Full TC evaluation

```python
import extremeweatherbench as ewb
from extremeweatherbench import inputs, derived, metrics

# Forecast: FCNv2 with IFS initialisation from the CIRA icechunk store
forecast = inputs.get_cira_icechunk(
    model_name="FOUR_v200_IFS",
    variables=[derived.TropicalCycloneTrackVariables()],
)

# Target: IBTrACS best-track data (fetched from NCEI)
ibtracs_target = ewb.IBTrACS()

tc_metrics = [
    # Landfall position error (km); "first" uses the first landfall only
    metrics.LandfallDisplacement(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
    # Landfall timing error (hours; positive = forecast late)
    metrics.LandfallTimeMeanError(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
    # Landfall intensity error: max sustained surface wind speed (m/s)
    metrics.LandfallIntensityMeanAbsoluteError(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=tc_metrics,
        target=ibtracs_target,
        forecast=forecast,
    ),
]

# Load all cases and filter to tropical cyclones only
all_cases = ewb.load_cases()
tc_cases = [c for c in all_cases if c.event_type == "tropical_cyclone"]

runner = ewb.evaluation(
    case_metadata=tc_cases,
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()
outputs.to_csv("tc_evaluation.csv", index=False)
```

## Approach: `"first"` vs `"next"` landfall

`LandfallDisplacement`, `LandfallTimeMeanError`, and
`LandfallIntensityMeanAbsoluteError` all accept an `approach` argument:

| Approach | Behaviour |
|----------|-----------|
| `"first"` | Compare forecast to the first observed landfall for the entire storm. Ignores subsequent landfalls. Good for storms with a single dominant landfall (e.g. Hurricane Ida's US landfall). |
| `"next"` | For each init time, find the *next* landfall after that init time and compare. Good for storms with multiple landfalls or long tracks. |

```python
# Using "next" approach: each init time compared to the upcoming landfall
next_displacement = metrics.LandfallDisplacement(
    approach="next",
    forecast_variable="surface_wind_speed",
    target_variable="surface_wind_speed",
)
```

## Adding intensity metrics beyond landfall

To evaluate track intensity at all lead times (not just at landfall),
add continuous metrics alongside the landfall metrics:

```python
tc_track = derived.TropicalCycloneTrackVariables()

eval_objects = [
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=[
            metrics.MeanAbsoluteError(
                forecast_variable=tc_track,
                target_variable="surface_wind_speed",
            ),
            metrics.LandfallDisplacement(
                approach="first",
                forecast_variable="surface_wind_speed",
                target_variable="surface_wind_speed",
            ),
        ],
        target=ibtracs_target,
        forecast=forecast,
    ),
]
```

> **Detailed Explanation**: When `TropicalCycloneTrackVariables` is
> passed as `forecast_variable`, EWB runs the track algorithm on the
> forecast to produce `surface_wind_speed` and
> `air_pressure_at_mean_sea_level` at each detected track point. These
> are then aligned to the IBTrACS track using the IBTrACS time
> coordinate, and the metric is computed at matched times. Passing a
> plain string variable name (like `"surface_wind_speed"`) skips the
> track detection step and instead uses the gridded forecast wind field
> directly — which is appropriate for landfall metrics but will produce
> different results from track-detected intensity for off-track lead
> times.

## Interpreting the output

The output `DataFrame` for TC evaluations includes:

| Column | Description |
|--------|-------------|
| `metric` | `"landfall_displacement"`, `"landfall_time_me"`, `"landfall_intensity_mae"` |
| `value` | Error value in km (displacement), hours (timing), or m/s (intensity) |
| `init_time` | Forecast initialisation time |
| `case_id_number` | Matches the TC case number in the events list |
| `event_type` | Always `"tropical_cyclone"` |

```python
import pandas as pd

df = pd.read_csv("tc_evaluation.csv")
displacement = df[df["metric"] == "landfall_displacement"]
print(displacement.groupby("forecast_source")["value"].describe())
```

## Complete Example

```python
import extremeweatherbench as ewb
from extremeweatherbench import inputs, derived, metrics

# FCNv2 with IFS initialisation; TropicalCycloneTrackVariables specifies
# the four raw fields needed for track detection.
forecast = inputs.get_cira_icechunk(
    model_name="FOUR_v200_IFS",
    variables=[derived.TropicalCycloneTrackVariables()],
)

ibtracs_target = ewb.IBTrACS()

tc_metrics = [
    metrics.LandfallDisplacement(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
    metrics.LandfallTimeMeanError(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
    metrics.LandfallIntensityMeanAbsoluteError(
        approach="first",
        forecast_variable="surface_wind_speed",
        target_variable="surface_wind_speed",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="tropical_cyclone",
        metric_list=tc_metrics,
        target=ibtracs_target,
        forecast=forecast,
    ),
]

all_cases = ewb.load_cases()
tc_cases = [c for c in all_cases if c.event_type == "tropical_cyclone"]

runner = ewb.evaluation(
    case_metadata=tc_cases,
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()
outputs.to_csv("tc_evaluation.csv", index=False)
print(outputs[["metric", "value", "init_time", "case_id_number"]].head(20))
```
