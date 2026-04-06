# Near Real-Time Point Observations

EWB's GHCN target provides access to Global Historical Climatology
Network hourly (GHCNh) station observations through a GCS-hosted parquet
file covering 2020–2024. Because the file is updated periodically, it
can be used to evaluate recent forecasts against real station readings —
including events within weeks of a model run. This page shows you how to
use `GHCN` as a target, how point-to-grid alignment works, and how to
evaluate a forecast against station data for a recent event. See
[Data](../data.md) for dataset provenance and update cadence.

## Why point observations?

ERA5 reanalysis is produced with a multi-week delay and interpolates
observations onto a regular grid. Station data from GHCN is available
much sooner after an event, is not gridded (so it does not smooth
spatial extremes), and reflects what a person actually experienced at
a specific location. Point observations are therefore especially
valuable for heat wave and freeze evaluation, where peak temperatures
at individual stations are operationally important.

> **Detailed Explanation**: GHCN is stored as a polars-compatible parquet
> file at `gs://extremeweatherbench/datasets/ghcnh_all_2020_2024.parq`.
> Each row is one hourly observation from one station, with columns for
> `valid_time`, `latitude`, `longitude`, and `surface_air_temperature`
> (in °C; EWB converts to Kelvin internally). EWB opens it lazily with
> `polars.scan_parquet`, applies temporal and spatial filters per case,
> then collects into memory before converting to an `xr.Dataset` indexed
> by `(valid_time, latitude, longitude)`. Forecast grid points are then
> interpolated to each station location using nearest-neighbour
> interpolation.

## Basic usage

```python
import extremeweatherbench as ewb

forecast = ewb.ZarrForecast(
    source=(
        "gs://weatherbench2/datasets/hres/"
        "2016-2022-0012-1440x721.zarr"
    ),
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

# GHCN defaults: reads from the EWB GCS bucket, no credentials needed
ghcn_target = ewb.GHCN()

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
        target=ghcn_target,
        forecast=forecast,
    ),
]

cases = ewb.load_cases()
heatwave_cases = [c for c in cases if c.event_type == "heat_wave"]

runner = ewb.evaluation(
    case_metadata=heatwave_cases,
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()
outputs.to_csv("ghcn_heatwave_eval.csv", index=False)
```

## Using GHCN and ERA5 side by side

A common pattern is to run GHCN and ERA5 evaluations together so you
can compare station-verified skill against gridded-field skill in one
DataFrame:

```python
import extremeweatherbench as ewb

forecast = ewb.ZarrForecast(
    source=(
        "gs://weatherbench2/datasets/hres/"
        "2016-2022-0012-1440x721.zarr"
    ),
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

shared_metrics = [
    ewb.metrics.MeanAbsoluteError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature",
    ),
]

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=ewb.ERA5(variables=["surface_air_temperature"]),
        forecast=forecast,
    ),
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=shared_metrics,
        target=ewb.GHCN(),
        forecast=forecast,
    ),
]

runner = ewb.evaluation(
    case_metadata=ewb.load_cases(),
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()

# Compare by target
for source in ["ERA5", "GHCN"]:
    subset = outputs[outputs["target_source"] == source]
    print(f"{source} mean MAE:", subset["value"].mean())
```

## Using a custom GHCN-format parquet file

If you have a more recent parquet file (same schema as the EWB default),
point the `source` argument at it:

```python
import extremeweatherbench as ewb

recent_ghcn = ewb.GHCN(
    source="gs://my-bucket/ghcnh_2025.parq",
    storage_options={"anon": False},
)
```

The schema must have at minimum: `valid_time` (datetime), `latitude`
(float, degrees), `longitude` (float, degrees −180–180), and
`surface_air_temperature` (float, °C).

> **Detailed Explanation**: EWB's `GHCN._custom_convert_to_dataset`
> adds 273.15 K to the temperature column (converting °C to Kelvin) and
> converts longitude from −180–180 to 0–360 before building the
> `xr.Dataset`. If your custom file already stores temperatures in
> Kelvin or longitude in 0–360, override `_custom_convert_to_dataset`
> in a subclass to skip those conversions. See the
> [BYOT](your_own_target.md) recipe for guidance on subclassing
> `TargetBase`.

## Station density and sparse regions

GHCN station density varies significantly by region. In data-sparse
areas (ocean basins, polar regions, parts of Africa and South America),
very few stations may fall within a case's bounding box. EWB handles
empty cases gracefully — if no stations are found after spatial
filtering, the case is skipped and a warning is logged.

To inspect which stations contributed to a result, you can directly
query the parquet file:

```python
import polars as pl

ghcn = pl.scan_parquet(
    ewb.DEFAULT_GHCN_URI,
    storage_options={"anon": True},
)

# Stations within the 2021 Pacific Northwest heat dome bounding box
pnw_stations = (
    ghcn
    .filter(
        (pl.col("latitude")  >= 42.0) & (pl.col("latitude")  <= 52.0)
        & (pl.col("longitude") >= -126.0) & (pl.col("longitude") <= -113.0)
        & (pl.col("valid_time") >= "2021-06-26")
        & (pl.col("valid_time") <= "2021-07-02")
    )
    .select(["latitude", "longitude"])
    .unique()
    .collect()
)
print(f"Stations in bounding box: {len(pnw_stations)}")
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1J3I1ITWVup-FGFhZz0MkC6QR2bwlO3nt/view?usp=sharing)

## Complete Example

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2020 SW US Heat Wave — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9007,
    title="2020 SW US Heat Wave (demo)",
    start_date=datetime.datetime(2020, 9, 5),
    end_date=datetime.datetime(2020, 9, 8),
    location=BoundingBoxRegion.create_region(
        latitude_min=32.0,
        latitude_max=38.0,
        longitude_min=243.0,
        longitude_max=249.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]

forecast = ewb.ZarrForecast(
    source=(
        "gs://weatherbench2/datasets/hres/"
        "2016-2022-0012-1440x721.zarr"
    ),
    name="HRES",
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)

ghcn_target = ewb.GHCN()

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
print(mae.groupby("lead_time")["value"].mean())
```
