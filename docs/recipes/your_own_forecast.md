# BYOF (Bring Your Own Forecast)

ExtremeWeatherBench supports any gridded forecast that can be expressed
as an xarray `Dataset`. This page shows you how to wrap your forecast
in one of the three built-in `ForecastBase` subclasses and plug it into
an evaluation. If your data lives in a zarr store, a [kerchunk](https://fsspec.github.io/kerchunk/) reference,
or is already loaded in memory, a corresponding class exists for each
case. Where to go next: the [Usage](../usage.md) page shows the full
evaluation loop once your forecast object is ready.

## Requirements

Every forecast `Dataset` must have these dimensions before evaluation:

| Dimension | Description |
|-----------|-------------|
| `init_time` | Initialisation time (datetime64) |
| `lead_time` | Forecast lead time (timedelta64) |
| `latitude` | Latitude in degrees |
| `longitude` | Longitude in degrees (0–360) |

If your data uses different names, pass a `variable_mapping` dictionary
that maps the original names to EWB's conventions.

> **Detailed Explanation**: EWB derives `valid_time` internally as
> `init_time + lead_time`. You do not need a `valid_time` dimension in
> the raw data; EWB creates it during subsetting. Additional dimensions
> such as `level` (pressure level) are carried through automatically and
> are useful for 3-D variables like temperature and wind.

## Option 1 — ZarrForecast

Use `ZarrForecast` for any forecast stored as a zarr archive, including
remote cloud stores (GCS, S3, Azure) or local paths.

```python
import extremeweatherbench as ewb

hres_forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variables=["surface_air_temperature"],
    variable_mapping=ewb.HRES_metadata_variable_mapping,
    storage_options={"remote_options": {"anon": True}},
)
```

Key arguments:

- `source` — path or URI to the zarr store
- `name` — label that appears in output DataFrames
- `variables` — CF-convention variable names to load; can also be
  specified per-metric instead
- `variable_mapping` — maps your dataset's names to EWB's names (init
  time, lead time, variable names, etc.)
- `storage_options` — passed directly to `xarray.open_zarr`; use
  `{"remote_options": {"anon": True}}` for public buckets
- `chunks` — dask chunking strategy, defaults to `"auto"`

> **Detailed Explanation**: EWB ships several built-in variable mappings
> for common datasets: `ewb.HRES_metadata_variable_mapping`,
> `ewb.ERA5_metadata_variable_mapping`, and
> `ewb.CIRA_metadata_variable_mapping`. For any other model you need to
> supply a dictionary that at minimum maps the model's time coordinate
> names to `"init_time"` and `"lead_time"`, and each variable name to
> its EWB CF-convention equivalent. A full list of expected variable
> names lives in `ewb.DEFAULT_VARIABLE_NAMES`.

### Building a custom variable mapping

```python
my_mapping = {
    "forecast_reference_time": "init_time",
    "step":                     "lead_time",
    "t2m":                      "surface_air_temperature",
    "u10":                      "surface_eastward_wind",
    "v10":                      "surface_northward_wind",
    "msl":                      "air_pressure_at_mean_sea_level",
}

my_zarr_forecast = ewb.ZarrForecast(
    source="gs://my-bucket/my-model.zarr",
    name="MyModel",
    variables=["surface_air_temperature"],
    variable_mapping=my_mapping,
    storage_options={"remote_options": {"anon": True}},
)
```

## Option 2 — XarrayForecast

Use `XarrayForecast` when you have already opened the dataset yourself,
for example when you are assembling a collection of NetCDF files or
applying custom preprocessing before handing data off to EWB.

```python
import xarray as xr
import extremeweatherbench as ewb

# Open and combine your NetCDF files
ds = xr.open_mfdataset(
    "path/to/forecast/*.nc",
    combine="nested",
    concat_dim="init_time",
)

my_forecast = ewb.XarrayForecast(
    ds=ds,
    name="MyNcModel",
    variables=["surface_air_temperature"],
    variable_mapping={
        "forecast_reference_time": "init_time",
        "step":  "lead_time",
        "t2m":   "surface_air_temperature",
    },
)
```

The `ds` argument accepts any `xr.Dataset`. EWB will apply
`variable_mapping` when it first opens the data, so you can pass in a
dataset with the original variable names.

### Adding a custom preprocess function

If you need to transform the dataset before evaluation (unit conversion,
coordinate adjustments, etc.), pass a callable to `preprocess`:

```python
def celsius_to_kelvin(ds: xr.Dataset) -> xr.Dataset:
    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] + 273.15
    return ds

preprocessed_forecast = ewb.XarrayForecast(
    ds=ds,
    name="MyModel_K",
    variable_mapping=my_mapping,
    preprocess=celsius_to_kelvin,
)
```

## Option 3 — KerchunkForecast

Use `KerchunkForecast` for datasets referenced via kerchunk parquet or
JSON files. This is the access pattern used for CIRA MLWP data.

```python
import extremeweatherbench as ewb

cira_kerchunk = ewb.KerchunkForecast(
    source="s3://noaa-oar-mlwp-data/references/FOUR_v200_IFS.parq",
    name="FCNv2_IFS",
    variable_mapping=ewb.CIRA_metadata_variable_mapping,
    storage_options={
        "remote_protocol": "s3",
        "remote_options": {"anon": True},
    },
)
```

> **Detailed Explanation**: [Kerchunk](https://fsspec.github.io/kerchunk/) creates lightweight virtual
> reference files that point to byte ranges in existing NetCDF or HDF5
> archives. This avoids copying data while still allowing zarr-style
> chunked access. The `storage_options` dict is split into a
> `remote_protocol` key (the storage backend, e.g. `"s3"` or `"gcs"`)
> and a `remote_options` dict passed to `fsspec` for credentials. It is recommended to use [VirtualiZarr](https://virtualizarr.readthedocs.io/en/latest/) with an [icechunk](https://icechunk.io/) store over kerchunk in most situations, though this comes at a risk of needing to manage concurrent HTML requests when running many parallel jobs.

## Running an evaluation with your forecast

Once you have a forecast object, combine it with a target and a metric
list inside an `EvaluationObject`:

```python
import extremeweatherbench as ewb

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
        forecast=my_zarr_forecast,
    ),
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run_evaluation()
outputs.to_csv("results.csv")
```

The output is a pandas `DataFrame` with one row per
`(case, metric, lead_time, init_time)` combination. See
[Usage](../usage.md) for a full walkthrough of the output columns.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wSQ2Spznf7_V4tdvrb-WlzdXBcdCQ6Rs?usp=sharing)

## Complete Example

```python
import datetime
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2019 W European Heat Wave — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9002,
    title="2019 W European Heat Wave (demo)",
    start_date=datetime.datetime(2019, 6, 27),
    end_date=datetime.datetime(2019, 6, 30),
    location=BoundingBoxRegion.create_region(
        latitude_min=45.0,
        latitude_max=51.0,
        longitude_min=0.0,
        longitude_max=6.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]

forecast = ewb.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variables=["surface_air_temperature"],
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
    case_metadata=cases,
    evaluation_objects=eval_objects,
)
outputs = runner.run_evaluation()
print(outputs)
```
