# Using ExtremeWeatherBench

## Quickstart 

There are two main ways to use ExtremeWeatherBench, by script or by command line.

To run the Brightband-based evaluation on an existing AIWP model (FCN v2), which
includes the default cases for heat waves, freezes, severe convection,
tropical cyclones, and atmospheric rivers (see
[Case Studies](events/case_studies.md)):


```python
import extremeweatherbench as ewb

eval_objects = ewb.defaults.get_brightband_evaluation_objects()
cases = ewb.cases.load_cases()

runner = ewb.evaluate.ExtremeWeatherBench(
    case_metadata=cases, evaluation_objects=eval_objects
)

outputs = runner.run_evaluation()
outputs.to_csv("your_outputs.csv")
```

or:

```bash
ewb --default
```

## API Overview

ExtremeWeatherBench provides a submodule-based API. All classes and functions
are accessed through their submodule:

```python
import extremeweatherbench as ewb

# Main evaluation entry point
ewb.evaluate.ExtremeWeatherBench(...)

# Inputs: targets and forecasts
ewb.inputs.ERA5(...)
ewb.inputs.GHCN(...)
ewb.inputs.IBTrACS()
ewb.inputs.ZarrForecast(...)
ewb.inputs.KerchunkForecast(...)
ewb.inputs.EvaluationObject(...)

# Metrics
ewb.metrics.MeanAbsoluteError()
ewb.metrics.MaximumMeanAbsoluteError()

# Derived variables
ewb.derived.AtmosphericRiverVariables()
ewb.derived.TropicalCycloneTrackVariables()

# Regions
ewb.regions.BoundingBoxRegion(...)

# Cases
ewb.cases.IndividualCase
ewb.cases.load_cases()

# Defaults (pre-built targets, forecasts, and helpers)
ewb.defaults.era5_heatwave_target
ewb.defaults.get_climatology(quantile=0.85)

# Outputs (optional xarray Dataset helpers; pandas remains the default)
ewb.outputs.results_to_dataset(...)
ewb.outputs.write_results(...)
ewb.outputs.drop_empty_slices(...)
```
## Running an Evaluation for a Single Event Type

ExtremeWeatherBench has default event types and cases for heat waves, freezes, severe convection, tropical cyclones, and atmospheric rivers.

To run an evaluation, there are three components required: a forecast, a target, and an evaluation object.

ExtremeWeatherBench requires forecasts to have `init_time`, `lead_time`, `latitude`, and `longitude` dimensions at minimum. If not already in that naming convention, initializing a `ForecastBase` object with a `variable_mapping` to map to those names is required. Other dimensions such as pressure level (`level`) can be included.

Targets require at least a `valid_time` with at least one spatial dimension. Examples include `location`, `station`, or (`latitude`, `longitude`). Forecasts are aligned to targets during the steps immediately prior to evaluating a metric.

```python
import extremeweatherbench as ewb
```
There are three built-in `ForecastBase` classes to set up a forecast: `ZarrForecast`, `XarrayForecast`, and `KerchunkForecast`. Here is an example of a `ZarrForecast`, using Weatherbench2's HRES zarr store:

```python
hres_forecast = ewb.inputs.ZarrForecast(
    source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
    name="HRES",
    variables=["surface_air_temperature"],
    variable_mapping=ewb.inputs.HRES_metadata_variable_mapping,  # built-in mapping
    storage_options={"remote_options": {"anon": True}},
)
```

There are required arguments, namely:

- `source`
- `name`
- `variables`*
- `variable_mapping`

* `variables` can alternatively be defined within one or more metrics, instead of in a `ForecastBase` object.

> **Detailed Explanation**: A forecast needs a `source`, which is a link to the zarr store in this case. A `name` is required to identify the outputs. It also needs `variables` defined, which are based on CF Conventions. A list of variable namings exists in `defaults.py` as `DEFAULT_VARIABLE_NAMES`. Each forecast will likely have different names for their variables, so a `variable_mapping` dictionary is also essential to process the variables, as well as the coordinates and dimensions. EWB uses `lead_time`, `init_time`, and `valid_time` as time coordinates. The HRES data is mapped from `prediction_timedelta` to `lead_time`, as an example. `storage_options` define access patterns for the data if needed. These are passed to the opening function, e.g. `xarray.open_zarr`.

Next, a target dataset must be defined as well to evaluate against. For this evaluation, we'll use ERA5:

```python
era5_heatwave_target = ewb.inputs.ERA5(
    source=ewb.inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    storage_options={"remote_options": {"anon": True}},
    chunks=None,
)
```

Note that EWB provides defaults for arguments, so most users will be able to instead write this (if defining variables with the intent of it applying to all metrics):

```python
era5_heatwave_target = ewb.ERA5(variables=["surface_air_temperature"])
```

Or (if defining variables as arguments to the metrics):

```python
era5_heatwave_target = ewb.ERA5()
```

> **Detailed Explanation**: Similarly to forecasts, we need to define the `source`, which here is the ARCO ERA5 provided by Google. `variables` are used to subset `ewb.inputs.ERA5` in an evaluation; `variable_mapping` defaults to `ewb.inputs.ERA5_metadata_variable_mapping` for many existing variables and likely is not required to be set unless your use case is for less common variables. Both forecasts and targets, if relevant, have an optional `chunks` parameter which defaults to what should be the most efficient value - usually `None` or `'auto'`, but can be changed as seen above. *If using the ARCO ERA5 and setting `chunks=None`, it is critical to order your subsetting by variables -> time -> `.sel` or `.isel` latitude & longitude -> rechunk. [See this Github comment](https://github.com/pydata/xarray/issues/8902#issuecomment-2036435045).

We then set up an `EvaluationObject` list:

```python
heatwave_evaluation_list = [
    ewb.inputs.EvaluationObject(
        event_type="heat_wave",
        metric_list=[
            ewb.metrics.MaximumMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.RootMeanSquaredError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
            ewb.metrics.MaximumLowestMeanAbsoluteError(
                forecast_variable="surface_air_temperature",
                target_variable="surface_air_temperature",
            ),
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
case_yaml = ewb.cases.load_cases()

ewb_instance = ewb.evaluate.ExtremeWeatherBench(
    case_metadata=case_yaml,
    evaluation_objects=heatwave_evaluation_list,
)

outputs = ewb_instance.run_evaluation()
outputs.to_csv("your_file_name.csv")
```

Where the EWB default events YAML file is loaded in using
`ewb.cases.load_cases()`, then applied to an instance
of `ewb.evaluate.ExtremeWeatherBench` along with the `EvaluationObject` list.
Finally, we run the evaluation with the `.run_evaluation()` method, where defaults are
typically sufficient to run with a small to moderate-sized virtual machine.

Running locally is feasible but is typically bottlenecked heavily by IO and network bandwidth. Even on a gigabit connection, the rate of data access is significantly slower compared to within a cloud provider VM.

The outputs are returned as a pandas DataFrame and can be manipulated in the script, a notebook, etc.

## Xarray Output

`run_evaluation()` defaults to the long-form pandas DataFrame shown above
(`output_format="pandas"`). Pass `output_format="xarray"` instead to get an
`xarray.Dataset` with the same information, unflattened:

```python
outputs = runner.run_evaluation(output_format="xarray")
```

The Dataset has one dimension per metadata field (`case_id_number`,
`metric`, `forecast_source`, `target_source`), plus whatever dimensions
the metrics themselves preserved (`lead_time`, `init_time`, and/or
spatial dims such as `latitude`, `longitude`, `level`). There is one
data variable per forecast/target variable pair, named after the
variable if the forecast and target names match, or
`f"{forecast_variable}_vs_{target_variable}"` otherwise. `event_type`
rides along as a non-dim coordinate on `case_id_number` rather than a
dimension of its own:

```
<xarray.Dataset> Size: 532B
Dimensions:                  (lead_time: 4, init_time: 3, metric: 2,
                              forecast_source: 1, target_source: 1,
                              case_id_number: 1)
Coordinates:
  * lead_time                (lead_time) timedelta64[ns] 32B 00:00:00 ... NaT
  * init_time                (init_time) datetime64[ns] 24B 2021-06-25 ... NaT
  * metric                   (metric) <U20 160B 'OnsetError' 'RootMeanSquared...
  * forecast_source          (forecast_source) <U11 44B 'my_forecast'
  * target_source            (target_source) <U9 36B 'my_target'
  * case_id_number           (case_id_number) int64 8B 1
    event_type               (case_id_number) <U9 36B ...
Data variables:
    surface_air_temperature  (case_id_number, metric, forecast_source, target_source, init_time, lead_time) float64 ...
```

### Placeholder labels

Metrics reduce to different dimensions: an RMSE-like metric preserves
`lead_time`, while a metric like onset or duration error preserves
`init_time` instead. A flat cube needs a slot for both, so
`results_to_dataset` pads whichever dimension a given metric's result
lacks with a single out-of-band placeholder label (`NaT` for
datetime/timedelta dimensions, `NaN` otherwise). That's why the
`lead_time` and `init_time` coordinates above each carry one extra,
otherwise-unused label: it's where results lacking that dimension get
parked. Peak metrics (e.g. `MaximumMeanAbsoluteError`) are the
exception: they reduce over a run but report against the lead time at
which it verified, so their values genuinely occupy both `init_time`
and `lead_time` rather than needing either one padded. The values are
correct, and this mirrors the CSV, where an RMSE row already has an
empty `init_time` column, but selecting a single metric leaves that
placeholder label behind as an awkward extra row:

```python
rmse = outputs["surface_air_temperature"].sel(
    metric="RootMeanSquaredError",
    forecast_source="my_forecast",
    target_source="my_target",
    case_id_number=1,
)
```
```
<xarray.DataArray 'surface_air_temperature' (init_time: 3, lead_time: 4)> Size: 96B
array([[       nan,        nan,        nan,        nan],
       [       nan,        nan,        nan,        nan],
       [0.41520745, 0.48798442, 0.4550809 ,        nan]])
Coordinates:
  * init_time        (init_time) datetime64[ns] 24B 2021-06-25 2021-06-26 NaT
  * lead_time        (lead_time) timedelta64[ns] 32B 00:00:00 06:00:00 ... NaT
    ...
```

Use `drop_empty_slices` to collapse it back down to the clean series
you'd expect: it drops every label along a dimension where the data is
entirely missing, then drops any dimension left with only a single
placeholder label remaining.

```python
ewb.outputs.drop_empty_slices(rmse)
```
```
<xarray.DataArray 'surface_air_temperature' (lead_time: 3)> Size: 24B
array([0.41520745, 0.48798442, 0.4550809 ])
Coordinates:
  * lead_time        (lead_time) timedelta64[ns] 24B 00:00:00 06:00:00 12:00:00
    ...
```

`drop_empty_slices` is meant to be used after selecting down to a
single metric (and typically a single case); on the full, unselected
Dataset every dimension mixes real and placeholder labels, so nothing
is entirely missing and it is close to a no-op.

### `sparse=True`

For unaggregated spatial results (e.g. a metric that preserves
`latitude`/`longitude` per case), the dense, padded hypercube can
become unmanageably large: every case ends up padding out to the union
of every other case's spatial grid. Pass `sparse=True` to back the
Dataset's floating-point and datetime/timedelta data variables with
`sparse.COO` arrays instead of densifying them:

```python
outputs = runner.run_evaluation(output_format="xarray", sparse=True)
```

This also covers a run whose metrics preserve different dimensions on
the same variable (e.g. `RootMeanSquaredError`'s `lead_time` alongside
`DurationMeanError`'s `init_time`): `results_to_dataset` builds each
sparse data variable directly, without materializing the padded
hypercube or routing `sparse.COO` arrays through `xr.merge`.

Data variables that aren't floating-point, datetime64, or timedelta64
can't be given a well-defined `sparse.COO` fill value, so they stay
densely backed even under `sparse=True`. In practice this only
matters for non-dim coords promoted to data variables by metrics like
landfall displacement: `forecast_landfall_latitude`/`_longitude`/
`_valid_time` and the `target_landfall_*` equivalents are float or
datetime64 and do sparsify (with a `NaN`/`NaT` fill, respectively);
anything with an incompatible dtype would not.

`results_to_dataset` also logs a warning (without requiring
`sparse=True`) when the estimated dense element count for a run passes
a threshold, as a hint to reach for it. The limitations: `sparse.COO`
isn't a netCDF/zarr-representable format, so writing a sparse Dataset
to disk densifies it first (`sparse=True` only helps while the Dataset
stays in memory), and `drop_empty_slices` can't operate directly on
`sparse.COO`-backed input -- densify a selection first, e.g. with
`utils.maybe_densify_dataarray`, before calling it.

### Saving results

`write_results` writes a list of annotated results, or an
already-converted DataFrame/Dataset, to disk. A Dataset can be written
to `"netcdf"` or `"zarr"`; `"csv"` needs the raw results list or a
DataFrame instead, since a flat cube doesn't round-trip to CSV:

```python
ewb.outputs.write_results(outputs, "results.nc", output_format="netcdf")
ewb.outputs.write_results(outputs, "results.zarr", output_format="zarr")
```

The CLI exposes the same choices with `--output-format {csv,netcdf,zarr}`
(default `csv`) and `--sparse` (valid only alongside `--output-format
netcdf` or `zarr`):

```bash
ewb --default --output-format zarr --sparse
```

## Import Patterns

All of the following import styles work:

```python
import extremeweatherbench as ewb

ewb.inputs.ERA5(...)

from extremeweatherbench import inputs, metrics, cases, evaluate, outputs
from extremeweatherbench.inputs import ERA5
from extremeweatherbench.evaluate import ExtremeWeatherBench
```
