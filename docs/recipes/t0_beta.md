# Evaluate t0-beta

[t0-beta](https://huggingface.co/theforecastingcompany/t0-beta) is a general
purpose time-series model from The Forecasting Company. The preparation script
`data_prep/t0_forecasts.py` forecasts every selected variable, pressure level and
grid cell as a target in **one group**, then writes median forecasts for EWB's
`XarrayForecast`. This supports inputs for all five EWB event families.

For each variable, every non-time dimension is flattened into the model's variate
axis `V`. These arrays are concatenated into one `[V, time]` input. Surface fields
are not broadcast across pressure levels. Output is restored to each variable's
original dimensions and coordinates, with `init_time` and `lead_time` replacing
`time`. Shared group IDs let the model use relationships across variables,
levels and locations. These are forecasts, with no future covariates or fitting
on future observations. This recipe does not report a completed full-benchmark run.

## Generate forecasts

Run from a checkout of this repository, with the optional runtime installed:

```shell
uv sync --group dev --all-extras
uv pip install 'tfc-t0>=0.5.0'
uv run --no-sync python data_prep/t0_forecasts.py \
    --input weather_history.nc \
    --variable surface_air_temperature air_pressure_at_mean_sea_level \
        surface_eastward_wind surface_northward_wind \
        air_temperature eastward_wind northward_wind geopotential specific_humidity \
    --init-time 2021-06-25T00:00:00 \
    --context-length 512 --horizon 40 \
    --output t0_beta_forecasts.nc
```

`--variable` accepts multiple names and can be repeated; omit it to select all
input data variables. A temperature-only run can select just
`--variable surface_air_temperature`.

The input is a local NetCDF dataset with a shared regular, strictly increasing
`time` coordinate and a rectilinear `latitude`, `longitude` grid. Variables may
have additional dimensions, such as `level`. Rename variables and coordinates to
EWB conventions before generation. The script preserves units and does not
convert them. Use the following fields for the union of all default tasks:

| Fields | Dimensions besides `time` | Units |
| --- | --- | --- |
| `surface_air_temperature` | `latitude`, `longitude` | K |
| `air_pressure_at_mean_sea_level` | `latitude`, `longitude` | Pa |
| `surface_eastward_wind`, `surface_northward_wind` | `latitude`, `longitude` | m/s |
| `air_temperature` | `level`, `latitude`, `longitude` | K |
| `eastward_wind`, `northward_wind` | `level`, `latitude`, `longitude` | m/s |
| `geopotential` | `level`, `latitude`, `longitude` | m²/s² |
| `specific_humidity` | `level`, `latitude`, `longitude` | kg/kg |

`level` is pressure in **hPa**, including exactly 300 and 500 hPa. Supply a full
vertical profile from near the surface through the upper troposphere for CAPE;
the two levels alone are insufficient. Atmospheric rivers integrate humidity
and wind over supplied levels from 300 hPa downward. Severe convection uses
500 hPa winds for its shear calculation. Tropical cyclone evaluation derives
300–500 hPa thickness from the forecast geopotential.

With `N` spatial cells and `L` pressure levels, these nine physical fields produce
`V = N × (4 + 5L)` targets. For example, 357 cells and 13 levels produce **24,633
joint targets**, not nine targets and not 357 independent model calls.

Subset the grid to cover the event and any spatial buffers needed by its derived
fields before generation; input and output are assembled in memory. Cyclone
tracking needs the storm's surrounding domain, and atmospheric river detection
needs enough area to identify river geometry. Use a suitable grid resolution for
these spatial algorithms; the earlier coarse temperature pilot is not evidence
that the same resolution is adequate for all tasks.

The default `--grouping joint` passes all targets with the same `group_ids`
value in one call per initialization. `--batch-size` does not split this group.
Joint attention and longer contexts increase memory requirements. The optional
`--grouping independent --batch-size 32` provides an independent-series baseline.

Each `--init-time` (repeat the option for multiple initializations) must occur in
the input. The context includes that timestamp and the preceding
`context_length - 1` samples. Later observations are excluded. Lead times start
one input sampling interval after initialization; `horizon` counts samples,
not hours. With six-hourly input, `--horizon 40` produces ten days of forecasts.
Missing or infinite history values are rejected. Negative longitudes are
converted to the 0–360 convention and sorted with their data.

`--context-length` defaults to 512 samples (128 days at six-hourly cadence).
Ensure that much history is available through every requested initialization.
The runtime must be at least 0.5.0 to read t0-beta's normalization configuration.
Use `--device cuda` for GPU inference (CPU is the default). For reproducibility,
pass a Hugging Face commit hash via `--revision` and retain the runtime version,
input provenance and command alongside your results.

## Evaluate all event families

Use EWB's default evaluation objects to retain its targets, derived diagnostics,
thresholds and metrics, replacing only the forecast source:

| Event family | Forecast fields used | Default targets |
| --- | --- | --- |
| Heat wave, freeze | Surface temperature | ERA5, GHCN |
| Tropical cyclone | Sea-level pressure, surface winds, geopotential thickness | IBTrACS |
| Atmospheric river | Pressure-level winds and specific humidity | ERA5 |
| Severe convection | Temperature, winds, humidity and geopotential profiles; surface winds and sea-level pressure | PPH, LSR |

Prepare a forecast file `forecasts/case_<case_id_number>.nc` for **every** case
returned by `ewb.load_cases()`, using that case's required region, history,
initializations and forecast horizon. Each file may contain multiple
initializations. Files must cover the evaluation times and spatial region; the
single initialization in the generation example is not a full benchmark schedule.
Generating per-case files keeps unrelated event regions separate while keeping
all variables, levels and cells within each model call joint.

The following evaluates every case and fails if any forecast file is missing.
For a pilot, explicitly restrict `cases_to_run` before running the loop.

```python
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import extremeweatherbench as ewb

cases_to_run = ewb.load_cases()
templates = ewb.get_brightband_evaluation_objects()
results: list[pd.DataFrame] = []

for case in cases_to_run:
    path = Path("forecasts") / f"case_{case.case_id_number}.nc"
    with xr.open_dataset(path, decode_timedelta=True) as source:
        dataset = source.load()
    valid_times = dataset.init_time.values[:, None] + dataset.lead_time.values[None, :]
    dataset = dataset.assign_coords(valid_time=np.unique(valid_times))

    if case.event_type == "tropical_cyclone":
        # Match EWB's default forecast preprocessing (geopotential to metres).
        thickness = ewb.geopotential_thickness(
            dataset["geopotential"], top_level=300, bottom_level=500
        ) / 9.81
        thickness.attrs["units"] = "m"
        dataset = dataset.assign(geopotential_thickness=thickness)

    evaluation_objects = [
        replace(
            template,
            forecast=ewb.XarrayForecast(
                ds=dataset,
                name="t0-beta",
                variables=template.forecast.variables,
            ),
        )
        for template in templates
        if template.event_type == case.event_type
    ]
    runner = ewb.evaluation(
        case_metadata=[case], evaluation_objects=evaluation_objects
    )
    case_results = runner.run_evaluation()
    if case_results.empty:
        raise RuntimeError(f"No scores returned for case {case.case_id_number}")
    case_results.to_csv(f"t0_beta_case_{case.case_id_number}_results.csv", index=False)
    results.append(case_results)

pd.concat(results, ignore_index=True).to_csv("t0_beta_results.csv", index=False)
```

The auxiliary one-dimensional `valid_time` coordinate is the union of actual
forecast verification times. It lets EWB's initial availability check recognize
forecasts issued before an event starts; EWB subsequently derives its normal
verification coordinates from unchanged initialization and lead times.

The diagnostics (CAPE/shear, atmospheric river masks, cyclone tracks) are derived
from the reconstructed forecasts by EWB; they are not substituted with future
observations. The default evaluation also requires access to its remote target
archives and temperature climatology. Generating local forecasts does not cache
those datasets or guarantee their availability.

Historical reanalysis used as context makes this a reanalysis-initialized
hindcast experiment. It does not establish real-time forecast skill: account
for source availability delays in any operational comparison. Check the model's
training data coverage before claiming out-of-sample results on historical EWB
cases; the public model card does not establish a weather-data training cutoff.
