# BYOV (Bring Your Own Variable)

EWB ships several derived variables out of the box:
`AtmosphericRiverVariables` (IVT, AR mask, land intersection),
`CravenBrooksSignificantSevere` (mixed-layer CAPE × shear), and
`TropicalCycloneTrackVariables` (TempestExtremes-style track detection).
When you need a derived quantity that isn't covered — a custom composite
index, a variable computed from pressure levels, etc. — you can subclass
`DerivedVariable` and use it anywhere a variable name string is accepted.
See [Usage](../usage.md) for how variables feed into `EvaluationObject`.

## The `DerivedVariable` interface

Two things are required of every subclass:

1. A class-level `variables` list — the names of the raw inputs your
   derivation needs (EWB uses this to subset the dataset before calling
   `derive_variable`).
2. A `derive_variable(self, data, *args, **kwargs)` method that accepts
   an `xr.Dataset` and returns an `xr.DataArray` (or `xr.Dataset` for
   multi-output variables).

```python
class DerivedVariable(abc.ABC):
    variables: list[str]  # raw inputs required

    def derive_variable(
        self, data: xr.Dataset, *args, **kwargs
    ) -> xr.DataArray:
        ...
```

> **Detailed Explanation**: EWB calls `derive_variable` after subsetting
> the forecast (or target) dataset to the current case. The `data`
> argument therefore contains only the spatial and temporal extent of the
> active case, already renamed via `variable_mapping`. The `kwargs` may
> include `case_metadata` (an `IndividualCase` object) and, for derived
> variables that set `requires_target_dataset = True`, a
> `_target_dataset` keyword holding the subset target data.

## Example 1 — Surface dewpoint depression

This computes 2 m dewpoint depression (temperature minus dewpoint) as a
single derived variable:

```python
import xarray as xr
import extremeweatherbench as ewb
from extremeweatherbench.derived import DerivedVariable


class DewpointDepression(DerivedVariable):
    """2 m dewpoint depression (T2m - Td2m) in Kelvin."""

    variables = [
        "surface_air_temperature",
        "surface_dewpoint_temperature",
    ]

    def __init__(self, name: str = "dewpoint_depression"):
        super().__init__(name=name)

    def derive_variable(
        self, data: xr.Dataset, *args, **kwargs
    ) -> xr.DataArray:
        depression = (
            data["surface_air_temperature"]
            - data["surface_dewpoint_temperature"]
        )
        depression.name = self.name
        return depression
```

### Using a derived variable in an evaluation

Pass a `DerivedVariable` instance anywhere a variable name string is
accepted — inside `variables` on a forecast or target, or as
`forecast_variable` / `target_variable` on a metric:

```python
import extremeweatherbench as ewb

dd = DewpointDepression()

my_metric = ewb.metrics.MeanAbsoluteError(
    forecast_variable=dd,
    target_variable=dd,
)

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[my_metric],
        target=ewb.ERA5(),
        forecast=my_forecast,
    ),
]

cases = ewb.load_cases()
runner = ewb.evaluation(case_metadata=cases, evaluation_objects=eval_objects)
outputs = runner.run()
```

> **Detailed Explanation**: When EWB encounters a `DerivedVariable` in
> a metric's `forecast_variable` or `target_variable`, it uses the
> `variables` class attribute to pull the required raw inputs from the
> dataset, then calls `derive_variable` to produce the derived DataArray
> before computing the metric. Only one `DerivedVariable` per
> `EvaluationObject` is supported; if you need multiple derived
> variables, create separate `EvaluationObject` instances for each.

## Example 2 — Multi-level wind speed

For a variable that requires pressure-level data, include the 3-D
dimension variable names in `variables`:

```python
import numpy as np
import xarray as xr
from extremeweatherbench.derived import DerivedVariable


class WindSpeed500hPa(DerivedVariable):
    """Wind speed at 500 hPa from u and v components."""

    variables = ["eastward_wind", "northward_wind"]

    def __init__(self, name: str = "wind_speed_500hPa"):
        super().__init__(name=name)

    def derive_variable(
        self, data: xr.Dataset, *args, **kwargs
    ) -> xr.DataArray:
        u500 = data["eastward_wind"].sel(level=500)
        v500 = data["northward_wind"].sel(level=500)
        speed = np.sqrt(u500 ** 2 + v500 ** 2)
        speed.name = self.name
        return speed
```

## Example 3 — Multi-output derived variable

When your derivation produces several outputs (like `AtmosphericRiverVariables`),
return an `xr.Dataset` and specify which variables downstream code
should see via `output_variables`:

```python
import xarray as xr
from extremeweatherbench.derived import DerivedVariable


class HeatStressIndex(DerivedVariable):
    """Wet-bulb globe temperature approximation and heat index."""

    variables = [
        "surface_air_temperature",
        "surface_relative_humidity",
    ]

    def __init__(
        self,
        name: str = "heat_stress",
        output_variables=None,
    ):
        if output_variables is None:
            output_variables = ["heat_index", "wbgt_approx"]
        super().__init__(name=name, output_variables=output_variables)

    def derive_variable(
        self, data: xr.Dataset, *args, **kwargs
    ) -> xr.Dataset:
        t = data["surface_air_temperature"] - 273.15  # to °C
        rh = data["surface_relative_humidity"]

        # Simplified heat index (Rothfusz regression, °C)
        hi = (
            -8.784695
            + 1.61139411 * t
            + 2.338549 * rh
            - 0.14611605 * t * rh
            - 0.01230809 * t ** 2
            - 0.01642482 * rh ** 2
        )
        # Very rough WBGT approximation
        wbgt = 0.7 * (t - (100 - rh) / 5) + 0.3 * t

        return xr.Dataset(
            {"heat_index": hi, "wbgt_approx": wbgt}
        )
```

Specify a single output variable when creating the metric:

```python
import extremeweatherbench as ewb

hi_variable = HeatStressIndex(output_variables=["heat_index"])

hi_metric = ewb.metrics.MaximumMeanAbsoluteError(
    forecast_variable=hi_variable,
    target_variable=hi_variable,
)
```
