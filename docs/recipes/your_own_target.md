# BYOT (Bring Your Own Target)

EWB ships with built-in targets for ERA5, GHCN station observations,
IBTrACS tropical cyclone best-track data, Local Storm Reports (LSR), and
the Practically Perfect Hindcast (PPH). When none of these fit your use
case — for example, you have gridded reanalysis from another provider,
or proprietary station data — you can subclass `TargetBase` and plug
your data directly into the evaluation pipeline. The [Usage](../usage.md)
page shows how targets combine with forecasts and metrics.

## The `TargetBase` interface

Two abstract methods must be implemented:

| Method | Purpose |
|--------|---------|
| `_open_data_from_source` | Open (lazily, where possible) the full dataset |
| `subset_data_to_case` | Subset opened data to one `IndividualCase` |

An optional third method, `maybe_align_forecast_to_target`, controls how
the forecast grid is regridded or interpolated to match target
coordinates. The base implementation passes forecast and target through
unchanged; override it when your target has unusual spatial coordinates.

## Example 1 — Custom gridded zarr target

The following example wraps a MERRA-2 zarr store as an EWB target:

```python
import dataclasses
import xarray as xr
import extremeweatherbench as ewb
from extremeweatherbench import inputs, cases


@dataclasses.dataclass
class MERRA2(inputs.TargetBase):
    """Target class for MERRA-2 gridded reanalysis data."""

    name: str = "MERRA2"
    source: str = "gs://my-bucket/merra2.zarr"
    variable_mapping: dict = dataclasses.field(
        default_factory=lambda: {
            "T2M":  "surface_air_temperature",
            "time": "valid_time",
        }
    )

    def _open_data_from_source(self):
        return xr.open_zarr(
            self.source,
            storage_options=self.storage_options,
            chunks=None,
        )

    def subset_data_to_case(self, data, case_metadata, **kwargs):
        # Reuse EWB's built-in zarr subsetter for gridded targets
        return inputs.zarr_target_subsetter(data, case_metadata)
```

> **Detailed Explanation**: `zarr_target_subsetter` handles the two
> steps that every gridded target needs: subsetting time with `.sel`
> along `valid_time` (or `time`) and masking the dataset to the case's
> `location` region. If your time coordinate has a different name, pass
> `time_variable="my_time"` as a keyword argument. For targets that do
> not use `valid_time` as their time dimension name, the helper
> automatically checks for `"time"` as a fallback.

### Using the custom target

```python
merra2_target = MERRA2(
    variables=["surface_air_temperature"],
    storage_options={"anon": True},
)

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[ewb.metrics.MeanAbsoluteError(
            forecast_variable="surface_air_temperature",
            target_variable="surface_air_temperature",
        )],
        target=merra2_target,
        forecast=my_forecast,
    ),
]
```

## Example 2 — Custom tabular (point observation) target

Point observation data is typically stored in parquet or CSV files with
columns for `valid_time`, `latitude`, `longitude`, and one or more
observation variables. The following example wraps a custom station
dataset stored in parquet format:

```python
import dataclasses
import pandas as pd
import polars as pl
import xarray as xr
import extremeweatherbench as ewb
from extremeweatherbench import inputs, cases, utils


@dataclasses.dataclass
class MyStationObs(inputs.TargetBase):
    """Target class for a custom parquet station observation dataset."""

    name: str = "MyStations"
    source: str = "gs://my-bucket/station_obs.parquet"

    def _open_data_from_source(self):
        return pl.scan_parquet(
            self.source,
            storage_options=self.storage_options,
        )

    def subset_data_to_case(self, data, case_metadata, **kwargs):
        bounds = case_metadata.location.as_geopandas().total_bounds
        time_min = case_metadata.start_date - pd.Timedelta(days=1)
        time_max = case_metadata.end_date + pd.Timedelta(days=1)

        return data.filter(
            (pl.col("valid_time") >= time_min)
            & (pl.col("valid_time") <= time_max)
            & (pl.col("latitude")  >= bounds[1])
            & (pl.col("latitude")  <= bounds[3])
            & (pl.col("longitude") >= bounds[0])
            & (pl.col("longitude") <= bounds[2])
        )

    def _custom_convert_to_dataset(self, data):
        # EWB calls this when it needs an xr.Dataset from the LazyFrame
        df = data.collect(engine="streaming").to_pandas()
        df["longitude"] = utils.convert_longitude_to_360(df["longitude"])
        df = df.set_index(["valid_time", "latitude", "longitude"])
        return xr.Dataset.from_dataframe(
            df[~df.index.duplicated(keep="first")], sparse=True
        )

    def maybe_align_forecast_to_target(self, forecast_data, target_data):
        return inputs.align_forecast_to_target(forecast_data, target_data)
```

> **Detailed Explanation**: EWB's internal pipeline calls
> `maybe_convert_to_dataset` on the result of `subset_data_to_case`.
> The base `TargetBase` implementation handles `xr.Dataset` and
> `xr.DataArray` natively; for any other type (polars `LazyFrame`,
> pandas `DataFrame`) you must override `_custom_convert_to_dataset` to
> produce an `xr.Dataset` with `valid_time`, `latitude`, and `longitude`
> as index dimensions. `align_forecast_to_target` then interpolates the
> forecast to the target's spatial coordinates using nearest-neighbour
> interpolation.

## Tips

- Keep `_open_data_from_source` **lazy** — return a `LazyFrame`,
  `xr.Dataset` with `chunks`, or `dask`-backed array. EWB subsets
  per-case before loading into memory.
- Use `case_metadata.location.as_geopandas().total_bounds` to obtain
  `(lon_min, lat_min, lon_max, lat_max)` bounds for spatial filtering.
- Use `case_metadata.start_date` and `case_metadata.end_date` for time
  filtering; these are `datetime.datetime` objects.
- Longitude convention: EWB uses 0–360 internally. Convert from
  −180–180 using `ewb.convert_longitude_to_360`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brightbandtech/extremeweatherbench/blob/main/notebooks/your_own_target.ipynb)

## Complete Example

A custom point-observation target wrapping EWB's public GHCNh parquet file,
following the `MyStationObs` pattern from Example 2 above.

```python
import datetime
import dataclasses
import pandas as pd
import polars as pl
import xarray as xr
import extremeweatherbench as ewb
from extremeweatherbench import inputs, utils
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2018 Japan Heat Wave — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9003,
    title="2018 Japan Heat Wave (demo)",
    start_date=datetime.datetime(2018, 7, 21),
    end_date=datetime.datetime(2018, 7, 24),
    location=BoundingBoxRegion.create_region(
        latitude_min=33.0,
        latitude_max=39.0,
        longitude_min=133.0,
        longitude_max=139.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]


@dataclasses.dataclass
class CustomGHCN(inputs.TargetBase):
    """GHCNh parquet via TargetBase, with explicit unit handling."""

    name: str = "CustomGHCN"
    source: str = ewb.DEFAULT_GHCN_URI

    def _open_data_from_source(self):
        return pl.scan_parquet(
            self.source,
            storage_options={"anon": True},
        )

    def subset_data_to_case(self, data, case_metadata, **kwargs):
        bounds = case_metadata.location.as_geopandas().total_bounds
        time_min = case_metadata.start_date - pd.Timedelta(days=1)
        time_max = case_metadata.end_date + pd.Timedelta(days=1)
        return data.filter(
            (pl.col("valid_time") >= time_min)
            & (pl.col("valid_time") <= time_max)
            & (pl.col("latitude") >= bounds[1])
            & (pl.col("latitude") <= bounds[3])
            & (pl.col("longitude") >= bounds[0])
            & (pl.col("longitude") <= bounds[2])
        )

    def _custom_convert_to_dataset(self, data):
        df = data.collect().to_pandas()
        df["surface_air_temperature"] += 273.15
        df["longitude"] = utils.convert_longitude_to_360(
            df["longitude"]
        )
        df = df.set_index(
            ["valid_time", "latitude", "longitude"]
        )
        return xr.Dataset.from_dataframe(
            df[~df.index.duplicated(keep="first")], sparse=True
        )

    def maybe_align_forecast_to_target(
        self, forecast_data, target_data
    ):
        return inputs.align_forecast_to_target(
            forecast_data, target_data
        )


custom_target = CustomGHCN()

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
        ],
        target=custom_target,
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
