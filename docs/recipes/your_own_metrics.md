# BYOM (Bring Your Own Metrics)

EWB ships continuous metrics (MAE, RMSE, MSE, bias), threshold-based
categorical metrics (CSI, FAR, accuracy), and event-specific metrics
(landfall displacement, spatial displacement, early signal). When you
need something that does not exist in that set, you can subclass
`BaseMetric` (or one of its specialised children) and drop the result
into any `EvaluationObject`. The [Usage](../usage.md) page shows how
metrics slot into the evaluation pipeline.

## The `BaseMetric` interface

Only one abstract method is required:

```python
def _compute_metric(
    self,
    forecast: xr.DataArray,
    target: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    ...
```

The method receives one-dimensional or multi-dimensional DataArrays for
the forecast and target, already subset to a single case and variable.
It must return an `xr.DataArray`.

> **Detailed Explanation**: By the time `_compute_metric` is called, the
> forecast and target have been aligned in time and space by the
> evaluation pipeline. The `preserve_dims` attribute controls which
> dimensions survive aggregation — defaults to `"lead_time"`, producing
> a result indexed by lead time. Override `preserve_dims` in
> `__init__` to keep different dimensions (e.g. `"init_time"` for
> event-level metrics).

## Example 1 — Simple continuous metric

The following implements a mean absolute percentage error (MAPE):

```python
import xarray as xr
import extremeweatherbench as ewb


class MeanAbsolutePercentageError(ewb.BaseMetric):
    """Mean Absolute Percentage Error between forecast and target."""

    def __init__(self, name: str = "MAPE", **kwargs):
        super().__init__(name=name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> xr.DataArray:
        percentage_error = (
            (forecast - target).abs() / target.where(target != 0)
        ) * 100
        return percentage_error.mean(
            dim=[d for d in percentage_error.dims
                 if d != self.preserve_dims]
        )
```

### Using the custom metric

```python
mape = MeanAbsolutePercentageError(
    forecast_variable="surface_air_temperature",
    target_variable="surface_air_temperature",
)

eval_objects = [
    ewb.EvaluationObject(
        event_type="heat_wave",
        metric_list=[mape],
        target=ewb.ERA5(variables=["surface_air_temperature"]),
        forecast=my_forecast,
    ),
]
```

## Example 2 — Threshold-based metric

To build a metric that applies a binary threshold, subclass
`ThresholdMetric`. The parent class provides
`transformed_contingency_manager`, which handles binarisation and
creates a `scores.categorical.BinaryContingencyManager` for you.

The following computes the Probability of Detection (POD), also known
as the Hit Rate:

```python
import xarray as xr
import extremeweatherbench as ewb


class ProbabilityOfDetection(ewb.ThresholdMetric):
    """Probability of Detection (Hit Rate) from binary classifications."""

    def __init__(self, name: str = "ProbabilityOfDetection", **kwargs):
        super().__init__(name=name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ):
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        tp = counts["tp_count"]
        fn = counts["fn_count"]
        return tp / (tp + fn)
```

### Using with explicit thresholds

```python
pod = ProbabilityOfDetection(
    forecast_variable="surface_air_temperature",
    target_variable="surface_air_temperature",
    forecast_threshold=308.15,  # 35 °C in Kelvin
    target_threshold=308.15,
)
```

> **Detailed Explanation**: `ThresholdMetric` accepts both
> `forecast_threshold` and `target_threshold`. These can differ — for
> example, you might binarise the forecast at one percentile and the
> target at another. The `op_func` argument controls the comparison
> operator; it defaults to `operator.ge` (≥) but accepts any callable
> or the string equivalents `">", ">=", "<", "<=", "==", "!="`.

## Example 3 — Composite metric

If you want to compute several threshold metrics in a single pass
(reusing the contingency table), pass them as a list to `ThresholdMetric`:

```python
composite = ewb.ThresholdMetric(
    name="severe_wx_contingency",
    forecast_variable="craven_brooks_significant_severe",
    target_variable="craven_brooks_significant_severe",
    forecast_threshold=20_000,
    target_threshold=20_000,
    metrics=[
        ewb.CriticalSuccessIndex,
        ewb.FalseAlarmRatio,
        ewb.Accuracy,
    ],
)
```

EWB expands composite metrics internally, computing the contingency
table once and passing it to each sub-metric.

## Init-time vs. lead-time preservation

By default, metrics preserve the `lead_time` dimension. To keep
`init_time` instead (useful for event-level or case-level summaries),
set `preserve_dims="init_time"` in the constructor:

```python
case_level_mae = ewb.metrics.MeanAbsoluteError(
    forecast_variable="surface_air_temperature",
    target_variable="surface_air_temperature",
    preserve_dims="init_time",
)
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1k-LLNZw8BCKnEM0hWak-GB3chPMcLL01/view?usp=sharing)

## Complete Example

Both custom metrics from this page combined in a single heat wave evaluation.

```python
import datetime
import xarray as xr
import extremeweatherbench as ewb
from extremeweatherbench.cases import IndividualCase
from extremeweatherbench.regions import BoundingBoxRegion

# Mini-case: 2022 Southern Plains Heat Wave — Colab-optimized
demo_case = IndividualCase(
    case_id_number=9004,
    title="2022 Southern Plains Heat Wave (demo)",
    start_date=datetime.datetime(2022, 7, 19),
    end_date=datetime.datetime(2022, 7, 22),
    location=BoundingBoxRegion.create_region(
        latitude_min=31.0,
        latitude_max=37.0,
        longitude_min=260.0,
        longitude_max=266.0,
    ),
    event_type="heat_wave",
)
cases = [demo_case]


class MeanAbsolutePercentageError(ewb.BaseMetric):
    """Mean Absolute Percentage Error."""

    def __init__(self, name: str = "MAPE", **kwargs):
        super().__init__(name=name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ) -> xr.DataArray:
        percentage_error = (
            (forecast - target).abs()
            / target.where(target != 0)
        ) * 100
        return percentage_error.mean(
            dim=[
                d
                for d in percentage_error.dims
                if d != self.preserve_dims
            ]
        )


class ProbabilityOfDetection(ewb.ThresholdMetric):
    """Probability of Detection (Hit Rate)."""

    def __init__(
        self, name: str = "ProbabilityOfDetection", **kwargs
    ):
        super().__init__(name=name, **kwargs)

    def _compute_metric(
        self,
        forecast: xr.DataArray,
        target: xr.DataArray,
        **kwargs,
    ):
        transformed = kwargs.get("transformed_manager")
        if transformed is None:
            transformed = self.transformed_contingency_manager(
                forecast=forecast,
                target=target,
                forecast_threshold=self.forecast_threshold,
                target_threshold=self.target_threshold,
                preserve_dims=self.preserve_dims,
            )
        counts = transformed.get_counts()
        tp = counts["tp_count"]
        fn = counts["fn_count"]
        return tp / (tp + fn)


mape = MeanAbsolutePercentageError(
    forecast_variable="surface_air_temperature",
    target_variable="surface_air_temperature",
)

pod = ProbabilityOfDetection(
    forecast_variable="surface_air_temperature",
    target_variable="surface_air_temperature",
    forecast_threshold=308.15,
    target_threshold=308.15,
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
        metric_list=[mape, pod],
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
