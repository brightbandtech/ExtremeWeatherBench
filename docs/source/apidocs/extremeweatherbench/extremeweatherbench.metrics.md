# {py:mod}`extremeweatherbench.metrics`

```{py:module} extremeweatherbench.metrics
```

```{autodoc2-docstring} extremeweatherbench.metrics
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Metric <extremeweatherbench.metrics.Metric>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.Metric
    :summary:
    ```
* - {py:obj}`RegionalRMSE <extremeweatherbench.metrics.RegionalRMSE>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.RegionalRMSE
    :summary:
    ```
* - {py:obj}`MaximumMAE <extremeweatherbench.metrics.MaximumMAE>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.MaximumMAE
    :summary:
    ```
* - {py:obj}`MaxOfMinTempMAE <extremeweatherbench.metrics.MaxOfMinTempMAE>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.MaxOfMinTempMAE
    :summary:
    ```
* - {py:obj}`OnsetME <extremeweatherbench.metrics.OnsetME>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.OnsetME
    :summary:
    ```
* - {py:obj}`DurationME <extremeweatherbench.metrics.DurationME>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.DurationME
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <extremeweatherbench.metrics.logger>`
  - ```{autodoc2-docstring} extremeweatherbench.metrics.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: extremeweatherbench.metrics.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} extremeweatherbench.metrics.logger
```

````

`````{py:class} Metric
:canonical: extremeweatherbench.metrics.Metric

```{autodoc2-docstring} extremeweatherbench.metrics.Metric
```

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.Metric.compute
:abstractmethod:

```{autodoc2-docstring} extremeweatherbench.metrics.Metric.compute
```

````

````{py:property} name
:canonical: extremeweatherbench.metrics.Metric.name
:type: str

```{autodoc2-docstring} extremeweatherbench.metrics.Metric.name
```

````

`````

`````{py:class} RegionalRMSE
:canonical: extremeweatherbench.metrics.RegionalRMSE

Bases: {py:obj}`extremeweatherbench.metrics.Metric`

```{autodoc2-docstring} extremeweatherbench.metrics.RegionalRMSE
```

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.RegionalRMSE.compute

````

`````

`````{py:class} MaximumMAE
:canonical: extremeweatherbench.metrics.MaximumMAE

Bases: {py:obj}`extremeweatherbench.metrics.Metric`

```{autodoc2-docstring} extremeweatherbench.metrics.MaximumMAE
```

````{py:attribute} time_deviation_tolerance
:canonical: extremeweatherbench.metrics.MaximumMAE.time_deviation_tolerance
:type: int
:value: >
   48

```{autodoc2-docstring} extremeweatherbench.metrics.MaximumMAE.time_deviation_tolerance
```

````

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.MaximumMAE.compute

````

`````

`````{py:class} MaxOfMinTempMAE
:canonical: extremeweatherbench.metrics.MaxOfMinTempMAE

Bases: {py:obj}`extremeweatherbench.metrics.Metric`

```{autodoc2-docstring} extremeweatherbench.metrics.MaxOfMinTempMAE
```

````{py:attribute} time_deviation_tolerance
:canonical: extremeweatherbench.metrics.MaxOfMinTempMAE.time_deviation_tolerance
:type: int
:value: >
   48

```{autodoc2-docstring} extremeweatherbench.metrics.MaxOfMinTempMAE.time_deviation_tolerance
```

````

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.MaxOfMinTempMAE.compute

````

`````

`````{py:class} OnsetME
:canonical: extremeweatherbench.metrics.OnsetME

Bases: {py:obj}`extremeweatherbench.metrics.Metric`

```{autodoc2-docstring} extremeweatherbench.metrics.OnsetME
```

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.OnsetME.compute
:abstractmethod:

````

`````

`````{py:class} DurationME
:canonical: extremeweatherbench.metrics.DurationME

Bases: {py:obj}`extremeweatherbench.metrics.Metric`

```{autodoc2-docstring} extremeweatherbench.metrics.DurationME
```

````{py:method} compute(forecast: xarray.DataArray, observation: xarray.DataArray)
:canonical: extremeweatherbench.metrics.DurationME.compute
:abstractmethod:

````

`````
