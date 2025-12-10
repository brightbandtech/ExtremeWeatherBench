# {py:mod}`extremeweatherbench.evaluate`

```{py:module} extremeweatherbench.evaluate
```

```{autodoc2-docstring} extremeweatherbench.evaluate
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CaseEvaluationInput <extremeweatherbench.evaluate.CaseEvaluationInput>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput
    :summary:
    ```
* - {py:obj}`CaseEvaluationData <extremeweatherbench.evaluate.CaseEvaluationData>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_dataset_subsets <extremeweatherbench.evaluate.build_dataset_subsets>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.build_dataset_subsets
    :summary:
    ```
* - {py:obj}`_subset_gridded_obs <extremeweatherbench.evaluate._subset_gridded_obs>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._subset_gridded_obs
    :summary:
    ```
* - {py:obj}`_subset_point_obs <extremeweatherbench.evaluate._subset_point_obs>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._subset_point_obs
    :summary:
    ```
* - {py:obj}`_check_and_subset_forecast_availability <extremeweatherbench.evaluate._check_and_subset_forecast_availability>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._check_and_subset_forecast_availability
    :summary:
    ```
* - {py:obj}`evaluate <extremeweatherbench.evaluate.evaluate>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.evaluate
    :summary:
    ```
* - {py:obj}`_maybe_evaluate_individual_cases_loop <extremeweatherbench.evaluate._maybe_evaluate_individual_cases_loop>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._maybe_evaluate_individual_cases_loop
    :summary:
    ```
* - {py:obj}`_maybe_evaluate_individual_case <extremeweatherbench.evaluate._maybe_evaluate_individual_case>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._maybe_evaluate_individual_case
    :summary:
    ```
* - {py:obj}`_open_forecast_dataset <extremeweatherbench.evaluate._open_forecast_dataset>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._open_forecast_dataset
    :summary:
    ```
* - {py:obj}`_open_obs_datasets <extremeweatherbench.evaluate._open_obs_datasets>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate._open_obs_datasets
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <extremeweatherbench.evaluate.logger>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.logger
    :summary:
    ```
* - {py:obj}`DEFAULT_FORECAST_SCHEMA_CONFIG <extremeweatherbench.evaluate.DEFAULT_FORECAST_SCHEMA_CONFIG>`
  - ```{autodoc2-docstring} extremeweatherbench.evaluate.DEFAULT_FORECAST_SCHEMA_CONFIG
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: extremeweatherbench.evaluate.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} extremeweatherbench.evaluate.logger
```

````

````{py:data} DEFAULT_FORECAST_SCHEMA_CONFIG
:canonical: extremeweatherbench.evaluate.DEFAULT_FORECAST_SCHEMA_CONFIG
:value: >
   'ForecastSchemaConfig(...)'

```{autodoc2-docstring} extremeweatherbench.evaluate.DEFAULT_FORECAST_SCHEMA_CONFIG
```

````

`````{py:class} CaseEvaluationInput
:canonical: extremeweatherbench.evaluate.CaseEvaluationInput

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput
```

````{py:attribute} observation_type
:canonical: extremeweatherbench.evaluate.CaseEvaluationInput.observation_type
:type: typing.Literal[gridded, point]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput.observation_type
```

````

````{py:attribute} observation
:canonical: extremeweatherbench.evaluate.CaseEvaluationInput.observation
:type: typing.Optional[xarray.DataArray]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput.observation
```

````

````{py:attribute} forecast
:canonical: extremeweatherbench.evaluate.CaseEvaluationInput.forecast
:type: typing.Optional[xarray.DataArray]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput.forecast
```

````

````{py:method} load_data()
:canonical: extremeweatherbench.evaluate.CaseEvaluationInput.load_data

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationInput.load_data
```

````

`````

`````{py:class} CaseEvaluationData
:canonical: extremeweatherbench.evaluate.CaseEvaluationData

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData
```

````{py:attribute} individual_case
:canonical: extremeweatherbench.evaluate.CaseEvaluationData.individual_case
:type: extremeweatherbench.case.IndividualCase
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData.individual_case
```

````

````{py:attribute} forecast
:canonical: extremeweatherbench.evaluate.CaseEvaluationData.forecast
:type: xarray.Dataset
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData.forecast
```

````

````{py:attribute} observation_type
:canonical: extremeweatherbench.evaluate.CaseEvaluationData.observation_type
:type: typing.Literal[gridded, point]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData.observation_type
```

````

````{py:attribute} observation
:canonical: extremeweatherbench.evaluate.CaseEvaluationData.observation
:type: typing.Optional[typing.Union[xarray.Dataset | pandas.DataFrame]]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.evaluate.CaseEvaluationData.observation
```

````

`````

````{py:function} build_dataset_subsets(case_evaluation_data: extremeweatherbench.evaluate.CaseEvaluationData, compute: bool = True, existing_forecast: typing.Optional[xarray.Dataset] = None) -> extremeweatherbench.evaluate.CaseEvaluationInput
:canonical: extremeweatherbench.evaluate.build_dataset_subsets

```{autodoc2-docstring} extremeweatherbench.evaluate.build_dataset_subsets
```
````

````{py:function} _subset_gridded_obs(case_evaluation_data: extremeweatherbench.evaluate.CaseEvaluationData) -> extremeweatherbench.evaluate.CaseEvaluationInput
:canonical: extremeweatherbench.evaluate._subset_gridded_obs

```{autodoc2-docstring} extremeweatherbench.evaluate._subset_gridded_obs
```
````

````{py:function} _subset_point_obs(case_evaluation_data: extremeweatherbench.evaluate.CaseEvaluationData, compute: bool = True) -> extremeweatherbench.evaluate.CaseEvaluationInput
:canonical: extremeweatherbench.evaluate._subset_point_obs

```{autodoc2-docstring} extremeweatherbench.evaluate._subset_point_obs
```
````

````{py:function} _check_and_subset_forecast_availability(case_evaluation_data: extremeweatherbench.evaluate.CaseEvaluationData) -> typing.Optional[xarray.DataArray]
:canonical: extremeweatherbench.evaluate._check_and_subset_forecast_availability

```{autodoc2-docstring} extremeweatherbench.evaluate._check_and_subset_forecast_availability
```
````

````{py:function} evaluate(eval_config: extremeweatherbench.config.Config, forecast_schema_config: extremeweatherbench.config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG, dry_run: bool = False, dry_run_event_type: typing.Optional[str] = 'HeatWave') -> dict[typing.Any, dict[typing.Any, typing.Optional[dict[str, typing.Any]]]]
:canonical: extremeweatherbench.evaluate.evaluate

```{autodoc2-docstring} extremeweatherbench.evaluate.evaluate
```
````

````{py:function} _maybe_evaluate_individual_cases_loop(event: extremeweatherbench.events.EventContainer, forecast_dataset: xarray.Dataset, gridded_obs: typing.Optional[xarray.Dataset] = None, point_obs: typing.Optional[pandas.DataFrame] = None) -> dict[typing.Any, typing.Optional[dict[str, typing.Any]]]
:canonical: extremeweatherbench.evaluate._maybe_evaluate_individual_cases_loop

```{autodoc2-docstring} extremeweatherbench.evaluate._maybe_evaluate_individual_cases_loop
```
````

````{py:function} _maybe_evaluate_individual_case(individual_case: extremeweatherbench.case.IndividualCase, forecast_dataset: typing.Optional[xarray.Dataset], gridded_obs: typing.Optional[xarray.Dataset], point_obs: typing.Optional[pandas.DataFrame]) -> typing.Optional[dict[str, xarray.Dataset]]
:canonical: extremeweatherbench.evaluate._maybe_evaluate_individual_case

```{autodoc2-docstring} extremeweatherbench.evaluate._maybe_evaluate_individual_case
```
````

````{py:function} _open_forecast_dataset(eval_config: extremeweatherbench.config.Config, forecast_schema_config: extremeweatherbench.config.ForecastSchemaConfig = DEFAULT_FORECAST_SCHEMA_CONFIG)
:canonical: extremeweatherbench.evaluate._open_forecast_dataset

```{autodoc2-docstring} extremeweatherbench.evaluate._open_forecast_dataset
```
````

````{py:function} _open_obs_datasets(eval_config: extremeweatherbench.config.Config)
:canonical: extremeweatherbench.evaluate._open_obs_datasets

```{autodoc2-docstring} extremeweatherbench.evaluate._open_obs_datasets
```
````
