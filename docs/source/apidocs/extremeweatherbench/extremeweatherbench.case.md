# {py:mod}`extremeweatherbench.case`

```{py:module} extremeweatherbench.case
```

```{autodoc2-docstring} extremeweatherbench.case
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IndividualCase <extremeweatherbench.case.IndividualCase>`
  - ```{autodoc2-docstring} extremeweatherbench.case.IndividualCase
    :summary:
    ```
* - {py:obj}`IndividualHeatWaveCase <extremeweatherbench.case.IndividualHeatWaveCase>`
  - ```{autodoc2-docstring} extremeweatherbench.case.IndividualHeatWaveCase
    :summary:
    ```
* - {py:obj}`IndividualFreezeCase <extremeweatherbench.case.IndividualFreezeCase>`
  - ```{autodoc2-docstring} extremeweatherbench.case.IndividualFreezeCase
    :summary:
    ```
* - {py:obj}`CaseEventType <extremeweatherbench.case.CaseEventType>`
  - ```{autodoc2-docstring} extremeweatherbench.case.CaseEventType
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_case_event_dataclass <extremeweatherbench.case.get_case_event_dataclass>`
  - ```{autodoc2-docstring} extremeweatherbench.case.get_case_event_dataclass
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <extremeweatherbench.case.logger>`
  - ```{autodoc2-docstring} extremeweatherbench.case.logger
    :summary:
    ```
* - {py:obj}`CASE_EVENT_TYPE_MATCHER <extremeweatherbench.case.CASE_EVENT_TYPE_MATCHER>`
  - ```{autodoc2-docstring} extremeweatherbench.case.CASE_EVENT_TYPE_MATCHER
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: extremeweatherbench.case.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} extremeweatherbench.case.logger
```

````

`````{py:class} IndividualCase
:canonical: extremeweatherbench.case.IndividualCase

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase
```

````{py:attribute} id
:canonical: extremeweatherbench.case.IndividualCase.id
:type: int
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.id
```

````

````{py:attribute} title
:canonical: extremeweatherbench.case.IndividualCase.title
:type: str
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.title
```

````

````{py:attribute} start_date
:canonical: extremeweatherbench.case.IndividualCase.start_date
:type: datetime.datetime
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.start_date
```

````

````{py:attribute} end_date
:canonical: extremeweatherbench.case.IndividualCase.end_date
:type: datetime.datetime
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.end_date
```

````

````{py:attribute} location
:canonical: extremeweatherbench.case.IndividualCase.location
:type: extremeweatherbench.utils.Location
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.location
```

````

````{py:attribute} bounding_box_degrees
:canonical: extremeweatherbench.case.IndividualCase.bounding_box_degrees
:type: float
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.bounding_box_degrees
```

````

````{py:attribute} event_type
:canonical: extremeweatherbench.case.IndividualCase.event_type
:type: str
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.event_type
```

````

````{py:attribute} data_vars
:canonical: extremeweatherbench.case.IndividualCase.data_vars
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.data_vars
```

````

````{py:attribute} cross_listed
:canonical: extremeweatherbench.case.IndividualCase.cross_listed
:type: typing.Optional[typing.List[str]]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.cross_listed
```

````

````{py:method} perform_subsetting_procedure(dataset: xarray.Dataset) -> xarray.Dataset
:canonical: extremeweatherbench.case.IndividualCase.perform_subsetting_procedure
:abstractmethod:

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase.perform_subsetting_procedure
```

````

````{py:method} _subset_data_vars(dataset: xarray.Dataset) -> xarray.Dataset
:canonical: extremeweatherbench.case.IndividualCase._subset_data_vars

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase._subset_data_vars
```

````

````{py:method} _subset_valid_times(dataset: xarray.Dataset) -> xarray.Dataset
:canonical: extremeweatherbench.case.IndividualCase._subset_valid_times

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase._subset_valid_times
```

````

````{py:method} _check_for_forecast_data_availability(forecast_dataset: xarray.Dataset) -> bool
:canonical: extremeweatherbench.case.IndividualCase._check_for_forecast_data_availability

```{autodoc2-docstring} extremeweatherbench.case.IndividualCase._check_for_forecast_data_availability
```

````

`````

`````{py:class} IndividualHeatWaveCase
:canonical: extremeweatherbench.case.IndividualHeatWaveCase

Bases: {py:obj}`extremeweatherbench.case.IndividualCase`

```{autodoc2-docstring} extremeweatherbench.case.IndividualHeatWaveCase
```

````{py:attribute} metrics_list
:canonical: extremeweatherbench.case.IndividualHeatWaveCase.metrics_list
:type: typing.List[typing.Type[extremeweatherbench.metrics.Metric]]
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.case.IndividualHeatWaveCase.metrics_list
```

````

````{py:attribute} data_vars
:canonical: extremeweatherbench.case.IndividualHeatWaveCase.data_vars
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.case.IndividualHeatWaveCase.data_vars
```

````

````{py:method} __post_init__()
:canonical: extremeweatherbench.case.IndividualHeatWaveCase.__post_init__

```{autodoc2-docstring} extremeweatherbench.case.IndividualHeatWaveCase.__post_init__
```

````

````{py:method} perform_subsetting_procedure(dataset: xarray.Dataset) -> xarray.Dataset
:canonical: extremeweatherbench.case.IndividualHeatWaveCase.perform_subsetting_procedure

````

`````

`````{py:class} IndividualFreezeCase
:canonical: extremeweatherbench.case.IndividualFreezeCase

Bases: {py:obj}`extremeweatherbench.case.IndividualCase`

```{autodoc2-docstring} extremeweatherbench.case.IndividualFreezeCase
```

````{py:attribute} metrics_list
:canonical: extremeweatherbench.case.IndividualFreezeCase.metrics_list
:type: typing.List[typing.Type[extremeweatherbench.metrics.Metric]]
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.case.IndividualFreezeCase.metrics_list
```

````

````{py:attribute} data_vars
:canonical: extremeweatherbench.case.IndividualFreezeCase.data_vars
:type: typing.List[str]
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.case.IndividualFreezeCase.data_vars
```

````

````{py:method} __post_init__()
:canonical: extremeweatherbench.case.IndividualFreezeCase.__post_init__

```{autodoc2-docstring} extremeweatherbench.case.IndividualFreezeCase.__post_init__
```

````

````{py:method} perform_subsetting_procedure(dataset) -> xarray.Dataset
:canonical: extremeweatherbench.case.IndividualFreezeCase.perform_subsetting_procedure

````

`````

`````{py:class} CaseEventType()
:canonical: extremeweatherbench.case.CaseEventType

Bases: {py:obj}`enum.StrEnum`

```{autodoc2-docstring} extremeweatherbench.case.CaseEventType
```

```{rubric} Initialization
```

```{autodoc2-docstring} extremeweatherbench.case.CaseEventType.__init__
```

````{py:attribute} HEAT_WAVE
:canonical: extremeweatherbench.case.CaseEventType.HEAT_WAVE
:value: >
   'heat_wave'

```{autodoc2-docstring} extremeweatherbench.case.CaseEventType.HEAT_WAVE
```

````

````{py:attribute} FREEZE
:canonical: extremeweatherbench.case.CaseEventType.FREEZE
:value: >
   'freeze'

```{autodoc2-docstring} extremeweatherbench.case.CaseEventType.FREEZE
```

````

`````

````{py:data} CASE_EVENT_TYPE_MATCHER
:canonical: extremeweatherbench.case.CASE_EVENT_TYPE_MATCHER
:type: dict[extremeweatherbench.case.CaseEventType, type[extremeweatherbench.case.IndividualCase]]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.case.CASE_EVENT_TYPE_MATCHER
```

````

````{py:function} get_case_event_dataclass(case_type: str) -> typing.Type[extremeweatherbench.case.IndividualCase]
:canonical: extremeweatherbench.case.get_case_event_dataclass

```{autodoc2-docstring} extremeweatherbench.case.get_case_event_dataclass
```
````
