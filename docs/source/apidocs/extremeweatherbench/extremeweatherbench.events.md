# {py:mod}`extremeweatherbench.events`

```{py:module} extremeweatherbench.events
```

```{autodoc2-docstring} extremeweatherbench.events
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EventContainer <extremeweatherbench.events.EventContainer>`
  - ```{autodoc2-docstring} extremeweatherbench.events.EventContainer
    :summary:
    ```
* - {py:obj}`HeatWave <extremeweatherbench.events.HeatWave>`
  - ```{autodoc2-docstring} extremeweatherbench.events.HeatWave
    :summary:
    ```
* - {py:obj}`Freeze <extremeweatherbench.events.Freeze>`
  - ```{autodoc2-docstring} extremeweatherbench.events.Freeze
    :summary:
    ```
````

### API

`````{py:class} EventContainer
:canonical: extremeweatherbench.events.EventContainer

```{autodoc2-docstring} extremeweatherbench.events.EventContainer
```

````{py:attribute} cases
:canonical: extremeweatherbench.events.EventContainer.cases
:type: typing.List[extremeweatherbench.case.IndividualCase]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.events.EventContainer.cases
```

````

````{py:attribute} event_type
:canonical: extremeweatherbench.events.EventContainer.event_type
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.events.EventContainer.event_type
```

````

````{py:method} subset_cases(subset) -> typing.List[extremeweatherbench.case.IndividualCase]
:canonical: extremeweatherbench.events.EventContainer.subset_cases

```{autodoc2-docstring} extremeweatherbench.events.EventContainer.subset_cases
```

````

````{py:method} __post_init__()
:canonical: extremeweatherbench.events.EventContainer.__post_init__

```{autodoc2-docstring} extremeweatherbench.events.EventContainer.__post_init__
```

````

`````

`````{py:class} HeatWave
:canonical: extremeweatherbench.events.HeatWave

Bases: {py:obj}`extremeweatherbench.events.EventContainer`

```{autodoc2-docstring} extremeweatherbench.events.HeatWave
```

````{py:attribute} event_type
:canonical: extremeweatherbench.events.HeatWave.event_type
:type: str
:value: >
   'heat_wave'

```{autodoc2-docstring} extremeweatherbench.events.HeatWave.event_type
```

````

`````

`````{py:class} Freeze
:canonical: extremeweatherbench.events.Freeze

Bases: {py:obj}`extremeweatherbench.events.EventContainer`

```{autodoc2-docstring} extremeweatherbench.events.Freeze
```

````{py:attribute} event_type
:canonical: extremeweatherbench.events.Freeze.event_type
:type: str
:value: >
   'freeze'

```{autodoc2-docstring} extremeweatherbench.events.Freeze.event_type
```

````

`````
