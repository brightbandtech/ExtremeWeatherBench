# {py:mod}`extremeweatherbench.config`

```{py:module} extremeweatherbench.config
```

```{autodoc2-docstring} extremeweatherbench.config
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Config <extremeweatherbench.config.Config>`
  - ```{autodoc2-docstring} extremeweatherbench.config.Config
    :summary:
    ```
* - {py:obj}`ForecastSchemaConfig <extremeweatherbench.config.ForecastSchemaConfig>`
  - ```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DATA_DIR <extremeweatherbench.config.DATA_DIR>`
  - ```{autodoc2-docstring} extremeweatherbench.config.DATA_DIR
    :summary:
    ```
* - {py:obj}`DEFAULT_OUTPUT_DIR <extremeweatherbench.config.DEFAULT_OUTPUT_DIR>`
  - ```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_OUTPUT_DIR
    :summary:
    ```
* - {py:obj}`DEFAULT_FORECAST_DIR <extremeweatherbench.config.DEFAULT_FORECAST_DIR>`
  - ```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_FORECAST_DIR
    :summary:
    ```
* - {py:obj}`DEFAULT_CACHE_DIR <extremeweatherbench.config.DEFAULT_CACHE_DIR>`
  - ```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_CACHE_DIR
    :summary:
    ```
* - {py:obj}`ARCO_ERA5_FULL_URI <extremeweatherbench.config.ARCO_ERA5_FULL_URI>`
  - ```{autodoc2-docstring} extremeweatherbench.config.ARCO_ERA5_FULL_URI
    :summary:
    ```
* - {py:obj}`ISD_POINT_OBS_URI <extremeweatherbench.config.ISD_POINT_OBS_URI>`
  - ```{autodoc2-docstring} extremeweatherbench.config.ISD_POINT_OBS_URI
    :summary:
    ```
* - {py:obj}`POINT_OBS_STORAGE_OPTIONS <extremeweatherbench.config.POINT_OBS_STORAGE_OPTIONS>`
  - ```{autodoc2-docstring} extremeweatherbench.config.POINT_OBS_STORAGE_OPTIONS
    :summary:
    ```
* - {py:obj}`GRIDDED_OBS_STORAGE_OPTIONS <extremeweatherbench.config.GRIDDED_OBS_STORAGE_OPTIONS>`
  - ```{autodoc2-docstring} extremeweatherbench.config.GRIDDED_OBS_STORAGE_OPTIONS
    :summary:
    ```
````

### API

````{py:data} DATA_DIR
:canonical: extremeweatherbench.config.DATA_DIR
:value: >
   'Path(...)'

```{autodoc2-docstring} extremeweatherbench.config.DATA_DIR
```

````

````{py:data} DEFAULT_OUTPUT_DIR
:canonical: extremeweatherbench.config.DEFAULT_OUTPUT_DIR
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_OUTPUT_DIR
```

````

````{py:data} DEFAULT_FORECAST_DIR
:canonical: extremeweatherbench.config.DEFAULT_FORECAST_DIR
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_FORECAST_DIR
```

````

````{py:data} DEFAULT_CACHE_DIR
:canonical: extremeweatherbench.config.DEFAULT_CACHE_DIR
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.DEFAULT_CACHE_DIR
```

````

````{py:data} ARCO_ERA5_FULL_URI
:canonical: extremeweatherbench.config.ARCO_ERA5_FULL_URI
:value: >
   'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

```{autodoc2-docstring} extremeweatherbench.config.ARCO_ERA5_FULL_URI
```

````

````{py:data} ISD_POINT_OBS_URI
:canonical: extremeweatherbench.config.ISD_POINT_OBS_URI
:value: >
   'gs://extremeweatherbench/isd_minimal_qc.parquet'

```{autodoc2-docstring} extremeweatherbench.config.ISD_POINT_OBS_URI
```

````

````{py:data} POINT_OBS_STORAGE_OPTIONS
:canonical: extremeweatherbench.config.POINT_OBS_STORAGE_OPTIONS
:value: >
   'dict(...)'

```{autodoc2-docstring} extremeweatherbench.config.POINT_OBS_STORAGE_OPTIONS
```

````

````{py:data} GRIDDED_OBS_STORAGE_OPTIONS
:canonical: extremeweatherbench.config.GRIDDED_OBS_STORAGE_OPTIONS
:value: >
   'dict(...)'

```{autodoc2-docstring} extremeweatherbench.config.GRIDDED_OBS_STORAGE_OPTIONS
```

````

`````{py:class} Config
:canonical: extremeweatherbench.config.Config

```{autodoc2-docstring} extremeweatherbench.config.Config
```

````{py:attribute} event_types
:canonical: extremeweatherbench.config.Config.event_types
:type: typing.List[extremeweatherbench.events.EventContainer]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.event_types
```

````

````{py:attribute} output_dir
:canonical: extremeweatherbench.config.Config.output_dir
:type: pathlib.Path
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.output_dir
```

````

````{py:attribute} forecast_dir
:canonical: extremeweatherbench.config.Config.forecast_dir
:type: pathlib.Path
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.forecast_dir
```

````

````{py:attribute} cache_dir
:canonical: extremeweatherbench.config.Config.cache_dir
:type: typing.Optional[pathlib.Path]
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.cache_dir
```

````

````{py:attribute} gridded_obs_path
:canonical: extremeweatherbench.config.Config.gridded_obs_path
:type: str
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.gridded_obs_path
```

````

````{py:attribute} point_obs_path
:canonical: extremeweatherbench.config.Config.point_obs_path
:type: str
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.config.Config.point_obs_path
```

````

````{py:attribute} gridded_obs_storage_options
:canonical: extremeweatherbench.config.Config.gridded_obs_storage_options
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.config.Config.gridded_obs_storage_options
```

````

````{py:attribute} point_obs_storage_options
:canonical: extremeweatherbench.config.Config.point_obs_storage_options
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} extremeweatherbench.config.Config.point_obs_storage_options
```

````

`````

`````{py:class} ForecastSchemaConfig
:canonical: extremeweatherbench.config.ForecastSchemaConfig

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig
```

````{py:attribute} surface_air_temperature
:canonical: extremeweatherbench.config.ForecastSchemaConfig.surface_air_temperature
:type: typing.Optional[str]
:value: >
   't2m'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.surface_air_temperature
```

````

````{py:attribute} surface_eastward_wind
:canonical: extremeweatherbench.config.ForecastSchemaConfig.surface_eastward_wind
:type: typing.Optional[str]
:value: >
   'u10'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.surface_eastward_wind
```

````

````{py:attribute} surface_northward_wind
:canonical: extremeweatherbench.config.ForecastSchemaConfig.surface_northward_wind
:type: typing.Optional[str]
:value: >
   'v10'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.surface_northward_wind
```

````

````{py:attribute} air_temperature
:canonical: extremeweatherbench.config.ForecastSchemaConfig.air_temperature
:type: typing.Optional[str]
:value: >
   't'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.air_temperature
```

````

````{py:attribute} eastward_wind
:canonical: extremeweatherbench.config.ForecastSchemaConfig.eastward_wind
:type: typing.Optional[str]
:value: >
   'u'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.eastward_wind
```

````

````{py:attribute} northward_wind
:canonical: extremeweatherbench.config.ForecastSchemaConfig.northward_wind
:type: typing.Optional[str]
:value: >
   'v'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.northward_wind
```

````

````{py:attribute} air_pressure_at_mean_sea_level
:canonical: extremeweatherbench.config.ForecastSchemaConfig.air_pressure_at_mean_sea_level
:type: typing.Optional[str]
:value: >
   'msl'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.air_pressure_at_mean_sea_level
```

````

````{py:attribute} lead_time
:canonical: extremeweatherbench.config.ForecastSchemaConfig.lead_time
:type: typing.Optional[str]
:value: >
   'time'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.lead_time
```

````

````{py:attribute} init_time
:canonical: extremeweatherbench.config.ForecastSchemaConfig.init_time
:type: typing.Optional[str]
:value: >
   'init_time'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.init_time
```

````

````{py:attribute} fhour
:canonical: extremeweatherbench.config.ForecastSchemaConfig.fhour
:type: typing.Optional[str]
:value: >
   'fhour'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.fhour
```

````

````{py:attribute} level
:canonical: extremeweatherbench.config.ForecastSchemaConfig.level
:type: typing.Optional[str]
:value: >
   'level'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.level
```

````

````{py:attribute} latitude
:canonical: extremeweatherbench.config.ForecastSchemaConfig.latitude
:type: typing.Optional[str]
:value: >
   'latitude'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.latitude
```

````

````{py:attribute} longitude
:canonical: extremeweatherbench.config.ForecastSchemaConfig.longitude
:type: typing.Optional[str]
:value: >
   'longitude'

```{autodoc2-docstring} extremeweatherbench.config.ForecastSchemaConfig.longitude
```

````

`````
