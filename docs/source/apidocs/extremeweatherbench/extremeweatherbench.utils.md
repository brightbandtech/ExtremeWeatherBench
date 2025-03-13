# {py:mod}`extremeweatherbench.utils`

```{py:module} extremeweatherbench.utils
```

```{autodoc2-docstring} extremeweatherbench.utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`convert_longitude_to_360 <extremeweatherbench.utils.convert_longitude_to_360>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.convert_longitude_to_360
    :summary:
    ```
* - {py:obj}`convert_longitude_to_180 <extremeweatherbench.utils.convert_longitude_to_180>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.convert_longitude_to_180
    :summary:
    ```
* - {py:obj}`clip_dataset_to_bounding_box_degrees <extremeweatherbench.utils.clip_dataset_to_bounding_box_degrees>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.clip_dataset_to_bounding_box_degrees
    :summary:
    ```
* - {py:obj}`convert_day_yearofday_to_time <extremeweatherbench.utils.convert_day_yearofday_to_time>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.convert_day_yearofday_to_time
    :summary:
    ```
* - {py:obj}`remove_ocean_gridpoints <extremeweatherbench.utils.remove_ocean_gridpoints>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.remove_ocean_gridpoints
    :summary:
    ```
* - {py:obj}`_open_mlwp_kerchunk_reference <extremeweatherbench.utils._open_mlwp_kerchunk_reference>`
  - ```{autodoc2-docstring} extremeweatherbench.utils._open_mlwp_kerchunk_reference
    :summary:
    ```
* - {py:obj}`map_era5_vars_to_forecast <extremeweatherbench.utils.map_era5_vars_to_forecast>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.map_era5_vars_to_forecast
    :summary:
    ```
* - {py:obj}`expand_lead_times_to_6_hourly <extremeweatherbench.utils.expand_lead_times_to_6_hourly>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.expand_lead_times_to_6_hourly
    :summary:
    ```
* - {py:obj}`process_dataarray_for_output <extremeweatherbench.utils.process_dataarray_for_output>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.process_dataarray_for_output
    :summary:
    ```
* - {py:obj}`center_forecast_on_time <extremeweatherbench.utils.center_forecast_on_time>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.center_forecast_on_time
    :summary:
    ```
* - {py:obj}`temporal_align_dataarrays <extremeweatherbench.utils.temporal_align_dataarrays>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.temporal_align_dataarrays
    :summary:
    ```
* - {py:obj}`align_observations_temporal_resolution <extremeweatherbench.utils.align_observations_temporal_resolution>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.align_observations_temporal_resolution
    :summary:
    ```
* - {py:obj}`truncate_incomplete_days <extremeweatherbench.utils.truncate_incomplete_days>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.truncate_incomplete_days
    :summary:
    ```
* - {py:obj}`return_max_min_timestamp <extremeweatherbench.utils.return_max_min_timestamp>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.return_max_min_timestamp
    :summary:
    ```
* - {py:obj}`load_events_yaml <extremeweatherbench.utils.load_events_yaml>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.load_events_yaml
    :summary:
    ```
* - {py:obj}`read_event_yaml <extremeweatherbench.utils.read_event_yaml>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.read_event_yaml
    :summary:
    ```
* - {py:obj}`location_subset_point_obs <extremeweatherbench.utils.location_subset_point_obs>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.location_subset_point_obs
    :summary:
    ```
* - {py:obj}`align_point_obs_from_gridded <extremeweatherbench.utils.align_point_obs_from_gridded>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.align_point_obs_from_gridded
    :summary:
    ```
* - {py:obj}`derive_indices_from_init_time_and_lead_time <extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <extremeweatherbench.utils.logger>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.logger
    :summary:
    ```
* - {py:obj}`Location <extremeweatherbench.utils.Location>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.Location
    :summary:
    ```
* - {py:obj}`ERA5_MAPPING <extremeweatherbench.utils.ERA5_MAPPING>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.ERA5_MAPPING
    :summary:
    ```
* - {py:obj}`ISD_MAPPING <extremeweatherbench.utils.ISD_MAPPING>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.ISD_MAPPING
    :summary:
    ```
* - {py:obj}`POINT_OBS_METADATA_VARS <extremeweatherbench.utils.POINT_OBS_METADATA_VARS>`
  - ```{autodoc2-docstring} extremeweatherbench.utils.POINT_OBS_METADATA_VARS
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: extremeweatherbench.utils.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} extremeweatherbench.utils.logger
```

````

````{py:data} Location
:canonical: extremeweatherbench.utils.Location
:value: >
   'namedtuple(...)'

```{autodoc2-docstring} extremeweatherbench.utils.Location
```

````

````{py:data} ERA5_MAPPING
:canonical: extremeweatherbench.utils.ERA5_MAPPING
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.utils.ERA5_MAPPING
```

````

````{py:data} ISD_MAPPING
:canonical: extremeweatherbench.utils.ISD_MAPPING
:value: >
   None

```{autodoc2-docstring} extremeweatherbench.utils.ISD_MAPPING
```

````

````{py:data} POINT_OBS_METADATA_VARS
:canonical: extremeweatherbench.utils.POINT_OBS_METADATA_VARS
:value: >
   ['time', 'station', 'call', 'name', 'latitude', 'longitude', 'elev', 'id']

```{autodoc2-docstring} extremeweatherbench.utils.POINT_OBS_METADATA_VARS
```

````

````{py:function} convert_longitude_to_360(longitude: float) -> float
:canonical: extremeweatherbench.utils.convert_longitude_to_360

```{autodoc2-docstring} extremeweatherbench.utils.convert_longitude_to_360
```
````

````{py:function} convert_longitude_to_180(dataset: typing.Union[xarray.Dataset, xarray.DataArray], longitude_name: str = 'longitude') -> typing.Union[xarray.Dataset, xarray.DataArray]
:canonical: extremeweatherbench.utils.convert_longitude_to_180

```{autodoc2-docstring} extremeweatherbench.utils.convert_longitude_to_180
```
````

````{py:function} clip_dataset_to_bounding_box_degrees(dataset: xarray.Dataset, location_center: extremeweatherbench.utils.Location, box_degrees: typing.Union[tuple, float]) -> xarray.Dataset
:canonical: extremeweatherbench.utils.clip_dataset_to_bounding_box_degrees

```{autodoc2-docstring} extremeweatherbench.utils.clip_dataset_to_bounding_box_degrees
```
````

````{py:function} convert_day_yearofday_to_time(dataset: xarray.Dataset, year: int) -> xarray.Dataset
:canonical: extremeweatherbench.utils.convert_day_yearofday_to_time

```{autodoc2-docstring} extremeweatherbench.utils.convert_day_yearofday_to_time
```
````

````{py:function} remove_ocean_gridpoints(dataset: xarray.Dataset) -> xarray.Dataset
:canonical: extremeweatherbench.utils.remove_ocean_gridpoints

```{autodoc2-docstring} extremeweatherbench.utils.remove_ocean_gridpoints
```
````

````{py:function} _open_mlwp_kerchunk_reference(file, forecast_schema_config, remote_protocol: str = 's3')
:canonical: extremeweatherbench.utils._open_mlwp_kerchunk_reference

```{autodoc2-docstring} extremeweatherbench.utils._open_mlwp_kerchunk_reference
```
````

````{py:function} map_era5_vars_to_forecast(forecast_schema_config, forecast_dataset, era5_dataset)
:canonical: extremeweatherbench.utils.map_era5_vars_to_forecast

```{autodoc2-docstring} extremeweatherbench.utils.map_era5_vars_to_forecast
```
````

````{py:function} expand_lead_times_to_6_hourly(dataarray: xarray.DataArray, max_fcst_hour: int = 240, fcst_output_cadence: int = 6) -> xarray.DataArray
:canonical: extremeweatherbench.utils.expand_lead_times_to_6_hourly

```{autodoc2-docstring} extremeweatherbench.utils.expand_lead_times_to_6_hourly
```
````

````{py:function} process_dataarray_for_output(da_list: typing.List[xarray.DataArray]) -> xarray.DataArray
:canonical: extremeweatherbench.utils.process_dataarray_for_output

```{autodoc2-docstring} extremeweatherbench.utils.process_dataarray_for_output
```
````

````{py:function} center_forecast_on_time(da: xarray.DataArray, time: pandas.Timestamp, hours: int)
:canonical: extremeweatherbench.utils.center_forecast_on_time

```{autodoc2-docstring} extremeweatherbench.utils.center_forecast_on_time
```
````

````{py:function} temporal_align_dataarrays(forecast: xarray.DataArray, observation: xarray.DataArray, init_time_datetime: datetime.datetime) -> tuple[xarray.DataArray, xarray.DataArray]
:canonical: extremeweatherbench.utils.temporal_align_dataarrays

```{autodoc2-docstring} extremeweatherbench.utils.temporal_align_dataarrays
```
````

````{py:function} align_observations_temporal_resolution(forecast: xarray.DataArray, observation: xarray.DataArray) -> xarray.DataArray
:canonical: extremeweatherbench.utils.align_observations_temporal_resolution

```{autodoc2-docstring} extremeweatherbench.utils.align_observations_temporal_resolution
```
````

````{py:function} truncate_incomplete_days(da: xarray.DataArray) -> xarray.DataArray
:canonical: extremeweatherbench.utils.truncate_incomplete_days

```{autodoc2-docstring} extremeweatherbench.utils.truncate_incomplete_days
```
````

````{py:function} return_max_min_timestamp(da: xarray.DataArray) -> pandas.Timestamp
:canonical: extremeweatherbench.utils.return_max_min_timestamp

```{autodoc2-docstring} extremeweatherbench.utils.return_max_min_timestamp
```
````

````{py:function} load_events_yaml()
:canonical: extremeweatherbench.utils.load_events_yaml

```{autodoc2-docstring} extremeweatherbench.utils.load_events_yaml
```
````

````{py:function} read_event_yaml(input_pth: str | pathlib.Path) -> dict
:canonical: extremeweatherbench.utils.read_event_yaml

```{autodoc2-docstring} extremeweatherbench.utils.read_event_yaml
```
````

````{py:function} location_subset_point_obs(df: pandas.DataFrame, min_lat: float, max_lat: float, min_lon: float, max_lon: float, lat_name: str = 'latitude', lon_name: str = 'longitude')
:canonical: extremeweatherbench.utils.location_subset_point_obs

```{autodoc2-docstring} extremeweatherbench.utils.location_subset_point_obs
```
````

````{py:function} align_point_obs_from_gridded(forecast_ds: xarray.Dataset, case_subset_point_obs_df: pandas.DataFrame, data_var: typing.List[str], point_obs_metadata_vars: typing.List[str]) -> typing.Tuple[xarray.Dataset, xarray.Dataset]
:canonical: extremeweatherbench.utils.align_point_obs_from_gridded

```{autodoc2-docstring} extremeweatherbench.utils.align_point_obs_from_gridded
```
````

````{py:function} derive_indices_from_init_time_and_lead_time(dataset: xarray.Dataset, start_date: datetime.datetime, end_date: datetime.datetime) -> numpy.ndarray
:canonical: extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time

```{autodoc2-docstring} extremeweatherbench.utils.derive_indices_from_init_time_and_lead_time
```
````
