from extremeweatherbench import config, inputs, metrics

era5_heatwave_target_config = config.TargetConfig(
    target=inputs.ERA5,
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

ghcn_heatwave_target_config = config.TargetConfig(
    target=inputs.GHCN,
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


cira_forecast_config = config.ForecastConfig(
    forecast=inputs.KerchunkForecast,
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)


BRIGHTBAND_METRICS = [
    config.MetricEvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target_config=era5_heatwave_target_config,
        forecast_config=cira_forecast_config,
    ),
]
