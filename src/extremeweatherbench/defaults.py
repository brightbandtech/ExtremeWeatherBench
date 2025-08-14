from extremeweatherbench import inputs, metrics

# ERA5 targets
era5_heatwave_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=["surface_air_temperature"],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

era5_freeze_target = inputs.ERA5(
    source=inputs.ARCO_ERA5_FULL_URI,
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "2m_temperature": "surface_air_temperature",
        "10m_u_component_of_wind": "surface_eastward_wind",
        "10m_v_component_of_wind": "surface_northward_wind",
        "time": "valid_time",
    },
    storage_options={"remote_options": {"anon": True}},
)

# TODO: Re-enable when atmospheric river target is implemented
# era5_atmospheric_river_target = inputs.ERA5(
#     source=inputs.ARCO_ERA5_FULL_URI,
#     variables=[
#         derived.IntegratedVaporTransport,
#         derived.AtmosphericRiverMask,
#     ],
#     variable_mapping={
#         "u_component_of_wind": "eastward_wind",
#         "v_component_of_wind": "northward_wind",
#         "temperature": "air_temperature",
#         "vertical_integral_of_northward_water_vapour_flux": "northward_water_vapour_flux",
#         "vertical_integral_of_eastward_water_vapour_flux": "eastward_water_vapour_flux",
#     },
#     storage_options={"remote_options": {"anon": True}},
# )

# GHCN targets
ghcn_heatwave_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

ghcn_freeze_target = inputs.GHCN(
    source=inputs.DEFAULT_GHCN_URI,
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "surface_temperature": "surface_air_temperature",
        "surface_eastward_wind": "surface_eastward_wind",
        "surface_northward_wind": "surface_northward_wind",
    },
    storage_options={"remote_options": {"anon": True}},
)

# LSR/PPH target
# TODO: Re-enable when severe convection is implemented
# lsr_target = inputs.LSR(
#     source=inputs.LSR_URI,
#     variables=["local_storm_reports"],
#     variable_mapping={},
#     storage_options={"remote_options": {"anon": True}},
# )

# pph_target = inputs.PPH(
#     source=inputs.PPH_URI,
#     variables=["practically_perfect_hindcast"],
#     variable_mapping={},
#     storage_options={"remote_options": {"anon": True}},
# )

# IBTrACS target

# TODO: Re-enable when IBTrACS target is implemented
# ibtracs_target = inputs.IBTrACS(
#     source=inputs.IBTRACS_URI,
#     variables=[derived.TCTrackVariables],
#     variable_mapping={
#         "vmax": "surface_wind_speed",
#         "slp": "air_pressure_at_mean_sea_level",
#     },
#     storage_options={"remote_options": {"anon": True}},
# )

# Forecast Examples

cira_heatwave_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=["surface_air_temperature"],
    variable_mapping={"t2": "surface_air_temperature"},
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

cira_freeze_forecast = inputs.KerchunkForecast(
    source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
    variables=[
        "surface_air_temperature",
        "surface_eastward_wind",
        "surface_northward_wind",
    ],
    variable_mapping={
        "t2": "surface_air_temperature",
        "10u": "surface_eastward_wind",
        "10v": "surface_northward_wind",
    },
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
)

# TODO: Re-enable when atmospheric river forecast is implemented
# cira_atmospheric_river_forecast = inputs.KerchunkForecast(
#     source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
#     variables=[
#         derived.IntegratedVaporTransport,
#         derived.AtmosphericRiverMask,
#     ],
#     variable_mapping={
#         "u_component_of_wind": "eastward_wind",
#         "v_component_of_wind": "northward_wind",
#         "specific_humidity": "specific_humidity",
#         "temperature": "air_temperature",
#         "vertical_integral_of_northward_water_vapour_flux": "northward_water_vapour_flux",
#         "vertical_integral_of_eastward_water_vapour_flux": "eastward_water_vapour_flux",
#     },
#     storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
# )

# TODO: Re-enable when CravenSignificantSevereParameter is implemented
# cira_severe_convection_forecast = inputs.KerchunkForecast(
#     source="gs://extremeweatherbench/FOUR_v200_GFS.parq",
#     variables=[derived.CravenSignificantSevereParameter],
#     variable_mapping={
#         "t": "air_temperature",
#         "t2": "surface_air_temperature",
#         "z": "geopotential",
#         "r": "relative_humidity",
#         "u": "eastward_wind",
#         "v": "northward_wind",
#         "10u": "surface_eastward_wind",
#         "10v": "surface_northward_wind",
#     },
#     storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
# )

BRIGHTBAND_EVALUATION_OBJECTS = [
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=era5_heatwave_target,
        forecast=cira_heatwave_forecast,
    ),
    inputs.EvaluationObject(
        event_type="heat_wave",
        metric=[
            metrics.MaximumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
            metrics.MaxMinMAE,
        ],
        target=ghcn_heatwave_target,
        forecast=cira_heatwave_forecast,
    ),
    inputs.EvaluationObject(
        event_type="freeze",
        metric=[
            metrics.MinimumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=era5_freeze_target,
        forecast=cira_freeze_forecast,
    ),
    inputs.EvaluationObject(
        event_type="freeze",
        metric=[
            metrics.MinimumMAE,
            metrics.RMSE,
            metrics.OnsetME,
            metrics.DurationME,
        ],
        target=ghcn_freeze_target,
        forecast=cira_freeze_forecast,
    ),
    # TODO: Re-enable when severe convection forecast is implemented
    # inputs.EvaluationObject(
    #     event_type="severe_convection",
    #     metric=[
    #         metrics.CSI,
    #         metrics.FAR,
    #         metrics.RegionalHitsMisses,
    #         metrics.HitsMisses,
    #     ],
    #     target=lsr_target,
    #     forecast=cira_severe_convection_forecast,
    # ),
    # TODO: Re-enable when atmospheric river forecast is implemented
    # inputs.EvaluationObject(
    #     event_type="atmospheric_river",
    #     metric=[metrics.CSI, metrics.SpatialDisplacement, metrics.EarlySignal],
    #     target=era5_atmospheric_river_target,
    #     forecast=cira_atmospheric_river_forecast,
    # ),
    # TODO: Re-enable when tropical cyclone forecast is implemented
    # inputs.EvaluationObject(
    #     event_type="tropical_cyclone",
    #     metric=[
    #         metrics.EarlySignal,
    #         metrics.LandfallDisplacement,
    #         metrics.LandfallTimeME,
    #         metrics.LandfallIntensityMAE,
    #     ],
    #     target=ibtracs_target,
    #     forecast=cira_tropical_cyclone_forecast,
    # ),
]
