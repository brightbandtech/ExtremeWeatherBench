"""ExtremeWeatherBench: A benchmarking framework for extreme weather forecasts.

This module provides the public API for ExtremeWeatherBench. Users can import
the package and access all key functionality:

    import extremeweatherbench as ewb

    # Main entry point for evaluation
    ewb.evaluation(case_metadata=..., evaluation_objects=...)

    # Hierarchical access via namespace submodules
    ewb.targets.ERA5(...)
    ewb.forecasts.ZarrForecast(...)
    ewb.metrics.MeanAbsoluteError(...)

    # Also available at top level
    ewb.ERA5(...)
    ewb.load_cases()
"""

from types import SimpleNamespace

# Import actual modules for backwards compatibility
from extremeweatherbench import calc, cases, defaults, derived, metrics, regions, utils

# Import specific items for top-level access
from extremeweatherbench.calc import (
    convert_from_cartesian_to_latlon,
    geopotential_thickness,
    great_circle_mask,
    haversine_distance,
    maybe_calculate_wind_speed,
    mixing_ratio,
    orography,
    pressure_at_surface,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    specific_humidity_from_relative_humidity,
)
from extremeweatherbench.cases import (
    CaseOperator,
    IndividualCase,
    build_case_operators,
    load_ewb_events_yaml_into_case_list,
    load_individual_cases,
    load_individual_cases_from_yaml,
    read_incoming_yaml,
)
from extremeweatherbench.defaults import (
    DEFAULT_COORDINATE_VARIABLES,
    DEFAULT_VARIABLE_NAMES,
    cira_fcnv2_atmospheric_river_forecast,
    cira_fcnv2_freeze_forecast,
    cira_fcnv2_heatwave_forecast,
    cira_fcnv2_severe_convection_forecast,
    cira_fcnv2_tropical_cyclone_forecast,
    era5_atmospheric_river_target,
    era5_freeze_target,
    era5_heatwave_target,
    get_brightband_evaluation_objects,
    get_climatology,
    ghcn_freeze_target,
    ghcn_heatwave_target,
    ibtracs_target,
    lsr_target,
    pph_target,
)
from extremeweatherbench.derived import (
    AtmosphericRiverVariables,
    CravenBrooksSignificantSevere,
    DerivedVariable,
    TropicalCycloneTrackVariables,
    maybe_derive_variables,
    maybe_include_variables_from_derived_input,
)
from extremeweatherbench.evaluate import ExtremeWeatherBench
from extremeweatherbench.inputs import (
    ARCO_ERA5_FULL_URI,
    DEFAULT_GHCN_URI,
    ERA5,
    GHCN,
    IBTRACS_URI,
    LSR,
    LSR_URI,
    PPH,
    PPH_URI,
    CIRA_metadata_variable_mapping,
    ERA5_metadata_variable_mapping,
    EvaluationObject,
    ForecastBase,
    HRES_metadata_variable_mapping,
    IBTrACS,
    IBTrACS_metadata_variable_mapping,
    InputBase,
    KerchunkForecast,
    TargetBase,
    XarrayForecast,
    ZarrForecast,
    align_forecast_to_target,
    check_for_missing_data,
    maybe_subset_variables,
    open_kerchunk_reference,
    zarr_target_subsetter,
)
from extremeweatherbench.metrics import (
    Accuracy,
    BaseMetric,
    CompositeMetric,
    CriticalSuccessIndex,
    DurationMeanError,
    EarlySignal,
    FalseAlarmRatio,
    FalseNegatives,
    FalsePositives,
    LandfallDisplacement,
    LandfallIntensityMeanAbsoluteError,
    LandfallMetric,
    LandfallTimeMeanError,
    MaximumLowestMeanAbsoluteError,
    MaximumMeanAbsoluteError,
    MeanAbsoluteError,
    MeanError,
    MeanSquaredError,
    MinimumMeanAbsoluteError,
    RootMeanSquaredError,
    SpatialDisplacement,
    ThresholdMetric,
    TrueNegatives,
    TruePositives,
)
from extremeweatherbench.regions import (
    REGION_TYPES,
    BoundingBoxRegion,
    CenteredRegion,
    Region,
    RegionSubsetter,
    ShapefileRegion,
    map_to_create_region,
    subset_cases_to_region,
    subset_results_to_region,
)
from extremeweatherbench.utils import (
    check_for_vars,
    convert_day_yearofday_to_time,
    convert_init_time_to_valid_time,
    convert_longitude_to_180,
    convert_longitude_to_360,
    convert_valid_time_to_init_time,
    derive_indices_from_init_time_and_lead_time,
    determine_temporal_resolution,
    extract_tc_names,
    filter_kwargs_for_callable,
    find_common_init_times,
    idx_to_coords,
    interp_climatology_to_target,
    is_valid_landfall,
    load_land_geometry,
    maybe_cache_and_compute,
    maybe_densify_dataarray,
    maybe_get_closest_timestamp_to_center_of_valid_times,
    maybe_get_operator,
    min_if_all_timesteps_present,
    min_if_all_timesteps_present_forecast,
    read_event_yaml,
    remove_ocean_gridpoints,
    stack_dataarray_from_dims,
)

# Aliases
evaluation = ExtremeWeatherBench
load_cases = load_ewb_events_yaml_into_case_list

# Namespace submodules for convenient grouping (these don't shadow actual modules)
targets = SimpleNamespace(
    TargetBase=TargetBase,
    ERA5=ERA5,
    GHCN=GHCN,
    IBTrACS=IBTrACS,
    LSR=LSR,
    PPH=PPH,
)

forecasts = SimpleNamespace(
    ForecastBase=ForecastBase,
    ZarrForecast=ZarrForecast,
    KerchunkForecast=KerchunkForecast,
    XarrayForecast=XarrayForecast,
)

__all__ = [
    # Core evaluation
    "evaluation",
    "ExtremeWeatherBench",
    # Modules
    "calc",
    "cases",
    "defaults",
    "derived",
    "metrics",
    "regions",
    "utils",
    # Namespace submodules
    "targets",
    "forecasts",
    # Aliases
    "load_cases",
    # calc
    "convert_from_cartesian_to_latlon",
    "geopotential_thickness",
    "great_circle_mask",
    "haversine_distance",
    "maybe_calculate_wind_speed",
    "mixing_ratio",
    "orography",
    "pressure_at_surface",
    "saturation_mixing_ratio",
    "saturation_vapor_pressure",
    "specific_humidity_from_relative_humidity",
    # cases
    "CaseOperator",
    "IndividualCase",
    "build_case_operators",
    "load_ewb_events_yaml_into_case_list",
    "load_individual_cases",
    "load_individual_cases_from_yaml",
    "read_incoming_yaml",
    # defaults
    "DEFAULT_COORDINATE_VARIABLES",
    "DEFAULT_VARIABLE_NAMES",
    "cira_fcnv2_atmospheric_river_forecast",
    "cira_fcnv2_freeze_forecast",
    "cira_fcnv2_heatwave_forecast",
    "cira_fcnv2_severe_convection_forecast",
    "cira_fcnv2_tropical_cyclone_forecast",
    "era5_atmospheric_river_target",
    "era5_freeze_target",
    "era5_heatwave_target",
    "get_brightband_evaluation_objects",
    "get_climatology",
    "ghcn_freeze_target",
    "ghcn_heatwave_target",
    "ibtracs_target",
    "lsr_target",
    "pph_target",
    # derived
    "AtmosphericRiverVariables",
    "CravenBrooksSignificantSevere",
    "DerivedVariable",
    "TropicalCycloneTrackVariables",
    "maybe_derive_variables",
    "maybe_include_variables_from_derived_input",
    # inputs
    "ARCO_ERA5_FULL_URI",
    "CIRA_metadata_variable_mapping",
    "DEFAULT_GHCN_URI",
    "ERA5",
    "ERA5_metadata_variable_mapping",
    "EvaluationObject",
    "ForecastBase",
    "GHCN",
    "HRES_metadata_variable_mapping",
    "IBTrACS",
    "IBTrACS_metadata_variable_mapping",
    "IBTRACS_URI",
    "InputBase",
    "KerchunkForecast",
    "LSR",
    "LSR_URI",
    "PPH",
    "PPH_URI",
    "TargetBase",
    "XarrayForecast",
    "ZarrForecast",
    "align_forecast_to_target",
    "check_for_missing_data",
    "maybe_subset_variables",
    "open_kerchunk_reference",
    "zarr_target_subsetter",
    # metrics
    "Accuracy",
    "BaseMetric",
    "CompositeMetric",
    "CriticalSuccessIndex",
    "DurationMeanError",
    "EarlySignal",
    "FalseAlarmRatio",
    "FalseNegatives",
    "FalsePositives",
    "LandfallDisplacement",
    "LandfallIntensityMeanAbsoluteError",
    "LandfallMetric",
    "LandfallTimeMeanError",
    "MaximumLowestMeanAbsoluteError",
    "MaximumMeanAbsoluteError",
    "MeanAbsoluteError",
    "MeanError",
    "MeanSquaredError",
    "MinimumMeanAbsoluteError",
    "RootMeanSquaredError",
    "SpatialDisplacement",
    "ThresholdMetric",
    "TrueNegatives",
    "TruePositives",
    # regions
    "BoundingBoxRegion",
    "CenteredRegion",
    "REGION_TYPES",
    "Region",
    "RegionSubsetter",
    "ShapefileRegion",
    "map_to_create_region",
    "subset_cases_to_region",
    "subset_results_to_region",
    # utils
    "check_for_vars",
    "convert_day_yearofday_to_time",
    "convert_init_time_to_valid_time",
    "convert_longitude_to_180",
    "convert_longitude_to_360",
    "convert_valid_time_to_init_time",
    "derive_indices_from_init_time_and_lead_time",
    "determine_temporal_resolution",
    "extract_tc_names",
    "filter_kwargs_for_callable",
    "find_common_init_times",
    "idx_to_coords",
    "interp_climatology_to_target",
    "is_valid_landfall",
    "load_land_geometry",
    "maybe_cache_and_compute",
    "maybe_densify_dataarray",
    "maybe_get_closest_timestamp_to_center_of_valid_times",
    "maybe_get_operator",
    "min_if_all_timesteps_present",
    "min_if_all_timesteps_present_forecast",
    "read_event_yaml",
    "remove_ocean_gridpoints",
    "stack_dataarray_from_dims",
]
