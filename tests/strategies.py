"""Hypothesis strategies for synthetic EWB forecast/target dataset pairs."""

import dataclasses
import datetime

import numpy as np
import pandas as pd
import xarray as xr
from hypothesis import strategies as st

from extremeweatherbench import cases, inputs, regions

INIT_RESOLUTIONS_HOURS = [1, 2, 3, 6, 12, 24, 48]
LEAD_RESOLUTIONS_HOURS = [1, 2, 3, 6, 12, 24]
SPATIAL_RESOLUTIONS_DEGREES = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0]

DOMAIN_KINDS = ["mid_latitude", "antimeridian", "polar", "near_global"]
LONGITUDE_CONVENTIONS = ["0-360", "-180-180"]
TARGET_TIME_DIMS = ["time", "valid_time"]

COORD_INCONSISTENCY_MODES = [
    "none",
    "drop_init_times",
    "drop_lead_times",
    "duplicate_init_time",
    "shuffled_init_time",
    "offset_outside_window",
]
MISSING_DATA_MODES = [
    "none",
    "scattered_nan",
    "all_nan_lead_slab",
    "all_nan_time_slab",
    "spatial_hole",
    "all_nan",
]
MISSING_DATA_SIDES = ["forecast", "target", "both"]

# Fixed margins (lat_min, lat_max, lon_min, lon_max) in -180/180 convention,
# generous enough to cover the grids built for each domain kind below.
DOMAIN_REGION_BOUNDS = {
    "mid_latitude": (20.0, 90.0, -130.0, -10.0),
    "antimeridian": (20.0, 90.0, 150.0, -100.0),
    "polar": (55.0, 90.0, -180.0, 180.0),
    "near_global": (-90.0, 90.0, -180.0, 180.0),
}


@dataclasses.dataclass
class InMemoryERA5(inputs.ERA5):
    """ERA5 target backed by an in-memory dataset."""

    ds: xr.Dataset | None = None
    source: str = "memory"

    def _open_data_from_source(self) -> xr.Dataset:
        return self.ds


@dataclasses.dataclass(frozen=True)
class ForecastTargetCase:
    """A synthetic forecast/target pair plus matching case metadata."""

    forecast: xr.Dataset
    target: xr.Dataset
    case: cases.IndividualCase
    init_resolution_hours: int
    lead_resolution_hours: int
    lead_dtype_is_timedelta: bool
    forecast_spatial_resolution: float
    target_spatial_resolution: float
    domain_kind: str
    forecast_longitude_convention: str
    target_longitude_convention: str
    forecast_latitude_ascending: bool
    target_latitude_ascending: bool
    target_time_dim: str
    coord_inconsistency_mode: str
    missing_data_mode: str
    missing_data_side: str
    case_overlaps: bool


def build_latitude(domain_kind: str, resolution: float, n_points: int) -> np.ndarray:
    """Build a latitude axis for the given domain kind."""
    if domain_kind == "polar":
        return np.linspace(60.0, 90.0, n_points)
    if domain_kind == "near_global":
        return np.linspace(-85.0, 85.0, n_points)
    return 20.0 + np.arange(n_points) * resolution


def build_longitude(
    domain_kind: str, resolution: float, n_points: int, convention: str
) -> np.ndarray:
    """Build a longitude axis for the given domain kind and convention.

    Coordinates are always monotonic; antimeridian-crossing grids only wrap
    around the -180/180 seam when that is the requested convention, since a
    0-360 grid represents the same crossing as a plain interior value.
    """
    if domain_kind == "antimeridian":
        raw = 170.0 + np.arange(n_points) * resolution
        if convention == "0-360":
            return raw % 360.0
        return ((raw + 180.0) % 360.0) - 180.0
    if domain_kind in ("near_global", "polar"):
        if convention == "0-360":
            return np.linspace(1.0, 359.0, n_points)
        return np.linspace(-179.0, 179.0, n_points)
    raw = -100.0 + np.arange(n_points) * resolution
    return raw % 360.0 if convention == "0-360" else raw


def build_case_location(domain_kind: str) -> regions.Region:
    """Build a Region overlapping the grids generated for domain_kind."""
    lat_min, lat_max, lon_min, lon_max = DOMAIN_REGION_BOUNDS[domain_kind]
    return regions.BoundingBoxRegion(
        latitude_min=lat_min,
        latitude_max=lat_max,
        longitude_min=lon_min,
        longitude_max=lon_max,
    )


def build_lead_time(
    n_lead: int, lead_resolution_hours: int, is_timedelta: bool
) -> np.ndarray:
    """Build a lead_time axis, either timedelta64[ns] or plain int hours."""
    hours = np.arange(n_lead) * lead_resolution_hours
    if is_timedelta:
        return hours.astype("timedelta64[h]").astype("timedelta64[ns]")
    return hours


def build_forecast_dataset(
    rng: np.random.Generator,
    init_time: pd.DatetimeIndex,
    lead_time: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> xr.Dataset:
    """Build a small synthetic forecast dataset."""
    shape = (len(init_time), len(lead_time), len(latitude), len(longitude))
    data = 273.0 + 10.0 * rng.standard_normal(shape)
    return xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "lead_time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def build_target_dataset(
    rng: np.random.Generator,
    time: pd.DatetimeIndex,
    latitude: np.ndarray,
    longitude: np.ndarray,
    time_dim: str,
) -> xr.Dataset:
    """Build a small synthetic gridded target dataset."""
    shape = (len(time), len(latitude), len(longitude))
    data = 273.0 + 10.0 * rng.standard_normal(shape)
    return xr.Dataset(
        {"2m_temperature": ([time_dim, "latitude", "longitude"], data)},
        coords={time_dim: time, "latitude": latitude, "longitude": longitude},
    )


def apply_coord_inconsistency(
    forecast: xr.Dataset, mode: str, rng: np.random.Generator
) -> xr.Dataset:
    """Apply an irregular/duplicate/shuffled init_time or lead_time pattern."""
    n_init = forecast.sizes["init_time"]
    n_lead = forecast.sizes["lead_time"]
    if mode == "drop_init_times" and n_init > 1:
        keep = np.sort(rng.choice(n_init, size=n_init - 1, replace=False))
        forecast = forecast.isel(init_time=keep)
    elif mode == "drop_lead_times" and n_lead > 1:
        keep = np.sort(rng.choice(n_lead, size=n_lead - 1, replace=False))
        forecast = forecast.isel(lead_time=keep)
    elif mode == "duplicate_init_time":
        forecast = xr.concat([forecast, forecast.isel(init_time=[0])], dim="init_time")
    elif mode == "shuffled_init_time":
        order = rng.permutation(n_init)
        forecast = forecast.isel(init_time=order)
    elif mode == "offset_outside_window":
        offset_init_time = forecast.init_time.values.copy()
        offset_init_time[-1] = offset_init_time[-1] + np.timedelta64(30, "D")
        forecast = forecast.assign_coords(init_time=offset_init_time)
    return forecast


def apply_missing_data(
    dataset: xr.Dataset, variable: str, mode: str, rng: np.random.Generator
) -> xr.Dataset:
    """Introduce NaNs into a data variable according to mode."""
    if mode == "none":
        return dataset
    data = dataset[variable].values
    dims = dataset[variable].dims
    if mode == "all_nan":
        data[...] = np.nan
    elif mode == "scattered_nan":
        flat = data.reshape(-1)
        n_missing = max(1, flat.size // 10)
        idx = rng.choice(flat.size, size=n_missing, replace=False)
        flat[idx] = np.nan
    elif mode == "spatial_hole":
        lat_dim = dims.index("latitude")
        lon_dim = dims.index("longitude")
        index = [slice(None)] * data.ndim
        index[lat_dim] = slice(0, max(1, data.shape[lat_dim] // 3))
        index[lon_dim] = slice(0, max(1, data.shape[lon_dim] // 3))
        data[tuple(index)] = np.nan
    elif mode == "all_nan_lead_slab" and "lead_time" in dims:
        index = [slice(None)] * data.ndim
        index[dims.index("lead_time")] = 0
        data[tuple(index)] = np.nan
    elif mode == "all_nan_time_slab":
        time_dim = next(
            (d for d in ("init_time", "time", "valid_time") if d in dims), None
        )
        if time_dim is not None:
            index = [slice(None)] * data.ndim
            index[dims.index(time_dim)] = 0
            data[tuple(index)] = np.nan
    return dataset


@st.composite
def forecast_target_case(draw) -> ForecastTargetCase:
    """Draw a synthetic forecast/target dataset pair plus matching case."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    init_resolution_hours = draw(st.sampled_from(INIT_RESOLUTIONS_HOURS))
    n_init = draw(st.integers(min_value=2, max_value=4))
    lead_resolution_hours = draw(st.sampled_from(LEAD_RESOLUTIONS_HOURS))
    n_lead = draw(st.integers(min_value=2, max_value=6))
    lead_dtype_is_timedelta = draw(st.sampled_from([True, True, True, False]))

    forecast_spatial_resolution = draw(st.sampled_from(SPATIAL_RESOLUTIONS_DEGREES))
    target_spatial_resolution = draw(st.sampled_from(SPATIAL_RESOLUTIONS_DEGREES))
    n_forecast_lat = draw(st.integers(min_value=3, max_value=14))
    n_forecast_lon = draw(st.integers(min_value=3, max_value=14))
    n_target_lat = draw(st.integers(min_value=3, max_value=14))
    n_target_lon = draw(st.integers(min_value=3, max_value=14))
    n_target_time = draw(st.integers(min_value=3, max_value=10))

    domain_kind = draw(st.sampled_from(DOMAIN_KINDS))
    forecast_longitude_convention = draw(st.sampled_from(LONGITUDE_CONVENTIONS))
    target_longitude_convention = draw(st.sampled_from(LONGITUDE_CONVENTIONS))
    forecast_latitude_ascending = draw(st.booleans())
    target_latitude_ascending = draw(st.booleans())
    target_time_dim = draw(st.sampled_from(TARGET_TIME_DIMS))

    coord_inconsistency_mode = draw(st.sampled_from(COORD_INCONSISTENCY_MODES))
    missing_data_mode = draw(st.sampled_from(MISSING_DATA_MODES))
    missing_data_side = draw(st.sampled_from(MISSING_DATA_SIDES))
    case_overlaps = draw(st.sampled_from([True, True, True, False]))

    init_time = pd.date_range(
        "2021-06-20", periods=n_init, freq=f"{init_resolution_hours}h"
    )
    lead_time = build_lead_time(n_lead, lead_resolution_hours, lead_dtype_is_timedelta)
    target_time = pd.date_range(
        "2021-06-20", periods=n_target_time, freq=f"{lead_resolution_hours}h"
    )

    lead_deltas = (
        lead_time if lead_dtype_is_timedelta else lead_time.astype("timedelta64[h]")
    )
    forecast_valid_times = (init_time.values[:, None] + lead_deltas[None, :]).ravel()
    all_times = np.concatenate([forecast_valid_times, target_time.values])
    nominal_start = pd.Timestamp(all_times.min())
    nominal_end = pd.Timestamp(all_times.max())

    forecast_latitude = build_latitude(
        domain_kind, forecast_spatial_resolution, n_forecast_lat
    )
    forecast_longitude = build_longitude(
        domain_kind,
        forecast_spatial_resolution,
        n_forecast_lon,
        forecast_longitude_convention,
    )
    target_latitude = build_latitude(
        domain_kind, target_spatial_resolution, n_target_lat
    )
    target_longitude = build_longitude(
        domain_kind,
        target_spatial_resolution,
        n_target_lon,
        target_longitude_convention,
    )
    if not forecast_latitude_ascending:
        forecast_latitude = forecast_latitude[::-1]
    if not target_latitude_ascending:
        target_latitude = target_latitude[::-1]

    forecast = build_forecast_dataset(
        rng, init_time, lead_time, forecast_latitude, forecast_longitude
    )
    target = build_target_dataset(
        rng, target_time, target_latitude, target_longitude, target_time_dim
    )

    forecast = apply_coord_inconsistency(forecast, coord_inconsistency_mode, rng)

    if missing_data_side in ("forecast", "both"):
        forecast = apply_missing_data(
            forecast, "surface_air_temperature", missing_data_mode, rng
        )
    if missing_data_side in ("target", "both"):
        target = apply_missing_data(target, "2m_temperature", missing_data_mode, rng)

    if case_overlaps:
        start_date = (nominal_start - pd.Timedelta(hours=1)).to_pydatetime()
        end_date = (nominal_end + pd.Timedelta(hours=1)).to_pydatetime()
    else:
        start_date = datetime.datetime(2000, 1, 1)
        end_date = datetime.datetime(2000, 1, 3)

    case = cases.IndividualCase(
        case_id_number=draw(st.integers(min_value=1, max_value=10_000)),
        title="synthetic hypothesis case",
        start_date=start_date,
        end_date=end_date,
        location=build_case_location(domain_kind),
        event_type="synthetic",
    )

    return ForecastTargetCase(
        forecast=forecast,
        target=target,
        case=case,
        init_resolution_hours=init_resolution_hours,
        lead_resolution_hours=lead_resolution_hours,
        lead_dtype_is_timedelta=lead_dtype_is_timedelta,
        forecast_spatial_resolution=forecast_spatial_resolution,
        target_spatial_resolution=target_spatial_resolution,
        domain_kind=domain_kind,
        forecast_longitude_convention=forecast_longitude_convention,
        target_longitude_convention=target_longitude_convention,
        forecast_latitude_ascending=forecast_latitude_ascending,
        target_latitude_ascending=target_latitude_ascending,
        target_time_dim=target_time_dim,
        coord_inconsistency_mode=coord_inconsistency_mode,
        missing_data_mode=missing_data_mode,
        missing_data_side=missing_data_side,
        case_overlaps=case_overlaps,
    )
