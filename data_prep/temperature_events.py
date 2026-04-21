#!/usr/bin/env python3
"""ERA5 temperature-event tooling: detection, expansion, and plotting.

Three subcommands, each producing csv and/or png output:

  plot    Render the consecutive-exceedance-days map for a single
          case_id_number from events.yaml.

  case    Re-expand the bounding boxes of curated heat_wave / freeze
          cases in base_temp_events.yaml and write the refreshed
          bounds back to events.yaml.

  global  Scan ERA5 2m temperature over a date range, detect events
          that stay within a climatology-quantile band for at least
          MIN_CONSECUTIVE_DAYS days over land, and write a csv plus
          a set of summary plots.

Examples:

  python temperature_events.py plot --case-id-number 30 \\
      --quantile 0.85 --operator ">="

  python temperature_events.py case \\
      --output heat_cold_yaml.csv --n-workers 4

  python temperature_events.py global \\
      --start-date 2023-06-01 --end-date 2023-09-01 \\
      --quantile-lower 0.85 --operator-lower ">=" \\
      --output heat_q85_2023.csv
"""

import argparse
import importlib
import logging
import operator as op_module
import pathlib
import time as time_module
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
import regionmask
import scipy.ndimage as ndimage
import xarray as xr
from dask.distributed import Client, LocalCluster
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ruamel.yaml import YAML
from xarray.core.indexes import IndexSelResult, PandasIndex, _query_slice
from xarray.core.indexing import _expand_slice

from extremeweatherbench import cases, defaults, inputs, regions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_DAYS = 3
MIN_GRIDPOINTS = 500
MIN_AREA_KM2 = 200000.0
EXPANSION_DEGREES = 1
MAX_SPATIAL_ITERATIONS = 20
EDGE_VALIDITY_THRESHOLD = 0.5
AREA_DECLINE_FRACTION = 0.5
TEMPORAL_LOAD_BUFFER_DAYS = 14
MAX_TEMPORAL_DAYS = 21

VALID_QUANTILES = [0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90]

_OP_MAP: dict[str, Callable] = {
    ">": op_module.gt,
    ">=": op_module.ge,
    "<": op_module.lt,
    "<=": op_module.le,
    "==": op_module.eq,
}

_HW_COLOR = "#d73027"
_FZ_COLOR = "#4575b4"
_BOX_ALPHA = 0.35
_BOX_EDGE_ALPHA = 0.85

EdgeName = Literal["north", "south", "east", "west"]


class PeriodicBoundaryIndex(PandasIndex):
    """xarray index for a 1-D coordinate that wraps at a period.

    Subclasses PandasIndex and intercepts slice queries so a selection
    that crosses the period boundary (e.g. longitude 350 to 10) is
    returned as two concatenated index arrays rather than an empty
    slice.
    """

    period: float
    _min: float
    _max: float

    __slots__ = ("index", "dim", "coord_dtype", "period", "_max", "_min")

    def __init__(self, *args, period=360, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self._min = self.index.min()
        self._max = self.index.max()

    @classmethod
    def from_variables(self, variables, options):
        obj = super().from_variables(variables, options={})
        obj.period = options.get("period", obj.period)
        return obj

    def _wrap_periodically(self, label_value: float) -> float:
        return self._min + (label_value - self._max) % self.period

    def _split_slice_across_boundary(self, label: slice) -> np.ndarray:
        """Return concatenated integer indices for a slice that wraps."""
        first_slice = slice(label.start, self._max, label.step)
        second_slice = slice(self._min, label.stop, label.step)

        first_as_index_slice = _query_slice(self.index, first_slice)
        second_as_index_slice = _query_slice(self.index, second_slice)

        first_as_indices = _expand_slice(first_as_index_slice, self.index.size)
        second_as_indices = _expand_slice(second_as_index_slice, self.index.size)

        return np.concatenate([first_as_indices, second_as_indices])

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        """Remap out-of-range labels back into the index range."""
        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            start, stop, step = label.start, label.stop, label.step
            if stop < start:
                return super().sel({coord_name: []})

            assert self._min < self._max

            wrapped_start = self._wrap_periodically(label.start)
            wrapped_stop = self._wrap_periodically(label.stop)
            wrapped_label = slice(wrapped_start, wrapped_stop, step)

            if wrapped_start < wrapped_stop:
                return super().sel({coord_name: wrapped_label})
            # Slice crosses the wrap boundary; split in two.
            wrapped_indices = self._split_slice_across_boundary(wrapped_label)
            return IndexSelResult({self.dim: wrapped_indices})

        wrapped_label = self._wrap_periodically(label)  # type: ignore
        return super().sel(
            {coord_name: wrapped_label}, method=method, tolerance=tolerance
        )

    def __repr__(self) -> str:
        return f"PeriodicBoundaryIndex(period={self.period})"


def resolve_op(op_str: str) -> Callable:
    """Return the callable for a comparison-operator string.

    Args:
        op_str: One of ``>``, ``>=``, ``<``, ``<=``, ``==``.

    Raises:
        ValueError: If ``op_str`` is not recognised.
    """
    if op_str not in _OP_MAP:
        raise ValueError(f"Unknown operator {op_str!r}; choose from {list(_OP_MAP)}")
    return _OP_MAP[op_str]


def detect_time_dim(obj: xr.Dataset | xr.DataArray) -> str:
    """Return the name of the time dimension (``valid_time`` or ``time``)."""
    for name in ("valid_time", "time"):
        if name in obj.dims:
            return name
    raise ValueError(f"No time dimension found. Available: {list(obj.dims)}")


def _to_plot_lon(lon: float) -> float:
    """Wrap 0-360 longitude to -180..180 for PlateCarree plotting."""
    return lon - 360.0 if lon > 180.0 else lon


def max_consecutive_days(mask_3d: np.ndarray) -> np.ndarray:
    """Return the per-grid-point max run length of True along axis 0.

    Args:
        mask_3d: Boolean array of shape (time, lat, lon).

    Returns:
        Int32 array of shape (lat, lon) holding the longest run of
        consecutive True values along the time axis for each cell.
    """
    nt, nlat, nlon = mask_3d.shape
    flat = mask_3d.reshape(nt, -1).astype(np.int8)
    result = np.zeros(flat.shape[1], dtype=np.int32)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        result[k] = int((ends - starts).max())
    return result.reshape(nlat, nlon)


def _consecutive_filter_np(
    mask: np.ndarray,
    min_days: int,
    max_grace_days: int,
) -> np.ndarray:
    struct = np.zeros((min_days, 1, 1), dtype=bool)
    struct[:, 0, 0] = True

    # Strict min_days-long runs with no grace applied.
    strict = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                mask,
                structure=struct,
                border_value=False,
            ),
            structure=struct,
            border_value=False,
        )
        & mask
    )

    if max_grace_days <= 0:
        return strict

    # Bridge gaps of up to max_grace_days days.
    close_k = np.zeros((2 * max_grace_days + 1, 1, 1), dtype=bool)
    close_k[:, 0, 0] = True
    filled = ndimage.binary_closing(
        mask,
        structure=close_k,
        border_value=False,
    )

    filled_runs = (
        ndimage.binary_dilation(
            ndimage.binary_erosion(
                filled,
                structure=struct,
                border_value=False,
            ),
            structure=struct,
            border_value=False,
        )
        & filled
    )

    # Label each temporal run independently per grid point.
    lbl_struct = np.zeros((3, 3, 3), dtype=int)
    lbl_struct[0, 1, 1] = 1
    lbl_struct[1, 1, 1] = 1
    lbl_struct[2, 1, 1] = 1
    labels, _ = ndimage.label(filled_runs, structure=lbl_struct)

    # Keep only runs that contain a strict min_days-day block.
    valid = np.unique(labels[strict & (labels > 0)])
    return np.isin(labels, valid) & filled_runs


def apply_consecutive_filter(
    mask: xr.DataArray,
    min_days: int = MIN_CONSECUTIVE_DAYS,
    max_grace_days: int = 1,
) -> xr.DataArray:
    """Keep temporal runs of ``min_days``+ True days along axis 0.

    Once a strict run of ``min_days`` True days exists at a grid
    point, gaps of up to ``max_grace_days`` are bridged so a single
    short-lived break does not split the event. Runs that never reach
    ``min_days`` strict consecutive days are discarded.

    Args:
        mask: Boolean DataArray of shape (time, lat, lon). Any
            xindexes (e.g. ``PeriodicBoundaryIndex`` on longitude) are
            preserved on the returned DataArray.
        min_days: Minimum strict run length. Default 3.
        max_grace_days: Maximum gap length to bridge. Default 1.

    Returns:
        Boolean DataArray of the same shape and indexes with only
        qualifying runs retained.
    """
    out_np = _consecutive_filter_np(mask.values.astype(bool), min_days, max_grace_days)
    return mask.copy(data=out_np)


def open_era5_t2m(start_date: str, end_date: str) -> xr.DataArray:
    """Open ERA5 2m temperature lazily for an inclusive date range.

    Restricts to 6-hourly timesteps (0/6/12/18 UTC) to match the
    climatology base and sorts latitude to ascending order.

    Args:
        start_date: Inclusive start date string (YYYY-MM-DD).
        end_date: Inclusive end date string (YYYY-MM-DD).
    """
    ds = xr.open_zarr(
        inputs.ARCO_ERA5_FULL_URI,
        storage_options={"token": "anon"},
        chunks=None,
    )
    tdim = detect_time_dim(ds)
    t2m = ds["2m_temperature"].sel({tdim: slice(start_date, end_date)})
    six_hourly = t2m[tdim].dt.hour.isin([0, 6, 12, 18])
    t2m = t2m.sel({tdim: six_hourly})
    t2m = t2m.chunk({"latitude": -1, "longitude": -1})
    return t2m.sortby("latitude")


def _align_climatology(
    clim: xr.DataArray,
    daily: xr.DataArray,
    tdim: str,
) -> xr.DataArray:
    """Align a dayofyear-indexed climatology to a daily DataArray.

    Computes the climatology into memory (366 x lat x lon) to avoid
    dask chunk multiplication, then indexes by day-of-year via numpy.

    Args:
        clim: Climatology DataArray indexed by ``dayofyear``.
        daily: Daily DataArray to align against.
        tdim: Name of the time dimension in ``daily``.
    """
    clim_vals = clim.compute().values
    clim_doy = clim.dayofyear.values
    doy = daily[tdim].dt.dayofyear.values
    max_doy = int(clim_doy.max())
    doy_idx = np.clip(doy, 1, max_doy) - int(clim_doy.min())

    clim_lats = clim.latitude.values
    clim_lons = clim.longitude.values
    daily_lats = daily.latitude.values
    daily_lons = daily.longitude.values

    if (
        clim_lats.shape == daily_lats.shape
        and np.allclose(clim_lats, daily_lats)
        and clim_lons.shape == daily_lons.shape
        and np.allclose(clim_lons, daily_lons)
    ):
        selected = clim_vals[doy_idx]
    else:
        lat_idx = np.array([np.argmin(np.abs(clim_lats - dl)) for dl in daily_lats])
        lon_idx = np.array([np.argmin(np.abs(clim_lons - dl)) for dl in daily_lons])
        selected = clim_vals[np.ix_(doy_idx, lat_idx, lon_idx)]

    return xr.DataArray(selected, dims=daily.dims, coords=daily.coords)


def get_climatology_bounds(
    q_lower: Optional[float] = None,
    q_upper: Optional[float] = None,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Return climatology DataArrays for the lower and/or upper bound.

    Args:
        q_lower: Quantile for the lower bound, or None.
        q_upper: Quantile for the upper bound, or None.

    Raises:
        ValueError: If both ``q_lower`` and ``q_upper`` are None.
    """
    if q_lower is None and q_upper is None:
        raise ValueError("At least one of q_lower or q_upper must be set.")
    clim_lower = (
        defaults.get_climatology(q_lower).sortby("latitude")
        if q_lower is not None
        else None
    )
    clim_upper = (
        defaults.get_climatology(q_upper).sortby("latitude")
        if q_upper is not None
        else None
    )
    return clim_lower, clim_upper


def build_land_mask(
    lons: xr.DataArray,
    lats: xr.DataArray,
) -> xr.DataArray:
    """Return a boolean DataArray (True = land) for the given grid."""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lons, lats)
    return mask == 0


def compute_grid_cell_area(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return a (lat, lon) array of grid-cell areas in km^2.

    Uses the spherical-Earth approximation
    ``area = R^2 * dlat_rad * dlon_rad * cos(lat)``.

    Args:
        lats: 1-D latitude array in degrees.
        lons: 1-D longitude array in degrees; only its length is used.
    """
    R_KM = 6371.0
    dlat = float(np.abs(np.diff(lats[:2]))[0]) if len(lats) > 1 else 0.25
    dlon = float(np.abs(np.diff(lons[:2]))[0]) if len(lons) > 1 else 0.25
    cell_km2 = R_KM**2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lats))
    return np.outer(cell_km2, np.ones(len(lons)))


def build_exceedance_mask(
    t2m: xr.DataArray,
    clim_lower: Optional[xr.DataArray],
    clim_upper: Optional[xr.DataArray],
    land_mask: xr.DataArray,
    op_lower: str = ">",
    op_upper: str = "<",
) -> xr.DataArray:
    """Build a daily land-only mask of passed-bound steps.

    A day passes only when every 6-hourly step satisfies every
    provided bound. The result is then masked to land.

    Args:
        t2m: 6-hourly 2m temperature DataArray.
        clim_lower: Climatology for the lower bound, or None.
        clim_upper: Climatology for the upper bound, or None.
        land_mask: Boolean DataArray (True = land) on t2m's grid.
        op_lower: Operator for the lower bound. Default ``>``.
        op_upper: Operator for the upper bound. Default ``<``.
    """
    tdim = detect_time_dim(t2m)
    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    ref = clim_lower if clim_lower is not None else clim_upper
    assert ref is not None
    max_clim_doy = int(ref.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    mask_6h = xr.ones_like(t2m, dtype=bool)

    if clim_lower is not None:
        cmp = resolve_op(op_lower)
        aligned = clim_lower.sel(dayofyear=doy_capped, hour=hour).reindex_like(
            t2m, method="nearest"
        )
        mask_6h = mask_6h & cmp(t2m, aligned)

    if clim_upper is not None:
        cmp = resolve_op(op_upper)
        aligned = clim_upper.sel(dayofyear=doy_capped, hour=hour).reindex_like(
            t2m, method="nearest"
        )
        mask_6h = mask_6h & cmp(t2m, aligned)

    return mask_6h.resample({tdim: "1D"}).min().astype(bool) & land_mask


def _edge_valid_fraction(
    mask_2d: np.ndarray,
    land_2d: np.ndarray,
    edge: str,
    band_pts: int,
) -> float:
    """Return the fraction of land grid points on an edge that are active.

    Args:
        mask_2d: 2-D boolean activity array (lat, lon).
        land_2d: 2-D boolean land mask (lat, lon); True = land.
        edge: One of ``north``, ``south``, ``east``, ``west``.
        band_pts: Width of the edge strip in grid points.

    Returns:
        Fraction in [0, 1] of land points in the strip that are
        active, or 0.0 if the strip contains no land points.

    Raises:
        ValueError: If ``edge`` is not one of the recognised values.
    """
    if edge == "north":
        strip = mask_2d[-band_pts:, :]
        land_strip = land_2d[-band_pts:, :]
    elif edge == "south":
        strip = mask_2d[:band_pts, :]
        land_strip = land_2d[:band_pts, :]
    elif edge == "west":
        strip = mask_2d[:, :band_pts]
        land_strip = land_2d[:, :band_pts]
    elif edge == "east":
        strip = mask_2d[:, -band_pts:]
        land_strip = land_2d[:, -band_pts:]
    else:
        raise ValueError(f"Unknown edge: {edge}")
    n_land = int(land_strip.sum())
    if n_land == 0:
        return 0.0
    return float((strip & land_strip).sum()) / n_land


def expand_event_bounds(
    event: Dict,
    filt_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    land_mask_np: np.ndarray,
) -> Dict:
    """Expand a detected event's bounding box using case-script logic.

    Starting from the blob-tracked bounding box, expands each edge
    outward while >= EDGE_VALIDITY_THRESHOLD of land points on that
    edge are active on the peak-footprint day. Also computes and stores
    max_consecutive_days and mean_consecutive_days within the expanded
    region.

    Args:
        event: Event dict from detect_events.
        filt_mask: Boolean array (time, lat, lon) with consecutive-day
            filtering already applied.
        dates: 1-D datetime64 array of daily timestamps (axis 0).
        lats: 1-D latitude array (axis 1).
        lons: 1-D longitude array (axis 2).
        land_mask_np: 2-D boolean array (lat, lon); True = land.

    Returns:
        The same event dict with updated lat_min/lat_max/lon_min/
        lon_max, max_consecutive_days, and mean_consecutive_days.
    """
    grid_res = float(np.abs(np.diff(lats[:2]))[0]) if len(lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    start_date = np.datetime64(event["start"])
    end_date = np.datetime64(event["end"])
    t_mask = (dates >= start_date) & (dates <= end_date)
    ev_filt = filt_mask[t_mask]

    if ev_filt.shape[0] == 0:
        event["max_consecutive_days"] = 0
        event["mean_consecutive_days"] = 0.0
        return event

    def _lat_idx(val: float) -> int:
        return int(np.argmin(np.abs(lats - val)))

    def _lon_idx(val: float) -> int:
        return int(np.argmin(np.abs(lons - val)))

    idx_s0 = _lat_idx(event["lat_min"])
    idx_n0 = _lat_idx(event["lat_max"])
    idx_w0 = _lon_idx(event["lon_min"])
    idx_e0 = _lon_idx(event["lon_max"])

    daily_counts = ev_filt.sum(axis=(1, 2))
    max_count = daily_counts.max()
    tied_days = np.where(daily_counts == max_count)[0]
    n_tied = len(tied_days)
    if n_tied <= 2:
        peak_day = int(tied_days[-1]) if n_tied == 2 else int(tied_days[0])
    else:
        peak_day = int(tied_days[(n_tied - 1) // 2])
    peak_mask = ev_filt[peak_day]

    idx_s, idx_n, idx_w, idx_e = idx_s0, idx_n0, idx_w0, idx_e0

    init_region_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    edges_active: Dict[str, bool] = {}
    for edge in ("north", "south", "east", "west"):
        if edge == "north":
            strip = init_region_land[-band_pts:, :]
        elif edge == "south":
            strip = init_region_land[:band_pts, :]
        elif edge == "west":
            strip = init_region_land[:, :band_pts]
        else:
            strip = init_region_land[:, -band_pts:]
        land_frac = strip.sum() / max(strip.size, 1)
        edges_active[edge] = bool(land_frac >= 0.25)

    for _ in range(MAX_SPATIAL_ITERATIONS):
        region = peak_mask[idx_s : idx_n + 1, idx_w : idx_e + 1]
        land_region = land_mask_np[idx_s : idx_n + 1, idx_w : idx_e + 1]

        all_done = True
        for edge in list(edges_active.keys()):
            if not edges_active[edge]:
                continue
            frac = _edge_valid_fraction(region, land_region, edge, band_pts)
            if frac < EDGE_VALIDITY_THRESHOLD:
                edges_active[edge] = False
            else:
                all_done = False
                if edge == "north":
                    idx_n = min(len(lats) - 1, idx_n + band_pts)
                elif edge == "south":
                    idx_s = max(0, idx_s - band_pts)
                elif edge == "east":
                    idx_e = min(len(lons) - 1, idx_e + band_pts)
                elif edge == "west":
                    idx_w = max(0, idx_w - band_pts)

        if all_done:
            break

    fin_filt = ev_filt[:, idx_s : idx_n + 1, idx_w : idx_e + 1]
    consec = max_consecutive_days(fin_filt)
    event["lat_min"] = float(lats[idx_s])
    event["lat_max"] = float(lats[idx_n])
    event["lon_min"] = float(lons[idx_w])
    event["lon_max"] = float(lons[idx_e])
    event["max_consecutive_days"] = int(consec.max()) if consec.size > 0 else 0
    active = consec[consec >= MIN_CONSECUTIVE_DAYS]
    event["mean_consecutive_days"] = (
        round(float(active.mean()), 2) if active.size > 0 else 0.0
    )
    return event


def process_event(
    single_case: cases.IndividualCase,
    out_dir: pathlib.Path = pathlib.Path("."),
    quantile: float | None = None,
    op_str: str | None = None,
) -> Optional[Dict]:
    """Re-expand one curated case's time window and spatial bounds.

    Grows the time window forward from ``start_date - 3`` one day at
    a time until the single-day exceedance fraction drops below 50%
    of its peak, then grows each spatial edge by
    ``EXPANSION_DEGREES`` per iteration while the edge-validity
    threshold is met.

    Args:
        single_case: Case with a ``CenteredRegion`` location.
        out_dir: Directory to write PNG plots into.
        quantile: Climatology quantile. Defaults to 0.85 for
            heat_wave or 0.15 for freeze when not provided.
        op_str: Comparison operator (``>``, ``>=``, ``<``, ``<=``).
            Defaults to ``>`` for heat_wave or ``<`` for freeze.

    Returns:
        A result dict (see call sites) or None if the case was
        skipped (wrong location type, empty grid, or no land).
    """
    is_heatwave = single_case.event_type == "heat_wave"
    logger.info(
        "Processing case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    loc = single_case.location
    if not isinstance(loc, regions.CenteredRegion):
        logger.warning(
            "  Case %d location is %s, not CenteredRegion — skipping",
            single_case.case_id_number,
            type(loc).__name__,
        )
        return None

    center_lat = loc.latitude
    center_lon = loc.longitude
    if isinstance(loc.bounding_box_degrees, tuple):
        half_lat = loc.bounding_box_degrees[0] / 2.0
        half_lon = loc.bounding_box_degrees[1] / 2.0
    else:
        half_lat = loc.bounding_box_degrees / 2.0
        half_lon = loc.bounding_box_degrees / 2.0

    box_lat_min = center_lat - half_lat
    box_lat_max = center_lat + half_lat
    box_lon_min = center_lon - half_lon
    box_lon_max = center_lon + half_lon

    logger.info(
        "  Initial box (%.1f°): center (%.2f, %.2f)"
        "  lat [%.2f, %.2f], lon [%.2f, %.2f]",
        loc.bounding_box_degrees
        if not isinstance(loc.bounding_box_degrees, tuple)
        else loc.bounding_box_degrees[0],
        center_lat,
        center_lon,
        box_lat_min,
        box_lat_max,
        box_lon_min,
        box_lon_max,
    )

    start_date = pd.Timestamp(single_case.start_date) - pd.Timedelta(days=3)
    end_date = pd.Timestamp(single_case.end_date) + pd.Timedelta(
        days=TEMPORAL_LOAD_BUFFER_DAYS
    )

    pot_lat_min = max(
        -90,
        box_lat_min - MAX_SPATIAL_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lat_max = min(
        90,
        box_lat_max + MAX_SPATIAL_ITERATIONS * EXPANSION_DEGREES,
    )
    pot_lon_min = box_lon_min - MAX_SPATIAL_ITERATIONS * EXPANSION_DEGREES
    pot_lon_max = box_lon_max + MAX_SPATIAL_ITERATIONS * EXPANSION_DEGREES

    ds = xr.open_zarr(
        inputs.ARCO_ERA5_FULL_URI,
        storage_options={"token": "anon"},
        chunks={},
    )
    tdim = detect_time_dim(ds)
    t2m = ds["2m_temperature"].sel(
        {tdim: slice(str(start_date), str(end_date))},
    )
    six_hourly = t2m[tdim].dt.hour.isin([0, 6, 12, 18])
    t2m = t2m.sel({tdim: six_hourly}).sortby("latitude")

    if float(t2m.longitude.max()) > 180:
        t2m = t2m.assign_coords(
            longitude=(t2m.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    pot_lon_min_era = pot_lon_min
    pot_lon_max_era = pot_lon_max
    if pot_lon_min_era > 180:
        pot_lon_min_era -= 360
    if pot_lon_max_era > 180:
        pot_lon_max_era -= 360

    t2m = t2m.sel(
        latitude=slice(pot_lat_min, pot_lat_max),
        longitude=slice(
            min(pot_lon_min_era, pot_lon_max_era),
            max(pot_lon_min_era, pot_lon_max_era),
        ),
    )

    if t2m.latitude.size == 0 or t2m.longitude.size == 0:
        logger.warning(
            "  Case %d: empty spatial selection — skipping",
            single_case.case_id_number,
        )
        return None

    if quantile is None:
        quantile = 0.85 if is_heatwave else 0.15
    if op_str is None:
        op_str = ">" if is_heatwave else "<"
    cmp = resolve_op(op_str)

    clim = defaults.get_climatology(quantile).sortby("latitude")

    if float(clim.longitude.max()) > 180:
        clim = clim.assign_coords(
            longitude=(clim.longitude.values + 180) % 360 - 180,
        ).sortby("longitude")

    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    max_clim_doy = int(clim.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    clim_aligned = clim.sel(
        dayofyear=doy_capped,
        hour=hour,
    ).reindex_like(t2m, method="nearest")

    exc_6h = cmp(t2m, clim_aligned)
    daily_all_pass = exc_6h.resample({tdim: "1D"}).min().astype(bool)

    land_reg = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = (
        land_reg.mask(
            daily_all_pass.longitude,
            daily_all_pass.latitude,
        )
        == 0
    )

    exc_mask = daily_all_pass & land_mask

    logger.info("  Computing exceedance mask...")
    mask_np = exc_mask.compute().values.astype(bool)

    all_lats = daily_all_pass.latitude.values
    all_lons = daily_all_pass.longitude.values
    grid_res = np.abs(np.diff(all_lats[:2]))[0] if len(all_lats) > 1 else 0.25
    band_pts = max(1, int(round(EXPANSION_DEGREES / grid_res)))

    def _lat_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lats - val)))

    def _lon_idx(val: float) -> int:
        return int(np.argmin(np.abs(all_lons - val)))

    box_lon_min_era = _to_plot_lon(box_lon_min)
    box_lon_max_era = _to_plot_lon(box_lon_max)

    idx_s0 = _lat_idx(box_lat_min)
    idx_n0 = _lat_idx(box_lat_max)
    idx_w0 = _lon_idx(box_lon_min_era)
    idx_e0 = _lon_idx(box_lon_max_era)

    land_mask_np = land_mask.compute().values.astype(bool)
    box_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    n_land_pts = int(box_land.sum())
    if n_land_pts == 0:
        logger.warning(
            "  Case %d: no land points in initial box — skipping",
            single_case.case_id_number,
        )
        return None

    n_days_total = min(mask_np.shape[0], MAX_TEMPORAL_DAYS)
    final_t = n_days_total
    established = False

    for t in range(n_days_total):
        if t < 2:
            continue

        # "Currently active" = last 3 days all exceed threshold.
        box_sl = (
            slice(idx_s0, idx_n0 + 1),
            slice(idx_w0, idx_e0 + 1),
        )
        currently_active = (
            mask_np[t][box_sl] & mask_np[t - 1][box_sl] & mask_np[t - 2][box_sl]
        )
        active_land = int((currently_active & box_land).sum())
        frac = active_land / n_land_pts

        if t == 2 and frac < EDGE_VALIDITY_THRESHOLD:
            logger.warning(
                "  Day 3: only %.1f%% of land points have 3 consecutive days (< 50%%)",
                frac * 100,
            )

        if not established:
            if frac >= EDGE_VALIDITY_THRESHOLD:
                established = True
                logger.info(
                    "  Event established at day %d (%.1f%% of land points active)",
                    t,
                    frac * 100,
                )
        else:
            if frac < EDGE_VALIDITY_THRESHOLD:
                final_t = t + 1
                logger.info(
                    "  Temporal stop at day %d (%.1f%% < 50%% of land points)",
                    t,
                    frac * 100,
                )
                break

        if t == 9:
            logger.warning(
                "  Exceeded 10 days (frac %.1f%%), continuing...",
                frac * 100,
            )

    if not established:
        logger.warning(
            "  Case %d: event never reached 50%% of land points — using all %d days",
            single_case.case_id_number,
            final_t,
        )

    logger.info(
        "  Using %d of %d available days",
        final_t,
        mask_np.shape[0],
    )

    mask_np = mask_np[:final_t]
    filtered = _consecutive_filter_np(mask_np, MIN_CONSECUTIVE_DAYS, 1)

    # Spatial expansion starts from the peak-footprint day. For tied
    # counts: 2nd of 2, middle of odd count, or first-middle of even.
    daily_counts = filtered.sum(axis=(1, 2))
    max_count = daily_counts.max()
    (tied_days,) = np.where(daily_counts == max_count)
    n_tied = len(tied_days)
    if n_tied <= 2:
        peak_day = int(tied_days[-1]) if n_tied == 2 else int(tied_days[0])
    else:
        peak_day = int(tied_days[(n_tied - 1) // 2])
    peak_mask = filtered[peak_day]
    logger.info(
        "  Peak footprint on day %d (%d active grid points, %d tied days)",
        peak_day,
        int(max_count),
        n_tied,
    )

    idx_s = idx_s0
    idx_n = idx_n0
    idx_w = idx_w0
    idx_e = idx_e0

    init_region_land = land_mask_np[idx_s0 : idx_n0 + 1, idx_w0 : idx_e0 + 1]
    edges_active = {}
    for edge in ("north", "south", "east", "west"):
        if edge == "north":
            strip = init_region_land[-band_pts:, :]
        elif edge == "south":
            strip = init_region_land[:band_pts, :]
        elif edge == "west":
            strip = init_region_land[:, :band_pts]
        else:
            strip = init_region_land[:, -band_pts:]
        land_frac = strip.sum() / max(strip.size, 1)
        edges_active[edge] = bool(land_frac >= 0.25)
        if not edges_active[edge]:
            logger.info(
                "  %s edge disabled (%.1f%% land in initial box)",
                edge.capitalize(),
                land_frac * 100,
            )
    n_iter = 0

    for iteration in range(MAX_SPATIAL_ITERATIONS):
        n_iter = iteration + 1
        region = peak_mask[idx_s : idx_n + 1, idx_w : idx_e + 1]
        land_region = land_mask_np[idx_s : idx_n + 1, idx_w : idx_e + 1]

        all_done = True
        for edge in list(edges_active.keys()):
            if not edges_active[edge]:
                continue

            frac = _edge_valid_fraction(region, land_region, edge, band_pts)
            if frac < EDGE_VALIDITY_THRESHOLD:
                edges_active[edge] = False
            else:
                all_done = False
                if edge == "north":
                    idx_n = min(
                        len(all_lats) - 1,
                        idx_n + band_pts,
                    )
                elif edge == "south":
                    idx_s = max(0, idx_s - band_pts)
                elif edge == "east":
                    idx_e = min(
                        len(all_lons) - 1,
                        idx_e + band_pts,
                    )
                elif edge == "west":
                    idx_w = max(0, idx_w - band_pts)

        if all_done:
            logger.info(
                "  Spatial expansion converged at iteration %d",
                n_iter,
            )
            break
    else:
        logger.info(
            "  Reached max spatial iterations (%d)",
            MAX_SPATIAL_ITERATIONS,
        )

    fin_lats = all_lats[idx_s : idx_n + 1]
    fin_lons = all_lons[idx_w : idx_e + 1]
    fin_filtered = filtered[:, idx_s : idx_n + 1, idx_w : idx_e + 1]
    consec = max_consecutive_days(fin_filtered)
    peak_gridpoints = int((consec >= MIN_CONSECUTIVE_DAYS).sum())

    result = {
        "case_id": single_case.case_id_number,
        "title": single_case.title,
        "event_type": single_case.event_type,
        "start_date": str(single_case.start_date),
        "end_date": str(single_case.end_date),
        "latitude_min": float(all_lats[idx_s]),
        "latitude_max": float(all_lats[idx_n]),
        "longitude_min": float(all_lons[idx_w]),
        "longitude_max": float(all_lons[idx_e]),
        "n_iterations": n_iter,
        "final_days": final_t,
        "_consec": consec,
        "_lats": fin_lats,
        "_lons": fin_lons,
        "_peak_gridpoints": peak_gridpoints,
    }
    logger.info(
        "  Final bounds: lat [%.2f, %.2f], lon [%.2f, %.2f] (%d days, %d iterations)",
        result["latitude_min"],
        result["latitude_max"],
        result["longitude_min"],
        result["longitude_max"],
        final_t,
        n_iter,
    )

    kind = "heatwave" if is_heatwave else "freeze"
    start = result["start_date"][:10]
    end = result["end_date"][:10]
    out_png = str(out_dir / f"case_{result['case_id']}_consecutive_{kind}_days.png")
    plot_consecutive_map(
        consec,
        fin_lats,
        fin_lons,
        cast(Literal["heat_wave", "cold_snap"], single_case.event_type),
        title=(
            f"Consecutive {kind.capitalize()} Days"
            f" — {result['title']}\n{start} to {end}"
        ),
        output_path=out_png,
    )
    logger.info("  Saved plot: %s", out_png)

    initial_box = (box_lat_min, box_lat_max, box_lon_min, box_lon_max)
    final_box = (
        result["latitude_min"],
        result["latitude_max"],
        result["longitude_min"],
        result["longitude_max"],
    )
    peak_png = str(out_dir / f"case_{result['case_id']}_peak_day_{kind}.png")
    plot_peak_day_with_bounds(
        peak_mask,
        all_lats,
        all_lons,
        initial_box,
        final_box,
        single_case.event_type,
        title=(
            f"Peak Footprint (day {peak_day}) — {result['title']}\n{start} to {end}"
        ),
        output_path=peak_png,
    )

    return result


@nb.njit(cache=True)
def _count_overlaps_nb(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_a: int,
    n_b: int,
) -> np.ndarray:
    """Return an (n_a, n_b) pixel-count overlap matrix between two grids.

    ``mat[i, j]`` is the number of pixels where ``labels_a == i + 1``
    and ``labels_b == j + 1``.
    """
    mat = np.zeros((n_a, n_b), dtype=np.int32)
    rows, cols = labels_a.shape
    for r in range(rows):
        for c in range(cols):
            a = labels_a[r, c]
            b = labels_b[r, c]
            if a > 0 and b > 0:
                mat[a - 1, b - 1] += 1
    return mat


def _resolve_event(
    oid: int,
    overlap_mat: Optional[np.ndarray],
    prev_map: Dict[int, int],
    events: Dict[int, Dict],
    cur_map: Dict[int, int],
) -> Optional[int]:
    """Find and merge prior-day events overlapping with blob ``oid``.

    Returns the surviving event ID after any merges, or None if no
    prior event overlaps the current blob.

    Args:
        oid: Current-day blob label (1-indexed).
        overlap_mat: Pixel-count overlap matrix, or None.
        prev_map: Mapping from previous-day blob label to event ID.
        events: Live events keyed by event ID (mutated).
        cur_map: Mapping from current-day blob label to event ID
            (mutated).
    """
    if overlap_mat is None:
        return None
    col = overlap_mat[:, oid - 1]
    prev_blob_ids = np.where(col > 0)[0] + 1
    eids = {prev_map[p] for p in prev_blob_ids if p in prev_map}
    alive = {e for e in eids if e in events and not events[e]["done"]}
    if not alive:
        return None

    eid = min(alive)
    for other in alive - {eid}:
        merged = events.pop(other)
        tgt = events[eid]
        tgt["lat_min"] = min(tgt["lat_min"], merged["lat_min"])
        tgt["lat_max"] = max(tgt["lat_max"], merged["lat_max"])
        tgt["lon_min"], tgt["lon_max"] = _merge_lon_extent(
            tgt["lon_min"],
            tgt["lon_max"],
            merged["lon_min"],
            merged["lon_max"],
        )
        tgt["peak"] = max(tgt["peak"], merged["peak"])
        tgt["peak_area_km2"] = max(tgt["peak_area_km2"], merged["peak_area_km2"])
        tgt["start"] = min(tgt["start"], merged["start"])
        for k, v in list(cur_map.items()):
            if v == other:
                cur_map[k] = eid
    return eid


def _terminate_declined_events(
    events: Dict[int, Dict],
    cur_map: Dict[int, int],
) -> None:
    """Mark events as done if absent today or below 50% of peak area."""
    active = set(cur_map.values())
    for eid, ev in events.items():
        if ev["done"]:
            continue
        if eid not in active:
            ev["done"] = True
        elif ev["area"] < AREA_DECLINE_FRACTION * ev["peak"]:
            ev["done"] = True


def _bbox_from_blob(
    li: np.ndarray,
    lo: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` for one blob.

    When the blob touches both column 0 and column ``len(lons)-1``
    it is assumed to cross the 0°/360° seam, and the bbox is emitted
    using the wrap convention ``lon_min > lon_max``.
    """
    lat_min = float(lats[li].min())
    lat_max = float(lats[li].max())
    cols = np.unique(lo)
    if cols[0] == 0 and cols[-1] == len(lons) - 1:
        # Largest gap between columns marks the seam; cols on either
        # side of the gap are the western (low-lon) and eastern
        # (high-lon) clusters of the blob.
        gap_idx = int(np.diff(cols).argmax())
        west_max = int(cols[gap_idx])
        east_min = int(cols[gap_idx + 1])
        return lat_min, lat_max, float(lons[east_min]), float(lons[west_max])
    return lat_min, lat_max, float(lons[lo].min()), float(lons[lo].max())


def _arc_contains_point(start: float, end: float, p: float) -> bool:
    """True if longitude ``p`` lies on the eastward arc start→end."""
    if end >= start:
        return start <= p <= end
    return p >= start or p <= end


def _arc_span(start: float, end: float) -> float:
    """Eastward arc length from ``start`` to ``end`` in degrees."""
    return (end - start) % 360.0


def _merge_lon_extent(
    cur_min: float, cur_max: float, new_min: float, new_max: float
) -> tuple[float, float]:
    """Union two longitude extents on a 360°-periodic axis.

    Each input pair may use the wrap convention ``lon_min > lon_max``
    (eastward arc that crosses the 0°/360° seam). Returns the
    smallest enclosing eastward arc as ``(lon_min, lon_max)``.
    """
    arcs = ((cur_min, cur_max), (new_min, new_max))
    pts = sorted({cur_min, cur_max, new_min, new_max})
    best_span = float("inf")
    best: tuple[float, float] = (cur_min, cur_max)
    for s in pts:
        for e in pts:
            span = _arc_span(s, e)
            if span >= best_span:
                continue
            if not all(
                _arc_contains_point(s, e, a)
                and _arc_contains_point(s, e, b)
                and _arc_span(s, e) >= _arc_span(a, b)
                for a, b in arcs
            ):
                continue
            best_span = span
            best = (s, e)
    return best


def detect_events(
    filtered_mask: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    event_type: str,
    area_grid: Optional[np.ndarray] = None,
    min_seed_area_km2: float = 0.0,
) -> List[Dict]:
    """Track spatiotemporal events from a filtered boolean mask.

    A new event is seeded only when the current-day blob AND the prior
    ``MIN_CONSECUTIVE_DAYS - 1`` days share a single contiguous region
    of at least ``min_seed_area_km2``. The joint intersection of the
    blob footprint with all prior filtered masks is computed; the
    largest connected component must meet the area threshold. Blobs
    that fail the seed test but overlap an established event still
    extend it.

    Args:
        filtered_mask: Boolean array of shape (time, lat, lon) with
            consecutive-day filtering already applied.
        dates: 1-D array of date labels aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array aligned with axis 2.
        event_type: Label string stored in each returned event dict.
        area_grid: Optional 2-D array (lat, lon) of grid-cell areas
            in km².
        min_seed_area_km2: Minimum contiguous area in km² for the
            seed test. Default 0.0 (all blobs seed).
    """
    filt_np = filtered_mask

    n_days = filt_np.shape[0]
    events: Dict[int, Dict] = {}
    next_id = 1
    prev_labels: Optional[np.ndarray] = None
    prev_n_obj: int = 0
    prev_map: Dict[int, int] = {}

    for di in range(n_days):
        day = filt_np[di]

        if not day.any():
            for ev in events.values():
                if not ev["done"]:
                    ev["done"] = True
            prev_labels = None
            prev_n_obj = 0
            prev_map = {}
            continue

        labels, n_obj = ndimage.label(day)
        labels = labels.astype(np.int32)
        cur_map: Dict[int, int] = {}

        overlap_mat: Optional[np.ndarray] = None
        if prev_labels is not None and n_obj > 0 and prev_n_obj > 0:
            overlap_mat = _count_overlaps_nb(
                prev_labels,
                labels,
                prev_n_obj,
                n_obj,
            )

        for oid in range(1, n_obj + 1):
            om = labels == oid
            area = int(om.sum())
            li, lo = np.where(om)
            blob_area_km2 = (
                float(area_grid[li, lo].sum()) if area_grid is not None else 0.0
            )

            eid = _resolve_event(
                oid,
                overlap_mat,
                prev_map,
                events,
                cur_map,
            )

            if eid is None:
                if blob_area_km2 < min_seed_area_km2:
                    continue
                # Seed test: the blob footprint must have been
                # >= min_seed_area_km2 on each of the prior
                # MIN_CONSECUTIVE_DAYS - 1 days as a single
                # contiguous region. Skip when history is too short.
                if min_seed_area_km2 > 0:
                    if di < MIN_CONSECUTIVE_DAYS - 1:
                        continue
                    joint_mask = om.copy()
                    for k in range(1, MIN_CONSECUTIVE_DAYS):
                        joint_mask &= filt_np[di - k]
                    if not joint_mask.any():
                        continue
                    j_lbl, j_n = ndimage.label(joint_mask)
                    if j_n == 0:
                        continue
                    best_area = max(
                        float(area_grid[j_lbl == c].sum())
                        if area_grid is not None
                        else float((j_lbl == c).sum())
                        for c in range(1, j_n + 1)
                    )
                    if best_area < min_seed_area_km2:
                        continue
                eid = next_id
                next_id += 1
                _lat_min, _lat_max, _lon_min, _lon_max = _bbox_from_blob(
                    li, lo, lats, lons
                )
                events[eid] = {
                    "type": event_type,
                    "start": dates[di],
                    "end": dates[di],
                    "lat_min": _lat_min,
                    "lat_max": _lat_max,
                    "lon_min": _lon_min,
                    "lon_max": _lon_max,
                    "peak": area,
                    "peak_area_km2": blob_area_km2,
                    "area": area,
                    "done": False,
                    "initial_bbox": (
                        _lat_min,
                        _lat_max,
                        _lon_min,
                        _lon_max,
                    ),
                }
            else:
                ev = events[eid]
                ev["end"] = dates[di]
                ev["peak_area_km2"] = max(ev["peak_area_km2"], blob_area_km2)
                blob_lat_min, blob_lat_max, blob_lon_min, blob_lon_max = (
                    _bbox_from_blob(li, lo, lats, lons)
                )
                ev["lat_min"] = min(ev["lat_min"], blob_lat_min)
                ev["lat_max"] = max(ev["lat_max"], blob_lat_max)
                ev["lon_min"], ev["lon_max"] = _merge_lon_extent(
                    ev["lon_min"], ev["lon_max"], blob_lon_min, blob_lon_max
                )
                ev["peak"] = max(ev["peak"], area)
                ev["area"] = area

            cur_map[oid] = eid

        _terminate_declined_events(events, cur_map)
        prev_labels = labels
        prev_n_obj = n_obj
        prev_map = cur_map

    for ev in events.values():
        ev["done"] = True

    return list(events.values())


def include_temps_with_events(
    events: List[Dict],
    t2m_daily_np: np.ndarray,
    exc_filt: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> List[Dict]:
    """Add mean and minimum temperature (°C) to each event dict.

    Uses a pre-computed global daily-mean temperature array so all
    events share a single ERA5 fetch. Statistics are computed over
    active (exceedant) grid point-days within each event's expanded
    bounding box and time window.

    Args:
        events: Event dicts with lat_min/max, lon_min/max, start, end.
        t2m_daily_np: Float32 array (days, lat, lon) of daily-mean
            2m temperature in Celsius, aligned with dates/lats/lons.
        exc_filt: Boolean array (time, lat, lon) from
            ``apply_consecutive_filter``, aligned with dates/lats/lons.
        dates: 1-D datetime64 array aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array aligned with axis 2.

    Returns:
        The same event list with ``mean_temp_c`` and ``min_temp_c``
        added (NaN when no active points exist).
    """

    def _idx(arr: np.ndarray, val: float) -> int:
        return int(np.argmin(np.abs(arr - val)))

    for ev in events:
        start_date = np.datetime64(ev["start"])
        end_date = np.datetime64(ev["end"])
        t_indices = np.where((dates >= start_date) & (dates <= end_date))[0]

        idx_s = _idx(lats, ev["lat_min"])
        idx_n = _idx(lats, ev["lat_max"])
        idx_w = _idx(lons, ev["lon_min"])
        idx_e = _idx(lons, ev["lon_max"])

        exc_ev = exc_filt[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]
        t2m_ev = t2m_daily_np[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]

        n_days = min(t2m_ev.shape[0], exc_ev.shape[0])
        active_temps = t2m_ev[:n_days][exc_ev[:n_days]]
        if active_temps.size > 0:
            ev["mean_temp_c"] = round(float(active_temps.mean()), 2)
            ev["min_temp_c"] = round(float(active_temps.min()), 2)
        else:
            ev["mean_temp_c"] = float("nan")
            ev["min_temp_c"] = float("nan")

    return events


def events_to_dataframe(
    events: List[Dict],
    min_gridpoints: int = MIN_GRIDPOINTS,
    min_area_km2: float = MIN_AREA_KM2,
) -> pd.DataFrame:
    """Filter raw events and return a labelled DataFrame.

    Drops events below ``min_gridpoints`` peak cells, below
    ``min_area_km2`` peak area, or with fewer than
    ``MIN_CONSECUTIVE_DAYS`` consecutive days.
    """
    columns = [
        "label",
        "event_type",
        "start_date",
        "end_date",
        "latitude_min",
        "latitude_max",
        "longitude_min",
        "longitude_max",
        "max_consecutive_days",
        "mean_consecutive_days",
        "mean_temp_c",
        "min_temp_c",
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    n_before = len(events)
    events = [
        e
        for e in events
        if e["peak"] >= min_gridpoints
        and e.get("peak_area_km2", 0.0) >= min_area_km2
        and e.get("max_consecutive_days", 0) >= MIN_CONSECUTIVE_DAYS
    ]
    logger.info(
        "  Filtered %d events (< %d pts, < %.0f km², or < %d consec days); %d remain",
        n_before - len(events),
        min_gridpoints,
        min_area_km2,
        MIN_CONSECUTIVE_DAYS,
        len(events),
    )

    if not events:
        return pd.DataFrame(columns=columns)

    rows = [
        {
            "event_type": e["type"],
            "start_date": str(e["start"]),
            "end_date": str(e["end"]),
            "latitude_min": e["lat_min"],
            "latitude_max": e["lat_max"],
            "longitude_min": e["lon_min"],
            "longitude_max": e["lon_max"],
            "max_consecutive_days": e.get("max_consecutive_days", 0),
            "mean_consecutive_days": e.get("mean_consecutive_days", 0.0),
            "mean_temp_c": e.get("mean_temp_c", float("nan")),
            "min_temp_c": e.get("min_temp_c", float("nan")),
        }
        for e in events
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


def load_case(case_id_number: int) -> cases.IndividualCase:
    """Load a single case from events.yaml by ``case_id_number``.

    Raises:
        ValueError: If no case with the given ID exists.
    """
    all_cases = cases.load_ewb_events_yaml_into_case_list()
    for c in all_cases:
        if c.case_id_number == case_id_number:
            return c
    hw_ids = ", ".join(
        str(c.case_id_number) for c in all_cases if c.event_type == "heat_wave"
    )
    cs_ids = ", ".join(
        str(c.case_id_number) for c in all_cases if c.event_type == "cold_snap"
    )
    raise ValueError(
        f"Case {case_id_number} not found.\n"
        f"  heat_wave ids: {hw_ids}\n"
        f"  cold_snap ids:    {cs_ids}"
    )


def compute_consecutive_field(
    single_case: cases.IndividualCase,
    quantile: float | None = None,
    op_str: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(consec, lats, lons)`` for a case's exceedance mask.

    ``consec`` is an int32 (lat, lon) array of the max consecutive
    event days over the case's window; lats and lons match the grid
    of the case's bounding box.

    Args:
        single_case: The case to compute the field for.
        quantile: Climatology quantile. Defaults to 0.85 for
            heat_wave or 0.15 for freeze.
        op_str: Comparison operator. Defaults to ``>=`` for
            heat_wave or ``<=`` for freeze.
    """
    is_heatwave = single_case.event_type == "heat_wave"
    if quantile is None:
        quantile = 0.85 if is_heatwave else 0.15
    if op_str is None:
        op_str = ">=" if is_heatwave else "<="
    cmp = resolve_op(op_str)

    start = str(single_case.start_date.date())
    end = str(single_case.end_date.date())
    bounds = single_case.location.as_geopandas().total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds

    if lon_min < 0 and lon_max > 0:
        lon_min += 360
        if lon_max > 0:
            lon_max += 360
    if lon_max < 0:
        lon_max += 360

    logger.info("Opening ERA5 for %s to %s ...", start, end)
    t2m = open_era5_t2m(start, end)
    t2m = t2m.drop_indexes("longitude").set_xindex(
        "longitude", index_cls=PeriodicBoundaryIndex, period=360
    )

    t2m = t2m.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )

    logger.info("  Subset: %s", dict(t2m.sizes))

    tdim = detect_time_dim(t2m)

    logger.info(
        "Loading q=%.2f climatology (op=%s)...",
        quantile,
        op_str,
    )
    clim = defaults.get_climatology(quantile).sortby("latitude")
    clim = clim.drop_indexes("longitude").set_xindex(
        "longitude", index_cls=PeriodicBoundaryIndex, period=360
    )
    clim = clim.sel(
        latitude=t2m.latitude,
        longitude=t2m.longitude,
        method="nearest",
    )
    doy = t2m[tdim].dt.dayofyear
    hour = t2m[tdim].dt.hour
    max_clim_doy = int(clim.dayofyear.max())
    doy_capped = doy.clip(max=max_clim_doy)

    clim_aligned = clim.sel(
        dayofyear=doy_capped,
        hour=hour,
    )

    logger.info("Computing 6-hourly exceedance...")
    exc_6h = cmp(t2m, clim_aligned)
    daily_all_pass = exc_6h.resample({tdim: "1D"}).min().astype(bool)

    logger.info("Building land mask...")
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask(daily_all_pass.longitude, daily_all_pass.latitude) == 0

    exc = daily_all_pass & land_mask
    mask_np = exc.compute().values.astype(bool)
    logger.info("Computing max consecutive days...")
    consec = max_consecutive_days(mask_np)
    return consec, daily_all_pass.latitude.values, daily_all_pass.longitude.values


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert ``process_event`` result dicts to a labelled DataFrame.

    Events below ``MIN_GRIDPOINTS`` peak grid points are dropped.
    None entries (skipped cases) are ignored. Rows are sorted by
    ``start_date``.
    """
    columns = [
        "label",
        "case_id",
        "title",
        "event_type",
        "start_date",
        "end_date",
        "latitude_min",
        "latitude_max",
        "longitude_min",
        "longitude_max",
    ]
    if not results:
        return pd.DataFrame(columns=columns)

    valid = [r for r in results if r is not None]
    n_before = len(valid)
    valid = [r for r in valid if r["_peak_gridpoints"] >= MIN_GRIDPOINTS]
    logger.info(
        "  Filtered %d events below %d gridpoints; %d remain",
        n_before - len(valid),
        MIN_GRIDPOINTS,
        len(valid),
    )

    rows = [
        {
            "event_type": r["event_type"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "latitude_min": r["latitude_min"],
            "latitude_max": r["latitude_max"],
            "longitude_min": r["longitude_min"],
            "longitude_max": r["longitude_max"],
        }
        for r in valid
    ]
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    df.insert(0, "label", range(1, len(df) + 1))
    return df


def write_bounds_to_yaml(
    results: List[Dict],
    yaml_path: pathlib.Path,
) -> None:
    """Write refreshed bounded_region parameters back into events.yaml.

    Uses ruamel.yaml round-trip mode so indentation, quoting style,
    key order, blank lines, and comments are preserved. Only cases
    present in ``results`` with a non-None value are modified.
    """
    result_map = {r["case_id"]: r for r in results if r is not None}
    if not result_map:
        logger.warning("write_bounds_to_yaml: no valid results, skipping.")
        return

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True

    with yaml_path.open("r") as fh:
        data = yaml.load(fh)

    updated = 0
    for entry in data:
        cid = entry.get("case_id_number")
        if cid not in result_map:
            continue
        r = result_map[cid]
        params = entry["location"]["parameters"]
        params["latitude_min"] = round(r["latitude_min"], 2)
        params["latitude_max"] = round(r["latitude_max"], 2)
        params["longitude_min"] = round(r["longitude_min"], 2)
        params["longitude_max"] = round(r["longitude_max"], 2)
        updated += 1

    with yaml_path.open("w") as fh:
        yaml.dump(data, fh)

    logger.info(
        "write_bounds_to_yaml: updated %d cases in %s",
        updated,
        yaml_path,
    )


def _load_base_temp_events() -> list[cases.IndividualCase]:
    """Load all cases from the bundled base_temp_events.yaml."""
    import extremeweatherbench.data

    old_yaml = importlib.resources.files(
        extremeweatherbench.data,
    ).joinpath("base_temp_events.yaml")
    with importlib.resources.as_file(old_yaml) as f:
        raw = cases.read_incoming_yaml(f)
    return cases.load_individual_cases(raw)


def _add_map_features(ax) -> None:
    """Add coastlines, borders, states, lakes, rivers to a GeoAxes."""
    ax.coastlines(linewidth=0.5, zorder=10)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.6,
        edgecolor="grey",
    )
    ax.add_feature(
        cfeature.STATES,
        linewidth=0.4,
        edgecolor="lightgrey",
    )
    ax.add_feature(
        cfeature.LAKES,
        linewidth=0.3,
        edgecolor="none",
        facecolor="none",
    )
    ax.add_feature(
        cfeature.RIVERS,
        linewidth=0.2,
        edgecolor="none",
    )


def plot_consecutive_map(
    consec: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    event_type: Literal["heat_wave", "cold_snap"],
    title: str,
    output_path: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    ax=None,
    add_colorbar: bool = True,
    vmax: int | None = None,
):
    """Plot a max-consecutive-days field using the standard style.

    When ``ax`` is provided the plot is drawn into that axis and
    nothing is saved; ``output_path`` is ignored. When ``ax`` is
    None a new figure is created and saved to ``output_path``.

    Args:
        consec: Int32 (nlat, nlon) array of max consecutive days.
        lats: 1-D latitude array matching ``consec`` rows.
        lons: 1-D longitude array (0-360) matching ``consec`` cols.
        event_type: ``"heat_wave"`` (Reds) or ``"cold_snap"`` (Blues).
        title: Two-line figure title (left-aligned).
        output_path: Destination PNG path. Required when ``ax`` is
            None.
        extent: (lon_min, lon_max, lat_min, lat_max) in -180..180.
            Derived from lats/lons when None.
        ax: Existing cartopy GeoAxes to draw into. When provided no
            figure is created or saved.
        add_colorbar: If False, skip the per-axes colorbar so the
            caller can supply a single shared colorbar.
        vmax: Override the colormap upper bound (useful for sharing
            one normalization across multiple panels).

    Returns:
        The pcolormesh mappable, so callers can build a shared
        colorbar.
    """
    plot_data = consec.astype(float)
    plot_data[consec < MIN_CONSECUTIVE_DAYS] = np.nan

    if vmax is None:
        valid = plot_data[~np.isnan(plot_data)]
        vmax = int(valid.max()) if valid.size > 0 else MIN_CONSECUTIVE_DAYS
    else:
        vmax = int(vmax)

    cmap_name = "Reds" if event_type == "heat_wave" else "Blues"
    n_levels = vmax - MIN_CONSECUTIVE_DAYS + 1
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(n_levels)
    bin_edges = np.arange(MIN_CONSECUTIVE_DAYS - 0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

    plot_lons = np.array([_to_plot_lon(x) for x in lons])

    if extent is None:
        lon_min, lon_max = float(plot_lons.min()), float(plot_lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
    else:
        lon_min, lon_max, lat_min, lat_max = extent

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(10, 8),
        )
    else:
        fig = ax.get_figure()

    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)

    im = ax.pcolormesh(
        plot_lons,
        lats,
        plot_data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    _add_map_features(ax)

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="3.75%",
            pad=0.1,
            axes_class=plt.Axes,
        )
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks(range(MIN_CONSECUTIVE_DAYS, vmax + 1))
        cbar.ax.set_ylabel(
            "Consecutive Days",
            rotation=270,
            labelpad=8,
            va="center",
            fontsize=12,
        )

    ax.set_title(title, loc="left", fontsize=13)

    if own_fig:
        if output_path is None:
            raise ValueError("output_path is required when ax is not provided.")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved consecutive-days plot to %s", output_path)

    return im


def plot_event_bounds(
    df: pd.DataFrame,
    csv_path: str,
    title: str = "Detected Heat Wave and Cold Snap Events",
) -> None:
    """Draw bounding boxes for all events on a global Robinson map.

    The PNG is saved alongside ``csv_path`` with the same stem.
    """
    if df.empty:
        logger.warning("No events to plot -- skipping bounds plot.")
        return

    out_path = str(pathlib.Path(csv_path).with_suffix(".png"))
    hw_count = int((df["event_type"] == "heat_wave").sum())
    fz_count = int((df["event_type"] == "cold_snap").sum())

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(14, 7),
    )
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.coastlines(linewidth=0.5, zorder=2)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.3,
        edgecolor="grey",
        zorder=2,
    )

    for _, row in df.iterrows():
        lon0 = _to_plot_lon(float(row["longitude_min"]))
        lon1 = _to_plot_lon(float(row["longitude_max"]))
        lat0, lat1 = float(row["latitude_min"]), float(row["latitude_max"])
        if lon1 < lon0:
            lon1 += 360.0
        color = _HW_COLOR if row["event_type"] == "heat_wave" else _FZ_COLOR
        kw = dict(
            width=lon1 - lon0,
            height=lat1 - lat0,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(
            mpatches.Rectangle(
                xy=(lon0, lat0),
                facecolor=color,
                edgecolor=color,
                alpha=_BOX_ALPHA,
                linewidth=1.0,
                zorder=3,
                **kw,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                xy=(lon0, lat0),
                facecolor="none",
                edgecolor=color,
                alpha=_BOX_EDGE_ALPHA,
                linewidth=1.0,
                zorder=4,
                **kw,
            )
        )

    legend_handles = []
    if hw_count:
        legend_handles.append(
            mpatches.Patch(
                facecolor=_HW_COLOR,
                alpha=0.7,
                label=f"Heat wave ({hw_count})",
            )
        )
    if fz_count:
        legend_handles.append(
            mpatches.Patch(
                facecolor=_FZ_COLOR,
                alpha=0.7,
                label=f"Cold Snap ({fz_count})",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=10,
            framealpha=0.85,
        )

    ax.set_title(title, loc="left", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bounds plot to %s", out_path)


def plot_peak_day_with_bounds(
    peak_mask: np.ndarray,
    all_lats: np.ndarray,
    all_lons: np.ndarray,
    initial_box: tuple,
    final_box: tuple,
    event_type: str,
    title: str,
    output_path: str,
) -> None:
    """Plot the peak-footprint day with initial and final boxes.

    Args:
        peak_mask: 2-D boolean array (lat, lon) for the peak day.
        all_lats: 1-D latitude array matching rows.
        all_lons: 1-D longitude array (0-360) matching columns.
        initial_box: (lat_min, lat_max, lon_min, lon_max) of the
            initial centered region.
        final_box: (lat_min, lat_max, lon_min, lon_max) after
            spatial expansion.
        event_type: ``"heat_wave"`` or ``"freeze"``.
        title: Figure title.
        output_path: Destination PNG path.
    """
    plot_lons = np.array([_to_plot_lon(x) for x in all_lons])
    lon_min = float(plot_lons.min()) - 1
    lon_max = float(plot_lons.max()) + 1
    lat_min = float(all_lats.min()) - 1
    lat_max = float(all_lats.max()) + 1

    cmap = mcolors.ListedColormap(
        ["white", "firebrick" if event_type == "heat_wave" else "steelblue"]
    )
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(10, 8),
    )
    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)

    ax.pcolormesh(
        plot_lons,
        all_lats,
        peak_mask.astype(float),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    def _add_rect(box, color, label, ls="-"):
        blat_min, blat_max, blon_min, blon_max = box
        blon_min = _to_plot_lon(blon_min)
        blon_max = _to_plot_lon(blon_max)
        rect = mpatches.Rectangle(
            (blon_min, blat_min),
            blon_max - blon_min,
            blat_max - blat_min,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle=ls,
            transform=ccrs.PlateCarree(),
            label=label,
        )
        ax.add_patch(rect)

    _add_rect(initial_box, "blue", "Initial box", ls="--")
    _add_rect(final_box, "green", "Final bounds", ls="-")

    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved peak-day plot: %s", output_path)


def _add_lon_rect(
    ax: plt.Axes,
    lon_min_0360: float,
    lon_max_0360: float,
    lat_min: float,
    lat_max: float,
    color: tuple,
    linewidth: float = 0.8,
    zorder: int = 4,
) -> None:
    """Add a lat/lon rectangle, splitting at the antimeridian if needed.

    Inputs are 0-360 longitude. When the event wraps across the
    0°/360° boundary the box is drawn as two rectangles so matplotlib
    does not produce an inverted or near-zero-width patch.
    """

    def _to_180(lon: float) -> float:
        return lon if lon <= 180 else lon - 360

    lmin = _to_180(lon_min_0360)
    lmax = _to_180(lon_max_0360)

    def _rect(x0: float, x1: float) -> mpatches.Rectangle:
        return mpatches.Rectangle(
            (x0, lat_min),
            x1 - x0,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=(*color[:3], 0.25),
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )

    if lmin <= lmax:
        ax.add_patch(_rect(lmin, lmax))
    else:
        ax.add_patch(_rect(lmin, 180.0))
        ax.add_patch(_rect(-180.0, lmax))


def plot_events_global(
    df: pd.DataFrame,
    event_type: str,
    title: str,
    output_path: str,
) -> None:
    """Plot detected events as bounding boxes on a global Robinson map.

    Boxes are coloured by start date using ``Reds`` for heat waves or
    ``Blues`` for cold snaps.
    """
    cmap_name = "Reds" if event_type == "heat_wave" else "Blues"
    cmap = plt.colormaps.get_cmap(cmap_name)

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=1)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.5, zorder=2)

    if df.empty:
        ax.set_title(f"{title}\n(no events)", fontsize=11, loc="left")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Events plot saved to %s (no events)", output_path)
        return

    start_dates = pd.to_datetime(df["start_date"])
    date_min = start_dates.min()
    date_max = start_dates.max()
    span = max((date_max - date_min).days, 1)
    norm = mcolors.Normalize(vmin=0, vmax=span)

    for _, row in df.iterrows():
        days = (pd.to_datetime(row["start_date"]) - date_min).days
        color = cmap(norm(days))
        _add_lon_rect(
            ax,
            row["longitude_min"],
            row["longitude_max"],
            row["latitude_min"],
            row["latitude_max"],
            color,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    tick_days = np.linspace(0, span, min(7, span + 1))
    cbar.set_ticks(tick_days)
    cbar.set_ticklabels(
        [(date_min + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in tick_days]
    )
    cbar.set_label("Event start date", fontsize=9)
    ax.set_title(f"{title}\n{len(df)} events", fontsize=11, loc="left")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Events plot saved to %s", output_path)


def plot_events_high_consec(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    min_consec: int = 5,
) -> None:
    """Plot events with >= ``min_consec`` days, coloured by run length.

    Uses the ``plasma_r`` colormap (neutral, not event-type specific).
    """
    sub = df[df["max_consecutive_days"] >= min_consec].copy()

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=1)
    ax.coastlines(linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="grey", alpha=0.5, zorder=2)

    if sub.empty:
        ax.set_title(
            f"{title}\n(no events with >= {min_consec} consecutive days)",
            fontsize=11,
            loc="left",
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            "High-consec plot saved to %s (no qualifying events)",
            output_path,
        )
        return

    vmin = min_consec
    vmax = int(sub["max_consecutive_days"].max())
    n_levels = vmax - vmin + 1
    cmap = plt.colormaps.get_cmap("plasma_r").resampled(n_levels)
    bin_edges = np.arange(vmin - 0.5, vmax + 1.5, 1)
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

    for _, row in sub.iterrows():
        consec_val = int(row["max_consecutive_days"])
        color = cmap(norm(consec_val))
        _add_lon_rect(
            ax,
            row["longitude_min"],
            row["longitude_max"],
            row["latitude_min"],
            row["latitude_max"],
            color,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_ticks(range(vmin, vmax + 1))
    cbar.set_label("Max Consecutive Days", fontsize=9)
    ax.set_title(f"{title}\n{len(sub)} events", fontsize=11, loc="left")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("High-consec plot saved to %s", output_path)


def _to_plot_lon_axis(lons_0360: np.ndarray) -> np.ndarray:
    """Convert a wrap-aware lon slab to a monotonic axis for plotting.

    Inputs are 0-360 longitudes that may step backwards across the
    seam (e.g. ``[350.0, ..., 359.75, 0.0, ..., 5.0]``). Returns the
    same values shifted so they ascend monotonically (and potentially
    extend slightly negative, e.g. ``[-10.0, ..., -0.25, 0.0, ..., 5.0]``).
    """
    if lons_0360.size == 0:
        return lons_0360
    diffs = np.diff(lons_0360)
    wrap = np.where(diffs < 0)[0]
    if wrap.size > 0:
        split = int(wrap[0]) + 1
        return np.concatenate([lons_0360[:split] - 360.0, lons_0360[split:]])
    return np.where(lons_0360 > 180.0, lons_0360 - 360.0, lons_0360)


def plot_event_consec_maps(
    df: pd.DataFrame,
    exc_filt: np.ndarray,
    dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    stem: str,
    min_consec: int = 3,
) -> None:
    """Save a per-event consecutive-days map for each qualifying event.

    For each row in ``df`` with ``max_consecutive_days >= min_consec``,
    slices ``exc_filt`` to the event's bounding box and time window,
    computes per-grid-point max consecutive days, and writes a
    ``plasma_r`` pcolormesh to
    ``{stem}_event_{label:04d}_consec.png``.

    Args:
        df: DataFrame from ``events_to_dataframe``.
        exc_filt: Boolean array (time, lat, lon) from
            ``apply_consecutive_filter``, aligned with
            dates/lats/lons.
        dates: 1-D datetime64 array aligned with axis 0.
        lats: 1-D latitude array aligned with axis 1.
        lons: 1-D longitude array (0-360) aligned with axis 2.
        stem: Output path stem.
        min_consec: Minimum max_consecutive_days to plot. Default 3.
    """
    qualifying = df[df["max_consecutive_days"] >= min_consec]
    if qualifying.empty:
        logger.info(
            "No events with >= %d consecutive days to plot individually.",
            min_consec,
        )
        return

    logger.info(
        "Plotting %d per-event consecutive-days maps...",
        len(qualifying),
    )

    def _idx(arr: np.ndarray, val: float) -> int:
        return int(np.argmin(np.abs(arr - val)))

    plot_lons = np.where(lons > 180, lons - 360.0, lons)

    for _, row in qualifying.iterrows():
        label = int(row["label"])
        start_date = np.datetime64(row["start_date"])
        end_date = np.datetime64(row["end_date"])
        t_indices = np.where((dates >= start_date) & (dates <= end_date))[0]

        idx_s = _idx(lats, row["latitude_min"])
        idx_n = _idx(lats, row["latitude_max"])
        idx_w = _idx(lons, row["longitude_min"])
        idx_e = _idx(lons, row["longitude_max"])

        ev_filt = exc_filt[t_indices][:, idx_s : idx_n + 1, idx_w : idx_e + 1]
        if ev_filt.size == 0:
            continue
        consec = max_consecutive_days(ev_filt)

        ev_lats = lats[idx_s : idx_n + 1]
        ev_lons = plot_lons[idx_w : idx_e + 1]

        plot_data = consec.astype(float)
        plot_data[consec < MIN_CONSECUTIVE_DAYS] = np.nan
        valid = plot_data[~np.isnan(plot_data)]
        if valid.size == 0:
            continue

        vmax = int(valid.max())
        n_levels = vmax - MIN_CONSECUTIVE_DAYS + 1
        cmap = plt.colormaps.get_cmap("plasma_r").resampled(n_levels)
        bin_edges = np.arange(MIN_CONSECUTIVE_DAYS - 0.5, vmax + 1.5, 1)
        norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_levels)

        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(10, 8),
        )
        ax.set_extent(
            [
                float(ev_lons.min()) - 1,
                float(ev_lons.max()) + 1,
                float(ev_lats.min()) - 1,
                float(ev_lats.max()) + 1,
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        im = ax.pcolormesh(
            ev_lons,
            ev_lats,
            plot_data,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.coastlines(linewidth=0.5, zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="grey")
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="lightgrey")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(MIN_CONSECUTIVE_DAYS, vmax + 1))
        cbar.set_label("Consecutive Days", fontsize=12)

        start_str = str(row["start_date"])[:10]
        end_str = str(row["end_date"])[:10]
        ax.set_title(
            f"Event {label}: Consecutive Exceedance Days\n{start_str} to {end_str}",
            loc="left",
            fontsize=13,
        )
        out_path = f"{stem}_event_{label:04d}_consec.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Event %04d saved to %s", label, out_path)


def _default_output(case_id_number: int, event_type: str) -> str:
    kind = "heatwave" if event_type == "heat_wave" else "cold_snap"
    return f"case_{case_id_number}_consecutive_{kind}_days.png"


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    """Populate the ``plot`` subparser."""
    parser.add_argument(
        "--case-id-number",
        type=int,
        required=True,
        help="case_id_number from events.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path "
            "(default: case_N_consecutive_{heatwave|cold_snap}_days.png)"
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help=(
            f"Climatology quantile (one of {VALID_QUANTILES}). "
            "Defaults to 0.85 for heat_wave or 0.15 for freeze."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Comparison operator string (>, >=, <, <=, ==). "
            "Defaults to > for heat_wave or < for freeze."
        ),
    )


def _add_case_args(parser: argparse.ArgumentParser) -> None:
    """Populate the ``case`` subparser."""
    parser.add_argument(
        "--output",
        default="heat_cold_yaml.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (joblib)",
    )
    parser.add_argument(
        "--case-min",
        type=int,
        default=None,
        help="Minimum case_id_number to process (inclusive)",
    )
    parser.add_argument(
        "--case-max",
        type=int,
        default=None,
        help="Maximum case_id_number to process (inclusive)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help=(
            f"Climatology quantile ({VALID_QUANTILES}). "
            "Defaults to 0.85 for heat_wave or 0.15 for freeze."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Comparison operator (>, >=, <, <=, ==). "
            "Defaults to > for heat_wave or < for freeze."
        ),
    )


def _add_global_args(parser: argparse.ArgumentParser) -> None:
    """Populate the ``global`` subparser."""
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--output",
        default="events_global.csv",
        help="Output CSV path stem (actual file will be {stem}_events.csv)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of dask workers",
    )
    parser.add_argument(
        "--dask-batch",
        type=int,
        default=1,
        help=(
            "Number of monthly chunks to submit to Dask at once. "
            "Default 1 (one month per batch). Dask still parallelizes "
            "daily time steps within each month across workers. Larger "
            "values increase cross-month parallelism but cause "
            "rechunk-merge bottlenecks when months span many zarr "
            "chunks."
        ),
    )
    parser.add_argument(
        "--quantile-lower",
        type=float,
        default=None,
        help=(
            f"Lower-bound climatology quantile ({VALID_QUANTILES}). "
            "At least one of --quantile-lower or --quantile-upper required."
        ),
    )
    parser.add_argument(
        "--operator-lower",
        default=">",
        help="Comparison operator for the lower bound (default: >)",
    )
    parser.add_argument(
        "--quantile-upper",
        type=float,
        default=None,
        help=(
            f"Upper-bound climatology quantile ({VALID_QUANTILES}). "
            "Combine with --quantile-lower to define a band."
        ),
    )
    parser.add_argument(
        "--operator-upper",
        default="<",
        help="Comparison operator for the upper bound (default: <)",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=-60.0,
        help=(
            "Minimum latitude to include in detection. Default -60.0 "
            "(excludes Antarctica, whose polar grid geometry produces "
            "full-circumference ring blobs that trivially meet the area "
            "threshold). Pass -90.0 explicitly to include Antarctica."
        ),
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=90.0,
        help="Maximum latitude to include in detection. Default 90.0",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=1,
        help=(
            "Process the exceedance mask this many months at a time "
            "to stay within Dask worker memory limits. Set to 0 to "
            "compute the full date range at once. Default is 1."
        ),
    )
    parser.add_argument(
        "--min-area-km2",
        type=float,
        default=200_000.0,
        help=(
            "Minimum contiguous blob area in km² required to seed a "
            "new event AND on each of the prior MIN_CONSECUTIVE_DAYS-1 "
            "days (joint-area check). Always applied, including in "
            "sensitivity mode. Default 200 000."
        ),
    )
    parser.add_argument(
        "--sensitivity-thresholds",
        type=float,
        nargs="+",
        default=None,
        metavar="KM2",
        help=(
            "Count surviving events at each supplied area threshold "
            "(km²) and write a summary CSV. Events are seeded at "
            "--min-area-km2 first; thresholds below that value will "
            "equal the seed-filter count. Example: "
            "--sensitivity-thresholds 200000 300000 500000 1000000"
        ),
    )


def _run_plot(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run the ``plot`` subcommand."""
    t0 = time_module.time()
    single_case = load_case(args.case_id_number)
    logger.info(
        "Case %d: %s (%s)",
        single_case.case_id_number,
        single_case.title,
        single_case.event_type,
    )

    if args.output is None:
        args.output = _default_output(
            args.case_id_number,
            single_case.event_type,
        )

    consec, lats, lons = compute_consecutive_field(
        single_case,
        quantile=args.quantile,
        op_str=args.operator,
    )

    kind = "Heat Wave" if single_case.event_type == "heat_wave" else "Cold Snap"
    start_str = str(single_case.start_date.date())
    end_str = str(single_case.end_date.date())

    plot_consecutive_map(
        consec,
        lats,
        lons,
        cast(Literal["heat_wave", "cold_snap"], single_case.event_type),
        title=f"Consecutive {kind} Days\n{start_str} to {end_str}",
        output_path=args.output,
    )
    logger.info("Done in %.1f s", time_module.time() - t0)


def _run_case(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run the ``case`` subcommand."""
    wall_start = time_module.time()

    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    old_cases = _load_base_temp_events()
    hw_fz = [
        e
        for e in old_cases
        if e.event_type in ("heat_wave", "freeze")
        and (args.case_min is None or e.case_id_number >= args.case_min)
        and (args.case_max is None or e.case_id_number <= args.case_max)
    ]
    logger.info(
        "Found %d heat_wave / freeze events in base_temp_events.yaml",
        len(hw_fz),
    )

    out_dir = pathlib.Path(args.output).parent

    results = joblib.Parallel(n_jobs=args.n_workers)(
        joblib.delayed(process_event)(
            c,
            out_dir,
            quantile=args.quantile,
            op_str=args.operator,
        )
        for c in hw_fz
    )

    df = results_to_dataframe(results)
    out_path = pathlib.Path(args.output)
    if out_path.exists():
        df.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df.to_csv(out_path, index=False)

    yaml_path = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "extremeweatherbench"
        / "data"
        / "events.yaml"
    )
    write_bounds_to_yaml(results, yaml_path)

    elapsed = time_module.time() - wall_start
    logger.info(
        "Done: %d events -> %s (%.1f s)",
        len(df),
        args.output,
        elapsed,
    )
    client.close()


def _run_global(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Run the ``global`` subcommand."""
    if args.quantile_lower is None and args.quantile_upper is None:
        parser.error(
            "At least one of --quantile-lower or --quantile-upper must be provided."
        )

    ql, qu = args.quantile_lower, args.quantile_upper
    if ql is not None and qu is not None:
        event_type = f"q{ql:.2f}-q{qu:.2f}"
    elif ql is not None:
        event_type = f"q{ql:.2f}+"
    else:
        event_type = f"q{qu:.2f}-"

    if ql is not None and ql >= 0.5:
        plot_event_type = "heat_wave"
    elif qu is not None and qu <= 0.5:
        plot_event_type = "cold_snap"
    else:
        plot_event_type = "heat_wave"

    is_sensitivity = args.sensitivity_thresholds is not None
    if is_sensitivity:
        below = [t for t in args.sensitivity_thresholds if t < args.min_area_km2]
        if below:
            logger.warning(
                "Sensitivity thresholds %s are below the seed filter "
                "(%.0f km²); those counts will equal the seed-filter total.",
                below,
                args.min_area_km2,
            )

    wall_start = time_module.time()
    client = Client(LocalCluster(n_workers=args.n_workers))
    logger.info("Dask dashboard: %s", client.dashboard_link)

    logger.info("Pre-warming numba JIT...")
    _tiny = np.zeros((2, 2), dtype=np.int32)
    _tiny[0, 0] = 1
    _tiny[1, 1] = 1
    _count_overlaps_nb(_tiny, _tiny, 1, 1)
    logger.info("  numba ready")

    logger.info("Opening ERA5 data...")
    t2m = open_era5_t2m(args.start_date, args.end_date)
    if args.lat_min != -90.0 or args.lat_max != 90.0:
        t2m = t2m.sel(latitude=slice(args.lat_min, args.lat_max))
        logger.info(
            "  Latitude filtered to [%.1f, %.1f]",
            args.lat_min,
            args.lat_max,
        )
    logger.info("  sizes=%s", dict(t2m.sizes))

    logger.info(
        "Loading climatology bounds (lower=%s, upper=%s)...",
        args.quantile_lower,
        args.quantile_upper,
    )
    clim_lower, clim_upper = get_climatology_bounds(
        q_lower=args.quantile_lower,
        q_upper=args.quantile_upper,
    )

    logger.info("Building land mask and grid-cell area array...")
    land_mask = build_land_mask(t2m.longitude, t2m.latitude)
    land_mask_np = land_mask.values.astype(bool)
    area_grid = compute_grid_cell_area(t2m.latitude.values, t2m.longitude.values)

    tdim = detect_time_dim(t2m)

    # Build the exceedance mask and daily temperature together so
    # both passes read the same ERA5 data once. Lazy month-chunks are
    # handed to a single dask.compute() per batch so Dask can schedule
    # months in parallel.
    t0 = time_module.time()
    if args.chunk_months > 0:
        logger.info(
            "Building lazy graphs for exceedance mask + temperature "
            "(%d-month chunks, parallel)...",
            args.chunk_months,
        )
        month_starts = pd.date_range(args.start_date, args.end_date, freq="MS")
        chunk_labels = []
        exc_lazy_list = []
        t2m_lazy_list = []
        for i in range(0, len(month_starts), args.chunk_months):
            ms = month_starts[i]
            me = month_starts[min(i + args.chunk_months, len(month_starts)) - 1]
            me = me + pd.offsets.MonthEnd(1)
            chunk_labels.append(f"{ms.strftime('%Y-%m')} - {me.strftime('%Y-%m')}")
            t2m_chunk = t2m.sel({tdim: slice(str(ms.date()), str(me.date()))})
            exc_lazy_list.append(
                build_exceedance_mask(
                    t2m_chunk,
                    clim_lower,
                    clim_upper,
                    land_mask,
                    op_lower=args.operator_lower,
                    op_upper=args.operator_upper,
                )
            )
            t2m_lazy_list.append(t2m_chunk.resample({tdim: "1D"}).mean())

        n_chunks = len(exc_lazy_list)
        batch = args.dask_batch
        logger.info(
            "  Computing %d chunks, %d per batch, via Dask...",
            n_chunks,
            batch,
        )
        exc_parts: list = []
        t2m_parts: list = []
        for b0 in range(0, n_chunks, batch):
            b1 = min(b0 + batch, n_chunks)
            b_exc = exc_lazy_list[b0:b1]
            b_t2m = t2m_lazy_list[b0:b1]
            b_sz = b1 - b0
            b_results = dask.compute(*b_exc, *b_t2m)
            exc_parts.extend(b_results[:b_sz])
            t2m_parts.extend(b_results[b_sz:])
            for label in chunk_labels[b0:b1]:
                logger.info("  chunk %s done", label)

        exc_da = xr.concat(exc_parts, dim=tdim)
        t2m_daily_da = xr.concat(t2m_parts, dim=tdim)
        del exc_parts, t2m_parts
    else:
        logger.info(
            "Building lazy graphs (all-at-once) for exceedance mask + temperature..."
        )
        exc_lazy = build_exceedance_mask(
            t2m,
            clim_lower,
            clim_upper,
            land_mask,
            op_lower=args.operator_lower,
            op_upper=args.operator_upper,
        )
        t2m_lazy = t2m.resample({tdim: "1D"}).mean()
        exc_da, t2m_daily_da = dask.compute(exc_lazy, t2m_lazy)

    logger.info(
        "  exceedance mask + temperature done in %.1f s",
        time_module.time() - t0,
    )

    dates = exc_da[tdim].values
    lats = exc_da.latitude.values
    lons = exc_da.longitude.values
    exc_np = exc_da.values.astype(bool)
    del exc_da

    t2m_daily_np = (t2m_daily_da.values - 273.15).astype(np.float32)
    del t2m_daily_da

    logger.info(
        "Applying %d-day consecutive filter...",
        MIN_CONSECUTIVE_DAYS,
    )
    exc_filt = _consecutive_filter_np(exc_np, MIN_CONSECUTIVE_DAYS, 1)
    logger.info(
        "  %d -> %d True cells",
        exc_np.sum(),
        exc_filt.sum(),
    )
    del exc_np

    logger.info(
        "Detecting events (min_seed_area_km2=%.0f)...",
        args.min_area_km2,
    )
    t0 = time_module.time()
    events = detect_events(
        exc_filt,
        dates,
        lats,
        lons,
        event_type,
        area_grid=area_grid,
        min_seed_area_km2=args.min_area_km2,
    )
    logger.info("  %d raw events (%.1f s)", len(events), time_module.time() - t0)

    logger.info(
        "Expanding event bounds (%d-deg steps)...",
        EXPANSION_DEGREES,
    )
    t0 = time_module.time()
    events = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(expand_event_bounds)(
            ev, exc_filt, dates, lats, lons, land_mask_np
        )
        for ev in events
    )
    logger.info("  done in %.1f s", time_module.time() - t0)

    logger.info("Computing per-event temperature statistics...")
    t0 = time_module.time()
    events = include_temps_with_events(
        events, t2m_daily_np, exc_filt, dates, lats, lons
    )
    del t2m_daily_np
    logger.info("  done in %.1f s", time_module.time() - t0)

    stem = str(pathlib.Path(args.output).with_suffix(""))

    df = events_to_dataframe(events, min_area_km2=args.min_area_km2)
    events_csv = f"{stem}_events.csv"
    df.to_csv(events_csv, index=False)
    logger.info("%d events written to %s", len(df), events_csv)

    logger.info("Computing max-consecutive-days plot...")
    consec = max_consecutive_days(exc_filt)
    plot_consecutive_map(
        consec,
        lats,
        lons,
        cast(Literal["heat_wave", "cold_snap"], plot_event_type),
        title=(
            f"Consecutive Exceedance Days ({event_type})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_consec.png",
    )
    del consec

    plot_events_global(
        df,
        event_type=plot_event_type,
        title=(
            f"Detected events ({event_type.replace('_', ' ')})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_events.png",
    )
    plot_events_high_consec(
        df,
        title=(
            f"Events \u22656 consecutive days ({event_type.replace('_', ' ')})"
            f"\n{args.start_date} to {args.end_date}"
        ),
        output_path=f"{stem}_high_consec.png",
    )

    plot_event_consec_maps(df, exc_filt, dates, lats, lons, stem)

    del exc_filt

    if is_sensitivity:
        thresholds = sorted(args.sensitivity_thresholds)
        logger.info(
            "Sensitivity test: counting events at %d thresholds...",
            len(thresholds),
        )
        rows = []
        for thresh in thresholds:
            df_thresh = events_to_dataframe(events, min_area_km2=thresh)
            rows.append({"area_threshold_km2": thresh, "n_events": len(df_thresh)})
        sens_df = pd.DataFrame(rows)
        print(sens_df.to_string(index=False))
        sens_path = f"{stem}_sensitivity.csv"
        sens_df.to_csv(sens_path, index=False)
        logger.info("Sensitivity table saved to %s", sens_path)

    elapsed = time_module.time() - wall_start
    logger.info("Done in %.1f s", elapsed)
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="temperature_events",
        description=(
            "ERA5 temperature-event tooling: plot a single case,"
            " re-expand curated cases, or scan a date range globally."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    plot_p = sub.add_parser(
        "plot",
        help="Plot one case from events.yaml.",
        description=(
            "Render a consecutive-exceedance-days map for one case"
            " identified by case_id_number."
        ),
    )
    _add_plot_args(plot_p)

    case_p = sub.add_parser(
        "case",
        help="Re-expand curated heat_wave / freeze cases.",
        description=(
            "Iterate every centered_region case in base_temp_events.yaml,"
            " expand its time and spatial bounds, and write the"
            " refreshed bounds back to events.yaml."
        ),
    )
    _add_case_args(case_p)

    global_p = sub.add_parser(
        "global",
        help="Detect temperature events over a date range.",
        description=(
            "Scan ERA5 2m temperature over a date range, detect events"
            " that stay within a climatology-quantile band for at least"
            " MIN_CONSECUTIVE_DAYS days over land, and write a CSV plus"
            " plots."
        ),
    )
    _add_global_args(global_p)

    args = parser.parse_args()
    if args.cmd == "plot":
        _run_plot(args, plot_p)
    elif args.cmd == "case":
        _run_case(args, case_p)
    elif args.cmd == "global":
        _run_global(args, global_p)
    else:  # pragma: no cover -- argparse enforces this
        parser.error(f"Unknown subcommand: {args.cmd!r}")


if __name__ == "__main__":
    main()
