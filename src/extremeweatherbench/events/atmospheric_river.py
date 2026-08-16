from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from numba import float64, guvectorize, njit
from scipy import ndimage

from extremeweatherbench import calc, utils


@njit(cache=True)
def _ufind(parent: np.ndarray, x: int) -> int:
    """Path-compressing union-find parent lookup."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _label_time_linked_stack(mask: np.ndarray) -> np.ndarray:
    """6-connected labels on (time, extra, y, x); extras stay unlinked."""
    n_t, n_extra, n_y, n_x = mask.shape
    out = np.zeros((n_t, n_extra, n_y, n_x), dtype=np.int32)
    n_pix = n_t * n_y * n_x
    parent = np.empty(n_pix, dtype=np.int32)
    remap = np.empty(n_pix, dtype=np.int32)
    stride_t = n_y * n_x
    next_id = 1
    for e in range(n_extra):
        for i in range(n_pix):
            parent[i] = i
            remap[i] = 0
        # Union True pixels with west, south, and previous-time neighbors.
        for t in range(n_t):
            for y in range(n_y):
                for x in range(n_x):
                    if not mask[t, e, y, x]:
                        continue
                    idx = t * stride_t + y * n_x + x
                    if x > 0 and mask[t, e, y, x - 1]:
                        ra = _ufind(parent, idx)
                        rb = _ufind(parent, idx - 1)
                        if ra != rb:
                            parent[rb] = ra
                    if y > 0 and mask[t, e, y - 1, x]:
                        ra = _ufind(parent, idx)
                        rb = _ufind(parent, idx - n_x)
                        if ra != rb:
                            parent[rb] = ra
                    if t > 0 and mask[t - 1, e, y, x]:
                        ra = _ufind(parent, idx)
                        rb = _ufind(parent, idx - stride_t)
                        if ra != rb:
                            parent[rb] = ra
        # Assign dense labels; extras do not share IDs.
        next_local = 0
        for t in range(n_t):
            for y in range(n_y):
                for x in range(n_x):
                    if not mask[t, e, y, x]:
                        continue
                    idx = t * stride_t + y * n_x + x
                    root = _ufind(parent, idx)
                    lab = remap[root]
                    if lab == 0:
                        next_local += 1
                        lab = next_local
                        remap[root] = lab
                    out[t, e, y, x] = lab + (next_id - 1)
        next_id += next_local
    return out


def _label_objects_time_linked(
    mask: npt.NDArray[np.bool_],
    dims: Sequence[str],
    time_dimension: str,
) -> npt.NDArray[np.int32]:
    """Label True pixels, linking through time but not across other dims."""
    keep = {time_dimension, "latitude", "longitude"}
    if not keep <= set(dims):
        labeled, _ = ndimage.label(mask)
        return np.asarray(labeled, dtype=np.int32)

    order = (
        [dims.index(time_dimension)]
        + [i for i, name in enumerate(dims) if name not in keep]
        + [dims.index("latitude"), dims.index("longitude")]
    )
    moved = np.ascontiguousarray(np.transpose(mask, order))
    n_t, *extra, n_y, n_x = moved.shape
    n_extra = int(np.prod(extra, initial=1))
    stacked = moved.reshape(n_t, n_extra, n_y, n_x)
    out = _label_time_linked_stack(stacked)
    return np.transpose(out.reshape(n_t, *extra, n_y, n_x), np.argsort(order))


def _filter_labels_size_and_lat(
    labeled: npt.NDArray[np.int32],
    latitudes: npt.NDArray[np.floating],
    lat_axis: int,
    min_size: int,
    min_abs_lat: float = 15.0,
) -> npt.NDArray[np.bool_]:
    """Keep labels with enough pixels and |mean lat| above the tropics."""
    if labeled.size == 0:
        return np.zeros(labeled.shape, dtype=bool)
    n = int(labeled.max()) + 1
    flat = labeled.ravel()
    counts = np.bincount(flat, minlength=n)
    lat_shape = [1] * labeled.ndim
    lat_shape[lat_axis] = np.asarray(latitudes).size
    weights = np.broadcast_to(
        np.asarray(latitudes, dtype=np.float64).reshape(lat_shape),
        labeled.shape,
    )
    lat_sums = np.bincount(flat, weights=weights.ravel(), minlength=n)
    valid = np.zeros(n, dtype=np.bool_)
    if n > 1:
        denom = np.maximum(counts[1:], 1)
        valid[1:] = (counts[1:] >= min_size) & (
            np.abs(lat_sums[1:] / denom) > min_abs_lat
        )
    return valid[labeled]


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:], float64, float64[:])],
    "(n),(n),(n),(n),()->()",
    nopython=True,
    target="cpu",
)
def _ivt_fused_kernel(u, v, q, x, g0, out):
    """Trapezoid-integrate u*q and v*q, then hypot / g0."""
    east = 0.0
    north = 0.0
    for i in range(len(u) - 1):
        uq0 = u[i] * q[i]
        uq1 = u[i + 1] * q[i + 1]
        vq0 = v[i] * q[i]
        vq1 = v[i + 1] * q[i + 1]
        dx = x[i + 1] - x[i]
        if not (np.isnan(uq0) or np.isnan(uq1)):
            east += dx * (uq0 + uq1) / 2.0
        if not (np.isnan(vq0) or np.isnan(vq1)):
            north += dx * (vq0 + vq1) / 2.0
    east /= g0
    north /= g0
    out[()] = np.hypot(east, north)


def _dilated_high_laplacian(
    ivt: xr.DataArray,
    sigma: float,
    laplacian_threshold: float,
    dilation_radius: int,
) -> xr.DataArray:
    """Threshold the blurred Laplacian and dilate on the same IVT chunks."""

    def _ufunc(data, sigma_, lap_thresh, radius):
        lap = calc._compute_blurred_laplacian_ufunc(data, sigma_)
        high = np.abs(lap) >= lap_thresh
        return calc._binary_dilation_ufunc(high, radius)

    return xr.apply_ufunc(
        _ufunc,
        ivt,
        sigma,
        laplacian_threshold,
        dilation_radius,
        input_core_dims=[["latitude", "longitude"], [], [], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[np.int8],
    )


def _finalize_ar_mask(
    intersection: npt.NDArray[np.bool_],
    coords_dict: dict,
    min_size_gridpoints: int,
    time_dimension: str,
) -> xr.DataArray:
    """Label, size-filter, and drop tropical features from a bool intersection."""
    labeled_array = _label_objects_time_linked(
        np.asarray(intersection, dtype=bool),
        list(coords_dict.keys()),
        time_dimension,
    )
    latitudes = np.asarray(coords_dict["latitude"].values)
    lat_axis = list(coords_dict.keys()).index("latitude")
    feature_mask = _filter_labels_size_and_lat(
        labeled_array, latitudes, lat_axis, min_size_gridpoints
    )
    ar_mask = xr.DataArray(
        feature_mask.astype(np.int8), coords=coords_dict, dims=coords_dict.keys()
    )
    ar_mask.name = "atmospheric_river_mask"
    return ar_mask


def _finalize_ar_mask_by_lead(
    intersection: xr.DataArray,
    min_size_gridpoints: int,
    time_dimension: str,
) -> xr.DataArray:
    """Label each lead independently so masks can stream as they compute."""
    if "lead_time" in intersection.dims:
        slices = [
            intersection.isel(lead_time=i)
            for i in range(intersection.sizes["lead_time"])
        ]
    else:
        slices = [intersection]
    parts = []
    for sl in slices:
        coords = {dim: sl.coords[dim] for dim in sl.dims}
        parts.append(
            _finalize_ar_mask(
                utils.values_as_bool(sl),
                coords,
                min_size_gridpoints,
                time_dimension,
            )
        )
    if "lead_time" not in intersection.dims:
        return parts[0]
    out = xr.concat(parts, dim="lead_time")
    return out.assign_coords(lead_time=intersection.lead_time).transpose(
        *intersection.dims
    )


def atmospheric_river_mask(
    ivt: xr.DataArray,
    ivt_laplacian: xr.DataArray,
    laplacian_threshold: float = 2.5,
    ivt_threshold: float = 400,
    dilation_radius: int = 8,
    min_size_gridpoints: int = 500,
    time_dimension: str = "valid_time",
) -> xr.DataArray:
    """Calculate atmospheric river mask using IVT and Laplacian thresholds.

    The current implementation uses standard grid spacing of 0.25 degrees.
    Users must convert their data to this grid spacing before using this
    function, otherwise unexpected results may occur. Parameter defaults
    are based on Newell et al. 1992, Mo 2024, TempestExtremes v2.1
    criteria (Ullrich et al. 2021), and visual inspection of ERA5 outputs.

    Args:
        ivt: the input DataArray containing integrated_vapor_transport
        ivt_laplacian: the input DataArray containing
            integrated_vapor_transport_laplacian
        laplacian_threshold: the threshold for the Laplacian in kg/m^2/s^2
        ivt_threshold: the threshold for the IVT in kg/m/s
        dilation_radius: the radius for the dilation of the Laplacian in
            gridpoints
        min_size_gridpoints: the minimum size of the atmospheric river in
            gridpoints
        time_dimension: name of time dimension. Defaults to the EWB standard
            'valid_time'.

    Returns:
        The atmospheric river mask as a DataArray
    """

    has_high_laplacian = np.abs(ivt_laplacian) >= laplacian_threshold
    has_high_ivt = ivt >= ivt_threshold
    dilated_laplacian = xr.apply_ufunc(
        calc._binary_dilation_ufunc,
        has_high_laplacian,
        dilation_radius,
        input_core_dims=[["latitude", "longitude"], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[np.int8],
    )
    initial_intersection = dilated_laplacian.astype(bool) & has_high_ivt.astype(bool)
    return _finalize_ar_mask_by_lead(
        initial_intersection,
        min_size_gridpoints,
        time_dimension,
    )


def integrated_vapor_transport(
    specific_humidity: xr.DataArray,
    eastward_wind: xr.DataArray,
    northward_wind: xr.DataArray,
) -> xr.DataArray:
    """Compute integrated vapor transport from humidity and winds.

    Args:
        specific_humidity: a DataArray containing specific humidity
        eastward_wind: a DataArray containing eastward wind (u-component)
        northward_wind: a DataArray containing northward wind (v-component)

    Returns:
        Integrated vapor transport as a DataArray
    """

    q = specific_humidity.chunk({"level": -1})
    u = eastward_wind.chunk({"level": -1})
    v = northward_wind.chunk({"level": -1})
    levels_pa = q["level"] * 100
    ivt_magnitude = xr.apply_ufunc(
        _ivt_fused_kernel,
        u,
        v,
        q,
        levels_pa,
        calc.g0,
        input_core_dims=[["level"], ["level"], ["level"], ["level"], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    ivt_magnitude.name = "integrated_vapor_transport"
    return ivt_magnitude


def integrated_vapor_transport_laplacian(
    ivt: xr.DataArray, sigma: float = 3
) -> xr.DataArray:
    """Compute the blurred Laplacian of IVT.

    Args:
        ivt: integrated vapor transport DataArray
        sigma: Gaussian filter sigma for smoothing

    Returns:
        The blurred Laplacian of IVT
    """
    laplacian = xr.apply_ufunc(
        calc._compute_blurred_laplacian_ufunc,
        ivt,
        sigma,
        input_core_dims=[["latitude", "longitude"], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[float],
    )
    laplacian.name = "integrated_vapor_transport_blurred_laplacian"
    return laplacian


def build_atmospheric_river_mask_and_land_intersection(
    data: xr.Dataset,
    output_variables: list[str] | None = None,
) -> xr.Dataset:
    """Calculate atmospheric river mask and land intersection.

    Args:
        data: Dataset with atmospheric data. Must contain eastward_wind,
            northward_wind, specific_humidity, and level.
        output_variables: If set, only these names are kept in the result.
            IVT is still computed for the mask but dropped when omitted.

    Returns:
        Dataset containing atmospheric river mask and land intersection.
    """
    working = utils.stack_valid_time_pairs(data)
    ivt_data = integrated_vapor_transport(
        specific_humidity=working["specific_humidity"],
        eastward_wind=working["eastward_wind"],
        northward_wind=working["northward_wind"],
    )
    dilated = _dilated_high_laplacian(
        ivt_data, sigma=3, laplacian_threshold=2.5, dilation_radius=8
    )
    initial_intersection = dilated.astype(bool) & (ivt_data >= 400)
    ivt_data = utils.unstack_valid_time_pairs(ivt_data, like=data)
    initial_intersection = utils.unstack_valid_time_pairs(
        initial_intersection, like=data
    )
    ar_mask_result = _finalize_ar_mask_by_lead(
        initial_intersection,
        min_size_gridpoints=500,
        time_dimension="valid_time",
    )

    land_intersection = calc.find_land_intersection(ar_mask_result)
    land_intersection.name = "atmospheric_river_land_intersection"

    result: dict[str, xr.DataArray] = {
        "atmospheric_river_mask": ar_mask_result,
        "atmospheric_river_land_intersection": land_intersection,
        "integrated_vapor_transport": ivt_data,
    }
    if output_variables is not None:
        result = {k: v for k, v in result.items() if k in output_variables}
    return xr.Dataset(result)
