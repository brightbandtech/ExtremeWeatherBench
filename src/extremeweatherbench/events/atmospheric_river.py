from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc


def _label_time_linked_tyx(mask_tyx: npt.NDArray[np.bool_]) -> npt.NDArray[np.int32]:
    """Label a (time, lat, lon) mask with 6-connectivity.

    2D connected components plus union-find across consecutive times.
    Matches ndimage.label with generate_binary_structure(3, 1).
    """
    n_t = mask_tyx.shape[0]
    labeled = np.empty(mask_tyx.shape, dtype=np.int32)
    n_feat = np.empty(n_t, dtype=np.int32)
    for t in range(n_t):
        labeled[t], n_feat[t] = ndimage.label(mask_tyx[t])
    total = int(n_feat.sum())
    if total == 0:
        return labeled

    glob = np.where(
        labeled > 0, labeled + (np.cumsum(n_feat) - n_feat)[:, None, None], 0
    ).astype(np.int32)
    parent = np.arange(total + 1, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for t in range(n_t - 1):
        both = (glob[t] > 0) & (glob[t + 1] > 0)
        if not both.any():
            continue
        pairs = np.unique(np.stack((glob[t][both], glob[t + 1][both]), axis=1), axis=0)
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

    remap = np.zeros(total + 1, dtype=np.int32)
    n = 0
    for i in range(1, total + 1):
        root = find(i)
        if remap[root] == 0:
            n += 1
            remap[root] = n
        remap[i] = remap[root]
    return remap[glob]


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
    moved = np.transpose(mask, order)
    n_t, *extra, n_y, n_x = moved.shape
    n_extra = int(np.prod(extra, initial=1))
    moved = moved.reshape(n_t, n_extra, n_y, n_x)
    out = np.empty_like(moved, dtype=np.int32)
    next_id = 1
    for i in range(n_extra):
        lab = _label_time_linked_tyx(moved[:, i])
        n_lab = int(lab.max())
        if n_lab:
            lab = np.where(lab, lab + (next_id - 1), 0)
            next_id += n_lab
        out[:, i] = lab
    return np.transpose(out.reshape(n_t, *extra, n_y, n_x), np.argsort(order))


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

    # Get all coordinates except level for the intersection DataArray
    coords_dict = {dim: ivt.coords[dim] for dim in ivt.dims if dim != "level"}

    # Create boolean masks for each condition
    has_high_laplacian, has_high_ivt = (
        np.abs(ivt_laplacian) >= laplacian_threshold,
        ivt >= ivt_threshold,
    )

    # For the Laplacian condition, we want to check if there's a value >=
    # laplacian_threshold within 8 gridpoints (0.25 degrees).
    # Apply binary dilation lazily via apply_ufunc; this keeps dilation in the Dask
    # graph after ivt/laplacian tasks
    dilated_laplacian = xr.apply_ufunc(
        calc._binary_dilation_ufunc,
        has_high_laplacian.chunk({time_dimension: 1, "latitude": -1, "longitude": -1}),
        dilation_radius,
        input_core_dims=[["latitude", "longitude"], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="parallelized",
        keep_attrs=True,
        output_dtypes=[np.int8],
    )

    # Combine conditions without tropical restriction initially
    initial_intersection = xr.where(dilated_laplacian & has_high_ivt, 1, 0)

    # Label 2D slices and merge across consecutive valid_time only.
    labeled_array = _label_objects_time_linked(
        np.asarray(initial_intersection, dtype=bool),
        list(initial_intersection.dims),
        time_dimension,
    )
    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

    # Filter by size first (excluding background label 0)
    size_valid_labels = unique_labels[
        np.where((label_counts >= min_size_gridpoints) & (unique_labels != 0))
    ]

    # Filter out tropical ARs: the mean latitude of each labeled feature
    # must be > 15 degrees N or S of the equator.
    latitudes = ivt.coords["latitude"].values
    lat_axis = list(coords_dict.keys()).index("latitude")
    lat_shape = [1] * labeled_array.ndim
    lat_shape[lat_axis] = len(latitudes)
    lat_grid = np.broadcast_to(latitudes.reshape(lat_shape), labeled_array.shape)
    if size_valid_labels.size == 0:
        valid_labels = size_valid_labels
    else:
        mean_lat = ndimage.mean(lat_grid, labels=labeled_array, index=size_valid_labels)
        valid_labels = size_valid_labels[np.abs(mean_lat) > 15]

    # Create final mask using valid features
    feature_mask = np.isin(labeled_array, valid_labels)

    # Final result with size threshold applied
    ar_mask = xr.DataArray(
        xr.where(feature_mask, 1, 0), coords=coords_dict, dims=coords_dict.keys()
    )
    ar_mask.name = "atmospheric_river_mask"
    return ar_mask


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

    # Compute IVT components using nantrapezoid_pressure_levels
    eastward_ivt = (
        calc.nantrapezoid_pressure_levels(
            da=eastward_wind * specific_humidity,
        )
        / calc.g0
    )

    northward_ivt = (
        calc.nantrapezoid_pressure_levels(
            da=northward_wind * specific_humidity,
        )
        / calc.g0
    )

    # Compute IVT using components
    ivt_magnitude = xr.ufuncs.hypot(eastward_ivt, northward_ivt)
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


def build_atmospheric_river_mask_and_land_intersection(data: xr.Dataset) -> xr.Dataset:
    """Calculate atmospheric river mask and land intersection.

    Args:
        data: Dataset with atmospheric data. Must contain eastward_wind,
            northward_wind, specific_humidity, and level.

    Returns:
        Dataset containing atmospheric river mask and land intersection.
    """
    # Generate IVT
    ivt_data = integrated_vapor_transport(
        specific_humidity=data["specific_humidity"],
        eastward_wind=data["eastward_wind"],
        northward_wind=data["northward_wind"],
    )

    # Compute IVT Laplacian
    ivt_laplacian = integrated_vapor_transport_laplacian(ivt=ivt_data, sigma=3)

    # Compute AR mask with default parameters
    ar_mask_result = atmospheric_river_mask(ivt=ivt_data, ivt_laplacian=ivt_laplacian)

    # Compute land intersection
    land_intersection = calc.find_land_intersection(ar_mask_result)

    return xr.Dataset(
        {
            "atmospheric_river_mask": ar_mask_result,
            "atmospheric_river_land_intersection": land_intersection,
            "integrated_vapor_transport": ivt_data,
        }
    )
