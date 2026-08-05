from typing import Optional

import numpy as np
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc, utils

#: Features whose mean latitude sits equatorward of this are tropical moisture
#: plumes rather than atmospheric rivers.
MIN_ABS_LATITUDE_DEGREES = 15.0

#: Above this many size-passing features, one weighted pass over the volume
#: beats scanning it once per feature. Below it the scan is cheaper, and the
#: size filter keeps the count in single digits on real fields.
_LATITUDE_SCAN_MAX_FEATURES = 8


def _label_ar_features_ufunc(
    intersection: np.ndarray,
    latitudes: np.ndarray,
    min_size_gridpoints: int,
    min_abs_latitude: float,
) -> np.ndarray:
    """Label connected features in each (time, lat, lon) volume and filter them.

    Operates on one volume at a time so that connectivity never crosses a
    leading batch dimension. Callers put the axis features should connect
    along (valid_time for analyses, lead_time within a single forecast) in
    the time position, and anything that must stay independent in the batch
    dimensions.

    Sizes come from a bincount over the labels, so the size filter costs one
    pass however many features there are. Mean latitude is then needed only
    for the few features that survive that filter, so it is measured by
    scanning for those, falling back to a single weighted pass once enough
    survive that the scans would cost more.

    Args:
        intersection: Array of shape (..., time, lat, lon), nonzero where the
            IVT and Laplacian criteria are both met.
        latitudes: Latitude in degrees for each row of the grid.
        min_size_gridpoints: Smallest feature retained, in gridpoints.
        min_abs_latitude: Smallest absolute mean latitude retained, in degrees.

    Returns:
        Array of the same shape holding 1 inside retained features, 0 outside.
    """
    n_time, n_lat, n_lon = intersection.shape[-3:]
    batch_shape = intersection.shape[:-3]
    volumes = intersection.reshape((-1, n_time, n_lat, n_lon))
    labeled_output = np.zeros(volumes.shape, dtype=np.int64)

    # Latitude varies only down the grid, so one time slice of weights is
    # enough to accumulate per-label sums a step at a time. Broadcasting it
    # over the whole volume instead would allocate a float copy of the data.
    latitude_values = np.asarray(latitudes, dtype=np.float64)
    latitude_per_cell = np.broadcast_to(
        latitude_values.reshape(n_lat, 1), (n_lat, n_lon)
    ).ravel()
    latitude_volume = np.broadcast_to(
        latitude_values.reshape(1, n_lat, 1), (n_time, n_lat, n_lon)
    )

    for index, volume in enumerate(volumes):
        labels, n_features = ndimage.label(volume)
        if n_features == 0:
            continue

        sizes = np.bincount(labels.ravel(), minlength=n_features + 1)
        sizes[0] = 0  # label 0 is background
        candidates = np.flatnonzero(sizes >= min_size_gridpoints)
        if candidates.size == 0:
            continue

        # Only reached when something survived the size filter, which keeps
        # the latitude pass off the common path where nothing does.
        if candidates.size > _LATITUDE_SCAN_MAX_FEATURES:
            latitude_sums = np.zeros(n_features + 1, dtype=np.float64)
            for time_step in labels:
                latitude_sums += np.bincount(
                    time_step.ravel(),
                    weights=latitude_per_cell,
                    minlength=n_features + 1,
                )
            mean_abs_latitude = np.abs(latitude_sums[candidates] / sizes[candidates])
        else:
            mean_abs_latitude = np.array(
                [abs(latitude_volume[labels == label].mean()) for label in candidates]
            )

        retained = candidates[mean_abs_latitude > min_abs_latitude]
        if retained.size == 0:
            continue

        keep = np.zeros(n_features + 1, dtype=np.int64)
        keep[retained] = 1
        labeled_output[index] = keep[labels]

    return labeled_output.reshape(batch_shape + (n_time, n_lat, n_lon))


def _resolve_connectivity_dimension(
    data: xr.DataArray, time_dimension: Optional[str]
) -> str:
    """Pick the single axis along which AR features may connect."""
    if time_dimension is not None:
        return time_dimension
    if "lead_time" in data.dims:
        return "lead_time"
    if "valid_time" in data.dims:
        return "valid_time"
    raise ValueError(
        "atmospheric_river_mask needs a lead_time or valid_time dimension to "
        f"connect features along; got dimensions {tuple(data.dims)}"
    )


def atmospheric_river_mask(
    ivt: xr.DataArray,
    ivt_laplacian: xr.DataArray,
    laplacian_threshold: float = 2.5,
    ivt_threshold: float = 400,
    dilation_radius: int = 8,
    min_size_gridpoints: int = 500,
    time_dimension: Optional[str] = None,
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
        time_dimension: name of the axis features connect along. Defaults to
            lead_time when present, otherwise the EWB standard 'valid_time'.

    Returns:
        The atmospheric river mask as a DataArray
    """
    has_high_laplacian = np.abs(ivt_laplacian) >= laplacian_threshold
    has_high_ivt = ivt >= ivt_threshold

    # Dilation answers "is there a value over the threshold within
    # dilation_radius gridpoints", so it stays 2-D per time step and runs on
    # whatever time chunking the caller supplied. Only the labeling needs the
    # time axis gathered, so any rechunk waits until then.
    if has_high_laplacian.chunks is not None:
        has_high_laplacian = has_high_laplacian.chunk({"latitude": -1, "longitude": -1})

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

    intersection = dilated_laplacian & has_high_ivt
    latitudes = ivt.coords["latitude"].values
    output_dims = [dim for dim in ivt.dims if dim != "level"]

    # A forecast carries both lead_time and valid_time, and each
    # initialization is an independent realization whose features must not
    # merge with another's. init_time/lead_time is the layout that separates
    # them, so relabel there and convert back. Only this one boolean field
    # makes the trip, rather than the two float fields it came from.
    if "lead_time" in intersection.dims and "valid_time" in intersection.dims:
        original_valid_time = intersection["valid_time"]
        # Filling with 0 rather than NaN keeps both legs of the round trip in
        # the mask's own dtype; NaN would widen an int8 field to float64.
        by_init = utils.convert_valid_time_to_init_time(
            intersection.astype(np.int8), fill_value=0
        )

        labeled = _label_ar_features(
            by_init, "lead_time", latitudes, min_size_gridpoints
        )

        ar_mask = utils.convert_init_time_to_valid_time(
            labeled.to_dataset(name="atmospheric_river_mask"), fill_value=0
        )["atmospheric_river_mask"]
        # The round trip spans every init/lead combination, which reaches
        # further than the valid times asked for.
        ar_mask = ar_mask.sel(valid_time=original_valid_time)
    else:
        connect_dim = _resolve_connectivity_dimension(intersection, time_dimension)
        ar_mask = _label_ar_features(
            intersection, connect_dim, latitudes, min_size_gridpoints
        )

    ar_mask = ar_mask.transpose(*output_dims)
    ar_mask.name = "atmospheric_river_mask"
    return ar_mask


def _label_ar_features(
    intersection: xr.DataArray,
    connect_dim: str,
    latitudes: np.ndarray,
    min_size_gridpoints: int,
) -> xr.DataArray:
    """Label and filter features, connecting only along ``connect_dim``."""
    volume_dims = [connect_dim, "latitude", "longitude"]
    if intersection.chunks is not None:
        intersection = intersection.chunk({dim: -1 for dim in volume_dims})

    # The latitudes and thresholds go through kwargs rather than as operands,
    # so apply_ufunc does not broadcast and align them against the data.
    return xr.apply_ufunc(
        _label_ar_features_ufunc,
        intersection,
        kwargs={
            "latitudes": latitudes,
            "min_size_gridpoints": min_size_gridpoints,
            "min_abs_latitude": MIN_ABS_LATITUDE_DEGREES,
        },
        input_core_dims=[volume_dims],
        output_core_dims=[volume_dims],
        dask="parallelized",
        output_dtypes=[np.int64],
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
