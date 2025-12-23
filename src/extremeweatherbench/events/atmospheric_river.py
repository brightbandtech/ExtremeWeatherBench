import numpy as np
import xarray as xr
from scipy import ndimage
from skimage import filters

from extremeweatherbench import calc


def atmospheric_river_mask(
    ivt: xr.DataArray,
    ivt_laplacian: xr.DataArray,
    laplacian_threshold: float = 2.5,
    ivt_threshold: float = 400,
    dilation_radius: int = 8,
    min_size_gridpoints: int = 500,
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

    Returns:
        The atmospheric river mask as a DataArray
    """

    # Get all coordinates except level for the intersection DataArray
    coords_dict = {dim: ivt.coords[dim] for dim in ivt.dims if dim != "level"}

    # Create boolean masks for each condition
    has_high_laplacian = np.abs(ivt_laplacian) >= laplacian_threshold
    has_high_ivt = ivt >= ivt_threshold

    # For the Laplacian condition, we want to check if there's a value >=
    # laplacian_threshold within 8 gridpoints (0.25 degrees)
    dilation_radius = dilation_radius * 2 + 1
    struct = np.ones((dilation_radius, dilation_radius))
    dilated_laplacian = ndimage.binary_dilation(
        has_high_laplacian, structure=struct, axes=(-2, -1)
    )

    # Combine conditions without tropical restriction initially
    initial_intersection = xr.where(dilated_laplacian & has_high_ivt, 1, 0)

    # Label connected components and get their sizes
    labeled_array, _ = ndimage.label(initial_intersection)
    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

    # Filter by size first (excluding background label 0)
    size_valid_labels = unique_labels[
        np.where((label_counts >= min_size_gridpoints) & (unique_labels != 0))
    ]

    # Collect all size-valid feature labels
    valid_features = []
    for label_num in size_valid_labels:
        valid_features.append(label_num)

    # Create final mask using valid features
    feature_mask = np.isin(labeled_array, valid_features)

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
    levels: xr.DataArray,
) -> xr.DataArray:
    """Compute integrated vapor transport from humidity and winds.

    Args:
        specific_humidity: a DataArray containing specific humidity
        eastward_wind: a DataArray containing eastward wind (u-component)
        northward_wind: a DataArray containing northward wind (v-component)
        levels: a DataArray containing pressure levels

    Returns:
        Integrated vapor transport as a DataArray
    """

    # Get required coordinates excluding level dimension
    coords_dict = {
        dim: specific_humidity.coords[dim]
        for dim in specific_humidity.dims
        if dim != "level"
    }
    level_axis = list(specific_humidity.dims).index("level")

    # Compute IVT components using nantrapezoid
    eastward_ivt_arr = (
        calc.nantrapezoid(
            eastward_wind * specific_humidity,
            x=levels * 100,  # convert to Pa
            axis=level_axis,
        )
        / calc.g0
    )
    eastward_ivt = xr.DataArray(
        eastward_ivt_arr, coords=coords_dict, dims=coords_dict.keys()
    )

    northward_ivt_arr = (
        calc.nantrapezoid(
            northward_wind * specific_humidity,
            x=levels * 100,  # convert to Pa
            axis=level_axis,
        )
        / calc.g0
    )
    northward_ivt = xr.DataArray(
        northward_ivt_arr, coords=coords_dict, dims=coords_dict.keys()
    )

    # Compute IVT using components
    ivt_magnitude = xr.ufuncs.hypot(eastward_ivt, northward_ivt)
    ivt_magnitude.name = "integrated_vapor_transport"
    return ivt_magnitude


def _compute_blurred_laplacian_ufunc(data: xr.DataArray, sigma: float) -> xr.DataArray:
    """Compute blurred Laplacian using scipy filters.

    Args:
        data: IVT data to compute the blurred Laplacian of; data must be 2D
        sigma: the standard deviation for the Gaussian filter

    Returns:
        The blurred Laplacian of IVT
    """
    laplace_data = filters.laplace(data)
    return ndimage.gaussian_filter(laplace_data, sigma=sigma)


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
    # TODO(189): determine if numba can be used to speed up this computation
    laplacian = xr.apply_ufunc(
        _compute_blurred_laplacian_ufunc,
        ivt,
        sigma,
        input_core_dims=[["latitude", "longitude"], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[float],
    )

    # Add name to the dataarray
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
        levels=data["level"],
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
