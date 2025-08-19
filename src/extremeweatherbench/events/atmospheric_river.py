import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage import filters


def blurred_laplacian(da: xr.DataArray, sigma: float = 3) -> xr.DataArray:
    """Calculate the Laplacian of a dataarray.

    Args:
        da: The dataarray to calculate the Laplacian from.
        sigma: The sigma of the Gaussian filter on the Laplacian.

    Returns:
        The blurred Laplacian of the input dataarray.
    """
    return gaussian_filter(filters.laplace(da), sigma=sigma)


def ar_mask(
    data: xr.Dataset, laplacian_threshold: float = 2.5, ivt_threshold: float = 400
) -> xr.DataArray:
    """Calculate the atmospheric river mask.

    Args:
        data: The input xarray dataset.
        laplacian_threshold: The threshold for the Laplacian in kg/m^2/s^2.
        ivt_threshold: The threshold for the IVT in kg/m/s.

    Returns:
        The atmospheric river mask.
    """
    coords_dict = {dim: data.coords[dim] for dim in data.dims if dim != "level"}
    # Create boolean masks for each condition
    has_high_laplacian = (
        np.abs(data["integrated_vapor_transport_laplacian"]) >= laplacian_threshold
    )
    has_high_ivt = data["integrated_vapor_transport"] >= ivt_threshold
    # For the laplacian condition, we want to check if there's a value >= 2.5 within 6
    # gridpoints
    struct = np.ones((17, 17))
    dilated_laplacian = binary_dilation(
        has_high_laplacian, structure=struct, axes=(-2, -1)
    )
    # Combine conditions without tropical restriction initially
    initial_intersection = xr.where(dilated_laplacian & has_high_ivt, 1, 0)

    # Label connected components and get their sizes
    labeled_array, _ = label(initial_intersection)
    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

    # Filter by size first (excluding background label 0)
    size_valid_labels = unique_labels[
        np.where((label_counts >= 500) & (unique_labels != 0))
    ]

    # Check centroids of each feature
    valid_features = []
    for label_num in size_valid_labels:
        # TODO: check if the centroid is outside the tropics (>25N/S)
        valid_features.append(label_num)

    # Create final mask using valid features
    feature_mask = np.isin(labeled_array, valid_features)

    # Final result with size threshold and centroid restriction applied
    ivt_laplacian_intersection = xr.DataArray(
        xr.where(feature_mask, 1, 0), coords=coords_dict, dims=coords_dict.keys()
    )
    return ivt_laplacian_intersection
