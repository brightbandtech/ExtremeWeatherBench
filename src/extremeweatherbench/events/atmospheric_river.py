import numpy as np
import regionmask
import scores.categorical as cat
import xarray as xr
from scipy import ndimage
from skimage import filters

from extremeweatherbench import calc


def atmospheric_river_mask(
    data: xr.Dataset,
    laplacian_threshold: float = 2.5,
    ivt_threshold: float = 400,
    dilation_radius: int = 8,
    min_size_gridpoints: int = 500,
) -> xr.DataArray:
    """Calculate the atmospheric river mask using thresholds for IVT and its Laplacian.

    The incoming dataset must contain the variables integrated_vapor_transport
    and integrated_vapor_transport_laplacian. The current implementation uses
    standard grid spacing of 0.25 degrees (same as ERA5); users must convert their
    data to this grid spacing before using this function as of v1.0.0.

    Parameter defaults for the mask are based on Newell et al. 1992, Mo 2024,
    TempestExtremes v2.1 criteria (Ullrich et al. 2021), and visual inspection of
    ERA5 outputs.

    Args:
        data: the input dataset containing integrated_vapor_transport as a variable and
        integrated_vapor_transport_laplacian as a variable.
        laplacian_threshold: the threshold for the Laplacian in kg/m^2/s^2
        ivt_threshold: the threshold for the IVT in kg/m/s
        dilation_radius: the radius for the dilation of the Laplacian in gridpoints
        min_size_gridpoints: the minimum size of the atmospheric river in gridpoints
    Returns:
        The atmospheric river mask
    """
    # Get all coordinates except level for the intersection DataArray
    coords_dict = {dim: data.coords[dim] for dim in data.dims if dim != "level"}
    # Create boolean masks for each condition
    has_high_laplacian = (
        np.abs(data["integrated_vapor_transport_laplacian"]) >= laplacian_threshold
    )
    has_high_ivt = data["integrated_vapor_transport"] >= ivt_threshold
    # For the Laplacian condition, we want to check if there's a value >= 2.5 within 8
    # gridpoints (0.25 degrees)
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

    # Check centroids of each feature
    valid_features = []
    for label_num in size_valid_labels:
        valid_features.append(label_num)

    # Create final mask using valid features
    feature_mask = np.isin(labeled_array, valid_features)

    # Final result with size threshold and centroid restriction applied
    ivt_laplacian_intersection = xr.DataArray(
        xr.where(feature_mask, 1, 0), coords=coords_dict, dims=coords_dict.keys()
    )
    ivt_laplacian_intersection.name = "atmospheric_river_mask"
    return ivt_laplacian_intersection


def compute_ivt(data: xr.Dataset) -> xr.DataArray:
    """Compute integrated vapor transport from eastward and northward winds.

    Args:
        data: dataset containing wind and humidity variables

    Returns:
        An integrated vapor transport dataarray
    """
    # Return if integrated_vapor_transport is already in the dataset
    if "integrated_vapor_transport" in data.data_vars:
        return data["integrated_vapor_transport"]

    # Get required coordinates excluding level dimension
    coords_dict = {dim: data.coords[dim] for dim in data.dims if dim != "level"}

    # Ensure standard surface pressure is available. Standard surface pressure is
    # used to remove pressure levels below the surface
    if "surface_standard_pressure" not in data.data_vars:
        # Calculate orography from geopotential at the surface if not available
        if "orography" not in data.variables:
            orography = calc.orography(data)
        else:
            orography = data["orography"]
        data["surface_standard_pressure"] = calc.calculate_pressure_at_surface(
            orography
        )

    # Ensure specific humidity is available
    if "specific_humidity" not in data.data_vars:
        if "relative_humidity" not in data.data_vars:
            raise ValueError(
                "specific_humidity or relative_humidity must be in the dataset"
            )
        data["specific_humidity"] = (
            calc.compute_specific_humidity_from_relative_humidity(data)
        )

    # Find the level axis
    level_axis = list(data.dims).index("level")

    # Transform all data to the same shape
    data_broadcast, level_broadcast, sfc_pres_broadcast = xr.broadcast(
        data, data["level"], data["surface_standard_pressure"]
    )

    # Only include levels > 200 hPa (levels lower than 200 hPa have negligible moisture)
    data_broadcast["adjusted_level"] = xr.where(
        (level_broadcast * 100 < sfc_pres_broadcast) & (data["level"] > 200),
        data["level"],
        np.nan,
    )

    # Compute IVT components using nantrapezoid
    eastward_ivt = xr.DataArray(
        calc.nantrapezoid(
            data_broadcast["eastward_wind"] * data_broadcast["specific_humidity"],
            x=data_broadcast.adjusted_level * 100,  # convert to Pa
            axis=level_axis,
        )
        / calc.g0,
        coords=coords_dict,
        dims=coords_dict.keys(),
    )

    northward_ivt = xr.DataArray(
        calc.nantrapezoid(
            data_broadcast["northward_wind"] * data_broadcast["specific_humidity"],
            x=data_broadcast.adjusted_level * 100,  # convert to Pa
            axis=level_axis,
        )
        / calc.g0,
        coords=coords_dict,
        dims=coords_dict.keys(),
    )

    # Compute IVT
    ivt_magnitude = np.hypot(eastward_ivt, northward_ivt)

    return ivt_magnitude


def _compute_laplacian_ufunc(data, sigma):
    """Compute Laplacian using scipy filters.

    Args:
        data: IVT data to compute the Laplacian of
        sigma: the standard deviation for the Gaussian filter

    Returns:
        The Laplacian of IVT
    """
    return ndimage.gaussian_filter(filters.laplace(data), sigma=sigma)


def compute_ivt_laplacian(ivt: xr.DataArray, sigma: float = 3) -> xr.DataArray:
    """Compute the Laplacian of IVT.

    Args:
        ivt: integrated vapor transport DataArray
        sigma: Gaussian filter sigma for smoothing

    Returns:
        The Laplacian of IVT
    """
    # TODO: determine if numba can be used to speed up this computation
    laplacian = xr.apply_ufunc(
        _compute_laplacian_ufunc,
        ivt,
        sigma,
        input_core_dims=[["latitude", "longitude"], []],
        output_core_dims=[["latitude", "longitude"]],
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[float],
    )

    # Add name to the dataarray
    laplacian.name = "integrated_vapor_transport_laplacian"
    return laplacian


def find_land_intersection(atmospheric_river_mask: xr.DataArray) -> xr.DataArray:
    """Find points where an atmospheric river mask intersects with land.

    Args:
        atmospheric_river_mask: a boolean mask of AR locations

    Returns:
        a mask of points where AR overlaps with land
    """

    mask_parent = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        atmospheric_river_mask.longitude, atmospheric_river_mask.latitude
    )
    mask = mask_parent.where(np.isnan(mask_parent), 1).where(mask_parent == 0, 0)
    contingency_manager = cat.BinaryContingencyManager(atmospheric_river_mask, mask)
    # return the true positive mask, where AR is true and land is true
    land_intersection = contingency_manager.tp
    land_intersection.name = "atmospheric_river_land_intersection"
    return land_intersection


def build_mask_and_land_intersection(data: xr.Dataset) -> xr.Dataset:
    """Calculate atmospheric river mask and land intersection.

    Args:
        data: data with wind and humidity data. Must contain eastward_wind,
        northward_wind, specific_humidity, and level.

    Returns:
        Dataset containing atmospheric river mask and land intersection
    """
    # First compute IVT
    ivt_data = compute_ivt(data)

    # Compute IVT Laplacian
    ivt_laplacian = compute_ivt_laplacian(ivt_data)

    # Convert IVT DataArray to Dataset and merge with Laplacian
    ivt_dataset = ivt_data.to_dataset(name="integrated_vapor_transport")
    full_data = xr.merge([ivt_dataset, ivt_laplacian])

    # Compute AR mask with default parameters
    ar_mask_result = atmospheric_river_mask(full_data)

    # Compute land intersection
    land_intersection = find_land_intersection(ar_mask_result)

    return xr.Dataset(
        {
            "atmospheric_river_mask": ar_mask_result,
            "atmospheric_river_land_intersection": land_intersection,
        }
    )
