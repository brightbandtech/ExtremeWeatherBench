import numpy as np
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


def compute_ivt(
    specific_humidity: xr.DataArray,
    eastward_wind: xr.DataArray,
    northward_wind: xr.DataArray,
    levels: xr.DataArray,
) -> xr.DataArray:
    """Compute integrated vapor transport from eastward and northward winds.

    Args:
        data: dataset containing wind and humidity variables

    Returns:
        An integrated vapor transport dataarray
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
    ivt_magnitude = np.hypot(eastward_ivt, northward_ivt)
    ivt_magnitude.name = "integrated_vapor_transport"
    return ivt_magnitude


def _compute_laplacian_ufunc(data: xr.DataArray, sigma: float) -> xr.DataArray:
    """Compute Laplacian using scipy filters.

    Args:
        data: IVT data to compute the Laplacian of; data must be 2D
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
    # TODO(189): determine if numba can be used to speed up this computation
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


def _maybe_build_atmospheric_river_variables(data: xr.Dataset) -> xr.Dataset:
    """Maybe build atmospheric river variables in the dataset if not already available.

    Args:
        data: dataset containing wind and humidity variables; if the dataset does not
        contain the variables, they will be calculated and added to the dataset

    Returns:
        Dataset containing required atmospheric river variables of surface standard
        pressure, specific humidity, and adjusted broadcasted level.
    """
    if "surface_standard_pressure" not in data.data_vars:
        # Calculate orography from geopotential at the surface if not available
        if "orography" not in data.variables:
            orography = calc.orography(data)
        else:
            orography = data["orography"]
        data["surface_standard_pressure"] = calc.pressure_at_surface(orography)

    # Ensure specific humidity is available
    if "specific_humidity" not in data.data_vars:
        if "relative_humidity" not in data.data_vars:
            raise ValueError(
                "specific_humidity or relative_humidity must be in the dataset"
            )
        data["specific_humidity"] = calc.specific_humidity_from_relative_humidity(
            air_temperature=data["air_temperature"],
            relative_humidity=data["relative_humidity"],
            levels=data["level"],
        )

    # Broadcast level to match all dimensions including valid_time
    # Only include levels > 200 hPa (levels lower than 200 hPa have negligible
    # moisture)
    level_broadcasted = data["level"].broadcast_like(data)
    data["adjusted_level"] = xr.where(
        (level_broadcasted * 100 < data["surface_standard_pressure"])
        & (level_broadcasted > 200),
        level_broadcasted,
        np.nan,
    )
    return data


def build_atmospheric_river_mask_and_land_intersection(data: xr.Dataset) -> xr.Dataset:
    """Calculate atmospheric river mask and land intersection.

    Args:
        data: data with wind and humidity data. Must contain eastward_wind,
        northward_wind, specific_humidity, and level.

    Returns:
        Dataset containing atmospheric river mask and land intersection
    """
    if "integrated_vapor_transport" not in data.data_vars:
        # Ensure standard surface pressure is available. Standard surface pressure is
        # used to remove pressure levels below the surface

        # Build required atmospheric river variables if not already available
        data = _maybe_build_atmospheric_river_variables(data)

        # First compute IVT
        ivt_data = compute_ivt(
            specific_humidity=data["specific_humidity"],
            eastward_wind=data["eastward_wind"],
            northward_wind=data["northward_wind"],
            levels=data["adjusted_level"],
        )

    # Compute IVT Laplacian
    ivt_laplacian = compute_ivt_laplacian(ivt=ivt_data, sigma=3)

    # Convert IVT DataArray to Dataset and merge with Laplacian
    full_data = xr.merge([ivt_data, ivt_laplacian])

    # Compute AR mask with default parameters
    ar_mask_result = atmospheric_river_mask(data=full_data)

    # Compute land intersection
    land_intersection = calc.find_land_intersection(ar_mask_result)

    return xr.Dataset(
        {
            "atmospheric_river_mask": ar_mask_result,
            "atmospheric_river_land_intersection": land_intersection,
            "integrated_vapor_transport": ivt_data,
        }
    )
