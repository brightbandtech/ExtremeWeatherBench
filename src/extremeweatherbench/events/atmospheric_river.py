import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage import filters

from extremeweatherbench import calc


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


def compute_ivt(data: xr.Dataset) -> xr.Dataset:
    """Compute Integrated Vapor Transport using xr.apply_ufunc.

    Args:
        data: Dataset containing wind and humidity variables

    Returns:
        Dataset with IVT components and magnitude
    """
    if "integrated_vapor_transport" in data.data_vars:
        return data

    # Get required coordinates excluding level dimension
    coords_dict = {dim: data.coords[dim] for dim in data.dims if dim != "level"}

    # Ensure surface pressure is available
    if "surface_standard_pressure" not in data.data_vars:
        data["surface_standard_pressure"] = calc.calculate_pressure_at_surface(
            calc.orography(data)
        )

    if "specific_humidity" not in data.data_vars:
        data["specific_humidity"] = _compute_specific_humidity_from_relative_humidity(
            data
        )

    # Find the level axis
    level_axis = list(data.dims).index("level")

    # Use original approach with broadcasting for level filtering
    data_broadcast, level_broadcast, sfc_pres_broadcast = xr.broadcast(
        data, data["level"], data["surface_standard_pressure"]
    )

    # Only include levels > 200 hPa (levels below 200 hPa)
    data_broadcast["adjusted_level"] = xr.where(
        (level_broadcast * 100 < sfc_pres_broadcast) & (data["level"] > 200),
        data["level"],
        np.nan,
    )

    # Compute IVT components using original method but with DataArrays
    eastward_ivt = xr.DataArray(
        calc.nantrapezoid(
            data_broadcast["eastward_wind"] * data_broadcast["specific_humidity"],
            x=data_broadcast.adjusted_level * 100,
            axis=level_axis,
        )
        / 9.80665,
        coords=coords_dict,
        dims=coords_dict.keys(),
    )

    northward_ivt = xr.DataArray(
        calc.nantrapezoid(
            data_broadcast["northward_wind"] * data_broadcast["specific_humidity"],
            x=data_broadcast.adjusted_level * 100,
            axis=level_axis,
        )
        / 9.80665,
        coords=coords_dict,
        dims=coords_dict.keys(),
    )

    # Compute IVT magnitude using apply_ufunc for the hypot calculation
    ivt_magnitude = xr.apply_ufunc(
        np.hypot,
        eastward_ivt,
        northward_ivt,
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[float],
    )

    return xr.Dataset(
        {
            "vertical_integral_of_eastward_water_vapour_flux": eastward_ivt,
            "vertical_integral_of_northward_water_vapour_flux": northward_ivt,
            "integrated_vapor_transport": ivt_magnitude,
        }
    )


def _compute_specific_humidity_from_relative_humidity(data: xr.Dataset) -> xr.DataArray:
    """Compute specific humidity from relative humidity and air temperature."""

    # Compute saturation mixing ratio; air temperature must be in Kelvin;
    # level must be in hPa
    sat_mixing_ratio = calc.saturation_mixing_ratio(
        data["level"], data["air_temperature"] - 273.15
    )

    # Calculate specific humidity using saturation mixing ratio, epsilon,
    # and relative humidity
    mixing_ratio = (
        calc.epsilon
        * sat_mixing_ratio
        * data["relative_humidity"]
        / (calc.epsilon + sat_mixing_ratio * (1 - data["relative_humidity"]))
    )
    specific_humidity = mixing_ratio / (1 + mixing_ratio)
    return specific_humidity


def _compute_laplacian_ufunc(data, sigma):
    """Compute Laplacian using scipy filters via apply_ufunc."""
    return gaussian_filter(filters.laplace(data), sigma=sigma)


def compute_ivt_laplacian(ivt: xr.DataArray, sigma: float = 3) -> xr.DataArray:
    """Compute the Laplacian of IVT using xr.apply_ufunc.

    Args:
        ivt: Integrated vapor transport DataArray
        sigma: Gaussian filter sigma for smoothing

    Returns:
        Laplacian of IVT
    """
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

    laplacian.name = "integrated_vapor_transport_laplacian"
    return laplacian


def find_land_intersection(ar_mask: xr.DataArray) -> xr.DataArray:
    """
    Finds points where an atmospheric river mask intersects with land.

    Args:
        ar_mask: xarray DataArray containing boolean mask of AR locations

    Returns:
        xarray DataArray containing only the points where AR overlaps with land
    """
    import regionmask
    import scores.categorical as cat

    mask_parent = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        ar_mask.longitude, ar_mask.latitude
    )
    mask = mask_parent.where(np.isnan(mask_parent), 1).where(mask_parent == 0, 0)
    contingency_manager = cat.BinaryContingencyManager(mask, ar_mask)
    return contingency_manager


def compute_atmospheric_river_mask_ufunc(data: xr.Dataset) -> xr.Dataset:
    """Compute atmospheric river mask using xr.apply_ufunc approach.

    Args:
        data: Dataset containing wind and humidity data

    Returns:
        Dataset containing atmospheric river mask and land intersection
    """
    # First compute IVT using apply_ufunc
    ivt_data = compute_ivt(data)

    # Compute IVT Laplacian using apply_ufunc
    ivt_laplacian = compute_ivt_laplacian(ivt_data["integrated_vapor_transport"])

    # Merge IVT data with Laplacian
    full_data = xr.merge([ivt_data, ivt_laplacian])

    # Compute AR mask using existing function
    ar_mask_result = ar_mask(full_data)

    # Compute land intersection
    land_intersection = find_land_intersection(ar_mask_result)

    return xr.Dataset(
        {
            "atmospheric_river_mask": ar_mask_result,
            "atmospheric_river_land_intersection": land_intersection,
        }
    )
