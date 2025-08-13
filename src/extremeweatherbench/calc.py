import dataclasses
from collections import namedtuple
from itertools import product
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import spatial
from skimage import measure
from skimage.feature import peak_local_max

from extremeweatherbench import inputs

Location = namedtuple("Location", ["latitude", "longitude"])


# TODO: determine if this is needed or can be removed for the refactor
@dataclasses.dataclass
class TC:
    id: int
    valid_time: pd.Timestamp
    coordinate: Location
    vmax: float
    slp: float


# TODO: determine if this is needed or can be removed for the refactor
@dataclasses.dataclass
class TCTracks:
    id: int
    track: list[TC]

    def plot(self, ax: plt.Axes, *args, **kwargs):
        """
        Plot the TCTrack line on the given axes.
        """
        # Extract lat/lon coordinates from track
        lats = [tc.coordinate.latitude for tc in self.track]
        lons = [tc.coordinate.longitude for tc in self.track]

        # Plot the track line
        line = ax.plot(lons, lats, *args, **kwargs)
        return line

    @property
    def valid_times(self) -> list[pd.Timestamp]:
        return [tc.valid_time for tc in self.track]

    @property
    def lats(self):
        return [tc.coordinate.latitude for tc in self.track]

    @property
    def lons(self):
        return [tc.coordinate.longitude for tc in self.track]

    @property
    def vmax(self) -> list[float]:
        return [tc.vmax for tc in self.track]

    @property
    def slp(self) -> list[float]:
        return [tc.slp for tc in self.track]


def find_furthest_contour_from_point(
    contour: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find the two points in a contour that are furthest apart.

    From
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-in-plot-with-many-points-in-python

    Args:
        contour: The contour to find the furthest point from.
        point: The point to find the furthest point from.

    Returns:
        The furthest point from the contour as a tuple of x,y coordinates.
    """

    # Calculate distances from point to all points in contour
    distances = spatial.distance.cdist([point], contour)[0]
    # Find index of point with maximum distance
    furthest_idx = np.argmax(distances)

    # Get the furthest point
    i = furthest_idx
    return contour[i]


def convert_from_cartesian_to_latlon(
    input_point: np.ndarray, ds_mapping: xr.Dataset
) -> tuple[float, float]:
    """Convert a point from the cartesian coordinate system to the lat/lon coordinate system.

    Args:
        input_point: The point to convert, represented as a tuple (y, x) in the cartesian coordinate system.
        ds_mapping: The dataset containing the latitude and longitude coordinates.

    Returns:
        The point in the lat/lon coordinate system, represented as a tuple (latitude, longitude) in degrees.
    """
    return (
        ds_mapping.isel(
            latitude=int(input_point[0]), longitude=int(input_point[1])
        ).latitude.values,
        ds_mapping.isel(
            latitude=int(input_point[0]), longitude=int(input_point[1])
        ).longitude.values,
    )


def calculate_haversine_degree_distance(
    input_a: Sequence[float], input_b: Sequence[float]
) -> float:
    """Calculate the great-circle distance between two points on the Earth's surface.

    Args:
        input_a: The first point, represented as an ndarray of shape (2,n) in degrees lat/lon.
        input_b: The second point(s), represented as an ndarray of shape (2,n) in degrees lat/lon.

    Returns:
        The great-circle distance between the two points in kilometers.
    """
    # Convert to radians for calculations
    lat1 = np.radians(input_a[0])
    lon1 = np.radians(input_a[1])
    lat2 = np.radians(input_b[0])
    lon2 = np.radians(input_b[1])

    # Haversine formula for great circle distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = np.degrees(c)  # Convert back to degrees
    return distance


def create_great_circle_mask(
    ds: xr.Dataset, latlon_point: tuple[float, float], radius_degrees: float
) -> xr.DataArray:
    """Create a circular mask based on great circle distance for an xarray dataset.

    Args:
        ds: Dataset with 'latitude' and 'longitude' coordinates.
        latlon_point: Tuple containing (latitude, longitude) of the center point.
        radius_degrees: Radius in degrees of great circle distance.

    Returns:
        Boolean mask where True indicates points within the radius.
    """

    distance = calculate_haversine_degree_distance(
        latlon_point, (ds.latitude, ds.longitude)
    )
    # Create mask as xarray DataArray
    mask = distance <= radius_degrees

    return mask


def find_contours_from_point_specified_field(
    field: xr.DataArray, point: tuple[float, float], level: float
) -> Sequence[tuple[float, float]]:
    """Find the contours from a point for a specified field.

    Args:
        field: The field to find the contours from.
        point: The point at which the field is subtracted from to find the anomaly contours.
        level: The anomaly level to find the contours at.

    Returns:
        The contours as a list of tuples of latitude and longitude.
    """
    field_at_point = field - field.isel(latitude=point[0], longitude=point[1])
    contours = measure.find_contours(
        field_at_point.values, level=level, positive_orientation="high"
    )
    return contours


def find_valid_contour_from_point(
    contour: Sequence[tuple[float, float]],
    point: tuple[float, float],
    ds_mapping: xr.Dataset,
) -> tuple[float, float]:
    """Find the great circle distance from a point to a contour.

    Args:
        contour: The contour to find the great circle distance to.
        point: The point to find the great circle distance to.
        ds_mapping: The xarray dataset to map the point to.

    Returns:
        The great circle distance from the point to the contour.
    """
    gc_distance_point = find_furthest_contour_from_point(contour, point)
    gc_distance_point_latlon = convert_from_cartesian_to_latlon(
        gc_distance_point, ds_mapping
    )
    point_latlon = convert_from_cartesian_to_latlon(point, ds_mapping)
    gc_distance_contour_distance = calculate_haversine_degree_distance(
        gc_distance_point_latlon, point_latlon
    )
    return gc_distance_contour_distance


def find_valid_candidates(
    slp_contours: Sequence[Sequence[tuple[float, float]]],
    dz_contours: Sequence[Sequence[tuple[float, float]]],
    point: tuple[float, float],
    ds_mapping: xr.Dataset,
    time_counter: int,
    max_gc_distance_slp_contour: float = 5.5,
    max_gc_distance_dz_contour: float = 6.5,
    orography_filter_threshold: float = 150,
) -> dict[pd.Timestamp, tuple[float, float]]:
    """Find valid candidate coordinate for a TC.

    Defaults use the TempestExtremes criteria for TC track detection, where the slp must increase by at least n hPa
    within 5.5 great circle degrees from the center point, geopotential thickness must increase by at least m meters
    within 6.5 great circle degrees from the center point, and the terrain must be less than 150m.

    Args:
        slp_contours: List of SLP contours.
        dz_contours: List of DZ contours.
        point: The point to find the valid candidate for.
        ds_mapping: The xarray dataset to map the point to.
        time_counter: The time counter.
        max_gc_distance_slp_contour: The maximum great circle distance for the SLP contour.
        max_gc_distance_dz_contour: The maximum great circle distance for the DZ contour.
        orography_filter_threshold: The threshold for the orography filter.

    Returns:
        The valid candidate coordinate as a dict of timestamp and tuple of latitude and longitude.
    """
    latitude = convert_from_cartesian_to_latlon(point, ds_mapping)[0]
    longitude = convert_from_cartesian_to_latlon(point, ds_mapping)[1]
    orography_filter = (
        ds_mapping["orography"]
        .sel(latitude=latitude, longitude=longitude, method="nearest")
        .min()
        .values
        < orography_filter_threshold
        if time_counter < 8
        else True
    )
    latitude_filter = abs(latitude) < 50 if time_counter < 10 else True
    for slp_contour, dz_contour in product(slp_contours, dz_contours):
        if (
            # Only check closed contours
            all(np.isclose(slp_contour[-1], slp_contour[0]))
            and all(np.isclose(dz_contour[-1], dz_contour[0]))
            and
            # Check if point is inside both contour types
            measure.points_in_poly([[point[0], point[1]]], slp_contour)[0]
            and measure.points_in_poly([[point[0], point[1]]], dz_contour)[0]
            and
            # Check if the contour is within the max great circle distance
            find_valid_contour_from_point(slp_contour, point, ds_mapping)
            < max_gc_distance_slp_contour
            and find_valid_contour_from_point(dz_contour, point, ds_mapping)
            < max_gc_distance_dz_contour
            and orography_filter
            and latitude_filter
        ):
            valid_candidate = Location(latitude=latitude, longitude=longitude)
            return valid_candidate
    return None


def create_tctracks(tcs: list[TC]) -> list[TCTracks]:
    """Create a TCTrack from a list of TCs.

    Groups TCs into tracks if they are:
    1. At unique timesteps
    2. Within 8 great circle degrees of the previous timestep
    3. Within 54 hours of the previous timestep

    Parameters from TempestExtremes: https://gmd.copernicus.org/articles/14/5023/2021/#section3
    Args:
        tcs: List of TC objects.

    Returns:
        A list of TCTracks objects.
    """
    # Sort TCs by time
    tcs = sorted(tcs, key=lambda x: x.valid_time)

    tracks = []
    used_tc_ids = set()  # Track TCs by their IDs instead of objects
    track_id = 0
    for tc in tcs:
        if tc.id in used_tc_ids:
            continue

        # Start a new track with this TC
        current_track = [tc]
        used_tc_ids.add(tc.id)

        # Find subsequent TCs that could be part of this track
        last_tc = tc
        for next_tc in tcs:
            if next_tc.id in used_tc_ids:
                continue

            # Check if within 54 hours
            time_diff = (next_tc.valid_time - last_tc.valid_time).total_seconds() / 3600
            if time_diff > 54:
                continue

            # Check if within 8 degrees
            distance = calculate_haversine_degree_distance(
                (last_tc.coordinate.latitude, last_tc.coordinate.longitude),
                (next_tc.coordinate.latitude, next_tc.coordinate.longitude),
            )

            if distance <= 8:
                current_track.append(next_tc)
                used_tc_ids.add(next_tc.id)
                last_tc = next_tc

        # Create track if it has points
        if len(current_track) > 1:
            tracks.append(TCTracks(id=track_id, track=current_track))
            track_id += 1
    return tracks


# TODO: review for potential improvements (vectorization, better readability, etc.)
def create_tctracks_from_dataset(
    cyclone_dataset: xr.Dataset,
    slp_contour_magnitude=200,
    dz_contour_magnitude=-6,
    min_distance=12,
):
    # Find the SLP minima
    slp = cyclone_dataset["air_pressure_at_mean_sea_level"]
    # Find the DZ maxima
    dz = cyclone_dataset["geopotential_thickness"]
    valid_candidates = {}
    # Initialize the id number to increment for each new TC
    id_number = 0
    # Initialize a time counter for time slice filters
    for init_time in slp.init_time[:]:
        # If there are no valid times for the init_time, skip
        if init_time.isnull().all():
            continue
        else:
            valid_candidates[init_time[0].values] = []

            # TODO: get init time working here
            slp_time = slp.sel(init_time=init_time)
            dz_time = dz.sel(init_time=init_time)
            for time_counter, t in enumerate(slp_time.valid_time):
                candidate_slp_points = peak_local_max(
                    -(slp_time.values),
                    min_distance=min_distance,
                    # exclude_border=0 required as points <= min_distance from border are not included
                    exclude_border=0,
                )
                for point in candidate_slp_points:
                    slp_contours = find_contours_from_point_specified_field(
                        slp_time, point, slp_contour_magnitude
                    )
                    dz_contours = find_contours_from_point_specified_field(
                        dz_time, point, dz_contour_magnitude
                    )
                    candidate = find_valid_candidates(
                        slp_contours, dz_contours, point, cyclone_dataset, time_counter
                    )
                    if candidate is not None:
                        vmax_mask = create_great_circle_mask(
                            cyclone_dataset, candidate, 2
                        )
                        vmax = np.max(
                            cyclone_dataset.where(vmax_mask)["surface_wind_speed"]
                            .max()
                            .values
                        )
                        tc = TC(
                            id=id_number,
                            valid_time=pd.to_datetime(t.values),
                            coordinate=candidate,
                            vmax=vmax,
                            slp=slp_time.sel(
                                latitude=candidate[0], longitude=candidate[1]
                            ).values,
                        )
                        valid_candidates[init_time].append(tc)
                        # Increment the id number for the next individual TC point
                        id_number += 1

    tctracks = create_tctracks(valid_candidates)
    return tctracks


def tctracks_to_3d_dataset(tctracks: list[TCTracks]) -> xr.Dataset:
    """Convert list of TCTracks objects to 3D xarray Dataset with time, latitude, and longitude dimensions.

    Args:
        tctracks: List of TCTracks objects.

    Returns:
        A 3D xarray Dataset with time, lat, lon dimensions.
    """
    # Get all unique times, latitudes, and longitudes
    all_times = []
    all_lats = []
    all_lons = []

    for track in tctracks:
        for tc in track.track:
            all_times.append(tc.valid_time)
            all_lats.append(float(tc.coordinate.latitude))
            all_lons.append(float(tc.coordinate.longitude))

    # Get unique sorted coordinates
    unique_times = sorted(list(set(all_times)))
    unique_lats = sorted(list(set(all_lats)))
    unique_lons = sorted(list(set(all_lons)))

    # Create coordinate arrays
    time_coords = pd.to_datetime(unique_times)
    lat_coords = np.array(unique_lats)
    lon_coords = np.array(unique_lons)

    # Initialize data arrays with NaN
    vmax_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
    slp_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
    track_id_data = np.full(
        (len(time_coords), len(lat_coords), len(lon_coords)), np.nan
    )

    # Fill in the data
    for track in tctracks:
        for tc in track.track:
            time_idx = time_coords.get_loc(tc.valid_time)
            lat_idx = np.where(lat_coords == float(tc.coordinate.latitude))[0][0]
            lon_idx = np.where(lon_coords == float(tc.coordinate.longitude))[0][0]

            vmax_data[time_idx, lat_idx, lon_idx] = float(tc.vmax)
            slp_data[time_idx, lat_idx, lon_idx] = float(tc.slp)
            track_id_data[time_idx, lat_idx, lon_idx] = track.id

    # Create 3D dataset
    ds_3d = xr.Dataset(
        {
            "wind_speed": (["time", "latitude", "longitude"], vmax_data),
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                slp_data,
            ),
            "track_id": (["time", "latitude", "longitude"], track_id_data),
        },
        coords={
            "valid_time": time_coords,
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
    )

    return ds_3d


def orography(ds: xr.Dataset) -> xr.DataArray:
    """Calculate the orography from the geopotential at the surface using ERA5.

    Args:
        ds: The potential xarray dataset to calculate the orography from.

    Returns:
        The orography as an xarray DataArray.
    """
    if "geopotential_at_surface" in ds.variables:
        return ds["geopotential_at_surface"] / 9.80665
    else:
        era5 = xr.open_zarr(
            inputs.ARCO_ERA5_FULL_URI,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        return (
            era5.isel(time=1000000)["geopotential_at_surface"].sel(
                latitude=ds.latitude, longitude=ds.longitude
            )
            / 9.80665
        )


def calculate_pressure_at_surface(orography_da: xr.DataArray) -> xr.DataArray:
    """Calculate the pressure at the surface, based on orography.

    The dataarray is orography (geopotential at the surface/g0).

    Args:
        orography_da: The orography dataarray.

    Returns:
        The pressure at the surface of the dataarray in Pa.
    """
    return 101325 * (1 - 2.25577e-5 * orography_da) ** 5.25579


def calculate_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate wind speed from available wind data.

    Args:
        ds: The xarray dataset to calculate the wind speed from.

    Returns:
        The wind speed as an xarray DataArray.
    """
    if "surface_wind_speed" in ds.data_vars:
        return ds["surface_wind_speed"]
    elif (
        "surface_eastward_wind" in ds.data_vars
        and "surface_northward_wind" in ds.data_vars
    ):
        return np.hypot(ds["surface_eastward_wind"], ds["surface_northward_wind"])
    else:
        raise ValueError("No suitable wind speed variables found in dataset")


def subset_variable_and_maybe_levels(
    ds: xr.Dataset,
    var_name: str,
    level_name: str = "level",
    level_value: Optional[int | Sequence[int]] = None,
) -> xr.DataArray:
    """Subset a variable and its levels from an xarray dataset.

    Args:
        ds: The xarray dataset to subset.
        var_name: The name of the variable to subset.
        level_name: The name of the level to subset.
        level_value: The value of the level to subset.

    Returns:
        The subsetted variable as an xarray DataArray.
    """
    if level_value is None:
        return ds[var_name]
    else:
        return ds[var_name].sel({level_name: level_value})


def generate_geopotential_thickness(
    ds: xr.Dataset,
    var_name: str = "geopotential",
    level_name: str = "level",
    top_level_value: int | Sequence[int] = [200, 300, 400],
    bottom_level_value: int = 500,
) -> xr.DataArray:
    """Generate the geopotential thickness from the geopotential heights.

    Args:
        ds: The xarray dataset to generate the geopotential thickness from.
        var_name: The name of the variable to generate the geopotential thickness from.
        level_name: The name of the level to generate the geopotential thickness from.
        top_level_value: The value of the top level to generate the geopotential thickness from.
        bottom_level_value: The value of the bottom level to generate the geopotential thickness from.

    Returns:
        The geopotential thickness as an xarray DataArray.
    """
    geopotential_heights = subset_variable_and_maybe_levels(
        ds=ds, var_name=var_name, level_name=level_name, level_value=top_level_value
    )
    geopotential_height_bottom = subset_variable_and_maybe_levels(
        ds=ds,
        var_name=var_name,
        level_name=level_name,
        level_value=bottom_level_value,
    )
    geopotential_thickness = (
        geopotential_heights - geopotential_height_bottom
    ) / 9.80665
    geopotential_thickness.attrs = dict(
        description="Geopotential thickness of level and 500 hPa", units="m"
    )
    return geopotential_thickness


def generate_tc_variables(ds: xr.Dataset) -> xr.Dataset:
    """Generate the variables needed for the TC track calculation.

    Args:
        ds: The xarray dataset to subset from.

    Returns:
        The subset variables as an xarray Dataset.
    """

    output = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": subset_variable_and_maybe_levels(
                ds, var_name="air_pressure_at_mean_sea_level"
            ),
            "geopotential_thickness": generate_geopotential_thickness(
                ds, top_level_value=300, bottom_level_value=500
            ),
            "surface_wind_speed": calculate_wind_speed(ds),
        },
    )

    return output


def nantrapezoid(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
):
    """
    Trapezoid rule for arrays with nans.

    Identical to np.trapezoid but with nans handled correctly in the summation.
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    if y.ndim != d.ndim:
        d = np.expand_dims(d, axis=1)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        # This is the only location different from np.trapezoid
        ret = np.nansum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret
