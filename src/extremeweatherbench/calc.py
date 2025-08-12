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


class CycloneDatasetBuilder:
    """
    A class to find cyclones in a dataset using the TempestExtremes criteria.
    <insert citation here>
    """

    def orography(self, ds: xr.Dataset) -> xr.DataArray:
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

    def calculate_wind_speed(self, ds: xr.Dataset) -> xr.DataArray:
        """Calculate wind speed from available wind data."""
        if "surface_wind_speed" in ds.data_vars:
            return ds["surface_wind_speed"]
        elif (
            "surface_eastward_wind" in ds.data_vars
            and "surface_northward_wind" in ds.data_vars
        ):
            return np.hypot(ds["surface_eastward_wind"], ds["surface_northward_wind"])
        else:
            raise ValueError("No suitable wind speed variables found in dataset")

    def subset_variable(
        self,
        ds: xr.Dataset,
        var_name: str,
        level_name: str = "level",
        level_value: Optional[int | Sequence[int]] = None,
    ):
        if level_value is None:
            return ds[var_name]
        else:
            return ds[var_name].sel({level_name: level_value})

    def generate_geopotential_thickness(
        self,
        ds: xr.Dataset,
        var_name: str = "geopotential",
        level_name: str = "level",
        top_level_value: int | Sequence[int] = [200, 300, 400],
        bottom_level_value: int = 500,
    ):
        geopotential_heights = self.subset_variable(
            ds=ds, var_name=var_name, level_name=level_name, level_value=top_level_value
        )
        geopotential_height_bottom = self.subset_variable(
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

    def generate_variables(self, ds: xr.Dataset):
        """Subset the variables based on the variables argument.

        Using "min" will subset the minimum required variables - sea level pressure and geopotential thickness.
        Using "max" will subset the maximum required variables - sea level pressure, geopotential thickness, u and v
        surface winds, u and v 850 hPa winds, and air temperature at 400 hPa.
        Args:
            vars: The variable pattern to subset.

        Returns:
            The subsetted variables.
        """

        output = xr.Dataset(
            {
                "air_pressure_at_mean_sea_level": self.subset_variable(
                    ds, var_name="air_pressure_at_mean_sea_level"
                ),
                "geopotential_thickness": self.generate_geopotential_thickness(
                    ds, top_level_value=300, bottom_level_value=500
                ),
                "surface_wind_speed": self.calculate_wind_speed(ds),
            },
        )

        return output


Location = namedtuple("Location", ["latitude", "longitude"])


@dataclasses.dataclass
class TC:
    id: int
    valid_time: pd.Timestamp
    coordinate: Location
    vmax: float
    slp: float


@dataclasses.dataclass
class TCTrack:
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
    """Find the two points in a contour that are furthest apart. From
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-in-plot-with-many-points-in-python
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
    """
    Create a circular mask based on great circle distance for an xarray dataset.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with 'latitude' and 'longitude' coordinates
    center_lat : float
        Latitude of the center point
    center_lon : float
        Longitude of the center point
    radius_degrees : float
        Radius in degrees of great circle distance

    Returns:
    --------
    mask : xarray.DataArray
        Boolean mask where True indicates points within the radius
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


def create_tctracks(tcs: list[TC]) -> list[TCTrack]:
    """
    Create a TCTrack from a list of TCs.

    Groups TCs into tracks if they are:
    1. At unique timesteps
    2. Within 8 great circle degrees of the previous timestep
    3. Within 54 hours of the previous timestep

    Returns a list of TCTrack objects.
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
            tracks.append(TCTrack(id=track_id, track=current_track))
            track_id += 1
    return tracks


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
        if slp.init_time[:, 11][0].isnull():
            continue
        else:
            valid_candidates[init_time[0].values] = []
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


def tctracks_to_3d_dataset(tctracks: list[TCTrack]) -> xr.Dataset:
    """Convert list of TCTrack objects to 3D xarray Dataset with time, lat, lon dimensions."""
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
            "vmax": (["time", "latitude", "longitude"], vmax_data),
            "slp": (["time", "latitude", "longitude"], slp_data),
            "track_id": (["time", "latitude", "longitude"], track_id_data),
        },
        coords={"time": time_coords, "latitude": lat_coords, "longitude": lon_coords},
    )

    return ds_3d
