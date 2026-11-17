"""Equivalence tests for the localised spatial mask / contour search.

The tracker restricts two searches to the neighbourhood of the point they are
about: ``_create_spatial_mask`` builds its distance field in a window instead of
globally, and ``find_contours_from_point_specified_field`` contours a window
instead of the whole (global) field. Both must return *exactly* what the
whole-field computation returns for anything that can be accepted. These tests
pin that claim against reference implementations of the previous code.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc
from extremeweatherbench.events import tropical_cyclone as tc


# --------------------------------------------------------------------------
# reference implementations (the whole-field code this change replaces)
# --------------------------------------------------------------------------
def reference_spatial_mask(
    lat_coords, lon_coords, nearby_tc_track_data, max_distance_degrees
):
    lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing="ij")
    if nearby_tc_track_data.empty:
        return np.zeros_like(lat_grid, dtype=bool)
    tc_lats = nearby_tc_track_data["latitude"].to_numpy(dtype=float)
    tc_lons = nearby_tc_track_data["longitude"].to_numpy(dtype=float)
    spatial_mask = np.zeros_like(lat_grid, dtype=bool)
    for start in range(0, len(tc_lats), 32):
        stop = start + 32
        distances = calc.haversine_distance(
            [lat_grid[np.newaxis], lon_grid[np.newaxis]],
            [
                tc_lats[start:stop, np.newaxis, np.newaxis],
                tc_lons[start:stop, np.newaxis, np.newaxis],
            ],
            units="degrees",
        )
        spatial_mask |= np.any(distances <= max_distance_degrees, axis=0)
    return spatial_mask


# --------------------------------------------------------------------------
# _create_spatial_mask
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dlat", [0.25, 1.0])
@pytest.mark.parametrize("radius", [1.0, 5.0, 12.0])
@pytest.mark.parametrize("n_points", [1, 7, 40])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spatial_mask_matches_whole_field(dlat, radius, n_points, seed):
    rng = np.random.default_rng(
        1000 * seed + int(dlat * 7) + int(radius) + int(n_points)
    )
    lat = np.arange(90, -90.01, -dlat)
    lon = np.arange(0, 360, dlat)
    df = pd.DataFrame(
        {
            "latitude": rng.uniform(-80, 80, n_points),
            "longitude": rng.uniform(0, 360, n_points),
        }
    )
    np.testing.assert_array_equal(
        tc._create_spatial_mask(lat, lon, df, radius),
        reference_spatial_mask(lat, lon, df, radius),
    )


def test_spatial_mask_matches_whole_field_at_the_dateline_and_poles():
    """The hard cases: a disc straddling lon=0/360, and discs on the pole."""
    lat = np.arange(90, -90.01, -0.25)
    lon = np.arange(0, 360, 0.25)
    df = pd.DataFrame(
        {
            "latitude": [89.9, -89.9, 0.0, 45.0, 60.0, -30.0],
            "longitude": [0.0, 359.9, 0.125, 359.8, 0.0, 179.99],
        }
    )
    np.testing.assert_array_equal(
        tc._create_spatial_mask(lat, lon, df, 5.0),
        reference_spatial_mask(lat, lon, df, 5.0),
    )


def test_spatial_mask_matches_whole_field_on_wrapped_and_ascending_grids():
    """ERA5 ships 0-360 with descending latitude; model grids may not."""
    cases = [
        (np.arange(90, -90.01, -0.5), np.arange(0, 360, 0.5)),
        (np.arange(-90, 90.01, 0.5), np.arange(-180, 180, 0.5)),
        (np.linspace(90, -90, 91), np.linspace(0, 359, 180)),
    ]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"latitude": rng.uniform(-70, 70, 12), "longitude": rng.uniform(0, 360, 12)}
    )
    for lat, lon in cases:
        df_loc = df.copy()
        if lon.min() < 0:  # express the points in this grid's longitude convention
            df_loc["longitude"] = (df["longitude"] + 180) % 360 - 180
        np.testing.assert_array_equal(
            tc._create_spatial_mask(lat, lon, df_loc, 5.0),
            reference_spatial_mask(lat, lon, df_loc, 5.0),
        )


def test_spatial_mask_empty_and_degenerate_grids():
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.arange(0, 10.0)
    df = pd.DataFrame({"latitude": [1.0], "longitude": [5.0]})
    np.testing.assert_array_equal(
        tc._create_spatial_mask(lat, lon, df, 1.5),
        reference_spatial_mask(lat, lon, df, 1.5),
    )
    assert tc._create_spatial_mask(
        lat, lon, pd.DataFrame({"latitude": [], "longitude": []}), 5.0
    ).shape == (3, 10)
    assert not tc._create_spatial_mask(
        lat, lon, pd.DataFrame({"latitude": [], "longitude": []}), 5.0
    ).any()


def test_spatial_mask_falls_back_on_non_uniform_and_nan_grids():
    """The window sizes itself on the first cell spacing; grids where that is
    not the spacing everywhere (or where a coordinate is NaN) must keep the
    whole-field result exactly."""
    lat = np.concatenate([np.arange(90, 0, -2.0), np.arange(0, -90.01, -0.5)])
    lon = np.concatenate([np.arange(0, 180, 2.0), np.arange(180, 360, 0.5)])
    df = pd.DataFrame({"latitude": [-40.0, 30.0], "longitude": [200.0, 100.0]})
    np.testing.assert_array_equal(
        tc._create_spatial_mask(lat, lon, df, 5.0),
        reference_spatial_mask(lat, lon, df, 5.0),
    )
    lat_nan = np.arange(90, -90.01, -0.25)
    lat_nan[-1] = np.nan
    lon_ok = np.arange(0, 360, 0.25)
    df_nan = pd.DataFrame({"latitude": [20.0], "longitude": [100.0]})
    np.testing.assert_array_equal(
        tc._create_spatial_mask(lat_nan, lon_ok, df_nan, 5.0),
        reference_spatial_mask(lat_nan, lon_ok, df_nan, 5.0),
    )


def test_coordinate_window_rejects_non_uniform_and_nan_grids():
    lat = np.arange(90, -90.01, -0.25)
    lon = np.arange(0, 360, 0.25)
    assert tc._coordinate_window((400, 800), 7.5, lat, lon) is not None
    lon_bad = lon.copy()
    lon_bad[800] += 0.1  # one misplaced cell
    assert tc._coordinate_window((400, 800), 7.5, lat, lon_bad) is None
    lat_bad = lat.copy()
    lat_bad[-1] = np.nan
    assert tc._coordinate_window((400, 800), 7.5, lat_bad, lon) is None


def test_windowed_contours_equal_global_on_non_uniform_grids():
    """With the fallback in place a windowed call is the global call."""
    field = _low_field(nlat=201, nlon=400, nlow=12, seed=13)
    lon = field.longitude.values.copy()
    lon[1] += 0.05  # non-uniform in exactly one gap
    field = field.assign_coords(longitude=lon)
    for point in _minima(field)[:10]:
        point = tuple(int(v) for v in point)
        glob = tc.find_contours_from_point_specified_field(field, point, 300.0)
        loc = tc.find_contours_from_point_specified_field(field, point, 300.0, 7.5)
        assert len(glob) == len(loc)
        for c1, c2 in zip(glob, loc):
            np.testing.assert_allclose(c1, c2, atol=0.0)


# --------------------------------------------------------------------------
# find_contours_from_point_specified_field
# --------------------------------------------------------------------------
def _low_field(nlat=401, nlon=800, nlow=25, seed=3):
    """A flat background with localised closed lows, like a real SLP chart."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(60, -60, nlat)
    lon = np.linspace(0, 359.5, nlon)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    f = np.full(la.shape, 101_325.0)
    for _ in range(nlow):
        lat0, lon0 = rng.uniform(-50, 50), rng.uniform(0, 355)
        # compact lows: a +200 Pa contour must sit inside the ~5.5 degree
        # acceptance radius for find_valid_candidates to accept anything at all
        amp, sig = rng.uniform(700, 1500), rng.uniform(1.2, 2.4)
        dlat = la - lat0
        dlon = ((lo - lon0 + 180) % 360 - 180) * np.cos(np.radians(lat0))
        f -= amp * np.exp(-(dlat**2 + dlon**2) / (2 * sig**2))
    return xr.DataArray(
        f, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon}
    )


def _minima(field, min_distance=8):
    from skimage.feature import peak_local_max

    return peak_local_max(
        -field.values,
        min_distance=min_distance,
        exclude_border=True,
        threshold_abs=-101_000.0,
    )


@pytest.mark.parametrize("level", [200.0, 400.0, 800.0])
@pytest.mark.parametrize("radius", [6.5, 8.0])
def test_windowed_contours_are_the_same_contours(level, radius):
    """Every contour the window finds exists, identically, in the global search."""
    field = _low_field()
    for point in _minima(field):
        point = tuple(int(v) for v in point)
        glob = tc.find_contours_from_point_specified_field(field, point, level)
        loc = tc.find_contours_from_point_specified_field(field, point, level, radius)
        # The precise property: the contours that lie WELL INSIDE the window
        # (so the window cannot cut them) are exactly the same objects either
        # way. A contour reaching the window edge extends >= radius degrees from
        # the peak, so it fails the tracker's "furthest point within
        # max_gc_distance" test in both implementations and is not compared.
        limit = radius - 1.5

        def accept(cs, point=point, limit=limit):
            out = []
            for c in cs:
                if not np.allclose(c[0], c[-1]):
                    continue
                if (
                    tc.find_valid_contour_from_point(
                        c, point, field.latitude, field.longitude
                    )
                    > limit
                ):
                    continue
                out.append(c)
            return sorted(out, key=lambda c: (len(c), float(c[0, 0]), float(c[0, 1])))

        a, b = accept(glob), accept(loc)
        assert len(a) == len(b), (len(a), len(b), point, level, radius)
        for c1, c2 in zip(a, b):
            np.testing.assert_allclose(c1, c2, atol=1e-9)


def test_low_field_fixture_producestightly_bound_contours():
    """Guard: the equivalence tests above would be vacuous on an empty fixture."""
    field = _low_field()
    n = 0
    for point in _minima(field):
        point = tuple(int(v) for v in point)
        for c in tc.find_contours_from_point_specified_field(field, point, 200.0):
            if (
                np.allclose(c[0], c[-1])
                and tc.find_valid_contour_from_point(
                    c, point, field.latitude, field.longitude
                )
                < 5.0
            ):
                n += 1
    assert n > 0


@pytest.mark.parametrize("radius", [6.5, 8.0])
def test_contour_validation_decisions_are_unchanged(radius):
    """find_valid_candidates accepts exactly the same peaks either way."""
    field = _low_field(seed=5)
    # warm core: thickness is HIGHEST over the low, so the -6 m anomaly
    # contour closes around the centre the way a real TC's does
    dz = 5_000.0 + (101_325.0 - field.values) * 0.05
    lat, lon = field.latitude, field.longitude
    n = 0
    for point in _minima(field):
        point = tuple(int(v) for v in point)
        out = {}
        for tag, r in (("global", None), ("window", radius)):
            slp_c = tc.find_contours_from_point_specified_field(field, point, 400.0, r)
            dz_c = tc.find_contours_from_point_specified_field(
                xr.DataArray(dz, dims=field.dims, coords=field.coords), point, -6.0, r
            )
            out[tag] = (
                tc.find_valid_candidates(
                    slp_c,
                    dz_c,
                    point,
                    lat,
                    lon,
                    max_gc_distance_slp_contour=5.5,
                    max_gc_distance_dz_contour=5.0,
                    latitude_max_degrees=70.0,
                )
                is not None
            )
        assert out["global"] == out["window"], point
        n += out["global"]
    assert n > 0, "the fixture must produce at least one validated candidate"


@pytest.mark.parametrize("radius", [None, 6.5, 8.0])
def test_find_peaks_for_time_slice_is_unchanged_by_the_window(radius):
    """End to end: the peak set does not move when the window is used."""
    field = _low_field(nlat=241, nlon=400, nlow=18, seed=11)
    lat = field.latitude.values
    lon = field.longitude.values
    dz = xr.DataArray(
        5_000.0 + (101_325.0 - field.values) * 0.05,
        dims=field.dims,
        coords=field.coords,
    )
    peaks = tc._find_peaks_for_time_slice(
        field.values,
        np.asarray(dz.values),
        pd.Timestamp("2023-10-01 06:00"),
        pd.DataFrame(
            {
                "latitude": lat[[60, 120, 180]],
                "longitude": lon[[50, 150, 300]],
                "valid_time": [pd.Timestamp("2023-10-01 06:00")] * 3,
            }
        ),
        min_distance_between_peaks=6,
        lat_coords=lat,
        lon_coords=lon,
        max_spatial_distance_degrees=25.0,
        surface_pressure_threshold=101_000.0,
        latitude_max_degrees=70.0,
        max_temporal_hours=12.0,
        max_gc_distance_slp_contour_degrees=5.5,
        max_gc_distance_dz_contour_degrees=6.5,
    )
    assert isinstance(peaks, np.ndarray)
    # the same call with the window disabled must agree exactly
    from unittest import mock

    real = tc.find_contours_from_point_specified_field

    def global_only(field_, point_, level_, search_radius_degrees=None):
        return real(field_, point_, level_, None)

    with mock.patch.object(tc, "find_contours_from_point_specified_field", global_only):
        peaks_global = tc._find_peaks_for_time_slice(
            field.values,
            np.asarray(dz.values),
            pd.Timestamp("2023-10-01 06:00"),
            pd.DataFrame(
                {
                    "latitude": lat[[60, 120, 180]],
                    "longitude": lon[[50, 150, 300]],
                    "valid_time": [pd.Timestamp("2023-10-01 06:00")] * 3,
                }
            ),
            min_distance_between_peaks=6,
            lat_coords=lat,
            lon_coords=lon,
            max_spatial_distance_degrees=25.0,
            surface_pressure_threshold=101_000.0,
            latitude_max_degrees=70.0,
            max_temporal_hours=12.0,
            max_gc_distance_slp_contour_degrees=5.5,
            max_gc_distance_dz_contour_degrees=6.5,
        )
    a = sorted(map(tuple, np.asarray(peaks, dtype=int).reshape(-1, 2)))
    b = sorted(map(tuple, np.asarray(peaks_global, dtype=int).reshape(-1, 2)))
    assert a == b
    assert len(a) > 0, "the fixture must actually produce validated peaks"
