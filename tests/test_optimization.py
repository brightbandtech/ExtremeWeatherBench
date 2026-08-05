"""Tests that drive and guard the xarray/dask optimization work.

Two kinds of test live here.

Performance-contract tests assert things like "the result is still lazy" or
"this graph does not grow quadratically with lead_time". They fail on the
unoptimized code, so they can drive a change test-first.

Characterization tests pin numeric output against hand-derived literals so an
optimization cannot silently change results. They pass before and after, so
they guard rather than drive.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from scipy import ndimage

from extremeweatherbench import calc, metrics, utils
from extremeweatherbench.events import atmospheric_river as ar
from tests import optimization_helpers as opt


def _forecast_dataset(n_init, n_lead, chunk=True, lead_unit=None):
    """Forecast-shaped dataset with init_time and lead_time dimensions."""
    lead = np.arange(0, n_lead * 6, 6)
    if lead_unit is not None:
        lead = pd.to_timedelta(lead, unit=lead_unit)
    ds = xr.Dataset(
        {
            "t": (
                ["init_time", "lead_time", "latitude"],
                np.arange(float(n_init * n_lead * 4)).reshape(n_init, n_lead, 4),
            )
        },
        coords={
            "init_time": pd.date_range("2020-01-01", periods=n_init, freq="D"),
            "lead_time": lead,
            "latitude": np.linspace(0, 3, 4),
        },
    )
    return ds.chunk({"init_time": 1}) if chunk else ds


class TestTimeCoordinateConversionScaling:
    """The init/valid reshape must not build a subgraph per lead time.

    Building one subgraph per lead and concatenating with coordinate
    comparison makes both graph size and runtime grow quadratically, which
    is what these tests forbid.
    """

    def test_init_to_valid_graph_does_not_scale_quadratically_with_lead_time(self):
        few = utils.convert_init_time_to_valid_time(_forecast_dataset(6, 4))
        many = utils.convert_init_time_to_valid_time(_forecast_dataset(6, 16))

        ratio = opt.graph_size(many) / opt.graph_size(few)
        assert ratio < 6.0, (
            f"quadrupling lead_time grew the graph {ratio:.1f}x; a vectorized "
            "reshape should grow it roughly linearly"
        )

    def test_valid_to_init_graph_does_not_scale_quadratically_with_lead_time(self):
        few = utils.convert_valid_time_to_init_time(
            utils.convert_init_time_to_valid_time(_forecast_dataset(6, 4))["t"]
        )
        many = utils.convert_valid_time_to_init_time(
            utils.convert_init_time_to_valid_time(_forecast_dataset(6, 16))["t"]
        )

        ratio = opt.graph_size(many) / opt.graph_size(few)
        assert ratio < 6.0, (
            f"quadrupling lead_time grew the graph {ratio:.1f}x; a vectorized "
            "reshape should grow it roughly linearly"
        )


class TestTimeCoordinateConversionOutput:
    """Characterization: the reshape must keep producing these exact values.

    Expected values are hand-derived from the init/lead grid rather than
    computed by the function under test.
    """

    def test_overlapping_valid_times_land_on_a_shared_axis(self):
        """Two inits a day apart with 0 h and 24 h leads share a valid time.

        init 2020-01-01 at lead 24h and init 2020-01-02 at lead 0h both land
        on 2020-01-02, so the union axis has three entries, not four.
        """
        ds = xr.Dataset(
            {"t": (["init_time", "lead_time"], [[0.0, 1.0], [2.0, 3.0]])},
            coords={
                "init_time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "lead_time": pd.to_timedelta([0, 24], unit="h"),
            },
        )

        result = utils.convert_init_time_to_valid_time(ds)

        assert list(result.valid_time.values) == list(
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        )
        actual = result["t"].transpose("lead_time", "valid_time").values
        expected = np.array(
            [
                [0.0, 2.0, np.nan],  # lead 0h: inits 01-01, 01-02, then nothing
                [np.nan, 1.0, 3.0],  # lead 24h: nothing, then inits 01-01, 01-02
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_integer_lead_times_are_added_as_nanoseconds(self):
        """Integer lead_time is added directly to datetime64, i.e. as ns.

        This pins existing behavior. Callers that mean hours must pass a
        timedelta; an optimization must not silently reinterpret the unit.
        """
        ds = xr.Dataset(
            {"t": (["init_time", "lead_time"], [[0.0, 1.0]])},
            coords={
                "init_time": pd.to_datetime(["2020-01-01"]),
                "lead_time": [0, 24],
            },
        )

        result = utils.convert_init_time_to_valid_time(ds)

        assert list(result.valid_time.values) == [
            np.datetime64("2020-01-01T00:00:00.000000000"),
            np.datetime64("2020-01-01T00:00:00.000000024"),
        ]

    def test_init_time_is_retained_as_a_two_dimensional_coordinate(self):
        """init_time survives as a (lead_time, valid_time) coord, NaT if absent."""
        ds = xr.Dataset(
            {"t": (["init_time", "lead_time"], [[0.0, 1.0], [2.0, 3.0]])},
            coords={
                "init_time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "lead_time": pd.to_timedelta([0, 24], unit="h"),
            },
        )

        result = utils.convert_init_time_to_valid_time(ds)

        init_coord = result.init_time.transpose("lead_time", "valid_time").values
        assert init_coord.shape == (2, 3)
        assert init_coord[0, 0] == np.datetime64("2020-01-01")
        assert init_coord[0, 1] == np.datetime64("2020-01-02")
        assert pd.isna(init_coord[0, 2])
        assert pd.isna(init_coord[1, 0])
        assert init_coord[1, 1] == np.datetime64("2020-01-01")
        assert init_coord[1, 2] == np.datetime64("2020-01-02")

    def test_non_dimension_coordinates_are_carried_through_and_masked(self):
        """A bool (init_time, lead_time) coord survives the reshape.

        ForecastBase.subset_data_to_case attaches valid_time_mask this way,
        so the reshape has to carry it across without dropping it.
        """
        ds = xr.Dataset(
            {"t": (["init_time", "lead_time"], [[0.0, 1.0], [2.0, 3.0]])},
            coords={
                "init_time": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "lead_time": pd.to_timedelta([0, 24], unit="h"),
                "valid_time_mask": (
                    ["init_time", "lead_time"],
                    np.array([[False, True], [True, True]]),
                ),
            },
        )

        result = utils.convert_init_time_to_valid_time(ds)

        assert "valid_time_mask" in result.coords
        mask = result.valid_time_mask.transpose("lead_time", "valid_time").values
        assert not bool(mask[0, 0])
        assert bool(mask[0, 1])
        assert bool(mask[1, 1])

    def test_chunked_input_gives_the_same_values_as_unchunked(self):
        chunked = utils.convert_init_time_to_valid_time(_forecast_dataset(4, 3))
        unchunked = utils.convert_init_time_to_valid_time(
            _forecast_dataset(4, 3, chunk=False)
        )
        xr.testing.assert_allclose(chunked.compute(), unchunked)

    def test_round_trip_back_to_init_time_recovers_the_original_values(self):
        ds = _forecast_dataset(4, 3, chunk=False)
        forward = utils.convert_init_time_to_valid_time(ds)
        back = utils.convert_valid_time_to_init_time(forward["t"])

        recovered = back.transpose("init_time", "lead_time", "latitude")
        np.testing.assert_allclose(
            recovered.sel(init_time=ds.init_time).values, ds["t"].values
        )


def _ar_inputs(
    time_dim="valid_time",
    n_time=3,
    n_lat=40,
    n_lon=40,
    extra_dim=None,
    n_extra=1,
    blobs=(),
    chunk=True,
):
    """IVT and Laplacian fields with rectangular blobs above both thresholds.

    Each blob is (time_index, extra_index, lat_slice, lon_slice). Values are
    set so the AR criteria are met exactly inside the blob and nowhere else,
    which makes the resulting feature sizes predictable.
    """
    lat = np.linspace(20.0, 59.0, n_lat)
    lon = np.linspace(-160.0, -121.0, n_lon)

    dims = [time_dim, "latitude", "longitude"]
    shape = [n_time, n_lat, n_lon]
    coords = {
        time_dim: (
            pd.date_range("2023-01-01", periods=n_time, freq="6h")
            if time_dim == "valid_time"
            else pd.to_timedelta(np.arange(n_time) * 6, unit="h")
        ),
        "latitude": lat,
        "longitude": lon,
    }
    if extra_dim is not None:
        dims.insert(0, extra_dim)
        shape.insert(0, n_extra)
        coords[extra_dim] = pd.date_range("2023-01-01", periods=n_extra, freq="D")

    ivt_values = np.zeros(shape)
    lap_values = np.zeros(shape)
    for blob in blobs:
        t_idx, extra_idx, lat_slice, lon_slice = blob
        index: tuple = (t_idx, lat_slice, lon_slice)
        if extra_dim is not None:
            index = (extra_idx,) + index
        ivt_values[index] = 800.0
        lap_values[index] = 5.0

    ivt = xr.DataArray(ivt_values, dims=dims, coords=coords)
    lap = xr.DataArray(lap_values, dims=dims, coords=coords)
    if chunk:
        chunking = {extra_dim: 1} if extra_dim is not None else {time_dim: -1}
        ivt, lap = ivt.chunk(chunking), lap.chunk(chunking)
    return ivt, lap


class TestAtmosphericRiverMaskLaziness:
    """The AR mask must stay in the dask graph instead of materializing."""

    def test_mask_from_dask_inputs_is_still_lazy(self):
        ivt, lap = _ar_inputs(blobs=[(0, 0, slice(10, 30), slice(10, 30))])

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, min_size_gridpoints=50
        )

        opt.assert_lazy(
            result,
            "ndimage.label on a lazy array pulls the whole volume into memory "
            "while the graph is being built",
        )

    def test_pipeline_integrates_ivt_once_rather_than_twice(self):
        """The mask and the returned IVT must share one integration.

        The integration kernel is counted rather than the wrapper, because
        the wrapper only builds graph nodes. If the mask materializes eagerly
        the kernel runs once while the graph is built and again when the
        caller computes the dataset.
        """
        ds = _ar_pipeline_dataset()

        with opt.spy(calc, "nantrapezoid_nd") as integration_only:
            ar.integrated_vapor_transport(
                specific_humidity=ds["specific_humidity"],
                eastward_wind=ds["eastward_wind"],
                northward_wind=ds["northward_wind"],
            ).compute()

        with opt.spy(calc, "nantrapezoid_nd") as whole_pipeline:
            ar.build_atmospheric_river_mask_and_land_intersection(ds).compute()

        assert integration_only.count > 0, "the spy did not observe the kernel"
        assert whole_pipeline.count == integration_only.count, (
            f"the pipeline ran the integration kernel {whole_pipeline.count} "
            f"times against {integration_only.count} for IVT alone; the mask "
            "and the returned IVT should share one integration"
        )


def _ar_pipeline_dataset(n_time=2, n_level=4, n_lat=24, n_lon=24):
    """Minimal dataset accepted by the full AR pipeline."""
    rng = np.random.default_rng(3)
    levels = np.array([1000.0, 850.0, 700.0, 500.0])[:n_level]
    dims = ["valid_time", "level", "latitude", "longitude"]
    shape = (n_time, n_level, n_lat, n_lon)
    coords = {
        "valid_time": pd.date_range("2023-01-01", periods=n_time, freq="6h"),
        "level": levels,
        "latitude": np.linspace(20.0, 43.0, n_lat),
        "longitude": np.linspace(-160.0, -137.0, n_lon),
    }
    return xr.Dataset(
        {
            "eastward_wind": (dims, rng.uniform(10, 40, shape)),
            "northward_wind": (dims, rng.uniform(10, 40, shape)),
            "specific_humidity": (dims, rng.uniform(0.001, 0.02, shape)),
        },
        coords=coords,
    ).chunk({"valid_time": 1})


class TestAtmosphericRiverLabelConnectivity:
    """Features connect along exactly one time-like axis.

    An AR keeps its identity as it evolves, so connectivity along the single
    time axis is wanted. Connecting across separate forecast initializations
    is not: two different forecasts are independent realizations, and merging
    them lets a feature that is too small in either one survive the size
    filter.
    """

    def test_features_connect_across_valid_time_in_an_analysis(self):
        """Two 36-cell blobs at consecutive valid times form one 72-cell feature."""
        ivt, lap = _ar_inputs(
            time_dim="valid_time",
            n_time=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (1, 0, slice(10, 16), slice(10, 16)),
            ],
        )

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        assert int(result.sum()) == 72, (
            "blobs at consecutive valid times should merge into one feature "
            "large enough to survive the size filter"
        )

    def test_features_do_not_connect_across_separate_initializations(self):
        """The same 36-cell blob in two forecasts stays two small features.

        Merging them across init_time would produce 72 cells and wrongly pass
        a 50-cell size filter.
        """
        ivt, lap = _ar_inputs(
            time_dim="lead_time",
            n_time=1,
            extra_dim="init_time",
            n_extra=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (0, 1, slice(10, 16), slice(10, 16)),
            ],
        )

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        assert int(result.sum()) == 0, (
            "a 36-cell feature is below the 50-cell threshold in each forecast "
            "separately and must not be rescued by merging across init_time"
        )

    def test_features_connect_across_lead_time_within_one_initialization(self):
        ivt, lap = _ar_inputs(
            time_dim="lead_time",
            n_time=2,
            extra_dim="init_time",
            n_extra=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (1, 0, slice(10, 16), slice(10, 16)),
            ],
        )

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        by_init = result.sum(dim=["lead_time", "latitude", "longitude"])
        assert int(by_init[0]) == 72
        assert int(by_init[1]) == 0


class TestLandIntersection:
    """Only the true-positive quadrant is needed, so only it should be built."""

    @staticmethod
    def _mask_and_land():
        lat = np.linspace(20.0, 29.0, 10)
        lon = np.linspace(-160.0, -151.0, 10)
        rng = np.random.default_rng(7)
        mask = xr.DataArray(
            (rng.random((3, 10, 10)) > 0.5).astype(np.int64),
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=3, freq="6h"),
                "latitude": lat,
                "longitude": lon,
            },
        )
        land = xr.DataArray(
            (rng.random((10, 10)) > 0.5).astype(float),
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon},
        )
        return mask, land

    def test_matches_the_contingency_table_true_positives(self):
        """Characterization against the scores library implementation."""
        from scores.categorical import BinaryContingencyManager

        mask, land = self._mask_and_land()
        expected = BinaryContingencyManager(mask, land).tp

        actual = calc.find_land_intersection(mask, land_mask=land)

        xr.testing.assert_equal(actual, expected)

    def test_propagates_nan_from_either_input(self):
        mask, land = self._mask_and_land()
        mask = mask.where(mask.latitude > mask.latitude[0])
        land = land.where(land.longitude > land.longitude[0])

        from scores.categorical import BinaryContingencyManager

        expected = BinaryContingencyManager(mask, land).tp
        actual = calc.find_land_intersection(mask, land_mask=land)

        xr.testing.assert_equal(actual, expected)

    def test_does_not_build_a_full_contingency_table(self):
        """The other three quadrants and the counts table are wasted work.

        On a lazy mask that construction dominates, costing more than the
        computation it precedes.
        """
        import scores.categorical

        mask, land = self._mask_and_land()

        with opt.spy(scores.categorical, "BinaryContingencyManager") as counter:
            calc.find_land_intersection(mask.chunk({"valid_time": 1}), land_mask=land)

        assert counter.count == 0, (
            "find_land_intersection built a full contingency table to use only "
            "its true-positive quadrant"
        )

    def test_stays_lazy_for_a_lazy_mask(self):
        mask, land = self._mask_and_land()
        result = calc.find_land_intersection(
            mask.chunk({"valid_time": 1}), land_mask=land
        )
        opt.assert_lazy(result)


class TestAtmosphericRiverMaskOutput:
    """Characterization: thresholds, size and latitude filters keep working."""

    def test_blob_above_all_criteria_is_retained(self):
        ivt, lap = _ar_inputs(blobs=[(0, 0, slice(5, 25), slice(5, 25))])
        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()
        assert int(result.sum()) == 400
        assert set(np.unique(result.values)) <= {0, 1}

    def test_a_blob_exactly_at_the_size_threshold_is_kept(self):
        # A 7x7 blob is 49 gridpoints; the filter keeps features of at least
        # min_size_gridpoints, so 49 is the smallest surviving size.
        ivt, lap = _ar_inputs(blobs=[(0, 0, slice(5, 12), slice(5, 12))])

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=49
        )

        assert int(result.sum()) == 49

    def test_a_blob_one_gridpoint_under_the_threshold_is_dropped(self):
        ivt, lap = _ar_inputs(blobs=[(0, 0, slice(5, 12), slice(5, 12))])

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        )

        assert int(result.sum()) == 0

    def test_blob_below_the_size_threshold_is_dropped(self):
        ivt, lap = _ar_inputs(blobs=[(0, 0, slice(5, 8), slice(5, 8))])
        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()
        assert int(result.sum()) == 0

    def test_tropical_feature_is_dropped_on_mean_latitude(self):
        """A large feature centered below 15 degrees is not an AR."""
        lat = np.linspace(0.0, 39.0, 40)
        lon = np.linspace(-160.0, -121.0, 40)
        values = np.zeros((1, 40, 40))
        values[0, 0:14, 5:35] = 1.0  # latitudes 0 to ~13 degrees
        ivt = xr.DataArray(
            values * 800.0,
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=1),
                "latitude": lat,
                "longitude": lon,
            },
        )
        lap = ivt / 160.0

        result = ar.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        )
        assert int(np.asarray(result).sum()) == 0

    def test_chunked_and_unchunked_inputs_agree(self):
        blobs = [(0, 0, slice(5, 25), slice(5, 25)), (2, 0, slice(8, 20), slice(8, 20))]
        chunked = ar.atmospheric_river_mask(
            ivt=_ar_inputs(blobs=blobs)[0],
            ivt_laplacian=_ar_inputs(blobs=blobs)[1],
            min_size_gridpoints=50,
        )
        unchunked = ar.atmospheric_river_mask(
            ivt=_ar_inputs(blobs=blobs, chunk=False)[0],
            ivt_laplacian=_ar_inputs(blobs=blobs, chunk=False)[1],
            min_size_gridpoints=50,
        )
        np.testing.assert_array_equal(np.asarray(chunked), np.asarray(unchunked))


def _spatial_dataarray(n_time=6, n_lat=8, n_lon=8, chunk=True):
    """Small (valid_time, latitude, longitude) array for reduction tests."""
    rng = np.random.default_rng(3)
    da = xr.DataArray(
        rng.uniform(280.0, 310.0, (n_time, n_lat, n_lon)),
        dims=["valid_time", "latitude", "longitude"],
        coords={
            "valid_time": pd.date_range("2023-01-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(30.0, 30.0 + n_lat - 1, n_lat),
            "longitude": np.linspace(-120.0, -120.0 + n_lon - 1, n_lon),
        },
    )
    return da.chunk({"valid_time": 1}) if chunk else da


class TestReduceDataArrayLaziness:
    """A spatial mean must not be a compute barrier by default.

    Computing inside every reduction breaks the graph into one eager step per
    call site, so a metric that reduces a forecast and a target pays two
    separate passes over the data instead of one fused pass.
    """

    def test_string_method_reduction_stays_lazy(self):
        da = _spatial_dataarray()
        result = utils.reduce_dataarray(
            da, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
        )
        opt.assert_lazy(result)

    def test_callable_reduction_stays_lazy(self):
        da = _spatial_dataarray()
        result = utils.reduce_dataarray(
            da, method=np.nanmean, reduce_dims=["latitude", "longitude"]
        )
        opt.assert_lazy(result)

    def test_compute_true_is_still_available_for_data_dependent_indexing(self):
        da = _spatial_dataarray()
        result = utils.reduce_dataarray(
            da,
            method="mean",
            reduce_dims=["latitude", "longitude"],
            compute=True,
            skipna=True,
        )
        assert isinstance(result.data, np.ndarray)


class TestReduceDataArrayOutput:
    """Characterization: laziness must not change the reduced values."""

    def test_lazy_and_eager_reductions_agree(self):
        da = _spatial_dataarray()
        lazy = utils.reduce_dataarray(
            da, method="mean", reduce_dims=["latitude", "longitude"], skipna=True
        )
        eager = utils.reduce_dataarray(
            da,
            method="mean",
            reduce_dims=["latitude", "longitude"],
            compute=True,
            skipna=True,
        )
        xr.testing.assert_allclose(lazy.compute(), eager)

    def test_mean_matches_a_hand_computed_column(self):
        da = _spatial_dataarray(n_time=2, n_lat=2, n_lon=2, chunk=False)
        da.values = np.array([[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]])
        result = utils.reduce_dataarray(
            da.chunk({"valid_time": 1}),
            method="mean",
            reduce_dims=["latitude", "longitude"],
            skipna=True,
        )
        np.testing.assert_allclose(result.compute().values, [2.5, 25.0])


def _unchunked_target(n_time=40, n_lat=64, n_lon=64):
    """Numpy-backed target dataset, as an unchunked zarr source produces."""
    rng = np.random.default_rng(5)
    return xr.Dataset(
        {
            "surface_air_temperature": (
                ["valid_time", "latitude", "longitude"],
                rng.uniform(280.0, 310.0, (n_time, n_lat, n_lon)),
            )
        },
        coords={
            "valid_time": pd.date_range("2021-06-20", periods=n_time, freq="6h"),
            "latitude": np.linspace(30.0, 60.0, n_lat),
            "longitude": np.linspace(-130.0, -100.0, n_lon),
        },
    )


class _PassThroughCase:
    """Case metadata whose mask is the identity, to isolate the chunking."""

    def __init__(self, data):
        self.start_date = pd.Timestamp("2021-06-20")
        self.end_date = pd.Timestamp("2021-06-29")
        self.location = self

    def mask(self, data, drop=False):
        return data


class TestZarrTargetSubsetterChunking:
    """An unchunked source must not collapse into a single chunk.

    ``chunk()`` with no arguments puts the whole array in one chunk, so every
    later step over that target runs single-threaded and has to hold the
    entire case window in memory at once.
    """

    def test_unchunked_source_is_split_along_time(self):
        from extremeweatherbench import inputs

        data = _unchunked_target()
        with dask.config.set({"array.chunk-size": "256kiB"}):
            result = inputs.zarr_target_subsetter(data, _PassThroughCase(data))
        time_chunks = result.chunks["valid_time"]
        assert len(time_chunks) > 1, (
            f"expected the case window to be split along valid_time, got a "
            f"single chunk of {time_chunks}"
        )

    def test_space_and_level_stay_whole(self):
        """Vertical integration requires whole columns, so level is not split."""
        from extremeweatherbench import inputs

        data = _unchunked_target()
        data = data.expand_dims(level=[1000.0, 850.0, 500.0])
        with dask.config.set({"array.chunk-size": "256kiB"}):
            result = inputs.zarr_target_subsetter(data, _PassThroughCase(data))
        assert len(result.chunks["level"]) == 1
        assert len(result.chunks["latitude"]) == 1
        assert len(result.chunks["longitude"]) == 1

    def test_already_chunked_source_is_left_alone(self):
        from extremeweatherbench import inputs

        data = _unchunked_target().chunk({"valid_time": 7})
        case = _PassThroughCase(data)
        result = inputs.zarr_target_subsetter(data, case)
        expected = data.sel(valid_time=slice(case.start_date, case.end_date))
        assert result.chunks["valid_time"] == expected.chunks["valid_time"]


class TestZarrTargetSubsetterOutput:
    """Characterization: chunking must not change the subset values."""

    def test_values_match_the_unchunked_subset(self):
        from extremeweatherbench import inputs

        data = _unchunked_target()
        case = _PassThroughCase(data)
        with dask.config.set({"array.chunk-size": "256kiB"}):
            result = inputs.zarr_target_subsetter(data, case)
        expected = data.sel(valid_time=slice(case.start_date, case.end_date))
        xr.testing.assert_identical(result.compute(), expected)


def _column_dataarray(n_time=4, n_lat=3, n_lon=3, levels=(1000.0, 850.0, 700.0, 500.0)):
    """Array with a level dimension, as the vertical integrals expect."""
    rng = np.random.default_rng(7)
    shape = (n_time, len(levels), n_lat, n_lon)
    return xr.DataArray(
        rng.uniform(0.001, 0.02, shape),
        dims=["valid_time", "level", "latitude", "longitude"],
        coords={
            "valid_time": pd.date_range("2023-01-01", periods=n_time, freq="6h"),
            "level": np.array(levels),
            "latitude": np.linspace(30.0, 40.0, n_lat),
            "longitude": np.linspace(-120.0, -110.0, n_lon),
        },
    )


class TestVerticalIntegralChunkingPrecondition:
    """The level -1 rechunk is required for correctness, not just for speed.

    nantrapezoid_nd integrates whichever levels it is handed, so it has to be
    handed a whole column. These tests pin the rechunk as the thing that
    guarantees that, so it is not later mistaken for redundant work: without
    it apply_ufunc refuses to build the graph, and the answer a partial
    column would give is genuinely different.
    """

    def test_level_is_whole_in_every_chunk_the_kernel_sees(self):
        da = _column_dataarray().chunk({"valid_time": 1, "level": 2})
        seen = []

        original = calc.nantrapezoid_nd

        def recording(values, levels):
            seen.append(values.shape[-1])
            return original(values, levels)

        calc.nantrapezoid_nd = recording
        try:
            calc.nantrapezoid_pressure_levels(da).compute()
        finally:
            calc.nantrapezoid_nd = original

        assert seen, "the integration kernel never ran"
        assert set(seen) == {da.sizes["level"]}, (
            f"kernel saw partial columns of sizes {sorted(set(seen))}; it must "
            f"always see all {da.sizes['level']} levels"
        )

    def test_without_the_rechunk_apply_ufunc_refuses_to_build(self):
        """The same call without the rechunk is an error, not a slow path."""
        da = _column_dataarray().chunk({"level": 2})
        with pytest.raises(ValueError, match="core dimension"):
            xr.apply_ufunc(
                calc.nantrapezoid_nd,
                da,
                da["level"] * 100,
                input_core_dims=[["level"], ["level"]],
                output_core_dims=[[]],
                dask="parallelized",
                output_dtypes=[float],
            )

    def test_a_partial_column_integrates_to_a_different_value(self):
        """Shows the rechunk is load-bearing rather than cosmetic."""
        da = _column_dataarray()
        levels_pa = (da["level"] * 100).values
        column = da.isel(valid_time=0, latitude=0, longitude=0).values

        whole = calc.nantrapezoid_nd(column, levels_pa)
        halves = calc.nantrapezoid_nd(column[:2], levels_pa[:2]) + calc.nantrapezoid_nd(
            column[2:], levels_pa[2:]
        )

        assert not np.isclose(whole, halves), (
            "integrating two half columns matched the whole column, so this "
            "test no longer demonstrates why the rechunk is needed"
        )

    def test_rechunk_is_applied_even_when_level_arrives_split(self):
        da = _column_dataarray().chunk({"level": 1})
        result = calc.nantrapezoid_pressure_levels(da)
        opt.assert_lazy(result)
        expected = calc.nantrapezoid_pressure_levels(
            _column_dataarray().chunk({"level": -1})
        )
        np.testing.assert_allclose(result.compute().values, expected.compute().values)


_CAPE_LEVELS = np.array(
    [
        1000.0,
        975,
        950,
        925,
        900,
        850,
        800,
        750,
        700,
        650,
        600,
        550,
        500,
        450,
        400,
        350,
        300,
        250,
        200,
        150,
        100,
    ]
)


def _reference_moist_adiabat(p_target, p_lcl, t_lcl, steps):
    """Forward-Euler moist adiabat from the LCL, at arbitrary resolution.

    Deliberately written out here rather than imported so it stays a fixed
    reference: refining `steps` converges on the true adiabat, so it can
    judge whether a change to the production scheme moved toward or away
    from the right answer.
    """
    from extremeweatherbench import _cape

    if p_target >= p_lcl:
        return t_lcl * (p_target / p_lcl) ** _cape.KAPPA

    d_log_p = (np.log(p_target) - np.log(p_lcl)) / steps
    t = t_lcl
    log_p = np.log(p_lcl)
    for _ in range(steps):
        e_s = _cape.saturation_vapor_pressure_inline(t)
        w_s = _cape.mixing_ratio_inline(np.exp(log_p), e_s)
        latent = _cape.L_V_0 - _cape.L_V_TEMP_COEFF * (t - _cape.KELVIN_TO_CELSIUS)
        numerator = 1.0 + latent * w_s / (_cape.Rd * t)
        denominator = 1.0 + latent * latent * w_s * _cape.EPSILON / (
            _cape.Cp * _cape.Rd * t * t
        )
        t += _cape.KAPPA * t * numerator / denominator * d_log_p
        log_p += d_log_p
    return t


class TestMoistAscentCost:
    """The parcel ascent must not restart from the LCL at every level.

    Restarting means the lower part of the adiabat is re-integrated once per
    level above the LCL, and because each restart spreads a fixed step budget
    over a longer interval, the steps also get coarser the higher you go. A
    single march up the profile does less work *and* keeps the steps short.
    """

    P_LCL = 900.0
    T_LCL = 295.0

    def _targets(self):
        return _CAPE_LEVELS[_CAPE_LEVELS < self.P_LCL]

    def test_ascent_is_closer_to_a_high_resolution_reference(self):
        from extremeweatherbench import _cape

        targets = self._targets()
        reference = np.array(
            [
                _reference_moist_adiabat(p, self.P_LCL, self.T_LCL, 40000)
                for p in targets
            ]
        )
        restarting = np.array(
            [
                _reference_moist_adiabat(
                    p, self.P_LCL, self.T_LCL, _cape.MOIST_ASCENT_STEPS
                )
                for p in targets
            ]
        )

        marched = np.empty(targets.size)
        t = self.T_LCL
        p_previous = self.P_LCL
        for i, p in enumerate(targets):
            t = _cape.moist_ascent_gap(p, p_previous, t)
            marched[i] = t
            p_previous = p

        restart_error = np.abs(restarting - reference).max()
        march_error = np.abs(marched - reference).max()
        assert march_error < restart_error, (
            f"marching the adiabat should track the reference better than "
            f"restarting from the LCL, but got {march_error:.4f} K of error "
            f"against {restart_error:.4f} K"
        )

    def test_marching_uses_fewer_integration_steps(self):
        from extremeweatherbench import _cape

        n_above_lcl = self._targets().size
        restarting_steps = n_above_lcl * _cape.MOIST_ASCENT_STEPS
        marching_steps = n_above_lcl * _cape.MOIST_ASCENT_SUBSTEPS
        assert marching_steps < restarting_steps

    def test_a_single_gap_matches_integrating_that_gap_directly(self):
        from extremeweatherbench import _cape

        got = _cape.moist_ascent_gap(850.0, self.P_LCL, self.T_LCL)
        expected = _reference_moist_adiabat(
            850.0, self.P_LCL, self.T_LCL, _cape.MOIST_ASCENT_SUBSTEPS
        )
        assert got == pytest.approx(expected, rel=1e-12)

    def test_a_zero_width_gap_leaves_the_parcel_alone(self):
        from extremeweatherbench import _cape

        assert _cape.moist_ascent_gap(
            self.P_LCL, self.P_LCL, self.T_LCL
        ) == pytest.approx(self.T_LCL, rel=1e-12)


class TestCapeOutput:
    """Characterization: CAPE stays physically sound as the ascent changes."""

    def _profile(self):
        surface_t = 305.0
        temperature = surface_t - np.linspace(0.0, 75.0, _CAPE_LEVELS.size)
        dewpoint = temperature - np.linspace(3.0, 30.0, _CAPE_LEVELS.size)
        geopotential = np.linspace(100.0, 160000.0, _CAPE_LEVELS.size)
        return _CAPE_LEVELS.copy(), temperature, dewpoint, geopotential

    def test_convective_profile_still_has_cape_and_no_cin(self):
        from extremeweatherbench._cape import compute_ml_cape_cin_from_profile

        cape, cin = compute_ml_cape_cin_from_profile(*self._profile())
        assert cape > 500.0
        assert cin >= 0.0

    def test_batched_and_single_profile_agree(self):
        from extremeweatherbench._cape import (
            compute_ml_cape_cin_batched,
            compute_ml_cape_cin_from_profile,
        )

        pressure, temperature, dewpoint, geopotential = self._profile()
        single = compute_ml_cape_cin_from_profile(
            pressure, temperature, dewpoint, geopotential
        )
        batched = compute_ml_cape_cin_batched(
            np.ascontiguousarray(pressure[None, :]),
            np.ascontiguousarray(temperature[None, :]),
            np.ascontiguousarray(dewpoint[None, :]),
            np.ascontiguousarray(geopotential[None, :]),
        )
        np.testing.assert_allclose(batched[0][0], single[0], rtol=1e-10)
        np.testing.assert_allclose(batched[1][0], single[1], rtol=1e-10)

    def test_cape_stays_within_a_percent_of_the_restarting_scheme(self):
        """Marching shifts the parcel path slightly; it must not reshape it.

        The reference value was recorded from the restart-per-level scheme.
        Marching integrates the same adiabat with shorter steps, so a small
        move is expected and a large one means something else changed.
        """
        from extremeweatherbench._cape import compute_ml_cape_cin_from_profile

        cape, _ = compute_ml_cape_cin_from_profile(*self._profile())
        assert cape == pytest.approx(3894.9, rel=0.01)

    def test_reversed_pressure_order_gives_no_cape(self):
        from extremeweatherbench._cape import compute_ml_cape_cin_from_profile

        pressure, temperature, dewpoint, geopotential = self._profile()
        cape, cin = compute_ml_cape_cin_from_profile(
            pressure[::-1].copy(),
            temperature[::-1].copy(),
            dewpoint[::-1].copy(),
            geopotential[::-1].copy(),
        )
        assert cape == 0.0


def _tc_grid(n_lat=181, n_lon=360):
    """Global-ish grid, so a local storm covers a small part of it."""
    return (
        np.linspace(-90.0, 90.0, n_lat),
        np.linspace(0.0, 359.0, n_lon),
    )


def _tc_track_frame(n_rows=4, latitude=15.0, longitude=140.0):
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2021-08-01", periods=n_rows, freq="6h"),
            "latitude": np.full(n_rows, latitude),
            "longitude": np.full(n_rows, longitude),
        }
    )


class TestSpatialMaskWork:
    """The storm-proximity mask must not sweep the whole globe per track row.

    Each track row only marks points within a few degrees of itself, so
    evaluating the distance formula at every grid point for every row does
    work proportional to rows x whole grid to fill in a small patch.
    """

    def test_distance_is_not_evaluated_over_the_whole_grid(self):
        from extremeweatherbench.events import tropical_cyclone as tc

        lat_coords, lon_coords = _tc_grid()
        frame = _tc_track_frame(n_rows=4)
        full_grid_points = lat_coords.size * lon_coords.size

        evaluated = []
        original = calc.haversine_distance

        def recording(input_a, input_b, units="km"):
            result = original(input_a, input_b, units=units)
            evaluated.append(np.size(result))
            return result

        calc.haversine_distance = recording
        try:
            tc._create_spatial_mask(lat_coords, lon_coords, frame, 5.0)
        finally:
            calc.haversine_distance = original

        assert evaluated, "no distance was evaluated"
        assert max(evaluated) < full_grid_points, (
            f"the mask evaluated distance at {max(evaluated)} points, the whole "
            f"{full_grid_points}-point grid; it should be confined to the band "
            f"the search radius can reach"
        )

    def test_mask_matches_the_whole_grid_calculation(self):
        """Characterization: narrowing the search must not narrow the mask."""
        from extremeweatherbench.events import tropical_cyclone as tc

        lat_coords, lon_coords = _tc_grid(n_lat=91, n_lon=180)
        frame = pd.DataFrame(
            {
                "valid_time": pd.date_range("2021-08-01", periods=3, freq="6h"),
                "latitude": [12.0, 18.0, -40.0],
                "longitude": [140.0, 145.0, 300.0],
            }
        )
        radius = 6.0

        got = tc._create_spatial_mask(lat_coords, lon_coords, frame, radius)

        lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing="ij")
        expected = np.zeros_like(lat_grid, dtype=bool)
        for _, row in frame.iterrows():
            distances = calc.haversine_distance(
                [lat_grid, lon_grid],
                [row["latitude"], row["longitude"]],
                units="degrees",
            )
            expected |= distances <= radius

        np.testing.assert_array_equal(got, expected)

    def test_empty_track_frame_gives_an_empty_mask(self):
        from extremeweatherbench.events import tropical_cyclone as tc

        lat_coords, lon_coords = _tc_grid(n_lat=20, n_lon=40)
        frame = _tc_track_frame(n_rows=0)
        mask = tc._create_spatial_mask(lat_coords, lon_coords, frame, 5.0)
        assert mask.shape == (20, 40)
        assert not mask.any()

    def test_a_row_outside_the_grid_marks_nothing(self):
        from extremeweatherbench.events import tropical_cyclone as tc

        lat_coords = np.linspace(0.0, 20.0, 21)
        lon_coords = np.linspace(100.0, 160.0, 61)
        frame = _tc_track_frame(n_rows=1, latitude=80.0, longitude=130.0)
        mask = tc._create_spatial_mask(lat_coords, lon_coords, frame, 5.0)
        assert not mask.any()


def _daily_series(n_days, timesteps_per_day=4, chunk=True, drop_last=0):
    """Hourly-ish series over whole days, optionally with a truncated last day."""
    n_steps = n_days * timesteps_per_day - drop_last
    freq = f"{24 // timesteps_per_day}h"
    valid_time = pd.date_range("2021-06-01", periods=n_steps, freq=freq)
    values = np.arange(float(n_steps))
    da = xr.DataArray(values, dims=["valid_time"], coords={"valid_time": valid_time})
    if chunk:
        da = da.chunk({"valid_time": timesteps_per_day})
    return da


def _daily_min_via_map(da, time_resolution_hours):
    """The groupby().map() formulation this phase replaces."""
    return da.groupby("valid_time.dayofyear").map(
        utils.min_if_all_timesteps_present,
        time_resolution_hours=time_resolution_hours,
    )


class TestDailyMinAggregation:
    """Phase 10: the per-day minimum should be one grouped reduction."""

    def test_result_is_still_lazy(self):
        da = _daily_series(n_days=8)
        result = utils.daily_min_over_complete_days(da, 6.0)
        opt.assert_lazy(
            result, "the daily minimum should not materialize on construction"
        )

    def test_no_python_callback_runs_once_per_day(self):
        da = _daily_series(n_days=16)
        with opt.spy(utils, "min_if_all_timesteps_present") as calls:
            utils.daily_min_over_complete_days(da, 6.0).compute()
        assert calls.count == 0, (
            f"the per-day helper ran {calls.count} times; the daily minimum "
            "should be one reduction over the whole series"
        )

    def test_graph_is_smaller_than_the_groupby_map_graph(self):
        da = _daily_series(n_days=32)
        mine = opt.graph_size(utils.daily_min_over_complete_days(da, 6.0))
        theirs = opt.graph_size(_daily_min_via_map(da, 6.0))
        assert mine < theirs, (
            f"graph has {mine} tasks against {theirs} for groupby().map(), so "
            "the per-day concatenation has not actually been removed"
        )

    def test_a_dask_backed_series_does_not_hit_the_flox_grouped_path(self):
        # flox 0.10.8 raises "Cannot call len() on object with unknown chunk
        # size" for any dask-backed grouped reduction against dask 2025.12, so
        # the daily minimum must reach its answer without one.
        da = _daily_series(n_days=8)
        result = utils.daily_min_over_complete_days(da, 6.0).compute()
        np.testing.assert_allclose(result.values, np.arange(8) * 4.0)

    def test_complete_days_match_the_groupby_map_result(self):
        da = _daily_series(n_days=5, chunk=False)
        got = utils.daily_min_over_complete_days(da, 6.0)
        expected = _daily_min_via_map(da, 6.0)
        np.testing.assert_allclose(got.values, expected.values)

    def test_an_incomplete_day_is_nan(self):
        da = _daily_series(n_days=5, chunk=False, drop_last=1)
        got = utils.daily_min_over_complete_days(da, 6.0)

        assert np.isnan(got.values[-1]), "the truncated last day should be NaN"
        np.testing.assert_allclose(got.values[:-1], [0.0, 4.0, 8.0, 12.0])

    def test_lead_time_is_preserved_for_forecast_shaped_input(self):
        base = _daily_series(n_days=4, chunk=False)
        da = base.expand_dims(lead_time=[0, 6, 12]).copy()
        got = utils.daily_min_over_complete_days(da, 6.0)

        assert got.dims == ("lead_time", "dayofyear")
        np.testing.assert_allclose(got.sel(lead_time=6).values, [0.0, 4.0, 8.0, 12.0])

    def test_days_stay_matched_to_their_labels_across_a_year_boundary(self):
        # dayofyear restarts below the days already seen, so the day labels no
        # longer arrive in time order.
        da = _daily_series(n_days=400, chunk=False)
        got = utils.daily_min_over_complete_days(da, 6.0)
        expected = _daily_min_via_map(da, 6.0)

        np.testing.assert_array_equal(got.dayofyear.values, expected.dayofyear.values)
        np.testing.assert_allclose(got.values, expected.values)

    def test_all_days_incomplete_gives_nan_on_the_day_axis(self):
        da = _daily_series(n_days=1, chunk=False, drop_last=1)
        got = utils.daily_min_over_complete_days(da, 6.0)
        assert got.dims == ("dayofyear",)
        assert np.isnan(got.values).all()


def _track_dataarray(lons, lats, chunk=False):
    """Track-shaped DataArray with latitude/longitude along valid_time."""
    valid_time = pd.date_range("2021-08-28", periods=len(lons), freq="6h")
    da = xr.DataArray(
        np.arange(float(len(lons))),
        dims=["valid_time"],
        coords={
            "valid_time": valid_time,
            "latitude": ("valid_time", np.asarray(lats, dtype=float)),
            "longitude": ("valid_time", np.asarray(lons, dtype=float)),
        },
    )
    return da.chunk({"valid_time": 4}) if chunk else da


def _gulf_coast_track(n_points):
    """Track running north out of the Gulf of Mexico onto Louisiana."""
    lons = np.linspace(-91.5, -91.0, n_points)
    lats = np.linspace(24.0, 32.0, n_points)
    return lons, lats


class TestLandfallDetectionScaling:
    """Phase 8: landfall detection should not test points one at a time."""

    def test_no_per_point_python_predicate(self):
        lons, lats = _gulf_coast_track(48)
        track = _track_dataarray(lons, lats)
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()

        with opt.spy(calc, "_is_true_landfall") as calls:
            calc._detect_landfalls_wrapper(track, land, ocean)

        assert calls.count == 0, (
            f"the scalar landfall predicate ran {calls.count} times; the "
            "point-in-polygon tests should be vectorized over the track"
        )

    def test_predicate_cost_does_not_scale_with_track_length(self):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()

        counts = []
        for n_points in (16, 256):
            track = _track_dataarray(*_gulf_coast_track(n_points))
            with opt.spy(shapely, "contains_xy") as calls:
                calc._detect_landfalls_wrapper(track, land, ocean)
            counts.append(calls.count)

        assert counts[0] > 0, (
            "contains_xy was never called, so the point tests are still going "
            "through Python one pair at a time"
        )
        assert counts[0] == counts[1], (
            f"contains_xy ran {counts} times for 16 and 256 track points; the "
            "number of predicate calls should not depend on track length"
        )


class TestLandfallDetectionOutput:
    """Phase 8: the detected landfalls must not move."""

    @staticmethod
    def _reference_mask(track, land, ocean):
        lats = track.coords["latitude"].values
        lons = (track.coords["longitude"].values + 180) % 360 - 180
        mask = np.zeros(lons.size, dtype=bool)
        for i in range(lons.size - 1):
            mask[i] = calc._is_true_landfall(
                lons[i], lats[i], lons[i + 1], lats[i + 1], land, ocean
            )
        return mask

    @pytest.mark.parametrize("n_points", [8, 33, 60])
    def test_mask_matches_the_scalar_predicate(self, n_points):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()
        track = _track_dataarray(*_gulf_coast_track(n_points))

        got = calc._detect_landfalls_wrapper(track, land, ocean)
        expected = self._reference_mask(track, land, ocean)

        np.testing.assert_array_equal(got.values, expected)
        assert expected.any(), "this track should make landfall somewhere"

    def test_a_track_that_stays_at_sea_finds_nothing(self):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()
        lons = np.linspace(-40.0, -38.0, 12)
        lats = np.full(12, 25.0)
        track = _track_dataarray(lons, lats)

        got = calc._detect_landfalls_wrapper(track, land, ocean)
        assert not got.values.any()

    def test_a_track_starting_over_land_is_not_a_landfall(self):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()
        # Kansas outward; leaving land is not landfall, and no point starts
        # in the open ocean.
        lons = np.linspace(-98.0, -96.0, 10)
        lats = np.full(10, 38.5)
        track = _track_dataarray(lons, lats)

        got = calc._detect_landfalls_wrapper(track, land, ocean)
        assert not got.values.any()

    def test_nan_positions_are_not_landfalls(self):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()
        lons, lats = _gulf_coast_track(12)
        lons = lons.copy()
        lats = lats.copy()
        lons[3] = np.nan
        lats[7] = np.nan
        track = _track_dataarray(lons, lats)

        got = calc._detect_landfalls_wrapper(track, land, ocean)
        # A pair with a NaN endpoint cannot be evaluated, so neither the pair
        # starting at the NaN nor the one ending at it may be flagged.
        assert not got.values[2:4].any()
        assert not got.values[6:8].any()

    def test_longitudes_given_in_0_360_are_wrapped(self):
        land = utils.load_land_geometry()
        ocean = utils.load_ocean_geometry()
        lons, lats = _gulf_coast_track(24)
        signed = _track_dataarray(lons, lats)
        unsigned = _track_dataarray(lons % 360, lats)

        np.testing.assert_array_equal(
            calc._detect_landfalls_wrapper(signed, land, ocean).values,
            calc._detect_landfalls_wrapper(unsigned, land, ocean).values,
        )


def _pattern_stack(n_lead=3, n_valid=4, n_lat=9, n_lon=11, chunk=False, seed=0):
    """Stack of 2-D non-negative fields, the shape SpatialDisplacement sees."""
    rng = np.random.default_rng(seed)
    values = rng.random((n_lead, n_valid, n_lat, n_lon))
    values[values < 0.4] = 0.0
    da = xr.DataArray(
        values,
        dims=["lead_time", "valid_time", "latitude", "longitude"],
        coords={
            "lead_time": np.arange(0, n_lead * 6, 6),
            "valid_time": pd.date_range("2021-02-01", periods=n_valid, freq="6h"),
            "latitude": np.linspace(30.0, 46.0, n_lat),
            "longitude": np.linspace(-130.0, -110.0, n_lon),
        },
    )
    return da.chunk({"lead_time": 1, "valid_time": 2}) if chunk else da


def _center_of_mass_reference(da):
    """Per-slice ndimage.center_of_mass, the formulation being replaced."""
    from scipy import ndimage as ndi

    values = da.transpose("lead_time", "valid_time", "latitude", "longitude").values
    lat_idx = np.empty(values.shape[:2])
    lon_idx = np.empty(values.shape[:2])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            field = values[i, j]
            if (field > 0).any():
                lat_idx[i, j], lon_idx[i, j] = ndi.center_of_mass(field)
            else:
                lat_idx[i, j], lon_idx[i, j] = np.nan, np.nan
    return lat_idx, lon_idx


class TestCenterOfMassLaziness:
    """Phase 9: the center of mass should be a reduction, not a per-slice loop."""

    def test_result_is_lazy_for_dask_input(self):
        da = _pattern_stack(chunk=True)
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        opt.assert_lazy(lat_idx, "the center of mass should compose lazily")
        opt.assert_lazy(lon_idx, "the center of mass should compose lazily")

    def test_ndimage_center_of_mass_is_not_called_per_slice(self):
        da = _pattern_stack(n_lead=4, n_valid=5, chunk=True)
        with opt.spy(ndimage, "center_of_mass") as calls:
            lat_idx, _ = metrics._center_of_mass_indices(da)
            lat_idx.compute()
        assert calls.count == 0, (
            f"ndimage.center_of_mass ran {calls.count} times; the center of "
            "mass is a weighted mean and should be one reduction"
        )


class TestCenterOfMassOutput:
    """Phase 9: the center of mass values must not move."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_indices_match_ndimage(self, seed):
        da = _pattern_stack(seed=seed)
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        expected_lat, expected_lon = _center_of_mass_reference(da)

        np.testing.assert_allclose(lat_idx.values, expected_lat, rtol=1e-12)
        np.testing.assert_allclose(lon_idx.values, expected_lon, rtol=1e-12)

    def test_an_all_zero_field_is_nan(self):
        da = _pattern_stack(n_lead=1, n_valid=2)
        da = xr.zeros_like(da)
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        assert np.isnan(lat_idx.values).all()
        assert np.isnan(lon_idx.values).all()

    def test_a_single_lit_cell_sits_on_that_cell(self):
        da = xr.zeros_like(_pattern_stack(n_lead=1, n_valid=1))
        da[0, 0, 3, 7] = 1.0
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        assert lat_idx.values.item() == 3.0
        assert lon_idx.values.item() == 7.0

    def test_a_nan_in_the_field_propagates_like_ndimage(self):
        da = _pattern_stack(n_lead=1, n_valid=1)
        da[0, 0, 2, 2] = np.nan
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        expected_lat, expected_lon = _center_of_mass_reference(da)

        np.testing.assert_array_equal(np.isnan(lat_idx.values), np.isnan(expected_lat))
        np.testing.assert_array_equal(np.isnan(lon_idx.values), np.isnan(expected_lon))

    def test_dask_and_numpy_paths_agree(self):
        eager = _pattern_stack(chunk=False)
        lazy = _pattern_stack(chunk=True)
        lat_eager, lon_eager = metrics._center_of_mass_indices(eager)
        lat_lazy, lon_lazy = metrics._center_of_mass_indices(lazy)

        np.testing.assert_allclose(lat_eager.values, lat_lazy.compute().values)
        np.testing.assert_allclose(lon_eager.values, lon_lazy.compute().values)


class TestSpatialDisplacementOutput:
    """Phase 9: the metric result must not move."""

    def test_displacement_matches_the_per_slice_formulation(self):
        forecast = _pattern_stack(seed=3)
        target = _pattern_stack(seed=4)
        metric = metrics.SpatialDisplacement(preserve_dims=["lead_time"])

        got = metric._compute_metric(forecast=forecast, target=target)
        expected = _spatial_displacement_reference(forecast, target, ["lead_time"])

        np.testing.assert_allclose(got.values, expected, rtol=1e-10)

    def test_result_keeps_the_preserved_dimension(self):
        forecast = _pattern_stack(seed=5)
        target = _pattern_stack(seed=6)
        metric = metrics.SpatialDisplacement(preserve_dims=["lead_time"])
        got = metric._compute_metric(forecast=forecast, target=target)
        assert got.dims == ("lead_time",)


def _spatial_displacement_reference(forecast, target, preserve_dims):
    """SpatialDisplacement computed through the per-slice center of mass."""
    pieces = []
    for da in (forecast, target):
        lat_idx, lon_idx = _center_of_mass_reference(da)
        lat_coords, lon_coords = utils.idx_to_coords(
            np.round(lat_idx),
            np.round(lon_idx),
            da.latitude.values,
            da.longitude.values,
        )
        pieces.append(np.array([lat_coords, lon_coords]))

    distance = calc.haversine_distance(pieces[0], pieces[1])
    result = xr.DataArray(
        distance,
        coords={
            "lead_time": forecast.lead_time,
            "valid_time": forecast.valid_time,
        },
        dims=["lead_time", "valid_time"],
    )
    reduce_dims = [d for d in result.dims if d not in preserve_dims]
    if reduce_dims:
        result = result.mean(dim=reduce_dims)
    return result.values


def _landfall_dataarray(n_init=64, chunk=True, all_nan=False):
    """Landfall-shaped DataArray indexed by init_time."""
    values = np.full(n_init, np.nan) if all_nan else np.arange(float(n_init))
    da = xr.DataArray(
        values,
        dims=["init_time"],
        coords={"init_time": pd.date_range("2021-08-20", periods=n_init, freq="6h")},
    )
    return da.chunk({"init_time": 8}) if chunk else da


class TestLandfallValidityGuard:
    """Phase 11: the validity guard should reduce, not materialize."""

    def test_guard_does_not_materialize_the_array(self):
        da = _landfall_dataarray(n_init=64, chunk=True)
        with opt.spy_largest_materialized_input(np, "isnan") as seen:
            utils.is_valid_landfall(da)

        assert seen.largest == 0, (
            f"the null test was handed a materialized array of {seen.largest} "
            "elements; the guard should fold into the dask graph instead"
        )

    @pytest.mark.parametrize("chunk", [False, True])
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, True),
            ({"all_nan": True}, False),
        ],
    )
    def test_verdict_is_unchanged(self, chunk, kwargs, expected):
        da = _landfall_dataarray(n_init=16, chunk=chunk, **kwargs)
        assert utils.is_valid_landfall(da) is expected

    def test_none_and_scalar_are_still_invalid(self):
        assert utils.is_valid_landfall(None) is False
        assert utils.is_valid_landfall(xr.DataArray(np.nan)) is False

    def test_missing_init_time_is_invalid(self):
        da = xr.DataArray([1.0, 2.0], dims=["landfall"])
        assert utils.is_valid_landfall(da) is False

    def test_a_single_real_value_among_nans_is_valid(self):
        da = _landfall_dataarray(n_init=16, chunk=True, all_nan=True)
        da = da.copy()
        da[5] = 3.0
        assert utils.is_valid_landfall(da) is True


def _level_dataset(chunk=True):
    """Pressure-level dataset shaped like the AR derived-variable input."""
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            name: (
                ["valid_time", "level", "latitude", "longitude"],
                rng.random((6, levels.size, 12, 16), dtype="float32"),
            )
            for name in ("specific_humidity", "eastward_wind", "northward_wind")
        },
        coords={
            "valid_time": pd.date_range("2021-01-01", periods=6, freq="6h"),
            "level": levels,
            "latitude": np.linspace(20.0, 60.0, 12),
            "longitude": np.linspace(-160.0, -120.0, 16),
        },
    )
    return ds.chunk({"valid_time": 2, "level": -1}) if chunk else ds


class TestLevelSubsetting:
    """Phase 11: the pressure-level subset should stay a contiguous slice."""

    def test_costs_no_more_graph_than_a_contiguous_slice(self):
        from extremeweatherbench import derived

        ds = _level_dataset()
        top = 300
        result = derived._subset_to_top_pressure_level(ds, top)

        keep = np.flatnonzero(ds.indexes["level"].to_numpy() >= top)
        reference = ds.isel(level=slice(keep[0], keep[-1] + 1))

        assert opt.graph_size(result["specific_humidity"]) == opt.graph_size(
            reference["specific_humidity"]
        ), (
            "selecting the levels by boolean mask adds a shuffle that a "
            "contiguous slice does not"
        )

    def test_selected_levels_are_unchanged(self):
        from extremeweatherbench import derived

        ds = _level_dataset(chunk=False)
        top = 300
        result = derived._subset_to_top_pressure_level(ds, top)
        expected = ds.sel(level=ds.level[ds.level >= top])

        np.testing.assert_array_equal(result.level.values, expected.level.values)
        xr.testing.assert_identical(result, expected)

    def test_ascending_levels_select_the_same_set(self):
        from extremeweatherbench import derived

        ds = _level_dataset(chunk=False).isel(level=slice(None, None, -1))
        top = 300
        result = derived._subset_to_top_pressure_level(ds, top)
        expected = ds.sel(level=ds.level[ds.level >= top])

        np.testing.assert_array_equal(result.level.values, expected.level.values)

    def test_non_monotonic_levels_fall_back_to_the_mask(self):
        from extremeweatherbench import derived

        ds = _level_dataset(chunk=False)
        shuffled = ds.isel(level=[0, 5, 1, 9, 2, 11, 3, 4, 6, 7, 8, 10])
        top = 300
        result = derived._subset_to_top_pressure_level(shuffled, top)
        expected = shuffled.sel(level=shuffled.level[shuffled.level >= top])

        np.testing.assert_array_equal(result.level.values, expected.level.values)

    def test_every_level_kept_when_top_is_below_the_grid(self):
        from extremeweatherbench import derived

        ds = _level_dataset(chunk=False)
        result = derived._subset_to_top_pressure_level(ds, 50)
        np.testing.assert_array_equal(result.level.values, ds.level.values)


class TestCoordinateOrderGuards:
    """Phase 11: coordinate order is read off the index, not via DataArrays."""

    @pytest.mark.parametrize("descending", [False, True])
    def test_region_mask_handles_both_latitude_directions(self, descending):
        from extremeweatherbench import regions

        latitudes = np.linspace(-90, 90, 73)
        if descending:
            latitudes = latitudes[::-1]
        ds = xr.Dataset(
            {"t": (["latitude", "longitude"], np.zeros((73, 144)))},
            coords={"latitude": latitudes, "longitude": np.linspace(-180, 177.5, 144)},
        )
        region = regions.BoundingBoxRegion(
            latitude_min=20.0,
            latitude_max=50.0,
            longitude_min=-130.0,
            longitude_max=-100.0,
        )
        masked = region.mask(ds)

        kept = masked.latitude.values
        assert kept.size > 0, "the latitude slice came back empty"
        assert kept.min() >= 20.0 and kept.max() <= 50.0

    @pytest.mark.parametrize("hours", [1, 3, 6, 12])
    def test_temporal_resolution_is_unchanged(self, hours):
        ds = xr.Dataset(
            {"t": ("valid_time", np.zeros(10))},
            coords={
                "valid_time": pd.date_range("2021-01-01", periods=10, freq=f"{hours}h")
            },
        )
        assert utils.determine_temporal_resolution(ds) == float(hours)

    def test_temporal_resolution_of_a_single_timestep_is_none(self):
        ds = xr.Dataset(
            {"t": ("valid_time", np.zeros(1))},
            coords={"valid_time": pd.date_range("2021-01-01", periods=1)},
        )
        assert utils.determine_temporal_resolution(ds) is None

    def test_temporal_resolution_takes_the_finest_of_mixed_spacings(self):
        times = pd.to_datetime(
            ["2021-01-01T00", "2021-01-01T06", "2021-01-01T18", "2021-01-02T00"]
        )
        ds = xr.Dataset(
            {"t": ("valid_time", np.zeros(4))}, coords={"valid_time": times}
        )
        assert utils.determine_temporal_resolution(ds) == 6.0

    def test_temporal_resolution_works_without_a_valid_time_index(self):
        ds = xr.Dataset(
            {"t": ("step", np.zeros(5))},
            coords={
                "step": np.arange(5),
                "valid_time": (
                    "step",
                    pd.date_range("2021-01-01", periods=5, freq="3h"),
                ),
            },
        )
        assert utils.determine_temporal_resolution(ds) == 3.0


@pytest.fixture
def shapefile_region(tmp_path):
    """A ShapefileRegion backed by a small on-disk box shapefile."""
    import geopandas as gpd

    from extremeweatherbench import regions

    frame = gpd.GeoDataFrame(
        {"name": ["box"]},
        geometry=[shapely.geometry.box(-130.0, 20.0, -100.0, 50.0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "box.shp"
    frame.to_file(path)
    return regions.ShapefileRegion(path)


def _global_grid_dataset():
    """Coarse global grid for region masking."""
    return xr.Dataset(
        {"t": (["latitude", "longitude"], np.zeros((73, 144)))},
        coords={
            "latitude": np.linspace(-90, 90, 73),
            "longitude": np.linspace(-180, 177.5, 144),
        },
    )


class TestShapefileCaching:
    """Phase 12: a shapefile should be read from disk once."""

    def test_read_once_within_a_single_mask_call(self, shapefile_region):
        import geopandas as gpd

        ds = _global_grid_dataset()
        with opt.spy(gpd, "read_file") as calls:
            shapefile_region.mask(ds)

        assert calls.count <= 1, (
            f"the shapefile was read {calls.count} times for one mask call; "
            "the bounds lookup and the mask should share one read"
        )

    def test_read_once_across_two_mask_calls(self, shapefile_region):
        import geopandas as gpd

        ds = _global_grid_dataset()
        with opt.spy(gpd, "read_file") as calls:
            shapefile_region.mask(ds)
            shapefile_region.mask(ds)

        assert calls.count <= 1, (
            f"the shapefile was read {calls.count} times across two mask "
            "calls; the parsed frame should be held on the region"
        )

    def test_as_geopandas_returns_equal_frames_each_time(self, shapefile_region):
        first = shapefile_region.as_geopandas()
        second = shapefile_region.as_geopandas()
        assert first.total_bounds.tolist() == second.total_bounds.tolist()
        assert len(first) == len(second)

    def test_mask_result_is_unchanged_by_caching(self, shapefile_region):
        ds = _global_grid_dataset()
        first = shapefile_region.mask(ds)
        second = shapefile_region.mask(ds)
        xr.testing.assert_identical(first, second)

        kept_lat = first.latitude.values
        kept_lon = first.longitude.values
        assert kept_lat.min() >= 20.0 and kept_lat.max() <= 50.0
        assert kept_lon.min() >= -130.0 and kept_lon.max() <= -100.0

    def test_a_missing_shapefile_still_raises(self, tmp_path):
        from extremeweatherbench import regions

        region = regions.ShapefileRegion(tmp_path / "absent.shp")
        with pytest.raises(ValueError):
            region.as_geopandas()

    def test_the_failure_is_not_cached_as_a_success(self, tmp_path):
        from extremeweatherbench import regions

        region = regions.ShapefileRegion(tmp_path / "absent.shp")
        with pytest.raises(ValueError):
            region.as_geopandas()
        with pytest.raises(ValueError):
            region.as_geopandas()


def _tc_forecast_dataset(n_time=4, seed=0):
    """Minimal forecast fields the TC tracker asks for."""
    rng = np.random.default_rng(seed)
    dims = ["valid_time", "latitude", "longitude"]
    shape = (n_time, 8, 10)
    return xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (dims, rng.random(shape) + 1000.0),
            "surface_wind_speed": (dims, rng.random(shape) * 30.0),
        },
        coords={
            "valid_time": pd.date_range("2021-09-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(10.0, 30.0, 8),
            "longitude": np.linspace(-80.0, -60.0, 10),
        },
    )


def _tc_track_target(n_time=4):
    """Observed-track dataset used to filter forecast candidates."""
    return xr.Dataset(
        {"intensity": ("valid_time", np.arange(float(n_time)))},
        coords={
            "valid_time": pd.date_range("2021-09-01", periods=n_time, freq="6h"),
            "latitude": ("valid_time", np.linspace(15.0, 25.0, n_time)),
            "longitude": ("valid_time", np.linspace(-75.0, -65.0, n_time)),
        },
    )


@pytest.fixture
def counted_tracker(monkeypatch):
    """Replace the TC tracker with a counting stub returning a marker."""
    from extremeweatherbench.events import tropical_cyclone as tc

    calls = {"count": 0}
    marker = xr.Dataset({"tracks": ("track", [0.0])}, coords={"track": [0]})

    def stub(*args, **kwargs):
        calls["count"] += 1
        return marker

    monkeypatch.setattr(tc, "generate_tc_tracks_by_init_time", stub)
    return calls, marker


class TestTrackCaching:
    """Phase 12: the documented track cache should actually hold."""

    def test_tracks_computed_once_across_two_calls(self, counted_tracker):
        from extremeweatherbench import derived

        calls, _ = counted_tracker
        variable = derived.TropicalCycloneTrackVariables()
        data = _tc_forecast_dataset()
        target = _tc_track_target()

        variable.get_or_compute_tracks(data, _target_dataset=target)
        variable.get_or_compute_tracks(data, _target_dataset=target)

        assert calls["count"] == 1, (
            f"the tracker ran {calls['count']} times for the same inputs; "
            "get_or_compute_tracks documents that it caches"
        )

    def test_cached_result_is_the_same_tracks(self, counted_tracker):
        from extremeweatherbench import derived

        _, marker = counted_tracker
        variable = derived.TropicalCycloneTrackVariables()
        data = _tc_forecast_dataset()
        target = _tc_track_target()

        first = variable.get_or_compute_tracks(data, _target_dataset=target)
        second = variable.get_or_compute_tracks(data, _target_dataset=target)
        assert first is marker
        assert second is marker

    def test_different_forecast_data_is_recomputed(self, counted_tracker):
        from extremeweatherbench import derived

        calls, _ = counted_tracker
        variable = derived.TropicalCycloneTrackVariables()
        target = _tc_track_target()

        variable.get_or_compute_tracks(
            _tc_forecast_dataset(seed=0), _target_dataset=target
        )
        variable.get_or_compute_tracks(
            _tc_forecast_dataset(seed=1), _target_dataset=target
        )

        assert calls["count"] == 2, (
            "a different forecast dataset must not reuse the cached tracks"
        )

    def test_different_target_tracks_are_recomputed(self, counted_tracker):
        from extremeweatherbench import derived

        calls, _ = counted_tracker
        variable = derived.TropicalCycloneTrackVariables()
        data = _tc_forecast_dataset()

        variable.get_or_compute_tracks(data, _target_dataset=_tc_track_target())
        variable.get_or_compute_tracks(data, _target_dataset=_tc_track_target())

        assert calls["count"] == 2, (
            "a different target dataset must not reuse the cached tracks"
        )

    def test_two_instances_do_not_share_a_cache(self, counted_tracker):
        from extremeweatherbench import derived

        calls, _ = counted_tracker
        data = _tc_forecast_dataset()
        target = _tc_track_target()

        derived.TropicalCycloneTrackVariables().get_or_compute_tracks(
            data, _target_dataset=target
        )
        derived.TropicalCycloneTrackVariables().get_or_compute_tracks(
            data, _target_dataset=target
        )

        # Instances can be configured differently, so their tracks are not
        # interchangeable even for identical inputs.
        assert calls["count"] == 2

    def test_a_missing_target_dataset_still_raises(self, counted_tracker):
        from extremeweatherbench import derived

        variable = derived.TropicalCycloneTrackVariables()
        with pytest.raises(ValueError, match="No track data provided"):
            variable.get_or_compute_tracks(_tc_forecast_dataset())

    def test_a_target_without_track_coords_still_raises(self, counted_tracker):
        from extremeweatherbench import derived

        variable = derived.TropicalCycloneTrackVariables()
        bare = xr.Dataset({"intensity": ("t", np.zeros(3))}, coords={"t": np.arange(3)})
        with pytest.raises(ValueError, match="missing required"):
            variable.get_or_compute_tracks(_tc_forecast_dataset(), _target_dataset=bare)


def _chunked_global_grid(n_time=8, n_lat=181, n_lon=360):
    """Chunked global grid for region subsetting."""
    return xr.Dataset(
        {
            "t": (
                ["valid_time", "latitude", "longitude"],
                np.zeros((n_time, n_lat, n_lon), dtype="float32"),
            )
        },
        coords={
            "valid_time": pd.date_range("2021-06-01", periods=n_time, freq="6h"),
            "latitude": np.linspace(-90, 90, n_lat),
            "longitude": np.linspace(-180, 180, n_lon, endpoint=False),
        },
    ).chunk({"valid_time": 2})


def _fancy_index_subset(ds, region):
    """The where(drop=True) plus sel formulation this phase replaces."""
    lon_min, lat_min, lon_max, lat_max = region.get_adjusted_bounds(ds)
    latitudes = ds.latitude.where(
        (ds.latitude >= lat_min) & (ds.latitude <= lat_max), drop=True
    )
    if lon_min > lon_max:
        longitudes = ds.longitude.where(
            (ds.longitude >= lon_min) | (ds.longitude <= lon_max), drop=True
        )
    else:
        longitudes = ds.longitude.where(
            (ds.longitude >= lon_min) & (ds.longitude <= lon_max), drop=True
        )
    return ds.sel(latitude=latitudes, longitude=longitudes)


class TestRegionBoundingBoxSubset:
    """Phase 13: a plain box should be taken as a slice, not a gather."""

    @staticmethod
    def _box():
        from extremeweatherbench import regions

        return regions.BoundingBoxRegion(
            latitude_min=20.0,
            latitude_max=50.0,
            longitude_min=-130.0,
            longitude_max=-100.0,
        )

    def test_subset_costs_less_graph_than_fancy_indexing(self):
        ds = _chunked_global_grid()
        region = self._box()

        masked = region.mask(ds)
        fancy = _fancy_index_subset(ds, region)

        assert opt.graph_size(masked["t"]) < opt.graph_size(fancy["t"]), (
            "selecting with coordinate DataArrays adds a gather layer that a "
            "contiguous slice does not need"
        )

    def test_subset_values_match_fancy_indexing(self):
        ds = _chunked_global_grid()
        region = self._box()
        xr.testing.assert_identical(
            region.mask(ds).compute(), _fancy_index_subset(ds, region).compute()
        )

    def test_subset_is_narrowed_to_the_region(self):
        ds = _chunked_global_grid()
        masked = self._box().mask(ds)

        assert masked.longitude.values.min() >= -130.0
        assert masked.longitude.values.max() <= -100.0
        assert masked.latitude.values.min() >= 20.0
        assert masked.latitude.values.max() <= 50.0
        assert masked.sizes["longitude"] < ds.sizes["longitude"]

    @pytest.mark.parametrize("descending", [False, True])
    def test_descending_latitude_gives_the_same_rows(self, descending):
        ds = _chunked_global_grid()
        if descending:
            ds = ds.isel(latitude=slice(None, None, -1))
        region = self._box()

        masked = region.mask(ds)
        expected = _fancy_index_subset(ds, region)
        np.testing.assert_array_equal(masked.latitude.values, expected.latitude.values)

    def test_an_antimeridian_region_keeps_both_ends(self):
        from extremeweatherbench import regions

        ds = _chunked_global_grid()
        region = regions.BoundingBoxRegion(
            latitude_min=20.0,
            latitude_max=50.0,
            longitude_min=170.0,
            longitude_max=-170.0,
        )
        masked = region.mask(ds)
        expected = _fancy_index_subset(ds, region)

        np.testing.assert_array_equal(
            masked.longitude.values, expected.longitude.values
        )
        xr.testing.assert_identical(masked.compute(), expected.compute())

    def test_a_region_outside_the_grid_gives_an_empty_subset(self):
        from extremeweatherbench import regions

        ds = _chunked_global_grid()
        region = regions.BoundingBoxRegion(
            latitude_min=20.0,
            latitude_max=50.0,
            longitude_min=-130.0,
            longitude_max=-100.0,
        )
        trimmed = ds.sel(longitude=slice(0.0, 90.0))
        masked = region.mask(trimmed)
        assert masked.sizes["longitude"] == 0


def _sparse_dataarray():
    """DataArray backed by a sparse.COO array."""
    import sparse

    data = sparse.COO(
        coords=np.array([[0, 1, 2], [1, 2, 0]]),
        data=np.array([1.0, 2.0, 3.0]),
        shape=(4, 4),
    )
    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": np.linspace(0.0, 3.0, 4),
            "longitude": np.linspace(10.0, 13.0, 4),
        },
    )


class TestDensifyDoesNotMutateTheCaller:
    """Phase 14: densifying should hand back a new array, not rewrite one."""

    def test_the_original_dataarray_stays_sparse(self):
        import sparse

        da = _sparse_dataarray()
        utils.maybe_densify_dataarray(da)

        assert isinstance(da.data, sparse.COO), (
            "the caller's array was densified in place; maybe_densify_dataarray "
            "should return a new array instead"
        )

    def test_a_dataset_holding_the_array_stays_sparse(self):
        import sparse

        ds = xr.Dataset({"t": _sparse_dataarray()})
        utils.maybe_densify_dataarray(ds["t"])

        assert isinstance(ds["t"].data, sparse.COO), (
            "densifying a variable pulled out of a dataset rewrote the "
            "dataset's own copy"
        )

    def test_the_returned_array_is_dense_with_the_same_values(self):
        da = _sparse_dataarray()
        densified = utils.maybe_densify_dataarray(da)

        assert not hasattr(densified.data, "todense")
        np.testing.assert_allclose(densified.values, da.data.todense())

    def test_coordinates_and_dims_survive(self):
        da = _sparse_dataarray()
        densified = utils.maybe_densify_dataarray(da)

        assert densified.dims == da.dims
        np.testing.assert_array_equal(densified.latitude.values, da.latitude.values)
        np.testing.assert_array_equal(densified.longitude.values, da.longitude.values)

    def test_a_dense_array_is_passed_straight_through(self):
        da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=["y", "x"])
        assert utils.maybe_densify_dataarray(da) is da


class TestMaximumMeanAbsoluteErrorHonorsSpatialDims:
    """Phase 14: a configured reduce_spatial_dims should not be discarded."""

    @staticmethod
    def _pair():
        rng = np.random.default_rng(0)
        coords = {
            "valid_time": pd.date_range("2021-07-01", periods=6, freq="6h"),
            "latitude": np.linspace(30.0, 40.0, 5),
            "longitude": np.linspace(-110.0, -100.0, 4),
        }
        dims = ["valid_time", "latitude", "longitude"]
        target = xr.DataArray(
            rng.random((6, 5, 4)) * 10 + 280, dims=dims, coords=coords
        )
        forecast = target + rng.random((6, 5, 4))
        return forecast, target

    def test_reducing_only_longitude_keeps_latitude(self):
        forecast, target = self._pair()
        metric = metrics.MaximumMeanAbsoluteError(
            reduce_spatial_dims=["longitude"], preserve_dims=["latitude"]
        )
        result = metric._compute_metric(forecast=forecast, target=target)

        assert "latitude" in result.dims, (
            "reduce_spatial_dims=['longitude'] was ignored, so latitude was "
            "collapsed anyway"
        )
        assert result.sizes["latitude"] == target.sizes["latitude"]

    def test_the_default_configuration_is_unchanged(self):
        forecast, target = self._pair()
        # The default preserves lead_time, which only the forecast carries.
        forecast = forecast.expand_dims(lead_time=[0, 6, 12])
        default = metrics.MaximumMeanAbsoluteError()
        explicit = metrics.MaximumMeanAbsoluteError(
            reduce_spatial_dims=["latitude", "longitude"]
        )

        np.testing.assert_allclose(
            default._compute_metric(forecast=forecast, target=target).values,
            explicit._compute_metric(forecast=forecast, target=target).values,
        )


class TestAlignmentSqueeze:
    """Phase 14: pin that the dead squeeze never reached the output."""

    @staticmethod
    def _pair():
        coords = {
            "valid_time": pd.date_range("2021-07-01", periods=4, freq="6h"),
            "latitude": np.linspace(30.0, 33.0, 4),
            "longitude": np.linspace(-110.0, -107.0, 4),
        }
        dims = ["valid_time", "latitude", "longitude"]
        values = np.arange(4.0 * 4 * 4).reshape(4, 4, 4)
        target = xr.Dataset({"t": (dims, values)}, coords=coords)
        forecast = xr.Dataset({"t": (dims, values + 1)}, coords=coords)
        return forecast, target

    def test_a_length_one_dimension_is_not_squeezed_away(self):
        from extremeweatherbench import inputs

        forecast, target = self._pair()
        forecast = forecast.isel(valid_time=slice(0, 1))
        target = target.isel(valid_time=slice(0, 1))

        aligned_forecast, aligned_target = inputs.align_forecast_to_target(
            forecast, target
        )

        assert aligned_forecast.sizes["valid_time"] == 1
        assert aligned_target.sizes["valid_time"] == 1

    def test_aligned_values_are_unchanged(self):
        from extremeweatherbench import inputs

        forecast, target = self._pair()
        aligned_forecast, aligned_target = inputs.align_forecast_to_target(
            forecast, target
        )

        np.testing.assert_allclose(aligned_target["t"].values, target["t"].values)
        np.testing.assert_allclose(aligned_forecast["t"].values, forecast["t"].values)


class TestAtmosphericRiverPipelineParity:
    """Phase 15: the derived-variable path and the module path must agree.

    The two entry points carried the same pipeline twice, so any change had
    to be made in both. These pin that they produce identical output before
    one is expressed in terms of the other.
    """

    @staticmethod
    def _dataset():
        rng = np.random.default_rng(11)
        valid_time = pd.date_range("2021-01-01", periods=3, freq="6h")
        level = np.array([1000, 850, 700, 500, 300, 200])
        lat = np.linspace(30.0, 34.75, 20)
        lon = np.linspace(-130.0, -125.25, 20)
        shape = (len(valid_time), len(level), len(lat), len(lon))
        dims = ["valid_time", "level", "latitude", "longitude"]
        return xr.Dataset(
            {
                "eastward_wind": (dims, rng.uniform(-20, 40, shape)),
                "northward_wind": (dims, rng.uniform(-20, 40, shape)),
                "specific_humidity": (dims, rng.uniform(1e-4, 2e-2, shape)),
            },
            coords={
                "valid_time": valid_time,
                "level": level,
                "latitude": lat,
                "longitude": lon,
            },
        )

    def test_the_derived_variable_matches_the_module_pipeline(self):
        from extremeweatherbench import derived

        data = self._dataset()
        variable = derived.AtmosphericRiverVariables()
        subset = derived._subset_to_top_pressure_level(data, 300)

        from_derived = variable.derive_variable(data).compute()
        from_module = ar.build_atmospheric_river_mask_and_land_intersection(
            subset
        ).compute()

        assert set(from_derived.data_vars) == set(from_module.data_vars)
        for name in from_module.data_vars:
            np.testing.assert_allclose(
                from_derived[name].values, from_module[name].values
            )

    def test_the_pipeline_output_is_pinned(self):
        data = self._dataset()
        result = ar.build_atmospheric_river_mask_and_land_intersection(data).compute()

        # One space-time feature spans the box; the box is open Pacific, so
        # nothing intersects land.
        assert np.array_equal(
            np.unique(result["atmospheric_river_mask"].values), [0, 1]
        )
        assert int(result["atmospheric_river_mask"].sum()) == 1116
        np.testing.assert_allclose(
            float(result["integrated_vapor_transport"].max()), 4174.306510274923
        )
        assert float(np.nansum(result["atmospheric_river_land_intersection"])) == 0.0

    def test_the_land_intersection_lights_up_over_land(self):
        data = self._dataset()
        over_land = data.assign_coords(
            latitude=data["latitude"].values + 6.0,
            longitude=data["longitude"].values + 30.0,
        )

        result = ar.build_atmospheric_river_mask_and_land_intersection(
            over_land
        ).compute()

        assert int(result["atmospheric_river_mask"].sum()) == 1116
        assert float(np.nansum(result["atmospheric_river_land_intersection"])) == 1116.0


class TestThresholdMetricValues:
    """Phase 15: pin every contingency metric before collapsing the blocks."""

    @staticmethod
    def _pair():
        forecast = xr.DataArray(
            [[15500.0, 14000.0, 16000.0], [14500.0, 15100.0, 13000.0]],
            dims=["x", "y"],
        )
        target = xr.DataArray(
            [[0.4, 0.2, 0.5], [0.25, 0.1, 0.35]],
            dims=["x", "y"],
        )
        return forecast, target

    # Row x=0: forecast>=15000 -> [1, 0, 1]; target>=0.3 -> [1, 0, 1]
    #   tp=2 fp=0 tn=1 fn=0  -> csi=1.0     far=0.0     acc=1.0
    # Row x=1: forecast>=15000 -> [0, 1, 0]; target>=0.3 -> [0, 0, 1]
    #   tp=0 fp=1 tn=1 fn=1  -> csi=0.0     far=1.0     acc=1/3
    EXPECTED = {
        "CriticalSuccessIndex": [1.0, 0.0],
        "FalseAlarmRatio": [0.0, 1.0],
        "TruePositives": [2 / 3, 0.0],
        "FalsePositives": [0.0, 1 / 3],
        "TrueNegatives": [1 / 3, 1 / 3],
        "FalseNegatives": [0.0, 1 / 3],
        "Accuracy": [1.0, 1 / 3],
    }

    @pytest.mark.parametrize("metric_name", sorted(EXPECTED))
    def test_each_threshold_metric_value(self, metric_name):
        forecast, target = self._pair()
        metric = getattr(metrics, metric_name)(
            forecast_threshold=15000, target_threshold=0.3, preserve_dims="x"
        )

        result = metric.compute_metric(forecast, target)

        np.testing.assert_allclose(result.values, self.EXPECTED[metric_name])

    @pytest.mark.parametrize("metric_name", sorted(EXPECTED))
    def test_a_shared_manager_gives_the_same_value(self, metric_name):
        forecast, target = self._pair()
        composite = metrics.ThresholdMetric(
            forecast_threshold=15000,
            target_threshold=0.3,
            preserve_dims="x",
            metrics=[getattr(metrics, metric_name)],
        )
        kwargs = composite.maybe_prepare_composite_kwargs(forecast, target)
        (instance,) = composite.maybe_expand_composite()

        result = instance._compute_metric(forecast, target, **kwargs)

        np.testing.assert_allclose(result.values, self.EXPECTED[metric_name])

    def test_a_multi_metric_composite_gives_the_same_values(self):
        forecast, target = self._pair()
        wanted = ["CriticalSuccessIndex", "Accuracy", "TruePositives"]
        composite = metrics.ThresholdMetric(
            forecast_threshold=15000,
            target_threshold=0.3,
            preserve_dims="x",
            metrics=[getattr(metrics, name) for name in wanted],
        )
        kwargs = composite.maybe_prepare_composite_kwargs(forecast, target)

        for instance, name in zip(composite.maybe_expand_composite(), wanted):
            np.testing.assert_allclose(
                instance._compute_metric(forecast, target, **kwargs).values,
                self.EXPECTED[name],
            )

    def test_the_base_class_still_refuses_to_compute(self):
        forecast, target = self._pair()

        with pytest.raises(NotImplementedError):
            metrics.ThresholdMetric().compute_metric(forecast, target)

    @staticmethod
    def _flat_pair():
        # roc_curve_data needs a flat sample dimension. Samples two and three
        # sit exactly on a threshold, which pins the comparison as inclusive.
        forecast = xr.DataArray(
            [15500.0, 15000.0, 16000.0, 14500.0, 15200.0], dims=["sample"]
        )
        target = xr.DataArray([0.4, 0.35, 0.3, 0.25, 0.2], dims=["sample"])
        return forecast, target

    def test_the_roc_curve_is_unchanged(self):
        forecast, target = self._flat_pair()
        roc = metrics.ReceiverOperatingCharacteristic(
            forecast_threshold=15000, target_threshold=0.3, preserve_dims=None
        )

        result = roc.compute_metric(forecast, target)

        np.testing.assert_allclose(result["POD"].values, [1.0, 1.0, 0.0])
        np.testing.assert_allclose(result["POFD"].values, [1.0, 0.5, 0.0])
        np.testing.assert_allclose(float(result["AUC"]), 0.75)

    def test_the_roc_skill_score_is_unchanged(self):
        forecast, target = self._flat_pair()
        skill = metrics.ReceiverOperatingCharacteristicSkillScore(
            forecast_threshold=15000, target_threshold=0.3, preserve_dims=None
        )

        result = skill.compute_metric(forecast, target)

        np.testing.assert_allclose(float(result), 0.5)


class TestLandMaskNormalization:
    """Phase 15: pin the land-mask where chain before collapsing it."""

    @staticmethod
    def _mask(values):
        return xr.DataArray(
            np.asarray(values, dtype=float),
            dims=["latitude", "longitude"],
            coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    def test_land_and_ocean_intersection_values(self):
        mask = self._mask([[1.0, 1.0], [0.0, 1.0]])
        land = self._mask([[1.0, 0.0], [1.0, 0.0]])

        result = calc.find_land_intersection(mask, land_mask=land)

        np.testing.assert_array_equal(result.values, [[1.0, 0.0], [0.0, 0.0]])

    def test_nan_in_either_input_propagates(self):
        mask = self._mask([[1.0, np.nan], [1.0, 1.0]])
        land = self._mask([[1.0, 1.0], [np.nan, 1.0]])

        result = calc.find_land_intersection(mask, land_mask=land)

        assert np.isnan(result.values[0, 1])
        assert np.isnan(result.values[1, 0])
        np.testing.assert_array_equal(result.values[[0, 1], [0, 1]], [1.0, 1.0])

    def test_the_default_land_mask_is_one_over_land_and_zero_over_ocean(self):
        # A point in Kansas and a point in the mid Pacific.
        mask = xr.DataArray(
            np.ones((1, 2)),
            dims=["latitude", "longitude"],
            coords={"latitude": [38.0], "longitude": [-98.0, -150.0]},
        )

        result = calc.find_land_intersection(mask)

        np.testing.assert_array_equal(result.values, [[1.0, 0.0]])


class TestDilationKernelEquivalence:
    """Phase 15: the dilation must stay identical to the square structure."""

    @staticmethod
    def _reference(data, radius):
        size = radius * 2 + 1
        struct = np.ones((size, size))
        return ndimage.binary_dilation(data, structure=struct, axes=(-2, -1)).astype(
            np.int8
        )

    @pytest.mark.parametrize("radius", [0, 1, 2, 8])
    @pytest.mark.parametrize("shape", [(7, 11), (3, 7, 11), (2, 3, 9, 9)])
    def test_matches_the_square_structure(self, radius, shape):
        rng = np.random.default_rng(radius + len(shape))
        data = rng.random(shape) < 0.1

        np.testing.assert_array_equal(
            calc._binary_dilation_ufunc(data, radius),
            self._reference(data, radius),
        )

    @pytest.mark.parametrize("corner", [(0, 0), (0, 10), (6, 0), (6, 10)])
    def test_corners_are_not_padded_as_true(self, corner):
        data = np.zeros((7, 11), dtype=bool)
        data[corner] = True

        np.testing.assert_array_equal(
            calc._binary_dilation_ufunc(data, 2), self._reference(data, 2)
        )

    def test_the_output_dtype_is_int8(self):
        data = np.zeros((4, 4), dtype=bool)

        assert calc._binary_dilation_ufunc(data, 1).dtype == np.int8


class TestValidTimeMaskConstruction:
    """Phase 15: pin the init/lead validity mask before vectorizing it."""

    @staticmethod
    def _reference(unique_init_indices, indices, n_init, n_lead):
        mask = np.zeros((n_init, n_lead), dtype=bool)
        for i, j in zip(indices[0], indices[1]):
            init_pos = np.where(unique_init_indices == i)[0]
            if len(init_pos) > 0:
                mask[init_pos[0], j] = True
        return mask

    def _case_dataset(self):
        n_init, n_lead = 5, 8
        lead = pd.to_timedelta(np.arange(0, n_lead * 6, 6), unit="h")
        return xr.Dataset(
            {
                "t": (
                    ["init_time", "lead_time", "latitude", "longitude"],
                    np.arange(float(n_init * n_lead * 2 * 2)).reshape(
                        n_init, n_lead, 2, 2
                    ),
                )
            },
            coords={
                "init_time": pd.date_range("2021-06-01", periods=n_init, freq="D"),
                "lead_time": lead,
                "latitude": [30.0, 31.0],
                "longitude": [-100.0, -99.0],
            },
        )

    def test_the_mask_matches_the_loop_on_real_case_indices(self):
        from extremeweatherbench import inputs

        data = self._case_dataset()
        indices = utils.derive_indices_from_init_time_and_lead_time(
            data, pd.Timestamp("2021-06-02"), pd.Timestamp("2021-06-04")
        )
        unique_init_indices = np.unique(indices[0])
        n_lead = data.sizes["lead_time"]
        expected = self._reference(
            unique_init_indices, indices, len(unique_init_indices), n_lead
        )

        result = inputs._valid_combinations_mask(unique_init_indices, indices, n_lead)

        assert expected.any(), "fixture must exercise a non-trivial mask"
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "init_positions,lead_positions",
        [
            ([0], [0]),
            ([0, 0, 3], [1, 2, 0]),
            ([2, 5, 5, 9], [0, 3, 4, 7]),
        ],
    )
    def test_the_mask_matches_the_loop_on_synthetic_indices(
        self, init_positions, lead_positions
    ):
        from extremeweatherbench import inputs

        indices = (np.array(init_positions), np.array(lead_positions))
        unique_init_indices = np.unique(indices[0])
        n_lead = int(max(lead_positions)) + 1
        expected = self._reference(
            unique_init_indices, indices, len(unique_init_indices), n_lead
        )

        result = inputs._valid_combinations_mask(unique_init_indices, indices, n_lead)

        np.testing.assert_array_equal(result, expected)

    def test_an_empty_selection_gives_an_all_false_mask(self):
        from extremeweatherbench import inputs

        indices = (np.array([], dtype=int), np.array([], dtype=int))

        result = inputs._valid_combinations_mask(np.array([], dtype=int), indices, 4)

        assert result.shape == (0, 4)
