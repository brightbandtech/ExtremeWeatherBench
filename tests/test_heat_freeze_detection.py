"""Unit tests for heat wave and freeze detection scripts.

Uses synthetic xarray data -- no network access required.
Covers core detection logic, consecutive filtering, event
tracking, bounding box expansion, edge cases, and CSV output.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "data_prep"),
)

from heat_freeze_bounds_global import (
    apply_consecutive_filter,
    build_exceedance_masks,
    detect_events,
    events_to_dataframe,
)
from heat_freeze_bounds_case import (
    _apply_consecutive_filter as yaml_consecutive_filter,
    _edge_valid_fraction,
)


# ── helpers ──────────────────────────────────────────────────


def _make_mask(
    n_days: int,
    n_lat: int,
    n_lon: int,
    hot_regions: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a boolean mask with optional injected hot regions.

    ``hot_regions`` is a list of dicts, each with keys:
        day_start, day_end, lat_start, lat_end,
        lon_start, lon_end
    """
    mask = np.zeros((n_days, n_lat, n_lon), dtype=bool)
    dates = pd.date_range(
        "2023-07-01",
        periods=n_days,
        freq="1D",
    ).values
    lats = np.linspace(-10, 10, n_lat)
    lons = np.linspace(0, 20, n_lon)

    if hot_regions:
        for r in hot_regions:
            mask[
                r["day_start"] : r["day_end"],
                r["lat_start"] : r["lat_end"],
                r["lon_start"] : r["lon_end"],
            ] = True
    return mask, dates, lats, lons


# ── apply_consecutive_filter tests ───────────────────────────


class TestConsecutiveFilter:
    def test_run_of_3_survives(self):
        mask, *_ = _make_mask(
            5,
            4,
            4,
            [
                dict(
                    day_start=0,
                    day_end=3,
                    lat_start=0,
                    lat_end=2,
                    lon_start=0,
                    lon_end=2,
                ),
            ],
        )
        out = apply_consecutive_filter(mask, min_days=3)
        assert out[:3, :2, :2].all()

    def test_run_of_2_removed(self):
        mask, *_ = _make_mask(
            5,
            4,
            4,
            [
                dict(
                    day_start=0,
                    day_end=2,
                    lat_start=0,
                    lat_end=2,
                    lon_start=0,
                    lon_end=2,
                ),
            ],
        )
        out = apply_consecutive_filter(mask, min_days=3)
        assert not out.any()

    def test_run_of_5_fully_kept(self):
        mask, *_ = _make_mask(
            7,
            4,
            4,
            [
                dict(
                    day_start=1,
                    day_end=6,
                    lat_start=0,
                    lat_end=3,
                    lon_start=0,
                    lon_end=3,
                ),
            ],
        )
        out = apply_consecutive_filter(mask, min_days=3)
        assert out[1:6, :3, :3].all()
        assert not out[0, :3, :3].any()
        assert not out[6, :3, :3].any()

    def test_gap_splits_runs(self):
        mask, *_ = _make_mask(
            10,
            4,
            4,
            [
                dict(
                    day_start=0,
                    day_end=3,
                    lat_start=0,
                    lat_end=2,
                    lon_start=0,
                    lon_end=2,
                ),
                dict(
                    day_start=5,
                    day_end=7,
                    lat_start=0,
                    lat_end=2,
                    lon_start=0,
                    lon_end=2,
                ),
            ],
        )
        out = apply_consecutive_filter(mask, min_days=3)
        assert out[:3, :2, :2].all()
        assert not out[5:7, :2, :2].any()

    def test_yaml_filter_matches(self):
        mask, *_ = _make_mask(
            6,
            4,
            4,
            [
                dict(
                    day_start=0,
                    day_end=4,
                    lat_start=0,
                    lat_end=3,
                    lon_start=0,
                    lon_end=3,
                ),
            ],
        )
        a = apply_consecutive_filter(mask)
        b = yaml_consecutive_filter(mask)
        np.testing.assert_array_equal(a, b)


# ── detect_events tests ─────────────────────────────────────


class TestDetectEvents:
    def test_single_heatwave(self):
        mask, dates, lats, lons = _make_mask(
            7,
            20,
            20,
            [
                dict(
                    day_start=1,
                    day_end=6,
                    lat_start=5,
                    lat_end=15,
                    lon_start=5,
                    lon_end=15,
                ),
            ],
        )
        filt = apply_consecutive_filter(mask)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 1
        ev = evs[0]
        assert ev["type"] == "heat_wave"
        assert ev["lat_min"] <= lats[5]
        assert ev["lat_max"] >= lats[14]
        assert ev["lon_min"] <= lons[5]
        assert ev["lon_max"] >= lons[14]

    def test_no_events_when_below_min_duration(self):
        mask, dates, lats, lons = _make_mask(
            5,
            10,
            10,
            [
                dict(
                    day_start=0,
                    day_end=2,
                    lat_start=0,
                    lat_end=5,
                    lon_start=0,
                    lon_end=5,
                ),
            ],
        )
        filt = apply_consecutive_filter(mask)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 0

    def test_two_separate_events(self):
        mask, dates, lats, lons = _make_mask(
            7,
            30,
            30,
            [
                dict(
                    day_start=0,
                    day_end=4,
                    lat_start=0,
                    lat_end=5,
                    lon_start=0,
                    lon_end=5,
                ),
                dict(
                    day_start=0,
                    day_end=4,
                    lat_start=20,
                    lat_end=25,
                    lon_start=20,
                    lon_end=25,
                ),
            ],
        )
        filt = apply_consecutive_filter(mask)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 2
        bounds = [(e["lat_min"], e["lat_max"], e["lon_min"], e["lon_max"]) for e in evs]
        lat_ranges = [(b[0], b[1]) for b in bounds]
        assert (
            lat_ranges[0][1] < lat_ranges[1][0] or lat_ranges[1][1] < lat_ranges[0][0]
        )

    def test_merging_adjacent_events(self):
        """Two regions that touch on day 2 merge into one."""
        n_lat, n_lon = 20, 20
        mask = np.zeros((5, n_lat, n_lon), dtype=bool)
        dates = pd.date_range(
            "2023-07-01",
            periods=5,
            freq="1D",
        ).values
        lats = np.linspace(0, 10, n_lat)
        lons = np.linspace(0, 10, n_lon)

        mask[:, 0:5, 0:5] = True
        mask[:, 5:10, 5:10] = True
        mask[2:, 4:6, 4:6] = True

        filt = apply_consecutive_filter(mask)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        merged = [
            e
            for e in evs
            if e["lat_min"] <= lats[0] + 0.1 and e["lat_max"] >= lats[9] - 0.1
        ]
        assert len(merged) >= 1

    def test_area_decline_terminates_event(self):
        """Event that shrinks to < 50% of peak is terminated."""
        n = 20
        mask = np.zeros((8, n, n), dtype=bool)
        dates = pd.date_range(
            "2023-07-01",
            periods=8,
            freq="1D",
        ).values
        lats = np.linspace(0, 10, n)
        lons = np.linspace(0, 10, n)

        mask[0:3, 5:10, 5:10] = True
        mask[3:5, 0:20, 0:20] = True
        mask[5, 5:10, 5:10] = True
        mask[6:8, 18:20, 18:20] = True

        filt = apply_consecutive_filter(mask, min_days=3)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        for ev in evs:
            assert ev["done"]

    def test_freeze_event_type(self):
        mask, dates, lats, lons = _make_mask(
            5,
            10,
            10,
            [
                dict(
                    day_start=0,
                    day_end=4,
                    lat_start=2,
                    lat_end=8,
                    lon_start=2,
                    lon_end=8,
                ),
            ],
        )
        filt = apply_consecutive_filter(mask)
        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "freeze",
        )
        assert len(evs) == 1
        assert evs[0]["type"] == "freeze"

    def test_empty_mask_produces_no_events(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        dates = pd.date_range(
            "2023-07-01",
            periods=10,
            freq="1D",
        ).values
        lats = np.linspace(0, 5, 10)
        lons = np.linspace(0, 5, 10)
        evs = detect_events(
            mask,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 0

    def test_all_true_single_global_event(self):
        """Entire mask True for 5 days -> one event."""
        mask = np.ones((5, 10, 10), dtype=bool)
        dates = pd.date_range(
            "2023-07-01",
            periods=5,
            freq="1D",
        ).values
        lats = np.linspace(-5, 5, 10)
        lons = np.linspace(0, 10, 10)
        evs = detect_events(
            mask,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 1
        ev = evs[0]
        assert ev["lat_min"] == pytest.approx(lats.min())
        assert ev["lat_max"] == pytest.approx(lats.max())


# ── events_to_dataframe tests ───────────────────────────────


class TestEventsToDataframe:
    def test_empty_events_returns_correct_columns(self):
        df = events_to_dataframe([])
        expected = [
            "label",
            "event_type",
            "start_date",
            "end_date",
            "latitude_min",
            "latitude_max",
            "longitude_min",
            "longitude_max",
        ]
        assert list(df.columns) == expected
        assert len(df) == 0

    def test_labels_are_sequential(self):
        evs = [
            {
                "type": "heat_wave",
                "start": np.datetime64("2023-07-05"),
                "end": np.datetime64("2023-07-08"),
                "lat_min": 10,
                "lat_max": 20,
                "lon_min": 30,
                "lon_max": 40,
                "peak": 100,
                "area": 50,
                "done": True,
            },
            {
                "type": "freeze",
                "start": np.datetime64("2023-07-01"),
                "end": np.datetime64("2023-07-04"),
                "lat_min": -10,
                "lat_max": 0,
                "lon_min": 50,
                "lon_max": 60,
                "peak": 80,
                "area": 40,
                "done": True,
            },
        ]
        df = events_to_dataframe(evs)
        assert list(df["label"]) == [1, 2]
        assert df.iloc[0]["event_type"] == "freeze"
        assert df.iloc[1]["event_type"] == "heat_wave"

    def test_csv_roundtrip(self):
        evs = [
            {
                "type": "heat_wave",
                "start": np.datetime64("2023-07-01"),
                "end": np.datetime64("2023-07-05"),
                "lat_min": 5.0,
                "lat_max": 15.0,
                "lon_min": 100.0,
                "lon_max": 110.0,
                "peak": 200,
                "area": 100,
                "done": True,
            },
        ]
        df = events_to_dataframe(evs)
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as f:
            df.to_csv(f.name, index=False)
            back = pd.read_csv(f.name)
        assert len(back) == 1
        assert back.iloc[0]["event_type"] == "heat_wave"
        assert back.iloc[0]["latitude_min"] == pytest.approx(5.0)


# ── edge validity tests (yaml script) ───────────────────────


class TestEdgeValidity:
    def test_fully_valid_edge(self):
        mask = np.ones((5, 20, 20), dtype=bool)
        assert _edge_valid_fraction(mask, "north", 4) == 1.0
        assert _edge_valid_fraction(mask, "south", 4) == 1.0
        assert _edge_valid_fraction(mask, "east", 4) == 1.0
        assert _edge_valid_fraction(mask, "west", 4) == 1.0

    def test_fully_empty_edge(self):
        mask = np.zeros((5, 20, 20), dtype=bool)
        assert _edge_valid_fraction(mask, "north", 4) == 0.0

    def test_partial_edge(self):
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[:, -4:, :10] = True
        frac = _edge_valid_fraction(mask, "north", 4)
        assert 0.4 < frac < 0.6

    def test_south_edge(self):
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[:, :4, :] = True
        assert _edge_valid_fraction(mask, "south", 4) == 1.0

    def test_east_edge(self):
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[:, :, -4:] = True
        assert _edge_valid_fraction(mask, "east", 4) == 1.0

    def test_west_edge(self):
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[:, :, :4] = True
        assert _edge_valid_fraction(mask, "west", 4) == 1.0


# ── build_exceedance_masks with synthetic data ──────────────


class TestBuildExceedanceMasks:
    @pytest.fixture
    def synthetic_data(self):
        """Build synthetic ERA5 + climatology + land mask."""
        n_days, n_lat, n_lon = 10, 8, 8
        dates = pd.date_range(
            "2023-07-01",
            periods=n_days,
            freq="1D",
        )
        lats = np.linspace(30, 37, n_lat)
        lons = np.linspace(100, 107, n_lon)

        base_temp = 300.0
        t2m_data = np.full(
            (n_days, n_lat, n_lon),
            base_temp,
        )
        t2m_data[2:6, 2:6, 2:6] = 320.0

        t2m = xr.DataArray(
            t2m_data,
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": dates,
                "latitude": lats,
                "longitude": lons,
            },
        )

        doy = np.arange(1, 367)
        clim_max = xr.DataArray(
            np.full((366, n_lat, n_lon), 310.0),
            dims=["dayofyear", "latitude", "longitude"],
            coords={
                "dayofyear": doy,
                "latitude": lats,
                "longitude": lons,
            },
        )
        clim_min = xr.DataArray(
            np.full((366, n_lat, n_lon), 260.0),
            dims=["dayofyear", "latitude", "longitude"],
            coords={
                "dayofyear": doy,
                "latitude": lats,
                "longitude": lons,
            },
        )

        land_mask = xr.DataArray(
            np.ones((n_lat, n_lon), dtype=bool),
            dims=["latitude", "longitude"],
            coords={"latitude": lats, "longitude": lons},
        )

        return t2m, clim_max, clim_min, land_mask

    def test_heatwave_mask_detects_hot_region(
        self,
        synthetic_data,
    ):
        t2m, clim_max, clim_min, land_mask = synthetic_data
        hw, fz = build_exceedance_masks(
            t2m,
            clim_max,
            clim_min,
            land_mask,
        )
        hw_np = hw.values.astype(bool)
        assert hw_np[2:6, 2:6, 2:6].all()
        assert not hw_np[0, 0, 0]

    def test_freeze_mask_cold_region(self, synthetic_data):
        t2m, clim_max, clim_min, land_mask = synthetic_data
        cold_t2m = t2m.copy()
        cold_t2m.values[:] = 270.0
        cold_t2m.values[1:5, 3:7, 3:7] = 250.0
        _, fz = build_exceedance_masks(
            cold_t2m,
            clim_max,
            clim_min,
            land_mask,
        )
        fz_np = fz.values.astype(bool)
        assert fz_np[1:5, 3:7, 3:7].all()

    def test_ocean_masked_out(self, synthetic_data):
        t2m, clim_max, clim_min, _ = synthetic_data
        ocean_mask = xr.DataArray(
            np.zeros(
                (len(t2m.latitude), len(t2m.longitude)),
                dtype=bool,
            ),
            dims=["latitude", "longitude"],
            coords={
                "latitude": t2m.latitude,
                "longitude": t2m.longitude,
            },
        )
        hw, fz = build_exceedance_masks(
            t2m,
            clim_max,
            clim_min,
            ocean_mask,
        )
        assert not hw.values.any()
        assert not fz.values.any()


# ── end-to-end pipeline test ────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_synthetic(self):
        """Full pipeline on synthetic data: create mask, filter,
        detect, produce CSV."""
        n_days, n_lat, n_lon = 15, 30, 30
        mask, dates, lats, lons = _make_mask(
            n_days,
            n_lat,
            n_lon,
            [
                dict(
                    day_start=2,
                    day_end=8,
                    lat_start=5,
                    lat_end=15,
                    lon_start=5,
                    lon_end=15,
                ),
                dict(
                    day_start=10,
                    day_end=14,
                    lat_start=20,
                    lat_end=28,
                    lon_start=20,
                    lon_end=28,
                ),
            ],
        )

        filt = apply_consecutive_filter(mask)
        assert filt.sum() > 0

        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 2

        df = events_to_dataframe(evs)
        assert len(df) == 2
        assert list(df["label"]) == [1, 2]
        assert all(df["latitude_min"] < df["latitude_max"])
        assert all(df["longitude_min"] < df["longitude_max"])

        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as f:
            df.to_csv(f.name, index=False)
            back = pd.read_csv(f.name)
        assert len(back) == 2

    def test_mixed_heat_freeze(self):
        """Both heat wave and freeze events are tracked."""
        mask_hw, dates, lats, lons = _make_mask(
            10,
            20,
            20,
            [
                dict(
                    day_start=0,
                    day_end=5,
                    lat_start=0,
                    lat_end=10,
                    lon_start=0,
                    lon_end=10,
                ),
            ],
        )
        mask_fz, *_ = _make_mask(
            10,
            20,
            20,
            [
                dict(
                    day_start=3,
                    day_end=8,
                    lat_start=10,
                    lat_end=20,
                    lon_start=10,
                    lon_end=20,
                ),
            ],
        )

        hw_filt = apply_consecutive_filter(mask_hw)
        fz_filt = apply_consecutive_filter(mask_fz)

        hw_evs = detect_events(
            hw_filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        fz_evs = detect_events(
            fz_filt,
            dates,
            lats,
            lons,
            "freeze",
        )

        df = events_to_dataframe(hw_evs + fz_evs)
        types = set(df["event_type"])
        assert "heat_wave" in types
        assert "freeze" in types


# ── cross-boundary consecutive filter test ──────────────────


class TestCrossBoundary:
    def test_event_at_time_boundary(self):
        """Event starting near end of array is still detected."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[7:10, 3:7, 3:7] = True

        dates = pd.date_range(
            "2023-07-01",
            periods=10,
            freq="1D",
        ).values
        lats = np.linspace(0, 5, 10)
        lons = np.linspace(0, 5, 10)

        filt = apply_consecutive_filter(mask)
        assert filt[7:10, 3:7, 3:7].all()

        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) == 1
