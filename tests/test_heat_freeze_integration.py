"""Integration smoke tests using real ERA5 + climatology from GCS.

These tests require network access to Google Cloud Storage.
Skip in CI with:  pytest -m "not integration"
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "data_prep"),
)

from heat_freeze_bounds_global import (
    apply_consecutive_filter,
    build_exceedance_masks,
    build_land_mask,
    detect_events,
    events_to_dataframe,
    get_daily_climatology_thresholds,
    open_era5_t2m,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def era5_july_2023():
    """Open 10 days of ERA5 for July 2023 (small window)."""
    return open_era5_t2m("2023-07-01", "2023-07-10")


@pytest.fixture(scope="module")
def climatology_thresholds():
    return get_daily_climatology_thresholds()


class TestRealDataSmoke:
    def test_era5_opens_with_correct_dims(
        self,
        era5_july_2023,
    ):
        t2m = era5_july_2023
        assert "latitude" in t2m.dims
        assert "longitude" in t2m.dims
        tdim = [d for d in t2m.dims if d in ("valid_time", "time")][0]
        assert t2m.sizes[tdim] > 0

    def test_climatology_has_dayofyear(
        self,
        climatology_thresholds,
    ):
        clim_max, clim_min = climatology_thresholds
        assert "dayofyear" in clim_max.dims
        assert "dayofyear" in clim_min.dims
        assert clim_max.sizes["dayofyear"] >= 365

    def test_land_mask_shape(self, era5_july_2023):
        t2m = era5_july_2023
        land = build_land_mask(t2m.longitude, t2m.latitude)
        assert land.shape == (
            t2m.sizes["latitude"],
            t2m.sizes["longitude"],
        )
        assert land.any()

    def test_exceedance_masks_produce_results(
        self,
        era5_july_2023,
        climatology_thresholds,
    ):
        t2m = era5_july_2023
        clim_max, clim_min = climatology_thresholds
        land = build_land_mask(t2m.longitude, t2m.latitude)

        hw, fz = build_exceedance_masks(
            t2m,
            clim_max,
            clim_min,
            land,
        )
        hw_np = hw.compute().values.astype(bool)
        fz_np = fz.compute().values.astype(bool)  # noqa: F841

        assert hw_np.shape[0] > 0
        assert hw_np.any(), "Expected at least some exceedance in July"

    def test_full_pipeline_finds_events(
        self,
        era5_july_2023,
        climatology_thresholds,
    ):
        """10 days in July should yield at least one heat wave
        event somewhere on the globe."""
        t2m = era5_july_2023
        clim_max, clim_min = climatology_thresholds
        land = build_land_mask(t2m.longitude, t2m.latitude)

        hw, fz = build_exceedance_masks(
            t2m,
            clim_max,
            clim_min,
            land,
        )
        tdim = [d for d in hw.dims if d in ("valid_time", "time")][0]

        hw_da = hw.compute()
        hw_np = hw_da.values.astype(bool)
        dates = hw_da[tdim].values
        lats = hw_da.latitude.values
        lons = hw_da.longitude.values

        filt = apply_consecutive_filter(hw_np)
        if not filt.any():
            pytest.skip("No 3-day heat waves in this 10-day window")

        evs = detect_events(
            filt,
            dates,
            lats,
            lons,
            "heat_wave",
        )
        assert len(evs) >= 1

        df = events_to_dataframe(evs)
        assert len(df) >= 1
        assert all(df["latitude_min"] >= -90)
        assert all(df["latitude_max"] <= 90)
        assert all(df["longitude_min"] >= -180)
        assert all(df["longitude_max"] <= 360)

        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as f:
            df.to_csv(f.name, index=False)
            back = pd.read_csv(f.name)
        assert len(back) == len(df)
        assert not back.isna().any().any()
