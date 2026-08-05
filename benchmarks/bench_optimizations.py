"""Wall-clock benchmarks for the xarray/dask optimization work.

Run before and after a change and compare:

    python benchmarks/bench_optimizations.py --out benchmarks/baseline.json
    python benchmarks/bench_optimizations.py --compare benchmarks/baseline.json

Wall clock is deliberately kept out of the pytest suite, where it is flaky.
The test suite asserts deterministic proxies instead (laziness, graph size,
call counts); this script exists to confirm those proxies translate into
real speedups.
"""

import argparse
import json
import pathlib
import time
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr


def _time(fn: Callable, repeats: int = 3, warmup: int = 1) -> float:
    """Return the best wall-clock time of fn over several repeats."""
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings)


# The unoptimized time-coordinate reshape is quadratic in init x lead, so the
# baseline has to run at a size that finishes. 20x21 already takes ~20 s.
CONVERT_N_INIT = 12
CONVERT_N_LEAD = 13


def bench_convert_init_time_to_valid_time() -> float:
    """Phase 1: init/lead to lead/valid reshape."""
    from extremeweatherbench import utils

    n_init, n_lead = CONVERT_N_INIT, CONVERT_N_LEAD
    ds = xr.Dataset(
        {
            "t": (
                ["init_time", "lead_time", "latitude", "longitude"],
                np.zeros((n_init, n_lead, 20, 20), dtype="float32"),
            )
        },
        coords={
            "init_time": pd.date_range("2021-06-01", periods=n_init, freq="D"),
            "lead_time": np.arange(0, n_lead * 6, 6),
            "latitude": np.linspace(20, 50, 20),
            "longitude": np.linspace(-130, -100, 20),
        },
    ).chunk({"init_time": 1})

    return _time(
        lambda: utils.convert_init_time_to_valid_time(ds).compute(),
        repeats=2,
        warmup=0,
    )


def bench_atmospheric_river_mask() -> float:
    """Phase 2: AR mask over an analysis-shaped array."""
    from extremeweatherbench.events import atmospheric_river as ar

    rng = np.random.default_rng(0)
    time_coord = pd.date_range("2023-01-01", periods=24, freq="6h")
    lat = np.linspace(10, 60, 320)
    lon = np.linspace(-160, -100, 320)
    shape = (len(time_coord), len(lat), len(lon))

    ivt_values = rng.uniform(100, 300, shape)
    ivt_values[:, 80:190, 80:190] = 500
    lap_values = rng.uniform(-2, 2, shape)
    lap_values[:, 80:190, 80:190] = 3.0

    coords = {"valid_time": time_coord, "latitude": lat, "longitude": lon}
    dims = ["valid_time", "latitude", "longitude"]
    ivt = xr.DataArray(ivt_values, dims=dims, coords=coords).chunk({"valid_time": 1})
    lap = xr.DataArray(lap_values, dims=dims, coords=coords).chunk({"valid_time": 1})

    def run():
        result = ar.atmospheric_river_mask(ivt=ivt, ivt_laplacian=lap)
        if hasattr(result.data, "compute"):
            result = result.compute()
        return result

    return _time(run, repeats=2)


def bench_atmospheric_river_mask_forecast() -> float:
    """Phase 2: AR mask over a forecast-shaped array.

    This is the shape whose labeling semantics changed: features must not
    merge across initializations, so the round trip through init_time/
    lead_time happens here and not in the analysis-shaped benchmark.
    """
    from extremeweatherbench.events import atmospheric_river as ar

    rng = np.random.default_rng(0)
    init_time = pd.date_range("2023-01-01", periods=4, freq="D")
    lead_time = pd.to_timedelta(np.arange(0, 8 * 6, 6), unit="h")
    lat = np.linspace(10, 60, 240)
    lon = np.linspace(-160, -100, 240)
    shape = (len(init_time), len(lead_time), len(lat), len(lon))

    ivt_values = rng.uniform(100, 300, shape)
    ivt_values[..., 60:150, 60:150] = 500
    lap_values = rng.uniform(-2, 2, shape)
    lap_values[..., 60:150, 60:150] = 3.0

    from extremeweatherbench import utils

    dims = ["init_time", "lead_time", "latitude", "longitude"]
    coords = {
        "init_time": init_time,
        "lead_time": lead_time,
        "latitude": lat,
        "longitude": lon,
    }
    by_init = xr.Dataset(
        {"ivt": (dims, ivt_values), "lap": (dims, lap_values)}, coords=coords
    ).chunk({"init_time": 1})
    # The pipeline sees forecasts after this conversion, so lead_time and
    # valid_time are both dimensions by the time the mask runs.
    by_valid = utils.convert_init_time_to_valid_time(by_init)
    ivt, lap = by_valid["ivt"], by_valid["lap"]

    def run():
        result = ar.atmospheric_river_mask(ivt=ivt, ivt_laplacian=lap)
        if hasattr(result.data, "compute"):
            result = result.compute()
        return result

    return _time(run, repeats=2)


def bench_atmospheric_river_pipeline() -> float:
    """Phase 2: full AR pipeline, where IVT is shared by mask and output."""
    from extremeweatherbench.events import atmospheric_river as ar

    rng = np.random.default_rng(4)
    levels = np.array([1000.0, 925, 850, 700, 600, 500, 400, 300])
    time_coord = pd.date_range("2023-01-01", periods=32, freq="6h")
    lat = np.linspace(10, 60, 400)
    lon = np.linspace(-160, -110, 400)
    shape = (len(time_coord), len(levels), len(lat), len(lon))
    dims = ["valid_time", "level", "latitude", "longitude"]
    coords = {
        "valid_time": time_coord,
        "level": levels,
        "latitude": lat,
        "longitude": lon,
    }

    ds = xr.Dataset(
        {
            "eastward_wind": (dims, rng.uniform(5, 45, shape)),
            "northward_wind": (dims, rng.uniform(5, 45, shape)),
            "specific_humidity": (dims, rng.uniform(0.001, 0.02, shape)),
        },
        coords=coords,
    ).chunk({"valid_time": 1})

    def run():
        result = ar.build_atmospheric_river_mask_and_land_intersection(ds)
        return result.compute()

    return _time(run, repeats=2)


def bench_reduce_dataarray() -> float:
    """Phase 3: reduce a forecast, then keep only a tolerance window of it.

    This is the shape of MaximumMeanAbsoluteError. Reducing eagerly means
    averaging every timestep and then discarding all but the window; staying
    lazy lets dask drop the timesteps outside the window before doing the
    work. A reduction on its own would show nothing, since the cost being
    removed is work that the following selection makes unnecessary.
    """
    from extremeweatherbench import utils

    n_time = 120
    forecast = xr.DataArray(
        np.random.default_rng(1).standard_normal((n_time, 400, 600)),
        dims=["valid_time", "latitude", "longitude"],
        coords={
            "valid_time": pd.date_range("2021-06-20", periods=n_time, freq="6h"),
            "latitude": np.linspace(20, 50, 400),
            "longitude": np.linspace(-130, -100, 600),
        },
    ).chunk({"valid_time": 8})
    centre = forecast.valid_time.values[n_time // 2]
    half_window = np.timedelta64(12, "h")

    def run():
        spatial_mean = utils.reduce_dataarray(
            forecast,
            method="mean",
            reduce_dims=["latitude", "longitude"],
            skipna=True,
        )
        result = spatial_mean.where(
            (spatial_mean.valid_time >= centre - half_window)
            & (spatial_mean.valid_time <= centre + half_window),
            drop=True,
        ).max("valid_time")
        if hasattr(result.data, "compute"):
            result = result.compute()
        return result

    return _time(run)


def bench_tc_spatial_mask() -> float:
    """Phase 7: the storm-proximity mask over a global grid."""
    from extremeweatherbench.events import tropical_cyclone as tc

    lat_coords = np.linspace(-90.0, 90.0, 721)
    lon_coords = np.linspace(0.0, 359.75, 1440)
    frame = pd.DataFrame(
        {
            "valid_time": pd.date_range("2021-08-01", periods=12, freq="6h"),
            "latitude": np.linspace(12.0, 26.0, 12),
            "longitude": np.linspace(140.0, 128.0, 12),
        }
    )

    return _time(
        lambda: tc._create_spatial_mask(lat_coords, lon_coords, frame, 5.0),
        repeats=3,
    )


def bench_cape() -> float:
    """Phase 6: mixed-layer CAPE over a modest grid."""
    from extremeweatherbench.events import severe_convection as sc

    rng = np.random.default_rng(2)
    levels = np.array(
        [1000.0, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300, 200]
    )
    n_time, n_lat, n_lon = 2, 24, 24
    shape = (n_time, n_lat, n_lon, len(levels))

    temperature = 300 - 6.5 * (np.log(1000 / levels) * 8.0)
    temperature = np.broadcast_to(temperature, shape) + rng.normal(0, 1, shape)
    dewpoint = temperature - rng.uniform(1, 12, shape)
    geopotential = np.broadcast_to(
        (np.log(1000 / levels) * 8000.0 * 9.81), shape
    ).copy()

    coords = {"level": levels}
    dims = ["time", "latitude", "longitude", "level"]
    p = xr.DataArray(levels, dims=["level"], coords=coords)
    t = xr.DataArray(temperature, dims=dims, coords=coords)
    td = xr.DataArray(dewpoint, dims=dims, coords=coords)
    z = xr.DataArray(geopotential, dims=dims, coords=coords)

    def run():
        return sc.compute_mixed_layer_cape(
            pressure=p,
            temperature=t,
            dewpoint=td,
            geopotential=z,
            pressure_dim="level",
            parallel=False,
        ).values

    return _time(run, repeats=2)


def bench_region_mask() -> float:
    """Phase 13: bounding-box subsetting of a global grid."""
    from extremeweatherbench import regions

    ds = xr.Dataset(
        {
            "t": (
                ["valid_time", "latitude", "longitude"],
                np.zeros((10, 181, 360), dtype="float32"),
            )
        },
        coords={
            "valid_time": pd.date_range("2021-06-20", periods=10, freq="6h"),
            "latitude": np.linspace(-90, 90, 181),
            "longitude": np.linspace(-180, 179, 360),
        },
    ).chunk({"valid_time": 1})

    region = regions.BoundingBoxRegion(
        latitude_min=30.0,
        latitude_max=50.0,
        longitude_min=-120.0,
        longitude_max=-90.0,
    )

    return _time(lambda: region.mask(ds).compute())


def bench_center_of_mass() -> float:
    """Phase 9: center of mass of every 2-D field in a forecast stack."""
    from extremeweatherbench import metrics

    rng = np.random.default_rng(0)
    values = rng.random((20, 20, 181, 360))
    values[values < 0.4] = 0.0
    da = xr.DataArray(
        values,
        dims=["lead_time", "valid_time", "latitude", "longitude"],
        coords={
            "lead_time": np.arange(0, 120, 6),
            "valid_time": pd.date_range("2021-02-01", periods=20, freq="6h"),
            "latitude": np.linspace(-90, 90, 181),
            "longitude": np.linspace(-180, 179, 360),
        },
    )

    def run():
        lat_idx, lon_idx = metrics._center_of_mass_indices(da)
        return lat_idx.values, lon_idx.values

    return _time(run)


def bench_landfall_detection() -> float:
    """Phase 8: ocean-to-land crossing tests along a track."""
    from extremeweatherbench import calc, utils

    n_points = 480
    track = xr.DataArray(
        np.arange(float(n_points)),
        dims=["valid_time"],
        coords={
            "valid_time": pd.date_range("2021-08-28", periods=n_points, freq="1h"),
            "latitude": ("valid_time", np.linspace(22.0, 34.0, n_points)),
            "longitude": ("valid_time", np.linspace(-92.0, -90.0, n_points)),
        },
    )
    land = utils.load_land_geometry()
    ocean = utils.load_ocean_geometry()

    return _time(lambda: calc._detect_landfalls_wrapper(track, land, ocean).compute())


def bench_daily_min() -> float:
    """Phase 10: per-day minimum over a chunked series of complete days."""
    from extremeweatherbench import utils

    timesteps_per_day = 4
    n_days = 120
    n_steps = n_days * timesteps_per_day
    da = xr.DataArray(
        np.arange(float(n_steps)),
        dims=["valid_time"],
        coords={"valid_time": pd.date_range("2021-06-01", periods=n_steps, freq="6h")},
    ).chunk({"valid_time": timesteps_per_day})

    return _time(lambda: utils.daily_min_over_complete_days(da, 6.0).compute())


BENCHMARKS: dict[str, Callable[[], float]] = {
    "convert_init_time_to_valid_time": bench_convert_init_time_to_valid_time,
    "atmospheric_river_mask": bench_atmospheric_river_mask,
    "atmospheric_river_mask_forecast": bench_atmospheric_river_mask_forecast,
    "atmospheric_river_pipeline": bench_atmospheric_river_pipeline,
    "reduce_dataarray": bench_reduce_dataarray,
    "tc_spatial_mask": bench_tc_spatial_mask,
    "cape": bench_cape,
    "center_of_mass": bench_center_of_mass,
    "landfall_detection": bench_landfall_detection,
    "daily_min": bench_daily_min,
    "region_mask": bench_region_mask,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    selected = args.only or list(BENCHMARKS)
    results: dict[str, float] = {}

    for name in selected:
        try:
            elapsed = BENCHMARKS[name]()
        except Exception as exc:  # a broken benchmark must not hide the rest
            print(f"{name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        results[name] = elapsed
        print(f"{name}: {elapsed * 1000:.1f} ms")

    if args.compare and args.compare.exists():
        baseline = json.loads(args.compare.read_text())
        print("\ncomparison against baseline")
        for name, elapsed in results.items():
            if name not in baseline:
                continue
            ratio = baseline[name] / elapsed if elapsed else float("inf")
            verdict = (
                "faster"
                if ratio > 1.05
                else ("slower" if ratio < 0.95 else "unchanged")
            )
            print(
                f"  {name}: {baseline[name] * 1000:.1f} -> "
                f"{elapsed * 1000:.1f} ms ({ratio:.2f}x {verdict})"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
