"""Profile / price the tropical-cyclone detection hot path (v1.1.1 functions).

Synthetic-but-realistic global 0.25 deg SLP + geopotential-thickness field with
localised closed lows and an IBTrACS-like track frame; no network:

    .venv/bin/python benchmarks/tc_detection_profile.py [--steps 24] [--profile]

Prices the array-side stages per timestep — `_create_spatial_mask`,
`_find_peaks_for_time_slice` (peak_local_max + contour validation) — and
extrapolates to a 90-day 6-hourly analysis run and to a 20-init x 120-lead
forecast run (where the same valid time is visited many times, so the mask is
rebuilt for an identical result every time). `--profile` prints the cProfile
top of the peak-search path.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np
import pandas as pd

from extremeweatherbench.events import tropical_cyclone as tc

NLAT, NLON = 721, 1440
NLOW = 40
N_TC = 12  # storms alive per valid time (IBTrACS rows)


def synthetic_field(steps: int, seed: int = 0):
    """A flat background with LOCALISED closed lows.

    A broad sinusoid would put the SLP anomaly level across the whole globe and
    the contourer would spend its life tracing mid-latitude waves instead of
    cyclones — the benchmark would price the wrong thing.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(90, -90, NLAT)
    lon = np.linspace(0, 359.75, NLON)
    lats, lons = np.meshgrid(lat, lon, indexing="ij")

    lows = [
        (
            rng.uniform(-60, 60),
            rng.uniform(0, 360),
            rng.uniform(4, 10),
            rng.uniform(40, 120),
        )
        for _ in range(NLOW)
    ]

    def gauss(lat_deg, lon_deg, amp, sigma):
        dlat = lats - lat_deg
        dlon = (lons - lon_deg + 180) % 360 - 180
        dlon = dlon * np.cos(np.radians(lat_deg))
        return amp * np.exp(-(dlat**2 + dlon**2) / (2 * sigma**2))

    slp = np.empty((steps, NLAT, NLON), np.float32)
    dz = np.empty((steps, NLAT, NLON), np.float32)
    for t in range(steps):
        f = np.full((NLAT, NLON), 101_325.0)
        g = np.full((NLAT, NLON), 5_000.0)
        for lat0, lon0, amp, sigma in lows:
            f -= gauss(
                lat0 + 0.5 * t, lon0 + 0.6 * t, amp * 12.0, sigma
            )  # up to ~1200 Pa deep
            g += gauss(lat0 + 0.5 * t, lon0 + 0.6 * t, amp * 30.0, sigma)
        slp[t] = f
        dz[t] = g
    return lat, lon, slp, dz


def make_ibtracs(steps: int, n_tc: int = N_TC):
    """IBTrACS-like frame: n_tc rows at every 6-hourly valid time of the window."""
    t0 = pd.Timestamp("2023-10-01 06:00")
    rows = []
    for i in range(n_tc):
        lat0, lon0 = 12 + 4 * i, 30 + 25 * i
        for s in range(steps):
            rows.append(
                {
                    "sid": f"s{i}",
                    "latitude": lat0 + 0.4 * s,
                    "longitude": lon0 + 0.5 * s,
                    "valid_time": t0 + pd.Timedelta(hours=6 * s),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument(
        "--inits", type=int, default=20, help="forecast inits for the extrapolation"
    )
    ap.add_argument("--leads", type=int, default=120, help="lead times per init")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    lat, lon, slp, dz = synthetic_field(a.steps)
    df = make_ibtracs(a.steps)
    min_dist = tc._degrees_to_gridpoints(2.0, lat, lon)
    valid_times = [
        pd.Timestamp("2023-10-01 06:00") + pd.Timedelta(hours=6 * i)
        for i in range(a.steps)
    ]
    kwargs = {
        "tc_track_data_df": df,
        "min_distance_between_peaks": min_dist,
        "lat_coords": lat,
        "lon_coords": lon,
        "max_spatial_distance_degrees": 5.0,
        "max_temporal_hours": 12.0,
        "slp_contour_magnitude": 200.0,
        "dz_contour_magnitude": -6.0,
        "use_contour_validation": True,
        "surface_pressure_threshold": 101_000.0,
        "latitude_max_degrees": 50.0,
        "max_gc_distance_slp_contour_degrees": 5.5,
        "max_gc_distance_dz_contour_degrees": 6.5,
    }

    t0 = time.perf_counter()
    mask = tc._create_spatial_mask(lat, lon, df[df.valid_time == valid_times[3]], 5.0)
    t_mask = time.perf_counter() - t0
    n_rows = int((df.valid_time == valid_times[3]).sum())

    t_slice = 0.0
    peaks = np.array([]).reshape(0, 2)
    for i in range(a.steps):
        t1 = time.perf_counter()
        peaks = tc._find_peaks_for_time_slice(
            slp[i], dz[i], valid_times[i], is_first_timestep=(i == 0), **kwargs
        )
        t_slice += time.perf_counter() - t1
    per_step = t_slice / a.steps

    print(
        f"grid {NLAT}x{NLON} @0.25deg, {a.steps} steps, {NLOW} lows, "
        f"{N_TC} IBTrACS rows/step, min_distance={min_dist} gp"
    )
    print(
        f"  _create_spatial_mask          {t_mask * 1000:8.0f} ms  ({n_rows} rows, "
        f"{mask.mean() * 100:.2f}% of the globe)"
    )
    print(
        f"  _find_peaks_for_time_slice    {per_step * 1000:8.0f} ms/step  "
        f"({peaks.shape[0]} validated peaks)"
    )
    print(f"  spatial-mask share of a step  {t_mask / per_step * 100:8.0f}%")
    analysis = 90 * 4  # 90 days at 6-hourly
    fcst = a.inits * a.leads
    print(
        f"  => 90-day analysis ({analysis} steps): {per_step * analysis:6.0f} s peak search, "
        f"{t_mask * analysis:6.0f} s of it spatial mask"
    )
    print(
        f"  => {a.inits}-init x {a.leads}-lead forecast ({fcst} steps): "
        f"{per_step * fcst / 60:6.1f} min peak search, "
        f"{t_mask * fcst / 60:6.1f} min of it spatial mask"
    )

    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        for i in range(min(4, a.steps)):
            tc._find_peaks_for_time_slice(
                slp[i], dz[i], valid_times[i], is_first_timestep=(i == 0), **kwargs
            )
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()
