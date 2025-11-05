#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gcsfs>=2024.12.0",
#     "numpy>=1.24",
#     "metpy>=1.5",
#     "xarray>=2023.1",
#     "tqdm>=4.65",
#     "zarr>=3.1.0",
# ]
# ///
"""Generate reference CAPE/CIN data using MetPy for unit testing.

This script generates a reference dataset by:

1. Fetching ERA5 atmospheric profiles.
2. Computing CAPE/CIN with MetPy for all profiles, and then subsetting to get a diverse
sample of profiles for testing.
3. Creating synthetic pathological test profiles to help test edge cases.
4. Saving inputs and reference outputs for unit testing

We use MetPy to help filter the profiles to ensure they are valid as well as to get a reference
value that we can use for testing. Our goal is to functionally reproduce the MetPy
CAPE/CIN estimates with some tolerance for relative error.

Usage:
    # Be sure you've set ADC credentials so that you can read from the ARCO-ERA5 bucket
    gcloud auth application-default login
    # Run script with uv
    uv run generate_reference_data.py
"""

import datetime
import logging
import pathlib

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set the output directory to the tests/data directory, relative from this script's
# location in the repo.
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "data"


def compute_metpy_reference(
    pressure: np.ndarray, temperature: np.ndarray, dewpoint: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CAPE/CIN using MetPy for all profiles."""
    from metpy.calc import mixed_layer_cape_cin
    from metpy.units import units

    n_profiles = pressure.shape[0]

    cape_ref = np.zeros(n_profiles)
    cin_ref = np.zeros(n_profiles)

    logger.info(f"Computing MetPy reference for {n_profiles} profiles...")

    for i in range(n_profiles):
        p = pressure[i, :] * units.hPa
        t = temperature[i, :] * units.kelvin
        td = dewpoint[i, :] * units.kelvin

        try:
            cape, cin = mixed_layer_cape_cin(p, t, td)
            cape_ref[i] = cape.magnitude
            cin_ref[i] = cin.magnitude
        except Exception:
            cape_ref[i] = np.nan
            cin_ref[i] = np.nan

    return cape_ref, cin_ref


def fetch_era5_data(radius_deg: float = 5.0):
    """Fetch ERA5 data from ARCO.

    To be sure we're getting interesting profiles to compute CAPE, we choose a small
    region of interest in the SE USA from mid-day April 27, 2011.
    """

    logger.info(f"Fetching ERA5 data (radius={radius_deg}°)...")

    # Reference point/time
    ref_lat = 33.43303
    ref_lon = 360 - 88.89015
    ref_time = datetime.datetime(2011, 4, 27, 18, 0, 0)

    # ERA5 ARCO dataset
    url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

    # Open dataset
    logger.info("  Opening ARCO ERA5 dataset...")
    ds = xr.open_zarr(url, chunks=None, consolidated=True)

    # Select time (just one timestamp for testing)
    ds_time = ds.sel(time=ref_time)

    # Select spatial region
    lat_min, lat_max = ref_lat - radius_deg, ref_lat + radius_deg
    lon_min, lon_max = ref_lon - radius_deg, ref_lon + radius_deg

    ds_subset = ds_time.sel(
        latitude=slice(lat_max, lat_min),  # Descending
        longitude=slice(lon_min, lon_max),
    )

    # Standard pressure levels (hPa)
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    # Extract variables at pressure levels
    logger.info("  Extracting variables...")
    t_data = ds_subset["temperature"].sel(level=levels)
    z_data = ds_subset["geopotential"].sel(level=levels)
    logger.info(
        f"  Extracted temperature and geopotential data for {t_data.shape[0]} levels"
    )

    # For dewpoint, need specific humidity to compute
    q_data = ds_subset["specific_humidity"].sel(level=levels)

    # Convert to numpy and reshape
    logger.info("  Converting to arrays...")
    # Coerce arrays to shape (n_lat, n_lon, n_levels)
    t_array = t_data.transpose("latitude", "longitude", "level").values
    z_array = z_data.transpose("latitude", "longitude", "level").values
    q_array = q_data.transpose("latitude", "longitude", "level").values
    logger.info(f"  t_array shape: {t_array.shape}")

    # Reshape to (n_profiles, n_levels)
    n_lat, n_lon, n_levels = t_array.shape
    n_profiles = n_lat * n_lon

    logger.info(f"  Reshaped to (n_profiles, n_levels) = ({n_profiles}, {n_levels})")

    temperature = t_array.reshape(n_profiles, n_levels)
    geopotential = z_array.reshape(n_profiles, n_levels)
    specific_humidity = q_array.reshape(n_profiles, n_levels)

    # Broadcast pressure levels
    pressure = np.tile(levels, (n_profiles, 1)).astype(np.float64)

    # Convert specific humidity to dewpoint
    logger.info("  Computing dewpoint from specific humidity...")
    dewpoint = np.zeros_like(temperature)

    for i in range(n_profiles):
        for j in range(n_levels):
            p = pressure[i, j] * 100  # Convert to Pa
            q = specific_humidity[i, j]
            t = temperature[i, j]

            # Vapor pressure from specific humidity
            e = (q * p) / (0.622 + 0.378 * q)

            # Dewpoint from Clausius-Clapeyron (Bolton 1980)
            if e > 0:
                td = 243.5 / ((17.67 / np.log(e / 611.2)) - 1) + 273.15
            else:
                td = t - 10  # Fallback

            dewpoint[i, j] = td

    logger.info(f"  Fetched {n_profiles} profiles with {n_levels} levels")

    return pressure, temperature, dewpoint, geopotential


def create_pathological_profiles():
    """Create synthetic pathological test profiles.

    These test edge cases like:
    - No convection (stable atmosphere)
    - Surface-based convection
    - Elevated convection
    - Very weak CAPE
    - Very strong CAPE
    - Multiple crossings
    """
    profiles = {}

    # Profile 1: Completely stable (no convection)
    p1 = np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        dtype=np.float64,
    )
    t1 = np.array(
        [273, 270, 268, 260, 255, 250, 240, 230, 225, 220, 215, 210, 205],
        dtype=np.float64,
    )
    td1 = t1 - 20  # Very dry
    z1 = 29.3 * 273 * np.log(1000 / p1)

    profiles["stable_no_convection"] = {
        "pressure": p1,
        "temperature": t1,
        "dewpoint": td1,
        "geopotential": z1,
        "expected_cape": 0.0,
        "expected_cin": 0.0,  # Should be negative but we expect 0 CAPE
        "description": "Completely stable atmosphere, no convection possible",
    }

    # Profile 2: Strong surface-based convection
    p2 = np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        dtype=np.float64,
    )
    t2 = np.array(
        [303, 297, 290, 278, 270, 260, 245, 230, 225, 220, 215, 210, 205],
        dtype=np.float64,
    )
    td2 = np.array(
        [298, 290, 283, 268, 255, 245, 230, 215, 210, 205, 200, 195, 190],
        dtype=np.float64,
    )
    z2 = (
        np.array(
            [
                0,
                700,
                1500,
                3000,
                4200,
                5600,
                7200,
                9200,
                10400,
                11800,
                13500,
                16000,
                20500,
            ],
            dtype=np.float64,
        )
        * 9.81
    )

    profiles["strong_surface_based"] = {
        "pressure": p2,
        "temperature": t2,
        "dewpoint": td2,
        "geopotential": z2,
        "expected_cape": 2000.0,  # Approximate
        "expected_cin": 0.0,  # Minimal CIN
        "description": "Strong surface-based convection, high CAPE",
    }

    # Profile 3: Elevated convection with CIN
    p3 = np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        dtype=np.float64,
    )
    t3 = np.array(
        [295, 293, 291, 280, 272, 262, 247, 232, 227, 222, 217, 212, 207],
        dtype=np.float64,
    )
    td3 = np.array(
        [280, 278, 276, 270, 260, 250, 235, 220, 215, 210, 205, 200, 195],
        dtype=np.float64,
    )
    z3 = (
        np.array(
            [
                0,
                700,
                1500,
                3000,
                4200,
                5600,
                7200,
                9200,
                10400,
                11800,
                13500,
                16000,
                20500,
            ],
            dtype=np.float64,
        )
        * 9.81
    )

    profiles["elevated_with_cin"] = {
        "pressure": p3,
        "temperature": t3,
        "dewpoint": td3,
        "geopotential": z3,
        "expected_cape": 500.0,  # Moderate
        "expected_cin": -100.0,  # Significant CIN
        "description": "Elevated convection with significant CIN",
    }

    # Profile 4: Very weak CAPE (the problematic case)
    p4 = np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        dtype=np.float64,
    )
    t4 = np.array(
        [288, 284, 280, 270, 264, 256, 243, 230, 225, 220, 215, 210, 205],
        dtype=np.float64,
    )
    td4 = t4 - 5  # Moderate moisture
    z4 = 29.3 * 288 * np.log(1000 / p4)

    profiles["weak_cape"] = {
        "pressure": p4,
        "temperature": t4,
        "dewpoint": td4,
        "geopotential": z4,
        "expected_cape": 50.0,  # Very weak
        "expected_cin": 0.0,
        "description": "Very weak CAPE (<100 J/kg), tests low CAPE accuracy",
    }

    # Profile 5: Isothermal layer (edge case)
    p5 = np.array(
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        dtype=np.float64,
    )
    t5 = np.array(
        [288, 285, 283, 275, 275, 275, 245, 230, 225, 220, 215, 210, 205],
        dtype=np.float64,
    )
    td5 = t5 - 10
    z5 = (
        np.array(
            [
                0,
                700,
                1500,
                3000,
                4200,
                5600,
                7200,
                9200,
                10400,
                11800,
                13500,
                16000,
                20500,
            ],
            dtype=np.float64,
        )
        * 9.81
    )

    profiles["isothermal_layer"] = {
        "pressure": p5,
        "temperature": t5,
        "dewpoint": td5,
        "geopotential": z5,
        "expected_cape": 0.0,
        "expected_cin": 0.0,
        "description": "Isothermal layer in mid-levels",
    }

    return profiles


def generate_cape_reference_data(radius_deg: float = 2.0):
    logger.info("Fetching ERA5 data...")
    pressure, temperature, dewpoint, geopotential = fetch_era5_data(
        radius_deg=radius_deg
    )

    # Take a subset for testing (don't need all ~6400 profiles for unit tests)
    # Use a diverse sample
    n_test = min(500, pressure.shape[0])
    indices = np.linspace(0, pressure.shape[0] - 1, n_test, dtype=int)

    pressure_test = pressure[indices]
    temperature_test = temperature[indices]
    dewpoint_test = dewpoint[indices]
    geopotential_test = geopotential[indices]

    logger.info(f"Using {n_test} profiles for unit test reference")

    cape_ref, cin_ref = compute_metpy_reference(
        pressure_test, temperature_test, dewpoint_test
    )

    # Use the reference CAPE calculations to filter out any bad profiles
    valid = ~(np.isnan(cape_ref) | np.isnan(cin_ref))
    n_valid = np.sum(valid)

    logger.info(f"Valid profiles: {n_valid}/{n_test}")

    pressure_test = pressure_test[valid]
    temperature_test = temperature_test[valid]
    dewpoint_test = dewpoint_test[valid]
    geopotential_test = geopotential_test[valid]
    cape_ref = cape_ref[valid]
    cin_ref = cin_ref[valid]

    # 3. Save real profile reference data
    real_output = OUTPUT_DIR / "era5_reference.npz"
    logger.info(f"Saving ERA5 reference data to {real_output}...")
    np.savez(
        real_output,
        pressure=pressure_test,
        temperature=temperature_test,
        dewpoint=dewpoint_test,
        geopotential=geopotential_test,
        n_profiles=n_valid,
        cape_reference=cape_ref,
        cin_reference=cin_ref,
        description="ERA5 profiles with MetPy reference CAPE/CIN for unit testing",
    )

    # 4. Create and save pathological test profiles
    logger.info("Generating pathological test profiles...")
    pathological = create_pathological_profiles()

    # Save pathological profiles
    pathological_output = OUTPUT_DIR / "pathological_profiles.npz"
    logger.info(f"Saving pathological profiles to {pathological_output}...")

    # Convert to arrays for saving
    names = list(pathological.keys())
    n_patho = len(names)

    patho_pressure = np.array([pathological[name]["pressure"] for name in names])
    patho_temperature = np.array([pathological[name]["temperature"] for name in names])
    patho_dewpoint = np.array([pathological[name]["dewpoint"] for name in names])
    patho_geopotential = np.array(
        [pathological[name]["geopotential"] for name in names]
    )
    patho_cape_reference = np.array(
        [pathological[name]["expected_cape"] for name in names]
    )
    patho_cin_reference = np.array(
        [pathological[name]["expected_cin"] for name in names]
    )
    patho_descriptions = [pathological[name]["description"] for name in names]

    np.savez(
        pathological_output,
        names=names,
        pressure=patho_pressure,
        temperature=patho_temperature,
        dewpoint=patho_dewpoint,
        geopotential=patho_geopotential,
        cape_reference=patho_cape_reference,
        cin_reference=patho_cin_reference,
        descriptions=patho_descriptions,
        n_profiles=n_patho,
    )


if __name__ == "__main__":
    generate_cape_reference_data()
