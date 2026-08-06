import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click import testing

from extremeweatherbench import calc
from tests.fixtures.forecasts import (
    dataset_to_dataarray,
    make_sample_forecast_dataset,
    make_sample_forecast_with_valid_time,
    make_sample_results_dataarray_list,
)
from tests.fixtures.grids import make_sample_sparse_target_dataarray
from tests.fixtures.observations import (
    make_sample_era5_dataset,
    make_sample_ghcn_dataframe,
    make_sample_gridded_obs_dataarray,
    make_sample_gridded_obs_dataset,
    make_sample_lsr_dataframe,
    make_sample_point_obs_df_with_attrs,
)
from tests.fixtures.tracks import make_sample_ibtracs_dataframe


@pytest.fixture
def sample_forecast_dataset():
    sample_forecast_dataset = make_sample_forecast_dataset()
    return sample_forecast_dataset


@pytest.fixture
def sample_forecast_dataarray():
    sample_forecast_dataarray = dataset_to_dataarray(make_sample_forecast_dataset())
    return sample_forecast_dataarray


@pytest.fixture
def sample_subset_forecast_dataarray():
    sample_forecast_dataset = dataset_to_dataarray(make_sample_forecast_dataset())
    subset_sample_forecast_dataset = sample_forecast_dataset.sel(
        latitude=slice(40, 45), longitude=slice(100, 105)
    )
    return subset_sample_forecast_dataset


@pytest.fixture
def sample_results_dataarray_list():
    sample_results_dataarray_list = make_sample_results_dataarray_list()
    return sample_results_dataarray_list


@pytest.fixture
def runner():
    """Fixture for Click CLI runner."""
    return testing.CliRunner()


@pytest.fixture
def temp_config_dir():
    """Fixture that creates a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def sample_yaml_config():
    """Fixture that returns the path to the sample YAML config file."""
    return pathlib.Path(__file__).parent / "data" / "sample_config.yaml"


@pytest.fixture
def sample_gridded_obs_dataarray():
    """Fixture for sample gridded observations DataArray."""
    return make_sample_gridded_obs_dataarray()


@pytest.fixture
def sample_gridded_obs_dataset():
    """Fixture for sample gridded observations Dataset."""
    return make_sample_gridded_obs_dataset()


@pytest.fixture
def sample_point_obs_df_with_attrs():
    """Fixture for sample point observations DataFrame with attributes."""
    return make_sample_point_obs_df_with_attrs()


@pytest.fixture
def sample_era5_dataset():
    """Fixture for sample ERA5-like dataset."""
    return make_sample_era5_dataset()


@pytest.fixture
def sample_forecast_with_valid_time():
    """Fixture for forecast dataset with valid_time dimension."""
    return make_sample_forecast_with_valid_time()


@pytest.fixture
def sample_ghcn_dataframe():
    """Fixture for sample GHCN polars DataFrame."""
    return make_sample_ghcn_dataframe()


@pytest.fixture
def sample_lsr_dataframe():
    """Fixture for sample LSR DataFrame."""
    return make_sample_lsr_dataframe()


@pytest.fixture
def sample_ibtracs_dataframe():
    """Fixture for sample IBTrACS polars DataFrame."""
    return make_sample_ibtracs_dataframe()


@pytest.fixture
def temp_zarr_file():
    """Fixture that creates a temporary zarr file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        zarr_path = pathlib.Path(temp_dir) / "test.zarr"
        # Create a simple zarr dataset
        ds = make_sample_era5_dataset()
        ds.to_zarr(zarr_path)
        yield str(zarr_path)


@pytest.fixture
def temp_parquet_file():
    """Fixture that creates a temporary parquet file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = pathlib.Path(temp_dir) / "test.parquet"
        # Create a simple parquet file
        df = make_sample_ghcn_dataframe()
        df.write_parquet(parquet_path)
        yield str(parquet_path)


@pytest.fixture
def sample_calc_dataset():
    """Create a sample dataset for calc testing."""
    time = pd.date_range("2023-01-01", periods=5, freq="6h")
    lat = np.linspace(20, 50, 16)
    lon = np.linspace(-120, -80, 21)
    level = [1000, 850, 700, 500, 300, 200]

    # Create realistic meteorological data.
    # Use a seeded generator so results are bit-reproducible across Python
    # versions; the global numpy RNG state can differ between interpreters and
    # caused flaky failures when extreme draws produced physically impossible
    # surface heights (negative elevation far below sea level) that pushed
    # barometric-formula surface pressures above the 105 000 Pa test bound.
    rng = np.random.default_rng(42)

    data_shape_3d = (len(time), len(lat), len(lon))
    data_shape_4d = (len(time), len(level), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                rng.normal(101325, 1000, data_shape_3d),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude"],
                rng.normal(0, 10, data_shape_3d),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude"],
                rng.normal(0, 10, data_shape_3d),
            ),
            "geopotential": (
                ["time", "level", "latitude", "longitude"],
                rng.normal(5000, 1000, data_shape_4d) * calc.g0,
            ),
            "geopotential_at_surface": (
                ["time", "latitude", "longitude"],
                rng.normal(500, 200, data_shape_3d) * calc.g0,
            ),
            "eastward_wind": (
                ["time", "level", "latitude", "longitude"],
                rng.normal(0, 15, data_shape_4d),
            ),
            "northward_wind": (
                ["time", "level", "latitude", "longitude"],
                rng.normal(0, 15, data_shape_4d),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                rng.uniform(0.001, 0.02, data_shape_4d),
            ),
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "level": level,
        },
    )

    return dataset


@pytest.fixture
def sample_sparse_target_dataarray():
    """Fixture for sample sparse target dataarray."""
    return make_sample_sparse_target_dataarray()


@pytest.fixture
def sample_sparse_target_dataset():
    """Fixture for sample sparse target dataset."""
    return xr.Dataset(
        {
            "target": make_sample_sparse_target_dataarray(),
        },
    )
