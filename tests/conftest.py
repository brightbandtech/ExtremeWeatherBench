import pathlib
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr
from click import testing

from extremeweatherbench import calc


def is_lazy(obj: Any) -> bool:
    """Whether an xarray object is still dask-backed and unevaluated."""
    if isinstance(obj, xr.Dataset):
        return bool(obj.chunks)
    if isinstance(obj, xr.DataArray):
        return obj.chunks is not None
    return hasattr(obj, "dask")


def assert_lazy(obj: Any, msg: str = "") -> None:
    """Fail if a result has been computed instead of staying lazy.

    Guards against reintroducing eager evaluation into paths that are meant
    to compose into a single dask graph, which is otherwise invisible until
    something runs out of memory on real data.
    """
    if not is_lazy(obj):
        raise AssertionError(f"expected a lazy result, got a computed one. {msg}")


def make_sample_sparse_target_dataarray() -> xr.DataArray:
    # Create a simple sparse array with known coordinates
    coords = ([0, 1, 2], [0, 1, 0])  # (lat_indices, lon_indices)
    data = [1.0, 2.0, 3.0]  # values at those coordinates
    shape = (3, 2)  # (lat, lon)

    sparse_array = sparse.COO(coords, data, shape=shape)

    # Create xarray DataArray with sparse data
    da = xr.DataArray(
        sparse_array,
        dims=["latitude", "longitude"],
        coords={"latitude": [10.0, 20.0, 30.0], "longitude": [100.0, 110.0]},
    )
    return da


def make_sample_gridded_obs_dataset():
    time = pd.date_range("2021-06-20", freq="3h", periods=200)
    data = np.random.RandomState(21897820).standard_normal(size=(len(time), 181, 360))
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], 20 + 5 * data),
            "tp": (["time", "latitude", "longitude"], data),
            "10m_u_component_of_wind": (["time", "latitude", "longitude"], data),
            "10m_v_component_of_wind": (["time", "latitude", "longitude"], data),
        },
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["2m_temperature"].loc[
        dict(
            time="2021-06-21 18:00",
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 25
    return dataset


def make_sample_point_obs_df():
    # Create sample point observations DataFrame
    data = {
        "time": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 06:00"]),
        "station": ["A100", "B200"],
        "call": ["KWEW", "KBCE"],
        "name": ["WEST CENTRAL", "EAST CENTRAL"],
        "latitude": [40.5, 41.8],
        "longitude": [-99.5, -99.8],
        "elev": [1000, 1100],
        "id": [1, 2],
        "surface_air_temperature": [20.0, 21.0],
    }
    df = pd.DataFrame(data)
    return df


def make_sample_forecast_dataset():
    init_time = pd.date_range("2021-06-20", periods=5)
    lead_time = range(0, 241, 6)
    data = np.random.RandomState(21897820).standard_normal(
        size=(len(init_time), 181, 360, len(lead_time)),
    )
    latitudes = np.linspace(-90, 90, 181)
    longitudes = np.linspace(0, 359, 360)
    dataset = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "latitude", "longitude", "lead_time"],
                20 + 5 * data,
            ),
            "surface_eastward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
            "surface_northward_wind": (
                ["init_time", "latitude", "longitude", "lead_time"],
                data,
            ),
        },
        coords={
            "init_time": init_time,
            "latitude": latitudes,
            "longitude": longitudes,
            "lead_time": lead_time,
        },
    )
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-21 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 24
    # Set a specific value for a specific time and location to remove ambiguity
    dataset["surface_air_temperature"].loc[
        dict(
            init_time="2021-06-20 00:00",
            lead_time=42,
            latitude=slice(40, 45),
            longitude=slice(100, 105),
        )
    ] = 23
    return dataset


def make_sample_results_dataarray_list():
    results_da_list = [
        xr.DataArray(
            data=[5],
            dims=["lead_time"],
            coords={"lead_time": [0]},
        ),
        xr.DataArray(
            data=[6],
            dims=["lead_time"],
            coords={"lead_time": [6]},
        ),
    ]
    return results_da_list


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]


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


def make_sample_gridded_obs_dataarray():
    """Create a sample gridded observations DataArray."""
    dataset = make_sample_gridded_obs_dataset()
    return dataset["2m_temperature"]


def make_sample_point_obs_df_with_attrs():
    """Create sample point observations DataFrame with attributes."""
    df = make_sample_point_obs_df()
    df.attrs = {
        "metadata_vars": ["station", "call", "name", "elev", "id"],
    }
    return df


def make_sample_era5_dataset():
    """Create a sample ERA5-like dataset with time dimension."""
    time = pd.date_range("2021-06-20", freq="6h", periods=50)
    data = np.random.RandomState(12345).standard_normal(size=(len(time), 91, 180))
    latitudes = np.linspace(-90, 90, 91)
    longitudes = np.linspace(0, 359, 180)

    dataset = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], 273.15 + 10 * data),
            "mean_sea_level_pressure": (
                ["time", "latitude", "longitude"],
                101325 + 1000 * data,
            ),
        },
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )
    return dataset


def make_sample_forecast_with_valid_time():
    """Create a forecast dataset with valid_time dimension instead of
    init_time/lead_time."""
    valid_time = pd.date_range("2021-06-20", freq="6h", periods=40)
    data = np.random.RandomState(54321).standard_normal(size=(len(valid_time), 91, 180))
    latitudes = np.linspace(-90, 90, 91)
    longitudes = np.linspace(0, 359, 180)

    dataset = xr.Dataset(
        {
            "surface_air_temperature": (
                ["valid_time", "latitude", "longitude"],
                273.15 + 10 * data,
            ),
            "surface_pressure": (
                ["valid_time", "latitude", "longitude"],
                101325 + 1000 * data,
            ),
        },
        coords={
            "valid_time": valid_time,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    return dataset


def make_sample_ghcn_dataframe():
    """Create a sample GHCN-like polars DataFrame."""
    import polars as pl

    dates = pd.date_range("2021-06-20", periods=100, freq="6h")
    n_stations = 5

    # Create combinations of stations and times
    station_ids = [f"STATION_{i:03d}" for i in range(n_stations)]

    data = []
    for station_id in station_ids:
        for date in dates:
            lat = 40 + np.random.normal(0, 5)
            lon = -100 + np.random.normal(0, 10)
            temp = 273.15 + np.random.normal(20, 5)

            data.append(
                {
                    "valid_time": date,
                    "station_id": station_id,
                    "latitude": lat,
                    "longitude": lon,
                    "surface_air_temperature": temp,
                }
            )

    return pl.DataFrame(data)


def make_sample_lsr_dataframe():
    """Create a sample Local Storm Report DataFrame."""
    data = {
        "valid_time": pd.date_range("2021-06-20", periods=20, freq="1h"),
        "latitude": np.random.uniform(30, 50, 20),
        "longitude": np.random.uniform(-110, -90, 20),
        "report_type": np.random.choice(["wind", "hail", "tor"], 20),
        "magnitude": np.random.uniform(0, 100, 20),
    }
    return pd.DataFrame(data)


def make_sample_ibtracs_dataframe():
    """Create a sample IBTrACS-like polars DataFrame."""
    import polars as pl

    data = {
        "valid_time": [
            "2021-06-20 00:00:00",
            "2021-06-20 06:00:00",
            "2021-06-20 12:00:00",
        ],
        "tc_name": ["TESTCYCLONE", "TESTCYCLONE", "TESTCYCLONE"],
        "latitude": [25.0, 26.0, 27.0],
        "longitude": [280.0, 281.0, 282.0],
        "surface_wind_speed": [30.0, 35.0, 40.0],
        "air_pressure_at_mean_sea_level": [1010.0, 1005.0, 1000.0],
    }
    return pl.DataFrame(data)


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


# Dataset builders shared across the test modules.
def make_ar_input_dataset(
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


def make_chunked_global_grid_dataset(n_time=8, n_lat=181, n_lon=360):
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


def make_coarse_global_grid_dataset():
    """Coarse global grid for region masking."""
    return xr.Dataset(
        {"t": (["latitude", "longitude"], np.zeros((73, 144)))},
        coords={
            "latitude": np.linspace(-90, 90, 73),
            "longitude": np.linspace(-180, 177.5, 144),
        },
    )


def make_daily_series_dataarray(n_days, timesteps_per_day=4, chunk=True, drop_last=0):
    """Hourly-ish series over whole days, optionally with a truncated last day."""
    n_steps = n_days * timesteps_per_day - drop_last
    freq = f"{24 // timesteps_per_day}h"
    valid_time = pd.date_range("2021-06-01", periods=n_steps, freq=freq)
    values = np.arange(float(n_steps))
    da = xr.DataArray(values, dims=["valid_time"], coords={"valid_time": valid_time})
    if chunk:
        da = da.chunk({"valid_time": timesteps_per_day})
    return da


def make_global_grid_coords(n_lat=181, n_lon=360):
    """Global-ish grid, so a local storm covers a small part of it."""
    return (
        np.linspace(-90.0, 90.0, n_lat),
        np.linspace(0.0, 359.0, n_lon),
    )


def make_gulf_coast_track(n_points):
    """Track running north out of the Gulf of Mexico onto Louisiana."""
    lons = np.linspace(-91.5, -91.0, n_points)
    lats = np.linspace(24.0, 32.0, n_points)
    return lons, lats


def make_ibtracs_frame_for_dataset(ds: xr.Dataset) -> xr.Dataset:
    """IBTrACS stub matching the storm centre in ``make_single_init_tc_dataset``."""
    valid_times = ds.valid_time.values
    return xr.Dataset(
        {
            "latitude": (["valid_time"], np.full(len(valid_times), 20.0)),
            "longitude": (["valid_time"], np.full(len(valid_times), -70.0)),
        },
        coords={"valid_time": valid_times},
    )


def make_init_lead_forecast_dataset(n_init, n_lead, chunk=True, lead_unit=None):
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


def make_landfall_dataarray(n_init=64, chunk=True, all_nan=False):
    """Landfall-shaped DataArray indexed by init_time."""
    values = np.full(n_init, np.nan) if all_nan else np.arange(float(n_init))
    da = xr.DataArray(
        values,
        dims=["init_time"],
        coords={"init_time": pd.date_range("2021-08-20", periods=n_init, freq="6h")},
    )
    return da.chunk({"init_time": 8}) if chunk else da


def make_pattern_stack_dataarray(
    n_lead=3, n_valid=4, n_lat=9, n_lon=11, chunk=False, seed=0
):
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


def make_pressure_column_dataarray(
    n_time=4, n_lat=3, n_lon=3, levels=(1000.0, 850.0, 700.0, 500.0)
):
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


def make_pressure_level_dataset(chunk=True):
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


def make_single_init_tc_dataset(n_lead: int = 12, n_wind_strong: int = 10):
    """Build a minimal synthetic TC dataset with one clear init_time.

    Grid: 1° resolution, 21×21 points (lat 10–30°, lon -80 to -60°).
    Storm: SLP=98 000 Pa at centre (lat=20, lon=-70) every (lead, valid) pair
    on the diagonal (same init_time T0).
    Wind: 20 m/s at one gridpoint east of centre for the first
    ``n_wind_strong`` diagonal pairs; 2 m/s everywhere else.
    Contour validation is OFF so only the wind filter is exercised.

    With a 1° grid ``_degrees_to_gridpoints(2.0, ...)`` = 2 gridpoints, so
    the ±2-pt neighbourhood around the centre includes the +1-pt east cell.

    Expected: n_wind_strong detections have neighbourhood wind ≥ 10 m/s.
    """
    lat = np.arange(10.0, 31.0, 1.0)  # 21 pts, 1 ° spacing
    lon = np.arange(-80.0, -59.0, 1.0)  # 21 pts, 1 ° spacing
    n_lat, n_lon = len(lat), len(lon)
    c_lat, c_lon = 10, 10  # centre indices → lat=20°, lon=-70°

    T0 = pd.Timestamp("2023-09-10")
    lead_h = np.arange(n_lead) * 6  # hours
    lead_td = (lead_h * np.timedelta64(1, "h")).astype("timedelta64[ns]")
    valid_times = pd.date_range(T0, periods=n_lead, freq="6h")

    # init_time[lt, vt] = valid_time[vt] - lead_time[lt]
    init_2d = np.array(
        [
            [valid_times[vt].to_datetime64() - lead_td[lt] for vt in range(n_lead)]
            for lt in range(n_lead)
        ]
    )

    # SLP: 98 000 Pa at centre for every (lt, vt) pair; 102 000 elsewhere
    slp = np.full((n_lead, n_lead, n_lat, n_lon), 102000.0)
    for k in range(n_lead):
        slp[k, k, c_lat, c_lon] = 98000.0

    # Wind: 20 m/s east of centre for the first n_wind_strong diagonal pairs
    wind = np.full((n_lead, n_lead, n_lat, n_lon), 2.0)
    for k in range(n_wind_strong):
        wind[k, k, c_lat, c_lon + 1] = 20.0

    # Geopotential thickness: zeros (contour validation disabled)
    dz = np.zeros((n_lead, n_lead, n_lat, n_lon))

    ds = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                slp,
            ),
            "surface_wind_speed": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                wind,
            ),
            "geopotential_thickness": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                dz,
            ),
        },
        coords={
            "lead_time": lead_td,
            "valid_time": valid_times,
            "latitude": lat,
            "longitude": lon,
            "init_time": (["lead_time", "valid_time"], init_2d),
        },
    )
    return ds


def make_sparse_grid_dataarray():
    """DataArray backed by a sparse.COO array."""
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


def make_spatial_dataarray(n_time=6, n_lat=8, n_lon=8, chunk=True):
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


def make_tc_track_frame(n_rows=4, latitude=15.0, longitude=140.0):
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2021-08-01", periods=n_rows, freq="6h"),
            "latitude": np.full(n_rows, latitude),
            "longitude": np.full(n_rows, longitude),
        }
    )


def make_tc_track_target_dataset(n_time=4):
    """Observed-track dataset used to filter forecast candidates."""
    return xr.Dataset(
        {"intensity": ("valid_time", np.arange(float(n_time)))},
        coords={
            "valid_time": pd.date_range("2021-09-01", periods=n_time, freq="6h"),
            "latitude": ("valid_time", np.linspace(15.0, 25.0, n_time)),
            "longitude": ("valid_time", np.linspace(-75.0, -65.0, n_time)),
        },
    )


def make_tc_tracker_forecast_dataset(n_time=4, seed=0):
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


def make_track_dataarray(lons, lats, chunk=False):
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


def make_unchunked_target_dataset(n_time=40, n_lat=64, n_lon=64):
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
