import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner


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
    return CliRunner()


@pytest.fixture
def temp_config_dir():
    """Fixture that creates a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_yaml_config():
    """Fixture that returns the path to the sample YAML config file."""
    return Path(__file__).parent / "data" / "sample_config.yaml"


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
        zarr_path = Path(temp_dir) / "test.zarr"
        # Create a simple zarr dataset
        ds = make_sample_era5_dataset()
        ds.to_zarr(zarr_path)
        yield str(zarr_path)


@pytest.fixture
def temp_parquet_file():
    """Fixture that creates a temporary parquet file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "test.parquet"
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

    # Create realistic meteorological data
    data_shape_3d = (len(time), len(lat), len(lon))
    data_shape_4d = (len(time), len(level), len(lat), len(lon))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                np.random.normal(101325, 1000, data_shape_3d),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape_3d),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape_3d),
            ),
            "geopotential": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(5000, 1000, data_shape_4d) * 9.80665,
            ),
            "geopotential_at_surface": (
                ["time", "latitude", "longitude"],
                np.random.normal(500, 200, data_shape_3d) * 9.80665,
            ),
            "eastward_wind": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(0, 15, data_shape_4d),
            ),
            "northward_wind": (
                ["time", "level", "latitude", "longitude"],
                np.random.normal(0, 15, data_shape_4d),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                np.random.uniform(0.001, 0.02, data_shape_4d),
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
def sample_derived_dataset():
    """Create a sample xarray Dataset for testing."""
    time = pd.date_range("2021-06-20", freq="6h", periods=8)
    latitudes = np.linspace(30, 50, 11)
    longitudes = np.linspace(250, 270, 21)
    level = [1000, 850, 700, 500, 300, 200]

    # Create realistic sample data
    np.random.seed(42)
    base_data = np.random.normal(
        20, 5, size=(len(time), len(latitudes), len(longitudes))
    )
    level_data = np.random.normal(
        0, 10, size=(len(time), len(level), len(latitudes), len(longitudes))
    )

    dataset = xr.Dataset(
        {
            # Basic surface variables
            "air_pressure_at_mean_sea_level": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    101325, 1000, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_eastward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    5, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_northward_wind": (
                ["time", "latitude", "longitude"],
                np.random.normal(
                    2, 3, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            "surface_wind_speed": (
                ["time", "latitude", "longitude"],
                np.random.uniform(
                    0, 15, size=(len(time), len(latitudes), len(longitudes))
                ),
            ),
            # 3D atmospheric variables
            "eastward_wind": (
                ["time", "level", "latitude", "longitude"],
                level_data + np.random.normal(10, 5, size=level_data.shape),
            ),
            "northward_wind": (
                ["time", "level", "latitude", "longitude"],
                level_data + np.random.normal(3, 5, size=level_data.shape),
            ),
            "specific_humidity": (
                ["time", "level", "latitude", "longitude"],
                np.random.exponential(0.008, size=level_data.shape),
            ),
            "geopotential": (
                ["time", "level", "latitude", "longitude"],
                level_data * 100 + np.random.normal(50000, 5000, size=level_data.shape),
            ),
            # Test variables
            "test_variable_1": (["time", "latitude", "longitude"], base_data),
            "test_variable_2": (["time", "latitude", "longitude"], base_data + 5),
            "single_variable": (["time", "latitude", "longitude"], base_data * 2),
        },
        coords={
            "time": time,
            "latitude": latitudes,
            "longitude": longitudes,
            "level": level,
        },
    )

    return dataset


@pytest.fixture
def sample_tc_forecast_dataset():
    """Create a sample forecast dataset for TC testing."""
    valid_time = pd.date_range("2023-09-01", periods=3, freq="12h")
    prediction_timedelta = np.array([0, 12, 24, 36], dtype="timedelta64[h]")
    lat = np.linspace(10, 40, 16)
    lon = np.linspace(-90, -60, 16)

    # Create realistic meteorological data
    data_shape = (len(valid_time), len(lat), len(lon), len(prediction_timedelta))

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["valid_time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(101325, 1000, data_shape),
            ),
            "surface_eastward_wind": (
                ["valid_time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(0, 10, data_shape),
            ),
            "surface_northward_wind": (
                ["valid_time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(0, 10, data_shape),
            ),
            "geopotential": (
                ["valid_time", "latitude", "longitude", "prediction_timedelta"],
                np.random.normal(5000, 1000, data_shape) * 9.80665,
            ),
        },
        coords={
            "valid_time": valid_time,
            "latitude": lat,
            "longitude": lon,
            "prediction_timedelta": prediction_timedelta,
        },
    )

    return dataset


@pytest.fixture
def sample_tc_tracks_dataset():
    """Create a sample TC tracks dataset."""
    time = pd.date_range("2023-09-01", periods=3, freq="12h")
    prediction_timedelta = np.array([0, 12, 24, 36], dtype="timedelta64[h]")

    data_shape = (len(time), len(prediction_timedelta))

    dataset = xr.Dataset(
        {
            "tc_slp": (
                ["time", "prediction_timedelta"],
                np.random.normal(101000, 1000, data_shape),
            ),
            "tc_latitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(15, 35, data_shape),
            ),
            "tc_longitude": (
                ["time", "prediction_timedelta"],
                np.random.uniform(-85, -65, data_shape),
            ),
            "tc_vmax": (
                ["time", "prediction_timedelta"],
                np.random.uniform(20, 60, data_shape),
            ),
            "track_id": (
                ["time", "prediction_timedelta"],
                np.random.randint(1, 5, data_shape),
            ),
        },
        coords={
            "time": time,
            "prediction_timedelta": prediction_timedelta,
        },
    )

    return dataset


@pytest.fixture
def realistic_forecast_dataset(self):
    """Create dataset matching the structure that caused the original bug."""
    lead_times = np.arange(0, 42, 6)  # 7 lead times
    valid_times = pd.date_range(
        "2023-09-01", periods=20, freq="6h"
    )  # Smaller for testing
    lat = np.linspace(10, 40, 31)  # Smaller grid for testing
    lon = np.linspace(240, 360, 41)

    # Create init_time with (lead_time, valid_time) dimensions
    init_time_base = pd.date_range("2023-08-31", periods=len(lead_times), freq="6h")
    init_time_grid = np.broadcast_to(
        init_time_base.values.reshape(-1, 1), (len(lead_times), len(valid_times))
    )

    data_shape = (len(lead_times), len(valid_times), len(lat), len(lon))
    level_shape = (len(lead_times), len(valid_times), 6, len(lat), len(lon))

    return xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                np.random.normal(101325, 1000, data_shape),
            ),
            "surface_eastward_wind": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape),
            ),
            "surface_northward_wind": (
                ["lead_time", "valid_time", "latitude", "longitude"],
                np.random.normal(0, 10, data_shape),
            ),
            "geopotential": (
                ["lead_time", "valid_time", "level", "latitude", "longitude"],
                np.random.normal(5000, 1000, level_shape) * 9.80665,
            ),
        },
        coords={
            "lead_time": lead_times,
            "valid_time": valid_times,
            "latitude": lat,
            "longitude": lon,
            "level": [1000, 850, 700, 500, 300, 200],
            "init_time": (["lead_time", "valid_time"], init_time_grid),
        },
    )


@pytest.fixture
def sample_ibtracs_dataset():
    """Create a sample IBTrACS dataset for testing."""
    valid_time = pd.date_range("2023-09-01", periods=10, freq="6h")

    dataset = xr.Dataset(
        {
            "air_pressure_at_mean_sea_level": (
                ["valid_time"],
                np.random.uniform(950, 1010, len(valid_time)),
            ),
            "latitude": (["valid_time"], np.linspace(15, 30, len(valid_time))),
            "longitude": (["valid_time"], np.linspace(-80, -70, len(valid_time))),
        },
        coords={"valid_time": valid_time},
    )
    dataset.attrs["source"] = "IBTrACS"

    return dataset
