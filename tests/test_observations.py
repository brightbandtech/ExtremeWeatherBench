import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench.case import IndividualCase
from extremeweatherbench.observations import (
    ERA5,
    GHCN,
    LSR,
    DerivedVariable,
    IBTrACS,
    Observation,
)
from extremeweatherbench.utils import Location


class TestObservation:
    """Test the abstract Observation base class."""

    def test_observation_initialization(self):
        """Test that Observation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Observation()

    def test_maybe_derive_variables(self):
        """Test the _maybe_derive_variables method."""

        # Create a concrete subclass for testing
        class TestObservation(Observation):
            def _open_data_from_source(self, storage_options=None):
                pass

            def _subset_data_to_case(self, data, case, variables=None):
                pass

            def _maybe_convert_to_dataset(self, data):
                pass

        # Create a concrete DerivedVariable for testing
        class TestDerivedVariable(DerivedVariable):
            def compute(self, **kwargs):
                return xr.DataArray(
                    np.ones((2, 2)), dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1]}
                )

            def name(self):
                return "test_variable"

        obs = TestObservation()

        # Create test data
        data = xr.Dataset(
            {
                "existing_var": xr.DataArray(
                    np.ones((2, 2)), dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1]}
                )
            }
        )

        # Create a test case
        test_case = IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

        # Test with mixed string and DerivedVariable
        variables = ["existing_var", TestDerivedVariable]
        result = obs._maybe_derive_variables(data, variables, test_case)

        assert "existing_var" in result.data_vars
        assert "test_variable" in result.data_vars


class TestERA5:
    """Test the ERA5 observation class."""

    @pytest.fixture
    def era5_case(self):
        """Create a dummy case for ERA5 testing."""
        return IndividualCase(
            case_id_number=1,
            title="ERA5 Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
            data_vars=["2m_temperature", "10m_u_component_of_wind"],
        )

    @pytest.fixture
    def era5_observation(self):
        """Create an ERA5 observation instance."""
        return ERA5()

    @pytest.fixture
    def mock_era5_data(self):
        """Create mock ERA5 data."""
        time = pd.date_range("2021-06-20", "2021-06-25", freq="3h")
        lat = np.linspace(35, 45, 21)  # 35-45 degrees latitude
        lon = np.linspace(-105, -95, 21)  # -105 to -95 degrees longitude

        data = np.random.rand(len(time), len(lat), len(lon))

        return xr.Dataset(
            {
                "2m_temperature": (["time", "latitude", "longitude"], data + 20),
                "10m_u_component_of_wind": (["time", "latitude", "longitude"], data),
                "10m_v_component_of_wind": (["time", "latitude", "longitude"], data),
            },
            coords={"time": time, "latitude": lat, "longitude": lon},
        )

    def test_era5_initialization(self):
        """Test ERA5 initialization."""
        era5_obs = ERA5()
        assert (
            era5_obs.source
            == "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        )

    @patch("xarray.open_zarr")
    def test_open_data_from_source(
        self, mock_open_zarr, era5_observation, mock_era5_data
    ):
        """Test opening ERA5 data from source."""
        mock_open_zarr.return_value = mock_era5_data

        result = era5_observation._open_data_from_source()

        mock_open_zarr.assert_called_once_with(
            era5_observation.source, chunks=None, storage_options=dict(token="anon")
        )
        assert result.equals(mock_era5_data)

    def test_subset_data_to_case(self, era5_observation, era5_case, mock_era5_data):
        """Test subsetting ERA5 data to case."""
        variables = ["2m_temperature", "10m_u_component_of_wind"]

        result = era5_observation._subset_data_to_case(
            mock_era5_data, era5_case, variables
        )

        # Check that the result is a subset of the original data
        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "10m_u_component_of_wind" in result.data_vars
        assert len(result.time) <= len(mock_era5_data.time)

    def test_subset_data_to_case_invalid_variables(
        self, era5_observation, era5_case, mock_era5_data
    ):
        """Test that subsetting with invalid variables raises an error."""
        invalid_variables = ["nonexistent_variable"]

        with pytest.raises(
            ValueError, match="Variables .* not found in observation data"
        ):
            era5_observation._subset_data_to_case(
                mock_era5_data, era5_case, invalid_variables
            )

    def test_subset_data_to_case_invalid_data_type(self, era5_observation, era5_case):
        """Test that subsetting with invalid data type raises an error."""
        invalid_data = "not xarray data"

        with pytest.raises(ValueError, match="Expected xarray Dataset or DataArray"):
            era5_observation._subset_data_to_case(invalid_data, era5_case)

    def test_maybe_convert_to_dataset_dataarray(self, era5_observation, mock_era5_data):
        """Test converting DataArray to Dataset."""
        dataarray = mock_era5_data["2m_temperature"]
        result = era5_observation._maybe_convert_to_dataset(dataarray)

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars

    def test_maybe_convert_to_dataset_already_dataset(
        self, era5_observation, mock_era5_data
    ):
        """Test that Dataset is returned unchanged."""
        result = era5_observation._maybe_convert_to_dataset(mock_era5_data)

        assert result is mock_era5_data

    @patch("xarray.open_zarr")
    def test_run_pipeline(
        self, mock_open_zarr, era5_observation, era5_case, mock_era5_data
    ):
        """Test the complete ERA5 pipeline."""
        mock_open_zarr.return_value = mock_era5_data

        result = era5_observation.run_pipeline(
            era5_case, variables=["2m_temperature", "10m_u_component_of_wind"]
        )

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars
        assert "10m_u_component_of_wind" in result.data_vars


class TestGHCN:
    """Test the GHCN observation class."""

    @pytest.fixture
    def ghcn_case(self):
        """Create a dummy case for GHCN testing."""
        return IndividualCase(
            case_id_number=2,
            title="GHCN Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
            data_vars=["temperature", "humidity"],
        )

    @pytest.fixture
    def ghcn_observation(self):
        """Create a GHCN observation instance."""
        return GHCN()

    @pytest.fixture
    def mock_ghcn_data(self):
        """Create mock GHCN data as a polars LazyFrame."""
        # Create sample data
        dates = pd.date_range("2021-06-18", "2021-06-27", freq="1h")
        lats = np.linspace(35, 45, 11)
        lons = np.linspace(-105, -95, 11)

        data = []
        for date in dates:
            for lat in lats:
                for lon in lons:
                    data.append(
                        {
                            "time": date,
                            "latitude": lat,
                            "longitude": lon,
                            "temperature": np.random.rand() * 30 + 10,
                            "humidity": np.random.rand() * 100,
                        }
                    )

        df = pd.DataFrame(data)
        return pl.LazyFrame(df)

    def test_ghcn_initialization(self):
        """Test GHCN initialization."""
        ghcn_obs = GHCN()
        assert ghcn_obs.source == "gs://extremeweatherbench/datasets/ghcnh.parq"

    @patch("polars.scan_parquet")
    def test_open_data_from_source(
        self, mock_scan_parquet, ghcn_observation, mock_ghcn_data
    ):
        """Test opening GHCN data from source."""
        mock_scan_parquet.return_value = mock_ghcn_data

        result = ghcn_observation._open_data_from_source()

        mock_scan_parquet.assert_called_once_with(
            ghcn_observation.source, storage_options=None
        )
        assert result is mock_ghcn_data

    def test_subset_data_to_case(self, ghcn_observation, ghcn_case, mock_ghcn_data):
        """Test subsetting GHCN data to case."""
        variables = ["temperature", "humidity"]

        result = ghcn_observation._subset_data_to_case(
            mock_ghcn_data, ghcn_case, variables
        )

        # Check that the result is a LazyFrame
        assert isinstance(result, pl.LazyFrame)
        # Check that the result contains the expected columns
        schema_fields = [field for field in result.collect_schema()]
        assert "time" in schema_fields
        assert "latitude" in schema_fields
        assert "longitude" in schema_fields

    def test_subset_data_to_case_invalid_variables(
        self, ghcn_observation, ghcn_case, mock_ghcn_data
    ):
        """Test that subsetting with invalid variables raises an error."""
        invalid_variables = ["nonexistent_variable"]

        with pytest.raises(
            ValueError, match="Variables .* not found in observation data"
        ):
            ghcn_observation._subset_data_to_case(
                mock_ghcn_data, ghcn_case, invalid_variables
            )

    def test_subset_data_to_case_invalid_data_type(self, ghcn_observation, ghcn_case):
        """Test that subsetting with invalid data type raises an error."""
        invalid_data = "not a lazyframe"

        with pytest.raises(ValueError, match="Expected polars LazyFrame"):
            ghcn_observation._subset_data_to_case(invalid_data, ghcn_case)

    def test_maybe_convert_to_dataset_lazyframe(self, ghcn_observation):
        """Test converting LazyFrame to xarray Dataset."""
        # Create a simple LazyFrame with unique indexes
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-06-20", periods=3),
                "latitude": [40.0, 40.1, 40.2],
                "longitude": [-100.0, -100.1, -100.2],
                "temperature": [20.0, 21.0, 22.0],
            }
        )
        lazy_frame = pl.LazyFrame(df)

        result = ghcn_observation._maybe_convert_to_dataset(lazy_frame)

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars
        assert "time" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords

    def test_maybe_convert_to_dataset_with_duplicates(self, ghcn_observation):
        """Test converting LazyFrame with duplicates to xarray Dataset."""
        # Create a LazyFrame with duplicate indexes that will be resolved by drop_duplicates()
        df = pd.DataFrame(
            {
                "time": pd.date_range("2021-06-20", periods=2).repeat(2),
                "latitude": [40.0, 40.0, 40.1, 40.1],
                "longitude": [-100.0, -100.0, -100.1, -100.1],
                "temperature": [
                    20.0,
                    20.0,
                    21.0,
                    21.0,
                ],  # Same values to create true duplicates
            }
        )
        lazy_frame = pl.LazyFrame(df)

        with patch("extremeweatherbench.observations.logger") as mock_logger:
            result = ghcn_observation._maybe_convert_to_dataset(lazy_frame)

        assert isinstance(result, xr.Dataset)
        mock_logger.warning.assert_called_once()

    def test_maybe_convert_to_dataset_invalid_type(self, ghcn_observation):
        """Test that invalid data type raises an error."""
        invalid_data = "not a lazyframe"

        with pytest.raises(ValueError, match="Data is not a polars LazyFrame"):
            ghcn_observation._maybe_convert_to_dataset(invalid_data)

    @patch("polars.scan_parquet")
    def test_run_pipeline(
        self, mock_scan_parquet, ghcn_observation, ghcn_case, mock_ghcn_data
    ):
        """Test the complete GHCN pipeline."""
        mock_scan_parquet.return_value = mock_ghcn_data

        result = ghcn_observation.run_pipeline(
            ghcn_case, variables=["temperature", "humidity"]
        )

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars
        assert "humidity" in result.data_vars


class TestLSR:
    """Test the LSR observation class."""

    @pytest.fixture
    def lsr_case(self):
        """Create a dummy case for LSR testing."""
        return IndividualCase(
            case_id_number=3,
            title="LSR Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="severe_convective",
            data_vars=["report_type", "scale"],
        )

    @pytest.fixture
    def lsr_observation(self):
        """Create an LSR observation instance."""
        return LSR()

    @pytest.fixture
    def mock_lsr_data(self):
        """Create mock LSR data."""
        return pd.DataFrame(
            {
                "lat": ["40.0", "41.0", "42.0", "30.0", "50.0"],
                "lon": ["-100.0", "-101.0", "-102.0", "-80.0", "-120.0"],
                "time": [
                    "2021-06-21 12:00:00",
                    "2021-06-22 12:00:00",
                    "2021-06-23 12:00:00",
                    "2021-06-21 12:00:00",
                    "2021-06-21 12:00:00",
                ],
                "report_type": ["tor", "hail", "wind", "tor", "hail"],
                "scale": ["EF1", "1.5", "60", "EF2", "2.0"],
            }
        )

    def test_lsr_initialization(self):
        """Test LSR initialization."""
        lsr_obs = LSR()
        assert (
            lsr_obs.source
            == "gs://extremeweatherbench/datasets/lsr_01012020_04302025.parq"
        )

    @patch("pandas.read_parquet")
    def test_open_data_from_source(
        self, mock_read_parquet, lsr_observation, mock_lsr_data
    ):
        """Test opening LSR data from source."""
        mock_read_parquet.return_value = mock_lsr_data

        result = lsr_observation._open_data_from_source()

        mock_read_parquet.assert_called_once_with(
            lsr_observation.source, storage_options={"token": "anon"}
        )
        assert result.equals(mock_lsr_data)

    def test_subset_data_to_case(self, lsr_observation, lsr_case, mock_lsr_data):
        """Test subsetting LSR data to case."""
        variables = ["report_type", "scale"]

        result = lsr_observation._subset_data_to_case(
            mock_lsr_data, lsr_case, variables
        )

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check that the result contains the expected columns
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        assert "valid_time" in result.columns

    def test_subset_data_to_case_invalid_data_type(self, lsr_observation, lsr_case):
        """Test that subsetting with invalid data type raises an error."""
        invalid_data = "not a dataframe"

        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            lsr_observation._subset_data_to_case(invalid_data, lsr_case)

    def test_maybe_convert_to_dataset_dataframe(self, lsr_observation):
        """Test converting DataFrame to xarray Dataset."""
        df = pd.DataFrame(
            {
                "valid_time": pd.date_range("2021-06-20", periods=3),
                "latitude": [40.0, 40.1, 40.2],
                "longitude": [-100.0, -100.1, -100.2],
                "report_type": ["tor", "hail", "wind"],
            }
        )

        result = lsr_observation._maybe_convert_to_dataset(df)

        assert isinstance(result, xr.Dataset)
        assert "report_type" in result.data_vars
        assert "valid_time" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords

    def test_maybe_convert_to_dataset_with_duplicates(self, lsr_observation):
        """Test converting DataFrame with duplicates to xarray Dataset."""
        df = pd.DataFrame(
            {
                "valid_time": pd.date_range("2021-06-20", periods=2).repeat(2),
                "latitude": [40.0, 40.0, 40.1, 40.1],
                "longitude": [-100.0, -100.0, -100.1, -100.1],
                "report_type": ["tor", "tor", "hail", "hail"],
            }
        )

        with patch("extremeweatherbench.observations.logger") as mock_logger:
            result = lsr_observation._maybe_convert_to_dataset(df)

        assert isinstance(result, xr.Dataset)
        mock_logger.warning.assert_called_once()

    def test_maybe_convert_to_dataset_invalid_type(self, lsr_observation):
        """Test that invalid data type raises an error."""
        invalid_data = "not a dataframe"

        with pytest.raises(ValueError, match="Data is not a pandas DataFrame"):
            lsr_observation._maybe_convert_to_dataset(invalid_data)

    @patch("pandas.read_parquet")
    def test_run_pipeline(
        self, mock_read_parquet, lsr_observation, lsr_case, mock_lsr_data
    ):
        """Test the complete LSR pipeline."""
        mock_read_parquet.return_value = mock_lsr_data

        result = lsr_observation.run_pipeline(
            lsr_case, variables=["report_type", "scale"]
        )

        assert isinstance(result, xr.Dataset)
        assert "report_type" in result.data_vars
        assert "scale" in result.data_vars


class TestIBTrACS:
    """Test the IBTrACS observation class."""

    @pytest.fixture
    def ibtracs_case(self):
        """Create a dummy case for IBTrACS testing."""
        return IndividualCase(
            case_id_number=4,
            title="IBTrACS Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="tropical_cyclone",
            data_vars=["wind_speed", "pressure"],
        )

    @pytest.fixture
    def ibtracs_observation(self):
        """Create an IBTrACS observation instance."""
        return IBTrACS()

    def test_ibtracs_initialization(self):
        """Test IBTrACS initialization."""
        ibtracs_obs = IBTrACS()
        assert "ncei.noaa.gov" in ibtracs_obs.source

    @patch("polars.scan_csv")
    def test_open_data_from_source(self, mock_scan_csv, ibtracs_observation):
        """Test opening IBTrACS data from source."""
        mock_lazyframe = pl.LazyFrame({"test": [1, 2, 3]})
        mock_scan_csv.return_value = mock_lazyframe

        result = ibtracs_observation._open_data_from_source()

        mock_scan_csv.assert_called_once_with(
            ibtracs_observation.source, storage_options=None
        )
        assert result is mock_lazyframe

    def test_maybe_convert_to_dataset_invalid_type(self, ibtracs_observation):
        """Test that invalid data type raises an error."""
        invalid_data = "not a dataframe"

        with pytest.raises(ValueError):
            ibtracs_observation._maybe_convert_to_dataset(invalid_data)


class TestObservationIntegration:
    """Integration tests for observation classes."""

    def test_observation_inheritance(self):
        """Test that all observation classes properly inherit from Observation."""
        era5_obs = ERA5()
        ghcn_obs = GHCN()
        lsr_obs = LSR()
        ibtracs_obs = IBTrACS()

        assert isinstance(era5_obs, Observation)
        assert isinstance(ghcn_obs, Observation)
        assert isinstance(lsr_obs, Observation)
        assert isinstance(ibtracs_obs, Observation)

    def test_observation_sources(self):
        """Test that all observation classes have appropriate sources."""
        era5_obs = ERA5()
        ghcn_obs = GHCN()
        lsr_obs = LSR()
        ibtracs_obs = IBTrACS()

        assert "arco-era5" in era5_obs.source
        assert "ghcnh.parq" in ghcn_obs.source
        assert "lsr_" in lsr_obs.source
        assert "ibtracs" in ibtracs_obs.source.lower()

    def test_run_pipeline_with_derived_variables(self):
        """Test running pipeline with derived variables."""

        # Create a concrete DerivedVariable for testing
        class TestDerivedVariable(DerivedVariable):
            def compute(self, **kwargs):
                return xr.DataArray(
                    np.ones((2, 2)),
                    dims=["x", "y"],
                    coords={"x": [0, 1], "y": [0, 1]},
                )

            def name(self):
                return "test_variable"

        # Create a test case with derived variables
        test_case = IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
            data_vars=["existing_var"],
        )

        # Create a concrete observation class for testing
        class TestObservation(Observation):
            def _open_data_from_source(self, storage_options=None):
                return xr.Dataset(
                    {
                        "existing_var": xr.DataArray(
                            np.ones((2, 2)),
                            dims=["x", "y"],
                            coords={"x": [0, 1], "y": [0, 1]},
                        )
                    }
                )

            def _subset_data_to_case(self, data, case, variables=None):
                return data

            def _maybe_convert_to_dataset(self, data):
                return data

        obs = TestObservation()
        result = obs.run_pipeline(
            test_case, variables=["existing_var", TestDerivedVariable]
        )

        # Check that both the original variable and derived variable are present
        assert "existing_var" in result.data_vars
        assert "test_variable" in result.data_vars
