import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from extremeweatherbench.case import IndividualCase
from extremeweatherbench.observations import ERA5, GHCN, LSR, Observation
from extremeweatherbench.utils import Location


class TestObservation:
    """Test the abstract Observation base class."""

    def test_observation_initialization(self):
        """Test that Observation can be initialized with a case."""
        # Create a dummy case
        dummy_case = IndividualCase(
            id=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

        # Test that we can't instantiate the abstract class directly
        with pytest.raises(TypeError):
            Observation(dummy_case)


class TestERA5:
    """Test the ERA5 observation class."""

    @pytest.fixture
    def era5_case(self):
        """Create a dummy case for ERA5 testing."""
        return IndividualCase(
            id=1,
            title="ERA5 Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

    @pytest.fixture
    def era5_observation(self, era5_case):
        """Create an ERA5 observation instance."""
        return ERA5(era5_case)

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

    def test_era5_initialization(self, era5_case):
        """Test ERA5 initialization."""
        era5_obs = ERA5(era5_case)
        assert era5_obs.case == era5_case

    @patch("xarray.open_zarr")
    def test_open_data_from_source(
        self, mock_open_zarr, era5_observation, mock_era5_data
    ):
        """Test opening ERA5 data from source."""
        mock_open_zarr.return_value = mock_era5_data

        result = era5_observation._open_data_from_source("dummy_source")

        mock_open_zarr.assert_called_once_with(
            "dummy_source", chunks=None, storage_options=dict(token="anon")
        )
        assert result.equals(mock_era5_data)

    def test_subset_data_to_case(self, era5_observation, mock_era5_data):
        """Test subsetting ERA5 data to case."""
        variables = ["2m_temperature", "10m_u_component_of_wind"]

        result = era5_observation._subset_data_to_case(mock_era5_data, variables)

        # Check that bounding box coordinates were set
        assert hasattr(era5_observation.case, "latitude_min")
        assert hasattr(era5_observation.case, "latitude_max")
        assert hasattr(era5_observation.case, "longitude_min")
        assert hasattr(era5_observation.case, "longitude_max")

        # Check that the result contains only the specified variables
        assert set(result.data_vars) == set(variables)

        # Check that time range is correct
        assert result.time.min() >= pd.Timestamp("2021-06-20")
        assert result.time.max() <= pd.Timestamp("2021-06-25")

    def test_subset_data_to_case_invalid_variables(
        self, era5_observation, mock_era5_data
    ):
        """Test that subsetting with invalid variables raises an error."""
        invalid_variables = ["nonexistent_variable"]

        with pytest.raises(
            ValueError, match="Variables .* not found in observation data"
        ):
            era5_observation._subset_data_to_case(mock_era5_data, invalid_variables)

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
    def test_run_pipeline(self, mock_open_zarr, era5_observation, mock_era5_data):
        """Test the complete ERA5 pipeline."""
        mock_open_zarr.return_value = mock_era5_data
        variables = ["2m_temperature"]

        result = era5_observation.run_pipeline("dummy_source", variables=variables)

        assert isinstance(result, xr.Dataset)
        assert "2m_temperature" in result.data_vars


class TestGHCN:
    """Test the GHCN observation class."""

    @pytest.fixture
    def ghcn_case(self):
        """Create a dummy case for GHCN testing."""
        return IndividualCase(
            id=2,
            title="GHCN Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

    @pytest.fixture
    def ghcn_observation(self, ghcn_case):
        """Create a GHCN observation instance."""
        return GHCN(ghcn_case)

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

    def test_ghcn_initialization(self, ghcn_case):
        """Test GHCN initialization."""
        ghcn_obs = GHCN(ghcn_case)
        assert ghcn_obs.case == ghcn_case

    @patch("polars.scan_parquet")
    def test_open_data_from_source(
        self, mock_scan_parquet, ghcn_observation, mock_ghcn_data
    ):
        """Test opening GHCN data from source."""
        mock_scan_parquet.return_value = mock_ghcn_data

        result = ghcn_observation._open_data_from_source("dummy_source.parquet")

        mock_scan_parquet.assert_called_once_with(
            "dummy_source.parquet", storage_options=None
        )
        assert result is mock_ghcn_data

    def test_subset_data_to_case(self, ghcn_observation, mock_ghcn_data):
        """Test subsetting GHCN data to case."""
        variables = ["temperature", "humidity"]

        result = ghcn_observation._subset_data_to_case(mock_ghcn_data, variables)

        # Check that bounding box coordinates were set
        assert hasattr(ghcn_observation.case, "latitude_min")
        assert hasattr(ghcn_observation.case, "latitude_max")
        assert hasattr(ghcn_observation.case, "longitude_min")
        assert hasattr(ghcn_observation.case, "longitude_max")

        # Check that the result is still a LazyFrame
        assert isinstance(result, pl.LazyFrame)

        # Check that all required variables are selected
        collected = result.collect()
        expected_columns = variables + ["time", "latitude", "longitude"]
        assert all(col in collected.columns for col in expected_columns)

    def test_subset_data_to_case_invalid_variables(
        self, ghcn_observation, mock_ghcn_data
    ):
        """Test that subsetting with invalid variables raises an error."""
        invalid_variables = ["nonexistent_variable"]

        with pytest.raises(
            ValueError, match="Variables .* not found in observation data"
        ):
            ghcn_observation._subset_data_to_case(mock_ghcn_data, invalid_variables)

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
    def test_run_pipeline(self, mock_scan_parquet, ghcn_observation, mock_ghcn_data):
        """Test the complete GHCN pipeline."""
        mock_scan_parquet.return_value = mock_ghcn_data
        variables = ["temperature"]

        result = ghcn_observation.run_pipeline(
            "dummy_source.parquet", variables=variables
        )

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars


class TestLSR:
    """Test the LSR observation class."""

    @pytest.fixture
    def lsr_case(self):
        """Create a dummy case for LSR testing."""
        return IndividualCase(
            id=3,
            title="LSR Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="severe_convective",
        )

    @pytest.fixture
    def lsr_observation(self, lsr_case):
        """Create an LSR observation instance."""
        return LSR(lsr_case)

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

    def test_lsr_initialization(self, lsr_case):
        """Test LSR initialization."""
        lsr_obs = LSR(lsr_case)
        assert lsr_obs.case == lsr_case

    @patch("pandas.read_parquet")
    def test_open_data_from_source(
        self, mock_read_parquet, lsr_observation, mock_lsr_data
    ):
        """Test opening LSR data from source."""
        mock_read_parquet.return_value = mock_lsr_data

        result = lsr_observation._open_data_from_source("dummy_source.parquet")

        mock_read_parquet.assert_called_once_with(
            "dummy_source.parquet", storage_options=None
        )
        assert result.equals(mock_lsr_data)

    def test_subset_data_to_case(self, lsr_observation, mock_lsr_data):
        """Test subsetting LSR data to case."""
        variables = ["report_type", "scale"]

        result = lsr_observation._subset_data_to_case(mock_lsr_data, variables)

        # Check that bounding box coordinates were set to central CONUS
        assert hasattr(lsr_observation.case, "latitude_min")
        assert hasattr(lsr_observation.case, "latitude_max")
        assert hasattr(lsr_observation.case, "longitude_min")
        assert hasattr(lsr_observation.case, "longitude_max")

        # Check that data types were converted
        assert result["latitude"].dtype == float
        assert result["longitude"].dtype == float
        assert pd.api.types.is_datetime64_any_dtype(result["valid_time"])

        # Check that only data within the central CONUS bounding box is included
        assert all(result["latitude"] >= 24.0)
        assert all(result["latitude"] <= 49.0)
        assert all(result["longitude"] >= -109.0)
        assert all(result["longitude"] <= -89.0)

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
    def test_run_pipeline(self, mock_read_parquet, lsr_observation, mock_lsr_data):
        """Test the complete LSR pipeline."""
        mock_read_parquet.return_value = mock_lsr_data
        variables = ["report_type"]

        result = lsr_observation.run_pipeline(
            "dummy_source.parquet", variables=variables
        )

        assert isinstance(result, xr.Dataset)
        assert "report_type" in result.data_vars


class TestObservationIntegration:
    """Integration tests for observation classes."""

    def test_observation_inheritance(self):
        """Test that all observation classes properly inherit from Observation."""
        dummy_case = IndividualCase(
            id=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

        era5_obs = ERA5(dummy_case)
        ghcn_obs = GHCN(dummy_case)
        lsr_obs = LSR(dummy_case)

        assert isinstance(era5_obs, Observation)
        assert isinstance(ghcn_obs, Observation)
        assert isinstance(lsr_obs, Observation)

    def test_case_attribute_consistency(self):
        """Test that all observation classes properly set the case attribute."""
        dummy_case = IndividualCase(
            id=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 6, 20),
            end_date=datetime.datetime(2021, 6, 25),
            location=Location(latitude=40.0, longitude=-100.0),
            bounding_box_degrees=5.0,
            event_type="heat_wave",
        )

        era5_obs = ERA5(dummy_case)
        ghcn_obs = GHCN(dummy_case)
        lsr_obs = LSR(dummy_case)

        assert era5_obs.case == dummy_case
        assert ghcn_obs.case == dummy_case
        assert lsr_obs.case == dummy_case
