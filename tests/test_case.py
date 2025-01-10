import pytest
import xarray as xr
import numpy as np
import extremeweatherbench.case as case
import pandas as pd
import datetime
import rioxarray
from extremeweatherbench.utils import Location


class TestGoodCases:
    @pytest.fixture
    def mock_dataset(self):
        init_time = pd.date_range("2000-01-01", periods=5)
        lead_time = range(0, 241, 6)
        data = np.random.rand(5, 180, 360, len(lead_time))
        latitudes = np.linspace(-90, 90, 180)
        longitudes = np.linspace(-180, 179, 360)
        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ["init_time", "latitude", "longitude", "lead_time"],
                    data,
                ),
                "eastward_wind": (
                    ["init_time", "latitude", "longitude", "lead_time"],
                    data,
                ),
                "northward_wind": (
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
        lead_time_grid, init_time_grid = np.meshgrid(
            dataset.lead_time, dataset.init_time
        )
        # Step 2: Flatten the meshgrid and convert lead_time to timedelta
        valid_time = init_time_grid.flatten() + pd.to_timedelta(
            lead_time_grid.flatten(), unit="h"
        )
        dataset.coords["time"] = valid_time
        return dataset

    def test_check_for_forecast_data_availability(self, mock_dataset):
        base_case = case.IndividualCase(
            id=10,
            title="Test Case",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 14),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="heat_wave",
        )
        assert base_case._check_for_forecast_data_availability(mock_dataset) is True

    def test_perform_subsetting_procedure_heatwave(self, mock_dataset):
        heatwave_case = case.IndividualHeatWaveCase(
            id=20,
            title="Test Heatwave",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 14),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="heat_wave",
        )
        subset_dataset = heatwave_case.perform_subsetting_procedure(mock_dataset)
        assert "air_temperature" in subset_dataset
        assert "eastward_wind" not in subset_dataset
        assert "northward_wind" not in subset_dataset

    def test_perform_subsetting_procedure_freeze(self, mock_dataset):
        freeze_case = case.IndividualFreezeCase(
            id=10,
            title="Test Freeze",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 14),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="freeze",
        )
        subset_dataset = freeze_case.perform_subsetting_procedure(mock_dataset)
        assert "air_temperature" in subset_dataset
        assert "eastward_wind" in subset_dataset
        assert "northward_wind" in subset_dataset

    def test_individual_case(self, mock_dataset):
        base_case = case.IndividualCase(
            id=10,
            title="Test Case",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 14),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="heat_wave",
        )
        valid_case = {
            "id": 10,
            "title": "Test Case",
            "start_date": datetime.date(2000, 1, 1),
            "end_date": datetime.date(2000, 1, 14),
            "location": Location(latitude=40, longitude=-100),
            "bounding_box_km": 500,
            "event_type": "heat_wave",
            "cross_listed": None,
            "data_vars": None,
        }
        assert base_case.__dict__ == valid_case
        assert base_case._subset_data_vars(mock_dataset) is mock_dataset
        with pytest.raises(NotImplementedError):
            base_case.perform_subsetting_procedure(mock_dataset)
        assert base_case._check_for_forecast_data_availability(mock_dataset) is True
