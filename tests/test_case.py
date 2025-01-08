import pytest
import xarray as xr
import numpy as np
import extremeweatherbench.case as case
import pandas as pd
import datetime
import rioxarray
from extremeweatherbench.utils import Location


class TestCases:
    @pytest.fixture
    def mock_bad_dataset(self):
        data = np.random.rand(20, 180, 360)
        times = pd.date_range("2000-01-01", periods=20)
        latitudes = np.linspace(-90, 90, 180)
        longitudes = np.linspace(-180, 179, 360)
        dataset = xr.Dataset(
            {
                "air_temperature": (["time", "latitude", "longitude"], data),
                "eastward_wind": (["time", "latitude", "longitude"], data),
                "northward_wind": (["time", "latitude", "longitude"], data),
            },
            coords={
                "time": times,
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )
        return dataset

    @pytest.fixture
    def mock_good_dataset(self):
        init_time = pd.date_range("2000-01-01", periods=5)
        lead_time = range(0, 169, 6)
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
        return dataset

    @pytest.fixture
    def mock_datasets(self, mock_bad_dataset, mock_good_dataset):
        return [self.mock_bad_dataset, self.mock_good_dataset]

    def test_perform_subsetting_procedure_heatwave(self, dataset):
        heatwave_case = case.IndividualHeatWaveCase(
            id=20,
            title="Test Heatwave",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 10),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="heat_wave",
        )
        subset_dataset = heatwave_case.perform_subsetting_procedure(dataset())
        assert "air_temperature" in subset_dataset
        assert "eastward_wind" not in subset_dataset
        assert "northward_wind" not in subset_dataset

    def test_perform_subsetting_procedure_freeze(self, dataset):
        freeze_case = case.IndividualFreezeCase(
            id=10,
            title="Test Freeze",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 10),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="freeze",
        )
        subset_dataset = freeze_case.perform_subsetting_procedure(dataset())
        assert "air_temperature" in subset_dataset
        assert "eastward_wind" in subset_dataset
        assert "northward_wind" in subset_dataset

    @pytest.mark.parametrize("dataset", [mock_bad_dataset, mock_good_dataset])
    def test_individual_case(self, dataset):
        base_case = case.IndividualCase(
            id=10,
            title="Test Case",
            start_date=datetime.date(2000, 1, 1),
            end_date=datetime.date(2000, 1, 10),
            location={"latitude": 40, "longitude": -100},
            bounding_box_km=500,
            event_type="heat_wave",
        )
        valid_case = {
            "id": 10,
            "title": "Test Case",
            "start_date": datetime.date(2000, 1, 1),
            "end_date": datetime.date(2000, 1, 10),
            "location": Location(latitude=40, longitude=-100),
            "bounding_box_km": 500,
            "event_type": "heat_wave",
            "cross_listed": None,
            "data_vars": None,
        }
        assert base_case.__dict__ == valid_case
        assert base_case._subset_data_vars(dataset) is dataset
        assert base_case._subset_valid_times(dataset).time.shape[0] > 1
        with pytest.raises(NotImplementedError):
            base_case.perform_subsetting_procedure(dataset)
        assert base_case._check_for_forecast_data_availability(dataset) is True
