import pytest
import extremeweatherbench.case as case
import rioxarray  # noqa: F401
from extremeweatherbench.utils import Location
import datetime


class TestGoodCases:
    def test_perform_subsetting_procedure_heatwave(self, mock_forecast_dataset):
        heatwave_case = case.IndividualHeatWaveCase(
            id=20,
            title="Test Heatwave",
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2000, 1, 14),
            location=Location(latitude=40, longitude=-100),
            bounding_box_km=500,
            event_type="heat_wave",
        )
        subset_dataset = heatwave_case.perform_subsetting_procedure(
            mock_forecast_dataset
        )
        assert len(subset_dataset["latitude"]) > 0
        assert len(subset_dataset["longitude"]) > 0

    def test_individual_case(self, mock_forecast_dataset):
        base_case = case.IndividualCase(
            id=10,
            title="Test Case",
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2000, 1, 14),
            location=Location(latitude=40, longitude=-100),
            bounding_box_km=500,
            event_type="heat_wave",
        )
        valid_case = {
            "id": 10,
            "title": "Test Case",
            "start_date": datetime.datetime(2000, 1, 1),
            "end_date": datetime.datetime(2000, 1, 14),
            "location": Location(latitude=40, longitude=-100),
            "bounding_box_km": 500,
            "event_type": "heat_wave",
            "cross_listed": None,
            "data_vars": None,
        }
        assert base_case.__dict__ == valid_case
        assert (
            base_case._subset_data_vars(mock_forecast_dataset) is mock_forecast_dataset
        )
        with pytest.raises(NotImplementedError):
            base_case.perform_subsetting_procedure(mock_forecast_dataset)
