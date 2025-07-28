import datetime

from extremeweatherbench import case
from extremeweatherbench.regions import Region


class TestGoodCases:
    def test_subset_region_heatwave(self, sample_forecast_dataset):
        heatwave_case = case.IndividualHeatWaveCase(
            case_id_number=20,
            title="Test Heatwave",
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2000, 1, 14),
            location=Region.create(latitude=40, longitude=-100, bounding_box_degrees=5),
            event_type="heat_wave",
        )
        subset_dataset = heatwave_case.subset_region(sample_forecast_dataset)
        assert len(subset_dataset["latitude"]) > 0
        assert len(subset_dataset["longitude"]) > 0

    def test_individual_case(self, sample_forecast_dataset):
        # Create a single region instance to use for both the case and expected dict
        region = Region.create(latitude=40, longitude=-100, bounding_box_degrees=5)

        base_case = case.IndividualCase(
            case_id_number=10,
            title="Test Case",
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2000, 1, 14),
            location=region,
            event_type="heat_wave",
        )
        valid_case = {
            "case_id_number": 10,
            "title": "Test Case",
            "start_date": datetime.datetime(2000, 1, 1),
            "end_date": datetime.datetime(2000, 1, 14),
            "location": region,
            "event_type": "heat_wave",
            "data_vars": None,
            "cross_listed": None,
        }
        assert base_case.__dict__ == valid_case
        assert (
            base_case._subset_data_vars(sample_forecast_dataset)
            is sample_forecast_dataset
        )
        # Test that subset_region works
        subset = base_case.subset_region(sample_forecast_dataset)
        assert len(subset.latitude) < len(sample_forecast_dataset.latitude)
        assert len(subset.longitude) < len(sample_forecast_dataset.longitude)
