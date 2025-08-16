"""Tests for the cases module."""

import datetime
from unittest.mock import Mock

import pytest

from extremeweatherbench import cases
from extremeweatherbench.regions import BoundingBoxRegion, CenteredRegion


class TestIndividualCase:
    """Test the IndividualCase dataclass."""

    def test_individual_case_creation(self):
        """Test IndividualCase creation with valid parameters."""
        region = CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )

        case = cases.IndividualCase(
            case_id_number=10,
            title="Test Case",
            start_date=datetime.datetime(2000, 1, 1),
            end_date=datetime.datetime(2000, 1, 14),
            location=region,
            event_type="heat_wave",
        )

        assert case.case_id_number == 10
        assert case.title == "Test Case"
        assert case.start_date == datetime.datetime(2000, 1, 1)
        assert case.end_date == datetime.datetime(2000, 1, 14)
        assert case.location == region
        assert case.event_type == "heat_wave"

    def test_individual_case_with_different_region_types(self):
        """Test IndividualCase with different region types."""
        # Test with CenteredRegion
        centered_region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        case1 = cases.IndividualCase(
            case_id_number=1,
            title="Centered Case",
            start_date=datetime.datetime(2021, 6, 1),
            end_date=datetime.datetime(2021, 6, 15),
            location=centered_region,
            event_type="heat_wave",
        )

        assert isinstance(case1.location, CenteredRegion)

        # Test with BoundingBoxRegion
        bbox_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        case2 = cases.IndividualCase(
            case_id_number=2,
            title="Bbox Case",
            start_date=datetime.datetime(2021, 7, 1),
            end_date=datetime.datetime(2021, 7, 15),
            location=bbox_region,
            event_type="drought",
        )

        assert isinstance(case2.location, BoundingBoxRegion)

    def test_individual_case_validation(self):
        """Test IndividualCase with various edge cases."""
        region = CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )

        # Test with very short time period
        short_case = cases.IndividualCase(
            case_id_number=999,
            title="Short Case",
            start_date=datetime.datetime(2021, 1, 1, 12, 0),
            end_date=datetime.datetime(2021, 1, 1, 18, 0),
            location=region,
            event_type="storm",
        )

        assert short_case.end_date > short_case.start_date


class TestIndividualCaseCollection:
    """Test the IndividualCaseCollection dataclass."""

    def test_case_collection_creation(self):
        """Test IndividualCaseCollection creation."""
        region1 = CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )
        region2 = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=8.0
        )

        case1 = cases.IndividualCase(
            case_id_number=1,
            title="Case 1",
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 15),
            location=region1,
            event_type="heat_wave",
        )

        case2 = cases.IndividualCase(
            case_id_number=2,
            title="Case 2",
            start_date=datetime.datetime(2021, 6, 1),
            end_date=datetime.datetime(2021, 6, 15),
            location=region2,
            event_type="drought",
        )

        collection = cases.IndividualCaseCollection(cases=[case1, case2])

        assert len(collection.cases) == 2
        assert collection.cases[0] == case1
        assert collection.cases[1] == case2

    def test_empty_case_collection(self):
        """Test IndividualCaseCollection with no cases."""
        empty_collection = cases.IndividualCaseCollection(cases=[])
        assert len(empty_collection.cases) == 0


class TestCaseOperator:
    """Test the CaseOperator dataclass."""

    def test_case_operator_creation(self):
        """Test CaseOperator creation with mock objects."""
        region = CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )

        case = cases.IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 15),
            location=region,
            event_type="heat_wave",
        )

        # Create mock objects for metric, target, and forecast
        mock_metric = Mock()
        mock_target = Mock()
        mock_forecast = Mock()

        operator = cases.CaseOperator(
            case_metadata=case,
            metric=mock_metric,
            target=mock_target,
            forecast=mock_forecast,
        )

        assert operator.case_metadata == case
        assert operator.metric == mock_metric
        assert operator.target == mock_target
        assert operator.forecast == mock_forecast


class TestLoadIndividualCases:
    """Test the load_individual_cases function."""

    def test_load_individual_cases_basic(self):
        """Test loading individual cases from dictionary."""
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Test Case 1",
                    "start_date": datetime.datetime(2021, 1, 1),
                    "end_date": datetime.datetime(2021, 1, 15),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 40.0,
                            "longitude": -100.0,
                            "bounding_box_degrees": 5.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 2,
                    "title": "Test Case 2",
                    "start_date": datetime.datetime(2021, 6, 1),
                    "end_date": datetime.datetime(2021, 6, 15),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 35.0,
                            "latitude_max": 45.0,
                            "longitude_min": -110.0,
                            "longitude_max": -95.0,
                        },
                    },
                    "event_type": "drought",
                },
            ]
        }

        collection = cases.load_individual_cases(cases_dict)

        assert isinstance(collection, cases.IndividualCaseCollection)
        assert len(collection.cases) == 2

        # Check first case
        case1 = collection.cases[0]
        assert case1.case_id_number == 1
        assert case1.title == "Test Case 1"
        assert case1.event_type == "heat_wave"
        assert isinstance(case1.location, CenteredRegion)

        # Check second case
        case2 = collection.cases[1]
        assert case2.case_id_number == 2
        assert case2.title == "Test Case 2"
        assert case2.event_type == "drought"
        assert isinstance(case2.location, BoundingBoxRegion)

    def test_load_individual_cases_empty(self):
        """Test loading empty cases list."""
        empty_cases_dict = {"cases": []}

        collection = cases.load_individual_cases(empty_cases_dict)

        assert isinstance(collection, cases.IndividualCaseCollection)
        assert len(collection.cases) == 0


class TestBuildCaseOperators:
    """Test the build_case_operators function."""

    def test_build_case_operators_basic(self):
        """Test building case operators from cases and evaluation objects."""
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Heat Wave Case",
                    "start_date": datetime.datetime(2021, 1, 1),
                    "end_date": datetime.datetime(2021, 1, 15),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 40.0,
                            "longitude": -100.0,
                            "bounding_box_degrees": 5.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 2,
                    "title": "Drought Case",
                    "start_date": datetime.datetime(2021, 6, 1),
                    "end_date": datetime.datetime(2021, 6, 15),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 35.0,
                            "longitude": -105.0,
                            "bounding_box_degrees": 8.0,
                        },
                    },
                    "event_type": "drought",
                },
            ]
        }

        # Create mock evaluation objects
        mock_eval_obj1 = Mock()
        mock_eval_obj1.event_type = ["heat_wave"]
        mock_eval_obj1.metric = Mock()
        mock_eval_obj1.target = Mock()
        mock_eval_obj1.forecast = Mock()

        mock_eval_obj2 = Mock()
        mock_eval_obj2.event_type = ["drought", "heat_wave"]
        mock_eval_obj2.metric = Mock()
        mock_eval_obj2.target = Mock()
        mock_eval_obj2.forecast = Mock()

        evaluation_objects = [mock_eval_obj1, mock_eval_obj2]

        operators = cases.build_case_operators(cases_dict, evaluation_objects)

        # Should create 3 operators:
        # - Heat wave case with eval_obj1 (matches heat_wave)
        # - Heat wave case with eval_obj2 (matches heat_wave)
        # - Drought case with eval_obj2 (matches drought)
        assert len(operators) == 3

        # Check that all operators are CaseOperator instances
        for operator in operators:
            assert isinstance(operator, cases.CaseOperator)
            assert hasattr(operator, "case_metadata")
            assert hasattr(operator, "metric")
            assert hasattr(operator, "target")
            assert hasattr(operator, "forecast")

    def test_build_case_operators_no_matches(self):
        """Test building case operators when no event types match."""
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Storm Case",
                    "start_date": datetime.datetime(2021, 1, 1),
                    "end_date": datetime.datetime(2021, 1, 15),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 40.0,
                            "longitude": -100.0,
                            "bounding_box_degrees": 5.0,
                        },
                    },
                    "event_type": "storm",
                },
            ]
        }

        # Create evaluation object that doesn't match storm
        mock_eval_obj = Mock()
        mock_eval_obj.event_type = ["heat_wave", "drought"]
        mock_eval_obj.metric = Mock()
        mock_eval_obj.target = Mock()
        mock_eval_obj.forecast = Mock()

        evaluation_objects = [mock_eval_obj]

        operators = cases.build_case_operators(cases_dict, evaluation_objects)

        # Should create no operators since no event types match
        assert len(operators) == 0

    def test_build_case_operators_invalid_cases_dict(self):
        """Test build_case_operators with invalid cases_dict."""
        invalid_cases_dict = {"cases": "not_a_list"}
        mock_eval_obj = Mock()

        with pytest.raises(
            TypeError, match="cases_dict\\['cases'\\] must be a list of cases"
        ):
            cases.build_case_operators(invalid_cases_dict, [mock_eval_obj])


class TestCasesIntegration:
    """Integration tests for the cases module."""

    def test_full_workflow(self):
        """Test the full workflow from dictionary to case operators."""
        # Define cases dictionary as might come from YAML
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 100,
                    "title": "California Heat Wave 2021",
                    "start_date": datetime.datetime(2021, 6, 15),
                    "end_date": datetime.datetime(2021, 6, 25),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 37.7749,
                            "longitude": -122.4194,
                            "bounding_box_degrees": 5.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 101,
                    "title": "Texas Drought 2021",
                    "start_date": datetime.datetime(2021, 7, 1),
                    "end_date": datetime.datetime(2021, 8, 31),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 29.0,
                            "latitude_max": 34.0,
                            "longitude_min": -104.0,
                            "longitude_max": -94.0,
                        },
                    },
                    "event_type": "drought",
                },
            ]
        }

        # Load individual cases
        collection = cases.load_individual_cases(cases_dict)
        assert len(collection.cases) == 2

        # Create mock evaluation objects
        heat_wave_eval = Mock()
        heat_wave_eval.event_type = ["heat_wave"]
        heat_wave_eval.metric = Mock()
        heat_wave_eval.target = Mock()
        heat_wave_eval.forecast = Mock()

        multi_event_eval = Mock()
        multi_event_eval.event_type = ["heat_wave", "drought", "storm"]
        multi_event_eval.metric = Mock()
        multi_event_eval.target = Mock()
        multi_event_eval.forecast = Mock()

        evaluation_objects = [heat_wave_eval, multi_event_eval]

        # Build case operators
        operators = cases.build_case_operators(cases_dict, evaluation_objects)

        # Should create 3 operators:
        # - Heat wave case with heat_wave_eval
        # - Heat wave case with multi_event_eval
        # - Drought case with multi_event_eval
        assert len(operators) == 3

        # Verify the operators contain the correct cases
        heat_wave_operators = [
            op for op in operators if op.case_metadata.case_id_number == 100
        ]
        drought_operators = [
            op for op in operators if op.case_metadata.case_id_number == 101
        ]

        assert len(heat_wave_operators) == 2  # Both eval objects match heat_wave
        assert len(drought_operators) == 1  # Only multi_event_eval matches drought

        # Verify case metadata is preserved correctly
        for operator in operators:
            if operator.case_metadata.case_id_number == 100:
                assert operator.case_metadata.title == "California Heat Wave 2021"
                assert operator.case_metadata.event_type == "heat_wave"
                assert isinstance(operator.case_metadata.location, CenteredRegion)
            elif operator.case_metadata.case_id_number == 101:
                assert operator.case_metadata.title == "Texas Drought 2021"
                assert operator.case_metadata.event_type == "drought"
                assert isinstance(operator.case_metadata.location, BoundingBoxRegion)
