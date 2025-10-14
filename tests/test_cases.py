"""Tests for the cases module."""

import datetime
from unittest import mock

import pytest
import yaml

from extremeweatherbench import cases, regions


class TestIndividualCase:
    """Test the IndividualCase dataclass."""

    def test_individual_case_creation(self):
        """Test IndividualCase creation with valid parameters."""
        region = regions.CenteredRegion.create_region(
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
        centered_region = regions.CenteredRegion.create_region(
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

        assert isinstance(case1.location, regions.CenteredRegion)

        # Test with BoundingBoxRegion
        bbox_region = regions.BoundingBoxRegion.create_region(
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

        assert isinstance(case2.location, regions.BoundingBoxRegion)

    def test_individual_case_validation(self):
        """Test IndividualCase with various edge cases."""
        region = regions.CenteredRegion.create_region(
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
        region1 = regions.CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )
        region2 = regions.CenteredRegion.create_region(
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
        region = regions.CenteredRegion.create_region(
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
        mock_metric = mock.Mock()
        mock_target = mock.Mock()
        mock_forecast = mock.Mock()

        operator = cases.CaseOperator(
            case_metadata=case,
            metric_list=[mock_metric],
            target=mock_target,
            forecast=mock_forecast,
        )

        assert operator.case_metadata == case
        assert operator.metric_list == [mock_metric]
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
        assert isinstance(case1.location, regions.CenteredRegion)

        # Check second case
        case2 = collection.cases[1]
        assert case2.case_id_number == 2
        assert case2.title == "Test Case 2"
        assert case2.event_type == "drought"
        assert isinstance(case2.location, regions.BoundingBoxRegion)

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
        mock_eval_obj1 = mock.Mock()
        mock_eval_obj1.event_type = ["heat_wave"]
        mock_eval_obj1.metric_list = mock.Mock()
        mock_eval_obj1.target = mock.Mock()
        mock_eval_obj1.forecast = mock.Mock()

        mock_eval_obj2 = mock.Mock()
        mock_eval_obj2.event_type = ["drought", "heat_wave"]
        mock_eval_obj2.metric_list = mock.Mock()
        mock_eval_obj2.target = mock.Mock()
        mock_eval_obj2.forecast = mock.Mock()

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
            assert hasattr(operator, "metric_list")
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
        mock_eval_obj = mock.Mock()
        mock_eval_obj.event_type = ["heat_wave", "drought"]
        mock_eval_obj.metric_list = mock.Mock()
        mock_eval_obj.target = mock.Mock()
        mock_eval_obj.forecast = mock.Mock()

        evaluation_objects = [mock_eval_obj]

        operators = cases.build_case_operators(cases_dict, evaluation_objects)

        # Should create no operators since no event types match
        assert len(operators) == 0

    def test_build_case_operators_invalid_cases_dict(self):
        """Test build_case_operators with invalid cases_dict."""
        import dacite

        invalid_cases_dict = {"cases": "not_a_list"}
        mock_eval_obj = mock.Mock()

        with pytest.raises(
            dacite.exceptions.WrongTypeError,
            match="wrong value type",
        ):
            cases.build_case_operators(invalid_cases_dict, [mock_eval_obj])

    def test_build_case_operators_with_individual_case_collection_passthrough(self):
        """Test build_case_operators with IndividualCaseCollection passthrough."""
        # Create individual cases directly
        region1 = regions.CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )
        region2 = regions.CenteredRegion.create_region(
            latitude=35.0, longitude=-105.0, bounding_box_degrees=8.0
        )

        case1 = cases.IndividualCase(
            case_id_number=1,
            title="Heat Wave Case",
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 15),
            location=region1,
            event_type="heat_wave",
        )

        case2 = cases.IndividualCase(
            case_id_number=2,
            title="Drought Case",
            start_date=datetime.datetime(2021, 6, 1),
            end_date=datetime.datetime(2021, 6, 15),
            location=region2,
            event_type="drought",
        )

        # Create IndividualCaseCollection directly
        case_collection = cases.IndividualCaseCollection(cases=[case1, case2])

        # Create mock evaluation objects
        mock_eval_obj1 = mock.Mock()
        mock_eval_obj1.event_type = ["heat_wave"]
        mock_eval_obj1.metric_list = mock.Mock()
        mock_eval_obj1.target = mock.Mock()
        mock_eval_obj1.forecast = mock.Mock()

        mock_eval_obj2 = mock.Mock()
        mock_eval_obj2.event_type = ["drought", "heat_wave"]
        mock_eval_obj2.metric_list = mock.Mock()
        mock_eval_obj2.target = mock.Mock()
        mock_eval_obj2.forecast = mock.Mock()

        evaluation_objects = [mock_eval_obj1, mock_eval_obj2]

        # Test passthrough functionality - pass IndividualCaseCollection directly
        operators = cases.build_case_operators(case_collection, evaluation_objects)

        # Should create 3 operators:
        # - Heat wave case with eval_obj1 (matches heat_wave)
        # - Heat wave case with eval_obj2 (matches heat_wave)
        # - Drought case with eval_obj2 (matches drought)
        assert len(operators) == 3

        # Check that all operators are CaseOperator instances
        for operator in operators:
            assert isinstance(operator, cases.CaseOperator)
            assert hasattr(operator, "case_metadata")
            assert hasattr(operator, "metric_list")
            assert hasattr(operator, "target")
            assert hasattr(operator, "forecast")

        # Verify the correct cases were matched
        heat_wave_operators = [
            op for op in operators if op.case_metadata.case_id_number == 1
        ]
        drought_operators = [
            op for op in operators if op.case_metadata.case_id_number == 2
        ]

        assert len(heat_wave_operators) == 2  # Both eval objects match heat_wave
        assert len(drought_operators) == 1  # Only eval_obj2 matches drought

    def test_build_case_operators_dict_vs_collection_equivalence(self):
        """Test dict and IndividualCaseCollection inputs produce same results."""
        # Define cases as dictionary (traditional way)
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Test Heat Wave",
                    "start_date": datetime.datetime(2021, 7, 1),
                    "end_date": datetime.datetime(2021, 7, 10),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 42.0,
                            "longitude": -95.0,
                            "bounding_box_degrees": 6.0,
                        },
                    },
                    "event_type": "heat_wave",
                },
                {
                    "case_id_number": 2,
                    "title": "Test Storm",
                    "start_date": datetime.datetime(2021, 8, 1),
                    "end_date": datetime.datetime(2021, 8, 5),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 30.0,
                            "latitude_max": 40.0,
                            "longitude_min": -100.0,
                            "longitude_max": -85.0,
                        },
                    },
                    "event_type": "storm",
                },
            ]
        }

        # Create equivalent IndividualCaseCollection
        region1 = regions.CenteredRegion.create_region(
            latitude=42.0, longitude=-95.0, bounding_box_degrees=6.0
        )
        region2 = regions.BoundingBoxRegion.create_region(
            latitude_min=30.0,
            latitude_max=40.0,
            longitude_min=-100.0,
            longitude_max=-85.0,
        )

        case1 = cases.IndividualCase(
            case_id_number=1,
            title="Test Heat Wave",
            start_date=datetime.datetime(2021, 7, 1),
            end_date=datetime.datetime(2021, 7, 10),
            location=region1,
            event_type="heat_wave",
        )

        case2 = cases.IndividualCase(
            case_id_number=2,
            title="Test Storm",
            start_date=datetime.datetime(2021, 8, 1),
            end_date=datetime.datetime(2021, 8, 5),
            location=region2,
            event_type="storm",
        )

        case_collection = cases.IndividualCaseCollection(cases=[case1, case2])

        # Create evaluation objects
        heat_wave_eval = mock.Mock()
        heat_wave_eval.event_type = ["heat_wave"]
        heat_wave_eval.metric_list = mock.Mock()
        heat_wave_eval.target = mock.Mock()
        heat_wave_eval.forecast = mock.Mock()

        storm_eval = mock.Mock()
        storm_eval.event_type = ["storm"]
        storm_eval.metric_list = mock.Mock()
        storm_eval.target = mock.Mock()
        storm_eval.forecast = mock.Mock()

        evaluation_objects = [heat_wave_eval, storm_eval]

        # Build operators using dictionary
        operators_from_dict = cases.build_case_operators(cases_dict, evaluation_objects)

        # Build operators using IndividualCaseCollection (passthrough)
        operators_from_collection = cases.build_case_operators(
            case_collection, evaluation_objects
        )

        # Results should be equivalent
        assert len(operators_from_dict) == len(operators_from_collection)
        assert len(operators_from_dict) == 2  # One for each event type

        # Sort operators by case_id_number for comparison
        dict_ops_sorted = sorted(
            operators_from_dict, key=lambda x: x.case_metadata.case_id_number
        )
        collection_ops_sorted = sorted(
            operators_from_collection, key=lambda x: x.case_metadata.case_id_number
        )

        # Compare each operator pair
        for dict_op, collection_op in zip(dict_ops_sorted, collection_ops_sorted):
            # Case metadata should be equivalent
            assert (
                dict_op.case_metadata.case_id_number
                == collection_op.case_metadata.case_id_number
            )
            assert dict_op.case_metadata.title == collection_op.case_metadata.title
            assert (
                dict_op.case_metadata.start_date
                == collection_op.case_metadata.start_date
            )
            assert (
                dict_op.case_metadata.end_date == collection_op.case_metadata.end_date
            )
            assert (
                dict_op.case_metadata.event_type
                == collection_op.case_metadata.event_type
            )

            # Location types should be the same
            assert isinstance(
                dict_op.case_metadata.location,
                type(collection_op.case_metadata.location),
            )

            # Other attributes should reference the same mock objects
            assert dict_op.metric_list is collection_op.metric_list
            assert dict_op.target is collection_op.target
            assert dict_op.forecast is collection_op.forecast

    def test_build_case_operators_empty_individual_case_collection(self):
        """Test build_case_operators with empty IndividualCaseCollection."""
        empty_collection = cases.IndividualCaseCollection(cases=[])

        mock_eval_obj = mock.Mock()
        mock_eval_obj.event_type = ["heat_wave"]
        mock_eval_obj.metric_list = mock.Mock()
        mock_eval_obj.target = mock.Mock()
        mock_eval_obj.forecast = mock.Mock()

        operators = cases.build_case_operators(empty_collection, [mock_eval_obj])

        # Should create no operators from empty collection
        assert len(operators) == 0

    def test_build_case_operators_collection_with_no_matching_events(self):
        """Test IndividualCaseCollection passthrough with no matching event types."""
        region = regions.CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )

        flood_case = cases.IndividualCase(
            case_id_number=1,
            title="Flood Case",
            start_date=datetime.datetime(2021, 9, 1),
            end_date=datetime.datetime(2021, 9, 10),
            location=region,
            event_type="flood",
        )

        case_collection = cases.IndividualCaseCollection(cases=[flood_case])

        # Evaluation object that doesn't match flood
        mock_eval_obj = mock.Mock()
        mock_eval_obj.event_type = ["heat_wave", "drought"]
        mock_eval_obj.metric_list = mock.Mock()
        mock_eval_obj.target = mock.Mock()
        mock_eval_obj.forecast = mock.Mock()

        operators = cases.build_case_operators(case_collection, [mock_eval_obj])

        # Should create no operators since event types don't match
        assert len(operators) == 0


class TestLoadIndividualCasesFromYaml:
    """Test the load_individual_cases_from_yaml function."""

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading individual cases from YAML file."""
        yaml_content = {
            "cases": [
                {
                    "case_id_number": 42,
                    "title": "YAML Test Case",
                    "start_date": datetime.datetime(2022, 5, 1),
                    "end_date": datetime.datetime(2022, 5, 15),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 38.0,
                            "longitude": -95.0,
                            "bounding_box_degrees": 7.5,
                        },
                    },
                    "event_type": "severe_storm",
                }
            ]
        }

        yaml_file = tmp_path / "test_cases.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        collection = cases.load_individual_cases_from_yaml(yaml_file)

        assert isinstance(collection, cases.IndividualCaseCollection)
        assert len(collection.cases) == 1

        case = collection.cases[0]
        assert case.case_id_number == 42
        assert case.title == "YAML Test Case"
        assert case.event_type == "severe_storm"
        assert isinstance(case.location, regions.CenteredRegion)

    def test_load_from_yaml_string_path(self, tmp_path):
        """Test loading with string path instead of Path object."""
        yaml_content = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "String Path Test",
                    "start_date": datetime.datetime(2023, 1, 1),
                    "end_date": datetime.datetime(2023, 1, 10),
                    "location": {
                        "type": "bounded_region",
                        "parameters": {
                            "latitude_min": 30.0,
                            "latitude_max": 40.0,
                            "longitude_min": -110.0,
                            "longitude_max": -100.0,
                        },
                    },
                    "event_type": "wildfire",
                }
            ]
        }

        yaml_file = tmp_path / "test_string_path.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Pass as string, not Path object
        collection = cases.load_individual_cases_from_yaml(str(yaml_file))

        assert len(collection.cases) == 1
        assert collection.cases[0].title == "String Path Test"
        assert isinstance(collection.cases[0].location, regions.BoundingBoxRegion)

    def test_load_from_yaml_malformed_file(self, tmp_path):
        """Test error handling with malformed YAML."""
        malformed_yaml = "cases:\n  - invalid: [unclosed"

        yaml_file = tmp_path / "malformed.yaml"
        with open(yaml_file, "w") as f:
            f.write(malformed_yaml)

        with pytest.raises(yaml.YAMLError):
            cases.load_individual_cases_from_yaml(yaml_file)

    def test_load_from_yaml_missing_file(self):
        """Test error handling with non-existent file."""
        with pytest.raises(FileNotFoundError):
            cases.load_individual_cases_from_yaml("nonexistent.yaml")


class TestLoadEventsYaml:
    """Test the load_ewb_events_yaml_into_case_collection function."""

    @mock.patch("importlib.resources")
    def test_load_ewb_events_yaml_into_case_collection_success(self, mock_resources):
        """Test successful loading of events YAML."""
        # Mock the resource access
        mock_files = mock.Mock()
        mock_resources.files.return_value = mock_files
        mock_files.joinpath.return_value = "/mock/path/events.yaml"

        # Mock the file content
        mock_yaml_content = {
            "cases": [
                {
                    "case_id_number": 999,
                    "title": "Mock Event",
                    "start_date": datetime.datetime(2021, 1, 1),
                    "end_date": datetime.datetime(2021, 1, 5),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 0.0,
                            "longitude": 0.0,
                            "bounding_box_degrees": 1.0,
                        },
                    },
                    "event_type": "test_event",
                }
            ]
        }

        with mock.patch("importlib.resources.as_file") as mock_as_file:
            with mock.patch(
                "extremeweatherbench.cases.read_incoming_yaml"
            ) as mock_read:
                mock_read.return_value = mock_yaml_content
                mock_as_file.return_value.__enter__.return_value = "/mock/file"

                result = cases.load_ewb_events_yaml_into_case_collection()

                # Should return an IndividualCaseCollection, not the raw dict
                assert isinstance(result, cases.IndividualCaseCollection)
                assert len(result.cases) == 1

                # Verify the case was loaded correctly
                case = result.cases[0]
                assert case.case_id_number == 999
                assert case.title == "Mock Event"
                assert case.event_type == "test_event"

                mock_read.assert_called_once()


class TestReadIncomingYaml:
    """Test the read_incoming_yaml function."""

    def test_read_incoming_yaml_success(self, tmp_path):
        """Test successful reading of YAML file."""
        test_data = {
            "test_key": "test_value",
            "cases": [{"case_id_number": 1, "title": "Test"}],
        }

        yaml_file = tmp_path / "test_input.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        result = cases.read_incoming_yaml(yaml_file)

        assert result == test_data
        assert isinstance(result, dict)
        assert "test_key" in result
        assert result["test_key"] == "test_value"

    def test_read_incoming_yaml_with_string_path(self, tmp_path):
        """Test reading YAML with string path."""
        test_data = {"simple": "data"}

        yaml_file = tmp_path / "string_test.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        result = cases.read_incoming_yaml(str(yaml_file))
        assert result == test_data

    def test_read_incoming_yaml_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            cases.read_incoming_yaml("does_not_exist.yaml")

    def test_read_incoming_yaml_empty_file(self, tmp_path):
        """Test handling of empty YAML file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.touch()

        result = cases.read_incoming_yaml(empty_file)
        assert result is None


class TestCasesEdgeCases:
    """Test edge cases and error conditions."""

    def test_individual_case_date_validation(self):
        """Test cases with edge case dates."""
        region = regions.CenteredRegion.create_region(
            latitude=0.0, longitude=0.0, bounding_box_degrees=1.0
        )

        # Test with same start and end date
        case = cases.IndividualCase(
            case_id_number=1,
            title="Same Date Case",
            start_date=datetime.datetime(2021, 1, 1, 12, 0),
            end_date=datetime.datetime(2021, 1, 1, 12, 0),
            location=region,
            event_type="instantaneous",
        )

        assert case.start_date == case.end_date

    def test_case_operator_with_list_metrics(self):
        """Test CaseOperator with list of metrics."""
        region = regions.CenteredRegion.create_region(
            latitude=40.0, longitude=-100.0, bounding_box_degrees=5.0
        )

        case = cases.IndividualCase(
            case_id_number=1,
            title="Multi-Metric Case",
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 15),
            location=region,
            event_type="multi_metric_event",
        )

        # Test with list of metrics
        metric_list = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_target = mock.Mock()
        mock_forecast = mock.Mock()

        operator = cases.CaseOperator(
            case_metadata=case,
            metric_list=metric_list,
            target=mock_target,
            forecast=mock_forecast,
        )

        assert operator.metric_list == metric_list
        assert len(operator.metric_list) == 3

    def test_build_case_operators_empty_evaluation_objects(self):
        """Test building case operators with empty evaluation objects."""
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Lonely Case",
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
                    "event_type": "lonely_event",
                }
            ]
        }

        operators = cases.build_case_operators(cases_dict, [])
        assert len(operators) == 0

    def test_build_case_operators_partial_event_matches(self):
        """Test case operators with complex event type matching."""
        cases_dict = {
            "cases": [
                {
                    "case_id_number": 1,
                    "title": "Heat Wave",
                    "start_date": datetime.datetime(2021, 6, 1),
                    "end_date": datetime.datetime(2021, 6, 15),
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
                    "title": "Drought",
                    "start_date": datetime.datetime(2021, 7, 1),
                    "end_date": datetime.datetime(2021, 7, 31),
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
                {
                    "case_id_number": 3,
                    "title": "Flood",
                    "start_date": datetime.datetime(2021, 8, 1),
                    "end_date": datetime.datetime(2021, 8, 10),
                    "location": {
                        "type": "centered_region",
                        "parameters": {
                            "latitude": 45.0,
                            "longitude": -90.0,
                            "bounding_box_degrees": 3.0,
                        },
                    },
                    "event_type": "flood",
                },
            ]
        }

        # Evaluation object that only matches heat_wave and drought
        eval_obj = mock.Mock()
        eval_obj.event_type = ["heat_wave", "drought"]
        eval_obj.metric = mock.Mock()
        eval_obj.target = mock.Mock()
        eval_obj.forecast = mock.Mock()

        operators = cases.build_case_operators(cases_dict, [eval_obj])

        # Should create 2 operators (heat_wave and drought, but not flood)
        assert len(operators) == 2

        created_case_ids = [op.case_metadata.case_id_number for op in operators]
        assert 1 in created_case_ids  # heat_wave case
        assert 2 in created_case_ids  # drought case
        assert 3 not in created_case_ids  # flood case should be excluded


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
        heat_wave_eval = mock.Mock()
        heat_wave_eval.event_type = ["heat_wave"]
        heat_wave_eval.metric_list = mock.Mock()
        heat_wave_eval.target = mock.Mock()
        heat_wave_eval.forecast = mock.Mock()

        multi_event_eval = mock.Mock()
        multi_event_eval.event_type = ["heat_wave", "drought", "storm"]
        multi_event_eval.metric_list = mock.Mock()
        multi_event_eval.target = mock.Mock()
        multi_event_eval.forecast = mock.Mock()

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
                assert isinstance(
                    operator.case_metadata.location, regions.CenteredRegion
                )
            elif operator.case_metadata.case_id_number == 101:
                assert operator.case_metadata.title == "Texas Drought 2021"
                assert operator.case_metadata.event_type == "drought"
                assert isinstance(
                    operator.case_metadata.location, regions.BoundingBoxRegion
                )
