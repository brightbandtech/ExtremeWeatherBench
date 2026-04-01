#!/usr/bin/env python3
"""Validation script for events.yaml file.

This script validates that the events.yaml file follows the required format:
1. All start_date and end_date are in format YYYY-MM-DD HH:MM:SS
2. All spacing is consistent throughout the file
3. case_id_numbers are monotonically increasing (not necessarily by 1)
4. Locations have valid types and required parameters
5. Titles are strings
6. Comments are preserved
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def validate_datetime_format(
    date_value: Any, field_name: str, case_num: int
) -> List[str]:
    """Validate that a date value is in YYYY-MM-DD HH:MM:SS format."""
    errors: list[str] = []

    # If it's already a datetime object, check if it was parsed correctly
    if isinstance(date_value, datetime):
        # This means YAML successfully parsed it as a datetime, which is good
        return errors

    # If it's a string, this indicates the date is quoted or in wrong format
    if isinstance(date_value, str):
        # Check if it looks like a quoted date that should be unquoted
        if date_value.count("-") == 2 and len(date_value) in [10, 19]:
            errors.append(
                f"Case {case_num}: {field_name} '{date_value}' appears to be "
                f"quoted in YAML. Dates should be unquoted to be parsed as datetime "
                "objects. "
            )
        else:
            try:
                # Try to parse the datetime
                parsed = datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S")
                # Ensure the string format matches exactly
                if parsed.strftime("%Y-%m-%d %H:%M:%S") != date_value:
                    errors.append(
                        f"Case {case_num}: {field_name} '{date_value}' format mismatch"
                    )
            except ValueError:
                errors.append(
                    f"Case {case_num}: {field_name} '{date_value}' is not in "
                    f"YYYY-MM-DD HH:MM:SS format"
                )
    else:
        errors.append(
            f"Case {case_num}: {field_name} must be a datetime object, "
            f"got {type(date_value)}"
        )

    return errors


def validate_location(location: Dict[str, Any], case_num: int) -> List[str]:
    """Validate location structure."""
    errors = []

    if not isinstance(location, dict):
        errors.append(f"Case {case_num}: location must be a dictionary")
        return errors

    if "type" not in location:
        errors.append(f"Case {case_num}: location missing 'type' field")
        return errors

    location_type = location["type"]
    if "parameters" not in location:
        errors.append(f"Case {case_num}: location missing 'parameters' field")
        return errors

    params = location["parameters"]
    if not isinstance(params, dict):
        errors.append(f"Case {case_num}: location parameters must be a dictionary")
        return errors

    if location_type == "centered_region":
        required_fields = ["latitude", "longitude", "bounding_box_degrees"]
        for field in required_fields:
            if field not in params:
                errors.append(
                    f"Case {case_num}: centered_region missing '{field}' parameter"
                )
            elif not isinstance(params[field], (int, float)):
                errors.append(
                    f"Case {case_num}: centered_region '{field}' must be a number"
                )

    elif location_type == "bounded_region":
        required_fields = [
            "latitude_min",
            "latitude_max",
            "longitude_min",
            "longitude_max",
        ]
        for field in required_fields:
            if field not in params:
                errors.append(
                    f"Case {case_num}: bounded_region missing '{field}' parameter"
                )
            elif not isinstance(params[field], (int, float)):
                errors.append(
                    f"Case {case_num}: bounded_region '{field}' must be a number"
                )

    elif location_type == "shapefile_region":
        if "shapefile_path" not in params:
            errors.append(
                f"Case {case_num}: shapefile_region missing 'shapefile_path' parameter"
            )
        elif not isinstance(params["shapefile_path"], str):
            errors.append(
                f"Case {case_num}: shapefile_region 'shapefile_path' must be a string"
            )
    else:
        errors.append(
            f"Case {case_num}: invalid location type '{location_type}'. "
            f"Must be 'centered_region', 'bounded_region', or 'shapefile_region'"
        )

    return errors


def validate_yaml_spacing(file_path: Path) -> List[str]:
    """Validate that YAML spacing is consistent throughout the file."""
    errors = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Check for consistent indentation
    indent_levels = {}
    for line_num, line in enumerate(lines, 1):
        if line.strip() == "" or line.strip().startswith("#"):
            continue  # Skip empty lines and comments

        # Count leading spaces
        leading_spaces = len(line) - len(line.lstrip())

        # Determine indent level based on content
        content = line.strip()
        if content.startswith("- case_id_number:"):
            expected_indent = 0
            indent_levels[0] = leading_spaces
        elif (
            content.startswith("title:")
            or content.startswith("start_date:")
            or content.startswith("end_date:")
            or content.startswith("location:")
            or content.startswith("event_type:")
        ):
            expected_indent = 1
            if 0 in indent_levels:
                if 1 not in indent_levels:
                    indent_levels[1] = leading_spaces
        elif content.startswith("type:") or content.startswith("parameters:"):
            expected_indent = 2
            if 1 in indent_levels:
                if 2 not in indent_levels:
                    indent_levels[2] = leading_spaces
        elif (
            content.startswith("latitude")
            or content.startswith("longitude")
            or content.startswith("bounding_box")
            or content.startswith("shapefile_path")
        ):
            expected_indent = 3
            if 2 in indent_levels:
                if 3 not in indent_levels:
                    indent_levels[3] = leading_spaces
        else:
            continue  # Skip validation for other lines

        # Check if indentation matches expected
        if expected_indent in indent_levels:
            expected_spaces = indent_levels[expected_indent]
            if leading_spaces != expected_spaces:
                errors.append(
                    f"Line {line_num}: inconsistent indentation. "
                    f"Expected {expected_spaces} spaces, got {leading_spaces}"
                )

    return errors


def validate_events_yaml(file_path: Path) -> List[str]:
    """Main validation function for events.yaml."""
    errors = []

    if not file_path.exists():
        return [f"File {file_path} does not exist"]

    # Validate YAML spacing first
    spacing_errors = validate_yaml_spacing(file_path)
    errors.extend(spacing_errors)

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"Invalid YAML syntax: {e}"]

    if not isinstance(data, dict) and not isinstance(data, list):
        return ["Root element must be a dictionary or a list"]

    # Validate each case
    previous_case_id = None
    for i, case in enumerate(data):
        if not isinstance(case, dict):
            errors.append(f"Case {i + 1}: must be a dictionary: {data} {file_path}")
            continue

        # Check required fields
        required_fields = [
            "case_id_number",
            "title",
            "start_date",
            "end_date",
            "location",
            "event_type",
        ]
        for field in required_fields:
            if field not in case:
                errors.append(f"Case {i + 1}: missing required field '{field}'")

        # Validate case_id_number (must be monotonically increasing)
        if "case_id_number" in case:
            case_id = case["case_id_number"]
            if not isinstance(case_id, int):
                errors.append(f"Case {i + 1}: case_id_number must be an integer")
            elif previous_case_id is not None and case_id <= previous_case_id:
                errors.append(
                    f"Case {i + 1}: case_id_number {case_id} is not greater than "
                    f"previous case_id_number {previous_case_id}. "
                    f"Case IDs must be monotonically increasing."
                )
            previous_case_id = case_id

        # Validate title is a string
        if "title" in case:
            if not isinstance(case["title"], str):
                errors.append(
                    f"Case {case.get('case_id_number', i + 1)}: title must be a string"
                )

        # Validate date formats
        case_num = case.get("case_id_number", i + 1)
        if "start_date" in case:
            errors.extend(
                validate_datetime_format(case["start_date"], "start_date", case_num)
            )

        if "end_date" in case:
            errors.extend(
                validate_datetime_format(case["end_date"], "end_date", case_num)
            )

        # Validate location
        if "location" in case:
            errors.extend(validate_location(case["location"], case_num))

    return errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_events_yaml.py <path_to_events.yaml> [...]")
        sys.exit(1)

    # Validate all provided files
    all_errors = {}
    for file_arg in sys.argv[1:]:
        file_path = Path(file_arg)
        errors = validate_events_yaml(file_path)
        if errors:
            all_errors[file_path] = errors

    # Report results
    if all_errors:
        for file_path, errors in all_errors.items():
            print(f"Validation failed for {file_path}:")
            for error in errors:
                print(f"  - {error}")
        sys.exit(1)
    else:
        for file_arg in sys.argv[1:]:
            print(f"✓ {file_arg} validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
