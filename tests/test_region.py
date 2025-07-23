"""Tests for the region-related functionality in utils.py."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import utils


class TestRegionClasses:
    """Test the Region base class and its subclasses."""

    def test_region_base_class(self):
        """Test that Region is a base class that can be instantiated."""
        # Region is not actually abstract, it's just a base class
        region = utils.Region()
        assert isinstance(region, utils.Region)

    def test_centered_region_creation(self):
        """Test CenteredRegion creation with valid parameters."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0
        assert isinstance(region, utils.Region)

    def test_centered_region_with_tuple_bounding_box(self):
        """Test CenteredRegion creation with tuple bounding box."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_bounding_box_region_creation(self):
        """Test BoundingBoxRegion creation with valid parameters."""
        region = utils.BoundingBoxRegion(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0
        assert isinstance(region, utils.Region)

    def test_shapefile_region_creation(self):
        """Test ShapefileRegion creation with valid path."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = utils.ShapefileRegion(shapefile_path="/path/to/shapefile.shp")
            assert region.shapefile_path == Path("/path/to/shapefile.shp")
            assert isinstance(region, utils.Region)
            mock_read.assert_called_once_with(Path("/path/to/shapefile.shp"))

    def test_shapefile_region_with_string_path(self):
        """Test ShapefileRegion creation with string path."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = utils.ShapefileRegion(shapefile_path="shapefile.shp")
            assert region.shapefile_path == Path("shapefile.shp")

    def test_shapefile_region_read_error(self):
        """Test ShapefileRegion creation with invalid shapefile."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.side_effect = Exception("File not found")
            with pytest.raises(ValueError, match="Error reading shapefile"):
                utils.ShapefileRegion(shapefile_path="/invalid/path.shp")


class TestCreateRegion:
    """Test the create_region factory function."""

    def test_create_centered_region(self):
        """Test creating a CenteredRegion via factory function."""
        region = utils.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert isinstance(region, utils.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_create_centered_region_with_tuple(self):
        """Test creating a CenteredRegion with tuple bounding box."""
        region = utils.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        assert isinstance(region, utils.CenteredRegion)
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_create_bounding_box_region(self):
        """Test creating a BoundingBoxRegion via factory function."""
        region = utils.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert isinstance(region, utils.BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0

    def test_create_shapefile_region(self):
        """Test creating a ShapefileRegion via factory function."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = utils.create_region(shapefile_path="/path/to/shapefile.shp")
            assert isinstance(region, utils.ShapefileRegion)
            assert region.shapefile_path == Path("/path/to/shapefile.shp")

    def test_create_region_invalid_parameters(self):
        """Test create_region with invalid parameter combinations."""
        # Missing required parameters for CenteredRegion
        with pytest.raises(ValueError, match="Invalid parameters"):
            utils.create_region(latitude=45.0, longitude=-120.0)
            # Missing bounding_box_degrees

        # Missing required parameters for BoundingBoxRegion
        with pytest.raises(ValueError, match="Invalid parameters"):
            utils.create_region(
                latitude_min=40.0, latitude_max=50.0, longitude_min=-125.0
            )
            # Missing longitude_max

        # Mixed parameters that don't form a valid region
        with pytest.raises(ValueError, match="Invalid parameters"):
            utils.create_region(latitude=45.0, longitude_min=-125.0)

    def test_create_region_priority_order(self):
        """Test that shapefile_path takes priority over other parameters."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = utils.create_region(
                latitude=45.0,
                longitude=-120.0,
                bounding_box_degrees=10.0,
                shapefile_path="/path/to/shapefile.shp",
            )
            # Should create ShapefileRegion, not CenteredRegion
            assert isinstance(region, utils.ShapefileRegion)


class TestMapToCreateRegion:
    """Test the map_to_create_region function."""

    def test_map_to_create_region_centered(self):
        """Test mapping dictionary to CenteredRegion."""
        kwargs = {
            "latitude": 45.0,
            "longitude": -120.0,
            "bounding_box_degrees": 10.0,
        }
        region = utils.map_to_create_region(kwargs)
        assert isinstance(region, utils.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_map_to_create_region_bounding_box(self):
        """Test mapping dictionary to BoundingBoxRegion."""
        kwargs = {
            "latitude_min": 40.0,
            "latitude_max": 50.0,
            "longitude_min": -125.0,
            "longitude_max": -115.0,
        }
        region = utils.map_to_create_region(kwargs)
        assert isinstance(region, utils.BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0

    def test_map_to_create_region_shapefile(self):
        """Test mapping dictionary to ShapefileRegion."""
        with patch("extremeweatherbench.utils.gpd.read_file") as mock_read:
            mock_read.return_value = Mock()
            kwargs = {"shapefile_path": "/path/to/shapefile.shp"}
            region = utils.map_to_create_region(kwargs)
            assert isinstance(region, utils.ShapefileRegion)
            assert region.shapefile_path == Path("/path/to/shapefile.shp")


class TestClipDatasetToBoundingBoxDegrees:
    """Test the clip_dataset_to_bounding_box_degrees function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        lats = np.linspace(-90, 90, 181)
        lons = np.linspace(0, 359, 360)
        data = np.random.random((len(lats), len(lons)))
        return xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

    def test_clip_dataset_centered_region_single_box(self, sample_dataset):
        """Test clipping with single bounding box value."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Check that the clipped dataset is smaller
        assert len(clipped.latitude) < len(sample_dataset.latitude)
        assert len(clipped.longitude) < len(sample_dataset.longitude)

        # Check that the bounds are correct (approximately)
        assert clipped.latitude.min() >= 40.0  # 45 - 10/2
        assert clipped.latitude.max() <= 50.0  # 45 + 10/2

    def test_clip_dataset_centered_region_tuple_box(self, sample_dataset):
        """Test clipping with tuple bounding box values."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Check that the clipped dataset is smaller
        assert len(clipped.latitude) < len(sample_dataset.latitude)
        assert len(clipped.longitude) < len(sample_dataset.longitude)

    def test_clip_dataset_negative_longitude(self, sample_dataset):
        """Test clipping with negative longitude (should convert to 0-360)."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Should handle negative longitude conversion
        assert len(clipped.longitude) > 0

    def test_clip_dataset_crossing_dateline(self, sample_dataset):
        """Test clipping when the bounding box crosses the dateline."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=175.0, bounding_box_degrees=20.0
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Should handle dateline crossing correctly
        assert len(clipped.longitude) > 0

    def test_clip_dataset_small_region(self, sample_dataset):
        """Test clipping with a very small region."""
        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=1.0
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Should still return a valid dataset
        assert len(clipped.latitude) > 0
        assert len(clipped.longitude) > 0

    def test_clip_dataset_polar_region(self, sample_dataset):
        """Test clipping in polar regions."""
        region = utils.CenteredRegion(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        clipped = utils.clip_dataset_to_bounding_box_degrees(sample_dataset, region)

        # Should handle high latitude regions
        assert len(clipped.latitude) > 0
        assert len(clipped.longitude) > 0


class TestLongitudeConversion:
    """Test longitude conversion utilities."""

    @pytest.mark.parametrize(
        "input_lon,expected",
        [
            (0, 0),
            (180, 180),
            (360, 0),
            (-179, 181),
            (-360, 0),
            (540, 180),
            (359, 359),
        ],
    )
    def test_convert_longitude_to_360(self, input_lon, expected):
        """Test longitude conversion to 0-360 range."""
        result = utils.convert_longitude_to_360(input_lon)
        assert result == expected

    def test_convert_longitude_to_180_dataset(self):
        """Test converting dataset longitude to -180 to 180 range."""
        # Create dataset with 0-360 longitude
        lons = np.linspace(0, 359, 360)
        lats = np.linspace(-90, 90, 181)
        data = np.random.random((len(lats), len(lons)))
        dataset = xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

        converted = utils.convert_longitude_to_180(dataset)

        # Check that longitude is now in -180 to 180 range
        assert converted.longitude.min() >= -180
        assert converted.longitude.max() < 180
        assert len(converted.longitude) == len(dataset.longitude)

    def test_convert_longitude_to_180_custom_name(self):
        """Test converting dataset with custom longitude name."""
        lons = np.linspace(0, 359, 360)
        lats = np.linspace(-90, 90, 181)
        data = np.random.random((len(lats), len(lons)))
        dataset = xr.Dataset(
            {"temperature": (["latitude", "lon"], data)},
            coords={"latitude": lats, "lon": lons},
        )

        converted = utils.convert_longitude_to_180(dataset, longitude_name="lon")

        # Check that longitude is now in -180 to 180 range
        assert converted.lon.min() >= -180
        assert converted.lon.max() < 180


class TestRegionIntegration:
    """Integration tests for region functionality."""

    def test_region_with_dacite_integration(self):
        """Test that regions work correctly with dacite.from_dict."""

        # Test data similar to what would come from YAML
        region_data = {
            "latitude": 45.0,
            "longitude": -120.0,
            "bounding_box_degrees": 10.0,
        }

        # Test that map_to_create_region works correctly
        region = utils.map_to_create_region(region_data)

        assert isinstance(region, utils.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_region_with_individual_case_integration(self):
        """Test that regions work correctly with IndividualCase."""
        from extremeweatherbench import case

        region = utils.CenteredRegion(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        # Create a sample dataset
        lats = np.linspace(-90, 90, 181)
        lons = np.linspace(0, 359, 360)
        data = np.random.random((len(lats), len(lons)))
        dataset = xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

        # Create IndividualCase with the region
        individual_case = case.IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=region,
            event_type="test",
            data_vars=["temperature"],
        )

        # Test that subset_region works
        subset = individual_case.subset_region(dataset)
        assert len(subset.latitude) < len(dataset.latitude)
        assert len(subset.longitude) < len(dataset.longitude)
