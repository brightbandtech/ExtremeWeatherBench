"""Tests for the region-related functionality in regions.py."""

from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import Polygon

from extremeweatherbench import utils
from extremeweatherbench.regions import (
    BoundingBoxRegion,
    CenteredRegion,
    Region,
    ShapefileRegion,
    map_to_create_region,
)


class TestRegionClasses:
    """Test the Region base class and its subclasses."""

    def test_region_base_class(self):
        """Test that Region is an abstract base class that cannot be instantiated."""
        # Region is now abstract and cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Region()

    def test_centered_region_creation(self):
        """Test CenteredRegion creation with valid parameters."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0
        assert isinstance(region, Region)

    def test_centered_region_with_tuple_bounding_box(self):
        """Test CenteredRegion creation with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_bounding_box_region_creation(self):
        """Test BoundingBoxRegion creation with valid parameters."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0
        assert isinstance(region, Region)

    def test_shapefile_region_creation(self):
        """Test ShapefileRegion creation with valid path."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )
            assert region.shapefile_path == Path("/path/to/shapefile.shp")
            assert isinstance(region, Region)

            # build_region should be called when geopandas is accessed
            _ = region.geopandas
            mock_read.assert_called_once_with(Path("/path/to/shapefile.shp"))

    def test_shapefile_region_with_string_path(self):
        """Test ShapefileRegion creation with string path."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = ShapefileRegion.create_region(shapefile_path="shapefile.shp")
            assert region.shapefile_path == Path("shapefile.shp")

    def test_shapefile_region_read_error(self):
        """Test ShapefileRegion creation with invalid shapefile."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.side_effect = Exception("File not found")
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            # build_region should be called when geopandas is accessed
            with pytest.raises(ValueError, match="Error reading shapefile"):
                _ = region.geopandas


class TestRegionToGeopandas:
    """Test the to_geopandas() method for all Region subclasses."""

    def test_centered_region_to_geopandas_single_box(self):
        """Test CenteredRegion.geopandas with single bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        gdf = region.geopandas

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Verify the polygon coordinates
        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds (should be 40-50 lat, 235-245 lon after conversion to 0-360)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - 235.0) < 0.001  # min lon (converted to 0-360)
        assert abs(bounds[2] - 245.0) < 0.001  # max lon (converted to 0-360)

    def test_centered_region_to_geopandas_tuple_box(self):
        """Test CenteredRegion.geopandas with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        gdf = region.geopandas

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Check bounds (should be 42.5-47.5 lat, 235-245 lon after conversion to 0-360)
        bounds = gdf.geometry.iloc[0].bounds
        assert abs(bounds[1] - 42.5) < 0.001  # min lat
        assert abs(bounds[3] - 47.5) < 0.001  # max lat
        assert abs(bounds[0] - 235.0) < 0.001  # min lon (converted to 0-360)
        assert abs(bounds[2] - 245.0) < 0.001  # max lon (converted to 0-360)

    def test_bounding_box_region_to_geopandas(self):
        """Test BoundingBoxRegion.geopandas."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        gdf = region.geopandas

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Verify the polygon coordinates
        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds (longitude should be converted to 0-360)
        bounds = polygon.bounds
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - 235.0) < 0.001  # min lon (converted to 0-360)
        assert abs(bounds[2] - 245.0) < 0.001  # max lon (converted to 0-360)

    def test_shapefile_region_to_geopandas(self):
        """Test ShapefileRegion.geopandas."""
        # Create a mock GeoDataFrame
        mock_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            gdf = region.geopandas

            # Should return the same GeoDataFrame that was read
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 1
            assert gdf.geometry.iloc[0] == mock_polygon

    def test_region_to_geopandas_edge_cases(self):
        """Test geopandas with edge cases."""
        # Test CenteredRegion at poles
        polar_region = CenteredRegion.create_region(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        polar_gdf = polar_region.geopandas
        assert isinstance(polar_gdf, gpd.GeoDataFrame)

        # Test CenteredRegion crossing the dateline
        dateline_region = CenteredRegion.create_region(
            latitude=45.0, longitude=175.0, bounding_box_degrees=10.0
        )
        dateline_gdf = dateline_region.geopandas
        assert isinstance(dateline_gdf, gpd.GeoDataFrame)

        # Test with very small bounding box
        small_region = BoundingBoxRegion.create_region(
            latitude_min=44.9,
            latitude_max=45.1,
            longitude_min=-120.1,
            longitude_max=-119.9,
        )
        small_gdf = small_region.geopandas
        assert isinstance(small_gdf, gpd.GeoDataFrame)

    def test_region_to_geopandas_longitude_conversion(self):
        """Test that longitudes are properly converted to 0-360 range."""
        # Test with negative longitudes
        region_neg = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        gdf_neg = region_neg.geopandas
        bounds_neg = gdf_neg.geometry.iloc[0].bounds
        assert bounds_neg[0] >= 0  # min lon should be >= 0
        assert bounds_neg[2] >= 0  # max lon should be >= 0

        # Test with positive longitudes
        region_pos = CenteredRegion.create_region(
            latitude=45.0, longitude=120.0, bounding_box_degrees=10.0
        )
        gdf_pos = region_pos.geopandas
        bounds_pos = gdf_pos.geometry.iloc[0].bounds
        assert bounds_pos[0] >= 0  # min lon should be >= 0
        assert bounds_pos[2] >= 0  # max lon should be >= 0


class TestRegionMask:
    """Test the mask() method for all Region subclasses."""

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

    def test_centered_region_mask(self, sample_dataset):
        """Test CenteredRegion.mask() method."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        masked_dataset = region.mask(sample_dataset)

        # Verify the masked dataset has the same structure but masked values
        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        assert masked_dataset.sizes == sample_dataset.sizes

        # Check that some values are masked (NaN) outside the region
        assert np.any(np.isnan(masked_dataset.temperature.values))

    def test_bounding_box_region_mask(self, sample_dataset):
        """Test BoundingBoxRegion.mask() method."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        masked_dataset = region.mask(sample_dataset)

        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        assert masked_dataset.sizes == sample_dataset.sizes
        assert np.any(np.isnan(masked_dataset.temperature.values))

    def test_shapefile_region_mask(self, sample_dataset):
        """Test ShapefileRegion.mask() method."""
        mock_polygon = Polygon([(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            masked_dataset = region.mask(sample_dataset)

            assert isinstance(masked_dataset, xr.Dataset)
            assert "temperature" in masked_dataset.data_vars
            assert masked_dataset.sizes == sample_dataset.sizes
            assert np.any(np.isnan(masked_dataset.temperature.values))

    def test_region_mask_consistency(self, sample_dataset):
        """Test that different region types produce consistent mask behavior."""
        # Create regions with similar coverage
        centered = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        bbox = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        # Both should produce masked datasets
        centered_masked = centered.mask(sample_dataset)
        bbox_masked = bbox.mask(sample_dataset)

        assert isinstance(centered_masked, xr.Dataset)
        assert isinstance(bbox_masked, xr.Dataset)
        assert "temperature" in centered_masked.data_vars
        assert "temperature" in bbox_masked.data_vars


class TestRegionInheritance:
    """Test that all region types properly inherit from Region."""

    def test_region_inheritance(self):
        """Test that all region types properly inherit from Region."""
        # Test CenteredRegion
        centered = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert isinstance(centered, Region)

        # Test BoundingBoxRegion
        bbox = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert isinstance(bbox, Region)

        # Test ShapefileRegion
        with patch("geopandas.read_file", return_value=Mock()):
            shapefile = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )
            assert isinstance(shapefile, Region)

    def test_region_methods_consistency(self):
        """Test that all region types have consistent method behavior."""
        # Create regions with similar coverage
        centered = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        bbox = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        # Test that both can create GeoDataFrames
        centered_gdf = centered.geopandas
        bbox_gdf = bbox.geopandas

        assert isinstance(centered_gdf, gpd.GeoDataFrame)
        assert isinstance(bbox_gdf, gpd.GeoDataFrame)
        assert len(centered_gdf) == 1
        assert len(bbox_gdf) == 1


class TestCreateRegion:
    """Test the create_region factory function."""

    def test_create_centered_region(self):
        """Test creating a CenteredRegion via factory function."""
        region = map_to_create_region(
            {
                "type": "centered_region",
                "parameters": {
                    "latitude": 45.0,
                    "longitude": -120.0,
                    "bounding_box_degrees": 10.0,
                },
            }
        )
        assert isinstance(region, CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_create_centered_region_with_tuple(self):
        """Test creating a CenteredRegion with tuple bounding box."""
        region = map_to_create_region(
            {
                "type": "centered_region",
                "parameters": {
                    "latitude": 45.0,
                    "longitude": -120.0,
                    "bounding_box_degrees": (5.0, 10.0),
                },
            }
        )
        assert isinstance(region, CenteredRegion)
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_create_bounding_box_region(self):
        """Test creating a BoundingBoxRegion via factory function."""
        region = map_to_create_region(
            {
                "type": "bounded_region",
                "parameters": {
                    "latitude_min": 40.0,
                    "latitude_max": 50.0,
                    "longitude_min": -125.0,
                    "longitude_max": -115.0,
                },
            }
        )
        assert isinstance(region, BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0

    def test_create_shapefile_region(self):
        """Test creating a ShapefileRegion via factory function."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = map_to_create_region(
                {
                    "type": "shapefile_region",
                    "parameters": {"shapefile_path": "/path/to/shapefile.shp"},
                }
            )
            assert isinstance(region, ShapefileRegion)
            assert region.shapefile_path == Path("/path/to/shapefile.shp")

    def test_create_region_invalid_parameters(self):
        """Test create_region with invalid parameter combinations."""
        # Missing required parameters for CenteredRegion
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            map_to_create_region(
                {
                    "type": "centered_region",
                    "parameters": {"latitude": 45.0, "longitude": -120.0},
                }
            )
            # Missing bounding_box_degrees

        # Missing required parameters for BoundingBoxRegion
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            map_to_create_region(
                {
                    "type": "bounded_region",
                    "parameters": {
                        "latitude_min": 40.0,
                        "latitude_max": 50.0,
                        "longitude_min": -125.0,
                    },
                }
            )
            # Missing longitude_max

        # Mixed parameters that don't form a valid region
        with pytest.raises(
            TypeError,
            match="got an unexpected keyword argument 'longitude_min'. Did you mean 'longitude'?",
        ):
            map_to_create_region(
                {
                    "type": "centered_region",
                    "parameters": {"latitude": 45.0, "longitude_min": -125.0},
                }
            )

    def test_create_region_priority_order(self):
        """Test that shapefile_path takes priority over other parameters."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = Mock()
            region = map_to_create_region(
                {
                    "type": "shapefile_region",
                    "parameters": {
                        "shapefile_path": "/path/to/shapefile.shp",
                    },
                }
            )
            # Should create ShapefileRegion, not CenteredRegion
            assert isinstance(region, ShapefileRegion)


class TestMapToCreateRegion:
    """Test the map_to_create_region function."""

    def test_map_to_create_region_centered(self):
        """Test mapping dictionary to CenteredRegion."""
        kwargs = {
            "type": "centered_region",
            "parameters": {
                "latitude": 45.0,
                "longitude": -120.0,
                "bounding_box_degrees": 10.0,
            },
        }
        region = map_to_create_region(kwargs)
        assert isinstance(region, CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_map_to_create_region_bounding_box(self):
        """Test mapping dictionary to BoundingBoxRegion."""
        kwargs = {
            "type": "bounded_region",
            "parameters": {
                "latitude_min": 40.0,
                "latitude_max": 50.0,
                "longitude_min": -125.0,
                "longitude_max": -115.0,
            },
        }
        region = map_to_create_region(kwargs)
        assert isinstance(region, BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0

    def test_map_to_create_region_shapefile(self):
        """Test mapping dictionary to ShapefileRegion."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = Mock()
            kwargs = {
                "type": "shapefile_region",
                "parameters": {"shapefile_path": "/path/to/shapefile.shp"},
            }
            region = map_to_create_region(kwargs)
            assert isinstance(region, ShapefileRegion)
            assert region.shapefile_path == Path("/path/to/shapefile.shp")


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
            "type": "centered_region",
            "parameters": {
                "latitude": 45.0,
                "longitude": -120.0,
                "bounding_box_degrees": 10.0,
            },
        }

        # Test that map_to_create_region works correctly
        region = map_to_create_region(region_data)

        assert isinstance(region, CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_region_with_individual_case_integration(self):
        """Test that regions work correctly with IndividualCase."""
        from extremeweatherbench import case

        region = CenteredRegion.create_region(
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
