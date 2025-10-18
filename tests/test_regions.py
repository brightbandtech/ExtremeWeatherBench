"""Tests for the region-related functionality in regions.py."""

from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import MultiPolygon, Polygon

from extremeweatherbench import utils
from extremeweatherbench.regions import (
    BoundingBoxRegion,
    CenteredRegion,
    Region,
    RegionSubsetter,
    ShapefileRegion,
    map_to_create_region,
    subset_cases_to_region,
    subset_results_to_region,
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
            _ = region.as_geopandas()
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
                _ = region.as_geopandas()


class TestRegionToGeopandas:
    """Test the to_geopandas() method for all Region subclasses."""

    def test_centered_region_to_geopandas_single_box(self):
        """Test CenteredRegion.as_geopandas() with single bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        gdf = region.as_geopandas()

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Verify the polygon coordinates
        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds (should be 40-50 lat, -125 to -115 lon after conversion)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted)

    def test_centered_region_to_geopandas_tuple_box(self):
        """Test CenteredRegion.as_geopandas() with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        gdf = region.as_geopandas()

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Check bounds (should be 42.5-47.5 lat, -125 to -115 lon after conversion)
        bounds = gdf.geometry.iloc[0].bounds
        assert abs(bounds[1] - 42.5) < 0.001  # min lat
        assert abs(bounds[3] - 47.5) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted)

    def test_bounding_box_region_to_geopandas(self):
        """Test BoundingBoxRegion.as_geopandas()."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        gdf = region.as_geopandas()

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Verify the polygon coordinates
        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds (longitude should be converted to -180 to 180)
        bounds = polygon.bounds
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted to -180 to 180)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted to -180 to 180)

    def test_shapefile_region_to_geopandas(self):
        """Test ShapefileRegion.as_geopandas()."""
        # Create a mock GeoDataFrame
        mock_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            gdf = region.as_geopandas()

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
        polar_gdf = polar_region.as_geopandas()
        assert isinstance(polar_gdf, gpd.GeoDataFrame)

        # Test CenteredRegion crossing the dateline
        dateline_region = CenteredRegion.create_region(
            latitude=45.0, longitude=175.0, bounding_box_degrees=10.0
        )
        dateline_gdf = dateline_region.as_geopandas()
        assert isinstance(dateline_gdf, gpd.GeoDataFrame)

        # Test with very small bounding box
        small_region = BoundingBoxRegion.create_region(
            latitude_min=44.9,
            latitude_max=45.1,
            longitude_min=-120.1,
            longitude_max=-119.9,
        )
        small_gdf = small_region.as_geopandas()
        assert isinstance(small_gdf, gpd.GeoDataFrame)

    def test_region_to_geopandas_longitude_conversion(self):
        """Test that longitudes are properly converted to -180 to 180 range."""
        # Test with negative longitudes
        region_neg = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        gdf_neg = region_neg.as_geopandas()
        bounds_neg = gdf_neg.geometry.iloc[0].bounds
        assert bounds_neg[0] >= -180  # min lon should be >= -180
        assert bounds_neg[2] <= 180  # max lon should be <= 180

        # Test with positive longitudes
        region_pos = CenteredRegion.create_region(
            latitude=45.0, longitude=120.0, bounding_box_degrees=10.0
        )
        gdf_pos = region_pos.as_geopandas()
        bounds_pos = gdf_pos.geometry.iloc[0].bounds
        assert bounds_pos[0] >= 0  # min lon should be >= 0
        assert bounds_pos[2] >= 0  # max lon should be >= 0


class TestRegionMask:
    """Test the mask() method for all Region subclasses."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        lats = np.linspace(90, -90, 181)  # Monotonically decreasing
        lons = np.linspace(0, 359, 360)
        data = np.random.random((len(lats), len(lons)))
        return xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

    @pytest.fixture
    def sample_dataset_180(self):
        """Create a sample dataset with -180 to 180 longitude."""
        lats = np.linspace(90, -90, 181)  # Monotonically decreasing
        lons = np.linspace(-180, 179, 360)
        data = np.random.random((len(lats), len(lons)))
        return xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

    def test_centered_region_mask(self, sample_dataset):
        """Test CenteredRegion.mask() method."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=240.0, bounding_box_degrees=10.0
        )

        masked_dataset = region.mask(sample_dataset)

        # Verify the masked dataset is a subset of the original
        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        # Dataset should be sliced to bounding box
        assert masked_dataset.sizes["latitude"] < sample_dataset.sizes["latitude"]
        assert masked_dataset.sizes["longitude"] <= sample_dataset.sizes["longitude"]

        # Check latitude range is approximately correct
        assert masked_dataset.latitude.min() >= 40.0 - 1
        assert masked_dataset.latitude.max() <= 50.0 + 1

    def test_bounding_box_region_mask(self, sample_dataset):
        """Test BoundingBoxRegion.mask() method."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=235.0,
            longitude_max=245.0,
        )

        masked_dataset = region.mask(sample_dataset)

        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        # Dataset should be sliced to bounding box
        assert masked_dataset.sizes["latitude"] < sample_dataset.sizes["latitude"]
        assert masked_dataset.sizes["longitude"] <= sample_dataset.sizes["longitude"]

    def test_shapefile_region_mask(self, sample_dataset):
        """Test ShapefileRegion.mask() method."""
        mock_polygon = Polygon([(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            masked_dataset = region.mask(sample_dataset, drop=False)

            assert isinstance(masked_dataset, xr.Dataset)
            assert "temperature" in masked_dataset.data_vars
            # With drop=False, dataset should be sliced to bounding box
            # but keep original coords (with NaNs outside polygon)
            assert masked_dataset.sizes["latitude"] < sample_dataset.sizes["latitude"]
            assert (
                masked_dataset.sizes["longitude"] <= sample_dataset.sizes["longitude"]
            )
            # Check that values outside polygon are masked (NaN)
            assert np.any(np.isnan(masked_dataset.temperature.values))

    def test_region_mask_consistency(self, sample_dataset):
        """Test that different region types produce consistent mask behavior."""
        # Create regions with similar coverage (using 0-360 longitude)
        centered = CenteredRegion.create_region(
            latitude=45.0, longitude=240.0, bounding_box_degrees=10.0
        )
        bbox = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=235.0,
            longitude_max=245.0,
        )

        # Both should produce sliced datasets
        centered_masked = centered.mask(sample_dataset)
        bbox_masked = bbox.mask(sample_dataset)

        assert isinstance(centered_masked, xr.Dataset)
        assert isinstance(bbox_masked, xr.Dataset)
        assert "temperature" in centered_masked.data_vars
        assert "temperature" in bbox_masked.data_vars

    def test_centered_region_mask_with_180_longitude(self, sample_dataset_180):
        """Test CenteredRegion.mask() with -180 to 180 longitude."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        masked_dataset = region.mask(sample_dataset_180)

        # Verify the masked dataset is a subset of the original
        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        # Dataset should be sliced to bounding box
        assert masked_dataset.sizes["latitude"] < sample_dataset_180.sizes["latitude"]
        assert (
            masked_dataset.sizes["longitude"] <= sample_dataset_180.sizes["longitude"]
        )

        # Check latitude range is approximately correct
        assert masked_dataset.latitude.min() >= 40.0 - 1
        assert masked_dataset.latitude.max() <= 50.0 + 1

        # Check longitude range is approximately correct
        assert masked_dataset.longitude.min() >= -125.0 - 1
        assert masked_dataset.longitude.max() <= -115.0 + 1

    def test_bounding_box_region_mask_with_180_longitude(self, sample_dataset_180):
        """Test BoundingBoxRegion.mask() with -180 to 180 longitude."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        masked_dataset = region.mask(sample_dataset_180)

        assert isinstance(masked_dataset, xr.Dataset)
        assert "temperature" in masked_dataset.data_vars
        # Dataset should be sliced to bounding box
        assert masked_dataset.sizes["latitude"] < sample_dataset_180.sizes["latitude"]
        assert (
            masked_dataset.sizes["longitude"] <= sample_dataset_180.sizes["longitude"]
        )

        # Check coordinate ranges
        assert masked_dataset.latitude.min() >= 40.0 - 1
        assert masked_dataset.latitude.max() <= 50.0 + 1
        assert masked_dataset.longitude.min() >= -125.0 - 1
        assert masked_dataset.longitude.max() <= -115.0 + 1

    def test_shapefile_region_mask_with_180_longitude(self, sample_dataset_180):
        """Test ShapefileRegion.mask() with -180 to 180 longitude."""
        # Create polygon with -180 to 180 coordinates
        mock_polygon = Polygon(
            [(-120, 40), (-110, 40), (-110, 50), (-120, 50), (-120, 40)]
        )
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            masked_dataset = region.mask(sample_dataset_180, drop=False)

            assert isinstance(masked_dataset, xr.Dataset)
            assert "temperature" in masked_dataset.data_vars
            # Dataset should be sliced to bounding box
            assert (
                masked_dataset.sizes["latitude"] < sample_dataset_180.sizes["latitude"]
            )
            assert (
                masked_dataset.sizes["longitude"]
                <= sample_dataset_180.sizes["longitude"]
            )
            # Check that values outside polygon are masked (NaN)
            assert np.any(np.isnan(masked_dataset.temperature.values))

            # Check coordinate ranges
            assert masked_dataset.latitude.min() >= 40.0 - 1
            assert masked_dataset.latitude.max() <= 50.0 + 1
            assert masked_dataset.longitude.min() >= -120.0 - 1
            assert masked_dataset.longitude.max() <= -110.0 + 1

    def test_region_mask_at_180_boundary(self, sample_dataset_180):
        """Test masking at the -180/180 boundary."""
        # Test a region that's right at the edge of -180
        region_west = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-180.0,
            longitude_max=-170.0,
        )

        masked_west = region_west.mask(sample_dataset_180)

        assert isinstance(masked_west, xr.Dataset)
        assert masked_west.sizes["latitude"] < sample_dataset_180.sizes["latitude"]
        assert masked_west.sizes["longitude"] <= sample_dataset_180.sizes["longitude"]

        # Test a region that's right at the edge of 180
        region_east = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=170.0,
            longitude_max=179.0,
        )

        masked_east = region_east.mask(sample_dataset_180)

        assert isinstance(masked_east, xr.Dataset)
        assert masked_east.sizes["latitude"] < sample_dataset_180.sizes["latitude"]
        assert masked_east.sizes["longitude"] <= sample_dataset_180.sizes["longitude"]


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
        centered_gdf = centered.as_geopandas()
        bbox_gdf = bbox.as_geopandas()

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
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'longitude_max'"
        ):
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
            match="got an unexpected keyword argument 'longitude_min'",
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


class TestCreateGeopandasFromBounds:
    """Test the _create_geopandas_from_bounds helper function."""

    def test_normal_case_no_antimeridian_crossing(self):
        """Test normal case where region doesn't cross antimeridian."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=100.0,
            longitude_max=120.0,
            latitude_min=40.0,
            latitude_max=50.0,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds - should be converted to -180 to 180 range
        bounds = polygon.bounds
        assert abs(bounds[0] - 100.0) < 0.001  # min lon
        assert abs(bounds[2] - 120.0) < 0.001  # max lon
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat

    def test_antimeridian_crossing_case(self):
        """Test case where region crosses the antimeridian."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=170.0,  # This will become -10 after conversion
            longitude_max=190.0,  # This will become 10 after conversion
            latitude_min=40.0,
            latitude_max=50.0,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        geometry = gdf.geometry.iloc[0]
        # Should be a MultiPolygon for antimeridian crossing
        assert isinstance(geometry, MultiPolygon)

        # Check that we have two polygons
        assert len(geometry.geoms) == 2

    def test_negative_longitude_input(self):
        """Test with negative longitude inputs."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=-120.0,
            longitude_max=-100.0,
            latitude_min=40.0,
            latitude_max=50.0,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        # Check bounds - should be converted to -180 to 180 range
        bounds = polygon.bounds
        assert abs(bounds[0] - (-120.0)) < 0.001  # min lon
        assert abs(bounds[2] - (-100.0)) < 0.001  # max lon

    def test_longitude_conversion_to_180_range(self):
        """Test that longitudes are properly converted to -180 to 180 range."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        # Test with longitudes that need conversion
        gdf = _create_geopandas_from_bounds(
            longitude_min=200.0,  # Should become -160
            longitude_max=220.0,  # Should become -140
            latitude_min=40.0,
            latitude_max=50.0,
        )

        bounds = gdf.geometry.iloc[0].bounds
        assert abs(bounds[0] - (-160.0)) < 0.001  # min lon converted
        assert abs(bounds[2] - (-140.0)) < 0.001  # max lon converted

    def test_edge_case_180_degree_longitude(self):
        """Test edge case with 180 degree longitude."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=175.0,
            longitude_max=185.0,  # This crosses 180
            latitude_min=40.0,
            latitude_max=50.0,
        )

        geometry = gdf.geometry.iloc[0]
        # Should be a MultiPolygon
        assert isinstance(geometry, MultiPolygon)
        assert len(geometry.geoms) == 2

    def test_edge_case_0_degree_longitude(self):
        """Test edge case with 0 degree longitude."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=-5.0,
            longitude_max=5.0,  # This crosses 0
            latitude_min=40.0,
            latitude_max=50.0,
        )

        geometry = gdf.geometry.iloc[0]
        # Should be a single Polygon (no antimeridian crossing)
        assert isinstance(geometry, Polygon)

    def test_polar_regions(self):
        """Test with regions near the poles."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=0.0,
            longitude_max=360.0,  # Full circle
            latitude_min=80.0,
            latitude_max=90.0,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        geometry = gdf.geometry.iloc[0]
        # After conversion to -180 to 180, 360 becomes 0, so no antimeridian crossing
        assert isinstance(geometry, Polygon)

    def test_small_region(self):
        """Test with a very small region."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=45.0, longitude_max=45.1, latitude_min=40.0, latitude_max=40.1
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, Polygon)

        bounds = polygon.bounds
        assert abs(bounds[0] - 45.0) < 0.001
        assert abs(bounds[2] - 45.1) < 0.001


class TestTotalBounds:
    """Test the as_geopandas().total_bounds method for all Region subclasses."""

    def test_centered_region_total_bounds(self):
        """Test CenteredRegion.as_geopandas().total_bounds method."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.as_geopandas().total_bounds
        assert coords[0] == -125.0
        assert coords[1] == 40.0
        assert coords[2] == -115.0
        assert coords[3] == 50.0

    def test_centered_region_total_bounds_tuple_box(self):
        """Test CenteredRegion.as_geopandas().total_bounds with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        coords = region.as_geopandas().total_bounds
        assert coords[0] == -125.0
        assert coords[1] == 42.5
        assert coords[2] == -115.0
        assert coords[3] == 47.5

    def test_bounding_box_region_total_bounds(self):
        """Test BoundingBoxRegion.as_geopandas().total_bounds method."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        coords = region.as_geopandas().total_bounds

        # Check coordinate values
        assert coords[0] == -125.0
        assert coords[1] == 40.0
        assert coords[2] == -115.0
        assert coords[3] == 50.0

    def test_shapefile_region_bounding_coordinates(self):
        """Test ShapefileRegion.as_geopandas().total_bounds method."""
        # Create a mock polygon with known bounds
        mock_polygon = Polygon([(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            coords = region.as_geopandas().total_bounds

            # Check coordinate values (should match the polygon bounds)
            assert coords[0] == 240.0
            assert coords[1] == 40.0
            assert coords[2] == 250.0
            assert coords[3] == 50.0

    def test_bounding_coordinates_antimeridian_crossing(self):
        """Test as_geopandas().total_bounds with antimeridian crossing."""
        # Create a region that truly crosses the antimeridian (longitude spans > 180°)
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=20.0,  # 165° to 185° crosses antimeridian
        )

        coords = region.as_geopandas().total_bounds

        # For antimeridian crossing regions, coordinates should span -180 to 180
        assert coords[0] == -180.0
        assert coords[1] == 35.0
        assert coords[2] == 180.0
        assert coords[3] == 55.0

    def test_bounding_coordinates_near_antimeridian_no_crossing(self):
        """Test as_geopandas().total_bounds for region near but not crossing
        antimeridian."""
        # Create a region that goes exactly to 180° but doesn't cross it
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=10.0,  # 170° to 180°, no crossing
        )

        coords = region.as_geopandas().total_bounds

        # Should be a single polygon from 170° to 180°
        assert coords[0] == 170.0
        assert coords[1] == 40.0
        assert coords[2] == 180.0
        assert coords[3] == 50.0

    def test_total_bounds_longitude_conversion(self):
        """Test that as_geopandas().total_bounds handles longitude conversion."""
        # Test with positive longitude that should be converted
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=200.0,
            bounding_box_degrees=10.0,  # 200° = -160°
        )

        coords = region.as_geopandas().total_bounds

        # Coordinates should be in -180 to 180 range
        assert coords[0] >= -180
        assert coords[2] <= 180

        # Test specific expected values
        # Center at 200° (converted to -160°), ±5° box should be -165° to -155°
        expected_min_lon = -165.0
        expected_max_lon = -155.0
        assert coords[0] == expected_min_lon
        assert coords[2] == expected_max_lon

    def test_bounding_coordinates_edge_cases(self):
        """Test as_geopandas().total_bounds with edge cases."""
        # Test very small region
        small_region = BoundingBoxRegion.create_region(
            latitude_min=44.99,
            latitude_max=45.01,
            longitude_min=-120.01,
            longitude_max=-119.99,
        )
        coords = small_region.as_geopandas().total_bounds
        assert coords[1] > coords[0]
        assert coords[3] > coords[2]

        # Test polar region
        polar_region = CenteredRegion.create_region(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        polar_coords = polar_region.as_geopandas().total_bounds
        assert polar_coords[0] >= -90
        assert polar_coords[1] <= 90

    def test_bounding_coordinates_return_type(self):
        """Test that as_geopandas().total_bounds returns correct type."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.as_geopandas().total_bounds

        # Should return a numpy array with 4 elements
        assert hasattr(coords, "__len__")
        assert len(coords) == 4


@pytest.mark.integration
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
        from extremeweatherbench import cases

        region = CenteredRegion.create_region(
            latitude=45.0, longitude=240.0, bounding_box_degrees=10.0
        )

        # Create a sample dataset
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        data = np.random.random((len(lats), len(lons)))
        dataset = xr.Dataset(
            {"temperature": (["latitude", "longitude"], data)},
            coords={"latitude": lats, "longitude": lons},
        )

        # Create IndividualCase with the region
        individual_case = cases.IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=region,
            event_type="test",
        )
        # Test that mask works to subset coordinates
        subset = individual_case.location.mask(dataset)
        assert len(subset.latitude) < len(dataset.latitude)
        assert len(subset.longitude) < len(dataset.longitude)

        # Test that ShapefileRegion supports drop parameter
        mock_polygon = Polygon([(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            shapefile_region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )
            # drop=False keeps the sliced coordinates
            subset_no_drop = shapefile_region.mask(dataset, drop=False)
            assert len(subset_no_drop.latitude) < len(dataset.latitude)
            assert len(subset_no_drop.longitude) < len(dataset.longitude)

            # drop=True drops NaN values
            subset_drop = shapefile_region.mask(dataset, drop=True)
            assert len(subset_drop.latitude) <= len(subset_no_drop.latitude)
            assert len(subset_drop.longitude) <= len(subset_no_drop.longitude)
        # Test that mask with drop=False works to subset coordinates
        subset = individual_case.location.mask(dataset, drop=False)
        assert len(subset.latitude) < len(dataset.latitude)
        assert len(subset.longitude) < len(dataset.longitude)


class TestRegionGeometricOperations:
    """Test geometric operations on regions."""

    def test_region_intersects(self):
        """Test the intersects method."""
        # Create two overlapping regions
        region1 = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        region2 = BoundingBoxRegion.create_region(
            latitude_min=45.0,
            latitude_max=55.0,
            longitude_min=-120.0,
            longitude_max=-110.0,
        )

        # They should intersect
        assert region1.intersects(region2)
        assert region2.intersects(region1)

        # Test non-overlapping regions
        region3 = BoundingBoxRegion.create_region(
            latitude_min=60.0,
            latitude_max=70.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        assert not region1.intersects(region3)
        assert not region3.intersects(region1)

    def test_region_contains(self):
        """Test the contains method."""
        # Create a larger region
        large_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=60.0,
            longitude_min=-130.0,
            longitude_max=-110.0,
        )

        # Create a smaller region inside it
        small_region = BoundingBoxRegion.create_region(
            latitude_min=45.0,
            latitude_max=55.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        # Large should contain small, but not vice versa
        assert large_region.contains(small_region)
        assert not small_region.contains(large_region)

    def test_area_overlap_fraction(self):
        """Test the area_overlap_fraction method."""
        # Create two regions with known overlap
        region1 = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        region2 = BoundingBoxRegion.create_region(
            latitude_min=45.0,
            latitude_max=55.0,
            longitude_min=-120.0,
            longitude_max=-110.0,
        )

        # Calculate overlap fraction
        overlap_fraction = region1.area_overlap_fraction(region2)

        # Should be between 0 and 1
        assert 0.0 <= overlap_fraction <= 1.0

        # Test with non-overlapping regions
        region3 = BoundingBoxRegion.create_region(
            latitude_min=60.0,
            latitude_max=70.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        overlap_fraction_none = region1.area_overlap_fraction(region3)
        assert overlap_fraction_none == 0.0

        # Test region with itself (should be 1.0)
        self_overlap = region1.area_overlap_fraction(region1)
        assert abs(self_overlap - 1.0) < 0.01  # Allow small numerical error


class TestRegionSubsetter:
    """Test the RegionSubsetter class."""

    @pytest.fixture
    def target_region(self):
        """Create a target region for subsetting."""
        return BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

    @pytest.fixture
    def sample_cases(self):
        """Create sample individual cases for testing."""
        from extremeweatherbench import cases

        # Case that intersects with target region
        intersecting_case = cases.IndividualCase(
            case_id_number=1,
            title="Intersecting Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-03"),
            location=BoundingBoxRegion.create_region(
                latitude_min=45.0,
                latitude_max=55.0,
                longitude_min=-120.0,
                longitude_max=-110.0,
            ),
            event_type="heat_wave",
        )

        # Case completely within target region
        contained_case = cases.IndividualCase(
            case_id_number=2,
            title="Contained Case",
            start_date=pd.Timestamp("2021-02-01"),
            end_date=pd.Timestamp("2021-02-03"),
            location=BoundingBoxRegion.create_region(
                latitude_min=42.0,
                latitude_max=48.0,
                longitude_min=-123.0,
                longitude_max=-117.0,
            ),
            event_type="cold_wave",
        )

        # Case outside target region
        outside_case = cases.IndividualCase(
            case_id_number=3,
            title="Outside Case",
            start_date=pd.Timestamp("2021-03-01"),
            end_date=pd.Timestamp("2021-03-03"),
            location=BoundingBoxRegion.create_region(
                latitude_min=60.0,
                latitude_max=70.0,
                longitude_min=-125.0,
                longitude_max=-115.0,
            ),
            event_type="snow_storm",
        )

        return cases.IndividualCaseCollection(
            cases=[intersecting_case, contained_case, outside_case]
        )

    def test_subsetter_initialization_with_region(self, target_region):
        """Test RegionSubsetter initialization with Region object."""
        subsetter = RegionSubsetter(
            region=target_region, method="intersects", percent_threshold=0.5
        )

        assert subsetter.region == target_region
        assert subsetter.method == "intersects"
        assert subsetter.percent_threshold == 0.5

    def test_subsetter_initialization_with_dict(self):
        """Test RegionSubsetter initialization with dictionary."""
        region_dict = {
            "latitude_min": 40.0,
            "latitude_max": 50.0,
            "longitude_min": -125.0,
            "longitude_max": -115.0,
        }

        subsetter = RegionSubsetter(
            region=region_dict, method="percent", percent_threshold=0.75
        )

        assert isinstance(subsetter.region, BoundingBoxRegion)
        assert subsetter.method == "percent"
        assert subsetter.percent_threshold == 0.75

    def test_subset_case_collection_intersects(self, target_region, sample_cases):
        """Test subsetting with intersects method."""
        subsetter = RegionSubsetter(region=target_region, method="intersects")

        subset_cases = subsetter.subset_case_collection(sample_cases)

        # Should include intersecting and contained cases, but not outside
        assert len(subset_cases.cases) == 2
        case_ids = {case.case_id_number for case in subset_cases.cases}
        assert case_ids == {1, 2}  # intersecting and contained

    def test_subset_case_collection_all(self, target_region, sample_cases):
        """Test subsetting with all method."""
        subsetter = RegionSubsetter(region=target_region, method="all")

        subset_cases = subsetter.subset_case_collection(sample_cases)

        # Should only include contained case
        assert len(subset_cases.cases) == 1
        assert subset_cases.cases[0].case_id_number == 2  # contained case

    def test_subset_case_collection_percent(self, target_region, sample_cases):
        """Test subsetting with percent method."""
        subsetter = RegionSubsetter(
            region=target_region, method="percent", percent_threshold=0.5
        )

        subset_cases = subsetter.subset_case_collection(sample_cases)

        # This depends on the actual overlap, but should include at least the contained
        # case
        case_ids = {case.case_id_number for case in subset_cases.cases}
        assert 2 in case_ids  # contained case should always be included

    def test_subset_case_collection_different_thresholds(
        self, target_region, sample_cases
    ):
        """Test subsetting with different percent thresholds."""
        # Low threshold - should include more cases
        low_threshold_subsetter = RegionSubsetter(
            region=target_region, method="percent", percent_threshold=0.1
        )

        low_threshold_cases = low_threshold_subsetter.subset_case_collection(
            sample_cases
        )

        # High threshold - should include fewer cases
        high_threshold_subsetter = RegionSubsetter(
            region=target_region, method="percent", percent_threshold=0.9
        )

        high_threshold_cases = high_threshold_subsetter.subset_case_collection(
            sample_cases
        )

        # Low threshold should include at least as many as high threshold
        assert len(low_threshold_cases.cases) >= len(high_threshold_cases.cases)

    def test_subset_results_to_region(self, target_region, sample_cases):
        """Test subsetting results DataFrame."""
        # Create a mock results DataFrame
        results_df = pd.DataFrame(
            {
                "case_id_number": [1, 2, 3, 1, 2, 3],
                "metric": ["mae", "mae", "mae", "rmse", "rmse", "rmse"],
                "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "lead_time": [0, 0, 0, 6, 6, 6],
            }
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        subset_results = subset_results_to_region(subsetter, results_df, sample_cases)

        # Should only include results for cases 1 and 2 (intersecting and contained)
        assert len(subset_results) == 4  # 2 cases * 2 metrics
        case_ids = set(subset_results["case_id_number"])
        assert case_ids == {1, 2}

    def test_invalid_method_raises_error(self, target_region):
        """Test that invalid method raises ValueError."""
        subsetter = RegionSubsetter(region=target_region, method="intersects")

        # Manually set invalid method to test error handling
        subsetter.method = "invalid_method"

        from extremeweatherbench import cases

        dummy_case = cases.IndividualCase(
            case_id_number=1,
            title="Test",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=target_region,
            event_type="test",
        )

        with pytest.raises(ValueError, match="Unknown method"):
            subsetter._should_include_case(dummy_case)


class TestConvenienceFunctions:
    """Test convenience functions for region subsetting."""

    @pytest.fixture
    def sample_case_collection(self):
        """Create a sample case collection."""
        from extremeweatherbench import cases

        case1 = cases.IndividualCase(
            case_id_number=1,
            title="Case 1",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-03"),
            location=BoundingBoxRegion.create_region(
                latitude_min=45.0,
                latitude_max=55.0,
                longitude_min=-120.0,
                longitude_max=-110.0,
            ),
            event_type="heat_wave",
        )

        case2 = cases.IndividualCase(
            case_id_number=2,
            title="Case 2",
            start_date=pd.Timestamp("2021-02-01"),
            end_date=pd.Timestamp("2021-02-03"),
            location=BoundingBoxRegion.create_region(
                latitude_min=60.0,
                latitude_max=70.0,
                longitude_min=-125.0,
                longitude_max=-115.0,
            ),
            event_type="cold_wave",
        )

        return cases.IndividualCaseCollection(cases=[case1, case2])

    def test_subset_cases_to_region_with_region_object(self, sample_case_collection):
        """Test subset_cases_to_region with Region object."""
        target_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subset_cases = subset_cases_to_region(
            sample_case_collection, target_region, method="intersects"
        )

        # Should include case 1 (intersects) but not case 2 (outside)
        assert len(subset_cases.cases) == 1
        assert subset_cases.cases[0].case_id_number == 1

    def test_subset_cases_to_region_with_dict(self, sample_case_collection):
        """Test subset_cases_to_region with dictionary."""
        region_dict = {
            "latitude_min": 40.0,
            "latitude_max": 50.0,
            "longitude_min": -125.0,
            "longitude_max": -115.0,
        }

        subset_cases = subset_cases_to_region(
            sample_case_collection, region_dict, method="intersects"
        )

        assert len(subset_cases.cases) == 1
        assert subset_cases.cases[0].case_id_number == 1

    def test_subset_cases_to_region_with_percent_method(self, sample_case_collection):
        """Test subset_cases_to_region with percent method."""
        target_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subset_cases = subset_cases_to_region(
            sample_case_collection,
            target_region,
            method="percent",
            percent_threshold=0.3,
        )

        # Results depend on actual area overlap calculations
        assert isinstance(subset_cases, type(sample_case_collection))

    def test_subset_cases_to_region_all_method(self, sample_case_collection):
        """Test subset_cases_to_region with all method."""
        # Create a region that contains case 1 completely
        large_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=60.0,
            longitude_min=-130.0,
            longitude_max=-100.0,
        )

        subset_cases = subset_cases_to_region(
            sample_case_collection, large_region, method="all"
        )

        # Should include case 1 which is completely within the large region
        assert len(subset_cases.cases) >= 1

    def test_subset_results_to_region_convenience(self, sample_case_collection):
        """Test subset_results_to_region convenience function."""
        # Create mock results
        results_df = pd.DataFrame(
            {
                "case_id_number": [1, 2, 1, 2],
                "metric": ["mae", "mae", "rmse", "rmse"],
                "value": [0.1, 0.2, 0.3, 0.4],
            }
        )

        target_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        subset_results = subset_results_to_region(
            subsetter, results_df, sample_case_collection
        )

        # Should only include results for case 1
        assert len(subset_results) == 2  # 1 case * 2 metrics
        assert all(subset_results["case_id_number"] == 1)


class TestRegionSubsettingEdgeCases:
    """Test edge cases for region subsetting."""

    def test_empty_case_collection(self):
        """Test subsetting with empty case collection."""
        from extremeweatherbench import cases

        empty_collection = cases.IndividualCaseCollection(cases=[])
        target_region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        subset_cases = subsetter.subset_case_collection(empty_collection)
        assert len(subset_cases.cases) == 0

    def test_very_small_regions(self):
        """Test subsetting with very small regions."""
        from extremeweatherbench import cases

        # Create a very small case region
        tiny_case = cases.IndividualCase(
            case_id_number=1,
            title="Tiny Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=BoundingBoxRegion.create_region(
                latitude_min=45.0,
                latitude_max=45.1,
                longitude_min=-120.0,
                longitude_max=-119.9,
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[tiny_case])

        # Create a target region that should intersect
        target_region = BoundingBoxRegion.create_region(
            latitude_min=44.9,
            latitude_max=45.2,
            longitude_min=-120.1,
            longitude_max=-119.8,
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        subset_cases = subsetter.subset_case_collection(case_collection)
        assert len(subset_cases.cases) == 1

    def test_antimeridian_crossing_regions(self):
        """Test subsetting with regions that cross the antimeridian."""
        from extremeweatherbench import cases

        # Create a case near the dateline
        dateline_case = cases.IndividualCase(
            case_id_number=1,
            title="Dateline Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=CenteredRegion.create_region(
                latitude=45.0,
                longitude=175.0,  # Near dateline
                bounding_box_degrees=10.0,
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[dateline_case])

        # Create a target region that might cross antimeridian
        target_region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=180.0,  # At dateline
            bounding_box_degrees=15.0,
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        # Should handle antimeridian crossing gracefully
        subset_cases = subsetter.subset_case_collection(case_collection)
        assert isinstance(subset_cases, cases.IndividualCaseCollection)

    def test_polar_regions(self):
        """Test subsetting with polar regions."""
        from extremeweatherbench import cases

        # Create a case near the North Pole
        polar_case = cases.IndividualCase(
            case_id_number=1,
            title="Polar Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=CenteredRegion.create_region(
                latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[polar_case])

        # Create a target region at high latitudes
        target_region = BoundingBoxRegion.create_region(
            latitude_min=80.0,
            latitude_max=90.0,
            longitude_min=-30.0,
            longitude_max=30.0,
        )

        subsetter = RegionSubsetter(region=target_region, method="intersects")

        # Should handle polar coordinates gracefully
        subset_cases = subsetter.subset_case_collection(case_collection)
        assert isinstance(subset_cases, cases.IndividualCaseCollection)
