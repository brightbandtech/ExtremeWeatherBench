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

        # Check bounds (should be 40-50 lat, -125 to -115 lon after conversion to -180 to 180)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted to -180 to 180)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted to -180 to 180)

    def test_centered_region_to_geopandas_tuple_box(self):
        """Test CenteredRegion.geopandas with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        gdf = region.geopandas

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Check bounds (should be 42.5-47.5 lat, -125 to -115 lon after conversion to -180 to 180)
        bounds = gdf.geometry.iloc[0].bounds
        assert abs(bounds[1] - 42.5) < 0.001  # min lat
        assert abs(bounds[3] - 47.5) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted to -180 to 180)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted to -180 to 180)

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

        # Check bounds (longitude should be converted to -180 to 180)
        bounds = polygon.bounds
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted to -180 to 180)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted to -180 to 180)

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
        """Test that longitudes are properly converted to -180 to 180 range."""
        # Test with negative longitudes
        region_neg = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        gdf_neg = region_neg.geopandas
        bounds_neg = gdf_neg.geometry.iloc[0].bounds
        assert bounds_neg[0] >= -180  # min lon should be >= -180
        assert bounds_neg[2] <= 180  # max lon should be <= 180

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


class TestGetBoundingCoordinates:
    """Test the get_bounding_coordinates method for all Region subclasses."""

    def test_centered_region_bounding_coordinates(self):
        """Test CenteredRegion.get_bounding_coordinates method."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.get_bounding_coordinates

        # Check that it's a named tuple with correct attributes
        assert hasattr(coords, "min_lon")
        assert hasattr(coords, "min_lat")
        assert hasattr(coords, "max_lon")
        assert hasattr(coords, "max_lat")

        # Check coordinate values (longitude should be converted to -180 to 180 range)
        assert abs(coords.min_lat - 40.0) < 0.001
        assert abs(coords.max_lat - 50.0) < 0.001
        assert abs(coords.min_lon - (-125.0)) < 0.001
        assert abs(coords.max_lon - (-115.0)) < 0.001

    def test_centered_region_bounding_coordinates_tuple_box(self):
        """Test CenteredRegion.get_bounding_coordinates with tuple bounding box."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        coords = region.get_bounding_coordinates

        # Check coordinate values (should be 42.5-47.5 lat, -125 to -115 lon)
        assert abs(coords.min_lat - 42.5) < 0.001
        assert abs(coords.max_lat - 47.5) < 0.001
        assert abs(coords.min_lon - (-125.0)) < 0.001
        assert abs(coords.max_lon - (-115.0)) < 0.001

    def test_bounding_box_region_bounding_coordinates(self):
        """Test BoundingBoxRegion.get_bounding_coordinates method."""
        region = BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        coords = region.get_bounding_coordinates

        # Check coordinate values
        assert abs(coords.min_lat - 40.0) < 0.001
        assert abs(coords.max_lat - 50.0) < 0.001
        assert abs(coords.min_lon - (-125.0)) < 0.001
        assert abs(coords.max_lon - (-115.0)) < 0.001

    def test_shapefile_region_bounding_coordinates(self):
        """Test ShapefileRegion.get_bounding_coordinates method."""
        # Create a mock polygon with known bounds
        mock_polygon = Polygon([(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with patch("geopandas.read_file", return_value=mock_gdf):
            region = ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            coords = region.get_bounding_coordinates

            # Check coordinate values (should match the polygon bounds)
            assert abs(coords.min_lon - 240.0) < 0.001
            assert abs(coords.max_lon - 250.0) < 0.001
            assert abs(coords.min_lat - 40.0) < 0.001
            assert abs(coords.max_lat - 50.0) < 0.001

    def test_bounding_coordinates_antimeridian_crossing(self):
        """Test get_bounding_coordinates with antimeridian crossing."""
        # Create a region that truly crosses the antimeridian (longitude spans > 180°)
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=20.0,  # 165° to 185° crosses antimeridian
        )

        coords = region.get_bounding_coordinates

        # For antimeridian crossing regions, coordinates should span -180 to 180
        assert coords.min_lon == -180.0
        assert coords.max_lon == 180.0
        assert abs(coords.min_lat - 35.0) < 0.001  # 45 - 10
        assert abs(coords.max_lat - 55.0) < 0.001  # 45 + 10

    def test_bounding_coordinates_near_antimeridian_no_crossing(self):
        """Test get_bounding_coordinates for region near but not crossing antimeridian."""
        # Create a region that goes exactly to 180° but doesn't cross it
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=10.0,  # 170° to 180°, no crossing
        )

        coords = region.get_bounding_coordinates

        # Should be a single polygon from 170° to 180°
        assert abs(coords.min_lon - 170.0) < 0.001
        assert abs(coords.max_lon - 180.0) < 0.001
        assert abs(coords.min_lat - 40.0) < 0.001
        assert abs(coords.max_lat - 50.0) < 0.001

    def test_bounding_coordinates_longitude_conversion(self):
        """Test that bounding coordinates properly handle longitude conversion."""
        # Test with positive longitude that should be converted
        region = CenteredRegion.create_region(
            latitude=45.0,
            longitude=200.0,
            bounding_box_degrees=10.0,  # 200° = -160°
        )

        coords = region.get_bounding_coordinates

        # Coordinates should be in -180 to 180 range
        assert coords.min_lon >= -180
        assert coords.max_lon <= 180

        # Test specific expected values
        # Center at 200° (converted to -160°), ±5° box should be -165° to -155°
        expected_min_lon = -165.0
        expected_max_lon = -155.0
        assert abs(coords.min_lon - expected_min_lon) < 0.001
        assert abs(coords.max_lon - expected_max_lon) < 0.001

    def test_bounding_coordinates_edge_cases(self):
        """Test get_bounding_coordinates with edge cases."""
        # Test very small region
        small_region = BoundingBoxRegion.create_region(
            latitude_min=44.99,
            latitude_max=45.01,
            longitude_min=-120.01,
            longitude_max=-119.99,
        )
        coords = small_region.get_bounding_coordinates
        assert coords.max_lat > coords.min_lat
        assert coords.max_lon > coords.min_lon

        # Test polar region
        polar_region = CenteredRegion.create_region(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        polar_coords = polar_region.get_bounding_coordinates
        assert polar_coords.min_lat >= -90
        assert polar_coords.max_lat <= 90

    def test_bounding_coordinates_return_type(self):
        """Test that get_bounding_coordinates returns correct type."""
        region = CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.get_bounding_coordinates

        # Should return a tuple
        assert isinstance(coords, tuple)
        assert len(coords) == 4

        # Should have named attributes
        assert hasattr(coords, "min_lon")
        assert hasattr(coords, "min_lat")
        assert hasattr(coords, "max_lon")
        assert hasattr(coords, "max_lat")

        # All values should be floats
        assert isinstance(coords.min_lon, float)
        assert isinstance(coords.min_lat, float)
        assert isinstance(coords.max_lon, float)
        assert isinstance(coords.max_lat, float)


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
        individual_case = cases.IndividualCase(
            case_id_number=1,
            title="Test Case",
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-02"),
            location=region,
            event_type="test",
        )
        # Test that mask with drop=True works to subset coordinates
        subset = individual_case.location.mask(dataset, drop=True)
        assert len(subset.latitude) < len(dataset.latitude)
        assert len(subset.longitude) < len(dataset.longitude)

        # Test that mask with drop=False works to subset coordinates
        subset = individual_case.location.mask(dataset, drop=False)
        assert len(subset.latitude) == len(dataset.latitude)
        assert len(subset.longitude) == len(dataset.longitude)
