"""Tests for the regions module."""

from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

from extremeweatherbench import regions, utils


class TestRegionClasses:
    """Test the regions.Region base class and its subclasses."""

    def test_region_base_class(self):
        """Test that regions.Region is an abstract base class that cannot be
        instantiated."""
        # regions.Region is now abstract and cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            regions.Region()

    def test_centered_region_creation(self):
        """Test regions.CenteredRegion creation with valid parameters."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0
        assert isinstance(region, regions.Region)

    def test_centered_region_with_tuple_bounding_box(self):
        """Test regions.CenteredRegion creation with tuple bounding box."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_bounding_box_region_creation(self):
        """Test regions.BoundingBoxRegion creation with valid parameters."""
        region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0
        assert isinstance(region, regions.Region)

    def test_shapefile_region_creation(self):
        """Test regions.ShapefileRegion creation with valid path."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.return_value = mock.Mock()
            region = regions.ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )
            assert region.shapefile_path == Path("/path/to/shapefile.shp")
            assert isinstance(region, regions.Region)

            # build_region should be called when geopandas is accessed
            _ = region.geopandas
            mock_read.assert_called_once_with(Path("/path/to/shapefile.shp"))

    def test_shapefile_region_with_string_path(self):
        """Test regions.ShapefileRegion creation with string path."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.return_value = mock.Mock()
            region = regions.ShapefileRegion.create_region(
                shapefile_path="shapefile.shp"
            )
            assert region.shapefile_path == Path("shapefile.shp")

    def test_shapefile_region_read_error(self):
        """Test regions.ShapefileRegion creation with invalid shapefile."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.side_effect = Exception("File not found")
            region = regions.ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            # build_region should be called when geopandas is accessed
            with pytest.raises(ValueError, match="Error reading shapefile"):
                _ = region.geopandas


class TestRegionToGeopandas:
    """Test the to_geopandas() method for all regions.Region subclasses."""

    def test_centered_region_to_geopandas_single_box(self):
        """Test regions.CenteredRegion.geopandas with single bounding box."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        gdf = region.geopandas

        # Verify it's a GeoDataFrame
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        # Verify the polygon coordinates
        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, shapely.Polygon)

        # Check bounds (should be 40-50 lat, -125 to -115 lon after conversion)
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted)

    def test_centered_region_to_geopandas_tuple_box(self):
        """Test regions.CenteredRegion.geopandas with tuple bounding box."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        gdf = region.geopandas

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
        """Test regions.BoundingBoxRegion.geopandas."""
        region = regions.BoundingBoxRegion.create_region(
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
        assert isinstance(polygon, shapely.Polygon)

        # Check bounds (longitude should be converted to -180 to 180)
        bounds = polygon.bounds
        assert abs(bounds[1] - 40.0) < 0.001  # min lat
        assert abs(bounds[3] - 50.0) < 0.001  # max lat
        assert abs(bounds[0] - (-125.0)) < 0.001  # min lon (converted to -180 to 180)
        assert abs(bounds[2] - (-115.0)) < 0.001  # max lon (converted to -180 to 180)

    def test_shapefile_region_to_geopandas(self):
        """Test regions.ShapefileRegion.geopandas."""
        # Create a mock GeoDataFrame
        mock_polygon = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with mock.patch("geopandas.read_file", return_value=mock_gdf):
            region = regions.ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            gdf = region.geopandas

            # Should return the same GeoDataFrame that was read
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 1
            assert gdf.geometry.iloc[0] == mock_polygon

    def test_region_to_geopandas_edge_cases(self):
        """Test geopandas with edge cases."""
        # Test regions.CenteredRegion at poles
        polar_region = regions.CenteredRegion.create_region(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        polar_gdf = polar_region.geopandas
        assert isinstance(polar_gdf, gpd.GeoDataFrame)

        # Test regions.CenteredRegion crossing the dateline
        dateline_region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=175.0, bounding_box_degrees=10.0
        )
        dateline_gdf = dateline_region.geopandas
        assert isinstance(dateline_gdf, gpd.GeoDataFrame)

        # Test with very small bounding box
        small_region = regions.BoundingBoxRegion.create_region(
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
        region_neg = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        gdf_neg = region_neg.geopandas
        bounds_neg = gdf_neg.geometry.iloc[0].bounds
        assert bounds_neg[0] >= -180  # min lon should be >= -180
        assert bounds_neg[2] <= 180  # max lon should be <= 180

        # Test with positive longitudes
        region_pos = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=120.0, bounding_box_degrees=10.0
        )
        gdf_pos = region_pos.geopandas
        bounds_pos = gdf_pos.geometry.iloc[0].bounds
        assert bounds_pos[0] >= 0  # min lon should be >= 0
        assert bounds_pos[2] >= 0  # max lon should be >= 0


class TestRegionMask:
    """Test the mask() method for all regions.Region subclasses."""

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
        """Test regions.CenteredRegion.mask() method."""
        region = regions.CenteredRegion.create_region(
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
        """Test regions.BoundingBoxRegion.mask() method."""
        region = regions.BoundingBoxRegion.create_region(
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
        """Test regions.ShapefileRegion.mask() method."""
        mock_polygon = shapely.Polygon(
            [(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)]
        )
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with mock.patch("geopandas.read_file", return_value=mock_gdf):
            region = regions.ShapefileRegion.create_region(
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
        centered = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        bbox = regions.BoundingBoxRegion.create_region(
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
    """Test that all region types properly inherit from regions.Region."""

    def test_region_inheritance(self):
        """Test that all region types properly inherit from regions.Region."""
        # Test regions.CenteredRegion
        centered = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        assert isinstance(centered, regions.Region)

        # Test regions.BoundingBoxRegion
        bbox = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        assert isinstance(bbox, regions.Region)

        # Test regions.ShapefileRegion
        with mock.patch("geopandas.read_file", return_value=mock.Mock()):
            shapefile = regions.ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )
            assert isinstance(shapefile, regions.Region)

    def test_region_methods_consistency(self):
        """Test that all region types have consistent method behavior."""
        # Create regions with similar coverage
        centered = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )
        bbox = regions.BoundingBoxRegion.create_region(
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
        """Test creating a regions.CenteredRegion via factory function."""
        region = regions.map_to_create_region(
            {
                "type": "centered_region",
                "parameters": {
                    "latitude": 45.0,
                    "longitude": -120.0,
                    "bounding_box_degrees": 10.0,
                },
            }
        )
        assert isinstance(region, regions.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_create_centered_region_with_tuple(self):
        """Test creating a regions.CenteredRegion with tuple bounding box."""
        region = regions.map_to_create_region(
            {
                "type": "centered_region",
                "parameters": {
                    "latitude": 45.0,
                    "longitude": -120.0,
                    "bounding_box_degrees": (5.0, 10.0),
                },
            }
        )
        assert isinstance(region, regions.CenteredRegion)
        assert region.bounding_box_degrees == (5.0, 10.0)

    def test_create_bounding_box_region(self):
        """Test creating a regions.BoundingBoxRegion via factory function."""
        region = regions.map_to_create_region(
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
        assert isinstance(region, regions.BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0
        assert region.longitude_min == -125.0
        assert region.longitude_max == -115.0

    def test_create_shapefile_region(self):
        """Test creating a regions.ShapefileRegion via factory function."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.return_value = mock.Mock()
            region = regions.map_to_create_region(
                {
                    "type": "shapefile_region",
                    "parameters": {"shapefile_path": "/path/to/shapefile.shp"},
                }
            )
            assert isinstance(region, regions.ShapefileRegion)
            assert region.shapefile_path == Path("/path/to/shapefile.shp")

    def test_create_region_invalid_parameters(self):
        """Test create_region with invalid parameter combinations."""
        # Missing required parameters for regions.CenteredRegion
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            regions.map_to_create_region(
                {
                    "type": "centered_region",
                    "parameters": {"latitude": 45.0, "longitude": -120.0},
                }
            )
            # Missing bounding_box_degrees

        # Missing required parameters for regions.BoundingBoxRegion
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'longitude_max'"
        ):
            regions.map_to_create_region(
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
            regions.map_to_create_region(
                {
                    "type": "centered_region",
                    "parameters": {"latitude": 45.0, "longitude_min": -125.0},
                }
            )

    def test_create_region_priority_order(self):
        """Test that shapefile_path takes priority over other parameters."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.return_value = mock.Mock()
            region = regions.map_to_create_region(
                {
                    "type": "shapefile_region",
                    "parameters": {
                        "shapefile_path": "/path/to/shapefile.shp",
                    },
                }
            )
            # Should create regions.ShapefileRegion, not regions.CenteredRegion
            assert isinstance(region, regions.ShapefileRegion)


class TestMapToCreateRegion:
    """Test the regions.map_to_create_region function."""

    def test_map_to_create_region_centered(self):
        """Test mapping dictionary to regions.CenteredRegion."""
        kwargs = {
            "type": "centered_region",
            "parameters": {
                "latitude": 45.0,
                "longitude": -120.0,
                "bounding_box_degrees": 10.0,
            },
        }
        region = regions.map_to_create_region(kwargs)
        assert isinstance(region, regions.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_map_to_create_region_bounding_box(self):
        """Test mapping dictionary to regions.BoundingBoxRegion."""
        kwargs = {
            "type": "bounded_region",
            "parameters": {
                "latitude_min": 40.0,
                "latitude_max": 50.0,
                "longitude_min": -125.0,
                "longitude_max": -115.0,
            },
        }
        region = regions.map_to_create_region(kwargs)
        assert isinstance(region, regions.BoundingBoxRegion)
        assert region.latitude_min == 40.0
        assert region.latitude_max == 50.0

    def test_map_to_create_region_shapefile(self):
        """Test mapping dictionary to regions.ShapefileRegion."""
        with mock.patch("geopandas.read_file") as mock_read:
            mock_read.return_value = mock.Mock()
            kwargs = {
                "type": "shapefile_region",
                "parameters": {"shapefile_path": "/path/to/shapefile.shp"},
            }
            region = regions.map_to_create_region(kwargs)
            assert isinstance(region, regions.ShapefileRegion)
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
        assert isinstance(polygon, shapely.Polygon)

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
        # Should be a Multishapely.Polygon for antimeridian crossing
        assert isinstance(geometry, shapely.MultiPolygon)

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
        assert isinstance(polygon, shapely.Polygon)

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
        # Should be a Multishapely.Polygon
        assert isinstance(geometry, shapely.MultiPolygon)
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
        # Should be a single shapely.Polygon (no antimeridian crossing)
        assert isinstance(geometry, shapely.Polygon)

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
        assert isinstance(geometry, shapely.Polygon)

    def test_small_region(self):
        """Test with a very small region."""
        from extremeweatherbench.regions import _create_geopandas_from_bounds

        gdf = _create_geopandas_from_bounds(
            longitude_min=45.0, longitude_max=45.1, latitude_min=40.0, latitude_max=40.1
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 1

        polygon = gdf.geometry.iloc[0]
        assert isinstance(polygon, shapely.Polygon)

        bounds = polygon.bounds
        assert abs(bounds[0] - 45.0) < 0.001
        assert abs(bounds[2] - 45.1) < 0.001


class TestGetBoundingCoordinates:
    """Test the get_bounding_coordinates method for all regions.Region subclasses."""

    def test_centered_region_bounding_coordinates(self):
        """Test regions.CenteredRegion.get_bounding_coordinates method."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.get_bounding_coordinates

        # Check that it's a named tuple with correct attributes
        assert hasattr(coords, "longitude_min")
        assert hasattr(coords, "latitude_min")
        assert hasattr(coords, "longitude_max")
        assert hasattr(coords, "latitude_max")

        # Check coordinate values (longitude should be converted to -180 to 180 range)
        assert abs(coords.latitude_min - 40.0) < 0.001
        assert abs(coords.latitude_max - 50.0) < 0.001
        assert abs(coords.longitude_min - (-125.0)) < 0.001
        assert abs(coords.longitude_max - (-115.0)) < 0.001

    def test_centered_region_bounding_coordinates_tuple_box(self):
        """Test regions.CenteredRegion.get_bounding_coordinates with tuple bounding
        box."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=(5.0, 10.0)
        )

        coords = region.get_bounding_coordinates

        # Check coordinate values (should be 42.5-47.5 lat, -125 to -115 lon)
        assert abs(coords.latitude_min - 42.5) < 0.001
        assert abs(coords.latitude_max - 47.5) < 0.001
        assert abs(coords.longitude_min - (-125.0)) < 0.001
        assert abs(coords.longitude_max - (-115.0)) < 0.001

    def test_bounding_box_region_bounding_coordinates(self):
        """Test regions.BoundingBoxRegion.get_bounding_coordinates method."""
        region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        coords = region.get_bounding_coordinates

        # Check coordinate values
        assert abs(coords.latitude_min - 40.0) < 0.001
        assert abs(coords.latitude_max - 50.0) < 0.001
        assert abs(coords.longitude_min - (-125.0)) < 0.001
        assert abs(coords.longitude_max - (-115.0)) < 0.001

    def test_shapefile_region_bounding_coordinates(self):
        """Test regions.ShapefileRegion.get_bounding_coordinates method."""
        # Create a mock polygon with known bounds
        mock_polygon = shapely.Polygon(
            [(240, 40), (250, 40), (250, 50), (240, 50), (240, 40)]
        )
        mock_gdf = gpd.GeoDataFrame(geometry=[mock_polygon], crs="EPSG:4326")

        with mock.patch("geopandas.read_file", return_value=mock_gdf):
            region = regions.ShapefileRegion.create_region(
                shapefile_path="/path/to/shapefile.shp"
            )

            coords = region.get_bounding_coordinates

            # Check coordinate values (should match the polygon bounds)
            assert abs(coords.longitude_min - 240.0) < 0.001
            assert abs(coords.longitude_max - 250.0) < 0.001
            assert abs(coords.latitude_min - 40.0) < 0.001
            assert abs(coords.latitude_max - 50.0) < 0.001

    def test_bounding_coordinates_antimeridian_crossing(self):
        """Test get_bounding_coordinates with antimeridian crossing."""
        # Create a region that truly crosses the antimeridian (longitude spans > 180°)
        region = regions.CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=20.0,  # 165° to 185° crosses antimeridian
        )

        coords = region.get_bounding_coordinates

        # For antimeridian crossing regions, coordinates should span -180 to 180
        assert coords.longitude_min == -180.0
        assert coords.longitude_max == 180.0
        assert abs(coords.latitude_min - 35.0) < 0.001  # 45 - 10
        assert abs(coords.latitude_max - 55.0) < 0.001  # 45 + 10

    def test_bounding_coordinates_near_antimeridian_no_crossing(self):
        """Test get_bounding_coordinates for region near but not crossing
        antimeridian."""
        # Create a region that goes exactly to 180° but doesn't cross it
        region = regions.CenteredRegion.create_region(
            latitude=45.0,
            longitude=175.0,
            bounding_box_degrees=10.0,  # 170° to 180°, no crossing
        )

        coords = region.get_bounding_coordinates

        # Should be a single polygon from 170° to 180°
        assert abs(coords.longitude_min - 170.0) < 0.001
        assert abs(coords.longitude_max - 180.0) < 0.001
        assert abs(coords.latitude_min - 40.0) < 0.001
        assert abs(coords.latitude_max - 50.0) < 0.001

    def test_bounding_coordinates_longitude_conversion(self):
        """Test that bounding coordinates properly handle longitude conversion."""
        # Test with positive longitude that should be converted
        region = regions.CenteredRegion.create_region(
            latitude=45.0,
            longitude=200.0,
            bounding_box_degrees=10.0,  # 200° = -160°
        )

        coords = region.get_bounding_coordinates

        # Coordinates should be in -180 to 180 range
        assert coords.longitude_min >= -180
        assert coords.longitude_max <= 180

        # Test specific expected values
        # Center at 200° (converted to -160°), ±5° box should be -165° to -155°
        expected_min_lon = -165.0
        expected_max_lon = -155.0
        assert abs(coords.longitude_min - expected_min_lon) < 0.001
        assert abs(coords.longitude_max - expected_max_lon) < 0.001

    def test_bounding_coordinates_edge_cases(self):
        """Test get_bounding_coordinates with edge cases."""
        # Test very small region
        small_region = regions.BoundingBoxRegion.create_region(
            latitude_min=44.99,
            latitude_max=45.01,
            longitude_min=-120.01,
            longitude_max=-119.99,
        )
        coords = small_region.get_bounding_coordinates
        assert coords.latitude_max > coords.latitude_min
        assert coords.longitude_max > coords.longitude_min

        # Test polar region
        polar_region = regions.CenteredRegion.create_region(
            latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
        )
        polar_coords = polar_region.get_bounding_coordinates
        assert polar_coords.latitude_min >= -90
        assert polar_coords.latitude_max <= 90

    def test_bounding_coordinates_return_type(self):
        """Test that get_bounding_coordinates returns correct type."""
        region = regions.CenteredRegion.create_region(
            latitude=45.0, longitude=-120.0, bounding_box_degrees=10.0
        )

        coords = region.get_bounding_coordinates

        # Should return a tuple
        assert isinstance(coords, tuple)
        assert len(coords) == 4

        # Should have named attributes
        assert hasattr(coords, "longitude_min")
        assert hasattr(coords, "latitude_min")
        assert hasattr(coords, "longitude_max")
        assert hasattr(coords, "latitude_max")

        # All values should be floats
        assert isinstance(coords.longitude_min, float)
        assert isinstance(coords.latitude_min, float)
        assert isinstance(coords.longitude_max, float)
        assert isinstance(coords.latitude_max, float)


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

        # Test that regions.map_to_create_region works correctly
        region = regions.map_to_create_region(region_data)

        assert isinstance(region, regions.CenteredRegion)
        assert region.latitude == 45.0
        assert region.longitude == -120.0
        assert region.bounding_box_degrees == 10.0

    def test_region_with_individual_case_integration(self):
        """Test that regions work correctly with IndividualCase."""
        from extremeweatherbench import cases

        region = regions.CenteredRegion.create_region(
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


class TestRegionGeometricOperations:
    """Test geometric operations on regions."""

    def test_region_intersects(self):
        """Test the intersects method."""
        # Create two overlapping regions
        region1 = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        region2 = regions.BoundingBoxRegion.create_region(
            latitude_min=45.0,
            latitude_max=55.0,
            longitude_min=-120.0,
            longitude_max=-110.0,
        )

        # They should intersect
        assert region1.intersects(region2)
        assert region2.intersects(region1)

        # Test non-overlapping regions
        region3 = regions.BoundingBoxRegion.create_region(
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
        large_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=60.0,
            longitude_min=-130.0,
            longitude_max=-110.0,
        )

        # Create a smaller region inside it
        small_region = regions.BoundingBoxRegion.create_region(
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
        region1 = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )
        region2 = regions.BoundingBoxRegion.create_region(
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
        region3 = regions.BoundingBoxRegion.create_region(
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
    """Test the  regions.RegionSubsetter class."""

    @pytest.fixture
    def target_region(self):
        """Create a target region for subsetting."""
        return regions.BoundingBoxRegion.create_region(
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
            location=regions.BoundingBoxRegion.create_region(
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
            location=regions.BoundingBoxRegion.create_region(
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
            location=regions.BoundingBoxRegion.create_region(
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
        """Test  regions.RegionSubsetter initialization with Region object."""
        subsetter = regions.RegionSubsetter(
            region=target_region, method="intersects", percent_threshold=0.5
        )

        assert subsetter.region == target_region
        assert subsetter.method == "intersects"
        assert subsetter.percent_threshold == 0.5

    def test_subsetter_initialization_with_dict(self):
        """Test  regions.RegionSubsetter initialization with dictionary."""
        region_dict = {
            "latitude_min": 40.0,
            "latitude_max": 50.0,
            "longitude_min": -125.0,
            "longitude_max": -115.0,
        }

        subsetter = regions.RegionSubsetter(
            region=region_dict, method="percent", percent_threshold=0.75
        )

        assert isinstance(subsetter.region, regions.BoundingBoxRegion)
        assert subsetter.method == "percent"
        assert subsetter.percent_threshold == 0.75

    def test_subset_case_collection_intersects(self, target_region, sample_cases):
        """Test subsetting with intersects method."""
        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

        subset_cases = subsetter.subset_case_collection(sample_cases)

        # Should include intersecting and contained cases, but not outside
        assert len(subset_cases.cases) == 2
        case_ids = {case.case_id_number for case in subset_cases.cases}
        assert case_ids == {1, 2}  # intersecting and contained

    def test_subset_case_collection_all(self, target_region, sample_cases):
        """Test subsetting with all method."""
        subsetter = regions.RegionSubsetter(region=target_region, method="all")

        subset_cases = subsetter.subset_case_collection(sample_cases)

        # Should only include contained case
        assert len(subset_cases.cases) == 1
        assert subset_cases.cases[0].case_id_number == 2  # contained case

    def test_subset_case_collection_percent(self, target_region, sample_cases):
        """Test subsetting with percent method."""
        subsetter = regions.RegionSubsetter(
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
        low_threshold_subsetter = regions.RegionSubsetter(
            region=target_region, method="percent", percent_threshold=0.1
        )

        low_threshold_cases = low_threshold_subsetter.subset_case_collection(
            sample_cases
        )

        # High threshold - should include fewer cases
        high_threshold_subsetter = regions.RegionSubsetter(
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

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

        subset_results = regions.subset_results_to_region(
            subsetter, results_df, sample_cases
        )

        # Should only include results for cases 1 and 2 (intersecting and contained)
        assert len(subset_results) == 4  # 2 cases * 2 metrics
        case_ids = set(subset_results["case_id_number"])
        assert case_ids == {1, 2}

    def test_invalid_method_raises_error(self, target_region):
        """Test that invalid method raises ValueError."""
        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

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
            location=regions.BoundingBoxRegion.create_region(
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
            location=regions.BoundingBoxRegion.create_region(
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
        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subset_cases = regions.subset_cases_to_region(
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

        subset_cases = regions.subset_cases_to_region(
            sample_case_collection, region_dict, method="intersects"
        )

        assert len(subset_cases.cases) == 1
        assert subset_cases.cases[0].case_id_number == 1

    def test_subset_cases_to_region_with_percent_method(self, sample_case_collection):
        """Test subset_cases_to_region with percent method."""
        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subset_cases = regions.subset_cases_to_region(
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
        large_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=60.0,
            longitude_min=-130.0,
            longitude_max=-100.0,
        )

        subset_cases = regions.subset_cases_to_region(
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

        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

        subset_results = regions.subset_results_to_region(
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
        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=40.0,
            latitude_max=50.0,
            longitude_min=-125.0,
            longitude_max=-115.0,
        )

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

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
            location=regions.BoundingBoxRegion.create_region(
                latitude_min=45.0,
                latitude_max=45.1,
                longitude_min=-120.0,
                longitude_max=-119.9,
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[tiny_case])

        # Create a target region that should intersect
        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=44.9,
            latitude_max=45.2,
            longitude_min=-120.1,
            longitude_max=-119.8,
        )

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

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
            location=regions.CenteredRegion.create_region(
                latitude=45.0,
                longitude=175.0,  # Near dateline
                bounding_box_degrees=10.0,
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[dateline_case])

        # Create a target region that might cross antimeridian
        target_region = regions.CenteredRegion.create_region(
            latitude=45.0,
            longitude=180.0,  # At dateline
            bounding_box_degrees=15.0,
        )

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

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
            location=regions.CenteredRegion.create_region(
                latitude=85.0, longitude=0.0, bounding_box_degrees=10.0
            ),
            event_type="test",
        )

        case_collection = cases.IndividualCaseCollection(cases=[polar_case])

        # Create a target region at high latitudes
        target_region = regions.BoundingBoxRegion.create_region(
            latitude_min=80.0,
            latitude_max=90.0,
            longitude_min=-30.0,
            longitude_max=30.0,
        )

        subsetter = regions.RegionSubsetter(region=target_region, method="intersects")

        # Should handle polar coordinates gracefully
        subset_cases = subsetter.subset_case_collection(case_collection)
        assert isinstance(subset_cases, cases.IndividualCaseCollection)
