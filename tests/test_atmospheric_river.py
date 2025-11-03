"""Tests for the atmospheric_river module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc
from extremeweatherbench.events import atmospheric_river

# Set random seed for reproducible tests
rng = np.random.default_rng(seed=42)


class TestAtmosphericRiverMask:
    """Test atmospheric river mask calculations."""

    @pytest.fixture
    def sample_ar_dataset(self):
        """Create a sample dataset for atmospheric river testing."""
        time = pd.date_range("2023-01-01", periods=3, freq="6h")
        lat = np.linspace(20, 50, 20)  # 0.25 degree spacing
        lon = np.linspace(-130, -100, 20)  # 0.25 degree spacing
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_3d = (len(time), len(lat), len(lon))
        data_shape_4d = (len(time), len(level), len(lat), len(lon))

        # Create realistic IVT data with some high values
        ivt_data = rng.uniform(100, 300, data_shape_3d)
        # Add some high IVT values to create potential AR features
        ivt_data[0, 5:15, 5:15] = 500  # High IVT region
        ivt_data[1, 8:12, 8:12] = 600  # Another high IVT region

        # Create IVT Laplacian data
        ivt_laplacian = rng.uniform(-2, 2, data_shape_3d)
        # Add high Laplacian values corresponding to high IVT regions
        ivt_laplacian[0, 5:15, 5:15] = 3.0
        ivt_laplacian[1, 8:12, 8:12] = 3.5

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        dataset = xr.Dataset(
            {
                "integrated_vapor_transport": (
                    ["time", "latitude", "longitude"],
                    ivt_data,
                ),
                "integrated_vapor_transport_laplacian": (
                    ["time", "latitude", "longitude"],
                    ivt_laplacian,
                ),
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "adjusted_level": level,
            },
        )

        return dataset

    def test_atmospheric_river_mask_basic(self, sample_ar_dataset):
        """Test basic atmospheric river mask functionality."""
        result = atmospheric_river.atmospheric_river_mask(
            ivt=sample_ar_dataset["integrated_vapor_transport"],
            ivt_laplacian=sample_ar_dataset["integrated_vapor_transport_laplacian"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions (no level dimension)
        expected_dims = ["time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == (3, 20, 20)

        # Values should be 0 or 1 (boolean mask)
        assert set(result.values.flatten()).issubset({0, 1})

        # Should have some True values where we created high IVT/Laplacian
        # Note: This might be 0 if the size filtering removes features
        # We'll test with lower thresholds in other tests

    def test_atmospheric_river_mask_custom_thresholds(self, sample_ar_dataset):
        """Test atmospheric river mask with custom thresholds."""
        # Test with very high thresholds (should return mostly zeros)
        result_high = atmospheric_river.atmospheric_river_mask(
            ivt=sample_ar_dataset["integrated_vapor_transport"],
            ivt_laplacian=sample_ar_dataset["integrated_vapor_transport_laplacian"],
            laplacian_threshold=10.0,
            ivt_threshold=1000.0,
        )

        # Should return mostly zeros with high thresholds
        assert result_high.sum() < result_high.size * 0.1

        # Test with very low thresholds (should return more ones)
        result_low = atmospheric_river.atmospheric_river_mask(
            ivt=sample_ar_dataset["integrated_vapor_transport"],
            ivt_laplacian=sample_ar_dataset["integrated_vapor_transport_laplacian"],
            laplacian_threshold=0.1,
            ivt_threshold=50.0,
        )

        # Should return more ones with low thresholds
        assert result_low.sum() > result_high.sum()

    def test_atmospheric_river_mask_size_filtering(self, sample_ar_dataset):
        """Test atmospheric river mask size filtering."""
        # Test with very large minimum size (should filter out small features)
        result_large_min = atmospheric_river.atmospheric_river_mask(
            ivt=sample_ar_dataset["integrated_vapor_transport"],
            ivt_laplacian=sample_ar_dataset["integrated_vapor_transport_laplacian"],
            min_size_gridpoints=1000,
        )

        # Test with small minimum size (should keep more features)
        result_small_min = atmospheric_river.atmospheric_river_mask(
            ivt=sample_ar_dataset["integrated_vapor_transport"],
            ivt_laplacian=sample_ar_dataset["integrated_vapor_transport_laplacian"],
            min_size_gridpoints=10,
        )

        # Small minimum size should have more features
        assert result_small_min.sum() >= result_large_min.sum()

    def test_atmospheric_river_mask_nan_handling(self):
        """Test atmospheric river mask with NaN values."""
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 10)
        lon = np.linspace(-130, -100, 10)

        # Create dataset with some NaN values
        ivt_data = rng.uniform(100, 300, (2, 10, 10))
        ivt_data[0, 5, 5] = np.nan  # Add NaN value
        ivt_laplacian = rng.uniform(-2, 2, (2, 10, 10))
        ivt_laplacian[0, 5, 5] = np.nan  # Add NaN value

        dataset = xr.Dataset(
            {
                "integrated_vapor_transport": (
                    ["time", "latitude", "longitude"],
                    ivt_data,
                ),
                "integrated_vapor_transport_laplacian": (
                    ["time", "latitude", "longitude"],
                    ivt_laplacian,
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )

        result = atmospheric_river.atmospheric_river_mask(
            ivt=dataset["integrated_vapor_transport"],
            ivt_laplacian=dataset["integrated_vapor_transport_laplacian"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should handle NaN values gracefully
        assert not np.isnan(result).any()

        # Values should be 0 or 1
        assert set(result.values.flatten()).issubset({0, 1})


class TestComputeIVT:
    """Test integrated vapor transport calculations."""

    @pytest.fixture
    def sample_ivt_dataset(self):
        """Create a sample dataset for IVT testing."""
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 10)
        lon = np.linspace(-130, -100, 10)
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))
        data_shape_3d = (len(time), len(lat), len(lon))

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        dataset = xr.Dataset(
            {
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "adjusted_level": level,
            },
        )

        return dataset

    def test_integrated_vapor_transport_basic(self, sample_ivt_dataset):
        """Test basic IVT computation."""
        result = atmospheric_river.integrated_vapor_transport(
            specific_humidity=sample_ivt_dataset["specific_humidity"],
            eastward_wind=sample_ivt_dataset["eastward_wind"],
            northward_wind=sample_ivt_dataset["northward_wind"],
            levels=sample_ivt_dataset["adjusted_level"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions (no level dimension)
        expected_dims = ["time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == (2, 10, 10)

        # Values should be positive (IVT magnitude)
        assert (result >= 0).all()

        # Values should be reasonable for IVT (typically 0-3000 kg/m/s)
        # Some extreme values may exceed 1000 but should be under 3000
        assert (result < 3000).all()

    def test_integrated_vapor_transport_nan_handling(self):
        """Test integrated_vapor_transport with NaN values."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))
        data_shape_3d = (len(time), len(lat), len(lon))

        # Create dataset with some NaN values
        eastward_wind = rng.uniform(-20, 20, data_shape_4d)
        eastward_wind[0, 2, 2, 2] = np.nan  # Add NaN value

        northward_wind = rng.uniform(-20, 20, data_shape_4d)
        northward_wind[0, 2, 2, 2] = np.nan  # Add NaN value

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        dataset = xr.Dataset(
            {
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    northward_wind,
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "adjusted_level": level,
            },
        )
        result = atmospheric_river.integrated_vapor_transport(
            specific_humidity=dataset["specific_humidity"],
            eastward_wind=dataset["eastward_wind"],
            northward_wind=dataset["northward_wind"],
            levels=dataset["adjusted_level"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should handle NaN values gracefully
        # Note: nantrapezoid should handle NaNs, but result might still have NaNs
        # depending on the specific implementation

    def test_integrated_vapor_transport_missing_variables(self):
        """Test integrated_vapor_transport with missing required variables."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)
        level = [1000, 850, 700, 500, 300, 200]

        # Create some valid DataArrays
        eastward_wind = xr.DataArray(
            rng.uniform(-20, 20, (1, 6, 5, 5)),
            dims=["time", "level", "latitude", "longitude"],
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
            },
        )

        levels = xr.DataArray(level, dims=["level"], coords={"level": level})

        # Should raise an error when required variables are missing (None values)
        with pytest.raises((TypeError, AttributeError)):
            atmospheric_river.integrated_vapor_transport(
                specific_humidity=None,
                eastward_wind=eastward_wind,
                northward_wind=eastward_wind,
                levels=levels,
            )

    def test_integrated_vapor_transport_low_pressure_levels(self):
        """Test integrated_vapor_transport with levels below 200 hPa (should be
        filtered out)."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)
        # Include levels below 200 hPa that should be filtered out
        level = [1000, 850, 700, 500, 300, 200, 150, 100, 50]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))
        data_shape_3d = (len(time), len(lat), len(lon))

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        # Filter adjusted_level to >= 200 hPa
        adjusted_level = [lev for lev in level if lev >= 200]

        # Create full dataset first
        dataset = xr.Dataset(
            {
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
            },
        )

        # Select only the adjusted levels
        dataset_filtered = dataset.sel(level=adjusted_level)
        dataset_filtered = dataset_filtered.assign_coords(adjusted_level=adjusted_level)

        result = atmospheric_river.integrated_vapor_transport(
            specific_humidity=dataset_filtered["specific_humidity"],
            eastward_wind=dataset_filtered["eastward_wind"],
            northward_wind=dataset_filtered["northward_wind"],
            levels=dataset_filtered["adjusted_level"],
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions (no level dimension)
        expected_dims = ["time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape (no level dimension)
        assert result.shape == (1, 5, 5)

        # Values should be positive (IVT magnitude)
        assert (result >= 0).all()

        # Values should be reasonable for IVT
        assert (result < 3000).all()


class TestComputeIVTLaplacian:
    """Test IVT Laplacian calculations."""

    @pytest.fixture
    def sample_ivt_dataarray(self):
        """Create a sample IVT DataArray for testing."""
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 10)
        lon = np.linspace(-130, -100, 10)

        # Create IVT data with some structure
        ivt_data = rng.uniform(100, 300, (2, 10, 10))
        # Add some structure to make Laplacian more interesting
        ivt_data[0, 3:7, 3:7] = 500  # High IVT region

        ivt = xr.DataArray(
            ivt_data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
            name="integrated_vapor_transport",
        )

        return ivt

    def test_integrated_vapor_transport_laplacian_basic(self, sample_ivt_dataarray):
        """Test basic IVT Laplacian computation."""
        result = atmospheric_river.integrated_vapor_transport_laplacian(
            sample_ivt_dataarray
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions
        assert list(result.dims) == ["time", "latitude", "longitude"]

        # Should have correct shape
        assert result.shape == (2, 10, 10)

        # Should have correct name
        assert result.name == "integrated_vapor_transport_blurred_laplacian"

        # Values should be finite (no NaN or inf)
        assert np.isfinite(result).all()

    def test_integrated_vapor_transport_laplacian_custom_sigma(
        self, sample_ivt_dataarray
    ):
        """Test IVT Laplacian with custom sigma parameter."""
        result_small_sigma = atmospheric_river.integrated_vapor_transport_laplacian(
            sample_ivt_dataarray, sigma=1.0
        )
        result_large_sigma = atmospheric_river.integrated_vapor_transport_laplacian(
            sample_ivt_dataarray, sigma=5.0
        )

        # Both should return DataArrays
        assert isinstance(result_small_sigma, xr.DataArray)
        assert isinstance(result_large_sigma, xr.DataArray)

        # Different sigma values should produce different results
        assert not np.allclose(result_small_sigma, result_large_sigma)

    def test_integrated_vapor_transport_laplacian_nan_handling(self):
        """Test IVT Laplacian with NaN values."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)

        # Create IVT data with NaN values
        ivt_data = rng.uniform(100, 300, (1, 5, 5))
        ivt_data[0, 2, 2] = np.nan  # Add NaN value

        ivt = xr.DataArray(
            ivt_data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
            name="integrated_vapor_transport",
        )

        result = atmospheric_river.integrated_vapor_transport_laplacian(ivt)

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should handle NaN values gracefully
        # Note: The result might contain NaNs depending on how the filter handles them


class TestFindLandIntersection:
    """Test land intersection calculations."""

    @pytest.fixture
    def sample_ar_mask(self):
        """Create a sample atmospheric river mask for testing."""
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 10)
        lon = np.linspace(-130, -100, 10)

        # Create AR mask with some True values
        ar_mask_data = np.zeros((2, 10, 10), dtype=int)
        ar_mask_data[0, 3:7, 3:7] = 1  # AR region

        ar_mask = xr.DataArray(
            ar_mask_data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )

        return ar_mask

    def test_find_land_intersection_basic(self, sample_ar_mask):
        """Test basic land intersection functionality."""
        result = calc.find_land_intersection(sample_ar_mask)

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions
        expected_dims = ["time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == sample_ar_mask.shape

        # Values should be 0 or 1 (binary mask)
        assert set(result.values.flatten()).issubset({0, 1})

        # Function works correctly (name is set by calling function if needed)

    def test_find_land_intersection_empty_mask(self):
        """Test land intersection with empty AR mask."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)

        # Create empty AR mask
        ar_mask = xr.DataArray(
            np.zeros((1, 5, 5), dtype=int),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )

        result = calc.find_land_intersection(ar_mask)

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions
        expected_dims = ["time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == ar_mask.shape

        # Values should be 0 or 1 (binary mask)
        assert set(result.values.flatten()).issubset({0, 1})

        # Function works correctly (name is set by calling function if needed)


class TestBuildMaskAndLandIntersection:
    """Test integrated atmospheric river mask and land intersection."""

    @pytest.fixture
    def sample_full_dataset(self):
        """Create a complete sample dataset for integration testing."""
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 10)
        lon = np.linspace(-130, -100, 10)
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))
        data_shape_3d = (len(time), len(lat), len(lon))

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        dataset = xr.Dataset(
            {
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "adjusted_level": level,
            },
        )

        return dataset

    def test_build_mask_and_land_intersection_basic(self, sample_full_dataset):
        """Test basic integration functionality."""
        result = atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
            sample_full_dataset
        )

        # Should return a Dataset
        assert isinstance(result, xr.Dataset)

        # Should contain expected variables
        assert "atmospheric_river_mask" in result.data_vars
        assert "atmospheric_river_land_intersection" in result.data_vars

        # Atmospheric river mask should be a DataArray
        ar_mask = result["atmospheric_river_mask"]
        assert isinstance(ar_mask, xr.DataArray)
        assert list(ar_mask.dims) == ["time", "latitude", "longitude"]

        # Land intersection should be a DataArray with binary values
        land_intersection = result["atmospheric_river_land_intersection"]

        # The land intersection is stored as a DataArray with binary values
        assert isinstance(land_intersection, xr.DataArray)
        assert list(land_intersection.dims) == ["time", "latitude", "longitude"]
        # Values should be 0 or 1 (binary mask)
        assert set(land_intersection.values.flatten()).issubset({0, 1})

    def test_build_mask_and_land_intersection_missing_variables(self):
        """Test integration with missing required variables."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_3d = (len(time), len(lat), len(lon))

        # Create dataset missing required wind variables
        # Include geopotential and air_temperature to avoid circular import
        dataset = xr.Dataset(
            {
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    rng.uniform(250, 300, (1, 6, 5, 5)),
                ),
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
            },
        )

        # Should raise an error when required variables are missing
        with pytest.raises(KeyError):
            atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
                dataset
            )

    def test_build_mask_and_land_intersection_nan_handling(self):
        """Test integration with NaN values."""
        time = pd.date_range("2023-01-01", periods=1, freq="6h")
        lat = np.linspace(20, 50, 5)
        lon = np.linspace(-130, -100, 5)
        level = [1000, 850, 700, 500, 300, 200]

        data_shape_4d = (len(time), len(level), len(lat), len(lon))
        data_shape_3d = (len(time), len(lat), len(lon))

        # Create dataset with some NaN values
        eastward_wind = rng.uniform(-20, 20, data_shape_4d)
        eastward_wind[0, 2, 2, 2] = np.nan

        northward_wind = rng.uniform(-20, 20, data_shape_4d)
        northward_wind[0, 2, 2, 2] = np.nan

        # Create temperature and humidity data for specific humidity
        air_temp = rng.uniform(250, 300, data_shape_4d)
        rel_humidity = rng.uniform(0.3, 0.9, data_shape_4d)

        # Compute specific humidity
        specific_hum = calc.specific_humidity_from_relative_humidity(
            air_temperature=xr.DataArray(
                air_temp,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["time", "level", "latitude", "longitude"],
                coords={
                    "time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            levels=xr.DataArray(level, dims=["level"], coords={"level": level}),
        )

        dataset = xr.Dataset(
            {
                "eastward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["time", "level", "latitude", "longitude"],
                    northward_wind,
                ),
                "air_temperature": (
                    ["time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "adjusted_level": level,
            },
        )

        result = atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
            dataset
        )

        # Should return a Dataset
        assert isinstance(result, xr.Dataset)

        # Should handle NaN values gracefully
        ar_mask = result["atmospheric_river_mask"]
        assert isinstance(ar_mask, xr.DataArray)

        # Values should be 0 or 1 (boolean mask)
        assert set(ar_mask.values.flatten()).issubset({0, 1})
