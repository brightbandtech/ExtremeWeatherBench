"""Tests for the atmospheric_river module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import calc
from extremeweatherbench.events import atmospheric_river
from tests.fixtures.events import make_ar_input_dataset
from tests.fixtures.laziness import assert_lazy

# Set random seed for reproducible tests
rng = np.random.default_rng(seed=42)


class TestAtmosphericRiverVariables:
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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "latitude", "longitude"],
                    ivt_data,
                ),
                "integrated_vapor_transport_laplacian": (
                    ["valid_time", "latitude", "longitude"],
                    ivt_laplacian,
                ),
                "eastward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
        expected_dims = ["valid_time", "latitude", "longitude"]
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

    def test_atmospheric_river_mask_4d_input(self):
        """Regression test: 4D (lead_time, valid_time, lat, lon) input must not
        raise IndexError in _binary_dilation_ufunc."""
        lead = [0, 6, 12]
        time = pd.date_range("2023-01-01", periods=2, freq="6h")
        lat = np.linspace(20, 50, 20)
        lon = np.linspace(-130, -100, 20)

        shape_4d = (len(lead), len(time), len(lat), len(lon))
        ivt_data = rng.uniform(100, 300, shape_4d)
        ivt_data[0, 0, 5:15, 5:15] = 500
        ivt_laplacian = rng.uniform(-2, 2, shape_4d)
        ivt_laplacian[0, 0, 5:15, 5:15] = 3.0

        ivt = xr.DataArray(
            ivt_data,
            dims=["lead_time", "valid_time", "latitude", "longitude"],
            coords={
                "lead_time": lead,
                "valid_time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )
        ivt_lap = xr.DataArray(
            ivt_laplacian,
            dims=["lead_time", "valid_time", "latitude", "longitude"],
            coords={
                "lead_time": lead,
                "valid_time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt,
            ivt_laplacian=ivt_lap,
        )

        assert isinstance(result, xr.DataArray)
        # All non-lat/lon dims should be preserved in the result
        assert "latitude" in result.dims
        assert "longitude" in result.dims
        assert set(result.values.flatten()).issubset({0, 1})

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
                    ["valid_time", "latitude", "longitude"],
                    ivt_data,
                ),
                "integrated_vapor_transport_laplacian": (
                    ["valid_time", "latitude", "longitude"],
                    ivt_laplacian,
                ),
            },
            coords={
                "valid_time": time,
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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions (no level dimension)
        expected_dims = ["valid_time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape
        assert result.shape == (2, 10, 10)

        # Values should be positive (IVT magnitude)
        assert (result >= 0).all()

        # There needs to be many > 0
        assert (result > 0).any()

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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "level", "latitude", "longitude"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    northward_wind,
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
            dims=["valid_time", "level", "latitude", "longitude"],
            coords={
                "valid_time": time,
                "latitude": lat,
                "longitude": lon,
                "level": level,
            },
        )

        # Should raise an error when required variables are missing (None values)
        with pytest.raises((TypeError, AttributeError)):
            atmospheric_river.integrated_vapor_transport(
                specific_humidity=None,
                eastward_wind=eastward_wind,
                northward_wind=eastward_wind,
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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
        )

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions (no level dimension)
        expected_dims = ["valid_time", "latitude", "longitude"]
        assert list(result.dims) == expected_dims

        # Should have correct shape (no level dimension)
        assert result.shape == (1, 5, 5)

        # Values should be positive (IVT magnitude)
        assert (result >= 0).all()

        # There needs to be many > 0
        assert (result > 0).any()

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
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": time,
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
        assert list(result.dims) == ["valid_time", "latitude", "longitude"]

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
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": time,
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
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": time,
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
        expected_dims = ["valid_time", "latitude", "longitude"]
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
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": time,
                "latitude": lat,
                "longitude": lon,
            },
        )

        result = calc.find_land_intersection(ar_mask)

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Should have correct dimensions
        expected_dims = ["valid_time", "latitude", "longitude"]
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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(-20, 20, data_shape_4d),
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
        assert list(ar_mask.dims) == ["valid_time", "latitude", "longitude"]

        # Land intersection should be a DataArray with binary values
        land_intersection = result["atmospheric_river_land_intersection"]

        # The land intersection is stored as a DataArray with binary values
        assert isinstance(land_intersection, xr.DataArray)
        assert list(land_intersection.dims) == ["valid_time", "latitude", "longitude"]
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
                    ["valid_time", "level", "latitude", "longitude"],
                    rng.uniform(250, 300, (1, 6, 5, 5)),
                ),
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
                    "level": level,
                    "latitude": lat,
                    "longitude": lon,
                },
            ),
            relative_humidity=xr.DataArray(
                rel_humidity,
                dims=["valid_time", "level", "latitude", "longitude"],
                coords={
                    "valid_time": time,
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
                    ["valid_time", "level", "latitude", "longitude"],
                    eastward_wind,
                ),
                "northward_wind": (
                    ["valid_time", "level", "latitude", "longitude"],
                    northward_wind,
                ),
                "air_temperature": (
                    ["valid_time", "level", "latitude", "longitude"],
                    air_temp,
                ),
                "relative_humidity": (
                    ["valid_time", "level", "latitude", "longitude"],
                    rel_humidity,
                ),
                "specific_humidity": specific_hum,
                "geopotential_at_surface": (
                    ["valid_time", "latitude", "longitude"],
                    rng.uniform(0, 1000, data_shape_3d),
                ),
            },
            coords={
                "valid_time": time,
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


class TestAtmosphericRiverLabelConnectivity:
    """Features connect along exactly one time-like axis.

    An AR keeps its identity as it evolves, so connectivity along the single
    time axis is wanted. Connecting across separate forecast initializations
    is not: two different forecasts are independent realizations, and merging
    them lets a feature that is too small in either one survive the size
    filter.
    """

    def test_features_connect_across_valid_time_in_an_analysis(self):
        """Two 36-cell blobs at consecutive valid times form one 72-cell feature."""
        ivt, lap = make_ar_input_dataset(
            time_dim="valid_time",
            n_time=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (1, 0, slice(10, 16), slice(10, 16)),
            ],
        )

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        assert int(result.sum()) == 72, (
            "blobs at consecutive valid times should merge into one feature "
            "large enough to survive the size filter"
        )

    def test_features_do_not_connect_across_separate_initializations(self):
        """The same 36-cell blob in two forecasts stays two small features.

        Merging them across init_time would produce 72 cells and wrongly pass
        a 50-cell size filter.
        """
        ivt, lap = make_ar_input_dataset(
            time_dim="lead_time",
            n_time=1,
            extra_dim="init_time",
            n_extra=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (0, 1, slice(10, 16), slice(10, 16)),
            ],
        )

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        assert int(result.sum()) == 0, (
            "a 36-cell feature is below the 50-cell threshold in each forecast "
            "separately and must not be rescued by merging across init_time"
        )

    def test_features_connect_across_lead_time_within_one_initialization(self):
        ivt, lap = make_ar_input_dataset(
            time_dim="lead_time",
            n_time=2,
            extra_dim="init_time",
            n_extra=2,
            blobs=[
                (0, 0, slice(10, 16), slice(10, 16)),
                (1, 0, slice(10, 16), slice(10, 16)),
            ],
        )

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()

        by_init = result.sum(dim=["lead_time", "latitude", "longitude"])
        assert int(by_init[0]) == 72
        assert int(by_init[1]) == 0


class TestAtmosphericRiverMaskLaziness:
    """The AR mask must stay in the dask graph instead of materializing."""

    def test_mask_from_dask_inputs_is_still_lazy(self):
        ivt, lap = make_ar_input_dataset(blobs=[(0, 0, slice(10, 30), slice(10, 30))])

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, min_size_gridpoints=50
        )

        assert_lazy(
            result,
            "ndimage.label on a lazy array pulls the whole volume into memory "
            "while the graph is being built",
        )


class TestAtmosphericRiverMaskOutput:
    """Pins how the threshold, size and latitude filters shape the mask."""

    def test_blob_above_all_criteria_is_retained(self):
        ivt, lap = make_ar_input_dataset(blobs=[(0, 0, slice(5, 25), slice(5, 25))])
        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()
        assert int(result.sum()) == 400
        assert set(np.unique(result.values)) <= {0, 1}

    def test_a_blob_exactly_at_the_size_threshold_is_kept(self):
        # A 7x7 blob is 49 gridpoints; the filter keeps features of at least
        # min_size_gridpoints, so 49 is the smallest surviving size.
        ivt, lap = make_ar_input_dataset(blobs=[(0, 0, slice(5, 12), slice(5, 12))])

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=49
        )

        assert int(result.sum()) == 49

    def test_a_blob_one_gridpoint_under_the_threshold_is_dropped(self):
        ivt, lap = make_ar_input_dataset(blobs=[(0, 0, slice(5, 12), slice(5, 12))])

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        )

        assert int(result.sum()) == 0

    def test_blob_below_the_size_threshold_is_dropped(self):
        ivt, lap = make_ar_input_dataset(blobs=[(0, 0, slice(5, 8), slice(5, 8))])
        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        ).compute()
        assert int(result.sum()) == 0

    def test_tropical_feature_is_dropped_on_mean_latitude(self):
        """A large feature centered below 15 degrees is not an AR."""
        lat = np.linspace(0.0, 39.0, 40)
        lon = np.linspace(-160.0, -121.0, 40)
        values = np.zeros((1, 40, 40))
        values[0, 0:14, 5:35] = 1.0  # latitudes 0 to ~13 degrees
        ivt = xr.DataArray(
            values * 800.0,
            dims=["valid_time", "latitude", "longitude"],
            coords={
                "valid_time": pd.date_range("2023-01-01", periods=1),
                "latitude": lat,
                "longitude": lon,
            },
        )
        lap = ivt / 160.0

        result = atmospheric_river.atmospheric_river_mask(
            ivt=ivt, ivt_laplacian=lap, dilation_radius=0, min_size_gridpoints=50
        )
        assert int(np.asarray(result).sum()) == 0

    def test_chunked_and_unchunked_inputs_agree(self):
        blobs = [(0, 0, slice(5, 25), slice(5, 25)), (2, 0, slice(8, 20), slice(8, 20))]
        chunked = atmospheric_river.atmospheric_river_mask(
            ivt=make_ar_input_dataset(blobs=blobs)[0],
            ivt_laplacian=make_ar_input_dataset(blobs=blobs)[1],
            min_size_gridpoints=50,
        )
        unchunked = atmospheric_river.atmospheric_river_mask(
            ivt=make_ar_input_dataset(blobs=blobs, chunk=False)[0],
            ivt_laplacian=make_ar_input_dataset(blobs=blobs, chunk=False)[1],
            min_size_gridpoints=50,
        )
        np.testing.assert_array_equal(np.asarray(chunked), np.asarray(unchunked))


class TestAtmosphericRiverPipelineOutput:
    """The derived-variable path and the module path must agree.

    Both entry points run the same pipeline, so these pin that they keep
    producing identical output.
    """

    @staticmethod
    def _dataset():
        rng = np.random.default_rng(11)
        valid_time = pd.date_range("2021-01-01", periods=3, freq="6h")
        level = np.array([1000, 850, 700, 500, 300, 200])
        lat = np.linspace(30.0, 34.75, 20)
        lon = np.linspace(-130.0, -125.25, 20)
        shape = (len(valid_time), len(level), len(lat), len(lon))
        dims = ["valid_time", "level", "latitude", "longitude"]
        return xr.Dataset(
            {
                "eastward_wind": (dims, rng.uniform(-20, 40, shape)),
                "northward_wind": (dims, rng.uniform(-20, 40, shape)),
                "specific_humidity": (dims, rng.uniform(1e-4, 2e-2, shape)),
            },
            coords={
                "valid_time": valid_time,
                "level": level,
                "latitude": lat,
                "longitude": lon,
            },
        )

    def test_the_derived_variable_matches_the_module_pipeline(self):
        from extremeweatherbench import derived

        data = self._dataset()
        variable = derived.AtmosphericRiverVariables()
        subset = derived._subset_to_top_pressure_level(data, 300)

        from_derived = variable.derive_variable(data).compute()
        from_module = (
            atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
                subset
            ).compute()
        )

        assert set(from_derived.data_vars) == set(from_module.data_vars)
        for name in from_module.data_vars:
            np.testing.assert_allclose(
                from_derived[name].values, from_module[name].values
            )

    def test_the_pipeline_output_is_pinned(self):
        data = self._dataset()
        result = atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
            data
        ).compute()

        # One space-time feature spans the box; the box is open Pacific, so
        # nothing intersects land.
        assert np.array_equal(
            np.unique(result["atmospheric_river_mask"].values), [0, 1]
        )
        assert int(result["atmospheric_river_mask"].sum()) == 1116
        np.testing.assert_allclose(
            float(result["integrated_vapor_transport"].max()), 4174.306510274923
        )
        assert float(np.nansum(result["atmospheric_river_land_intersection"])) == 0.0

    def test_the_land_intersection_lights_up_over_land(self):
        data = self._dataset()
        over_land = data.assign_coords(
            latitude=data["latitude"].values + 6.0,
            longitude=data["longitude"].values + 30.0,
        )

        result = atmospheric_river.build_atmospheric_river_mask_and_land_intersection(
            over_land
        ).compute()

        assert int(result["atmospheric_river_mask"].sum()) == 1116
        assert float(np.nansum(result["atmospheric_river_land_intersection"])) == 1116.0
