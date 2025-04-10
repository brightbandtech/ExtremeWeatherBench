import pytest
import xarray as xr
from extremeweatherbench import config, data_loader, events
from unittest.mock import patch
import numpy as np
import pandas as pd


def test_open_and_preprocess_forecast_dataset_invalid_path(default_forecast_config):
    invalid_config = config.Config(
        event_types=[events.HeatWave],
        forecast_dir="invalid/path",
        gridded_obs_path="test/path",
    )
    with pytest.raises(TypeError):
        data_loader.open_and_preprocess_forecast_dataset(
            invalid_config, default_forecast_config
        )


def test_open_and_preprocess_forecast_dataset_zarr(
    mocker, sample_config, default_forecast_config, sample_forecast_dataset
):
    # Mock xr.open_zarr to avoid actually opening a zarr file
    mock_zarr_dataset = sample_forecast_dataset
    mocker.patch("xarray.open_zarr", return_value=mock_zarr_dataset)
    sample_config.forecast_dir = "test/test.zarr"
    forecast_dataset = data_loader.open_and_preprocess_forecast_dataset(
        sample_config, default_forecast_config
    )
    xr.testing.assert_equal(forecast_dataset, mock_zarr_dataset)


def test_open_and_preprocess_forecast_dataset_json(
    mocker, sample_config, default_forecast_config, sample_forecast_dataset
):
    mock_json_dataset = sample_forecast_dataset
    mocker.patch(
        "extremeweatherbench.data_loader.open_kerchunk_reference",
        return_value=mock_json_dataset,
    )
    sample_config.forecast_dir = "test/test.json"

    forecast_dataset = data_loader.open_and_preprocess_forecast_dataset(
        sample_config, default_forecast_config
    )
    xr.testing.assert_equal(forecast_dataset, mock_json_dataset)


def test_open_and_preprocess_forecast_dataset_multiple_types(
    sample_config, default_forecast_config
):
    # test for multiple other file types
    sample_config.forecast_dir = "test/other"

    with pytest.raises(TypeError):
        data_loader.open_and_preprocess_forecast_dataset(
            sample_config, default_forecast_config
        )


def test_open_and_preprocess_forecast_dataset_no_files(
    mocker, sample_config, default_forecast_config
):
    sample_config.forecast_dir = "test/"
    with pytest.raises(TypeError):
        data_loader.open_and_preprocess_forecast_dataset(
            sample_config, default_forecast_config
        )


def test_open_and_preprocess_forecast_dataset_invalid_file_type(
    mocker, sample_config, default_forecast_config
):
    mock_fs = mocker.Mock()
    mock_fs.ls.return_value = ["test.invalid"]  # Invalid file type
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    with pytest.raises(TypeError, match="Unknown file type found in forecast path."):
        data_loader.open_and_preprocess_forecast_dataset(
            sample_config, default_forecast_config
        )


def test_open_kerchunk_reference(sample_config):
    # Test with parquet file type
    mock_parquet_dataset = xr.Dataset({"random_var": ("time", range(0, 41))})
    with patch(
        "extremeweatherbench.data_loader.xr.open_dataset",
        return_value=mock_parquet_dataset,
    ):
        sample_config.forecast_dir = "test.parq"
        forecast_dataset = data_loader.open_kerchunk_reference(sample_config)
        assert "random_var" in forecast_dataset

    # Test with json and zarr file types
    mock_zarr_dataset = xr.Dataset({"random_var": ("time", range(0, 41))})
    with patch(
        "extremeweatherbench.data_loader.xr.open_dataset",
        return_value=mock_zarr_dataset,
    ):
        sample_config.forecast_dir = "test.json"
        forecast_dataset = data_loader.open_kerchunk_reference(sample_config)
        assert "random_var" in forecast_dataset


def test_rename_forecast_dataset(sample_forecast_dataset):
    """Test that _rename_forecast_dataset correctly renames variables according to the schema config."""
    # Create a copy of the sample dataset to avoid modifying the original
    ds = sample_forecast_dataset.copy(deep=True)

    # Rename variables to match what we'd expect from a raw dataset before renaming
    # This simulates the dataset before variable standardization
    ds = ds.rename(
        {
            "surface_air_temperature": "t2m",
            "surface_eastward_wind": "u10",
            "surface_northward_wind": "v10",
            "lead_time": "forecast_time",
        }
    )

    # Create a custom schema config for testing that matches our renamed variables
    schema_config = config.ForecastSchemaConfig(
        surface_air_temperature="t2m",
        surface_eastward_wind="u10",
        surface_northward_wind="v10",
        lead_time="forecast_time",
    )

    # Store original values for later comparison
    t2m_values = ds["t2m"].values
    u10_values = ds["u10"].values
    v10_values = ds["v10"].values
    forecast_time_values = ds["forecast_time"].values

    # Call the function being tested
    renamed_ds = data_loader._rename_forecast_dataset(ds, schema_config)

    # Verify that variables were correctly renamed
    assert "surface_air_temperature" in renamed_ds.data_vars
    assert "surface_eastward_wind" in renamed_ds.data_vars
    assert "surface_northward_wind" in renamed_ds.data_vars

    # Verify that coordinates were correctly renamed
    assert "lead_time" in renamed_ds.coords

    # Verify that original variable names are no longer present
    assert "t2m" not in renamed_ds.data_vars
    assert "u10" not in renamed_ds.data_vars
    assert "v10" not in renamed_ds.data_vars
    assert "forecast_time" not in renamed_ds.coords

    # Verify that the data values remain unchanged after renaming
    np.testing.assert_array_equal(
        renamed_ds["surface_air_temperature"].values, t2m_values
    )
    np.testing.assert_array_equal(
        renamed_ds["surface_eastward_wind"].values, u10_values
    )
    np.testing.assert_array_equal(
        renamed_ds["surface_northward_wind"].values, v10_values
    )
    np.testing.assert_array_equal(renamed_ds["lead_time"].values, forecast_time_values)


def test_rename_forecast_dataset_partial_mapping(sample_forecast_dataset):
    """Test that _rename_forecast_dataset correctly handles partial mappings."""
    # Create a copy of the sample dataset to avoid modifying the original
    ds = sample_forecast_dataset.copy(deep=True)

    # Rename only one variable to simulate partial mapping scenario
    ds = ds.rename({"surface_air_temperature": "t2m"})

    # Add a custom variable that doesn't match any schema mapping
    ds["custom_var"] = xr.DataArray(
        np.random.rand(*ds.surface_eastward_wind.shape),
        dims=ds.surface_eastward_wind.dims,
        coords=ds.surface_eastward_wind.coords,
    )

    # Store original values for later comparison
    t2m_values = ds["t2m"].values
    eastward_wind_values = ds["surface_eastward_wind"].values
    northward_wind_values = ds["surface_northward_wind"].values
    custom_var_values = ds["custom_var"].values

    # Create a schema config where only some variables match
    schema_config = config.ForecastSchemaConfig(
        surface_air_temperature="t2m",
        surface_eastward_wind="u10",  # Not in dataset
        surface_northward_wind="v10",  # Not in dataset
    )

    # Call the function being tested
    renamed_ds = data_loader._rename_forecast_dataset(ds, schema_config)

    # Verify that matching variables were renamed
    assert "surface_air_temperature" in renamed_ds.data_vars

    # Verify that non-matching variables were left unchanged
    assert "surface_eastward_wind" in renamed_ds.data_vars  # Already had standard name
    assert "surface_northward_wind" in renamed_ds.data_vars  # Already had standard name
    assert "custom_var" in renamed_ds.data_vars

    # Verify that original variable names of renamed variables are no longer present
    assert "t2m" not in renamed_ds.data_vars

    # Verify that the data values remain unchanged
    np.testing.assert_array_equal(
        renamed_ds["surface_air_temperature"].values, t2m_values
    )
    np.testing.assert_array_equal(
        renamed_ds["surface_eastward_wind"].values, eastward_wind_values
    )
    np.testing.assert_array_equal(
        renamed_ds["surface_northward_wind"].values, northward_wind_values
    )
    np.testing.assert_array_equal(renamed_ds["custom_var"].values, custom_var_values)


def test_maybe_convert_dataset_lead_time_to_int(sample_config):
    # Test with timedelta lead_time
    ds_timedelta = xr.Dataset(
        coords={
            "lead_time": pd.timedelta_range(start="0h", periods=5, freq="6h"),
        }
    )
    result_timedelta = data_loader._maybe_convert_dataset_lead_time_to_int(
        sample_config, ds_timedelta
    )
    assert result_timedelta["lead_time"].dtype == np.dtype("int64")
    assert all(result_timedelta["lead_time"].values == [0, 6, 12, 18, 24])

    # Test with already integer lead_time
    ds_int = xr.Dataset(
        coords={
            "lead_time": [0, 6, 12, 18, 24],
        }
    )
    result_int = data_loader._maybe_convert_dataset_lead_time_to_int(
        sample_config, ds_int
    )
    assert result_int["lead_time"].dtype == np.dtype("int64")
    assert all(result_int["lead_time"].values == [0, 6, 12, 18, 24])
