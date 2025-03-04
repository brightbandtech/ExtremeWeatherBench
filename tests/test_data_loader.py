import pytest
import xarray as xr
from extremeweatherbench import config, data_loader, events
from unittest.mock import patch


def test_open_forecast_dataset_invalid_path(default_forecast_config):
    invalid_config = config.Config(
        event_types=[events.HeatWave],
        forecast_dir="invalid/path",
        gridded_obs_path="test/path",
    )
    with pytest.raises(FileNotFoundError):
        data_loader.open_forecast_dataset(invalid_config, default_forecast_config)


def test_open_forecast_dataset_zarr(
    mocker, sample_config, default_forecast_config, sample_forecast_dataset
):
    # Mock the fsspec.filesystem and fs.ls methods
    mock_fs = mocker.Mock()
    mock_fs.ls.return_value = ["test.zarr"]
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    # Mock xr.open_zarr to avoid actually opening a zarr file
    mock_zarr_dataset = sample_forecast_dataset
    mocker.patch("xarray.open_zarr", return_value=mock_zarr_dataset)

    forecast_dataset = data_loader.open_forecast_dataset(
        sample_config, default_forecast_config
    )
    xr.testing.assert_equal(forecast_dataset, mock_zarr_dataset)


def test_open_forecast_dataset_json(
    mocker, sample_config, default_forecast_config, sample_forecast_dataset
):
    mock_fs = mocker.Mock()
    mock_fs.ls.return_value = ["test.json"]
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    mock_json_dataset = sample_forecast_dataset
    mocker.patch(
        "extremeweatherbench.data_loader.open_kerchunk_reference",
        return_value=mock_json_dataset,
    )

    forecast_dataset = data_loader.open_forecast_dataset(
        sample_config, default_forecast_config
    )
    xr.testing.assert_equal(forecast_dataset, mock_json_dataset)


def test_open_forecast_dataset_multiple_types(
    mocker, sample_config, default_forecast_config
):
    mock_fs = mocker.Mock()

    # test parquet kerchunk reference
    sample_config.forecast_dir = "test/parquet"
    mock_fs.ls.return_value = ["test.parquet", "test.txt"]

    # test for multiple other file types
    sample_config.forecast_dir = "test/other"
    mock_fs.ls.return_value = ["test.nc", "test.txt"]
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    with pytest.raises(TypeError, match="Multiple file types found in forecast path."):
        data_loader.open_forecast_dataset(sample_config, default_forecast_config)


def test_open_forecast_dataset_no_files(mocker, sample_config, default_forecast_config):
    mock_fs = mocker.Mock()
    mock_fs.ls.return_value = []  # No files found
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    with pytest.raises(FileNotFoundError, match="No files found in forecast path."):
        data_loader.open_forecast_dataset(sample_config, default_forecast_config)


def test_open_forecast_dataset_invalid_file_type(
    mocker, sample_config, default_forecast_config
):
    mock_fs = mocker.Mock()
    mock_fs.ls.return_value = ["test.invalid"]  # Invalid file type
    mocker.patch("fsspec.filesystem", return_value=mock_fs)

    with pytest.raises(TypeError, match="Unknown file type found in forecast path."):
        data_loader.open_forecast_dataset(sample_config, default_forecast_config)


def test_open_kerchunk_reference(mocker, default_forecast_config):
    # Test with parquet file type
    mock_parquet_dataset = xr.Dataset({"random_var": ("time", range(0, 41))})
    with patch(
        "extremeweatherbench.data_loader.xr.open_dataset",
        return_value=mock_parquet_dataset,
    ):
        forecast_dataset = data_loader.open_kerchunk_reference(
            "test.parq", default_forecast_config
        )
        assert "random_var" in forecast_dataset

    # Test with json and zarr file types
    mock_zarr_dataset = xr.Dataset({"random_var": ("time", range(0, 41))})
    with patch(
        "extremeweatherbench.data_loader.xr.open_dataset",
        return_value=mock_zarr_dataset,
    ):
        forecast_dataset = data_loader.open_kerchunk_reference(
            "test.json", default_forecast_config
        )
        assert "random_var" in forecast_dataset
