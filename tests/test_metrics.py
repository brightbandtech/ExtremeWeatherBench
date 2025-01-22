from extremeweatherbench import metrics
import xarray as xr
import pytest
import pandas as pd


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]


class TestRegionalRMSE:
    def test_regional_rmse_compute(
        self, mock_forecast_dataset, mock_gridded_obs_dataset
    ):
        # Instantiate the RegionalRMSE metric
        metric = metrics.RegionalRMSE()
        mock_forecast_dataarray = dataset_to_dataarray(mock_forecast_dataset)
        mock_gridded_obs_dataarray = dataset_to_dataarray(mock_gridded_obs_dataset)
        # Compute the RMSE
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)

        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims


class TestMaximumMAE:
    def test_base_compute(self, mock_forecast_dataset, mock_gridded_obs_dataset):
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
        # Turn fixtures into dataarrays
        mock_forecast_dataarray = dataset_to_dataarray(mock_forecast_dataset)
        mock_gridded_obs_dataarray = dataset_to_dataarray(mock_gridded_obs_dataset)
        # Compute the RMSE

        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)
        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_in_forecast_dataarray(
        self, mock_forecast_dataset, mock_gridded_obs_dataset_max_in_forecast
    ):
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()

        # Turn fixtures into dataarrays
        mock_forecast_dataarray = dataset_to_dataarray(mock_forecast_dataset)
        mock_gridded_obs_dataarray_max_in_forecast = dataset_to_dataarray(
            mock_gridded_obs_dataset_max_in_forecast
        )
        result = metric.compute(
            mock_forecast_dataarray, mock_gridded_obs_dataarray_max_in_forecast
        )
        assert all(
            [
                pd.isna(result.sel({"lead_time": lead_time}))
                for lead_time in [0, 24, 48, 72, 96, 168, 240]
            ]
        )
        assert pytest.approx(result.sel({"lead_time": 6}), 1e-7) == 24.94948657
        assert pytest.approx(result.sel({"lead_time": 30}), 1e-7) == 24.96504733


class TestMaxOfMinTempMAE:
    def test_max_of_min_temp_mae_compute(
        self, mock_forecast_dataset, mock_gridded_obs_dataset
    ):
        # Instantiate the MaxOfMinTempMAE metric
        metric = metrics.MaxOfMinTempMAE()
        mock_forecast_dataarray = dataset_to_dataarray(mock_forecast_dataset)
        mock_gridded_obs_dataarray = dataset_to_dataarray(mock_gridded_obs_dataset)
        # Compute the RMSE
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)
        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

        def test_metric_base_class():
            # Test base Metric class name property
            metric = metrics.Metric()
            assert metric.name == "Metric"

            # Test that compute raises NotImplementedError
            mock_forecast = xr.DataArray()
            mock_obs = xr.DataArray()
            with pytest.raises(NotImplementedError):
                metric.compute(mock_forecast, mock_obs)


def test_metric_align_datasets(mock_forecast_dataset, mock_gridded_obs_dataset):
    # Test dataset alignment method
    metric = metrics.Metric()
    mock_forecast = dataset_to_dataarray(mock_forecast_dataset)
    mock_obs = dataset_to_dataarray(mock_gridded_obs_dataset)

    init_time = mock_forecast.init_time[0].values
    init_time_dt = pd.Timestamp(init_time).to_pydatetime()

    aligned_forecast, aligned_obs = metric.align_datasets(
        mock_forecast, mock_obs, init_time_dt
    )

    # Check aligned datasets have same time coordinates
    assert (aligned_forecast.time == aligned_obs.time).all()

    # Check forecast was properly subset by init time
    assert aligned_forecast.init_time.size == 1
    assert pd.Timestamp(aligned_forecast.init_time.values) == pd.Timestamp(init_time)
