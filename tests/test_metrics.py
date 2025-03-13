from extremeweatherbench import metrics
import xarray as xr
import pytest
import pandas as pd
import numpy as np


def dataset_to_dataarray(dataset):
    """Convert an xarray Dataset to a DataArray."""
    mock_data_var = [data_var for data_var in dataset.data_vars][0]
    return dataset[mock_data_var]


class TestMetric:
    """Tests the Metric base class."""

    def test_name_property(self):
        """Test that the property decorator is applied properly."""
        metric = metrics.Metric()
        assert metric.name == "Metric"

    def test_compute_not_implemented(
        self, sample_forecast_dataarray, sample_gridded_obs_dataarray
    ):
        """Test that the base Metric compute method returns a NotImplementedError."""
        metric = metrics.Metric()
        with pytest.raises(NotImplementedError):
            metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)


class TestRegionalRMSE:
    """Tests the RegionalRMSE Metric child class."""

    def test_regional_rmse_compute_output_metadata_types(
        self, sample_forecast_dataarray, sample_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the RegionalRMSE metric
        metric = metrics.RegionalRMSE()
        result = metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)

        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_regional_rmse_values(
        self, sample_forecast_dataarray, sample_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        output = np.load("tests/data/arrays.npz")["RegionalRMSE"]
        metric = metrics.RegionalRMSE()
        result = metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)

        assert all(
            [
                pytest.approx(individual_result, 1e-7) == individual_output
                for individual_result, individual_output in zip(result.values, output)
            ]
        )


class TestMaximumMAE:
    """Tests the MaximumMAE Metric child class, which computes maximum temperature MAE during a case."""

    def test_base_compute(
        self, sample_forecast_dataarray, sample_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
        result = metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)
        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_maximum_mae_values(
        self, sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
        result = metric.compute(
            sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
        )
        assert all(
            [
                pd.isna(result.sel({"lead_time": lead_time}))
                for lead_time in [0, 24, 48, 72, 96, 168, 240]
            ]
        )
        assert pytest.approx(result.sel({"lead_time": 18}), 1e-7) == 1.0
        assert pytest.approx(result.sel({"lead_time": 42}), 1e-7) == 2.0


class TestMaxOfMinTempMAE:
    """Tests the MaxOfMinTempMAE Metric child class, which computes highest minimum temperature MAE during a case."""

    def test_max_of_min_temp_mae_compute(
        self, sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the MaxOfMinTempMAE metric
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(
            sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
        )

        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_max_of_min_temp_mae_values(
        self, sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(
            sample_subset_forecast_dataarray, sample_subset_gridded_obs_dataarray
        )

        assert all(
            [
                pd.isna(result.sel({"lead_time": lead_time}))
                for lead_time in [0, 6, 12, 30, 168, 240]
            ]
        )
        assert pytest.approx(result.sel({"lead_time": 18}), 1e-7) == 0.76977804


class TestOnsetME:
    """Tests the not-yet-implemented OnsetME Metric child class, which will identify
    the temporal mean bias error of a case's onset."""

    def test_compute(self, sample_forecast_dataarray, sample_gridded_obs_dataarray):
        """Test if compute returns the proper type and dimensions (in this case an error)."""
        metric = metrics.OnsetME()
        with pytest.raises(NotImplementedError):
            metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)


class TestDurationME:
    """Tests the not-yet-implemented DurationME Metric child class, which will identify
    the temporal mean bias error of a case's duration."""

    def test_compute(self, sample_forecast_dataarray, sample_gridded_obs_dataarray):
        """Test if compute returns the proper type and dimensions (in this case an error)."""
        metric = metrics.DurationME()
        with pytest.raises(NotImplementedError):
            metric.compute(sample_forecast_dataarray, sample_gridded_obs_dataarray)
