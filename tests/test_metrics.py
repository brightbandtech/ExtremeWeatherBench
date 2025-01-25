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
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test that the base Metric compute method returns a NotImplementedError."""
        metric = metrics.Metric()
        with pytest.raises(NotImplementedError):
            metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)

    def test_align_datasets(self, mock_forecast_dataarray, mock_gridded_obs_dataarray):
        """Test that the conversion from init time to valid time (named as time) produces an aligned
        dataarray to ensure metrics are applied properly."""
        metric = metrics.Metric()
        init_time_datetime = pd.Timestamp(
            mock_forecast_dataarray.init_time[0].values
        ).to_pydatetime()
        aligned_forecast, aligned_obs = metric.align_datasets(
            mock_forecast_dataarray, mock_gridded_obs_dataarray, init_time_datetime
        )

        # Check aligned datasets have same time coordinates
        assert (aligned_forecast.time == aligned_obs.time).all()

        # Check forecast was properly subset by init time
        assert aligned_forecast.init_time.size == 1
        assert pd.Timestamp(aligned_forecast.init_time.values) == pd.Timestamp(
            mock_forecast_dataarray.init_time[0].values
        )


class TestRegionalRMSE:
    """Tests the RegionalRMSE Metric child class."""

    def test_regional_rmse_compute_output_metadata_types(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the RegionalRMSE metric
        metric = metrics.RegionalRMSE()
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)

        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_regional_rmse_values(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        output = np.array(
            [
                7.06819423,
                7.07986864,
                7.08808892,
                7.06857515,
                7.07467895,
                7.07154156,
                7.05989394,
                7.06687087,
                7.0665981,
                7.05737303,
                7.08282287,
                7.06380817,
                7.07936601,
                7.07125565,
                7.06621689,
                7.0641209,
                7.06740009,
                7.07123912,
                7.07697865,
                7.05404263,
                7.07595125,
                7.07550436,
                7.05839933,
                7.06479538,
                7.08006438,
                7.06520873,
                7.07479995,
                7.06371806,
                7.08151001,
                7.08536655,
                7.06980819,
                7.07308411,
                7.08840835,
                7.06256349,
                7.07074064,
                7.08574952,
                7.08438375,
                7.07106463,
                7.06908532,
                7.08782645,
                7.07161996,
            ]
        )

        metric = metrics.RegionalRMSE()
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)

        assert all(
            [
                pytest.approx(individual_result, 1e-7) == individual_output
                for individual_result, individual_output in zip(result.values, output)
            ]
        )


class TestMaximumMAE:
    """Tests the MaximumMAE Metric child class, which computes maximum temperature MAE during a case."""

    def test_base_compute(self, mock_forecast_dataarray, mock_gridded_obs_dataarray):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)
        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_maximum_mae_values(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray_max_in_forecast
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
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
    """Tests the MaxOfMinTempMAE Metric child class, which computes highest minimum temperature MAE during a case."""

    def test_max_of_min_temp_mae_compute(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the MaxOfMinTempMAE metric
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)
        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_max_of_min_temp_mae_wrong_data_var(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """As this metric is meant to be only for surface temperature, make sure
        it fails if another variable is in its place."""
        mock_forecast_dataarray = mock_forecast_dataarray.rename("bad_name")
        metric = metrics.MaxOfMinTempMAE()
        with pytest.raises(KeyError):
            metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)

    def test_max_of_min_temp_mae_values(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray_max_in_forecast
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(
            mock_forecast_dataarray, mock_gridded_obs_dataarray_max_in_forecast
        )
        assert all(
            [
                pd.isna(result.sel({"lead_time": lead_time}))
                for lead_time in [6, 12, 18, 30, 72, 96, 168, 240]
            ]
        )
        assert pytest.approx(result.sel({"lead_time": 0}), 1e-7) == 0.04544667
        assert pytest.approx(result.sel({"lead_time": 24}), 1e-7) == 0.04979312


class TestOnsetME:
    """Tests the not-yet-implemented OnsetME Metric child class, which will identify
    the temporal mean bias error of a case's onset."""

    def test_compute(self, mock_forecast_dataarray, mock_gridded_obs_dataarray):
        """Test if compute returns the proper type and dimensions (in this case an error)."""
        metric = metrics.OnsetME()
        with pytest.raises(
            NotImplementedError, match="Onset mean error not yet implemented."
        ):
            metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)


class TestDurationME:
    """Tests the not-yet-implemented DurationME Metric child class, which will identify
    the temporal mean bias error of a case's duration."""

    def test_compute(self, mock_forecast_dataarray, mock_gridded_obs_dataarray):
        """Test if compute returns the proper type and dimensions (in this case an error)."""
        metric = metrics.DurationME()
        with pytest.raises(
            NotImplementedError, match="Duration mean error not yet implemented."
        ):
            metric.compute(mock_forecast_dataarray, mock_gridded_obs_dataarray)
