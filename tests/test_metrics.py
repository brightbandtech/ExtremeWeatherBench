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

    # TODO: tomorrow fix this up to ge 100% test coverage for metrics
    """Tests the _align_observations_temporal_resolution method of the Metric base class."""

    def test_obs_finer_temporal_resolution(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test when forecast and observation have same temporal resolution."""
        metric = metrics.Metric()
        aligned_obs = metric._align_observations_temporal_resolution(
            mock_forecast_dataarray, mock_gridded_obs_dataarray
        )
        # Check that observation was modified
        assert len(np.unique(np.diff(aligned_obs.time))) == 1
        assert np.unique(np.diff(aligned_obs.time)).astype("timedelta64[h]").astype(
            int
        ) == np.unique(np.diff(mock_forecast_dataarray.lead_time))

    def test_obs_finer_temporal_resolution_data(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test when observation has finer temporal resolution than forecast,
        that the outputs are the correct values at the hour."""
        metric = metrics.Metric()
        aligned_obs = metric._align_observations_temporal_resolution(
            mock_forecast_dataarray, mock_gridded_obs_dataarray
        )

        test_truncated_obs, test_aligned_obs = xr.align(
            mock_gridded_obs_dataarray, aligned_obs, join="inner"
        )
        # Check that observation was modified
        assert (test_aligned_obs == test_truncated_obs).all()

    def test_obs_coarser_temporal_resolution(
        self, mock_forecast_dataarray, mock_gridded_obs_dataarray
    ):
        """Test when observation has coarser temporal resolution than forecast,
        that observations are unmodified."""
        metric = metrics.Metric()
        aligned_obs = metric._align_observations_temporal_resolution(
            mock_forecast_dataarray,
            mock_gridded_obs_dataarray.resample(time="12h").first(),
        )
        # Check that observation was modified
        assert (aligned_obs == mock_gridded_obs_dataarray).all()


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
                7.072156,
                7.07325857,
                7.05412501,
                7.07429232,
                7.06224259,
                7.07307497,
                7.06065033,
                7.06077196,
                7.06397257,
                7.06501056,
                7.07663102,
                7.05498302,
                7.07148515,
                7.06898867,
                7.0632514,
                7.06386921,
                7.06310243,
                7.06324579,
                7.06674583,
                7.07391275,
                7.08054948,
                7.07530351,
                7.06474977,
                7.07237954,
                7.07761042,
                7.04927946,
                7.06237319,
                7.07482148,
                7.06780911,
                7.07967395,
                7.06711688,
                7.06797943,
                7.06700641,
                7.06965538,
                7.06657121,
                7.08633883,
                7.07582586,
                7.07056922,
                7.069464,
                7.09834288,
                7.06137688,
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

    def test_maximum_mae_wrong_data_var(
        self, mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
    ):
        """As this metric is meant to be only for surface temperature, make sure
        it fails if another variable is in its place."""
        mock_subset_forecast_dataarray = mock_subset_forecast_dataarray.rename(
            "bad_name"
        )
        metric = metrics.MaximumMAE()
        with pytest.raises(NotImplementedError):
            metric.compute(
                mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
            )

    def test_maximum_mae_values(
        self, mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        # Instantiate the MaximumMAE metric
        metric = metrics.MaximumMAE()
        result = metric.compute(
            mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
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
        self, mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
    ):
        """Test if compute returns the proper type and dimensions."""
        # Instantiate the MaxOfMinTempMAE metric
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(
            mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
        )

        # Check the result is an xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Check the dimensions of the result
        assert "lead_time" in result.dims

    def test_max_of_min_temp_mae_wrong_data_var(
        self, mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
    ):
        """As this metric is meant to be only for surface temperature, make sure
        it fails if another variable is in its place."""
        mock_subset_forecast_dataarray = mock_subset_forecast_dataarray.rename(
            "bad_name"
        )
        metric = metrics.MaxOfMinTempMAE()
        with pytest.raises(NotImplementedError):
            metric.compute(
                mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
            )

    def test_max_of_min_temp_mae_values(
        self, mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
    ):
        """Test if the numerical outputs of the metric are producing the correct results,
        to a reasonable (1e-7) precision."""
        metric = metrics.MaxOfMinTempMAE()
        result = metric.compute(
            mock_subset_forecast_dataarray, mock_subset_gridded_obs_dataarray
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
