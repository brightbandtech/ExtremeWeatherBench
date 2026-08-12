"""Tests for the outputs module."""

import logging

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from extremeweatherbench import outputs

METADATA_KWARGS = {
    "metric": "RMSE",
    "target_variable": "2m_temperature",
    "forecast_variable": "surface_air_temperature",
    "forecast_source": "test_forecast",
    "target_source": "test_target",
    "case_id_number": 1,
    "event_type": "heat_wave",
}


def _lead_time_result(values=(1.0, 2.0), lead_times=(0, 6)):
    result = xr.DataArray(
        data=list(values), dims=["lead_time"], coords={"lead_time": list(lead_times)}
    )
    return outputs.annotate_metric_result(result, **METADATA_KWARGS)


def _init_time_result(values=(3.0, 4.0)):
    init_times = pd.date_range("2021-06-20", periods=len(values), freq="D")
    result = xr.DataArray(
        data=list(values), dims=["init_time"], coords={"init_time": init_times}
    )
    return outputs.annotate_metric_result(result, **METADATA_KWARGS)


class TestAnnotateMetricResult:
    """Test annotate_metric_result attaches metadata without altering data."""

    def test_attaches_scalar_coords_without_altering_dims_or_values(self):
        result = xr.DataArray(
            data=[1.0, 2.0, 3.0],
            dims=["lead_time"],
            coords={"lead_time": [0, 6, 12]},
        )
        annotated = outputs.annotate_metric_result(result, **METADATA_KWARGS)

        assert annotated.dims == result.dims
        np.testing.assert_array_equal(annotated.values, result.values)
        for coord in outputs.METADATA_COORDS:
            assert coord in annotated.coords
            assert annotated.coords[coord].ndim == 0
            assert coord not in annotated.dims

    def test_does_not_mutate_the_input_array(self):
        result = xr.DataArray(data=[1.0], dims=["lead_time"], coords={"lead_time": [0]})
        outputs.annotate_metric_result(result, metric="RMSE")
        assert "metric" not in result.coords

    def test_metadata_values_are_correct(self):
        result = xr.DataArray(data=[1.0], dims=["lead_time"], coords={"lead_time": [0]})
        annotated = outputs.annotate_metric_result(result, **METADATA_KWARGS)
        for key, value in METADATA_KWARGS.items():
            assert annotated.coords[key].item() == value


class TestResultsToDataFrame:
    """Test conversion of annotated results into the long-form DataFrame."""

    def test_lead_time_result_has_output_columns(self):
        df = outputs.results_to_dataframe([_lead_time_result()])
        assert list(df.columns) == outputs.OUTPUT_COLUMNS

    def test_init_time_result_has_output_columns(self):
        df = outputs.results_to_dataframe([_init_time_result()])
        assert list(df.columns) == outputs.OUTPUT_COLUMNS

    def test_forecast_variable_is_dropped(self):
        df = outputs.results_to_dataframe([_lead_time_result()])
        assert "forecast_variable" not in df.columns

    def test_extra_coords_survive_as_trailing_columns(self):
        init_times = pd.date_range("2021-06-20", periods=2, freq="D")
        result = xr.DataArray(
            data=[3.0, 4.0], dims=["init_time"], coords={"init_time": init_times}
        )
        result = result.assign_coords(
            forecast_landfall_latitude=("init_time", [10.0, 11.0]),
            forecast_landfall_longitude=("init_time", [-80.0, -81.0]),
        )
        annotated = outputs.annotate_metric_result(result, **METADATA_KWARGS)

        df = outputs.results_to_dataframe([annotated])

        expected_extra = ["forecast_landfall_latitude", "forecast_landfall_longitude"]
        assert list(df.columns) == outputs.OUTPUT_COLUMNS + expected_extra

    def test_regression_guard_matches_prior_schema(self):
        """Guards the pre-refactor pandas schema: columns, order, dtypes, values."""
        lead_times = [0, 6, 12]
        values = [1.5, 2.5, 3.5]
        result = xr.DataArray(
            data=values, dims=["lead_time"], coords={"lead_time": lead_times}
        )
        annotated = outputs.annotate_metric_result(
            result,
            metric="RMSE",
            target_variable="2m_temperature",
            forecast_variable="surface_air_temperature",
            forecast_source="test_forecast",
            target_source="test_target",
            case_id_number=7,
            event_type="heat_wave",
        )

        df = outputs.results_to_dataframe([annotated])

        expected = pd.DataFrame(
            {
                "value": pd.Series(values, dtype="float64"),
                "lead_time": pd.Series(lead_times, dtype="int64"),
                "init_time": pd.Series([np.nan, np.nan, np.nan], dtype="float64"),
                "target_variable": ["2m_temperature"] * 3,
                "metric": ["RMSE"] * 3,
                "forecast_source": ["test_forecast"] * 3,
                "target_source": ["test_target"] * 3,
                "case_id_number": pd.Series([7, 7, 7], dtype="int64"),
                "event_type": ["heat_wave"] * 3,
            }
        )

        assert list(df.columns) == list(expected.columns)
        pd.testing.assert_frame_equal(df, expected)

    def test_empty_list_returns_empty_output_columns(self):
        df = outputs.results_to_dataframe([])
        assert list(df.columns) == outputs.OUTPUT_COLUMNS
        assert len(df) == 0

    def test_list_with_empty_result_is_dropped(self):
        empty_result = xr.DataArray(
            data=[], dims=["lead_time"], coords={"lead_time": []}
        )
        annotated_empty = outputs.annotate_metric_result(
            empty_result, **METADATA_KWARGS
        )
        good = _lead_time_result()

        df = outputs.results_to_dataframe([annotated_empty, good])

        assert len(df) == good.sizes["lead_time"]
        assert df["metric"].notna().all()


class TestSafeConcat:
    """Test the moved _safe_concat helper."""

    def test_filters_empty_and_all_na_frames(self):
        good = pd.DataFrame({"value": [1.0], "metric": ["RMSE"]})
        empty = pd.DataFrame(columns=["value", "metric"])
        all_na = pd.DataFrame({"value": [np.nan], "metric": [np.nan]})

        result = outputs._safe_concat([empty, all_na, good], ignore_index=True)

        assert len(result) == 1
        assert result["metric"].iloc[0] == "RMSE"

    def test_all_invalid_frames_returns_empty_output_columns(self):
        empty = pd.DataFrame(columns=["value"])
        all_na = pd.DataFrame({"value": [np.nan]})

        result = outputs._safe_concat([empty, all_na], ignore_index=True)

        assert list(result.columns) == outputs.OUTPUT_COLUMNS
        assert len(result) == 0

    def test_empty_list_returns_empty_output_columns(self):
        result = outputs._safe_concat([], ignore_index=True)
        assert list(result.columns) == outputs.OUTPUT_COLUMNS
        assert len(result) == 0


class TestEnsureOutputSchema:
    """Test the moved _ensure_output_schema helper."""

    def test_warns_when_a_non_time_column_is_missing(self, caplog):
        df = pd.DataFrame({"value": [1.0], "lead_time": [0]})
        with caplog.at_level(logging.WARNING, logger="extremeweatherbench.outputs"):
            result = outputs._ensure_output_schema(df)

        assert any("Missing expected columns" in r.getMessage() for r in caplog.records)
        assert "target_source" in result.columns

    def test_does_not_warn_when_only_lead_time_or_init_time_is_missing(self, caplog):
        df = pd.DataFrame(
            {
                "value": [1.0],
                "lead_time": [0],
                "target_variable": ["2m_temperature"],
                "metric": ["RMSE"],
                "forecast_source": ["fc"],
                "target_source": ["tg"],
                "case_id_number": [1],
                "event_type": ["heat_wave"],
            }
        )
        with caplog.at_level(logging.WARNING, logger="extremeweatherbench.outputs"):
            outputs._ensure_output_schema(df)

        assert not any(
            "Missing expected columns" in r.getMessage() for r in caplog.records
        )

    def test_reorders_columns_and_preserves_extras(self):
        df = pd.DataFrame(
            {
                "extra": [1],
                "lead_time": [0],
                "value": [1.0],
            }
        )
        result = outputs._ensure_output_schema(df)
        assert list(result.columns) == outputs.OUTPUT_COLUMNS + ["extra"]


@pytest.mark.parametrize(
    "coord",
    [
        "metric",
        "target_variable",
        "forecast_variable",
        "forecast_source",
        "target_source",
        "case_id_number",
        "event_type",
    ],
)
def test_metadata_coords_contains_expected_names(coord):
    assert coord in outputs.METADATA_COORDS


def _result(dims="lead_time", values=(1.0, 2.0), coord_values=None, **overrides):
    metadata = {**METADATA_KWARGS, **overrides}
    if dims == "lead_time":
        coord_values = list(coord_values) if coord_values is not None else [0, 6]
        result = xr.DataArray(
            data=list(values),
            dims=["lead_time"],
            coords={"lead_time": coord_values},
        )
    elif dims == "init_time":
        coord_values = (
            coord_values
            if coord_values is not None
            else pd.date_range("2021-06-20", periods=len(values), freq="D")
        )
        result = xr.DataArray(
            data=list(values), dims=["init_time"], coords={"init_time": coord_values}
        )
    else:
        raise ValueError(f"Unsupported dims kind: {dims}")
    return outputs.annotate_metric_result(result, **metadata)


class TestResultsToDataset:
    """Test the flat-Dataset builder for a whole run's results."""

    def test_empty_list_returns_empty_dataset(self):
        ds = outputs.results_to_dataset([])
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0
        assert len(ds.dims) == 0

    def test_dims_include_metadata_dims_and_own_dims(self):
        ds = outputs.results_to_dataset([_result()])
        assert set(ds.dims) >= {
            "case_id_number",
            "metric",
            "forecast_source",
            "target_source",
            "lead_time",
        }
        assert ds.sizes["case_id_number"] == 1
        assert ds.sizes["metric"] == 1
        assert ds.sizes["lead_time"] == 2

    def test_target_and_forecast_variable_are_dropped(self):
        ds = outputs.results_to_dataset([_result()])
        assert "target_variable" not in ds.dims
        assert "target_variable" not in ds.coords
        assert "forecast_variable" not in ds.coords

    def test_matching_variable_names_produce_single_named_data_var(self):
        result = _result(
            target_variable="2m_temperature", forecast_variable="2m_temperature"
        )
        ds = outputs.results_to_dataset([result])
        assert list(ds.data_vars) == ["2m_temperature"]

    def test_differing_variable_names_produce_merged_data_var_name(self):
        result = _result(
            target_variable="2m_temperature",
            forecast_variable="surface_air_temperature",
        )
        ds = outputs.results_to_dataset([result])
        assert list(ds.data_vars) == ["surface_air_temperature_vs_2m_temperature"]

    def test_event_type_is_a_non_dim_coord_along_case_id_number(self):
        result_1 = _result(case_id_number=1, event_type="heat_wave")
        result_2 = _result(case_id_number=2, event_type="freeze")
        ds = outputs.results_to_dataset([result_1, result_2])

        assert "event_type" not in ds.dims
        assert ds.coords["event_type"].dims == ("case_id_number",)
        by_case = dict(
            zip(ds.coords["case_id_number"].values, ds.coords["event_type"].values)
        )
        assert by_case == {1: "heat_wave", 2: "freeze"}

    def test_two_metrics_on_one_variable_merge_along_metric_dim(self):
        same_var = {
            "target_variable": "2m_temperature",
            "forecast_variable": "2m_temperature",
        }
        rmse = _result(
            metric="RMSE", coord_values=(0, 6), values=(1.0, 2.0), **same_var
        )
        mae = _result(metric="MAE", coord_values=(6, 12), values=(3.0, 4.0), **same_var)
        ds = outputs.results_to_dataset([rmse, mae])

        assert ds.sizes["metric"] == 2
        df = ds["2m_temperature"].to_dataframe(name="value").reset_index()
        df = df.dropna(subset=["value"])

        rmse_rows = df[df["metric"] == "RMSE"]
        assert dict(zip(rmse_rows["lead_time"], rmse_rows["value"])) == {0: 1.0, 6: 2.0}
        mae_rows = df[df["metric"] == "MAE"]
        assert dict(zip(mae_rows["lead_time"], mae_rows["value"])) == {6: 3.0, 12: 4.0}

    def test_lead_time_and_init_time_metrics_coexist_with_nan_padding(self):
        same_var = {
            "target_variable": "2m_temperature",
            "forecast_variable": "2m_temperature",
        }
        lead_result = _result(
            dims="lead_time", metric="RMSE", values=(1.0, 2.0), **same_var
        )
        init_result = _result(
            dims="init_time", metric="OnsetError", values=(5.0, 6.0), **same_var
        )
        ds = outputs.results_to_dataset([lead_result, init_result])

        assert "lead_time" in ds.dims
        assert "init_time" in ds.dims

        df = ds["2m_temperature"].to_dataframe(name="value").reset_index()
        df = df.dropna(subset=["value"])
        rmse_rows = df[df["metric"] == "RMSE"]
        assert sorted(rmse_rows["value"]) == [1.0, 2.0]
        assert sorted(rmse_rows["lead_time"]) == [0, 6]

        onset_rows = df[df["metric"] == "OnsetError"]
        assert sorted(onset_rows["value"]) == [5.0, 6.0]

    def test_landfall_extra_coords_are_promoted_and_preserved_per_case(self):
        def make(case_id, lat_val, lon_val):
            init_times = pd.date_range("2021-06-20", periods=2, freq="D")
            result = xr.DataArray(
                data=[1.0, 2.0], dims=["init_time"], coords={"init_time": init_times}
            )
            result = result.assign_coords(
                forecast_landfall_latitude=("init_time", [lat_val, lat_val + 1]),
                forecast_landfall_longitude=("init_time", [lon_val, lon_val + 1]),
            )
            return outputs.annotate_metric_result(
                result,
                metric="LandfallDisplacement",
                target_variable="surface_wind_speed",
                forecast_variable="surface_wind_speed",
                forecast_source="test_forecast",
                target_source="test_target",
                case_id_number=case_id,
                event_type="tropical_cyclone",
            )

        result_1 = make(1, 10.0, -80.0)
        result_2 = make(2, 50.0, -170.0)

        ds = outputs.results_to_dataset([result_1, result_2])

        assert "forecast_landfall_latitude" in ds.data_vars
        assert "forecast_landfall_latitude" not in ds.coords

        df = ds["forecast_landfall_latitude"].to_dataframe(name="value").reset_index()
        df = df.dropna(subset=["value"])
        case_1_vals = sorted(df[df["case_id_number"] == 1]["value"])
        case_2_vals = sorted(df[df["case_id_number"] == 2]["value"])
        assert case_1_vals == [10.0, 11.0]
        assert case_2_vals == [50.0, 51.0]

    def test_dataset_values_match_dataframe_values_for_each_result(self):
        same_var = {
            "target_variable": "2m_temperature",
            "forecast_variable": "2m_temperature",
        }
        results = [
            _result(
                metric="RMSE",
                case_id_number=1,
                values=(1.0, 2.0),
                coord_values=(0, 6),
                **same_var,
            ),
            _result(
                dims="init_time",
                metric="OnsetError",
                case_id_number=2,
                values=(5.0, 6.0),
                **same_var,
            ),
        ]
        df = outputs.results_to_dataframe(results)
        ds = outputs.results_to_dataset(results)

        for result in results:
            metadata = {
                c: result.coords[c].item()
                for c in (
                    "metric",
                    "forecast_source",
                    "target_source",
                    "case_id_number",
                )
            }
            var_name = outputs._variable_name(
                result.coords["forecast_variable"].item(),
                result.coords["target_variable"].item(),
            )
            own_dim = result.dims[0]

            for coord_value, value in zip(result.coords[own_dim].values, result.values):
                selectors = dict(metadata)
                selectors[own_dim] = coord_value
                for other_dim in ("lead_time", "init_time"):
                    if other_dim != own_dim and other_dim in ds.dims:
                        selectors[other_dim] = outputs._sentinel_for_dtype(
                            ds.coords[other_dim].dtype
                        )
                dataset_value = ds[var_name].sel(**selectors).compute().item()
                assert dataset_value == pytest.approx(value)

                df_row = df[
                    (df["metric"] == metadata["metric"])
                    & (df["case_id_number"] == metadata["case_id_number"])
                    & (df[own_dim] == coord_value)
                ]
                assert df_row["value"].iloc[0] == pytest.approx(dataset_value)

    def test_sparse_true_produces_sparse_backed_data_with_correct_values(self):
        result_1 = _result(case_id_number=1, values=(1.0, 2.0))
        result_2 = _result(case_id_number=2, values=(3.0, 4.0))
        ds = outputs.results_to_dataset([result_1, result_2], sparse=True)

        var = ds["surface_air_temperature_vs_2m_temperature"]
        assert isinstance(var.data, sparse.COO)
        assert np.isnan(var.data.fill_value)
        assert np.nansum(var.data.todense()) == pytest.approx(10.0)

    def test_result_is_dask_backed_by_default(self):
        ds = outputs.results_to_dataset([_result(case_id_number=1)])
        var = ds["surface_air_temperature_vs_2m_temperature"]
        assert isinstance(var.data, da.Array)

    def test_already_dask_backed_inputs_stay_dask_backed(self):
        result = _result(case_id_number=1)
        result = result.chunk()
        ds = outputs.results_to_dataset([result])
        var = ds["surface_air_temperature_vs_2m_temperature"]
        assert isinstance(var.data, da.Array)

    def test_warns_when_estimated_dense_size_exceeds_threshold(self, caplog):
        latitudes = np.arange(2000)
        longitudes = np.arange(2000)
        data = np.zeros((len(latitudes), len(longitudes)))
        result = xr.DataArray(
            data=data,
            dims=["latitude", "longitude"],
            coords={"latitude": latitudes, "longitude": longitudes},
        )
        result = outputs.annotate_metric_result(result, **METADATA_KWARGS)

        with caplog.at_level(logging.WARNING, logger="extremeweatherbench.outputs"):
            outputs.results_to_dataset([result], sparse=False)

        assert any("sparse=True" in r.getMessage() for r in caplog.records)

    def test_no_dense_size_warning_for_small_results(self, caplog):
        with caplog.at_level(logging.WARNING, logger="extremeweatherbench.outputs"):
            outputs.results_to_dataset([_result()], sparse=False)

        assert not any("sparse=True" in r.getMessage() for r in caplog.records)


class TestWriteResults:
    """Test write_results dispatching and on-disk round trips."""

    def test_unknown_output_format_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="csv"):
            outputs.write_results(
                [_result()], tmp_path / "out", output_format="parquet"
            )

    def test_csv_output_matches_dataframe_bytes(self, tmp_path):
        results = [_result(case_id_number=1)]
        expected_path = tmp_path / "expected.csv"
        outputs.results_to_dataframe(results).to_csv(expected_path, index=False)

        out_path = tmp_path / "out.csv"
        outputs.write_results(results, out_path, output_format="csv")

        assert out_path.read_bytes() == expected_path.read_bytes()

    def test_netcdf_round_trip_preserves_values_and_coords(self, tmp_path):
        results = [
            _result(metric="RMSE", case_id_number=1, values=(1.0, 2.0)),
            _result(
                dims="init_time",
                metric="OnsetError",
                case_id_number=2,
                values=(5.0, 6.0),
            ),
        ]
        expected_ds = outputs.results_to_dataset(results).compute()

        out_path = tmp_path / "out.nc"
        outputs.write_results(results, out_path, output_format="netcdf")

        actual = xr.open_dataset(out_path).compute()
        xr.testing.assert_allclose(expected_ds, actual, check_dim_order=False)
