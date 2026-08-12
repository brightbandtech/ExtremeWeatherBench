"""Tests for the outputs module."""

import logging

import numpy as np
import pandas as pd
import pytest
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
