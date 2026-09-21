"""Tests for preparing t0-beta forecasts without loading model weights."""

from pathlib import Path
import sys
import types
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from data_prep.t0_forecasts import generate_forecasts
from data_prep import t0_forecasts


def make_history() -> xr.DataArray:
    times: NDArray[np.datetime64] = np.arange(
        np.datetime64("2024-06-01T00", "h"),
        np.datetime64("2024-06-03T00", "h"),
        np.timedelta64(6, "h"),
    )
    values: NDArray[np.float64] = np.arange(48, dtype=float).reshape(8, 2, 3)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [50.0, 51.0], "longitude": [0, 90, 180]},
        name="surface_air_temperature",
        attrs={"units": "K"},
    )


def persistence(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
    return np.repeat(context[:, -1:], horizon, axis=1)


def test_batches_preserve_gridpoints_and_use_only_available_history() -> None:
    history: xr.DataArray = make_history()
    batches: list[NDArray[np.float32]] = []

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        batches.append(context.copy())
        return persistence(context, horizon)

    init_times: list[np.datetime64] = [history.time.values[2], history.time.values[4]]
    result: xr.Dataset = generate_forecasts(
        history,
        init_times,
        predict,
        context_length=3,
        horizon=2,
        batch_size=4,
        grouping="independent",
    )

    assert result.sizes == {
        "init_time": 2,
        "lead_time": 2,
        "latitude": 2,
        "longitude": 3,
    }
    np.testing.assert_array_equal(result.init_time.values, init_times)
    np.testing.assert_array_equal(
        result.lead_time.values, np.array([6, 12], dtype="timedelta64[h]")
    )
    np.testing.assert_array_equal(result.latitude.values, history.latitude.values)
    np.testing.assert_array_equal(result.longitude.values, history.longitude.values)
    assert len(batches) == 4
    assert [batch.shape for batch in batches] == [(4, 3), (2, 3), (4, 3), (2, 3)]
    for index, time_index in enumerate([2, 4]):
        contexts: NDArray[np.float32] = np.concatenate(
            batches[2 * index : 2 * index + 2]
        )
        expected: NDArray[np.float64] = (
            history.values[time_index - 2 : time_index + 1].reshape(3, 6).T
        )
        np.testing.assert_array_equal(contexts, expected)
        actual: NDArray[np.float32] = (
            result.surface_air_temperature.isel(init_time=index)
            .transpose("lead_time", "latitude", "longitude")
            .values
        )
        np.testing.assert_array_equal(
            actual, np.repeat(history.values[time_index][None], 2, axis=0)
        )


def test_future_observations_do_not_change_forecast() -> None:
    history: xr.DataArray = make_history()
    modified: xr.DataArray = history.copy(deep=True)
    modified.values[3:] = -99999
    init_times: list[np.datetime64] = [history.time.values[2]]
    original: xr.Dataset = generate_forecasts(
        history, init_times, persistence, context_length=3, horizon=2
    )
    changed: xr.Dataset = generate_forecasts(
        modified, init_times, persistence, context_length=3, horizon=2
    )
    xr.testing.assert_equal(original, changed)


def test_longitude_normalization_preserves_values() -> None:
    history: xr.DataArray = make_history().assign_coords(longitude=[-90, 0, 90])
    result: xr.Dataset = generate_forecasts(
        history, [history.time.values[2]], persistence, context_length=3, horizon=1
    )
    np.testing.assert_array_equal(result.longitude.values, [0, 90, 270])
    actual: NDArray[np.float32] = (
        result.surface_air_temperature.isel(init_time=0, lead_time=0)
        .transpose("latitude", "longitude")
        .values
    )
    np.testing.assert_array_equal(actual, history.values[2][:, [1, 2, 0]])


@pytest.mark.parametrize("parameter", ["context_length", "horizon", "batch_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_nonpositive_counts(parameter: str, value: int) -> None:
    history: xr.DataArray = make_history()
    options: dict[str, int] = {"context_length": 3, "horizon": 2, "batch_size": 4}
    options[parameter] = value
    with pytest.raises(ValueError):
        generate_forecasts(
            history,
            [history.time.values[2]],
            persistence,
            context_length=options["context_length"],
            horizon=options["horizon"],
            batch_size=options["batch_size"],
        )


def test_rejects_irregular_history() -> None:
    history: xr.DataArray = make_history().isel(time=[0, 1, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[3]], persistence, context_length=3
        )


def test_rejects_insufficient_history() -> None:
    history: xr.DataArray = make_history()
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[1]], persistence, context_length=3
        )


def test_rejects_initialization_not_in_history() -> None:
    history: xr.DataArray = make_history()
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [np.datetime64("2024-06-01T13")], persistence, context_length=2
        )


def test_rejects_empty_initializations() -> None:
    with pytest.raises(ValueError):
        generate_forecasts(make_history(), [], persistence, context_length=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_predictions(bad_value: float) -> None:
    history: xr.DataArray = make_history()

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        return np.full((context.shape[0], horizon), bad_value, dtype=np.float32)

    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[2]], predict, context_length=3, horizon=2
        )


@pytest.mark.parametrize("shape", [(1, 2), (4, 3), (4, 2, 1)])
def test_rejects_wrong_prediction_shape(shape: tuple[int, ...]) -> None:
    history: xr.DataArray = make_history()

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        return np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        generate_forecasts(
            history,
            [history.time.values[2]],
            predict,
            context_length=3,
            horizon=2,
            batch_size=4,
        )


def test_transposed_history_preserves_forecasts() -> None:
    history: xr.DataArray = make_history()
    init_times: list[np.datetime64] = [history.time.values[2]]
    expected: xr.Dataset = generate_forecasts(
        history, init_times, persistence, context_length=3, horizon=2
    )
    actual: xr.Dataset = generate_forecasts(
        history.transpose("longitude", "time", "latitude"),
        init_times,
        persistence,
        context_length=3,
        horizon=2,
    )
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_context(bad_value: float) -> None:
    history: xr.DataArray = make_history()
    history.values[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[2]], persistence, context_length=3, horizon=2
        )


def test_netcdf_roundtrip_preserves_forecast(tmp_path: Path) -> None:
    history: xr.DataArray = make_history()
    result: xr.Dataset = generate_forecasts(
        history, [history.time.values[2]], persistence, context_length=3, horizon=2
    )
    result.to_netcdf(tmp_path / "forecast.nc")
    with xr.open_dataset(tmp_path / "forecast.nc", decode_timedelta=True) as restored:
        xr.testing.assert_equal(result, restored)


def test_joint_group_preserves_cross_cell_context_and_all_targets() -> None:
    history: xr.DataArray = make_history()
    batches: list[NDArray[np.float32]] = []

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        batches.append(context.copy())
        return np.repeat(context[:, -1:] + context[:, -1].mean(), horizon, axis=1)

    result: xr.Dataset = generate_forecasts(
        history,
        [history.time.values[2], history.time.values[4]],
        predict,
        context_length=3,
        horizon=2,
        batch_size=2,
    )
    assert [batch.shape for batch in batches] == [(6, 3), (6, 3)]
    for index, stop in enumerate([2, 4]):
        np.testing.assert_array_equal(
            batches[index], history.values[stop - 2 : stop + 1].reshape(3, 6).T
        )
        last: NDArray[np.float64] = history.values[stop]
        expected: NDArray[np.float64] = np.repeat((last + last.mean())[None], 2, axis=0)
        np.testing.assert_array_equal(
            result.surface_air_temperature.isel(init_time=index).values, expected
        )


@pytest.mark.parametrize("selection", ["single", "multiple", "all"])
@pytest.mark.parametrize("grouping", ["joint", "independent"])
def test_cli_passes_group_ids_for_all_grid_targets(
    grouping: Literal["joint", "independent"],
    selection: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history: xr.DataArray = make_history()
    input_path: Path = tmp_path / "history.nc"
    output_path: Path = tmp_path / "forecast.nc"
    dataset: xr.Dataset = history.to_dataset()
    dataset["eastward_wind"] = xr.concat(
        [history + 100, history + 200], dim=xr.IndexVariable("level", [500, 850])
    )
    dataset.to_netcdf(input_path)
    targets: int = 6 if selection == "single" else 18
    model: MagicMock = MagicMock()

    def predict(
        context: NDArray[np.float32],
        *,
        horizon: int,
        quantile_levels: list[float],
        group_ids: NDArray[np.int64] | None,
    ) -> MagicMock:
        assert quantile_levels == [0.5]
        if grouping == "joint":
            assert len(context) == targets
            np.testing.assert_array_equal(group_ids, np.zeros(targets, dtype=np.int64))
        else:
            assert len(context) == 2
            assert group_ids is None
        result: MagicMock = MagicMock()
        result.median.cpu.return_value.numpy.return_value = persistence(
            context, horizon
        )
        return result

    model.predict.side_effect = predict
    factory: MagicMock = MagicMock()
    factory.from_pretrained.return_value.to.return_value.eval.return_value = model
    runtime: types.ModuleType = types.ModuleType("t0")
    setattr(runtime, "T0Forecaster", factory)
    monkeypatch.setitem(sys.modules, "t0", runtime)
    monkeypatch.setattr(t0_forecasts, "version", lambda name: "0.5.0")
    arguments: list[str] = [
        "t0_forecasts.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--init-time",
        "2024-06-01T12",
        "--context-length",
        "3",
        "--horizon",
        "2",
        "--batch-size",
        "2",
    ]
    if selection == "single":
        arguments.extend(["--variable", "surface_air_temperature"])
    elif selection == "multiple":
        arguments.extend(
            ["--variable", "surface_air_temperature", "--variable", "eastward_wind"]
        )
    if grouping == "independent":
        arguments.extend(["--grouping", "independent"])
    monkeypatch.setattr(sys, "argv", arguments)
    t0_forecasts.main()
    assert model.predict.call_count == (1 if grouping == "joint" else targets // 2)
    with xr.open_dataset(output_path, decode_timedelta=True) as result:
        assert result.attrs["grouping"] == grouping
        assert set(result.data_vars) == (
            {"surface_air_temperature"}
            if selection == "single"
            else set(dataset.data_vars)
        )
        if selection != "single":
            np.testing.assert_array_equal(
                result.eastward_wind.values[0, 0],
                dataset.eastward_wind.isel(time=2).values,
            )
        np.testing.assert_array_equal(
            result.surface_air_temperature.values[0, 0], history.values[2]
        )


def test_mixed_variables_levels_share_one_group_and_restore_dimensions(
    tmp_path: Path,
) -> None:
    surface: xr.DataArray = make_history().assign_coords(longitude=[-90, 0, 90])
    upper: xr.DataArray = xr.concat(
        [surface + 100, surface + 200], dim=xr.IndexVariable("pressure", [500, 850])
    ).transpose("longitude", "pressure", "time", "latitude")
    upper.attrs = {"units": "m s-1"}
    history: xr.Dataset = xr.Dataset(
        {"surface_air_temperature": surface, "eastward_wind": upper}
    )
    history.pressure.attrs["units"] = "hPa"
    calls: list[NDArray[np.float32]] = []

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        calls.append(context.copy())
        return persistence(context, horizon) + context[:, -1].mean()

    result: xr.Dataset = generate_forecasts(
        history,
        [surface.time.values[2], surface.time.values[4]],
        predict,
        context_length=3,
        horizon=2,
        batch_size=2,
    )
    assert [call.shape for call in calls] == [(18, 3), (18, 3)]
    assert result.surface_air_temperature.dims == (
        "init_time",
        "lead_time",
        "latitude",
        "longitude",
    )
    assert result.eastward_wind.sizes["pressure"] == 2
    normalized: xr.Dataset = history.assign_coords(
        longitude=history.longitude % 360
    ).sortby("longitude")
    for index, stop in enumerate([2, 4]):
        expected_context: NDArray[np.float32] = np.concatenate(
            [
                field.transpose("time", ...)
                .isel(time=slice(stop - 2, stop + 1))
                .values.reshape(3, -1)
                .T
                for field in normalized.data_vars.values()
            ]
        )
        # Row order may differ; every real series must appear exactly once.
        np.testing.assert_array_equal(
            np.sort(calls[index], axis=0), np.sort(expected_context, axis=0)
        )
        for name in history.data_vars:
            expected: xr.DataArray = (
                normalized[name].isel(time=stop, drop=True)
                + expected_context[:, -1].mean()
            )
            actual: xr.DataArray = (
                result[name]
                .isel(init_time=index, lead_time=0, drop=True)
                .transpose(*expected.dims)
            )
            xr.testing.assert_allclose(actual, expected)
            assert result[name].attrs == history[name].attrs
    assert result.pressure.attrs == history.pressure.attrs
    result.to_netcdf(tmp_path / "mixed.nc")
    with xr.open_dataset(tmp_path / "mixed.nc", decode_timedelta=True) as restored:
        xr.testing.assert_identical(result, restored)


def test_rejects_static_fields_in_target_dataset() -> None:
    history: xr.Dataset = make_history().to_dataset()
    history["static"] = history.surface_air_temperature.isel(time=0, drop=True)
    with pytest.raises(ValueError, match="time"):
        generate_forecasts(
            history, [history.time.values[2]], persistence, context_length=3
        )


def test_multiple_extra_axes_and_future_values_do_not_leak() -> None:
    surface: xr.DataArray = make_history()
    upper: xr.DataArray = surface.expand_dims(level=[300, 500], member=[0, 1, 2]).copy(
        deep=True
    )
    history: xr.Dataset = xr.Dataset(
        {"surface_air_temperature": surface, "air_temperature": upper}
    )
    modified: xr.Dataset = history.copy(deep=True)
    modified["surface_air_temperature"].loc[{"time": history.time.values[3:]}] = np.nan
    modified["air_temperature"].loc[{"time": history.time.values[3:]}] = np.nan
    original: xr.Dataset = generate_forecasts(
        history, [history.time.values[2]], persistence, context_length=3, horizon=2
    )
    actual: xr.Dataset = generate_forecasts(
        modified, [history.time.values[2]], persistence, context_length=3, horizon=2
    )
    xr.testing.assert_identical(actual, original)
    assert actual.air_temperature.dims == (
        "init_time",
        "lead_time",
        "level",
        "member",
        "latitude",
        "longitude",
    )
    assert actual.air_temperature.shape == (1, 2, 2, 3, 2, 3)


@pytest.mark.parametrize("axis", ["init_time", "lead_time"])
def test_rejects_reserved_auxiliary_coordinate_before_prediction(axis: str) -> None:
    history: xr.DataArray = make_history().assign_coords({axis: 0})

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        raise AssertionError("Invalid coordinates must be rejected before inference")

    with pytest.raises(ValueError, match="reserved"):
        generate_forecasts(history, [history.time.values[2]], predict, context_length=3)
