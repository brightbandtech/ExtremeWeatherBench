"""Generate gridded t0-beta median hindcasts for ExtremeWeatherBench."""

# Ruff parses jaxtyping shape strings as forward annotations.
# ruff: noqa: F722

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

if TYPE_CHECKING:
    from jaxtyping import Float

MODEL_ID: str = "theforecastingcompany/t0-beta"


def generate_forecasts(
    history: xr.DataArray | xr.Dataset,
    init_times: Sequence[np.datetime64],
    predict: Callable[
        [Float[NDArray[np.float32], "batch time"], int],
        Float[NDArray[np.float32], "batch horizon"],
    ],
    *,
    context_length: int = 512,
    horizon: int = 40,
    batch_size: int = 32,
    grouping: Literal["joint", "independent"] = "joint",
) -> xr.Dataset:
    """Stack every variable's non-time dimensions into joint target variates.

    ``predict`` receives [V, time] and returns [V, horizon]. Surface variables
    are not broadcast over pressure levels. Original dimensions and attributes
    are restored after inference. The caller must configure joint rows as one
    group; independent mode splits rows into batches of ``batch_size``.
    Variables must already use EWB names and units. Only finite histories on
    a shared regular time axis and rectilinear grid are supported.
    """
    if min(context_length, horizon, batch_size) < 1:
        raise ValueError("context_length, horizon and batch_size must be positive")
    if grouping not in ("joint", "independent"):
        raise ValueError("grouping must be 'joint' or 'independent'")
    if isinstance(history, xr.DataArray):
        if history.name is None:
            raise ValueError("Provide a named variable")
        history = history.to_dataset()
    if not history.data_vars:
        raise ValueError("Provide at least one target variable")
    for reserved in ("init_time", "lead_time"):
        if (
            reserved in history.coords
            or reserved in history.dims
            or reserved in history.data_vars
        ):
            raise ValueError(f"{reserved} is reserved for forecast output")
    for name, field in history.data_vars.items():
        if not {"time", "latitude", "longitude"}.issubset(field.dims):
            raise ValueError(f"{name} must have time, latitude, longitude axes")
        for axis in field.dims:
            if axis not in history.coords or history[axis].dims != (axis,):
                raise ValueError(f"{axis} must be a one-dimensional coordinate")
            if history.sizes[axis] == 0 or history[axis].to_index().has_duplicates:
                raise ValueError(f"{axis} must be nonempty and unique")
    for axis in ("time", "latitude", "longitude"):
        if axis not in history.coords or history[axis].dims != (axis,):
            raise ValueError(f"{axis} must be a one-dimensional coordinate")
        if history.sizes[axis] == 0:
            raise ValueError(f"{axis} must not be empty")
    times: NDArray[np.datetime64] = history.time.values
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise ValueError("time must contain valid datetime64 values")
    steps: NDArray[np.timedelta64] = np.diff(times)
    if len(steps) == 0 or steps[0] <= np.timedelta64(0, "ns"):
        raise ValueError("At least two strictly increasing timestamps are required")
    if not np.all(steps == steps[0]):
        raise ValueError("time must be regular and strictly increasing")
    initializations: NDArray[np.datetime64] = np.asarray(init_times, dtype=times.dtype)
    if initializations.ndim != 1 or len(initializations) == 0:
        raise ValueError("Provide at least one initialization timestamp")
    if len(np.unique(initializations)) != len(initializations):
        raise ValueError("Initialization timestamps must be unique")
    indices: NDArray[np.int64] = np.searchsorted(times, initializations)
    if np.any(indices >= len(times)) or not np.array_equal(
        times[indices], initializations
    ):
        raise ValueError(
            "Every initialization must occur in the history time coordinate"
        )
    if np.any(indices < context_length - 1):
        raise ValueError("Insufficient history for the requested context_length")
    for axis in ("latitude", "longitude"):
        if not np.isfinite(history[axis].values).all():
            raise ValueError(f"{axis} must contain finite coordinates")
    if np.any(np.abs(history.latitude.values) > 90):
        raise ValueError("latitude must lie between -90 and 90 degrees")
    history = history.assign_coords(longitude=history.longitude % 360).sortby(
        "longitude"
    )
    for axis in ("latitude", "longitude"):
        if len(np.unique(history[axis])) != history.sizes[axis]:
            raise ValueError(f"{axis} must contain unique coordinates")
    fields: list[xr.DataArray] = [
        field.transpose(
            "time",
            *[
                axis
                for axis in field.dims
                if axis not in ("time", "latitude", "longitude")
            ],
            "latitude",
            "longitude",
        )
        for field in history.data_vars.values()
    ]
    widths: list[int] = [int(np.prod(field.shape[1:])) for field in fields]
    forecasts: Float[NDArray[np.float32], "init V horizon"] = np.empty(
        (len(initializations), sum(widths), horizon), dtype=np.float32
    )
    for init_index, stop in enumerate(indices):
        context: Float[NDArray[np.float32], "V time"] = np.concatenate(
            [
                np.asarray(
                    field.isel(time=slice(stop - context_length + 1, stop + 1)).values,
                    dtype=np.float32,
                )
                .reshape(context_length, width)
                .T
                for field, width in zip(fields, widths, strict=True)
            ]
        )
        if not np.isfinite(context).all():
            raise ValueError("History context must contain only finite values")
        output: Float[NDArray[np.float32], "grid horizon"] = np.empty(
            (sum(widths), horizon), dtype=np.float32
        )
        rows_per_call: int = len(context) if grouping == "joint" else batch_size
        for start in range(0, len(context), rows_per_call):
            batch: Float[NDArray[np.float32], "batch time"] = context[
                start : start + rows_per_call
            ]
            median: Float[NDArray[np.float32], "batch horizon"] = np.asarray(
                predict(batch, horizon), dtype=np.float32
            )
            if median.shape != (len(batch), horizon):
                raise ValueError("Predictor must return shape (batch, horizon)")
            if not np.isfinite(median).all():
                raise ValueError("Predictor returned nonfinite values")
            output[start : start + len(batch)] = median
        forecasts[init_index] = output
    result: xr.Dataset = xr.Dataset(attrs=history.attrs.copy())
    offset: int = 0
    for field, width in zip(fields, widths, strict=True):
        result[str(field.name)] = xr.DataArray(
            forecasts[:, offset : offset + width, :]
            .transpose(0, 2, 1)
            .reshape(len(initializations), horizon, *field.shape[1:]),
            dims=("init_time", "lead_time", *field.dims[1:]),
            coords={
                "init_time": initializations.astype("datetime64[ns]"),
                "lead_time": np.arange(1, horizon + 1) * steps[0],
                **{
                    name: coord
                    for name, coord in field.coords.items()
                    if "time" not in coord.dims
                },
            },
            attrs=field.attrs.copy(),
        )
        offset += width
    return result


def main() -> None:
    """Load a local history file and write EWB-compatible NetCDF forecasts."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variable",
        action="extend",
        nargs="+",
        help="Target variables; defaults to all data variables. May be repeated.",
    )
    parser.add_argument("--init-time", required=True, action="append")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Target variates per call in independent mode only",
    )
    parser.add_argument(
        "--grouping",
        choices=("joint", "independent"),
        default="joint",
        help="Jointly forecast all variables, levels and grid cells as targets in one group (default)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    args: argparse.Namespace = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        from packaging.version import Version
        from t0 import T0Forecaster
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional runtime: uv pip install 'tfc-t0>=0.5.0'"
        ) from exc
    runtime_version: str = version("tfc-t0")
    if Version(runtime_version) < Version("0.5.0"):
        raise RuntimeError("t0-beta requires tfc-t0>=0.5.0 for correct normalization")
    model: T0Forecaster = (
        T0Forecaster.from_pretrained(MODEL_ID, revision=args.revision)
        .to(args.device)
        .eval()
    )

    def predict(
        context: Float[NDArray[np.float32], "batch time"], horizon: int
    ) -> Float[NDArray[np.float32], "batch horizon"]:
        return (
            model.predict(
                context,
                horizon=horizon,
                quantile_levels=[0.5],
                group_ids=np.zeros(len(context), dtype=np.int64)
                if args.grouping == "joint"
                else None,
            )
            .median.cpu()
            .numpy()
        )

    with xr.open_dataset(args.input) as dataset:
        forecasts: xr.Dataset = generate_forecasts(
            dataset[args.variable] if args.variable else dataset,
            [np.datetime64(value) for value in args.init_time],
            predict,
            context_length=args.context_length,
            horizon=args.horizon,
            batch_size=args.batch_size,
            grouping=args.grouping,
        )
    forecasts.attrs.update(
        model=MODEL_ID,
        model_revision=args.revision,
        runtime_version=runtime_version,
        context_length=args.context_length,
        forecast_statistic="median",
        grouping=args.grouping,
        inference="all variables, levels and grid cells as targets in one group"
        if args.grouping == "joint"
        else "independent target variates",
    )
    forecasts.to_netcdf(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
