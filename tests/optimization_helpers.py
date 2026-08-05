"""Helpers for the optimization test suite.

These support tests that assert *performance contracts* rather than numeric
output: that a result is still lazy, that a dask graph does not grow with a
dimension's length, or that an expensive kernel executes once instead of
twice. Such assertions fail on unoptimized code and pass afterwards, which
is what makes them usable to drive an optimization test-first.

Numeric equality is handled separately by characterization fixtures, since
those pass both before and after an optimization and therefore cannot drive
one.
"""

import contextlib
import functools
from typing import Any, Callable, Iterator

import dask
import dask.array as dask_array
import numpy as np
import xarray as xr


def is_lazy(obj: Any) -> bool:
    """Return whether obj is still backed by an uncomputed dask graph."""
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        return dask.is_dask_collection(obj) and bool(obj.chunks)
    return dask.is_dask_collection(obj)


def assert_lazy(obj: Any, msg: str = "") -> None:
    """Assert obj is still a dask collection that has not been materialized."""
    if not is_lazy(obj):
        backing = type(getattr(obj, "data", obj)).__name__
        raise AssertionError(
            f"expected a lazy dask-backed result, got {backing}. {msg}".strip()
        )


def graph_size(obj: Any) -> int:
    """Number of tasks in obj's dask graph.

    Used to assert a graph does not grow proportionally with a dimension,
    which is how a per-element Python loop over a dimension shows up.
    """
    if not dask.is_dask_collection(obj):
        raise AssertionError("graph_size requires a dask collection")
    return len(obj.__dask_graph__())


class _TaskCounter(dask.callbacks.Callback):
    """Count tasks executed while computing a dask collection."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        self.count += 1


@contextlib.contextmanager
def count_task_executions() -> Iterator[_TaskCounter]:
    """Count dask tasks executed inside the context.

    Distinguishes a graph that is computed once from one that is rebuilt and
    recomputed by a second consumer.
    """
    counter = _TaskCounter()
    with counter:
        yield counter


class _CallCounter:
    """Records how many times a spied function was called."""

    def __init__(self) -> None:
        self.count = 0


@contextlib.contextmanager
def spy(target: Any, name: str) -> Iterator[_CallCounter]:
    """Count calls to target.name while leaving the real function in place.

    This wraps rather than replaces the function, so behavior under test stays
    real and only the call count is observed. Use it to assert that an
    expensive routine runs once per column rather than once per level, or that
    a predicate is not evaluated per grid segment.

    The replacement is a plain function rather than a Mock so that spied
    functions embedded in a dask graph stay tokenizable.
    """
    counter = _CallCounter()
    original = getattr(target, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        counter.count += 1
        return original(*args, **kwargs)

    setattr(target, name, wrapper)
    try:
        yield counter
    finally:
        setattr(target, name, original)


class _SizeRecorder(_CallCounter):
    """Records the largest materialized array a spied function was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.largest = 0


@contextlib.contextmanager
def spy_largest_materialized_input(target: Any, name: str) -> Iterator[_SizeRecorder]:
    """Track the biggest in-memory array passed to target.name.

    Lazy arguments are ignored, so this distinguishes a guard that hands the
    predicate a dask graph to fold into the computation from one that reads
    .values first and tests the assembled result. Both touch the same data, so
    a task count cannot tell them apart; only the second one puts a full-size
    array in memory.
    """
    recorder = _SizeRecorder()
    original = getattr(target, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        recorder.count += 1
        for arg in args:
            if dask.is_dask_collection(arg):
                continue
            size = getattr(arg, "size", 0)
            if isinstance(size, int):
                recorder.largest = max(recorder.largest, size)
        return original(*args, **kwargs)

    setattr(target, name, wrapper)
    try:
        yield recorder
    finally:
        setattr(target, name, original)


def make_chunked_dataarray(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    coords: dict | None = None,
    chunks: dict | None = None,
    seed: int = 0,
) -> xr.DataArray:
    """Build a small dask-backed DataArray for laziness assertions."""
    rng = np.random.default_rng(seed)
    data = dask_array.from_array(rng.standard_normal(shape), chunks="auto")
    da = xr.DataArray(data, dims=dims, coords=coords or {})
    if chunks:
        da = da.chunk(chunks)
    return da


def counted_calls(func: Callable) -> tuple[Callable, _CallCounter]:
    """Wrap func so calls are counted, returning the wrapper and counter."""
    counter = _CallCounter()

    def wrapper(*args, **kwargs):
        counter.count += 1
        return func(*args, **kwargs)

    return wrapper, counter
