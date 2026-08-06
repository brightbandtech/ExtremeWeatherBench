"""Helpers for asserting that a result is still dask-backed."""

from typing import Any

import xarray as xr


def is_lazy(obj: Any) -> bool:
    """Whether an xarray object is still dask-backed and unevaluated."""
    if isinstance(obj, xr.Dataset):
        return bool(obj.chunks)
    if isinstance(obj, xr.DataArray):
        return obj.chunks is not None
    return hasattr(obj, "dask")


def assert_lazy(obj: Any, msg: str = "") -> None:
    """Fail if a result has been computed instead of staying lazy.

    Guards against reintroducing eager evaluation into paths that are meant
    to compose into a single dask graph, which is otherwise invisible until
    something runs out of memory on real data.
    """
    if not is_lazy(obj):
        raise AssertionError(f"expected a lazy result, got a computed one. {msg}")
