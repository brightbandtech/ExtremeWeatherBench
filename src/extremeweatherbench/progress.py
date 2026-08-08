"""Progress bar for ExtremeWeatherBench runs.

Builds the progress bar shown while evaluating ``CaseOperator``s, tracks
which bar is currently active, and updates it with dask task counts.

This module doesn't import any other extremeweatherbench module, so both
utils.py and evaluate.py can use it without an import cycle.

A worker process starts with no active bar registered, so calls to
set_phase from a worker are automatically no-ops.
"""

import contextlib
import logging
import os
import time
from typing import Iterator, Optional

import dask.callbacks
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Always shows a percentage, fraction, and elapsed/remaining time.
BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]{postfix}"
)


class _ProgressState:
    """Container for the single active-bar registry, mutated in place.

    Grouping this mutable state in one object means register_bar/clear_bar/
    set_phase can update its attributes without rebinding module globals.
    """

    def __init__(self) -> None:
        self.active_bar: Optional[tqdm] = None
        self.phase_updates_allowed: bool = False
        self.current_phase: str = ""


_state = _ProgressState()


def make_case_bar(total: int, disable: bool = False) -> tqdm:
    """Build the single main-process bar tracking CaseOperator completion.

    Args:
        total: The number of CaseOperators the bar should track.
        disable: If True, suppress the bar's output.

    Returns:
        A configured, not-yet-registered tqdm bar.
    """
    disable = disable or bool(os.environ.get("EWB_DISABLE_PROGRESS"))
    return tqdm(
        total=total,
        desc="Evaluating cases",
        unit="case",
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
        # Use an overall average rate instead of the default smoothing,
        # which gives an unstable ETA for a handful of long, uneven cases.
        smoothing=0,
        disable=disable,
    )


def register_bar(bar: tqdm, allow_phase_updates: bool = False) -> None:
    """Register bar as the single active bar for this process.

    Args:
        bar: The bar to register as active.
        allow_phase_updates: Whether set_phase() may write to this bar.
    """
    _state.active_bar = bar
    _state.phase_updates_allowed = allow_phase_updates
    _state.current_phase = ""


def clear_bar() -> None:
    """Clear the active-bar registry so later calls become no-ops."""
    _state.active_bar = None
    _state.phase_updates_allowed = False
    _state.current_phase = ""


@contextlib.contextmanager
def registered_bar(bar: tqdm, allow_phase_updates: bool = False) -> Iterator[tqdm]:
    """Register bar as active for the duration of the context.

    Args:
        bar: The bar to register as active.
        allow_phase_updates: Whether set_phase() may write to this bar.
            Keep False for parallel dispatch, since concurrent cases would
            otherwise fight over a single postfix.

    Yields:
        The same bar, for convenience.
    """
    register_bar(bar, allow_phase_updates=allow_phase_updates)
    try:
        yield bar
    finally:
        clear_bar()


def set_phase(text: str) -> None:
    """Write the current phase into the active bar's postfix.

    Does nothing if no bar is registered (e.g. in a worker process) or
    if phase updates are disabled for the active bar.

    Args:
        text: The phase description to display, e.g. "case 12 | RMSE".
    """
    if _state.active_bar is None or not _state.phase_updates_allowed:
        return
    _state.current_phase = text
    _state.active_bar.set_postfix_str(text)


class DaskTaskPostfix(dask.callbacks.Callback):
    """Show live dask task counts in the active bar's postfix.

    Only works with local schedulers (single-threaded, threaded,
    multiprocessing); dask.distributed computations aren't covered.
    """

    def __init__(self, throttle_seconds: float = 0.5):
        super().__init__()
        self.throttle_seconds = throttle_seconds
        self.total_tasks = 0
        self.completed_tasks = 0
        self._last_write = 0.0

    def _start_state(self, dsk, state) -> None:
        self.total_tasks = len(dsk)
        self.completed_tasks = 0
        self._last_write = 0.0

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        self.completed_tasks += 1
        now = time.monotonic()
        if now - self._last_write < self.throttle_seconds:
            return
        self._last_write = now
        self._write_postfix()

    def _finish(self, dsk, state, failed) -> None:
        self._write_postfix()

    def _write_postfix(self) -> None:
        if _state.active_bar is None:
            return
        prefix = f"{_state.current_phase} | " if _state.current_phase else ""
        _state.active_bar.set_postfix_str(
            f"{prefix}dask {self.completed_tasks}/{self.total_tasks}"
        )
