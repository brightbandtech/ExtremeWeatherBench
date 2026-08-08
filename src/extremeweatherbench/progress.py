"""Shared progress-reporting primitives for ExtremeWeatherBench runs.

This module owns the single unified progress bar shown while evaluating
``CaseOperator``s: a bar factory, a module-global registry of the one
currently active bar, and a dask callback that mirrors live task counts
into that bar's postfix. It must not import any other extremeweatherbench
module, so that utils.py and evaluate.py can both depend on it without
creating an import cycle.

The registry is a plain module global rather than something process-aware:
in a spawned worker process the registry starts out empty, so calls to
set_phase from that process are automatically silent no-ops, with no
env-var or PID-based detection required.
"""

import contextlib
import logging
import os
import time
from typing import Iterator, Optional

import dask.callbacks
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Guarantees a percentage, a fraction, and an elapsed<remaining ETA are
# always shown, regardless of terminal width or postfix content.
BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]{postfix}"
)

_active_bar: Optional[tqdm] = None
_phase_updates_allowed: bool = False
_current_phase: str = ""


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
        # The default EMA smoothing gives a wildly unstable ETA for a
        # handful of multi-minute, heterogeneous cases; an overall average
        # rate is far more useful here.
        smoothing=0,
        disable=disable,
    )


def register_bar(bar: tqdm, allow_phase_updates: bool = False) -> None:
    """Register bar as the single active bar for this process.

    Args:
        bar: The bar to register as active.
        allow_phase_updates: Whether set_phase() may write to this bar.
    """
    global _active_bar, _phase_updates_allowed, _current_phase
    _active_bar = bar
    _phase_updates_allowed = allow_phase_updates
    _current_phase = ""


def clear_bar() -> None:
    """Clear the active-bar registry so later calls become no-ops."""
    global _active_bar, _phase_updates_allowed, _current_phase
    _active_bar = None
    _phase_updates_allowed = False
    _current_phase = ""


@contextlib.contextmanager
def registered_bar(bar: tqdm, allow_phase_updates: bool = False) -> Iterator[tqdm]:
    """Register bar as active for the duration of the context.

    Args:
        bar: The bar to register as active.
        allow_phase_updates: Whether set_phase() may write to this bar.
            Kept False for parallel dispatch, since a thread-based joblib
            backend shares this registry with worker threads and
            concurrent cases would otherwise thrash a single postfix.

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

    A silent no-op when no bar is registered (e.g. in a spawned worker
    process, where this module-global registry is always empty) or when
    phase updates are disabled for the active bar.

    Args:
        text: The phase description to display, e.g. "case 12 | RMSE".
    """
    global _current_phase
    if _active_bar is None or not _phase_updates_allowed:
        return
    _current_phase = text
    _active_bar.set_postfix_str(text)


class DaskTaskPostfix(dask.callbacks.Callback):
    """Mirror live local-scheduler dask task counts into the active bar.

    Note: dask.callbacks only instruments the local schedulers (single
    threaded/threaded/multiprocessing get()); it never fires for
    dask.distributed computations, so those get no postfix coverage here.
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
        if _active_bar is None:
            return
        prefix = f"{_current_phase} | " if _current_phase else ""
        _active_bar.set_postfix_str(
            f"{prefix}dask {self.completed_tasks}/{self.total_tasks}"
        )
