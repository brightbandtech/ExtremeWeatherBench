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
from typing import Iterator, Optional, Union

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
        self.step_bar: Optional[tqdm] = None


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


def make_case_step_bar(
    case_id: Union[int, str],
    total_steps: int,
    position: int = 1,
    disable: bool = False,
) -> tqdm:
    """Build the nested bar tracking steps within a single case.

    Args:
        case_id: The case this bar reports on, shown in the description.
        total_steps: The number of steps the case will run.
        position: The tqdm line position, below the case bar.
        disable: If True, suppress the bar's output.

    Returns:
        A configured tqdm bar.
    """
    disable = disable or bool(os.environ.get("EWB_DISABLE_PROGRESS"))
    return tqdm(
        total=total_steps,
        desc=f"  case {case_id}",
        unit="step",
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        mininterval=0.5,
        position=position,
        leave=False,
        smoothing=0,
        disable=disable,
    )


def make_dask_task_bar(position: int = 2, disable: bool = False) -> tqdm:
    """Build the nested bar tracking dask tasks within the current step.

    The total is unknown until a compute starts, so the bar is created
    with total=0 and resized by the callback.

    Args:
        position: The tqdm line position, below the step bar.
        disable: If True, suppress the bar's output.

    Returns:
        A configured tqdm bar.
    """
    disable = disable or bool(os.environ.get("EWB_DISABLE_PROGRESS"))
    return tqdm(
        total=0,
        desc="    dask tasks",
        unit="task",
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        mininterval=0.5,
        position=position,
        leave=False,
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


def register_step_bar(bar: Optional[tqdm]) -> None:
    """Set (or clear) the nested step bar that set_phase advances.

    Args:
        bar: The step bar to advance, or None to stop advancing.
    """
    _state.step_bar = bar


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
    """Advance the step bar and write the phase to the active bar.

    The step bar (if registered) always advances, since it must keep
    working in parallel mode where the active bar disallows phase
    updates. Writing to the active bar's postfix stays gated on
    phase_updates_allowed, so concurrent cases don't fight over it.

    Args:
        text: The phase description to display, e.g. "case 12 | RMSE".
    """
    _state.current_phase = text
    if _state.step_bar is not None:
        # Clamp so an unexpected extra phase can't overshoot the total.
        if _state.step_bar.n < (_state.step_bar.total or 0):
            _state.step_bar.update(1)
        _state.step_bar.set_postfix_str(text)
    if _state.active_bar is None or not _state.phase_updates_allowed:
        return
    _state.active_bar.set_postfix_str(text)


class DaskTaskBar(dask.callbacks.Callback):
    """Drive a nested tqdm bar from local-scheduler dask task events.

    Only works with local schedulers (single-threaded, threaded,
    multiprocessing); dask.distributed computations aren't covered.

    The callbacks run on the scheduler's own loop, so writes are
    throttled to keep rendering off the critical path.
    """

    def __init__(self, bar: tqdm, throttle_seconds: float = 0.5):
        super().__init__()
        self.bar = bar
        self.throttle_seconds = throttle_seconds
        self.total_tasks = 0
        self.completed_tasks = 0
        self._last_write = 0.0

    def _start_state(self, dsk, state) -> None:
        self.total_tasks = len(dsk)
        self.completed_tasks = 0
        self._last_write = 0.0
        self.bar.reset(total=self.total_tasks)

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        self.completed_tasks += 1
        now = time.monotonic()
        if now - self._last_write < self.throttle_seconds:
            return
        self._last_write = now
        self._sync_bar()

    def _finish(self, dsk, state, failed) -> None:
        self._sync_bar()

    def _sync_bar(self) -> None:
        self.bar.update(self.completed_tasks - self.bar.n)


# Deprecated alias; DaskTaskBar renders to a bar instead of a postfix.
DaskTaskPostfix = DaskTaskBar
