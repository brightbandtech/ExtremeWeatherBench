"""Progress bar for ExtremeWeatherBench runs.

Builds the progress bar shown while evaluating ``CaseOperator``s, tracks
which bar is currently active, and updates it with dask task counts.

This module doesn't import any other extremeweatherbench module, so both
utils.py and evaluate.py can use it without an import cycle.

A worker process starts with no active bar registered, so calls to
set_phase from a worker only render locally if a sink has been
registered via register_sink; otherwise they publish nothing.
"""

import contextlib
import dataclasses
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional, Union

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
        self.sink: Optional[Callable[["ProgressEvent"], None]] = None
        self.case_id: Union[int, str] = ""
        self.total_steps: int = 0
        self.step: int = 0


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


def register_sink(
    sink: Optional[Callable[["ProgressEvent"], None]],
    case_id: Union[int, str] = "",
    total_steps: int = 0,
) -> None:
    """Route this process's progress events to sink.

    Called inside worker processes, where no bar exists to render to.

    Args:
        sink: Receives every ProgressEvent, or None to stop publishing.
        case_id: The case this process is currently computing.
        total_steps: How many steps that case will run.
    """
    _state.sink = sink
    _state.case_id = case_id
    _state.total_steps = total_steps
    _state.step = 0


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
    if _state.sink is not None:
        _state.step += 1
        _state.sink(
            ProgressEvent(
                case_id=_state.case_id,
                phase=text,
                step=_state.step,
                total_steps=_state.total_steps,
            )
        )
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


class DaskTaskSink(dask.callbacks.Callback):
    """Publish local-scheduler dask task counts to a progress sink.

    The worker-process counterpart of DaskTaskBar. Throttled because
    posttask callbacks run on the dask scheduler's own loop, so every
    publish is on the critical path of the compute.
    """

    def __init__(
        self,
        sink: Callable[["ProgressEvent"], None],
        case_id: Union[int, str],
        throttle_seconds: float = 0.5,
    ):
        super().__init__()
        self.sink = sink
        self.case_id = case_id
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
        self.sink(
            ProgressEvent(
                case_id=self.case_id,
                phase=_state.current_phase,
                step=_state.step,
                total_steps=_state.total_steps,
                dask_done=self.completed_tasks,
                dask_total=self.total_tasks,
            )
        )


@dataclasses.dataclass(frozen=True)
class ProgressEvent:
    """One progress update from wherever a case is being computed.

    Attributes:
        case_id: The case the event describes.
        phase: Human-readable description of the current step.
        step: How many steps of the case are done.
        total_steps: How many steps the case has in total.
        dask_done: Tasks completed in the in-flight dask compute.
        dask_total: Tasks in the in-flight dask compute.
        finished: True when the case is complete and its slot frees.
    """

    case_id: Union[int, str]
    phase: str = ""
    step: int = 0
    total_steps: int = 0
    dask_done: int = 0
    dask_total: int = 0
    finished: bool = False


class QueueSink:
    """Publish progress events to a cross-process queue.

    Used inside worker processes, where there is no bar to render to.
    Drops events rather than raising if the parent has gone away, since
    progress must never break an evaluation.
    """

    def __init__(self, event_queue) -> None:
        self.event_queue = event_queue

    def __call__(self, event: ProgressEvent) -> None:
        try:
            self.event_queue.put_nowait(event)
        except Exception:  # noqa: BLE001 - progress must never break a run
            logger.debug("Dropped progress event for case %s", event.case_id)


def supports_nested_bars() -> bool:
    """Report whether nested bars will render usefully here.

    Nested bars rely on cursor positioning, which produces unreadable
    output in CI logs and captured notebook cells.

    Returns:
        True when stderr is a terminal and progress is not disabled.
    """
    if os.environ.get("EWB_DISABLE_PROGRESS"):
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


class LogSink:
    """Report progress as throttled INFO logs instead of nested bars.

    Used when stderr is not a terminal, where cursor-positioned bars
    would produce thousands of unreadable lines.
    """

    def __init__(self, throttle_seconds: float = 5.0) -> None:
        self.throttle_seconds = throttle_seconds
        self._last_log: dict[Union[int, str], float] = {}

    def __call__(self, event: ProgressEvent) -> None:
        if event.finished:
            self._last_log.pop(event.case_id, None)
            return
        now = time.monotonic()
        previous = self._last_log.get(event.case_id, 0.0)
        if now - previous < self.throttle_seconds:
            return
        self._last_log[event.case_id] = now
        logger.info(
            "case %s: step %s/%s (%s)",
            event.case_id,
            event.step,
            event.total_steps,
            event.phase,
        )


class WorkerSlotRenderer:
    """Render worker progress events onto one fixed bar per worker slot.

    Loky reuses worker processes, so cases are mapped onto a fixed set of
    slots and the mapping is recycled as cases finish. Keeping the slot
    count fixed keeps the number of terminal lines stable.
    """

    def __init__(self, n_slots: int, disable: bool = False) -> None:
        self.slot_by_case: dict[Union[int, str], int] = {}
        self._free_slots = list(range(n_slots))
        self._bars = [
            make_case_step_bar(
                case_id="idle", total_steps=1, position=i + 1, disable=disable
            )
            for i in range(n_slots)
        ]
        self._queue: Optional[Any] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def handle(self, event: ProgressEvent) -> None:
        """Apply one event to the slot bars.

        Args:
            event: The event to render.
        """
        if event.finished:
            slot = self.slot_by_case.pop(event.case_id, None)
            if slot is not None:
                self._free_slots.append(slot)
                self._bars[slot].reset(total=1)
                self._bars[slot].set_description_str("  idle")
            return

        slot = self.slot_by_case.get(event.case_id)
        if slot is None:
            if not self._free_slots:
                # More cases in flight than slots; the case bar still
                # accounts for them, so drop the detail rather than
                # reflowing the display.
                return
            slot = self._free_slots.pop(0)
            self.slot_by_case[event.case_id] = slot
            self._bars[slot].reset(total=max(event.total_steps, 1))
            self._bars[slot].set_description_str(f"  case {event.case_id}")

        bar = self._bars[slot]
        bar.update(event.step - bar.n)
        if event.dask_total:
            bar.set_postfix_str(
                f"{event.phase} | dask {event.dask_done}/{event.dask_total}"
            )
        else:
            bar.set_postfix_str(event.phase)

    def start(self, event_queue) -> None:
        """Begin draining event_queue on a daemon thread.

        Args:
            event_queue: The cross-process queue to drain.
        """
        self._queue = event_queue
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self._queue is not None, "start() must be called before _drain()"
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                return
            self.handle(event)

    def close(self) -> None:
        """Stop draining and close every slot bar."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for bar in self._bars:
            bar.close()
