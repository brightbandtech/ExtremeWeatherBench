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
import logging.handlers
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import dask.callbacks
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Always shows a percentage, fraction, and elapsed/remaining time.
BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]{postfix}"
)

# Repaint budget for the fast-moving inner bars. Painting every task
# costs ~50us against ~0.4us for a throttled update(), so the counter is
# advanced on every task and only the repaint is rate-limited. 20fps
# reads as continuous while keeping terminal writes off the hot path.
FAST_BAR_MININTERVAL = 0.05

# Pair every fast bar with miniters=1 so mininterval is the only gate.
# Left at 0, tqdm auto-tunes miniters, and with smoothing=0 that path is
# miniters = max(miniters, dn): a ratchet that never falls back, so one
# quick burst makes the bar progressively less responsive for the rest
# of the graph.
FAST_BAR_MINITERS = 1


def _repaint(bar: tqdm) -> None:
    """Force an immediate repaint, bypassing tqdm's mininterval throttle.

    tqdm.refresh() repaints the terminal but does not update
    last_print_n (only update()'s own throttle check does that), so a
    caller checking last_print_n to see whether a frame was actually
    painted would still see it as stale. Setting it here keeps that
    bookkeeping consistent with the forced repaint.

    Args:
        bar: The bar to repaint.
    """
    bar.refresh()
    bar.last_print_n = bar.n


class _ProgressState:
    """Container for the single active-bar registry, mutated in place.

    Grouping this mutable state in one object means register_bar/clear_bar/
    set_phase can update its attributes without rebinding module globals.
    """

    def __init__(self) -> None:
        self.active_bar: tqdm | None = None
        self.phase_updates_allowed: bool = False
        self.current_phase: str = ""
        self.step_bar: tqdm | None = None
        self.sink: Callable[[ProgressEvent], None] | None = None
        self.case_id: int | str = ""
        self.total_steps: int = 0
        self.step: int = 0
        self.slot_key: int | str = ""
        self.label: str = ""


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


def make_precompute_bar(total: int, disable: bool = False) -> tqdm:
    """Build the bar tracking unique target pipelines before parallel work.

    Args:
        total: The number of unique targets the bar should track.
        disable: If True, suppress the bar's output.

    Returns:
        A configured, not-yet-registered tqdm bar.
    """
    disable = disable or bool(os.environ.get("EWB_DISABLE_PROGRESS"))
    return tqdm(
        total=total,
        desc="Precomputing targets",
        unit="target",
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        smoothing=0,
        disable=disable,
    )


def make_case_step_bar(
    case_id: int | str,
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
        mininterval=FAST_BAR_MININTERVAL,
        miniters=FAST_BAR_MINITERS,
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
        mininterval=FAST_BAR_MININTERVAL,
        miniters=FAST_BAR_MINITERS,
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


def register_step_bar(bar: tqdm | None) -> None:
    """Set (or clear) the nested step bar that set_phase advances.

    Args:
        bar: The step bar to advance, or None to stop advancing.
    """
    _state.step_bar = bar


def register_sink(
    sink: Callable[["ProgressEvent"], None] | None,
    case_id: int | str = "",
    total_steps: int = 0,
    slot_key: int | str = "",
    label: str = "",
) -> None:
    """Route this process's progress events to sink.

    Called inside worker processes, where no bar exists to render to.

    Args:
        sink: Receives every ProgressEvent, or None to stop publishing.
        case_id: The case this process is currently computing.
        total_steps: How many steps that case will run.
        slot_key: Unique run-scoped key identifying this dispatch, since
            two CaseOperators (e.g. one case with two EvaluationObjects)
            can share the same case_id.
        label: Precomputed display label for this dispatch.
    """
    _state.sink = sink
    _state.case_id = case_id
    _state.total_steps = total_steps
    _state.step = 0
    _state.slot_key = slot_key
    _state.label = label


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
                slot_key=_state.slot_key,
                label=_state.label,
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

    The callbacks run on the scheduler's own loop, so rendering is kept
    off the critical path. Rate limiting is left to the bar's own
    mininterval rather than duplicated here: every task advances the
    counter, and tqdm decides when that reaches the terminal. Setting
    throttle_seconds above 0 additionally drops task events before the
    bar sees them, which makes the count itself lag.
    """

    def __init__(self, bar: tqdm, throttle_seconds: float = 0.0):
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
        if self.throttle_seconds:
            now = time.monotonic()
            if now - self._last_write < self.throttle_seconds:
                return
            self._last_write = now
        self._sync_bar()

    def _finish(self, dsk, state, failed) -> None:
        # Defensive: on success, a shortfall against len(dsk) shouldn't
        # leave the bar visibly incomplete. Never fires today (verified
        # empirically across several graph shapes) and must not fire on
        # the failure path, where a partial bar is the honest display.
        if not failed and self.completed_tasks < (self.bar.total or 0):
            self.bar.total = self.completed_tasks
        self._sync_bar()
        # The last update() may have been throttled by mininterval,
        # leaving the terminal on a stale mid-graph value even though
        # self.n reached the total; force the final frame to paint.
        _repaint(self.bar)

    def _sync_bar(self) -> None:
        self.bar.update(self.completed_tasks - self.bar.n)


# Deprecated alias; DaskTaskBar renders to a bar instead of a postfix.
DaskTaskPostfix = DaskTaskBar


class DaskTaskSink(dask.callbacks.Callback):
    """Publish local-scheduler dask task counts to a progress sink.

    The worker-process counterpart of DaskTaskBar. Unlike that class
    this stays throttled, because a publish crosses a process boundary:
    a Manager-queue put measures ~57us, as costly as a full repaint, and
    posttask runs on the dask scheduler's own loop, so an unthrottled
    publish would put that on the critical path of every task.

    One case typically runs many sequential dask graphs. Alongside the
    per-graph done/total that the slot bar renders, this also publishes
    a count that accumulates across graphs and never resets, for sinks
    that want a strictly monotonic liveness signal instead.
    """

    def __init__(
        self,
        sink: Callable[["ProgressEvent"], None],
        case_id: int | str,
        throttle_seconds: float = FAST_BAR_MININTERVAL,
    ):
        super().__init__()
        self.sink = sink
        self.case_id = case_id
        self.throttle_seconds = throttle_seconds
        self.total_tasks = 0
        self.completed_tasks = 0
        self.cumulative_completed_tasks = 0
        self._last_write = 0.0

    def _start_state(self, dsk, state) -> None:
        # Per-graph bookkeeping only; cumulative_completed_tasks must
        # keep counting across graphs, so it is never reset here.
        self.total_tasks = len(dsk)
        self.completed_tasks = 0
        self._last_write = 0.0

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        self.completed_tasks += 1
        self.cumulative_completed_tasks += 1
        now = time.monotonic()
        if now - self._last_write < self.throttle_seconds:
            return
        self._last_write = now
        self._publish()

    def _finish(self, dsk, state, failed) -> None:
        # The throttle nearly always swallows a graph's last tasks, so
        # without an unthrottled publish here the slot would sit on a
        # partial fraction until the next graph resets it. On failure
        # the real total stays, since a partial count is honest there.
        if not failed and self.completed_tasks < self.total_tasks:
            self.total_tasks = self.completed_tasks
        self._publish()

    def _publish(self) -> None:
        self.sink(
            ProgressEvent(
                case_id=self.case_id,
                slot_key=_state.slot_key,
                label=_state.label,
                phase=_state.current_phase,
                step=_state.step,
                total_steps=_state.total_steps,
                dask_done=self.completed_tasks,
                dask_total=self.total_tasks,
                dask_tasks_done=self.cumulative_completed_tasks,
            )
        )


@dataclasses.dataclass(frozen=True)
class ProgressEvent:
    """One progress update from wherever a case is being computed.

    Attributes:
        case_id: The case the event describes, for display only. Two
            dispatches (e.g. one case with two EvaluationObjects) can
            share a case_id, so this must not be used as a slot key.
        slot_key: Unique run-scoped key identifying this dispatch.
        label: Precomputed display label for the slot bar description.
        phase: Human-readable description of the current step.
        step: How many steps of the case are done.
        total_steps: How many steps the case has in total.
        dask_done: Tasks completed in the in-flight dask graph.
        dask_total: Tasks in the in-flight dask graph.
        dask_tasks_done: Cumulative dask tasks completed across every
            graph run so far for this dispatch; never decreases. Not
            rendered by the slot bar, which shows dask_done/dask_total.
        finished: True when the case is complete and its slot frees.
    """

    case_id: int | str
    slot_key: int | str = ""
    label: str = ""
    phase: str = ""
    step: int = 0
    total_steps: int = 0
    dask_done: int = 0
    dask_total: int = 0
    dask_tasks_done: int = 0
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
    output in CI logs and captured notebook cells, so they are gated on
    stderr being a terminal. EWB_FORCE_PROGRESS overrides that gate for
    terminals that misreport isatty, or to capture bar output
    deliberately; EWB_DISABLE_PROGRESS still wins over it.

    Returns:
        True when nested bars should be rendered.
    """
    if os.environ.get("EWB_DISABLE_PROGRESS"):
        return False
    if os.environ.get("EWB_FORCE_PROGRESS"):
        return True
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


class LogSink:
    """Report progress as throttled INFO logs instead of nested bars.

    Used when stderr is not a terminal, where cursor-positioned bars
    would produce thousands of unreadable lines.
    """

    def __init__(self, throttle_seconds: float = 5.0) -> None:
        self.throttle_seconds = throttle_seconds
        self._last_log: dict[int | str, float] = {}

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
        self.slot_by_case: dict[int | str, int] = {}
        self._free_slots = list(range(n_slots))
        self._bars = [
            make_case_step_bar(
                case_id="", total_steps=1, position=i + 1, disable=disable
            )
            for i in range(n_slots)
        ]
        for bar in self._bars:
            # An unclaimed slot renders as a blank reserved line rather
            # than a "case idle" placeholder, since that's just noise
            # before anything has actually started.
            bar.set_description_str("")
            bar.bar_format = "{desc}"
        self._queue: Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def handle(self, event: ProgressEvent) -> None:
        """Apply one event to the slot bars.

        Events key on slot_key, not case_id: build_case_operators can
        emit multiple CaseOperators sharing a case_id (one case, several
        EvaluationObjects), and those must not collide onto one slot.

        Args:
            event: The event to render.
        """
        # slot_key defaults to "" for callers that never set it (e.g.
        # serial-mode helpers), so fall back to case_id in that case.
        key = event.slot_key if event.slot_key != "" else event.case_id

        if event.finished:
            slot = self.slot_by_case.pop(key, None)
            if slot is not None:
                self._free_slots.append(slot)
                # Leave the bar showing the case's final state; it is
                # only reset once a new case actually claims this slot,
                # so finishing mid-run doesn't cause visible churn. The
                # last update before this may have been throttled, so
                # force it to paint before the slot sits idle.
                _repaint(self._bars[slot])
            return

        slot = self.slot_by_case.get(key)
        if slot is None:
            if not self._free_slots:
                # More cases in flight than slots; the case bar still
                # accounts for them, so drop the detail rather than
                # reflowing the display.
                return
            slot = self._free_slots.pop(0)
            self.slot_by_case[key] = slot
            bar = self._bars[slot]
            bar.bar_format = BAR_FORMAT
            bar.reset(total=max(event.total_steps, 1))
            bar.set_description_str(f"  {event.label or f'case {event.case_id}'}")

        bar = self._bars[slot]
        bar.update(event.step - bar.n)
        # The description already names the case, so drop the phase's
        # own "case N | " prefix rather than printing it twice on a line
        # that is already close to the terminal width.
        phase = event.phase.removeprefix(f"case {event.case_id} | ")
        if event.dask_total:
            bar.set_postfix_str(f"{phase} | dask {event.dask_done}/{event.dask_total}")
        else:
            bar.set_postfix_str(phase)

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


@contextlib.contextmanager
def captured_warnings() -> Iterator[None]:
    """Route warnings.warn() through logging for the duration of the block.

    Idempotent with respect to a caller that already enabled capture: in
    that case this is a no-op on both enter and exit, so a library caller
    with its own warning handling is left untouched.
    """
    already_enabled = getattr(logging, "_warnings_showwarning", None) is not None
    if not already_enabled:
        logging.captureWarnings(True)
    try:
        yield
    finally:
        if not already_enabled:
            logging.captureWarnings(False)


class _ForwardingQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that drops records instead of raising or printing.

    Progress and logging must never break an evaluation, so a full queue
    or a parent that has gone away is dropped silently rather than
    falling back to the base class's default stderr traceback.
    """

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Exception:  # noqa: BLE001, S110
            pass


@contextlib.contextmanager
def forwarding_logs_to(log_queue: Any) -> Iterator[None]:
    """Route this process's log records (and warnings) to log_queue.

    Swaps the root logger's handlers for a single QueueHandler and
    enables warning capture, so every logger.* call and warnings.warn()
    call in this process reaches the parent instead of this process's
    own stderr. Restores the prior handlers on exit, even if the
    wrapped block raises.

    Args:
        log_queue: Cross-process queue to publish LogRecords to.
    """
    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    root.handlers = [_ForwardingQueueHandler(log_queue)]
    try:
        with captured_warnings():
            yield
    finally:
        root.handlers = previous_handlers


class LogQueueListener:
    """Drain worker log records and re-emit them via the parent's loggers.

    Uses handle() rather than log() because the worker already applied
    its own effective-level check, so handle() skips a redundant one.
    """

    def __init__(self) -> None:
        self._queue: Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, log_queue: Any) -> None:
        """Begin draining log_queue on a daemon thread.

        Args:
            log_queue: The cross-process queue of LogRecords to drain.
        """
        self._queue = log_queue
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        assert self._queue is not None, "start() must be called before _drain()"
        while not self._stop.is_set():
            try:
                record = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                return
            except Exception:  # noqa: BLE001, S112
                continue
            self._emit(record)

    def _emit(self, record: logging.LogRecord) -> None:
        try:
            logging.getLogger(record.name).handle(record)
        except Exception:  # noqa: BLE001, S110
            pass

    def close(self, drain_deadline: float = 2.0) -> None:
        """Stop draining, then flush any records still sitting in the queue.

        Args:
            drain_deadline: Max seconds to spend on the final flush, so a
                wedged queue can't hang shutdown.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._queue is None:
            return
        deadline = time.monotonic() + drain_deadline
        while time.monotonic() < deadline:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError, BrokenPipeError):
                break
            except Exception:  # noqa: BLE001 - logging must never break a run
                break
            self._emit(record)
