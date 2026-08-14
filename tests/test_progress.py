"""Tests for the progress module."""

import io
import logging
import logging.handlers
import multiprocessing
import queue
import sys
import time
import warnings

import dask
import dask.array as da
import joblib

from extremeweatherbench import progress


def test_bar_format_has_percentage_and_eta():
    """BAR_FORMAT must always render a percentage and an elapsed<remaining ETA."""
    assert "{percentage" in progress.BAR_FORMAT
    assert "{elapsed}" in progress.BAR_FORMAT
    assert "{remaining}" in progress.BAR_FORMAT
    assert "{postfix}" in progress.BAR_FORMAT


def test_make_case_bar_defaults():
    """make_case_bar builds a bar with the shared desc/unit and a total."""
    bar = progress.make_case_bar(3)
    try:
        assert bar.total == 3
        assert bar.desc == "Evaluating cases"
        assert bar.unit == "case"
    finally:
        bar.close()


def test_make_case_bar_disable_env_var(monkeypatch):
    """EWB_DISABLE_PROGRESS forces the bar into a disabled state."""
    monkeypatch.setenv("EWB_DISABLE_PROGRESS", "1")
    bar = progress.make_case_bar(3)
    try:
        assert bar.disable is True
    finally:
        bar.close()


def test_make_case_bar_disable_param_without_env_var(monkeypatch):
    """The disable parameter also disables the bar without the env var."""
    monkeypatch.delenv("EWB_DISABLE_PROGRESS", raising=False)
    bar = progress.make_case_bar(3, disable=True)
    try:
        assert bar.disable is True
    finally:
        bar.close()


def test_set_phase_is_noop_without_registered_bar():
    """set_phase must not raise when no bar has been registered."""
    progress.clear_bar()
    progress.set_phase("case 1 | RMSE")


def test_set_phase_updates_registered_bar():
    """set_phase writes to the active bar when phase updates are allowed."""
    # disable=False so tqdm actually initializes bar.postfix; pytest
    # captures the bar's stderr output so it doesn't clutter test runs.
    bar = progress.make_case_bar(1)
    try:
        with progress.registered_bar(bar, allow_phase_updates=True):
            progress.set_phase("case 1 | RMSE")
            assert bar.postfix == "case 1 | RMSE"
    finally:
        bar.close()


def test_set_phase_ignored_when_phase_updates_disallowed():
    """set_phase is a no-op when the active bar disallows phase updates."""
    bar = progress.make_case_bar(1)
    try:
        with progress.registered_bar(bar, allow_phase_updates=False):
            progress.set_phase("case 1 | RMSE")
            assert bar.postfix is None
    finally:
        bar.close()


def test_registered_bar_clears_on_exit():
    """The registered_bar context manager clears the registry on exit."""
    bar = progress.make_case_bar(1)
    try:
        with progress.registered_bar(bar, allow_phase_updates=True):
            progress.set_phase("case 1 | RMSE")
        # Outside the context, updates must be silent no-ops again.
        progress.set_phase("case 2 | MAE")
        assert bar.postfix == "case 1 | RMSE"
    finally:
        bar.close()


def test_case_step_bar_totals_and_advances():
    """The step bar carries the case id and advances once per phase."""
    bar = progress.make_case_step_bar(case_id=12, total_steps=4)
    try:
        assert bar.total == 4
        assert "case 12" in bar.desc
        bar.update(1)
        assert bar.n == 1
    finally:
        bar.close()


def test_dask_task_postfix_counts_tasks():
    """DaskTaskPostfix (now an alias for DaskTaskBar) counts dask tasks."""
    bar = progress.make_dask_task_bar()
    array = da.ones((4, 4), chunks=(2, 2)) + 1
    try:
        callback = progress.DaskTaskPostfix(bar, throttle_seconds=0.0)
        with callback:
            array.compute()
        assert callback.completed_tasks == callback.total_tasks
        assert callback.completed_tasks > 0
    finally:
        bar.close()


def test_set_phase_advances_registered_step_bar():
    """Each set_phase call advances the step bar by exactly one."""
    step_bar = progress.make_case_step_bar(case_id=7, total_steps=3)
    try:
        progress.register_step_bar(step_bar)
        progress.set_phase("case 7 | target pipeline")
        progress.set_phase("case 7 | forecast pipeline")
        assert step_bar.n == 2
    finally:
        progress.register_step_bar(None)
        step_bar.close()


def test_step_bar_does_not_overshoot_total():
    """Extra phases clamp at the total rather than exceeding it."""
    step_bar = progress.make_case_step_bar(case_id=7, total_steps=1)
    try:
        progress.register_step_bar(step_bar)
        progress.set_phase("one")
        progress.set_phase("two")
        assert step_bar.n == 1
    finally:
        progress.register_step_bar(None)
        step_bar.close()


def test_queue_sink_publishes_events():
    """A queue sink forwards events verbatim to its queue."""
    q: queue.Queue = queue.Queue()
    sink = progress.QueueSink(q)
    sink(progress.ProgressEvent(case_id=3, phase="target pipeline", step=1))
    event = q.get_nowait()
    assert event.case_id == 3
    assert event.phase == "target pipeline"
    assert event.step == 1


def test_supports_nested_bars_false_without_tty(monkeypatch):
    """Nested bars are suppressed when stderr is not a terminal."""
    monkeypatch.delenv("EWB_DISABLE_PROGRESS", raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    assert progress.supports_nested_bars() is False


def test_supports_nested_bars_false_when_disabled(monkeypatch):
    """EWB_DISABLE_PROGRESS suppresses nested bars even on a tty."""
    monkeypatch.setenv("EWB_DISABLE_PROGRESS", "1")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    assert progress.supports_nested_bars() is False


def test_supports_nested_bars_forced_without_tty(monkeypatch):
    """EWB_FORCE_PROGRESS overrides the isatty gate."""
    monkeypatch.delenv("EWB_DISABLE_PROGRESS", raising=False)
    monkeypatch.setenv("EWB_FORCE_PROGRESS", "1")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    assert progress.supports_nested_bars() is True


def test_supports_nested_bars_disable_beats_force(monkeypatch):
    """EWB_DISABLE_PROGRESS wins when both env vars are set."""
    monkeypatch.setenv("EWB_DISABLE_PROGRESS", "1")
    monkeypatch.setenv("EWB_FORCE_PROGRESS", "1")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    assert progress.supports_nested_bars() is False


def test_set_phase_publishes_to_registered_sink():
    """set_phase publishes a ProgressEvent to a registered sink."""
    published = []
    try:
        progress.register_sink(published.append, case_id=9, total_steps=2)
        progress.set_phase("case 9 | target pipeline")
        assert len(published) == 1
        assert published[0].case_id == 9
        assert published[0].phase == "case 9 | target pipeline"
        assert published[0].step == 1
        assert published[0].total_steps == 2
    finally:
        progress.register_sink(None)


def test_slot_renderer_assigns_and_reuses_slots():
    """Cases claim a free slot and release it when they finish."""
    renderer = progress.WorkerSlotRenderer(n_slots=2, disable=True)
    try:
        renderer.handle(progress.ProgressEvent(case_id=1, total_steps=3))
        renderer.handle(progress.ProgressEvent(case_id=2, total_steps=3))
        assert set(renderer.slot_by_case) == {1, 2}
        renderer.handle(progress.ProgressEvent(case_id=1, finished=True))
        assert 1 not in renderer.slot_by_case
        renderer.handle(progress.ProgressEvent(case_id=3, total_steps=3))
        assert 3 in renderer.slot_by_case
    finally:
        renderer.close()


def test_slot_renderer_drops_events_when_slots_exhausted():
    """More in-flight cases than slots must not raise."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=True)
    try:
        renderer.handle(progress.ProgressEvent(case_id=1, total_steps=3))
        renderer.handle(progress.ProgressEvent(case_id=2, total_steps=3))
        assert set(renderer.slot_by_case) == {1}
    finally:
        renderer.close()


def test_slot_renderer_disambiguates_same_case_id_via_slot_key():
    """Two operators sharing a case_id must claim two distinct slots.

    Regression test: build_case_operators can emit multiple CaseOperators
    for the same case_id (one per evaluation object), and they must not
    collide onto the same slot bar.
    """
    renderer = progress.WorkerSlotRenderer(n_slots=2, disable=True)
    try:
        renderer.handle(progress.ProgressEvent(case_id=305, slot_key=0, total_steps=3))
        renderer.handle(progress.ProgressEvent(case_id=305, slot_key=1, total_steps=3))
        assert set(renderer.slot_by_case) == {0, 1}
        assert renderer.slot_by_case[0] != renderer.slot_by_case[1]
    finally:
        renderer.close()


def test_slot_renderer_uses_label_for_description():
    """The slot bar description comes from the event's precomputed label."""
    # disable=True short-circuits tqdm's __init__ before .desc is even
    # set, so this test needs a real (non-disabled) bar to inspect it.
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(
            progress.ProgressEvent(
                case_id=305, slot_key=0, total_steps=3, label="case 305 | pph_target"
            )
        )
        slot = renderer.slot_by_case[0]
        assert renderer._bars[slot].desc == "  case 305 | pph_target"
    finally:
        renderer.close()


def test_slot_renderer_shows_dask_progress_as_a_fraction():
    """Slot bars show the in-flight graph's done/total, as serial does."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(
            progress.ProgressEvent(
                case_id=1,
                slot_key=0,
                total_steps=3,
                phase="RootMeanSquaredError",
                dask_done=341,
                dask_total=10699,
                dask_tasks_done=12904,
            )
        )
        slot = renderer.slot_by_case[0]
        assert "dask 341/10699" in renderer._bars[slot].postfix
    finally:
        renderer.close()


def test_slot_renderer_drops_case_prefix_duplicated_in_description():
    """The case id shows once, in the description, not again in the postfix."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(
            progress.ProgressEvent(
                case_id=220,
                slot_key=0,
                label="case 220 | IBTrACS | HRES",
                total_steps=5,
                phase="case 220 | forecast pipeline",
            )
        )
        slot = renderer.slot_by_case[0]
        assert renderer._bars[slot].postfix == "forecast pipeline"
        assert "case 220" in renderer._bars[slot].desc
    finally:
        renderer.close()


def test_dask_task_sink_publishes_completed_fraction_on_finish():
    """A graph's final publish lands on its total despite the throttle.

    The throttle nearly always drops a graph's last tasks, which would
    otherwise leave the slot showing a partial fraction until the next
    graph resets it.
    """
    published = []
    sink = progress.DaskTaskSink(published.append, case_id=1)
    sink._start_state(dsk={"a": 1, "b": 2, "c": 3}, state={})
    for _ in range(3):
        sink._posttask("k", None, {}, {}, 0)
    sink._finish(dsk={}, state={}, failed=False)
    assert published[-1].dask_done == published[-1].dask_total == 3


def test_slot_renderer_leaves_finished_bar_until_reclaimed():
    """A finished case's bar keeps its final state until a new case claims it."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(
            progress.ProgressEvent(
                case_id=1, slot_key=0, total_steps=2, phase="done", step=2
            )
        )
        slot_before = renderer.slot_by_case[0]
        desc_before = renderer._bars[slot_before].desc
        renderer.handle(progress.ProgressEvent(case_id=1, slot_key=0, finished=True))
        # No new case has claimed the slot yet: description is untouched.
        assert renderer._bars[slot_before].desc == desc_before
        assert 0 not in renderer.slot_by_case

        renderer.handle(progress.ProgressEvent(case_id=2, slot_key=1, total_steps=5))
        new_slot = renderer.slot_by_case[1]
        assert new_slot == slot_before
        assert renderer._bars[new_slot].desc != desc_before
    finally:
        renderer.close()


def test_slot_renderer_finished_case_paints_final_state():
    """A case's final slot update is actually painted, not just counted.

    Regression test: mininterval throttling can silently skip the
    repaint on the update just before a case finishes, leaving
    last_print_n (what was painted) behind n (the internal count)
    until something forces a repaint.
    """
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(progress.ProgressEvent(case_id=1, slot_key=0, total_steps=2))
        bar = renderer._bars[0]
        bar.mininterval = 1e9  # force the next update() to skip painting
        renderer.handle(
            progress.ProgressEvent(
                case_id=1, slot_key=0, total_steps=2, phase="done", step=2
            )
        )
        assert bar.n == 2
        assert bar.last_print_n != bar.n

        renderer.handle(progress.ProgressEvent(case_id=1, slot_key=0, finished=True))
        assert bar.last_print_n == bar.n
    finally:
        renderer.close()


def test_slot_renderer_initial_slots_are_blank():
    """An unclaimed slot renders as a blank line, not a placeholder."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        assert renderer._bars[0].desc == ""
    finally:
        renderer.close()


def test_dask_task_sink_publishes_monotonic_cumulative_count():
    """The cumulative dask count never decreases across two graphs."""
    published = []
    sink = progress.DaskTaskSink(published.append, case_id=1, throttle_seconds=0.0)
    with sink:
        (da.ones((4, 4), chunks=(2, 2)) + 1).compute()
        after_first_graph = published[-1].dask_tasks_done
        (da.ones((8, 8), chunks=(2, 2)) + 1).compute()
    counts = [e.dask_tasks_done for e in published]
    assert counts == sorted(counts)
    assert published[-1].dask_tasks_done > after_first_graph


def _publish_from_worker(case_id, event_queue):
    """Worker-side helper: emit two phases plus a finish event."""
    sink = progress.QueueSink(event_queue)
    progress.register_sink(sink, case_id=case_id, total_steps=2)
    try:
        progress.set_phase(f"case {case_id} | target pipeline")
        progress.set_phase(f"case {case_id} | forecast pipeline")
    finally:
        sink(progress.ProgressEvent(case_id=case_id, finished=True))
        progress.register_sink(None)
    return case_id


def test_progress_events_cross_loky_process_boundary():
    """Events published in loky workers arrive in the parent process."""
    manager = multiprocessing.Manager()
    event_queue = manager.Queue()
    try:
        with joblib.parallel_config(backend="loky", n_jobs=2):
            joblib.Parallel()(
                joblib.delayed(_publish_from_worker)(i, event_queue) for i in range(4)
            )
        received = []
        while not event_queue.empty():
            received.append(event_queue.get_nowait())
        reporting_cases = {e.case_id for e in received}
        assert reporting_cases == {0, 1, 2, 3}
        assert {e.case_id for e in received if e.finished} == {0, 1, 2, 3}
        assert max(e.step for e in received if e.case_id == 0) == 2
    finally:
        manager.shutdown()


def test_dask_task_bar_resets_between_computes():
    """Each compute resizes the dask bar and drains it to completion."""
    task_bar = progress.make_dask_task_bar()
    callback = progress.DaskTaskBar(task_bar, throttle_seconds=0.0)
    try:
        with callback:
            (da.ones((4, 4), chunks=(2, 2)) + 1).compute()
            first_total = task_bar.total
            (da.ones((8, 8), chunks=(2, 2)) + 1).compute()
        assert first_total > 0
        assert task_bar.total > first_total
        assert task_bar.n == task_bar.total
    finally:
        task_bar.close()


def test_dask_task_bar_paints_final_state_on_finish():
    """The bar's painted state reaches total when the graph finishes.

    Regression test: mininterval throttling could silently skip the
    final repaint, so n reached total while last_print_n (what was
    actually painted) lagged behind - the reported stall-then-reset.
    A test that only checked n would have passed before this fix.
    """
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        with progress.DaskTaskBar(task_bar):
            (da.ones((4, 4), chunks=(2, 2)) + 1).compute()
        assert task_bar.n == task_bar.total
        assert task_bar.last_print_n == task_bar.n
    finally:
        task_bar.close()


def test_dask_task_bar_writes_completed_frame_to_terminal():
    """The completed count reaches the output stream, not just the bar.

    Asserting on last_print_n alone would be circular, since the fix
    sets it; this checks the bytes tqdm actually wrote. The graph
    finishes well inside mininterval, so the only frame that can carry
    the full count is the forced repaint at finish.
    """
    buffer = io.StringIO()
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        task_bar.fp = buffer
        task_bar.sp = task_bar.status_printer(buffer)
        with progress.DaskTaskBar(task_bar):
            (da.ones((4, 4), chunks=(2, 2)) + 1).compute()
        total = task_bar.total
        assert f"{total}/{total}" in buffer.getvalue()
    finally:
        task_bar.close()


def test_dask_task_bar_counts_every_task():
    """No task is dropped before reaching the bar.

    The bar used to skip _sync_bar entirely for tasks arriving inside
    the throttle window, so the count itself lagged rather than just
    the repaint. Two back-to-back tasks land well inside any window.
    """
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        callback = progress.DaskTaskBar(task_bar)
        callback._start_state(dsk={"a": 1, "b": 2}, state={})
        callback._posttask("a", None, {}, {}, 0)
        callback._posttask("b", None, {}, {}, 0)
        assert task_bar.n == 2
    finally:
        task_bar.close()


def test_fast_bars_are_time_gated_only():
    """Inner bars gate on mininterval alone, not tqdm's miniters ratchet.

    With miniters left at 0 and smoothing=0, tqdm auto-tunes miniters as
    max(miniters, dn), which never falls back, so a fast burst makes the
    bar progressively less responsive for the rest of the run.
    """
    for bar in (
        progress.make_dask_task_bar(disable=False),
        progress.make_case_step_bar(case_id=1, total_steps=3, disable=False),
    ):
        try:
            assert bar.miniters == 1
            assert bar.dynamic_miniters is False
            assert bar.mininterval == progress.FAST_BAR_MININTERVAL
        finally:
            bar.close()


def test_dask_task_sink_still_throttles_publishes():
    """The worker sink keeps its throttle; each publish crosses processes.

    A Manager-queue put is about as costly as a repaint and runs on the
    dask scheduler's loop, so unthrottling this would put IPC on the
    critical path of every task.
    """
    published = []
    sink = progress.DaskTaskSink(published.append, case_id=1)
    sink._start_state(dsk={}, state={})
    for _ in range(50):
        sink._posttask("k", None, {}, {}, 0)
    assert sink.cumulative_completed_tasks == 50
    assert len(published) < 50


def test_dask_task_bar_repaints_several_times_during_a_slow_graph():
    """A multi-second graph paints repeatedly, not once at the end.

    This is the user-visible symptom: with a 0.5s app-level throttle
    stacked on a 0.5s mininterval, a short graph could reach the
    terminal only once.
    """
    buffer = io.StringIO()
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        task_bar.fp = buffer
        task_bar.sp = task_bar.status_printer(buffer)
        tasks = [dask.delayed(time.sleep)(0.02) for _ in range(20)]
        with progress.DaskTaskBar(task_bar):
            dask.compute(*tasks, scheduler="single-threaded")
        # 20 x 20ms spans ~8 windows of FAST_BAR_MININTERVAL; assert
        # well under that so the test isn't timing-fragile.
        assert buffer.getvalue().count("\r") >= 3
    finally:
        task_bar.close()


def test_dask_task_bar_finish_snaps_total_down_when_tasks_fall_short():
    """A success shortfall against len(dsk) still lands the bar full.

    This never happens in practice (verified across several real graph
    shapes), so the shortfall is simulated directly against a callback
    that was never attached to a real compute.
    """
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        callback = progress.DaskTaskBar(task_bar)
        task_bar.reset(total=10)
        callback.completed_tasks = 7
        callback._finish(dsk={}, state={}, failed=False)
        assert task_bar.total == 7
        assert task_bar.n == 7
        assert task_bar.last_print_n == 7
    finally:
        task_bar.close()


def test_dask_task_bar_finish_does_not_snap_total_on_failure():
    """A failed compute keeps its real total; a partial bar is honest."""
    task_bar = progress.make_dask_task_bar(disable=False)
    try:
        callback = progress.DaskTaskBar(task_bar)
        task_bar.reset(total=10)
        callback.completed_tasks = 7
        callback._finish(dsk={}, state={}, failed=True)
        assert task_bar.total == 10
        assert task_bar.n == 7
    finally:
        task_bar.close()


def test_captured_warnings_routes_warning_and_restores_state():
    """warnings.warn() becomes a py.warnings log record, then state resets."""
    before = logging._warnings_showwarning
    captured = []
    handler = logging.Handler()
    handler.emit = captured.append
    py_warnings_logger = logging.getLogger("py.warnings")
    py_warnings_logger.addHandler(handler)
    py_warnings_logger.setLevel(logging.WARNING)
    try:
        with progress.captured_warnings():
            assert logging._warnings_showwarning is not None
            warnings.warn("boom", RuntimeWarning)
    finally:
        py_warnings_logger.removeHandler(handler)
    assert any("boom" in record.getMessage() for record in captured)
    assert logging._warnings_showwarning == before


def test_captured_warnings_is_idempotent_when_already_enabled():
    """A caller's pre-existing capture state is left alone on exit."""
    logging.captureWarnings(True)
    try:
        marker = logging._warnings_showwarning
        with progress.captured_warnings():
            assert logging._warnings_showwarning is marker
        assert logging._warnings_showwarning is marker
    finally:
        logging.captureWarnings(False)


def test_forwarding_logs_to_routes_records_and_restores_handlers():
    """The root logger's handlers are swapped for a QueueHandler, then restored."""
    log_queue: queue.Queue = queue.Queue()
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    with progress.forwarding_logs_to(log_queue):
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.handlers.QueueHandler)
        logging.getLogger("extremeweatherbench.test_progress").warning(
            "hello %s", "world"
        )
    assert root.handlers == original_handlers
    record = log_queue.get_nowait()
    assert record.name == "extremeweatherbench.test_progress"
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "hello world"


def test_forwarding_logs_to_restores_handlers_when_block_raises():
    """A raising block must not leave the handler swap in place."""
    log_queue: queue.Queue = queue.Queue()
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    try:
        with progress.forwarding_logs_to(log_queue):
            raise ValueError("boom")
    except ValueError:
        pass
    assert root.handlers == original_handlers


def test_forwarding_logs_to_also_captures_warnings():
    """warnings.warn() inside forwarding_logs_to also reaches the queue."""
    log_queue: queue.Queue = queue.Queue()
    with progress.forwarding_logs_to(log_queue):
        warnings.warn("worker warning", RuntimeWarning)
    records = []
    while not log_queue.empty():
        records.append(log_queue.get_nowait())
    assert any("worker warning" in r.getMessage() for r in records)


def test_log_record_survives_manager_queue_round_trip():
    """A record keeps its level, logger name, and message through a Manager queue."""
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    try:
        with progress.forwarding_logs_to(log_queue):
            logging.getLogger("extremeweatherbench.test_progress").warning(
                "value=%s", 42
            )
        record = log_queue.get(timeout=2)
        assert record.name == "extremeweatherbench.test_progress"
        assert record.levelno == logging.WARNING
        assert record.getMessage() == "value=42"
    finally:
        manager.shutdown()


def test_log_queue_listener_emits_via_handle():
    """The listener drains a queue and re-emits records through logging."""
    log_queue: queue.Queue = queue.Queue()
    record = logging.LogRecord(
        name="extremeweatherbench.worker_test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="worker warning",
        args=None,
        exc_info=None,
    )
    log_queue.put(record)
    captured = []
    handler = logging.Handler()
    handler.emit = captured.append
    target_logger = logging.getLogger("extremeweatherbench.worker_test")
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.WARNING)
    listener = progress.LogQueueListener()
    try:
        listener.start(log_queue)
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
    finally:
        listener.close()
        target_logger.removeHandler(handler)
    assert any("worker warning" in r.getMessage() for r in captured)


def test_log_queue_listener_close_drains_leftover_records():
    """close() flushes records still queued after the drain thread stops."""
    log_queue: queue.Queue = queue.Queue()
    captured = []
    handler = logging.Handler()
    handler.emit = captured.append
    target_logger = logging.getLogger("extremeweatherbench.worker_test2")
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.WARNING)
    listener = progress.LogQueueListener()
    try:
        listener.start(log_queue)
        listener._stop.set()
        listener._thread.join(timeout=2)
        log_queue.put(
            logging.LogRecord(
                name="extremeweatherbench.worker_test2",
                level=logging.WARNING,
                pathname=__file__,
                lineno=1,
                msg="late warning",
                args=None,
                exc_info=None,
            )
        )
    finally:
        listener.close()
        target_logger.removeHandler(handler)
    assert any("late warning" in r.getMessage() for r in captured)


def test_log_queue_listener_drops_get_errors_without_raising():
    """A queue that raises on get() must not crash the drain thread."""

    class ExplodingQueue:
        def get(self, timeout=None):
            raise RuntimeError("boom")

        def get_nowait(self):
            raise queue.Empty()

    listener = progress.LogQueueListener()
    listener.start(ExplodingQueue())
    time.sleep(0.2)
    listener.close(drain_deadline=0.1)


def _warn_and_log_from_worker(msg, log_queue):
    """Worker-side helper: emit a logger.warning and a warnings.warn."""
    with progress.forwarding_logs_to(log_queue):
        logging.getLogger("extremeweatherbench.worker_test").warning(msg)
        warnings.warn(f"warn:{msg}", RuntimeWarning)
    return msg


def test_log_records_cross_loky_process_boundary():
    """logger.warning and warnings.warn in loky workers arrive in the parent."""
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    try:
        with joblib.parallel_config(backend="loky", n_jobs=2):
            joblib.Parallel()(
                joblib.delayed(_warn_and_log_from_worker)(f"case-{i}", log_queue)
                for i in range(4)
            )
        received = []
        while not log_queue.empty():
            received.append(log_queue.get_nowait())
        messages = [r.getMessage() for r in received]
        assert any("case-0" in m for m in messages)
        assert any("warn:case-0" in m for m in messages)
        assert len(received) == 8
    finally:
        manager.shutdown()
