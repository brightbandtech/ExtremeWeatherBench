"""Tests for the progress module."""

import logging
import logging.handlers
import multiprocessing
import queue
import sys
import time
import warnings

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
    q: "queue.Queue" = queue.Queue()
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
        renderer.handle(
            progress.ProgressEvent(case_id=305, slot_key=0, total_steps=3)
        )
        renderer.handle(
            progress.ProgressEvent(case_id=305, slot_key=1, total_steps=3)
        )
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


def test_slot_renderer_shows_cumulative_dask_count_without_denominator():
    """Slot bars show a bare cumulative dask count, not done/total."""
    renderer = progress.WorkerSlotRenderer(n_slots=1, disable=False)
    try:
        renderer.handle(
            progress.ProgressEvent(
                case_id=1,
                slot_key=0,
                total_steps=3,
                phase="RootMeanSquaredError",
                dask_tasks_done=12904,
            )
        )
        slot = renderer.slot_by_case[0]
        postfix = renderer._bars[slot].postfix
        assert "12904 tasks" in postfix
        assert "/" not in postfix
    finally:
        renderer.close()


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

        renderer.handle(
            progress.ProgressEvent(case_id=2, slot_key=1, total_steps=5)
        )
        new_slot = renderer.slot_by_case[1]
        assert new_slot == slot_before
        assert renderer._bars[new_slot].desc != desc_before
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
                joblib.delayed(_publish_from_worker)(i, event_queue)
                for i in range(4)
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
    log_queue: "queue.Queue" = queue.Queue()
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
    log_queue: "queue.Queue" = queue.Queue()
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
    log_queue: "queue.Queue" = queue.Queue()
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
    log_queue: "queue.Queue" = queue.Queue()
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
    log_queue: "queue.Queue" = queue.Queue()
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
