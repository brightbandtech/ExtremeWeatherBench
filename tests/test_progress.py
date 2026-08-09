"""Tests for the progress module."""

import multiprocessing
import queue
import sys

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
