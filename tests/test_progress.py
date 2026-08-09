"""Tests for the progress module."""

import dask.array as da

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
