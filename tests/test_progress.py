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


def test_dask_task_postfix_counts_tasks():
    """DaskTaskPostfix counts tasks over a small dask.array compute."""
    bar = progress.make_case_bar(1)
    array = da.ones((4, 4), chunks=(2, 2)) + 1
    try:
        with progress.registered_bar(bar, allow_phase_updates=True):
            callback = progress.DaskTaskPostfix(throttle_seconds=0.0)
            with callback:
                array.compute()
            assert callback.completed_tasks == callback.total_tasks
            assert callback.completed_tasks > 0
    finally:
        bar.close()
