# Implementation Plan: Intra-Case Progress Reporting

---
**Date:** 2026-08-08
**Author:** AI Assistant
**Status:** Draft
**Codebase state:** commit `1c3eb7d`, branch `feat/unified-progress-bars`
**Related Documents:**
- [Research: Intra-Case Progress Reporting](research-intra-case-progress.md)

---

## Overview

EWB currently shows a single progress bar counting completed `CaseOperator`s. When
one case takes a long time — which is common — the user has no way to tell whether
it is opening data, running a pipeline, or grinding through a metric, and no sense
of how far along it is. In parallel mode, the recommended and documented execution
mode, there is no intra-case signal at all: worker processes start with an empty
bar registry, so every `set_phase` call and every dask callback write is a silent
no-op.

This plan adds two additional levels of granularity beneath the case bar: a
**step** level (`2 + N` coarse steps per case, where `N` is the number of
metric×variable-pair evaluations) and a **dask task** level (the graph draining
within the current step). It makes both levels work in serial *and* in parallel by
introducing a process-agnostic event sink: in the parent, events render directly to
nested tqdm bars; in a loky worker, they are published to a
`multiprocessing.Manager().Queue()` that a parent daemon thread drains onto one
fixed bar per worker slot.

The work is phased so the cheapest, lowest-risk improvement lands first. Phase 1 is
a handful of log-message edits that immediately help today's parallel users. Phase 2
restructures serial rendering using plumbing that already exists. Phase 3 adds the
cross-process transport, which is the only genuinely new machinery.

**Goal:** A user watching any EWB run — serial or parallel — can see which case is
running, which step within that case is running, how many steps remain, and that
the dask graph for the current step is actively draining.

**Motivation:** Long cases are currently indistinguishable from hangs. Users cannot
tell a slow S3 read from a stuck compute, and have no basis for estimating whether
to wait or kill the run.

## Current State Analysis

**Existing Implementation:**

- `src/extremeweatherbench/progress.py:31` — `_ProgressState`, a single-slot registry
  holding `active_bar`, `phase_updates_allowed`, and `current_phase`.
- `src/extremeweatherbench/progress.py:47` — `make_case_bar`, the only bar factory;
  honours `EWB_DISABLE_PROGRESS` at `:57`.
- `src/extremeweatherbench/progress.py:112` — `set_phase`, writes free text to the
  active bar's postfix; returns early at `:121` when no bar is registered or phase
  updates are disallowed.
- `src/extremeweatherbench/progress.py:127` — `DaskTaskPostfix`, a
  `dask.callbacks.Callback` writing `dask <done>/<total>` into the same postfix,
  throttled to 0.5 s (`:134`), returning early at `:158` when no bar is registered.
- `src/extremeweatherbench/evaluate.py:266-277` — serial dispatch; registers the bar
  with `allow_phase_updates=True` and wraps the loop in `DaskTaskPostfix`.
- `src/extremeweatherbench/utils.py:572-582` — `ParallelTqdm.dispatch_one_batch`;
  registers the bar with `allow_phase_updates=False`.
- `src/extremeweatherbench/evaluate.py:421-508` — the metric evaluation loop; builds
  `explicitly_claimed_*` sets, then per metric computes `metrics_to_evaluate` and
  `variable_pairs` inline.
- `src/extremeweatherbench/evaluate.py:638`, `:821`, `:840` — existing INFO logs for
  phase transitions.
- `src/extremeweatherbench/evaluate.py:254` — `logging_redirect_tqdm` already wraps
  the whole run.

**Current Behavior:**

Serial runs show one bar whose postfix carries either a phase string or a dask task
count, whichever wrote last. Parallel runs show one bar with no postfix at all.

**Current Limitations:**

- Intra-case information is squeezed into a single postfix string, so the phase and
  the dask count overwrite each other.
- Nothing intra-case reaches the terminal in parallel mode.
- `"Computing metric %s"` (`evaluate.py:638`) and `"Running target pipeline... "`
  (`evaluate.py:821`) carry no case id, making them uninterpretable when several
  cases log concurrently.
- The number of steps in a case is never computed, so no intra-case ETA is possible.

## Desired End State

**New Behavior:**

Serial runs render three stacked bars:

```
Evaluating cases:  40%|████      | 2/5 [01:12<01:48]
  case 12:         33%|███       | 3/9 [00:21<00:42] RootMeanSquaredError
    dask tasks:    71%|███████   | 12904/18332
```

Parallel runs render the case bar plus one fixed bar per worker slot:

```
Evaluating cases:  40%|████      | 2/5 [01:12<01:48]
  slot 0 | case 12: 33%|███     | 3/9 [00:21<00:42] RootMeanSquaredError
  slot 1 | case 14: 88%|████████| 8/9 [00:19<00:02] dask 4001/4412
  slot 2 | idle
```

Non-tty output (CI, notebooks) falls back to throttled INFO log lines instead of
nested bars.

**Success Looks Like:**

- In serial mode, the step bar advances exactly `2 + N` times per case (plus 2 more
  when `cache_dir` is set) and reaches its total before the case bar ticks.
- In parallel mode with `n_jobs=4`, four slot bars appear, each showing a case id and
  a step fraction, and slots are reused as cases complete.
- `progress=False`, `--no-progress`, and `EWB_DISABLE_PROGRESS` suppress all three
  levels and skip creating the `Manager` process entirely.
- With `sys.stderr` not a tty, no bars are emitted and phase transitions appear as
  INFO logs at most once every 5 seconds per case.

## What We're NOT Doing

- [ ] Progress for the `dask.distributed` backend at the dask-task level.
- [ ] Replacing tqdm with `rich`.
- [ ] Changing the parallelism model, `joblib` usage, or `ParallelTqdm`'s case-level
      counting semantics.
- [ ] Per-metric wall-time profiling or a performance report.
- [ ] Making the dask task count drive the ETA.
- [ ] Restructuring the metric evaluation loop's caching behaviour at
      `evaluate.py:510-520`.

**Rationale:** `dask.callbacks.Callback` provably never fires under
`dask.distributed` (`progress.py:130-132`), so task-level counting there would need a
wholly different mechanism (scheduler plugins) — out of proportion to the benefit
when the dashboard already exists and is already documented at
`docs/parallelism.md:54-57`. The step-level bar *will* still work under that backend,
since it flows over the queue rather than the dask callback. `rich` would be a new
dependency and a rewrite of working code. The dask task total restarts on every
compute call (`progress.py:141`), so it is a liveness signal, not a completion
signal, and must not drive the ETA.

## Implementation Approach

**Technical Strategy:**

Keep the existing module-level registry pattern in `progress.py` and generalise the
destination of an event from "the active tqdm bar" to "the active **sink**". A sink
is anything that accepts a `ProgressEvent`. Three sinks exist:

- `BarSink` — renders to nested tqdm bars in the parent process (serial).
- `QueueSink` — publishes to a `Manager().Queue()` from a loky worker (parallel).
- `LogSink` — throttled INFO logging (non-tty fallback, either mode).

Because the sink is registered per process, the four existing `set_phase` call sites
(`evaluate.py:639`, `:822`, `:841`, `utils.py:879`) keep working unchanged and
automatically start reporting from workers. `set_phase` doubles as the step tick:
it is already called exactly once per step.

**Key Architectural Decisions:**

1. **Decision:** Step count, not dask task count, supplies the intra-case percentage
   and ETA.
   - **Rationale:** `_start_state` resets `total_tasks = len(dsk)` per compute call
     (`progress.py:141`), so with `2 + N` computes per case the dask fraction sweeps
     0→100% repeatedly and graph sizes differ by orders of magnitude between steps.
   - **Trade-offs:** Steps are coarse and unequal in duration, so the step ETA is
     rough. It is, however, monotonic and bounded, which the dask count is not.
   - **Alternatives considered:** Summing `len(dsk)` across steps — rejected because
     the totals are unknown until each compute begins.

2. **Decision:** One fixed bar per worker slot, reassigned as cases start.
   - **Rationale:** A stable line count avoids terminal reflow. Loky reuses workers,
     so a case→slot map with a free list matches the execution model.
   - **Trade-offs:** A slot sits visibly "idle" between cases. Accepted; it is
     honest about worker utilisation.
   - **Alternatives considered:** One bar per in-flight case (constant reflow); a
     single aggregate line (loses the per-case detail that motivated the work).

3. **Decision:** Worker-side throttling is time-based at 0.5 s, reusing the existing
   constant.
   - **Rationale:** The feasibility probe produced 1032 events from ~5100 tasks with
     only a count-based "every 5th task" throttle. Real graphs are far larger.
     Separately, `posttask` callbacks run on the scheduler's critical path
     (`dask.local.get_async` lines 168-170), so unthrottled publishing directly slows
     compute.
   - **Trade-offs:** Up to 0.5 s of staleness. Imperceptible for this purpose.
   - **Alternatives considered:** Count-based throttling — rejected, as it does not
     adapt to graph size.

4. **Decision:** The `Manager` process is created only when parallel *and* progress
   are both active, and only when stderr is a tty.
   - **Rationale:** Avoids paying for an extra process and queue traffic that nobody
     renders.
   - **Trade-offs:** The disable paths must be checked before constructing the
     transport rather than at render time.
   - **Alternatives considered:** Always creating it and discarding events — wasteful.

**Patterns to Follow:**

- Single-slot module registry with a context manager — see `progress.py:92`
  (`registered_bar`).
- `disable` resolution that ORs the parameter with the env var — see `progress.py:57`.
- Callback throttling via `time.monotonic()` deltas — see `progress.py:148-152`.
- Keeping `progress.py` free of EWB imports to avoid cycles — see `progress.py:6-7`.

## Implementation Phases

### Phase 1: Case-identified log lines

**Objective:** Make existing phase logs interpretable under concurrency, delivering
parallel-mode visibility before any bar work.

**Tasks:**

- [x] **Write the failing test** for case-identified metric logs.
  - File: `tests/test_evaluate.py` (append)

  ```python
  def test_metric_log_includes_case_id(caplog, sample_case_operator):
      """Metric logs must name the case so parallel output is readable."""
      caplog.set_level(logging.INFO, logger="extremeweatherbench.evaluate")
      case_id = sample_case_operator.case_metadata.case_id_number
      evaluate.compute_case_operator(sample_case_operator)
      metric_logs = [
          r.getMessage() for r in caplog.records if "Computing metric" in r.getMessage()
      ]
      assert metric_logs
      assert all(f"case {case_id}" in message for message in metric_logs)
  ```

- [x] **Run it, watch it fail:**
  `pytest tests/test_evaluate.py::test_metric_log_includes_case_id -v`
  → expect FAIL (log reads `Computing metric RootMeanSquaredError... ` with no case id)

- [x] **Implement** the log message change.
  - File: `src/extremeweatherbench/evaluate.py:638`

  ```python
  logger.info(
      "Computing metric %s for case %s... ",
      metric.name,
      case_operator.case_metadata.case_id_number,
  )
  ```

  - File: `src/extremeweatherbench/evaluate.py:821`

  ```python
  logger.info(
      "Running target pipeline for case %s... ",
      case_operator.case_metadata.case_id_number,
  )
  ```

  - File: `src/extremeweatherbench/evaluate.py:840`

  ```python
  logger.info(
      "Running forecast pipeline for case %s... ",
      case_operator.case_metadata.case_id_number,
  )
  ```

- [x] **Run it, watch it pass:**
  `pytest tests/test_evaluate.py::test_metric_log_includes_case_id -v` → expect PASS

- [x] **Commit:** `git commit -m "feat: identify the case in pipeline and metric logs"`

**Dependencies:** None.

**Verification:**
- [x] `pytest tests/test_evaluate.py -v` → all pass
- [x] `ruff check src/ tests/` → no errors

### Phase 2: Serial nested step and dask bars

**Objective:** Give serial runs a step bar with a real denominator and a dask task
bar, replacing the overloaded single postfix.

**Tasks:**

- [ ] **Write the failing test** for the hoisted evaluation plan.
  - File: `tests/test_evaluate.py` (append)

  ```python
  def test_plan_metric_evaluations_counts_pairs(sample_case_operator):
      """The plan enumerates every metric x variable-pair evaluation."""
      plan = evaluate._plan_metric_evaluations(sample_case_operator)
      assert plan
      expected = sum(len(expanded) * len(pairs) for _, expanded, pairs in plan)
      assert evaluate._count_metric_evaluations(sample_case_operator) == expected
  ```

- [ ] **Run it, watch it fail:**
  `pytest tests/test_evaluate.py::test_plan_metric_evaluations_counts_pairs -v`
  → expect FAIL (`AttributeError: module has no attribute '_plan_metric_evaluations'`)

- [ ] **Implement** the hoisted planner. Insert before `compute_case_operator` at
  `src/extremeweatherbench/evaluate.py:347`. The body is lifted verbatim from the
  existing logic at `evaluate.py:421-482`; nothing in it touches the datasets.

  ```python
  def _plan_metric_evaluations(
      case_operator: "cases.CaseOperator",
  ) -> list[tuple["metrics.BaseMetric", list["metrics.BaseMetric"], list[tuple]]]:
      """Enumerate the metric evaluations a case operator will perform.

      Depends only on metric and input metadata, never on the data, so the
      result is available before any pipeline runs and can size a progress
      bar.

      Args:
          case_operator: The case operator to plan evaluations for.

      Returns:
          One (metric, expanded_metrics, variable_pairs) tuple per metric,
          in the same order the evaluation loop will consume them.
      """
      explicitly_claimed_forecast_vars = set()
      explicitly_claimed_target_vars = set()
      for metric in case_operator.metric_list:
          if (metric.forecast_variable is not None) and (
              metric.target_variable is not None
          ):
              explicitly_claimed_forecast_vars.update(
                  _maybe_expand_derived_variable_to_output_variables(
                      metric.forecast_variable
                  )
              )
              explicitly_claimed_target_vars.update(
                  _maybe_expand_derived_variable_to_output_variables(
                      metric.target_variable
                  )
              )

      plan = []
      for metric in case_operator.metric_list:
          metrics_to_evaluate = metric.maybe_expand_composite()
          if (
              metric.forecast_variable is not None
              and metric.target_variable is not None
          ):
              forecast_vars = _maybe_expand_derived_variable_to_output_variables(
                  metric.forecast_variable
              )
              target_vars = _maybe_expand_derived_variable_to_output_variables(
                  metric.target_variable
              )
              variable_pairs = list(zip(forecast_vars, target_vars))
          else:
              forecast_vars_expanded = []
              for var in case_operator.forecast.variables:
                  forecast_vars_expanded.extend(
                      _maybe_expand_derived_variable_to_output_variables(var)
                  )
              target_vars_expanded = []
              for var in case_operator.target.variables:
                  target_vars_expanded.extend(
                      _maybe_expand_derived_variable_to_output_variables(var)
                  )
              forecast_vars_available = [
                  v
                  for v in forecast_vars_expanded
                  if v not in explicitly_claimed_forecast_vars
              ]
              target_vars_available = [
                  v
                  for v in target_vars_expanded
                  if v not in explicitly_claimed_target_vars
              ]
              variable_pairs = list(
                  zip(forecast_vars_available, target_vars_available)
              )
          plan.append((metric, metrics_to_evaluate, variable_pairs))
      return plan


  def _count_metric_evaluations(case_operator: "cases.CaseOperator") -> int:
      """Count the metric evaluations a case operator will perform.

      Args:
          case_operator: The case operator to count evaluations for.

      Returns:
          The number of individual metric evaluations.
      """
      return sum(
          len(expanded) * len(pairs)
          for _, expanded, pairs in _plan_metric_evaluations(case_operator)
      )
  ```

  Then replace the inline computation at `evaluate.py:415-482` with a call, leaving
  the body of the loop (including the caching block at `:510-520`) untouched:

  ```python
      for metric, metrics_to_evaluate, variable_pairs in _plan_metric_evaluations(
          case_operator
      ):
  ```

- [ ] **Run it, watch it pass:**
  `pytest tests/test_evaluate.py::test_plan_metric_evaluations_counts_pairs -v`
  → expect PASS

- [ ] **Run the regression suite** to confirm the hoist changed no behaviour:
  `pytest tests/test_evaluate.py -v` → expect all PASS

- [ ] **Commit:** `git commit -m "refactor: hoist metric evaluation planning out of the loop"`

- [ ] **Write the failing test** for the step bar.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run it, watch it fail:**
  `pytest tests/test_progress.py::test_case_step_bar_totals_and_advances -v`
  → expect FAIL (`AttributeError: module has no attribute 'make_case_step_bar'`)

- [ ] **Implement** the two new bar factories.
  - File: `src/extremeweatherbench/progress.py` (append after `make_case_bar` at `:70`)

  ```python
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
  ```

  Add `Union` to the `typing` import at `progress.py:17`.

- [ ] **Run it, watch it pass:**
  `pytest tests/test_progress.py::test_case_step_bar_totals_and_advances -v`
  → expect PASS

- [ ] **Write the failing test** for the dask bar resetting per compute.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run it, watch it fail:**
  `pytest tests/test_progress.py::test_dask_task_bar_resets_between_computes -v`
  → expect FAIL (`AttributeError: module has no attribute 'DaskTaskBar'`)

- [ ] **Implement** `DaskTaskBar`, replacing `DaskTaskPostfix` at `progress.py:127`.

  ```python
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
  ```

- [ ] **Run it, watch it pass:**
  `pytest tests/test_progress.py::test_dask_task_bar_resets_between_computes -v`
  → expect PASS

- [ ] **Implement** the serial wiring. Replace `evaluate.py:266-277` with:

  ```python
              run_results = []
              case_bar = progress_module.make_case_bar(
                  len(case_operators), disable=not progress
              )
              task_bar = progress_module.make_dask_task_bar(disable=not progress)
              with progress_module.registered_bar(case_bar, allow_phase_updates=True):
                  with progress_module.DaskTaskBar(task_bar):
                      for case_operator in case_operators:
                          step_bar = progress_module.make_case_step_bar(
                              case_operator.case_metadata.case_id_number,
                              _count_case_steps(case_operator, cache_dir),
                              disable=not progress,
                          )
                          progress_module.register_step_bar(step_bar)
                          try:
                              run_results.append(
                                  compute_case_operator(
                                      case_operator, cache_dir, **kwargs
                                  )
                              )
                          finally:
                              progress_module.register_step_bar(None)
                              step_bar.close()
                          case_bar.update(1)
              task_bar.close()
              case_bar.close()
  ```

  Add the step-count helper next to `_count_metric_evaluations`:

  ```python
  def _count_case_steps(
      case_operator: "cases.CaseOperator",
      cache_dir: Optional[pathlib.Path] = None,
  ) -> int:
      """Count the coarse steps a case will report progress for.

      Two pipeline runs, two cache writes when caching is on, and one step
      per metric evaluation. Mirrors the set_phase call sites.

      Args:
          case_operator: The case operator about to be computed.
          cache_dir: The cache directory, if caching is enabled.

      Returns:
          The total number of steps.
      """
      cache_steps = 2 if cache_dir is not None else 0
      return 2 + cache_steps + _count_metric_evaluations(case_operator)
  ```

  Extend `_ProgressState` at `progress.py:38-41` with `step_bar` and make `set_phase`
  tick it, so the four existing call sites become step ticks:

  ```python
  def register_step_bar(bar: Optional[tqdm]) -> None:
      """Set (or clear) the nested step bar that set_phase advances.

      Args:
          bar: The step bar to advance, or None to stop advancing.
      """
      _state.step_bar = bar
  ```

  and inside `set_phase`, after `_state.current_phase = text`:

  ```python
      if _state.step_bar is not None:
          # Clamp so an unexpected extra phase can't overshoot the total.
          if _state.step_bar.n < (_state.step_bar.total or 0):
              _state.step_bar.update(1)
          _state.step_bar.set_postfix_str(text)
  ```

  Note that `set_phase`'s early return at `progress.py:121` must move *below* the
  step-bar tick, or gate only the postfix write, so the step bar still advances in
  parallel mode where `phase_updates_allowed` is False.

- [ ] **Write the failing test** for the step tick.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run them:**
  `pytest tests/test_progress.py -v` → expect all PASS

- [ ] **Commit:** `git commit -m "feat: add nested step and dask task bars for serial runs"`

**Dependencies:** Requires Phase 1 only for ordering convenience, not technically.

**Verification:**
- [ ] `pytest tests/test_progress.py tests/test_evaluate.py -v` → all pass
- [ ] `mypy src/extremeweatherbench/` → no new errors
- [ ] Manual: run a two-case serial evaluation and observe three stacked bars

### Phase 3: Cross-process progress for parallel runs

**Objective:** Carry step and dask events out of loky workers and render them on one
fixed bar per worker slot.

**Tasks:**

- [ ] **Write the failing test** for the event type and queue sink.
  - File: `tests/test_progress.py` (append)

  ```python
  def test_queue_sink_publishes_events():
      """A queue sink forwards events verbatim to its queue."""
      q: "queue.Queue" = queue.Queue()
      sink = progress.QueueSink(q)
      sink(progress.ProgressEvent(case_id=3, phase="target pipeline", step=1))
      event = q.get_nowait()
      assert event.case_id == 3
      assert event.phase == "target pipeline"
      assert event.step == 1
  ```

- [ ] **Run it, watch it fail:**
  `pytest tests/test_progress.py::test_queue_sink_publishes_events -v`
  → expect FAIL (`AttributeError: module has no attribute 'ProgressEvent'`)

- [ ] **Implement** the event and sink.
  - File: `src/extremeweatherbench/progress.py` (append)

  ```python
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
  ```

  Add `import dataclasses` and `import queue` to the imports at `progress.py:13-20`.

- [ ] **Run it, watch it pass:**
  `pytest tests/test_progress.py::test_queue_sink_publishes_events -v` → expect PASS

- [ ] **Write the failing test** for slot assignment and reuse.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run them, watch them fail:**
  `pytest tests/test_progress.py -k slot_renderer -v`
  → expect FAIL (`AttributeError: module has no attribute 'WorkerSlotRenderer'`)

- [ ] **Implement** the renderer.
  - File: `src/extremeweatherbench/progress.py` (append)

  ```python
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
          self._queue = None
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
  ```

  Add `import threading` to the imports at `progress.py:13-20`.

- [ ] **Run them, watch them pass:**
  `pytest tests/test_progress.py -k slot_renderer -v` → expect PASS

- [ ] **Implement** worker-side publishing. Add the sink to `_ProgressState` and a
  registration function, then make `set_phase` publish:

  ```python
  def register_sink(
      sink: Optional[Callable[[ProgressEvent], None]],
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
  ```

  and inside `set_phase`, after the step-bar tick:

  ```python
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
  ```

  Add `Callable` to the `typing` import at `progress.py:17`.

- [ ] **Implement** the worker entry point. Add to
  `src/extremeweatherbench/evaluate.py`, next to `compute_case_operator` at `:347`:

  ```python
  def _compute_case_operator_with_progress(
      case_operator: "cases.CaseOperator",
      cache_dir: Optional[pathlib.Path] = None,
      event_queue=None,
      **kwargs,
  ) -> pd.DataFrame:
      """Compute a case operator, publishing progress to event_queue.

      Runs inside a worker process. Falls back to plain computation when no
      queue is supplied so serial callers are unaffected.

      Args:
          case_operator: The case operator to compute.
          cache_dir: The directory to cache mid-flight outputs.
          event_queue: Cross-process queue for progress events, or None.
          **kwargs: Additional arguments for compute_case_operator.

      Returns:
          A pd.DataFrame of results from the case operator.
      """
      if event_queue is None:
          return compute_case_operator(case_operator, cache_dir, **kwargs)

      case_id = case_operator.case_metadata.case_id_number
      sink = progress_module.QueueSink(event_queue)
      progress_module.register_sink(
          sink, case_id=case_id, total_steps=_count_case_steps(case_operator, cache_dir)
      )
      try:
          with progress_module.DaskTaskSink(sink, case_id):
              return compute_case_operator(case_operator, cache_dir, **kwargs)
      finally:
          sink(progress_module.ProgressEvent(case_id=case_id, finished=True))
          progress_module.register_sink(None)
  ```

- [ ] **Implement** `DaskTaskSink`, the worker-side counterpart of `DaskTaskBar`.
  - File: `src/extremeweatherbench/progress.py` (append)

  ```python
  class DaskTaskSink(dask.callbacks.Callback):
      """Publish local-scheduler dask task counts to a progress sink.

      The worker-process counterpart of DaskTaskBar. Throttled because
      posttask callbacks run on the dask scheduler's own loop, so every
      publish is on the critical path of the compute.
      """

      def __init__(
          self,
          sink: Callable[[ProgressEvent], None],
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
  ```

- [ ] **Implement** the parallel wiring. Replace `evaluate.py:325-339` with:

  ```python
      parallel_tqdm_kwargs: dict[str, Any] = {"total_tasks": len(case_operators)}
      if not progress:
          parallel_tqdm_kwargs["disable_progressbar"] = True

      manager = None
      event_queue = None
      renderer = None
      if progress and progress_module.supports_nested_bars():
          n_slots = parallel_config.get("n_jobs") or 1
          manager = multiprocessing.Manager()
          event_queue = manager.Queue()
          renderer = progress_module.WorkerSlotRenderer(n_slots=n_slots)
          renderer.start(event_queue)

      try:
          with joblib.parallel_config(**parallel_config):
              run_results = utils.ParallelTqdm(**parallel_tqdm_kwargs)(
                  joblib.delayed(_compute_case_operator_with_progress)(
                      case_operator,
                      cache_dir=cache_dir,
                      event_queue=event_queue,
                      **kwargs,
                  )
                  for case_operator in case_operators
              )
          return run_results
      finally:
          if renderer is not None:
              renderer.close()
          if manager is not None:
              manager.shutdown()
          if dask_client is not None:
              logger.info("Closing dask client")
              dask_client.close()
  ```

  Add `import multiprocessing` to the imports at `evaluate.py:3-22`.

- [ ] **Write the failing test** for the non-tty guard.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run them, watch them fail:**
  `pytest tests/test_progress.py -k supports_nested_bars -v`
  → expect FAIL (`AttributeError: module has no attribute 'supports_nested_bars'`)

- [ ] **Implement** the guard and the log fallback.
  - File: `src/extremeweatherbench/progress.py` (append)

  ```python
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
  ```

  Add `import sys` to the imports at `progress.py:13-20`.

- [ ] **Run them, watch them pass:**
  `pytest tests/test_progress.py -k supports_nested_bars -v` → expect PASS

- [ ] **Write the integration test** for end-to-end worker reporting.
  - File: `tests/test_progress.py` (append)

  ```python
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
  ```

- [ ] **Run it:**
  `pytest tests/test_progress.py::test_progress_events_cross_loky_process_boundary -v`
  → expect PASS (the mechanism was verified by probe before planning)

- [ ] **Commit:** `git commit -m "feat: report intra-case progress from parallel workers"`

**Dependencies:** Requires Phase 2 (`make_case_step_bar`, `_count_case_steps`,
`register_step_bar`, and the `set_phase` restructure).

**Verification:**
- [ ] `pytest tests/ -v` → all pass
- [ ] `mypy src/extremeweatherbench/` → no new errors
- [ ] `EWB_DISABLE_PROGRESS=1 python -c "import multiprocessing, ..."` confirms no
      `Manager` process is spawned (check via `multiprocessing.active_children()`)

### Phase 4: Documentation

**Objective:** Describe the new output so users know what they are looking at.

**Tasks:**

- [ ] **Rewrite** the Progress Reporting section at `docs/parallelism.md:43-57` to
  cover three levels, the slot model, the non-tty fallback, and the unchanged
  `dask.distributed` guidance. Include the two sample bar layouts from the *Desired
  End State* section above.

- [ ] **Update** the CLI help text at `src/extremeweatherbench/evaluate_cli.py:60`
  from `"Disable the case-level progress bar"` to `"Disable all progress bars"`.

- [ ] **Update** the `progress` argument docstrings at `evaluate.py:122`, `:168`,
  `:248`, and `:295` from `"Whether to display the case-level progress bar."` to
  `"Whether to display progress bars."`

- [ ] **Commit:** `git commit -m "docs: describe the nested progress bars"`

**Dependencies:** Requires Phase 3.

**Verification:**
- [ ] `ewb --help` shows the updated flag description
- [ ] `ruff check src/ tests/` → no errors

## Success Criteria

### Automated Verification

- [ ] `make dev-test` passes
- [ ] `pytest tests/test_progress.py -v` passes, including the new step-bar,
      slot-renderer, non-tty, and cross-process tests
- [ ] `pytest tests/test_evaluate.py -v` passes, confirming the hoist in Phase 2
      changed no evaluation behaviour
- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] `_count_case_steps` equals the observed number of `set_phase` calls for a
      sample case operator, asserted in `tests/test_evaluate.py`

### Manual Verification

- [ ] A serial run with `n_jobs=1` shows three stacked bars; the step bar reaches its
      total before the case bar ticks
- [ ] A parallel run with `n_jobs=4` shows exactly four slot bars, and slots visibly
      recycle between cases
- [ ] The dask task bar visibly moves during a long metric compute, confirming a slow
      case is distinguishable from a hang
- [ ] `ewb --default --no-progress` emits no bars
- [ ] Piping output to a file (`ewb --default > out.log 2>&1`) produces readable log
      lines with no escape-sequence noise
- [ ] Wall-clock time for a representative parallel run is within noise of the same
      run on `main` — the throttled publishing must not measurably slow compute

## Testing Strategy

Unit tests are written test-first within each phase. This section covers the
additional integration and manual coverage.

**Unit Test Coverage (summary, written in-phase):**
- [ ] Bar construction: totals, descriptions, positions, disable paths
- [ ] `set_phase` step ticking, including the overshoot clamp
- [ ] `DaskTaskBar` resizing and draining across consecutive computes
- [ ] Slot assignment, release, reuse, and exhaustion
- [ ] `supports_nested_bars` for tty and env-var combinations
- [ ] `_plan_metric_evaluations` / `_count_metric_evaluations` agreement
- [ ] Mock external dependencies: none needed; `dask.array` and a real
      `Manager().Queue()` are used directly, as in the existing
      `tests/test_progress.py:90`

**Integration Tests:**
- [ ] Events published from loky workers arrive in the parent, one `finished` event
      per case (Phase 3)
- [ ] A full `_run_evaluation` in serial mode with progress enabled produces results
      identical to progress disabled

**Manual Testing:**
- [ ] Scenario 1: `ewb --default --n-jobs 1` in a terminal; watch the three bars
- [ ] Scenario 2: `ewb --default --n-jobs 4` in a terminal; watch slot recycling
- [ ] Scenario 3: the same command redirected to a file; confirm log fallback
- [ ] Scenario 4: interrupt a run with Ctrl-C; confirm bars close and the terminal
      cursor is restored

**Test Data Requirements:**
- Existing fixtures in `tests/`; the cross-process test needs no EWB data, only
  `joblib` and a `Manager` queue.

## Migration Strategy

**Migration Steps:**
1. Phases land in order; each is independently shippable.
2. `DaskTaskPostfix` is replaced by `DaskTaskBar` in Phase 2. It is a public name in
   a lazily-exported module (`__init__.py:28`), so keep a thin alias for one release:

   ```python
   # Deprecated alias; DaskTaskBar renders to a bar instead of a postfix.
   DaskTaskPostfix = DaskTaskBar
   ```

**Rollback Plan:** Each phase is a separate commit; revert individually. Phase 3 can
also be neutralised at runtime by `EWB_DISABLE_PROGRESS=1`, which short-circuits
`supports_nested_bars` and skips the `Manager` entirely.

**Backward Compatibility:** `progress=True/False`, `--no-progress`, and
`EWB_DISABLE_PROGRESS` keep their meanings, now covering all levels. The case-level
bar's format and semantics are unchanged. `compute_case_operator` keeps its exact
signature; the queue is threaded through the new
`_compute_case_operator_with_progress` wrapper instead.

## Risk Assessment

**Potential Risks:**

1. **Risk:** The hoist in Phase 2 subtly changes metric evaluation ordering or the
   `explicitly_claimed_*` semantics.
   - **Likelihood:** Low — the extracted code is verbatim and provably
     data-independent (`evaluate.py:444-482` reads only metric and input metadata).
   - **Impact:** High — would silently change benchmark results.
   - **Mitigation:** The hoist is its own commit with the full `tests/test_evaluate.py`
     suite run before and after; the loop body and caching block are untouched.

2. **Risk:** Queue publishing slows compute because `posttask` runs on the dask
   scheduler's critical path.
   - **Likelihood:** Medium without throttling, Low with it.
   - **Impact:** Medium.
   - **Mitigation:** 0.5 s time-based throttle in `DaskTaskSink`; `put_nowait` never
     blocks; the wall-clock comparison is an explicit manual success criterion.

3. **Risk:** tqdm bars written by the renderer's daemon thread interleave badly with
   `ParallelTqdm.print_progress` (`utils.py:586`) writing the case bar.
   - **Likelihood:** Medium.
   - **Impact:** Low — cosmetic tearing only.
   - **Mitigation:** tqdm serialises writes through `tqdm.get_lock()`; the fixed slot
     count avoids reflow. Verified by manual Scenario 2.

4. **Risk:** The `Manager` process leaks if the run raises.
   - **Likelihood:** Low.
   - **Impact:** Medium — an orphaned process.
   - **Mitigation:** `manager.shutdown()` in the `finally` block alongside the
     existing `dask_client.close()` at `evaluate.py:340-344`.

## Edge Cases and Error Handling

**Edge Cases:**

1. **Case:** A case runs more phases than `_count_case_steps` predicted.
   - **Expected Behavior:** The step bar clamps at its total rather than exceeding it.
   - **Implementation:** The `n < total` guard in `set_phase`.

2. **Case:** More cases in flight than worker slots (joblib pre-dispatch).
   - **Expected Behavior:** The extra case gets no slot bar; the case-level bar still
     counts it.
   - **Implementation:** The empty-`_free_slots` early return in
     `WorkerSlotRenderer.handle`.

3. **Case:** `n_jobs=-1`, so the slot count is not a literal integer.
   - **Expected Behavior:** Resolve to the actual CPU count before sizing slots.
   - **Implementation:** `n_slots = parallel_config.get("n_jobs") or 1`; when negative,
     resolve via `joblib.effective_n_jobs(n_jobs)` before constructing the renderer.

4. **Case:** `backend="dask"` with a distributed client.
   - **Expected Behavior:** Step-level events still flow over the queue; dask task
     counts are simply absent, and `dask_total` stays 0 so the postfix omits them.
   - **Implementation:** The `if event.dask_total:` branch in `handle`.

5. **Case:** A case raises partway through.
   - **Expected Behavior:** The `finished` event still fires and the slot is released.
   - **Implementation:** The `finally` block in `_compute_case_operator_with_progress`.

**Error Scenarios:**

1. **Error:** The parent dies and the queue proxy breaks while a worker publishes.
   - **Handling:** `QueueSink.__call__` catches broadly and logs at DEBUG. Progress
     must never break an evaluation.

2. **Error:** The renderer thread's `queue.get` raises `EOFError` or `BrokenPipeError`
   after manager shutdown.
   - **Handling:** `_drain` returns cleanly on those exceptions.

## Performance Considerations

- **Expected Load:** One event per step (a handful per case) plus up to two events per
  second per worker from `DaskTaskSink`.
- **Performance Targets:** Parallel wall-clock within noise of `main`.
- **Optimization Strategy:** Time-based throttling at the 0.5 s convention already in
  `progress.py:134`; `put_nowait` so a full queue drops events rather than blocking a
  worker; no transport at all when progress is disabled or stderr is not a tty.

## Documentation Updates

- [ ] Rewrite `docs/parallelism.md:43-57` (Phase 4)
- [ ] Update the `--no-progress` help text at `evaluate_cli.py:60` (Phase 4)
- [ ] Update the four `progress` argument docstrings in `evaluate.py` (Phase 4)
- [ ] Docstrings on all new public functions and classes, max 88 characters per line

## Timeline Estimate

- Phase 1: very small — three log-line edits and one test
- Phase 2: the largest single chunk, dominated by the hoist and its regression run
- Phase 3: moderate — the mechanism is proven, so the work is wiring and tests
- Phase 4: small

**Note:** Estimates are rough and may change during implementation.

## Open Questions

None. The three decisions that were open after research — concurrent-case bar layout,
non-tty fallback, and whether the queue is opt-in — were resolved as fixed per-slot
bars, automatic log fallback, and on-by-default respectively, and are reflected in
*Key Architectural Decisions* above.

---

## References

**Research Documents:**
- [Research: Intra-Case Progress Reporting](research-intra-case-progress.md)

**Files Analyzed:**
- `src/extremeweatherbench/progress.py`
- `src/extremeweatherbench/evaluate.py`
- `src/extremeweatherbench/utils.py`
- `src/extremeweatherbench/evaluate_cli.py`
- `src/extremeweatherbench/__init__.py`
- `tests/test_progress.py`
- `docs/parallelism.md`
- `Makefile`
- `pyproject.toml`

**External Documentation:**
- [dask custom callbacks](https://docs.dask.org/en/stable/custom-collections.html)
- [joblib parallel_config](https://joblib.readthedocs.io/en/latest/generated/joblib.parallel_config.html)
- [dask dashboard](https://docs.dask.org/en/stable/dashboard.html)
- `dask.local.get_async` lines 168-170 — posttask callbacks run on the scheduler loop

---

## Review History

### Version 1.0 — 2026-08-08
- Initial plan created from `research-intra-case-progress.md`, incorporating the loky
  queue feasibility probe and three user design decisions.
