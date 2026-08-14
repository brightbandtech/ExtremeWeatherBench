# Implementation Plan: Name operator-group types

---
**Date:** 2026-08-14
**Author:** AI Assistant
**Status:** Complete (awaiting manual verification)
**Related Documents:**
- [Research: Simplify `_group_operators_sharing_forecast`](research-group-operators-sharing-forecast.md)

---

## Overview

Replace the anonymous
`list[list[tuple[int, CaseOperator]]]` grouping contract with named
types so the index, operator, and group layers are readable at every
call site. Behavior stays the same: first-seen forecast groups, original
indices, and `_ungroup_operator_results` still restore input order.

**Goal:** Grouping and ungrouping use `IndexedOperator` and
`IndexedOperatorResult` instead of raw tuples. Groups stay
`list[IndexedOperator]` (no type alias).

**Motivation:** The nested tuple type is hard to audit. Option 3
(drop ungroup) has no performance benefit — it only skips O(n) pointer
writes after the expensive pipelines finish.

## Current State Analysis

**Existing Implementation:**
- `src/extremeweatherbench/evaluate.py:466-483` —
  `_group_operators_sharing_forecast` returns
  `list[list[tuple[int, CaseOperator]]]`
- `src/extremeweatherbench/evaluate.py:450` — `dispatch_id=group[0][0]`
- `src/extremeweatherbench/evaluate.py:486-509` —
  `_ungroup_operator_results` duck-types `(int, result)` pairs
- `src/extremeweatherbench/evaluate.py:558-591` —
  `_compute_operator_group_with_progress` unpacks `orig_i, op`

**Current Behavior:** Parallel evaluation groups operators that share
`(case_id, forecast.name, forecast.source)`, runs each group in one
worker, and scatters results back to input order.

**Current Limitations:**
- The type does not name what the `int` or inner list means
- Call sites use `group[0][0]` and `orig_i, op`
- No direct unit tests for the grouper

## Desired End State

**New Behavior:** Same grouping and ungrouping, with named fields
(`.index`, `.operator`, `.results`) at every site.

**Success Looks Like:**
- Grouper return type is `list[list[IndexedOperator]]`
- Worker return type is `list[IndexedOperatorResult]`
- Tests assert grouping, index preservation, and ungroup restore
- Existing `TestRunParallel` tests still pass

## What We're NOT Doing

- [ ] Option 2 (`list[list[int]]` grouper)
- [ ] Option 3 (drop `_ungroup_operator_results`)
- [ ] Changing `build_case_operators` or `CaseOperator`
- [ ] Removing the unused group-level `dispatch_id` parameter
- [ ] Changing progress-bar `total_tasks` semantics
- [ ] Committing unless the user asks

**Rationale:** User chose option 1 for auditability after confirming
option 3 is not faster.

## Implementation Approach

**Technical Strategy:** Add frozen dataclass types in `evaluate.py` and
switch the grouper, worker, and ungroup helper to construct and read
those types. Frozen dataclasses match `ProgressEvent` and keep named
fields at every call site. Joblib pickles them without extra work.

**Key Architectural Decisions:**
1. **Decision:** `@dataclasses.dataclass(frozen=True)` over `NamedTuple`
   - **Rationale:** Matches existing evaluate/progress records; named
     fields without tuple semantics
   - **Trade-offs:** Not unpackable as `(index, operator)`
   - **Alternatives considered:** `NamedTuple` (first implementation;
     user preferred dataclasses)
2. **Decision:** Also name the worker result pair
   - **Rationale:** Ungroup is the other half of the same contract
   - **Trade-offs:** One extra type
   - **Alternatives considered:** Leave results as raw tuples
3. **Decision:** Ungroup checks `isinstance(..., IndexedOperatorResult)`
   - **Rationale:** More auditable than "list of 2-tuples starting
     with an int"; mock ParallelTqdm flat lists still pass through

**Patterns to Follow:**
- `TypeAlias` for union/list aliases — `src/extremeweatherbench/inputs.py:161`
- Private evaluate helpers tested directly — `tests/test_evaluate.py`

## Implementation Phases

### Phase 1: Named types and grouping tests

**Objective:** Introduce named types, wire them through group / ungroup
/ worker, and lock the contract with unit tests.

**Tasks:**
- [x] **Write the failing tests** in
  `tests/test_evaluate.py` (new class after `TestRunSerial`, before
  `TestRunParallel` at line 905)

  ```python
  class TestGroupOperatorsSharingForecast:
      def test_shared_forecast_is_one_group(self, sample_case_operator):
          op2 = dataclasses.replace(
              sample_case_operator,
              target=mock.Mock(spec=inputs.TargetBase),
          )
          sample_case_operator.forecast.name = "HRES"
          sample_case_operator.forecast.source = "hres://x"
          op2.forecast.name = "HRES"
          op2.forecast.source = "hres://x"
          groups = evaluate._group_operators_sharing_forecast(
              [sample_case_operator, op2]
          )
          assert len(groups) == 1
          assert [item.index for item in groups[0]] == [0, 1]
          assert all(
              isinstance(item, evaluate.IndexedOperator)
              for item in groups[0]
          )

      def test_different_forecasts_are_separate_groups(
          self, sample_case_operator
      ):
          other_forecast = mock.Mock(spec=inputs.ForecastBase)
          other_forecast.name = "other"
          other_forecast.source = "other://x"
          sample_case_operator.forecast.name = "HRES"
          sample_case_operator.forecast.source = "hres://x"
          op2 = dataclasses.replace(
              sample_case_operator, forecast=other_forecast
          )
          groups = evaluate._group_operators_sharing_forecast(
              [sample_case_operator, op2]
          )
          assert len(groups) == 2
          assert [item.index for item in groups[0]] == [0]
          assert [item.index for item in groups[1]] == [1]

      def test_interleaved_operators_keep_original_indices(
          self, sample_case_operator
      ):
          shared = sample_case_operator.forecast
          shared.name = "HRES"
          shared.source = "hres://x"
          other = mock.Mock(spec=inputs.ForecastBase)
          other.name = "other"
          other.source = "other://x"
          op_b = dataclasses.replace(
              sample_case_operator, forecast=other
          )
          op_c = dataclasses.replace(sample_case_operator)
          groups = evaluate._group_operators_sharing_forecast(
              [sample_case_operator, op_b, op_c]
          )
          assert [item.index for item in groups[0]] == [0, 2]
          assert [item.index for item in groups[1]] == [1]

      def test_ungroup_restores_input_order(self):
          da0 = xr.DataArray([0.0])
          da1 = xr.DataArray([1.0])
          nested = [
              [
                  evaluate.IndexedOperatorResult(2, [da1]),
                  evaluate.IndexedOperatorResult(0, [da0]),
              ]
          ]
          restored = evaluate._ungroup_operator_results(nested, 3)
          assert restored[0][0] is da0
          assert restored[2][0] is da1
          assert restored[1] == []

      def test_ungroup_passes_through_mock_flat_list(self):
          mock_results = [pd.DataFrame({"v": [1.0]})]
          assert (
              evaluate._ungroup_operator_results(mock_results, 1)
              is mock_results
          )
  ```

- [x] **Run it, watch it fail:**
  `source .venv/bin/activate && pytest tests/test_evaluate.py::TestGroupOperatorsSharingForecast -v`
  → expect FAIL (`IndexedOperator` / `IndexedOperatorResult` missing)

- [x] **Implement the types and wire call sites** in
  `src/extremeweatherbench/evaluate.py`

  Add imports: `NamedTuple`, `TypeAlias`.

  Insert before `_group_operators_sharing_forecast`:

  ```python
  class IndexedOperator(NamedTuple):
      """An operator tagged with its original input index."""

      index: int
      operator: "cases.CaseOperator"


  class IndexedOperatorResult(NamedTuple):
      """Metric results tagged with the originating operator index."""

      index: int
      results: list[xr.DataArray]


  OperatorGroup: TypeAlias = list[IndexedOperator]
  ```

  Grouper appends `IndexedOperator(i, op)` and returns
  `list[OperatorGroup]`. Worker takes `OperatorGroup` and returns
  `list[IndexedOperatorResult]`, using `.index` / `.operator`. Dispatch
  uses `group[0].index`. Ungroup checks
  `isinstance(first[0], IndexedOperatorResult)` and writes
  `run_results[item.index] = item.results`.

- [x] **Run it, watch it pass:**
  `source .venv/bin/activate && pytest tests/test_evaluate.py::TestGroupOperatorsSharingForecast tests/test_evaluate.py::TestRunParallel -v`

**Dependencies:** None

**Verification:**
- [x] Grouping tests pass
- [x] `TestRunParallel` still passes

## Success Criteria

### Automated Verification

- [x] `source .venv/bin/activate && pytest tests/test_evaluate.py::TestGroupOperatorsSharingForecast tests/test_evaluate.py::TestRunParallel -q`
- [x] `ruff check` UP040 on the new `type OperatorGroup` alias (pre-existing file lints left alone)

### Manual Verification

- [ ] Call sites in `_run_parallel_evaluation`, the grouper, the
      worker, and ungroup read `.index` / `.operator` / `.results`
      rather than `[0]` / `[1]`
- [ ] No behavior change: same groups, same restored order

## Testing Strategy

**Unit Test Coverage (summary, written in-phase):**
- Shared forecast → one group
- Different forecasts → two groups
- Interleaved operators keep original indices
- Ungroup restores order from `IndexedOperatorResult`
- Ungroup still passes through mock flat lists

**Integration Tests:**
- Existing `TestRunParallel` covers dispatch through ParallelTqdm

**Manual Testing:**
- Read the four call sites for named-field use

**Test Data Requirements:**
- Existing `sample_case_operator` fixture; explicit
  `forecast.name` / `forecast.source`

## Risk Assessment

**Potential Risks:**
1. **Risk:** `dataclasses.replace` on CaseOperator with mock forecast
   - **Likelihood:** Low
   - **Impact:** Low
   - **Mitigation:** Tests set name/source on the shared forecast
     object; replace only swaps forecast/target

2. **Risk:** Ungroup isinstance check rejects a later type change
   - **Likelihood:** Low
   - **Impact:** Medium
   - **Mitigation:** Tests cover both real results and mock passthrough

## Edge Cases and Error Handling

**Edge Cases:**
1. **Case:** Empty `nested_results`
   - **Expected Behavior:** Return unchanged (existing early return)
   - **Implementation:** `_ungroup_operator_results` first lines
2. **Case:** Singleton group
   - **Expected Behavior:** One `IndexedOperator` with its input index
   - **Implementation:** grouper `setdefault` + append

## Documentation Updates

- [ ] Type and function signatures/docstrings in `evaluate.py`
- [ ] Research doc already records why option 1 was chosen

## Open Questions

---

## References

**Research Documents:**
- [Research: Simplify `_group_operators_sharing_forecast`](research-group-operators-sharing-forecast.md)

**Files Analyzed:**
- `src/extremeweatherbench/evaluate.py`
- `tests/test_evaluate.py`
- `src/extremeweatherbench/cases.py`
- `src/extremeweatherbench/inputs.py`

---

## Review History

### Version 1.0 — 2026-08-14
- Initial plan for option 1 (named types)

### Version 1.1 — 2026-08-14
- Switch `NamedTuple` to frozen dataclasses per user preference

### Version 1.2 — 2026-08-14
- Drop the `OperatorGroup` type alias; annotate `list[IndexedOperator]`

### Version 1.3 — 2026-08-14
- Worker returns group-order result lists; parent zips `groups` to
  scatter. Deleted `IndexedOperatorResult` and the ungroup duck-type.
