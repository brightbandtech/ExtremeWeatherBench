# Research: Simplify `_group_operators_sharing_forecast`

**Date:** 2026-08-14
**Scope:** internal codebase
**Related Documents:** none
**Codebase state:** `origin/perf/shared-eval-speedups` @ `2f23ca4`
(2026-08-14). Line numbers below are from that ref, not `develop`.

## Question / Scope

Can the return type of `_group_operators_sharing_forecast` —
`list[list[tuple[int, CaseOperator]]]` — be simplified, including by
changing upstream producers or downstream consumers? This pass maps
what the type is doing, why the index exists, and which simplifications
are real versus cosmetic. No implementation.

## Codebase Findings

### What the function does

`_group_operators_sharing_forecast` lives only on
`perf/shared-eval-speedups`, in
`src/extremeweatherbench/evaluate.py:466-483`. It is a first-seen
groupby over a forecast-reuse key:

```text
(case_id_number, forecast.name, str(forecast.source))
```

Each group is a list of `(original_index, CaseOperator)` pairs. An
`OrderedDict` keeps first-seen key order. The function is used only by
`_run_parallel_evaluation` (`evaluate.py:439`).

The motivating case is in the docstring and in
`defaults.get_brightband_evaluation_objects`: PPH and LSR
`EvaluationObject`s share `cira_fcnv2_severe_convection_forecast`.
Heat-wave and freeze defaults do the same (ERA5 + GHCN against one
forecast). `build_case_operators`
(`src/extremeweatherbench/cases.py:79-91`) emits one `CaseOperator` per
`(case, EvaluationObject)` pair, so those pairs become two operators
that should share one forecast pipeline.

### Why the type is three layers deep

Each layer has a distinct job:

| Layer | Meaning |
|---|---|
| Outer `list` | One joblib job per reuse group |
| Inner `list` | Operators that must run in the same worker |
| `tuple[int, CaseOperator]` | Original input index plus the operator |

The index is not decorative. Groups are not 1:1 with input positions.
`build_case_operators` uses `itertools.product(cases, evaluation_objects)`,
so same-case pairs are usually adjacent, but a custom operator list can
interleave groups (case1-PPH, case2-PPH, case1-LSR). Joblib also
returns one result per group, not per operator.

`_compute_operator_group_with_progress` (`evaluate.py:558-591`) takes
that same `list[tuple[int, CaseOperator]]`, runs each operator against
a process-local `_pipeline_cache_var`, and returns
`list[tuple[int, list[xr.DataArray]]]`. `_ungroup_operator_results`
(`evaluate.py:486-509`) scatters those pairs back into
`list[list[xr.DataArray]]` aligned with the original
`case_operators` list.

### Upstream: grouping is a parallel-scheduling concern

`CaseOperator` (`cases.py:48-61`) is a plain dataclass: case metadata,
metrics, target, forecast. It has no reuse key and no group id.

`build_case_operators` only filters by `event_type` and builds the
product. It does not know about joblib, process-local caches, or
result order.

Serial `_run_evaluation` (`evaluate.py:321-352`) does **not** group. It
walks operators in input order and sets one process-wide
`_pipeline_cache_var`. Reuse in serial is automatic. Grouping exists
only so parallel loky/dask workers colocate operators that can share
that cache. A `Manager`-shared cache of `xr.Dataset`s would pickle
large arrays across processes; `cache_dir` is documented as serial-only.

Observation: moving grouping into `build_case_operators` or
`CaseOperator` would mix a joblib scheduling detail into the domain
model, and serial would ignore it.

### Downstream: the index is discarded almost immediately

The only production consumers of `_run_evaluation` are
`ExtremeWeatherBench.run` and `run_evaluation`
(`evaluate.py:148-159` and `206-217`). Both immediately flatten:

```text
[result for case_results in run_results for result in case_results]
```

then pass a flat `list[xr.DataArray]` to `_convert_results`. Annotated
results already carry `case_id_number`, `metric`, `target_source`, and
`forecast_source` (`evaluate.py:915-943`), so output identity does
not depend on list position.

So the per-operator aligned `list[list[xr.DataArray]]` contract exists
for:

1. Serial tests that assert `result[i]` matches operator `i`.
2. Parallel tests that mock `ParallelTqdm` and expect that shape.
3. Keeping serial and parallel return types identical.

It is not required by the public DataFrame/Dataset API.

### Group key vs pipeline cache key

The group key (`evaluate.py:477-481`) is coarser than
`_pipeline_cache_key` (`evaluate.py:519-531`):

| Field | Group key | Cache key |
|---|---|---|
| `case_id_number` | yes | yes |
| input type name | no | yes |
| `name` | yes | yes |
| `source` | yes | yes |
| variables | no | yes |

Operators with the same case/name/source but different variable sets
still share a worker (useful) and miss the cache (correct). PPH vs LSR
is the intended hit: same forecast, different targets.

### Bookkeeping that exists only because of grouping

`_ungroup_operator_results` duck-types the ParallelTqdm return value
(`evaluate.py:496-504`). If the first item is not a list of
`(int, result)` pairs, it returns the mock unchanged. That shim exists
because many `TestRunParallel` tests still return a flat list of
DataFrames, the pre-grouping shape.

`dispatch_id=group[0][0]` (`evaluate.py:450`) is passed into
`_compute_operator_group_with_progress` and then unused. The group
function re-passes `dispatch_id=orig_i` to each inner operator
(`evaluate.py:575-586`). The group-level `dispatch_id` is dead.

`parallel_tqdm_kwargs["total_tasks"]` is reset from
`len(case_operators)` to `len(groups)` (`evaluate.py:400`, `440`). The
case-level bar ticks once per group, not once per operator. Inner step
bars still run per operator.

There are no direct unit tests for
`_group_operators_sharing_forecast` or `_ungroup_operator_results`.
`test_run_parallel_evaluation_assigns_unique_dispatch_id_per_operator`
avoids grouping by giving the second operator a different forecast
name/source.

## Synthesis

The nested type is honest: parallel reuse needs (1) groups and (2) a
way back to input order. The convolution is real, but most of it is
bookkeeping for a per-operator list that the public API flattens away.

**Observation (not a design):** three simplification levels are
available. They are not mutually exclusive.

1. **Name the inner type, keep the structure.** A `NamedTuple` or
   `@dataclass` such as `IndexedOperator(index, operator)`, plus a
   type alias `OperatorGroup = list[IndexedOperator]`, makes the
   current contract readable. No behavior change. Lowest risk.

2. **Return indices only:** `list[list[int]]`. The grouper stays a
   pure partition. The call site looks up `case_operators[i]`. The
   function type gets simpler; the `(index, operator)` pair still
   appears at the joblib boundary unless option 3 is also taken.

3. **Drop index-preserving ungroup.** Have the grouper return
   `list[list[CaseOperator]]`. Have each worker return a flat
   `list[xr.DataArray]`. Concatenate group results in first-seen
   group order. Delete `_ungroup_operator_results` and its duck-type
   shim. This is the only option that removes a layer rather than
   renaming it. Cost: `_run_evaluation`'s "per-case-operator lists
   in input order" contract goes away (or becomes serial-only),
   several tests change, and DataFrame row order becomes group-major
   instead of input-major. Output coords still identify rows.

Upstream changes to `build_case_operators` / `CaseOperator` do not
remove the need to colocate parallel jobs. Serial already reuses
without grouping. A shared cross-process cache would avoid grouping
but is a larger, likely slower design.

Light recommendation: if the pain is the annotation, do (1) or (2).
If the pain is the bookkeeping (`_ungroup`, duck-typing, unused
`dispatch_id`), do (3) — the index exists to honor a contract the
public API does not use.

Open questions for planning:

- Is DataFrame/Dataset row order part of the public contract?
- Should `_run_evaluation` keep returning per-operator lists for
  serial/parallel consistency, or flatten at that boundary?
- Should grouping stay private to `_run_parallel_evaluation`?

## References / Sources

- `src/extremeweatherbench/evaluate.py:148-217` — flatten after
  `_run_evaluation`
- `src/extremeweatherbench/evaluate.py:290-354` — serial path, no
  grouping, process-wide cache
- `src/extremeweatherbench/evaluate.py:439-455` — parallel dispatch
- `src/extremeweatherbench/evaluate.py:466-509` — group / ungroup
- `src/extremeweatherbench/evaluate.py:519-555` — pipeline cache key
- `src/extremeweatherbench/evaluate.py:558-591` — group worker
- `src/extremeweatherbench/evaluate.py:915-943` — result metadata
- `src/extremeweatherbench/cases.py:48-91` — `CaseOperator` /
  `build_case_operators`
- `src/extremeweatherbench/defaults.py:437-448` — PPH + LSR share a
  forecast
- `tests/test_evaluate.py` — `TestRunParallel`; no direct grouper
  tests
- Commit `b5774e9` — `perf: reuse evaluation pipeline work across
  operators`
