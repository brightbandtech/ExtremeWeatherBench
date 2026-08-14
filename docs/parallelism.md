# A Note on Parallelism

Running an evaluation on ExtremeWeatherBench utilizes `joblib` for parallelism. When invoking `ExtremeWeatherBench.run_evaluation()`, users can choose to passthrough their own `parallel_config` (see `joblib` information on `parallel_config` [here](https://joblib.readthedocs.io/en/latest/generated/joblib.parallel_config.html)). The default and recommended method is to use `joblib`'s default engine, `loky`.

## What is Parallelized in EWB?

Each process in EWB evaluates one `CaseOperator`. As a reminder, `CaseOperator` is an object that processes:

- One `IndividualCase` object
- One `ForecastBase` object
- One `TargetBase` object
- Any number of metrics provided in an `EvaluationObject`

Depending on the machine used to run EWB, input data sources, and types of events being evaluated, the number of jobs (`n_jobs`) will need to be tuned to prevent scenarios like out-of-memory (OOM) issues. 

## Recommended Approach

In experimentation with varying configurations, it was found `loky` was the most consistent and prevented out-of-scope issues, e.g. threading concurrency challenges, untraceable errors, better management of memory, and more efficient saturation of cores and threads. Users can opt to use any of the built-in choices for `joblib` using `parallel_config` but **we cannot guarantee consistent behavior**.

```python
...
# Load events yaml
case_yaml = cases.load_ewb_events_yaml_into_case_list()

# Get default EvaluationObjects
evaluation_objects = defaults.get_brightband_evaluation_objects()

# Instantiate EWB runner class
ewb = evaluate.ExtremeWeatherBench(
    case_metadata=case_yaml,
    evaluation_objects=evaluation_objects,
)

# Define parallel_config for runner with n_jobs set to the number of EvaluationObjects
# The larger the machine, the larger n_jobs can be (a bit of an oversimplification)
parallel_config = {"backend": "loky", "n_jobs": len(evaluation_objects)}

outputs = ewb.run_evaluation(parallel_config=parallel_config)
```

The _safest_ approach is to run EWB in serial, with `n_jobs` set to 1. `Dask` will still be invoked during each `CaseOperator` when the case executes and computes the graph, only one at a time. That said, for evaluations with more cases this approach would likely be too time-consuming. 

## Progress Reporting

EWB reports progress at three nested levels: how many `CaseOperator`s are
done, how far the current case has progressed through its pipelines and
metrics, and how many dask tasks the in-flight compute has finished.

In serial mode, this renders as three stacked bars:

```
Evaluating cases:  40%|████      | 2/5 [01:12<01:48]
  case 12:         33%|███       | 3/9 [00:21<00:42] RootMeanSquaredError
    dask tasks:    71%|███████   | 12904/18332
```

The case bar tracks overall run completion. The case-step bar's total is
the number of pipeline runs, cache writes, and metric evaluations that
case will perform, and its postfix names the current phase. The dask
task bar resizes for each compute call and only reflects local-scheduler
computes; it never advances under `dask.distributed`.

In parallel mode, cases run concurrently across worker processes, so
each worker slot gets its own fixed bar, reused as cases finish and new
ones start:

```
Evaluating cases:  40%|████      | 2/5 [01:12<01:48]
  case 12 | pph_target | HRES: 33%|███ | 3/9 [00:21<00:42] RootMeanSquaredError
  case 14 | ir_target | GraphCast: 88%|███| 8/9 [00:19<00:02] dask 4183/10699

```

Each slot's label names the case, target _and_ forecast, since one case
can have several `CaseOperator`s (one per `EvaluationObject`) running in
different slots at once, and the case id alone can't tell those apart.
The dask fraction is per graph, as in serial, so it restarts each time
the case begins a new compute. A slot with no case assigned yet renders
as a blank line, and a slot whose case just finished keeps showing that
case's final state until a new case claims it, so finishing mid-run
doesn't cause visible churn.

The number of slots is `joblib.effective_n_jobs(n_jobs)`, so it stays
bounded even when `n_jobs` is negative (e.g. `-1` for "all but one
core") or `None`. Slot bars are rendered from progress events published
by worker processes over a `multiprocessing.Manager` queue, so this adds
one extra process only while parallel progress is actually being shown.

When `sys.stderr` isn't a terminal (CI logs, captured notebook cells),
nested bars would just produce unreadable escape-code noise, so EWB
skips them entirely: no `Manager` process is created, and phase
transitions instead appear as throttled `INFO` log lines (at most once
every 5 seconds per case). Set `EWB_FORCE_PROGRESS` to render bars
anyway.

Bars repaint up to 20 times a second, which keeps terminal writes off
the critical path of large dask graphs.

Log lines and `warnings.warn()` output no longer tear through the bars.
`logging.captureWarnings(True)` is enabled for the run (and restored
afterwards), so warnings become log records; in the parent process they
already flow through `tqdm`'s log redirect, and in parallel mode each
worker forwards its log records to the parent over a second
`Manager` queue, where a listener thread re-emits them the same way.
This queue and its listener thread are only created under the same
conditions as the slot renderer above.

To turn off all progress reporting, pass `progress=False` to
`run_evaluation()`, use `--no-progress` on the CLI, or set the
`EWB_DISABLE_PROGRESS` env var; none of the three levels render and no
extra process is created. If you're using
`parallel_config={"backend": "dask", ...}`, dask-task-level progress
isn't available through this mechanism; use the
[dask dashboard](https://docs.dask.org/en/stable/dashboard.html)
instead. The case- and step-level bars still work under that backend.