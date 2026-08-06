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
parallel_config = {"backend":"loky","n_jobs":len(evaluation_objects)}

outputs = ewb.run_evaluation(parallel_config=parallel_config)
```

The _safest_ approach is to run EWB in serial, with `n_jobs` set to 1. `Dask` will still be invoked during each `CaseOperator` when the case executes and computes the directed acyclic graph, only one at a time. That said, for evaluations with more cases this approach would likely be too time-consuming. 

## Optional: Query-Optimized Dask Arrays

If the optional [`dask-array`](https://github.com/mrocklin/dask-array) package is installed (`pip install "extremeweatherbench[dask-array]"` or `pip install dask-array`), EWB automatically registers it as xarray's chunk manager before each `CaseOperator` opens its target/forecast data, so its graphs get query-optimized (reordered/fused) instead of using the standard `dask.array` chunk manager. Nothing else needs to change; this happens transparently the first time a case operator runs in a given process.

Because parallel runs execute each `CaseOperator` in its own worker process (e.g. a `loky` subprocess, or a `dask` distributed worker), registration happens independently in every worker the first time it computes a case, rather than once globally.

To opt out and keep the standard `dask.array` chunk manager even with `dask-array` installed, set:

```python
from extremeweatherbench import evaluate

evaluate.USE_DASK_ARRAY_QUERY_OPTIMIZATION = False
```

**Known limitation:** registering a chunk manager only affects arrays created afterward. If you build your own `XarrayForecast` from a dataset you already opened and chunked yourself before handing it to EWB, that dataset keeps using whichever chunk manager was active when you opened it.