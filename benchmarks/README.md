# Benchmarks

Wall-clock benchmarks for the xarray/dask hot paths. These are deliberately
kept out of the pytest suite, where wall-clock assertions are flaky. The test
suite asserts deterministic proxies instead — laziness, dask graph size, and
call counts — and this harness exists to confirm those proxies translate into
real speedups.

## Usage

```bash
# Record a baseline for your machine
python benchmarks/bench_optimizations.py --out benchmarks/baseline.json

# Make a change, then compare against it
python benchmarks/bench_optimizations.py --compare benchmarks/baseline.json

# Run a subset
python benchmarks/bench_optimizations.py --only cape region_mask
```

## About `baseline.json`

The committed `baseline.json` records timings taken at commit `d7d1809`, before
the optimization work, so that the improvements are reproducible in principle.

**These numbers are machine specific.** They came from one developer machine and
are useful only as the "before" half of a before/after pair measured on the same
hardware. Do not treat them as performance targets or compare your own timings
against them directly. Record your own baseline with `--out` before making a
change, and compare against that.
