# test_progress.py
"""Test progress bar with actual workflow structure"""
import time
from tqdm.auto import tqdm
import tqdm as tqdm_module
from joblib import Parallel, delayed

fake_cases = [
    {"case_id": 101, "event_type": "heatwave"},
    {"case_id": 102, "event_type": "heatwave"},
    {"case_id": 201, "event_type": "freeze"},
    {"case_id": 301, "event_type": "tropical_cyclone"},
]

fake_metrics = ["MAE", "RMSE", "CSI"]


# ============================================
# Simulates _build_datasets()
# ============================================
def build_datasets(case, progress_callback=None):
    case_info = f"Case {case['case_id']} ({case['event_type']})"
    
    if progress_callback:
        progress_callback(f"{case_info} → Building target")
    time.sleep(0.2)  # Simulate target pipeline
    
    if progress_callback:
        progress_callback(f"{case_info} → Building forecast")
    time.sleep(0.2)  # Simulate forecast pipeline
    
    return {"target": "data", "forecast": "data"}


# ============================================
# Simulates _evaluate_metric_and_return_df()
# ============================================
def evaluate_metric(metric, progress_callback=None, case_info=""):
    if progress_callback:
        progress_callback(f"{case_info} → Computing {metric}")
    time.sleep(0.1)
    return {"metric": metric, "value": 0.5}


# ============================================
# Simulates compute_case_operator()
# ============================================
def compute_case_operator(case, progress_callback=None):
    case_info = f"Case {case['case_id']} ({case['event_type']})"
    
    # Build datasets
    datasets = build_datasets(case, progress_callback)
    
    # Compute metrics
    results = []
    for metric in fake_metrics:
        results.append(evaluate_metric(metric, progress_callback, case_info))
    
    return results


# ============================================
# SERIAL: With workflow stages
# ============================================
print("=== SERIAL with workflow stages ===")
pbar = tqdm(fake_cases, desc="Evaluating", unit="case")
for case in pbar:
    # Create callback to update progress bar
    def update_status(status):
        pbar.set_postfix_str(status)
    
    compute_case_operator(case, progress_callback=update_status)

print()


# ============================================
# PARALLEL: Challenge - can't easily update from workers
# ============================================
print("=== PARALLEL (simplified - no stage updates) ===")

class ParallelTqdmImproved(Parallel):
    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        unit: str = "tasks",
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError("verbose is not supported.")
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.unit = unit
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm_module.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            return super().__call__(iterable)
        finally:
            if self.progress_bar is not None:
                self.progress_bar.close()

    def dispatch_one_batch(self, iterator):
        if self.progress_bar is None:
            self.progress_bar = tqdm_module.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit=self.unit,
            )
        return super().dispatch_one_batch(iterator)

    def print_progress(self):
        if self.progress_bar is None:
            return
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


# For parallel, we can't easily pass callbacks, so just show case completion
results = ParallelTqdmImproved(
    total_tasks=len(fake_cases),
    desc="Evaluating",
    unit="case",
    n_jobs=2,
)(
    delayed(compute_case_operator)(case, None) for case in fake_cases
)

print(f"\nDone! Processed {len(results)} cases")