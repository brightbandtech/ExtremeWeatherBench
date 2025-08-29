"""Advanced progress tracking system for ExtremeWeatherBench workflows.

This module provides comprehensive, nested progress bars using tqdm for all major
ExtremeWeatherBench processes including:
- Overall workflow progress
- Case processing with detailed steps
- Data loading and preprocessing
- Variable derivation
- Metric computation
- Parallel execution tracking
- Dask/xarray computation progress
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional, Union

import xarray as xr
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Configure tqdm to work properly with logging
try:
    tqdm.set_lock_timeout(1)
except AttributeError:
    # Fallback for older tqdm versions
    pass


class EWBProgressTracker:
    """Enhanced progress tracking for ExtremeWeatherBench workflows.

    Features:
    - Nested progress bars for hierarchical operations
    - Data loading progress with size estimation
    - Metric computation tracking
    - Memory usage monitoring
    - Time estimation and ETA
    - Parallel execution awareness
    """

    def __init__(self, show_memory: bool = True, show_rate: bool = True):
        self.show_memory = show_memory
        self.show_rate = show_rate
        self.active_bars = {}
        self._start_time = None
        self._position_counter = 0
        self._dask_progress_enabled = False
        self._update_timer = None
        self._stop_timer = False
        self.enable_dask_progress()

    def _get_position(self):
        """Get next available position for progress bar."""
        self._position_counter += 1
        return self._position_counter

    def enable_dask_progress(self):
        """Enable dask progress tracking."""
        try:
            import importlib.util

            if importlib.util.find_spec("dask") is not None:
                self._dask_progress_enabled = True
            else:
                self._dask_progress_enabled = False
        except ImportError:
            self._dask_progress_enabled = False

    def _start_timer(self):
        """Start the timer for real-time updates."""
        self._stop_timer = False

        def update_timer():
            while not self._stop_timer:
                time.sleep(1.0)  # Update every second
                if not self._stop_timer:
                    self._refresh_bars()

        self._update_timer = threading.Thread(target=update_timer, daemon=True)
        self._update_timer.start()

    def _stop_timer_func(self):
        """Stop the timer for real-time updates."""
        self._stop_timer = True
        if self._update_timer and self._update_timer.is_alive():
            self._update_timer.join(timeout=1.0)

    def _refresh_bars(self):
        """Refresh all active progress bars to update elapsed time."""
        try:
            for bar in self.active_bars.values():
                if hasattr(bar, "refresh"):
                    bar.refresh()
        except Exception:
            # Ignore refresh errors to avoid breaking the main workflow
            pass

    @contextmanager
    def overall_workflow(self, total_operations: int, description: str):
        """Main workflow progress bar."""
        desc = f"🌤️  {description}"

        # Start the real-time timer
        self._start_timer()

        # Main progress bar stays at bottom, others above it
        with tqdm(
            total=total_operations,
            desc=desc,
            unit="ops",
            position=0,
            colour="blue",
            leave=False,
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
            mininterval=0.1,  # Allow frequent updates
            maxinterval=1.0,  # Force update at least every second
        ) as pbar:
            self._start_time = time.time()
            self.active_bars["main"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("main", None)
                self._stop_timer_func()
                elapsed = time.time() - self._start_time if self._start_time else 0

                # Ensure progress bar is at 100% before closing
                remaining = pbar.total - pbar.n
                if remaining > 0:
                    pbar.update(remaining)

                    # Use tqdm.write for clean completion message
                tqdm.write(f"✅ Workflow completed in {elapsed:.2f}s")

    @contextmanager
    def case_processing(self, case_id: int, case_title: str, metrics_count: int):
        """Progress for processing a single case with all its metrics."""
        desc = f"📊 Case {case_id}: {case_title[:50]}..."

        # Use tqdm.write for any logging during case processing
        with tqdm(
            total=metrics_count,
            desc=desc,
            unit="step",
            position=1,
            leave=False,
            colour="green",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            mininterval=0.1,
            maxinterval=1.0,
        ) as pbar:
            self.active_bars["case"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("case", None)

    @contextmanager
    def data_loading(self, source_type: str, estimated_size: Optional[str] = None):
        """Progress for data loading operations."""
        size_info = f" ({estimated_size})" if estimated_size else ""
        desc = f"📁 Loading {source_type}{size_info}"

        # For data loading, we'll use an indeterminate progress bar
        with tqdm(
            desc=desc,
            unit="chunks",
            position=2,
            leave=False,
            colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt} chunks [{elapsed}] {rate_fmt}",
        ) as pbar:
            self.active_bars["data"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("data", None)

    @contextmanager
    def variable_derivation(self, variables: list, case_id: int):
        """Progress for derived variable computation."""
        var_names = [
            getattr(v, "name", str(v)) for v in variables if not isinstance(v, str)
        ]
        if var_names:
            suffix = "..." if len(var_names) > 2 else ""
            desc = f"⚙️  Deriving: {', '.join(var_names[:2])}{suffix}"

            with tqdm(
                total=len(var_names),
                desc=desc,
                unit="var",
                position=3,
                leave=False,
                colour="yellow",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                self.active_bars["derive"] = pbar
                try:
                    yield pbar
                finally:
                    self.active_bars.pop("derive", None)
        else:
            # No derived variables, use a dummy context
            yield None

    @contextmanager
    def metric_computation(self, metric_name: str, data_size: Optional[str] = None):
        """Progress for individual metric computations."""
        size_info = f" {data_size}" if data_size else ""
        desc = f"🔢 Computing {metric_name}{size_info}"

        with tqdm(
            desc=desc,
            unit="step",
            position=7,
            leave=False,
            colour="magenta",
            bar_format="{l_bar}{bar}| {n_fmt} steps [{elapsed}] {rate_fmt}",
            mininterval=0.1,
            maxinterval=1.0,
        ) as pbar:
            self.active_bars["metric"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("metric", None)

    @contextmanager
    def parallel_execution(self, total_jobs: int, n_workers: int):
        """Progress for parallel execution tracking."""
        desc = f"🚀 Parallel execution ({n_workers} workers)"

        with tqdm(
            total=total_jobs,
            desc=desc,
            unit="job",
            position=0,
            colour="red",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        ) as pbar:
            self.active_bars["parallel"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("parallel", None)

    @contextmanager
    def dataset_alignment(self, forecast_shape: tuple, target_shape: tuple):
        """Progress for dataset alignment operations."""
        desc = f"🔄 Aligning datasets {forecast_shape} ↔ {target_shape}"

        with tqdm(
            desc=desc,
            unit="step",
            position=6,
            leave=False,
            colour="yellow",
            bar_format="{l_bar}{bar}| {n_fmt} steps [{elapsed}]",
        ) as pbar:
            self.active_bars["align"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("align", None)

    @contextmanager
    def dask_computation_context(self, show_progress: bool = True):
        """Context manager for dask/xarray computations with progress."""
        if self._dask_progress_enabled and show_progress:
            try:
                from dask.diagnostics import ProgressBar

                # Create a dask ProgressBar that outputs through tqdm
                class TqdmDaskProgressBar(ProgressBar):
                    def __init__(self):
                        super().__init__(out=None)  # Disable default output
                        self._tqdm_bar = None
                        self._task_count = 0

                    def _start(self, dsk):
                        # Create tqdm bar for dask computation
                        total = len(dsk)
                        self._tqdm_bar = tqdm(
                            total=total,
                            desc="🔄 Computing datasets",
                            unit="tasks",
                            position=4,
                            leave=False,
                            colour="cyan",
                            bar_format=(
                                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] "
                                "{rate_fmt}"
                            ),
                            dynamic_ncols=True,
                        )
                        self._task_count = 0

                    def _pretask(self, key, dsk, state):
                        # Task is starting, update progress
                        if self._tqdm_bar:
                            self._task_count += 1
                            self._tqdm_bar.update(1)

                    def _finish(self, dsk, state, errored):
                        if self._tqdm_bar:
                            # Ensure we're at 100% and close
                            remaining = self._tqdm_bar.total - self._tqdm_bar.n
                            if remaining > 0:
                                self._tqdm_bar.update(remaining)
                            self._tqdm_bar.close()

                with TqdmDaskProgressBar():
                    yield
            except ImportError:
                yield
        else:
            yield

    @contextmanager
    def xarray_computation(self, operation_name: str):
        """Progress for xarray computations."""
        desc = f"🔢 Computing {operation_name}"

        with tqdm(
            desc=desc,
            unit="chunk",
            position=5,
            leave=False,
            colour="cyan",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt} chunks [{elapsed}] {rate_fmt}",
            mininterval=0.1,
            maxinterval=1.0,
        ) as pbar:
            self.active_bars["xarray"] = pbar
            try:
                yield pbar
            finally:
                self.active_bars.pop("xarray", None)

    def update_memory_usage(self):
        """Update memory usage in active progress bars if enabled."""
        if not self.show_memory:
            return

        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            for bar_name, pbar in self.active_bars.items():
                if hasattr(pbar, "set_postfix"):
                    current_postfix = getattr(pbar, "_postfix", {})
                    current_postfix["Memory"] = f"{memory_mb:.0f}MB"
                    pbar.set_postfix(current_postfix)
        except ImportError:
            # psutil not available, skip memory tracking
            pass
        except Exception as e:
            logger.debug(f"Error updating memory usage: {e}")

    def estimate_data_size(self, dataset: Union[xr.Dataset, xr.DataArray]) -> str:
        """Estimate human-readable size of xarray dataset/dataarray."""
        try:
            if isinstance(dataset, xr.Dataset):
                total_bytes = sum(var.nbytes for var in dataset.data_vars.values())
            else:
                total_bytes = dataset.nbytes

            for unit in ["B", "KB", "MB", "GB"]:
                if total_bytes < 1024:
                    return f"{total_bytes:.1f}{unit}"
                total_bytes /= 1024
            return f"{total_bytes:.1f}TB"
        except Exception:
            return "Unknown"

    def update_data_progress(self, bar_key: str, increment: int = 1):
        """Update progress for data operations."""
        if bar_key in self.active_bars:
            self.active_bars[bar_key].update(increment)
            if self.show_memory:
                self.update_memory_usage()


# Global progress tracker instance
progress_tracker = EWBProgressTracker()


def format_dataset_info(dataset: Union[xr.Dataset, xr.DataArray]) -> str:
    """Format dataset information for progress displays."""
    if isinstance(dataset, xr.Dataset):
        vars_str = f"{len(dataset.data_vars)} vars"
        dims_str = f"{len(dataset.dims)} dims"
    else:
        vars_str = "1 var"
        dims_str = f"{len(dataset.dims)} dims"

    size = progress_tracker.estimate_data_size(dataset)
    return f"({vars_str}, {dims_str}, {size})"


@contextmanager
def enhanced_logging():
    """Context manager for enhanced logging with tqdm compatibility."""

    # Configure logging to write through tqdm
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    # Set up custom handler
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)

    # Configure both root logger and extremeweatherbench loggers
    root_logger = logging.getLogger()
    ewb_logger = logging.getLogger("extremeweatherbench")

    # Store original handlers
    original_root_handlers = root_logger.handlers[:]
    original_ewb_handlers = ewb_logger.handlers[:]
    original_ewb_level = ewb_logger.level
    original_propagate = ewb_logger.propagate

    try:
        # Remove existing handlers and add tqdm handler to both
        for handler in original_root_handlers:
            root_logger.removeHandler(handler)
        for handler in original_ewb_handlers:
            ewb_logger.removeHandler(handler)

        root_logger.addHandler(tqdm_handler)
        ewb_logger.addHandler(tqdm_handler)
        ewb_logger.setLevel(logging.INFO)
        ewb_logger.propagate = False  # Prevent double logging

        yield
    finally:
        # Restore original configuration
        root_logger.removeHandler(tqdm_handler)
        ewb_logger.removeHandler(tqdm_handler)

        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        for handler in original_ewb_handlers:
            ewb_logger.addHandler(handler)

        ewb_logger.setLevel(original_ewb_level)
        ewb_logger.propagate = original_propagate


def create_summary_progress_bar(description: str, items: list, **kwargs) -> tqdm:
    """Create a summary progress bar with smart formatting."""
    defaults = {
        "unit": "item",
        "bar_format": (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
        "dynamic_ncols": True,
    }
    defaults.update(kwargs)

    return tqdm(items, desc=description, total=len(items), **defaults)


def track_tropical_cyclone_processing(num_tracks: int, num_timesteps: int):
    """Specialized progress tracking for tropical cyclone computations."""
    desc = f"🌀 Processing TC tracks: {num_tracks} tracks × {num_timesteps} timesteps"
    return tqdm(
        total=num_tracks * num_timesteps,
        desc=desc,
        unit="track·step",
        colour="purple",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def track_atmospheric_river_computation(grid_points: int):
    """Specialized progress tracking for atmospheric river computations."""
    desc = f"🌊 Computing AR mask: {grid_points:,} grid points"
    return tqdm(
        total=grid_points,
        desc=desc,
        unit="point",
        colour="blue",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    )


def track_severe_convection_analysis(num_cells: int):
    """Specialized progress tracking for severe convection analysis."""
    desc = f"⛈️  Analyzing convection: {num_cells} cells"
    return tqdm(
        total=num_cells,
        desc=desc,
        unit="cell",
        colour="red",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    )
