import logging
import multiprocessing
import os

import joblib
import pandas as pd

from extremeweatherbench import cases, defaults, evaluate


def configure_root_logger():
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("joblib.log")
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)
    return root


# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load events yaml
case_yaml = cases.load_ewb_events_yaml_into_case_collection()

# Initialize ExtremeWeatherBench
ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=defaults.BRIGHTBAND_EVALUATION_OBJECTS,
)

# Get case operators
case_operators = ewb.case_operators

# Get the number of available CPUs for determining n_processes
n_threads_per_process = 10
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)

# Set environment variable to control threads per process
os.environ["OMP_NUM_THREADS"] = str(n_threads_per_process)
os.environ["MKL_NUM_THREADS"] = str(n_threads_per_process)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads_per_process)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads_per_process)


# Use joblib to parallelize with n processes, each with 10 threads
results = joblib.Parallel(n_jobs=n_processes, prefer="processes")(
    joblib.delayed(evaluate.compute_case_operator)(
        case_op, tolerance_range=48, pre_compute=True
    )
    for case_op in case_operators
)

results = pd.concat(results, ignore_index=True)
results.to_csv("brightband_evaluation_results.csv", index=False)
