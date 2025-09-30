import logging
import multiprocessing
import warnings

from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import cases, defaults, evaluate

# Suppress noisy log messages
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)


logging.getLogger("joblib").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress joblib/loky warnings about worker timeouts and child process errors
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Load events yaml
case_yaml = cases.load_ewb_events_yaml_into_case_collection()
# Initialize ExtremeWeatherBench
ewb = evaluate.ExtremeWeatherBench(
    cases=case_yaml,
    evaluation_objects=defaults.get_brightband_evaluation_objects(),
)
# Get the number of available CPUs for determining n_processes
n_threads_per_process = 10
n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)
with logging_redirect_tqdm(loggers=[logger]):
    results = ewb.run(n_jobs=n_processes, pre_compute=True)
results.to_csv("brightband_evaluation_results.csv", index=False)
