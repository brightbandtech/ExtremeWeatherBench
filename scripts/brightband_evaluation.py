import logging
import multiprocessing

from extremeweatherbench import cases, defaults, evaluate


def configure_logger(level=logging.INFO):
    """Configure the logger for the extremeweatherbench package."""
    logger = logging.getLogger("extremeweatherbench")
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("joblib.log")
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    return logger


logger = configure_logger()

if __name__ == "__main__":
    # Load events yaml
    case_yaml = cases.load_ewb_events_yaml_into_case_list()

    # Initialize ExtremeWeatherBench
    ewb = evaluate.ExtremeWeatherBench(
        case_metadata=case_yaml,
        evaluation_objects=defaults.get_brightband_evaluation_objects(),
    )
    # Get the number of available CPUs for determining n_processes and divide by 4 threads
    # per process; this is optional. Leaving n_jobs blank will use the joblib backend
    # default (which, if loky, is the number of available CPUs).
    n_threads_per_process = 4
    n_processes = max(1, multiprocessing.cpu_count() // n_threads_per_process)

    # Set up parallel configuration
    parallel_config = {"backend": "loky", "n_jobs": n_processes}
    results = ewb.run_evaluation(parallel_config=parallel_config)
    results.to_csv("brightband_evaluation_results.csv", index=False)
