"""Evaluation routines for use during ExtremeWeatherBench case studies."""

import logging
import pathlib
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

from extremeweatherbench import cases
from extremeweatherbench.evaluate_tools import (
    OUTPUT_COLUMNS,
    run_case_operators,
    safe_concat,
)

if TYPE_CHECKING:
    from extremeweatherbench import inputs, regions

logger = logging.getLogger(__name__)


class ExtremeWeatherBench:
    """A class to build and run the ExtremeWeatherBench workflow.

    This class is used to run the ExtremeWeatherBench workflow. It is ultimately
    a wrapper around case operators and evaluation objects to create a parallel
    or serial run to evaluate cases and metrics, returning a concatenated
    dataframe of the results.

    Attributes:
        case_metadata: A dictionary of cases or an IndividualCaseCollection to
            run.
        evaluation_objects: A list of evaluation objects to run.
        cache_dir: An optional directory to cache the mid-flight outputs of the
            workflow for serial runs.
        region_subsetter: An optional region subsetter to subset the cases that
            are part of the evaluation to a Region object or a dictionary of
            lat/lon bounds.
    """

    def __init__(
        self,
        case_metadata: Union[dict[str, list], "cases.IndividualCaseCollection"],
        evaluation_objects: list["inputs.EvaluationObject"],
        cache_dir: Optional[Union[str, pathlib.Path]] = None,
        region_subsetter: Optional["regions.RegionSubsetter"] = None,
    ):
        if isinstance(case_metadata, dict):
            self.case_metadata = cases.load_individual_cases(case_metadata)
        elif isinstance(case_metadata, cases.IndividualCaseCollection):
            self.case_metadata = case_metadata
        else:
            raise TypeError(
                "case_metadata must be a dictionary of cases or an "
                "IndividualCaseCollection"
            )
        self.evaluation_objects = evaluation_objects
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None

        # Instantiate cache dir if needed
        if self.cache_dir:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.region_subsetter = region_subsetter

    @property
    def case_operators(self) -> list["cases.CaseOperator"]:
        """Build the CaseOperator objects from case_metadata and eval objects."""
        if self.region_subsetter:
            subset_collection = self.region_subsetter.subset_case_collection(
                self.case_metadata
            )
        else:
            subset_collection = self.case_metadata
        return cases.build_case_operators(subset_collection, self.evaluation_objects)

    def run(
        self,
        n_jobs: Optional[int] = None,
        parallel_config: Optional[dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the ExtremeWeatherBench workflow.

        This method will run the workflow in the order of the case operators,
        optionally caching the mid-flight outputs of the workflow if cache_dir
        was provided for serial runs.

        Args:
            n_jobs: The number of jobs to run in parallel. If None, defaults to the
                joblib backend default value. If 1, the workflow will run serially.
                Ignored if parallel_config is provided.
            parallel_config: Optional dictionary of joblib parallel configuration.
                If provided, this takes precedence over n_jobs. If not provided and
                n_jobs is specified, a default config with loky backend is used.

        Returns:
            A concatenated dataframe of the evaluation results.
        """
        logger.info("Running ExtremeWeatherBench workflow...")

        # Check for serial or parallel configuration
        parallel_config = _parallel_serial_config_check(n_jobs, parallel_config)
        kwargs["parallel_config"] = parallel_config
        run_results = run_case_operators(
            self.case_operators, cache_dir=self.cache_dir, **kwargs
        )

        if run_results:
            return safe_concat(run_results, ignore_index=True)
        else:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _parallel_serial_config_check(
    n_jobs: Optional[int] = None,
    parallel_config: Optional[dict] = None,
) -> Optional[dict]:
    """Check if running in serial or parallel mode.

    Args:
        n_jobs: The number of jobs to run in parallel. If None, defaults to the
            joblib backend default value. If 1, the workflow will run serially.
        parallel_config: Optional dictionary of joblib parallel configuration. If
            provided, this takes precedence over n_jobs. If not provided and n_jobs is
            specified, a default config with loky backend is used.
    Returns:
        None if running in serial mode, otherwise a dictionary of joblib parallel
        configuration.
    """
    # Determine if running in serial or parallel mode
    # Serial: n_jobs=1 or (parallel_config with n_jobs=1)
    # Parallel: n_jobs>1 or (parallel_config with n_jobs>1)
    is_serial = (
        (n_jobs == 1)
        or (parallel_config is not None and parallel_config.get("n_jobs") == 1)
        or (n_jobs is None and parallel_config is None)
    )
    logger.debug("Running in %s mode.", "serial" if is_serial else "parallel")

    if not is_serial:
        # Build parallel_config if not provided
        if parallel_config is None and n_jobs is not None:
            logger.debug(
                "No parallel_config provided, using loky backend and %s jobs.",
                n_jobs,
            )
            parallel_config = {"backend": "loky", "n_jobs": n_jobs}
    # If running in serial mode, set parallel_config to None if not already
    else:
        parallel_config = None
    # Return the maybe updated kwargs
    return parallel_config
