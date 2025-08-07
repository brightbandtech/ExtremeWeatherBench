"""Evaluation routines for use during ExtremeWeatherBench case studies / analyses."""

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from extremeweatherbench import case, derived, utils

if TYPE_CHECKING:
    from extremeweatherbench import config, inputs, metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtremeWeatherBench:
    def __init__(
        self,
        cases: dict[str, list],
        metrics: list["config.MetricEvaluationObject"],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.cases = cases
        self.metrics = metrics
        self.cache_dir = cache_dir

    @property
    def case_operators(self) -> list["case.CaseOperator"]:
        return case.build_case_operators(self.cases, self.metrics)

    def run(
        self,
        **kwargs,
    ) -> pd.DataFrame:
        """Runs the workflow in the order of the event operators and cases inside the event operators."""

        # instantiate the cache directory if caching and build it if it does not exist
        if self.cache_dir:
            if isinstance(self.cache_dir, str):
                self.cache_dir = Path(self.cache_dir)
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        run_results = []
        with logging_redirect_tqdm():
            for case_operator in tqdm(self.case_operators):
                run_results.append(self.compute_case_operator(case_operator, **kwargs))

                # store the results of each case operator if caching
                if self.cache_dir:
                    pd.concat(run_results).to_pickle(
                        self.cache_dir / "case_results.pkl"
                    )
        return pd.concat(run_results, ignore_index=True)

    def compute_case_operator(self, case_operator: "case.CaseOperator", **kwargs):
        target_ds, forecast_ds = self._build_datasets(case_operator, **kwargs)

        # align the target and forecast datasets to ensure they have the same valid_time dimension
        target_ds, forecast_ds = xr.align(target_ds, forecast_ds)

        # compute and cache the datasets if requested
        if kwargs.get("pre_compute", False):
            target_ds, forecast_ds = self._compute_and_maybe_cache(
                target_ds, forecast_ds
            )

        logger.info(f"datasets built for case {case_operator.case.case_id_number}")
        results = []
        for variables, metric in itertools.product(
            zip(
                case_operator.forecast_config.variables,
                case_operator.target_config.variables,
            ),
            case_operator.metric,
        ):
            results.append(
                self._evaluate_metric_and_return_df(
                    target_ds=target_ds,
                    forecast_ds=forecast_ds,
                    forecast_variable=variables[0],
                    target_variable=variables[1],
                    metric=metric,
                    target_name=case_operator.target_config.target.name,
                    case_id_number=case_operator.case.case_id_number,
                    event_type=case_operator.case.event_type,
                    **kwargs,
                )
            )

            # cache the results of each metric if caching
            if self.cache_dir:
                results.to_pickle(self.cache_dir / "results.pkl")

        return pd.concat(results, ignore_index=True)

    def _compute_and_maybe_cache(self, *datasets: xr.Dataset):
        """Compute and cache the datasets if caching."""
        logger.info("computing datasets")
        computed_datasets = (dataset.compute() for dataset in datasets)
        if self.cache_dir:
            raise NotImplementedError("Caching is not implemented yet")
            # (computed_dataset.to_netcdf(self.cache_dir) for computed_dataset in computed_datasets)
        return computed_datasets

    def _evaluate_metric_and_return_df(
        self,
        forecast_ds: xr.Dataset,
        target_ds: xr.Dataset,
        forecast_variable: Union[str, "derived.DerivedVariable"],
        target_variable: Union[str, "derived.DerivedVariable"],
        metric: "metrics.BaseMetric",
        case_id_number: int,
        event_type: str,
        **kwargs,
    ):
        metric = metric()
        logger.info(f"computing metric {metric.name}")
        metric_result = metric.compute_metric(
            forecast_ds[forecast_variable],
            target_ds[target_variable],
            **kwargs,
        )

        # Convert to DataFrame and add metadata
        df = metric_result.to_dataframe(name="value").reset_index()
        df["target_variable"] = target_variable
        df["metric"] = metric.name
        df["target_source"] = target_ds.attrs["source"]
        df["forecast_source"] = forecast_ds.attrs["source"]
        df["case_id_number"] = case_id_number
        df["event_type"] = event_type
        return df

    def _build_datasets(self, case_operator: "case.CaseOperator", **kwargs):
        """Build the target and forecast datasets for a case operator.

        This method will process through all stages of the pipeline for the target and forecast datasets,
        including preprocessing, variable renaming, and subsetting.
        """
        target_input = case_operator.target_config.target(
            source=case_operator.target_config.source,
            variables=case_operator.target_config.variables,
            variable_mapping=case_operator.target_config.variable_mapping,
            storage_options=case_operator.target_config.storage_options,
            preprocess=case_operator.target_config.preprocess,
        )

        forecast_input = case_operator.forecast_config.forecast(
            source=case_operator.forecast_config.source,
            variables=case_operator.forecast_config.variables,
            variable_mapping=case_operator.forecast_config.variable_mapping,
            storage_options=case_operator.forecast_config.storage_options,
            preprocess=case_operator.forecast_config.preprocess,
        )

        logger.info("running target pipeline")
        target_ds = run_pipeline(
            input_data=target_input,
            case_operator=case_operator,
        )

        logger.info("running forecast pipeline")
        forecast_ds = run_pipeline(
            input_data=forecast_input,
            case_operator=case_operator,
        )
        return target_ds, forecast_ds


def run_pipeline(
    input_data: "inputs.InputBase",
    case_operator: "case.CaseOperator",
    **kwargs,
) -> xr.Dataset:
    """
    Shared method for running the target pipeline.

    Args:
        input_data: The input data to run the pipeline on.
        case_operator: The case operator to run the pipeline on.
        **kwargs: Additional keyword arguments to pass in as needed.

    Returns:
        The target data with a type determined by the user.
    """

    # Open data and process through pipeline steps
    data = (
        # opens data from user-defined source
        input_data.open_and_maybe_preprocess_data_from_source()
        # maps variable names to the target data if not already using EWB naming conventions
        .pipe(
            utils.maybe_map_variable_names,
            variable_mapping=input_data.variable_mapping,
        )
        # subsets the target data using the caseoperator metadata
        .pipe(
            input_data.subset_data_to_case,
            case_operator=case_operator,
        )
        # converts the target data to an xarray dataset if it is not already
        .pipe(input_data.maybe_convert_to_dataset)
        # derives variables from the target data if derived variables are defined
        .pipe(derived.maybe_derive_variables, variables=input_data.variables)
        .pipe(input_data.add_source_to_dataset_attrs)
    )
    return data
