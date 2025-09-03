#!/usr/bin/env python3
"""
Comprehensive Extreme Weather Bench Evaluation Script

This script runs all case operators for all event types and metrics,
handling errors gracefully and recording detailed results in CSV format.
"""

import csv
import datetime
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import pandas as pd

from extremeweatherbench import defaults
from extremeweatherbench.evaluate import (
    ExtremeWeatherBench,
    _build_datasets,
    _evaluate_metric_and_return_df,
    compute_case_operator,
)
from extremeweatherbench.evaluate_cli import _load_default_cases
from extremeweatherbench.progress import enhanced_logging, progress_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose loggers
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("botocore.httpchecksum").setLevel(logging.CRITICAL)


class ComprehensiveEvaluationResults:
    """Manages comprehensive evaluation results and CSV output."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize result files
        base_name = f"comprehensive_evaluation_{timestamp}"
        self.detailed_file = self.output_dir / f"{base_name}.csv"
        self.summary_file = self.output_dir / f"{base_name}_summary.csv"
        self.error_file = self.output_dir / f"{base_name}_errors.csv"

        # Initialize CSV files with headers
        self._init_csv_files()

        # Track statistics
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "event_type_stats": {},
            "metric_stats": {},
            "errors": [],
        }

    def _init_csv_files(self):
        """Initialize CSV files with appropriate headers."""
        # Detailed results file
        detailed_headers = [
            "case_id_number",
            "event_type",
            "metric_name",
            "target_source",
            "forecast_source",
            "target_variable",
            "status",
            "error_message",
            "evaluation_timestamp",
            "lead_time_count",
            "result_rows",
        ]

        with open(self.detailed_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(detailed_headers)

        # Error details file
        error_headers = [
            "case_id_number",
            "event_type",
            "metric_name",
            "error_type",
            "error_message",
            "traceback",
            "timestamp",
        ]

        with open(self.error_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(error_headers)

    def record_success(
        self,
        case_operator,
        metric_name: str,
        target_variable: str,
        result_df: pd.DataFrame,
    ):
        """Record a successful evaluation."""
        self.stats["total_evaluations"] += 1
        self.stats["successful_evaluations"] += 1

        # Update event type stats
        event_type = case_operator.case_metadata.event_type
        if event_type not in self.stats["event_type_stats"]:
            self.stats["event_type_stats"][event_type] = {"pass": 0, "fail": 0}
        self.stats["event_type_stats"][event_type]["pass"] += 1

        # Update metric stats
        if metric_name not in self.stats["metric_stats"]:
            self.stats["metric_stats"][metric_name] = {"pass": 0, "fail": 0}
        self.stats["metric_stats"][metric_name]["pass"] += 1

        # Write to detailed CSV
        row = [
            case_operator.case_metadata.case_id_number,
            event_type,
            metric_name,
            getattr(case_operator.target, "name", "unknown"),
            getattr(case_operator.forecast, "name", "unknown"),
            target_variable,
            "SUCCESS",
            "",
            datetime.datetime.now().isoformat(),
            (len(result_df.get("lead_time", [])) if hasattr(result_df, "get") else 0),
            len(result_df) if result_df is not None else 0,
        ]

        with open(self.detailed_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def record_failure(
        self, case_operator, metric_name: str, target_variable: str, error: Exception
    ):
        """Record a failed evaluation."""
        self.stats["total_evaluations"] += 1
        self.stats["failed_evaluations"] += 1

        # Update event type stats
        event_type = case_operator.case_metadata.event_type
        if event_type not in self.stats["event_type_stats"]:
            self.stats["event_type_stats"][event_type] = {"pass": 0, "fail": 0}
        self.stats["event_type_stats"][event_type]["fail"] += 1

        # Update metric stats
        if metric_name not in self.stats["metric_stats"]:
            self.stats["metric_stats"][metric_name] = {"pass": 0, "fail": 0}
        self.stats["metric_stats"][metric_name]["fail"] += 1

        # Store error details
        error_details = {
            "case_id": case_operator.case_metadata.case_id_number,
            "event_type": event_type,
            "metric": metric_name,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        self.stats["errors"].append(error_details)

        # Write to detailed CSV
        detailed_row = [
            case_operator.case_metadata.case_id_number,
            event_type,
            metric_name,
            getattr(case_operator.target, "name", "unknown"),
            getattr(case_operator.forecast, "name", "unknown"),
            target_variable,
            "FAIL",
            str(error)[:500],  # Truncate long error messages
            datetime.datetime.now().isoformat(),
            0,
            0,
        ]

        with open(self.detailed_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(detailed_row)

        # Write to error CSV
        error_row = [
            case_operator.case_metadata.case_id_number,
            event_type,
            metric_name,
            type(error).__name__,
            str(error)[:1000],  # Allow longer error messages in error file
            traceback.format_exc(),
            datetime.datetime.now().isoformat(),
        ]

        with open(self.error_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(error_row)

    def add_result(
        self,
        case_id: str,
        event_type: str,
        metric: str,
        value,
        status: str,
        error_message: str = "",
    ):
        """Add a result from the DataFrame processing approach."""
        self.stats["total_evaluations"] += 1

        if status == "PASS":
            self.stats["successful_evaluations"] += 1
        else:
            self.stats["failed_evaluations"] += 1

        # Update event type stats
        if event_type not in self.stats["event_type_stats"]:
            self.stats["event_type_stats"][event_type] = {"pass": 0, "fail": 0}

        if status == "PASS":
            self.stats["event_type_stats"][event_type]["pass"] += 1
        else:
            self.stats["event_type_stats"][event_type]["fail"] += 1

        # Update metric stats
        if metric not in self.stats["metric_stats"]:
            self.stats["metric_stats"][metric] = {"pass": 0, "fail": 0}

        if status == "PASS":
            self.stats["metric_stats"][metric]["pass"] += 1
        else:
            self.stats["metric_stats"][metric]["fail"] += 1

        # Write to detailed CSV
        row = [
            case_id,
            event_type,
            metric,
            "unknown",  # target_source
            "unknown",  # forecast_source
            "unknown",  # target_variable
            status,
            error_message,
            datetime.datetime.now().isoformat(),
            0,  # lead_time_count
            1 if status == "PASS" else 0,  # result_rows
        ]

        with open(self.detailed_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def generate_summary(self):
        """Generate summary statistics and save to CSV."""
        summary_data = []

        for event_type, stats in self.stats["event_type_stats"].items():
            total = stats["pass"] + stats["fail"]
            success_rate = (stats["pass"] / total * 100) if total > 0 else 0

            summary_data.append(
                {
                    "event_type": event_type,
                    "FAIL": stats["fail"],
                    "PASS": stats["pass"],
                    "total": total,
                    "success_rate": round(success_rate, 1),
                }
            )

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.summary_file, index=False)

        return summary_df


def evaluate_single_metric(case_operator, metric, forecast_var, target_var, **kwargs):
    """
    Evaluate a single metric for a case operator.

    Args:
        case_operator: The case operator
        metric: The metric instance to evaluate
        forecast_var: Forecast variable
        target_var: Target variable
        **kwargs: Additional arguments

    Returns:
        DataFrame with metric results
    """
    # Build datasets
    forecast_ds, target_ds = _build_datasets(case_operator)

    if len(forecast_ds) == 0 or len(target_ds) == 0:
        return pd.DataFrame()

    # Align datasets
    aligned_forecast_ds, aligned_target_ds = (
        case_operator.target.maybe_align_forecast_to_target(forecast_ds, target_ds)
    )

    # Handle derived variables by extracting their names
    forecast_var_name = (
        forecast_var.name if hasattr(forecast_var, "name") else str(forecast_var)
    )
    target_var_name = (
        target_var.name if hasattr(target_var, "name") else str(target_var)
    )

    # Evaluate the metric
    result_df = _evaluate_metric_and_return_df(
        forecast_ds=aligned_forecast_ds,
        target_ds=aligned_target_ds,
        forecast_variable=forecast_var_name,
        target_variable=target_var_name,
        metric=metric,
        case_id_number=case_operator.case_metadata.case_id_number,
        event_type=case_operator.case_metadata.event_type,
        **kwargs,
    )

    return result_df


def evaluate_single_case_operator(
    case_operator,
    results_manager: ComprehensiveEvaluationResults,  # noqa: E501
    **kwargs,
) -> Dict:
    """
    Evaluate a single case operator with all its metrics.

    Args:
        case_operator: The case operator to evaluate
        results_manager: Results manager for logging
        **kwargs: Additional arguments for evaluation

    Returns:
        Dict with evaluation results and statistics
    """
    case_id = case_operator.case_metadata.case_id_number
    event_type = case_operator.case_metadata.event_type

    logger.info(f"Evaluating case {case_id} ({event_type})")

    evaluation_results = {
        "case_id": case_id,
        "event_type": event_type,
        "metrics_attempted": 0,
        "metrics_successful": 0,
        "metrics_failed": 0,
        "errors": [],
    }

    try:
        # Get variable pairs
        variable_pairs = list(
            zip(
                case_operator.forecast.variables,
                case_operator.target.variables,
            )
        )

        # Iterate through each metric and variable pair
        for variables in variable_pairs:
            forecast_var, target_var = variables

            # Handle derived variables
            target_var_name = (
                target_var.name if hasattr(target_var, "name") else str(target_var)
            )

            for metric_class in case_operator.metric_list:
                evaluation_results["metrics_attempted"] += 1

                try:
                    # Instantiate metric if it's a class
                    if isinstance(metric_class, type):
                        metric = metric_class()
                    else:
                        metric = metric_class

                    metric_name = getattr(metric, "name", str(metric_class))

                    logger.debug(f"  Computing {metric_name} for {target_var_name}")  # noqa: E501

                    # Run the individual metric evaluation
                    result_df = evaluate_single_metric(
                        case_operator, metric, forecast_var, target_var, **kwargs
                    )

                    # Record success
                    results_manager.record_success(
                        case_operator, metric_name, target_var_name, result_df
                    )
                    evaluation_results["metrics_successful"] += 1

                    logger.debug(f"  ✓ {metric_name} completed successfully")

                except Exception as e:
                    # Record failure
                    error_msg = f"Error in {metric_name}: {str(e)}"
                    logger.warning(error_msg)

                    results_manager.record_failure(
                        case_operator, metric_name, target_var_name, e
                    )
                    evaluation_results["metrics_failed"] += 1
                    evaluation_results["errors"].append(error_msg)

                    # Continue with next metric
                    continue

    except Exception as e:
        # Handle case-level errors
        error_msg = f"Case-level error for {case_id}: {str(e)}"
        logger.error(error_msg)
        evaluation_results["errors"].append(error_msg)

        # Still record the failure for tracking
        results_manager.record_failure(case_operator, "CASE_LEVEL_ERROR", "unknown", e)

    return evaluation_results


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./outputs",
    help="Directory for evaluation outputs (default: ./outputs)",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Optional directory for caching intermediate data",
)
@click.option(
    "--max-cases",
    type=int,
    help="Maximum number of cases to evaluate (for testing)",
)
@click.option(
    "--event-types",
    multiple=True,
    help="Specific event types to evaluate (default: all)",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    default=True,
    help="Continue evaluation even if individual cases fail",
)
@click.option(
    "--precompute",
    is_flag=True,
    help="Pre-compute datasets to avoid recomputing (faster but more memory)",
)
def run_comprehensive_evaluation(
    output_dir: str,
    cache_dir: Optional[str],
    max_cases: Optional[int],
    event_types: Tuple[str, ...],
    continue_on_error: bool,
    precompute: bool,
):
    """
    Run comprehensive evaluation of all case operators and metrics.

    This script evaluates all available case operators across all event types
    and metrics, recording both successful and failed evaluations in CSV format.
    """
    start_time = datetime.datetime.now()

    click.echo("=== Extreme Weather Bench Comprehensive Evaluation ===")
    click.echo(f"Start time: {start_time}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Continue on error: {continue_on_error}")
    event_filter = event_types if event_types else "all"
    click.echo(f"Event types filter: {event_filter}")

    # Initialize results manager
    results_manager = ComprehensiveEvaluationResults(Path(output_dir))

    try:
        # Load default cases and evaluation objects
        click.echo("\nLoading default evaluation configuration...")
        evaluation_objects = defaults.BRIGHTBAND_EVALUATION_OBJECTS
        cases_dict = _load_default_cases()
        event_types = "tropical_cyclone"
        # Filter evaluation objects by event types if specified
        if event_types:
            evaluation_objects = [
                obj for obj in evaluation_objects if obj.event_type in event_types
            ]
            click.echo(f"Filtered to {len(evaluation_objects)} evaluation objects")

        # Create ExtremeWeatherBench instance
        click.echo("Initializing ExtremeWeatherBench...")
        ewb = ExtremeWeatherBench(
            cases=cases_dict, evaluation_objects=evaluation_objects, cache_dir=cache_dir
        )
        # Get case operators for stats
        case_operators = ewb.case_operators

        # Count unique case_id_numbers
        unique_case_ids = set(op.case_metadata.case_id_number for op in case_operators)
        total_unique_cases = len(unique_case_ids)

        click.echo(
            f"Built {len(case_operators)} case operators from {total_unique_cases} "
            "unique cases"
        )

        # Limit cases for testing if specified
        if max_cases:
            # When limiting, we want to limit by unique case_id_number, not case
            # operators
            limited_case_ids = set()
            limited_case_operators = []

            for case_operator in case_operators:
                case_id = case_operator.case_metadata.case_id_number
                if len(limited_case_ids) < max_cases:
                    limited_case_ids.add(case_id)
                    limited_case_operators.append(case_operator)
                elif case_id in limited_case_ids:
                    # Include this case operator if its case_id is already in our
                    # limited set
                    limited_case_operators.append(case_operator)

            case_operators = limited_case_operators
            click.echo(
                f"Limited to {len(limited_case_ids)} unique cases "
                f"({len(case_operators)} case operators) for testing"
            )

        # Recalculate unique cases after potential limiting
        unique_case_ids = set(op.case_metadata.case_id_number for op in case_operators)
        total_unique_cases = len(unique_case_ids)

        # Calculate total metrics for progress tracking
        total_metrics = sum(
            len(op.metric_list) * len(op.forecast.variables) for op in case_operators
        )
        click.echo(f"Total unique cases: {total_unique_cases}")
        click.echo(f"Total case operators: {len(case_operators)}")
        click.echo(f"Total metric evaluations to attempt: {total_metrics}")

        # Run evaluations using ExtremeWeatherBench
        click.echo("\nStarting comprehensive evaluation...")

        kwargs = {
            "cache_dir": cache_dir,
            "pre_compute": True,  # Always use pre_compute=True as requested
        }

        # Run the evaluation workflow with comprehensive error handling
        results_df = pd.DataFrame()

        # Use individual case processing for robust error handling
        click.echo(
            "Using individual case processing for comprehensive error handling..."
        )

        try:
            results_list = []
            total_operations = sum(
                len(case_op.metric_list) for case_op in case_operators
            )

            with enhanced_logging():
                with progress_tracker.overall_workflow(
                    total_operations,
                    f"Processing {total_unique_cases} unique cases "
                    f"({len(case_operators)} operators) with "
                    f"{total_operations} total metrics",
                ):
                    for case_operator in case_operators:
                        case_id = case_operator.case_metadata.case_id_number
                        case_title = case_operator.case_metadata.title
                        num_metrics = len(case_operator.metric_list)

                        try:
                            with progress_tracker.case_processing(
                                case_id, f"{case_title}", num_metrics
                            ) as case_pbar:
                                with progress_tracker.dask_computation_context():
                                    result = compute_case_operator(
                                        case_operator, **kwargs
                                    )
                                    if not result.empty:
                                        results_list.append(result)
                                case_pbar.update(num_metrics)
                        except Exception as case_error:
                            logger.error(
                                "Error processing case "
                                f"{case_operator.case_metadata.case_id_number}: "
                                f"{case_error}"
                            )
                            # Record the case-level failure
                            try:
                                results_manager.record_failure(
                                    case_operator,
                                    "CASE_LEVEL_ERROR",
                                    "unknown",
                                    case_error,
                                )
                            except Exception as record_error:
                                logger.error(
                                    f"Failed to record case failure: {record_error}"
                                )
                            continue

            if results_list:
                results_df = pd.concat(results_list, ignore_index=True)
                click.echo(f"Processing completed, got {len(results_df)} results")
            else:
                click.echo("No successful results from processing")

        except Exception as processing_error:
            logger.error(
                f"Error during individual processing: {processing_error}", exc_info=True
            )
            click.echo(f"Warning: Processing failed with error: {processing_error}")

            # Still try to concatenate any results we got before the error
            if results_list:
                try:
                    results_df = pd.concat(results_list, ignore_index=True)
                    click.echo(f"Partial results available: {len(results_df)} results")
                except Exception as concat_error:
                    logger.error(
                        f"Failed to concatenate partial results: {concat_error}"
                    )
                    results_df = pd.DataFrame()
        # Process results for our comprehensive evaluation format
        click.echo("Processing results...")
        if not results_df.empty:
            try:
                for _, row in results_df.iterrows():
                    try:
                        # Extract relevant data from each result row
                        case_id = row.get("case_id", "unknown")
                        event_type = row.get("event_type", "unknown")
                        metric_name = row.get("metric", "unknown")
                        metric_value = row.get("value", None)
                        status = "PASS" if pd.notna(metric_value) else "FAIL"
                        # Add to results manager
                        results_manager.add_result(
                            case_id=case_id,
                            event_type=event_type,
                            metric=metric_name,
                            value=metric_value,
                            status=status,
                            error_message=""
                            if status == "PASS"
                            else "No value computed",
                        )
                    except Exception as row_error:
                        logger.error(f"Error processing result row: {row_error}")
                        # Try to add a failure record with whatever data we can extract
                        try:
                            safe_case_id = (
                                str(row.get("case_id", "unknown"))
                                if hasattr(row, "get")
                                else "unknown"
                            )
                            safe_event_type = (
                                str(row.get("event_type", "unknown"))
                                if hasattr(row, "get")
                                else "unknown"
                            )
                            safe_metric = (
                                str(row.get("metric", "unknown"))
                                if hasattr(row, "get")
                                else "unknown"
                            )
                            results_manager.add_result(
                                case_id=safe_case_id,
                                event_type=safe_event_type,
                                metric=safe_metric,
                                value=None,
                                status="FAIL",
                                error_message=f"Error processing result: {row_error}",
                            )
                        except Exception as safe_error:
                            logger.error(
                                f"Failed to record processing error: {safe_error}"
                            )
                        continue
            except Exception as processing_error:
                logger.error(
                    f"Error during results processing: {processing_error}",
                    exc_info=True,
                )
                click.echo(f"Warning: Results processing failed: {processing_error}")
        else:
            click.echo("No results to process")

        # Generate and save summary
        click.echo("\nGenerating summary statistics...")
        summary_df = results_manager.generate_summary()

        # Print summary to console
        click.echo("\n=== EVALUATION SUMMARY ===")
        total_evals = results_manager.stats["total_evaluations"]
        successful_evals = results_manager.stats["successful_evaluations"]
        failed_evals = results_manager.stats["failed_evaluations"]

        click.echo(f"Total evaluations attempted: {total_evals}")
        click.echo(f"Successful evaluations: {successful_evals}")
        click.echo(f"Failed evaluations: {failed_evals}")

        overall_success_rate = successful_evals / max(total_evals, 1) * 100
        click.echo(f"Overall success rate: {overall_success_rate:.1f}%")

        click.echo("\nBy Event Type:")
        for _, row in summary_df.iterrows():
            event_type = row["event_type"]
            passes = row["PASS"]
            total = row["total"]
            success_rate = row["success_rate"]
            click.echo(f"  {event_type}: {passes}/{total} ({success_rate}%)")

        # Print file locations
        click.echo("Results saved to:")
        click.echo(f"  Detailed results: {results_manager.detailed_file}")
        click.echo(f"  Summary: {results_manager.summary_file}")
        click.echo(f"  Errors: {results_manager.error_file}")

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        click.echo(f"\nCompleted in {duration}")

    except KeyboardInterrupt:
        click.echo("\n\nEvaluation interrupted by user")
        click.echo("Partial results may be available in output files")
        # Generate summary with whatever data we have
        try:
            summary_df = results_manager.generate_summary()
            click.echo("Partial results saved")
        except Exception as summary_error:
            logger.error(
                f"Failed to generate summary after interruption: {summary_error}"
            )
    except Exception as e:
        click.echo(f"\n\nFatal error during evaluation: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        click.echo("Attempting to save any partial results...")
        try:
            summary_df = results_manager.generate_summary()
            click.echo("Partial results saved despite error")
            # Print file locations
            click.echo("Results saved to:")
            click.echo(f"  Detailed results: {results_manager.detailed_file}")
            click.echo(f"  Summary: {results_manager.summary_file}")
            click.echo(f"  Errors: {results_manager.error_file}")
        except Exception as summary_error:
            logger.error(
                f"Failed to generate summary after fatal error: {summary_error}"
            )
        # Don't re-raise - let the script complete and show what it could accomplish


if __name__ == "__main__":
    run_comprehensive_evaluation()
