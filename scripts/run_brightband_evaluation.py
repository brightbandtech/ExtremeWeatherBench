#!/usr/bin/env python3
"""Script to evaluate BRIGHTBAND_EVALUATION_OBJECTS against all cases in events.yaml.

This script creates a detailed CSV output showing pass/fail status for each
case-event-metric combination from events.yaml.
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from extremeweatherbench import cases, inputs
from extremeweatherbench.defaults import BRIGHTBAND_EVALUATION_OBJECTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DetailedBrightbandEvaluationRunner:
    """Runner class for detailed evaluation of each case-event-metric combination."""

    def __init__(self, output_file: str = "brightband_detailed_results.csv"):
        self.output_file = output_file
        self.results: List[Dict[str, Any]] = []
        self.evaluation_objects = BRIGHTBAND_EVALUATION_OBJECTS

        # Create mapping of event types to their metrics
        self.event_type_metrics = {}
        for eval_obj in self.evaluation_objects:
            if eval_obj.event_type not in self.event_type_metrics:
                self.event_type_metrics[eval_obj.event_type] = []
            self.event_type_metrics[eval_obj.event_type].extend(eval_obj.metric_list)

    def load_events_yaml(self) -> List[Dict[str, Any]]:
        """Load cases from events.yaml file.

        Returns:
            List of case dictionaries
        """
        events_path = Path("src/extremeweatherbench/data/events.yaml")
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")

        with open(events_path, "r") as f:
            data = yaml.safe_load(f)

        return data.get("cases", [])

    def test_metric_instantiation(self, metric_class: Any) -> Tuple[bool, str]:
        """Test if a metric can be instantiated successfully.

        Args:
            metric_class: The metric class or function to test

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Try to instantiate the metric
            if isinstance(metric_class, type):
                metric = metric_class()
            else:
                metric = metric_class

            # Check if the metric has required methods
            if not hasattr(metric, "compute_metric") and not hasattr(
                metric, "_compute_metric"
            ):
                return False, "Missing compute methods"

            # Check if metric has a name attribute
            if not hasattr(metric, "name"):
                return False, "Missing name attribute"

            return True, ""

        except NotImplementedError as e:
            return False, f"NotImplementedError: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def test_case_operator_creation(
        self, case: Dict[str, Any], eval_obj: inputs.EvaluationObject
    ) -> Tuple[bool, str]:
        """Test if case operators can be created for a case and evaluation object.

        Args:
            case: Case dictionary from events.yaml
            eval_obj: EvaluationObject to test with

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Create cases dict in expected format
            cases_dict = {"cases": [case]}

            # Try to build case operators
            case_operators = cases.build_case_operators(cases_dict, [eval_obj])

            if not case_operators:
                return False, "No case operators generated"

            return True, ""

        except Exception as e:
            return False, f"Case operator creation failed: {str(e)}"

    def evaluate_single_case_metric(
        self, case: Dict[str, Any], metric_class: Any, eval_obj: inputs.EvaluationObject
    ) -> Dict[str, Any]:
        """Evaluate a single case-metric combination.

        Args:
            case: Case dictionary from events.yaml
            metric_class: Metric class to test
            eval_obj: EvaluationObject containing the metric

        Returns:
            Dictionary with evaluation results
        """
        start_time = datetime.now()
        metric_name = getattr(metric_class, "__name__", str(metric_class))

        result = {
            "case_id_number": case.get("case_id_number"),
            "case_title": case.get("title", ""),
            "event_type": case.get("event_type"),
            "metric_name": metric_name,
            "start_date": case.get("start_date"),
            "end_date": case.get("end_date"),
            "target_source": getattr(eval_obj.target, "source", "unknown"),
            "forecast_source": getattr(eval_obj.forecast, "source", "unknown"),
            "start_time": start_time.isoformat(),
            "status": "UNKNOWN",
            "error_message": None,
            "processing_time_seconds": 0,
        }

        try:
            # Test 1: Can we instantiate the metric?
            metric_success, metric_error = self.test_metric_instantiation(metric_class)
            if not metric_success:
                result["status"] = "FAIL"
                result["error_message"] = f"Metric instantiation failed: {metric_error}"
                return result

            # Test 2: Can we create case operators for this case?
            case_op_success, case_op_error = self.test_case_operator_creation(
                case, eval_obj
            )
            if not case_op_success:
                result["status"] = "FAIL"
                result["error_message"] = (
                    f"Case operator creation failed: {case_op_error}"
                )
                return result

            # If we get here, both tests passed
            result["status"] = "PASS"
            result["error_message"] = None

        except Exception as e:
            result["status"] = "FAIL"
            result["error_message"] = f"Unexpected error: {str(e)}"
            logger.error(
                f"Unexpected error for case {case.get('case_id_number')}, "
                f"metric {metric_name}: {e}"
            )

        finally:
            end_time = datetime.now()
            result["end_time"] = end_time.isoformat()
            result["processing_time_seconds"] = (end_time - start_time).total_seconds()

        return result

    def run_detailed_evaluation(self) -> pd.DataFrame:
        """Run detailed evaluation for all case-event-metric combinations.

        Returns:
            DataFrame with one row per case-metric combination
        """
        # Load all cases from events.yaml
        try:
            all_cases = self.load_events_yaml()
            logger.info(f"Loaded {len(all_cases)} cases from events.yaml")
        except Exception as e:
            logger.error(f"Failed to load events.yaml: {e}")
            return pd.DataFrame()

        # Group cases by event type for processing
        cases_by_event_type = {}
        for case in all_cases:
            event_type = case.get("event_type")
            if event_type not in cases_by_event_type:
                cases_by_event_type[event_type] = []
            cases_by_event_type[event_type].append(case)

        # Calculate total combinations for progress tracking
        total_combinations = 0
        for event_type, event_cases in cases_by_event_type.items():
            if event_type in self.event_type_metrics:
                total_combinations += len(event_cases) * len(
                    self.event_type_metrics[event_type]
                )

        logger.info(f"Total case-metric combinations to evaluate: {total_combinations}")

        # Process each event type
        processed = 0
        for event_type, event_cases in cases_by_event_type.items():
            if event_type not in self.event_type_metrics:
                logger.warning(f"No metrics defined for event type: {event_type}")
                continue

            # Find the evaluation object for this event type
            eval_obj = None
            for obj in self.evaluation_objects:
                if obj.event_type == event_type:
                    eval_obj = obj
                    break

            if eval_obj is None:
                logger.warning(
                    f"No evaluation object found for event type: {event_type}"
                )
                continue

            logger.info(
                f"Processing {len(event_cases)} cases for event type: {event_type}"
            )

            # Process each case for this event type
            for case in event_cases:
                # Process each metric for this case
                for metric_class in self.event_type_metrics[event_type]:
                    processed += 1
                    # Progress update every 50 combinations
                    if processed % 50 == 0:
                        logger.info(
                            f"Progress: {processed}/{total_combinations} "
                            "combinations processed"
                        )

                    result = self.evaluate_single_case_metric(
                        case, metric_class, eval_obj
                    )
                    self.results.append(result)

        # Create DataFrame from results
        df = pd.DataFrame(self.results)

        if not df.empty:
            # Summary statistics
            total_rows = len(df)
            passed = len(df[df["status"] == "PASS"])
            failed = len(df[df["status"] == "FAIL"])
            success_rate = (passed / total_rows) * 100

            logger.info("\n" + "=" * 60)
            logger.info("DETAILED EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total case-metric combinations: {total_rows}")
            logger.info(f"Passed: {passed}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Success rate: {success_rate:.1f}%")

            # Summary by event type
            logger.info("\nBreakdown by event type:")
            summary_by_event = (
                df.groupby("event_type")
                .agg(
                    {"status": lambda x: (x == "PASS").sum(), "case_id_number": "count"}
                )
                .rename(columns={"status": "passed", "case_id_number": "total"})
            )
            summary_by_event["failed"] = (
                summary_by_event["total"] - summary_by_event["passed"]
            )
            summary_by_event["success_rate"] = (
                summary_by_event["passed"] / summary_by_event["total"] * 100
            ).round(1)

            for event_type, row in summary_by_event.iterrows():
                logger.info(
                    f"  {event_type}: {row['passed']}/{row['total']} passed "
                    f"({row['success_rate']}%)"
                )

        return df

    def save_results(self, df: pd.DataFrame) -> None:
        """Save results to CSV file.

        Args:
            df: DataFrame containing evaluation results
        """
        try:
            # Sort by case_id_number for easy reading
            df_sorted = df.sort_values(["case_id_number", "metric_name"])
            df_sorted.to_csv(self.output_file, index=False)
            logger.info(f"Detailed results saved to {self.output_file}")

            # Also save a summary file grouped by status
            summary_file = self.output_file.replace(".csv", "_summary.csv")

            if not df.empty:
                summary_df = (
                    df.groupby(["event_type", "status"]).size().unstack(fill_value=0)
                )
                summary_df["total"] = summary_df.sum(axis=1)
                if "PASS" in summary_df.columns:
                    summary_df["success_rate"] = (
                        summary_df["PASS"] / summary_df["total"] * 100
                    ).round(1)
                summary_df.to_csv(summary_file)
                logger.info(f"Summary saved to {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main function to run the detailed evaluation."""
    # Create output directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"brightband_detailed_evaluation_{timestamp}.csv"

    runner = DetailedBrightbandEvaluationRunner(str(output_file))

    try:
        results_df = runner.run_detailed_evaluation()

        if results_df.empty:
            logger.error("No results generated")
            return 2

        runner.save_results(results_df)

        # Print final status
        total_combinations = len(results_df)
        passed_combinations = len(results_df[results_df["status"] == "PASS"])

        if passed_combinations == total_combinations:
            logger.info("🎉 All case-metric combinations passed!")
            return 0
        else:
            failed_combinations = total_combinations - passed_combinations
            logger.warning(
                f"⚠️  {failed_combinations}/{total_combinations} "
                "case-metric combinations had issues"
            )
            return 1

    except Exception as e:
        logger.error(f"Evaluation failed with critical error: {e}")
        logger.error(traceback.format_exc())
        return 2


if __name__ == "__main__":
    sys.exit(main())
