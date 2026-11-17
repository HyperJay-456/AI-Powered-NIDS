"""Multi-Model Comparison & Generalization Analysis Engine (CP-07).

Provides comparative analysis across the three authorized algorithm families:
1. Support Vector Machine (LinearSVC)
2. Multilayer Perceptron (MLP)
3. Decision Tree (CART)

Features:
- Validation comparison table with key metrics, latency, and resource footprints.
- Class-wise comparative breakdown highlighting performance on rare attack classes.
- Generalization gap analysis comparing validation vs holdout test partitions for frozen candidates.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.evaluation.evaluation_manifest import (
    AlgorithmEvaluationResult,
    CandidateModelFreezeRecord,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelComparator:
    """Compares multiple evaluated models across validation and test partitions."""

    @staticmethod
    def build_validation_comparison_dataframe(
        results_by_algo: Dict[str, AlgorithmEvaluationResult],
    ) -> pd.DataFrame:
        """Construct a structured comparison DataFrame across all evaluated algorithms on validation data.

        Args:
            results_by_algo: Mapping of algorithm key to AlgorithmEvaluationResult.

        Returns:
            pd.DataFrame with formatted comparative metrics.
        """
        rows: List[Dict[str, Any]] = []

        for algo_key, res in results_by_algo.items():
            agg = res.aggregate_metrics
            sec = res.security_metrics

            rows.append({
                "Algorithm": res.algorithm_display_name,
                "Accuracy": f"{agg.accuracy * 100:.2f}%",
                "Balanced Acc": f"{agg.balanced_accuracy * 100:.2f}%",
                "Macro F1": f"{agg.macro_f1 * 100:.2f}%",
                "Weighted F1": f"{agg.weighted_f1 * 100:.2f}%",
                "Macro Recall": f"{agg.macro_recall * 100:.2f}%",
                "Macro Precision": f"{agg.macro_precision * 100:.2f}%",
                "MCC": f"{agg.matthews_corrcoef:.4f}",
                "Macro AUC": f"{res.macro_roc_auc:.4f}" if res.macro_roc_auc is not None else "N/A",
                "Macro AP": f"{res.macro_average_precision:.4f}" if res.macro_average_precision is not None else "N/A",
                "Attack Recall": f"{sec.attack_detection_recall * 100:.2f}%" if sec.is_available and sec.attack_detection_recall is not None else "N/A",
                "False Alarm Rate": f"{sec.benign_false_alarm_rate * 100:.2f}%" if sec.is_available and sec.benign_false_alarm_rate is not None else "N/A",
                "Latency (ms/rec)": f"{res.ms_per_record:.3f} ms",
                "Duration": f"{res.duration_seconds:.2f} s",
                "Evaluated Rows": f"{res.evaluated_rows:,}",
            })

        return pd.DataFrame(rows)

    @staticmethod
    def build_classwise_comparison_dataframe(
        results_by_algo: Dict[str, AlgorithmEvaluationResult],
    ) -> pd.DataFrame:
        """Construct a comparative table showing Recall and F1 per attack class across models.

        Args:
            results_by_algo: Mapping of algorithm key to AlgorithmEvaluationResult.

        Returns:
            pd.DataFrame comparing class-level recall across algorithms.
        """
        if not results_by_algo:
            return pd.DataFrame()

        # Extract classes from the first available result
        first_res = next(iter(results_by_algo.values()))
        classes = [m.class_name for m in first_res.per_class_metrics]

        rows: List[Dict[str, Any]] = []

        for idx, cls_name in enumerate(classes):
            row: Dict[str, Any] = {"Class": cls_name}

            # Get support from first result
            row["Support"] = first_res.per_class_metrics[idx].support

            for algo_key, res in results_by_algo.items():
                if idx < len(res.per_class_metrics):
                    m = res.per_class_metrics[idx]
                    row[f"{res.algorithm_display_name} Recall"] = f"{m.recall * 100:.1f}%"
                    row[f"{res.algorithm_display_name} F1"] = f"{m.f1_score * 100:.1f}%"
                else:
                    row[f"{res.algorithm_display_name} Recall"] = "N/A"
                    row[f"{res.algorithm_display_name} F1"] = "N/A"

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def build_generalization_gap_dataframe(
        val_results: Dict[str, AlgorithmEvaluationResult],
        test_results: Dict[str, AlgorithmEvaluationResult],
    ) -> pd.DataFrame:
        """Calculate validation versus test generalization differences for frozen candidates.

        Args:
            val_results: Mapping of algorithm key to validation results.
            test_results: Mapping of algorithm key to test holdout results.

        Returns:
            pd.DataFrame showing validation, test, and delta (generalization gap).
        """
        rows: List[Dict[str, Any]] = []

        common_algos = set(val_results.keys()).intersection(test_results.keys())

        for algo_key in sorted(common_algos):
            v = val_results[algo_key]
            t = test_results[algo_key]

            v_acc = v.aggregate_metrics.accuracy
            t_acc = t.aggregate_metrics.accuracy
            delta_acc = (t_acc - v_acc) * 100.0

            v_f1 = v.aggregate_metrics.macro_f1
            t_f1 = t.aggregate_metrics.macro_f1
            delta_f1 = (t_f1 - v_f1) * 100.0

            v_far = v.security_metrics.benign_false_alarm_rate or 0.0
            t_far = t.security_metrics.benign_false_alarm_rate or 0.0
            delta_far = (t_far - v_far) * 100.0

            v_rec = v.security_metrics.attack_detection_recall or 0.0
            t_rec = t.security_metrics.attack_detection_recall or 0.0
            delta_rec = (t_rec - v_rec) * 100.0

            rows.append({
                "Algorithm": v.algorithm_display_name,
                "Val Accuracy": f"{v_acc * 100:.2f}%",
                "Test Accuracy": f"{t_acc * 100:.2f}%",
                "Acc Gap (Delta)": f"{delta_acc:+.2f}%",
                "Val Macro F1": f"{v_f1 * 100:.2f}%",
                "Test Macro F1": f"{t_f1 * 100:.2f}%",
                "F1 Gap (Delta)": f"{delta_f1:+.2f}%",
                "Val Attack Recall": f"{v_rec * 100:.2f}%",
                "Test Attack Recall": f"{t_rec * 100:.2f}%",
                "Val False Alarm Rate": f"{v_far * 100:.2f}%",
                "Test False Alarm Rate": f"{t_far * 100:.2f}%",
            })

        return pd.DataFrame(rows)
