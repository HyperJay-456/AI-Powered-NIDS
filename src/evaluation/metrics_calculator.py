"""Multiclass Metrics Calculation Engine (CP-07).

Computes comprehensive multiclass aggregate and per-class diagnostic metrics:
1. Aggregate metrics:
   - Accuracy, Balanced Accuracy
   - Macro Precision, Recall, F1 (prominently surfaced for intrusion detection)
   - Weighted Precision, Recall, F1
   - Matthews Correlation Coefficient (MCC)
   - Cohen's Kappa
2. Per-class metrics:
   - Class name, index, support, predicted count
   - One-vs-rest TP, FP, FN, TN
   - Precision, Recall, F1
   - Specificity (TNR), False Positive Rate (FPR), False Negative Rate (FNR)
3. Explicit zero-division handling and documentation for unpredicted/absent classes.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.evaluation.evaluation_manifest import (
    AggregateMetricsRecord,
    PerClassMetricRecord,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculates scientifically rigorous multiclass classification metrics."""

    @staticmethod
    def calculate_aggregate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> AggregateMetricsRecord:
        """Calculate aggregate multiclass metrics with zero-division protection.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predicted labels.

        Returns:
            AggregateMetricsRecord.
        """
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0.0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0.0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

        weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0.0))
        weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0.0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0))

        try:
            mcc = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            mcc = 0.0

        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            kappa = 0.0

        return AggregateMetricsRecord(
            accuracy=acc,
            balanced_accuracy=bal_acc,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_precision=weighted_p,
            weighted_recall=weighted_r,
            weighted_f1=weighted_f1,
            matthews_corrcoef=mcc,
            cohen_kappa=kappa,
        )

    @staticmethod
    def calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
    ) -> List[PerClassMetricRecord]:
        """Calculate one-vs-rest per-class metrics and error rates.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predicted labels.
            classes: Canonical class names list from label mapping.

        Returns:
            List of PerClassMetricRecord for each class.
        """
        records: List[PerClassMetricRecord] = []

        for idx, cls_name in enumerate(classes):
            is_true = (y_true == idx)
            is_pred = (y_pred == idx)

            support = int(np.sum(is_true))
            pred_count = int(np.sum(is_pred))

            tp = int(np.sum(is_true & is_pred))
            fp = int(np.sum(~is_true & is_pred))
            fn = int(np.sum(is_true & ~is_pred))
            tn = int(np.sum(~is_true & ~is_pred))

            # Precision: TP / (TP + FP)
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            # F1: 2 * P * R / (P + R)
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Specificity (True Negative Rate): TN / (TN + FP)
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            # False Positive Rate (Fallout): FP / (FP + TN)
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            # False Negative Rate (Miss Rate): FN / (FN + TP)
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            notes_list = []
            if support == 0:
                notes_list.append("Zero ground-truth support in evaluated partition")
            if pred_count == 0:
                notes_list.append("Never predicted by model")

            notes = "; ".join(notes_list) if notes_list else None

            records.append(
                PerClassMetricRecord(
                    class_index=idx,
                    class_name=cls_name,
                    support=support,
                    predicted_count=pred_count,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    true_negatives=tn,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    specificity=specificity,
                    false_positive_rate=fpr,
                    false_negative_rate=fnr,
                    notes=notes,
                )
            )

        return records
