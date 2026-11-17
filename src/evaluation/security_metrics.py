"""Security-Oriented Benign vs Attack Summary Engine (CP-07).

Computes critical cybersecurity metrics by evaluating model efficacy on ATTACK vs BENIGN flows:
1. Attack Detection Recall (Sensitivity): Proportion of actual attacks successfully detected.
2. Attack Precision: Proportion of raised attack alarms that were genuine intrusions.
3. Attack F1-score: Harmonic mean of attack precision and detection recall.
4. Benign False Alarm Rate (FAR): Proportion of benign traffic incorrectly flagged as attacks.
5. Benign Specificity (True Negative Rate): Proportion of benign traffic correctly permitted.
6. Attack False Negative Rate (Miss Rate): Proportion of attacks that slipped past undetected.
7. Binary Accuracy & Binary Balanced Accuracy.

INVARIANTS:
- Preserves all multiclass classifications and does NOT modify original ground truth labels.
- Safely detects canonical BENIGN labels (case-insensitive "benign").
- If no recognizable benign class exists, returns is_available=False with clear reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.evaluation.evaluation_manifest import SecurityMetricRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SecurityMetricsEngine:
    """Calculates Benign vs Attack security metrics without modifying multiclass state."""

    BENIGN_CANDIDATES = ("benign", "normal", "legitimate")

    @classmethod
    def identify_benign_class(cls, classes: List[str]) -> Optional[Tuple[int, str]]:
        """Identify index and name of the benign class in the class list."""
        for idx, cls_name in enumerate(classes):
            norm = cls_name.strip().lower()
            if norm in cls.BENIGN_CANDIDATES:
                return idx, cls_name
        return None

    @classmethod
    def compute_security_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
    ) -> SecurityMetricRecord:
        """Calculate Benign vs Attack binary metrics.

        Args:
            y_true: Ground truth target labels.
            y_pred: Model predictions.
            classes: Canonical class names.

        Returns:
            SecurityMetricRecord.
        """
        benign_info = cls.identify_benign_class(classes)
        if benign_info is None:
            return SecurityMetricRecord(
                is_available=False,
                benign_label=None,
                attack_classes=[],
                unavailable_reason="No identifiable BENIGN class found in class vocabulary",
            )

        benign_idx, benign_name = benign_info
        attack_classes = [c for i, c in enumerate(classes) if i != benign_idx]

        # Binary mapping:
        # 0 = BENIGN (Negative)
        # 1 = ATTACK (Positive)
        y_true_binary = (y_true != benign_idx).astype(int)
        y_pred_binary = (y_pred != benign_idx).astype(int)

        benign_total = int(np.sum(y_true_binary == 0))
        attack_total = int(np.sum(y_true_binary == 1))

        # Contingency table:
        # TP: True Attack predicted as Attack
        tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
        # FP: True Benign predicted as Attack (False Alarm!)
        fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
        # FN: True Attack predicted as Benign (Missed Intrusion!)
        fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
        # TN: True Benign predicted as Benign
        tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))

        # 1. Attack Detection Recall (Sensitivity): TP / (TP + FN)
        attack_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # 2. Attack Precision: TP / (TP + FP)
        attack_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # 3. Attack F1: 2 * P * R / (P + R)
        denom_f1 = attack_prec + attack_recall
        attack_f1 = float(2 * attack_prec * attack_recall / denom_f1) if denom_f1 > 0 else 0.0

        # 4. Benign False Alarm Rate (FAR): FP / (FP + TN)
        benign_far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # 5. Benign Specificity: TN / (TN + FP) = 1.0 - FAR
        benign_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # 6. Attack False Negative Rate (FNR): FN / (FN + TP) = 1.0 - Recall
        attack_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # 7. Binary Accuracy & Balanced Accuracy
        total_records = len(y_true_binary)
        bin_acc = float((tp + tn) / total_records) if total_records > 0 else 0.0
        bin_bal_acc = float((attack_recall + benign_spec) / 2.0)

        return SecurityMetricRecord(
            is_available=True,
            benign_label=benign_name,
            attack_classes=attack_classes,
            attack_detection_recall=round(attack_recall, 4),
            attack_precision=round(attack_prec, 4),
            attack_f1=round(attack_f1, 4),
            benign_false_alarm_rate=round(benign_far, 4),
            benign_specificity=round(benign_spec, 4),
            attack_false_negative_rate=round(attack_fnr, 4),
            binary_accuracy=round(bin_acc, 4),
            binary_balanced_accuracy=round(bin_bal_acc, 4),
            benign_records_count=benign_total,
            attack_records_count=attack_total,
            unavailable_reason=None,
        )
