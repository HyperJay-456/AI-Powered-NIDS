"""Evaluation Manifest & Metrics Schema (CP-07).

Defines structured dataclasses and serialization schemas for multiclass evaluation:
- Aggregate classification metrics (Macro/Weighted, Accuracy, Balanced Acc, MCC, Kappa)
- Per-class metrics (Support, TP, FP, FN, TN, Specificity, FPR, FNR)
- Confusion matrices (raw counts and row-normalized)
- One-vs-rest ROC analysis (handling LinearSVC decision_function vs predict_proba)
- One-vs-rest Precision-Recall curves & Average Precision
- Benign vs Attack security metrics (FAR, Detection Recall, Attack Precision)
- Candidate model freezing records & immutable test holdout gating
- Master evaluation run manifests
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import platform
from typing import Any, Dict, List, Optional
import numpy as np
import scipy
import sklearn

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AggregateMetricsRecord:
    """Aggregate multiclass classification metrics."""

    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    matthews_corrcoef: float
    cohen_kappa: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerClassMetricRecord:
    """Class-wise performance and diagnostic metrics."""

    class_index: int
    class_name: str
    support: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ConfusionMatrixRecord:
    """Multiclass confusion matrix (raw counts and row-normalized percentages)."""

    classes: List[str]
    raw_matrix: List[List[int]]
    normalized_matrix: List[List[float]]
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ROCMetricRecord:
    """One-vs-rest ROC curve and AUC score for an individual class."""

    class_name: str
    is_available: bool
    auc: Optional[float] = None
    fpr: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    score_type: str = "predict_proba"  # "decision_function" or "predict_proba"
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PrecisionRecallMetricRecord:
    """One-vs-rest Precision-Recall curve and Average Precision for a class."""

    class_name: str
    is_available: bool
    average_precision: Optional[float] = None
    precision_points: List[float] = field(default_factory=list)
    recall_points: List[float] = field(default_factory=list)
    positive_samples: int = 0
    negative_samples: int = 0
    score_type: str = "predict_proba"
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SecurityMetricRecord:
    """Benign versus Attack binary summary metrics."""

    is_available: bool
    benign_label: Optional[str] = None
    attack_classes: List[str] = field(default_factory=list)
    attack_detection_recall: Optional[float] = None
    attack_precision: Optional[float] = None
    attack_f1: Optional[float] = None
    benign_false_alarm_rate: Optional[float] = None
    benign_specificity: Optional[float] = None
    attack_false_negative_rate: Optional[float] = None
    binary_accuracy: Optional[float] = None
    binary_balanced_accuracy: Optional[float] = None
    benign_records_count: int = 0
    attack_records_count: int = 0
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AlgorithmEvaluationResult:
    """Complete evaluation results for an algorithm on a specific partition."""

    algorithm: str  # "svm", "mlp", "decision_tree"
    algorithm_display_name: str
    training_run_id: str
    model_checksum: str
    partition: str  # "validation" or "test"
    evaluated_rows: int
    failed_rows: int
    duration_seconds: float
    ms_per_record: float
    score_type: str
    aggregate_metrics: AggregateMetricsRecord
    per_class_metrics: List[PerClassMetricRecord]
    confusion_matrix: ConfusionMatrixRecord
    roc_analysis: Dict[str, ROCMetricRecord]
    macro_roc_auc: Optional[float]
    pr_analysis: Dict[str, PrecisionRecallMetricRecord]
    macro_average_precision: Optional[float]
    security_metrics: SecurityMetricRecord
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    is_test_fixture: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "algorithm": self.algorithm,
            "algorithm_display_name": self.algorithm_display_name,
            "training_run_id": self.training_run_id,
            "model_checksum": self.model_checksum,
            "partition": self.partition,
            "evaluated_rows": self.evaluated_rows,
            "failed_rows": self.failed_rows,
            "duration_seconds": self.duration_seconds,
            "ms_per_record": self.ms_per_record,
            "score_type": self.score_type,
            "aggregate_metrics": self.aggregate_metrics.to_dict() if hasattr(self.aggregate_metrics, "to_dict") else self.aggregate_metrics,
            "per_class_metrics": [m.to_dict() if hasattr(m, "to_dict") else m for m in self.per_class_metrics],
            "confusion_matrix": self.confusion_matrix.to_dict() if hasattr(self.confusion_matrix, "to_dict") else self.confusion_matrix,
            "roc_analysis": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.roc_analysis.items()},
            "macro_roc_auc": self.macro_roc_auc,
            "pr_analysis": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.pr_analysis.items()},
            "macro_average_precision": self.macro_average_precision,
            "security_metrics": self.security_metrics.to_dict() if hasattr(self.security_metrics, "to_dict") else self.security_metrics,
            "warnings": self.warnings,
            "error_message": self.error_message,
            "is_test_fixture": self.is_test_fixture,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmEvaluationResult":
        """Reconstruct from dictionary."""
        d = dict(data)
        if isinstance(d.get("aggregate_metrics"), dict):
            d["aggregate_metrics"] = AggregateMetricsRecord(**d["aggregate_metrics"])
        if isinstance(d.get("per_class_metrics"), list):
            d["per_class_metrics"] = [
                PerClassMetricRecord(**item) if isinstance(item, dict) else item
                for item in d["per_class_metrics"]
            ]
        if isinstance(d.get("confusion_matrix"), dict):
            d["confusion_matrix"] = ConfusionMatrixRecord(**d["confusion_matrix"])
        if isinstance(d.get("roc_analysis"), dict):
            d["roc_analysis"] = {
                k: ROCMetricRecord(**v) if isinstance(v, dict) else v
                for k, v in d["roc_analysis"].items()
            }
        if isinstance(d.get("pr_analysis"), dict):
            d["pr_analysis"] = {
                k: PrecisionRecallMetricRecord(**v) if isinstance(v, dict) else v
                for k, v in d["pr_analysis"].items()
            }
        if isinstance(d.get("security_metrics"), dict):
            d["security_metrics"] = SecurityMetricRecord(**d["security_metrics"])
        return cls(**d)


@dataclass
class CandidateModelFreezeRecord:
    """Immutable record of candidate model artifacts frozen prior to test holdout evaluation."""

    algorithm: str
    algorithm_display_name: str
    training_run_id: str
    model_checksum: str
    frozen_at_utc: str
    selected_as_primary: bool = False
    selection_criterion: str = ""
    validation_macro_f1: Optional[float] = None
    validation_accuracy: Optional[float] = None
    validation_far: Optional[float] = None
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_candidate_set_hash(
    frozen_candidates: Dict[str, CandidateModelFreezeRecord],
) -> str:
    """Compute deterministic cryptographic hash of the complete candidate set.

    Guarantees that any modification to candidate algorithms, checksums, primary designation,
    or selection criterion produces a distinct hash.
    """
    items = []
    for k in sorted(frozen_candidates.keys()):
        rec = frozen_candidates[k]
        items.append(f"{k}:{rec.model_checksum}:{rec.selected_as_primary}:{rec.selection_criterion}")
    canonical_repr = "|".join(items)
    import hashlib
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


@dataclass
class EvaluationRunManifest:
    """Consolidated master manifest for an evaluation session."""

    evaluation_run_id: str
    engineering_run_id: str
    training_session_id: Optional[str]
    status: str  # "Completed", "Completed with Warnings", "Failed"
    created_at_utc: str
    completed_at_utc: Optional[str] = None
    evaluated_algorithms: List[str] = field(default_factory=list)
    partitions_evaluated: List[str] = field(default_factory=list)
    dataset_checksums: Dict[str, str] = field(default_factory=dict)
    model_checksums: Dict[str, str] = field(default_factory=dict)
    validation_results: Dict[str, AlgorithmEvaluationResult] = field(default_factory=dict)
    frozen_candidates: Dict[str, CandidateModelFreezeRecord] = field(default_factory=dict)
    test_gate_confirmed: bool = False
    test_gate_confirmed_at_utc: Optional[str] = None
    test_gate_confirmation_notes: Optional[str] = None
    test_holdout_locked: bool = False
    test_holdout_locked_at_utc: Optional[str] = None
    candidate_set_hash: Optional[str] = None
    test_results: Dict[str, AlgorithmEvaluationResult] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    is_test_fixture: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "engineering_run_id": self.engineering_run_id,
            "training_session_id": self.training_session_id,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "evaluated_algorithms": self.evaluated_algorithms,
            "partitions_evaluated": self.partitions_evaluated,
            "dataset_checksums": self.dataset_checksums,
            "model_checksums": self.model_checksums,
            "validation_results": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.validation_results.items()
            },
            "frozen_candidates": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.frozen_candidates.items()
            },
            "test_gate_confirmed": self.test_gate_confirmed,
            "test_gate_confirmed_at_utc": self.test_gate_confirmed_at_utc,
            "test_gate_confirmation_notes": self.test_gate_confirmation_notes,
            "test_holdout_locked": self.test_holdout_locked,
            "test_holdout_locked_at_utc": self.test_holdout_locked_at_utc,
            "candidate_set_hash": self.candidate_set_hash,
            "test_results": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.test_results.items()
            },
            "environment": self.environment,
            "is_test_fixture": self.is_test_fixture,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRunManifest":
        """Reconstruct from dictionary."""
        d = dict(data)
        if isinstance(d.get("validation_results"), dict):
            d["validation_results"] = {
                k: AlgorithmEvaluationResult.from_dict(v) if isinstance(v, dict) else v
                for k, v in d["validation_results"].items()
            }
        if isinstance(d.get("test_results"), dict):
            d["test_results"] = {
                k: AlgorithmEvaluationResult.from_dict(v) if isinstance(v, dict) else v
                for k, v in d["test_results"].items()
            }
        if isinstance(d.get("frozen_candidates"), dict):
            d["frozen_candidates"] = {
                k: CandidateModelFreezeRecord(**v) if isinstance(v, dict) else v
                for k, v in d["frozen_candidates"].items()
            }
        return cls(**d)


def capture_evaluation_environment() -> Dict[str, str]:
    """Capture deterministic runtime environment versions for MLOps reproducibility."""
    return {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
