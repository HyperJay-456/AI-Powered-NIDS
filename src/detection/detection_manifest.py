"""Detection Manifest & Provenance Schema (CP-08).

Defines the structured audit manifest recorded for every operational intrusion detection
inference run. Tracks exact provenance back to CP-04 ingestion records, CP-05 preprocessing
pipelines, CP-06 trained estimators, and CP-07 candidate freeze evaluations.
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
class ModelInferenceMetadata:
    """Individual model inference execution metadata."""

    algorithm: str  # "svm", "mlp", "decision_tree"
    algorithm_display_name: str
    training_run_id: str
    model_checksum: str
    engineering_run_id: str
    is_primary_candidate: bool = False
    candidate_freeze_time_utc: Optional[str] = None
    candidate_selection_criterion: Optional[str] = None
    status: str = "Completed"  # "Completed", "Failed"
    predicted_rows: int = 0
    score_type: str = "decision_margin"  # "decision_margin" (SVM) or "uncalibrated_probabilities" (MLP/DT)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInferenceMetadata":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class DisagreementSummary:
    """Summary of cross-model prediction consensus and disagreement."""

    total_compared_rows: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    agreement_percentage: float = 0.0
    disagreement_percentage: float = 0.0
    incomplete_count: int = 0
    pairwise_agreements: Dict[str, float] = field(default_factory=dict)
    top_disagreements: List[Dict[str, Any]] = field(default_factory=list)
    disagreement_disclaimer: str = (
        "Model disagreement indicates prediction uncertainty and warrants forensic analyst triage; "
        "it is NOT an automatic confirmation of an intrusion."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisagreementSummary":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class RowAccountingRecord:
    """Rigorous accounting verifying zero silent row loss."""

    input_rows: int
    successfully_predicted_rows: int
    rejected_rows: int
    skipped_rows: int
    accounting_invariant_holds: bool = True

    def __post_init__(self) -> None:
        expected = self.successfully_predicted_rows + self.rejected_rows + self.skipped_rows
        self.accounting_invariant_holds = (self.input_rows == expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RowAccountingRecord":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class DetectionRunManifest:
    """Consolidated master manifest for an intrusion detection inference run."""

    detection_run_id: str
    created_at_utc: str
    completed_at_utc: str
    status: str  # "Completed", "Completed with Warnings", "Failed"
    source_dataset_id: str
    source_filename: str
    source_sha256: str
    source_validation_status: str
    source_overlaps_training: bool
    source_overlap_details: Optional[str]
    has_ground_truth_target: bool
    ground_truth_column_name: Optional[str]
    selected_algorithms: List[str]
    models: Dict[str, ModelInferenceMetadata]
    engineering_run_id: str
    feature_schema_fingerprint: str
    batch_size: int
    row_accounting: RowAccountingRecord
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    disagreement_summary: Optional[DisagreementSummary] = None
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    run_type: str = "production"  # "production" or "test_fixture"
    is_test_fixture: bool = False
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        d = asdict(self)
        d["models"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.models.items()}
        if self.disagreement_summary and hasattr(self.disagreement_summary, "to_dict"):
            d["disagreement_summary"] = self.disagreement_summary.to_dict()
        if self.row_accounting and hasattr(self.row_accounting, "to_dict"):
            d["row_accounting"] = self.row_accounting.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionRunManifest":
        """Reconstruct manifest from dictionary representation."""
        d = dict(data)
        if isinstance(d.get("models"), dict):
            d["models"] = {
                k: ModelInferenceMetadata.from_dict(v) if isinstance(v, dict) else v
                for k, v in d["models"].items()
            }
        if isinstance(d.get("disagreement_summary"), dict):
            d["disagreement_summary"] = DisagreementSummary.from_dict(d["disagreement_summary"])
        if isinstance(d.get("row_accounting"), dict):
            d["row_accounting"] = RowAccountingRecord.from_dict(d["row_accounting"])
        return cls(**d)


def capture_detection_environment() -> Dict[str, str]:
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
