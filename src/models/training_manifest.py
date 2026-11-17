"""Training Manifest & Provenance Schema (CP-06).

Defines the structured audit manifest recorded for every ML training execution.
Tracks exact provenance back to CP-05 engineering artifacts, algorithm configurations,
solver convergence states, and cryptographic model checksums.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import platform
from typing import Any, Dict, List, Optional
import sklearn
import scipy
import numpy

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlgorithmTrainingManifest:
    """Individual algorithm training execution manifest."""

    training_run_id: str
    algorithm: str  # "svm", "mlp", "decision_tree"
    algorithm_display_name: str
    estimator_class: str
    status: str  # "Completed", "Completed with Warnings", "Failed"
    engineering_run_id: str
    source_dataset_names: List[str]
    source_checksums: Dict[str, str]
    training_partition_checksum: str
    feature_schema_checksum: str
    label_mapping_checksum: str
    training_samples: int
    feature_count: int
    class_count: int
    classes: List[str]
    input_representation: str  # "Scaled Sparse CSR" or "Unscaled Sparse CSR"
    random_seed: int
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    started_at_utc: str
    completed_at_utc: str
    convergence_status: str  # "Converged", "Iteration Limit Reached", "Failed"
    solver_iterations: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    artifact_checksum: Optional[str] = None
    artifact_path: Optional[str] = None
    model_size_bytes: int = 0
    preprocessing_artifact_references: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmTrainingManifest":
        """Reconstruct manifest from dictionary representation."""
        return cls(**data)


@dataclass
class MasterTrainingRunManifest:
    """Consolidated manifest for a training run session (single or multi-model)."""

    session_run_id: str
    engineering_run_id: str
    started_at_utc: str
    completed_at_utc: str
    overall_status: str  # "Completed", "Completed with Warnings", "Partial Success", "Failed"
    trained_algorithms: List[str]
    model_manifests: Dict[str, AlgorithmTrainingManifest]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        data["model_manifests"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.model_manifests.items()}
        return data


def capture_training_environment() -> Dict[str, str]:
    """Capture deterministic runtime environment versions for MLOps reproducibility."""
    return {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": numpy.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
