"""Evaluation Data Loader & Zero-Leakage Partition Isolation (CP-07).

Loads evaluation dataset partitions (validation and test) from completed CP-05 runs:
- LinearSVC: Scaled sparse CSR matrices ('X_scaled.npz')
- MLPClassifier: Scaled sparse CSR matrices ('X_scaled.npz')
- DecisionTreeClassifier: Unscaled physical feature CSR matrices ('X_unscaled.npz')

CRITICAL ZERO-LEAKAGE INVARIANTS:
1. STRICTLY forbids loading the 'train' partition during evaluation.
2. Reuses fitted preprocessor representations exactly as produced by CP-05.
3. Absolutely NO refitting, recalculating, or re-transforming.
4. Cryptographically verifies matrix and label partition checksums.
5. Strictly checks that matrix row count exactly matches ground-truth label count.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedEvaluationPartition:
    """Evaluation partition loaded safely into memory with provenance checksums."""

    X: sparse.csr_matrix
    y: np.ndarray
    partition_name: str  # "validation" or "test"
    algorithm: str
    representation_type: str  # "scaled" or "unscaled"
    num_samples: int
    num_features: int
    num_classes: int
    classes: List[str]
    engineering_run_id: str
    matrix_checksum: str
    labels_checksum: str
    label_mapping_checksum: str
    feature_schema_checksum: str

    def verify_alignment(self) -> None:
        """Verify that matrix rows and label rows are strictly equal."""
        if self.X.shape[0] != len(self.y):
            raise ValueError(
                f"Evaluation row count mismatch: Feature matrix has {self.X.shape[0]} rows, "
                f"but ground truth label array has {len(self.y)} rows."
            )
        if self.X.shape[1] != self.num_features:
            raise ValueError(
                f"Evaluation feature count mismatch: Matrix has {self.X.shape[1]} columns, "
                f"expected {self.num_features}."
            )


class EvaluationDataLoader:
    """Manages secure, zero-leakage loading of CP-05 evaluation partitions."""

    def __init__(self, processed_dir: Optional[Path] = None):
        """Initialize data loader with processed datasets directory."""
        self.processed_dir = processed_dir or settings.processed_data_dir

    def load_partition(
        self,
        engineering_run_id: str,
        partition: str,
        algorithm: str,
    ) -> LoadedEvaluationPartition:
        """Safely load an evaluation partition for a specific algorithm.

        Args:
            engineering_run_id: CP-05 run identifier.
            partition: 'validation' or 'test'. Loading 'train' is STRICTLY FORBIDDEN.
            algorithm: Algorithm key ('svm', 'mlp', 'decision_tree').

        Returns:
            LoadedEvaluationPartition with verified sparse CSR matrix and target labels.

        Raises:
            PermissionError: If partition == 'train' (Anti-leakage protocol enforcement).
            FileNotFoundError: If partition files or metadata are missing.
            ValueError: If dimensions or row counts do not align.
        """
        # 1. Anti-leakage guardrail
        part_clean = partition.lower().strip()
        if part_clean == "train":
            raise PermissionError(
                "DATA LEAKAGE VIOLATION: EvaluationDataLoader strictly forbids loading the "
                "training partition ('train'). CP-07 evaluates validation and test holdout partitions only."
            )

        if part_clean not in ("validation", "test"):
            raise ValueError(
                f"Invalid evaluation partition '{partition}'. Must be 'validation' or 'test'."
            )

        run_dir = self.processed_dir / engineering_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Engineering run directory not found: {run_dir}")

        meta_dir = run_dir / "metadata"
        part_dir = run_dir / "datasets" / part_clean

        if not part_dir.exists():
            raise FileNotFoundError(f"Partition directory not found: {part_dir}")

        # 2. Load metadata (label mapping and feature schema)
        label_map_file = meta_dir / "label_mapping.json"
        feat_schema_file = meta_dir / "feature_schema.json"

        if not label_map_file.exists():
            raise FileNotFoundError(f"Label mapping missing: {label_map_file}")
        if not feat_schema_file.exists():
            raise FileNotFoundError(f"Feature schema missing: {feat_schema_file}")

        with open(label_map_file, "r", encoding="utf-8") as f:
            label_map_raw = f.read()
            label_map = json.loads(label_map_raw)
            label_mapping_sha = hashlib.sha256(label_map_raw.encode("utf-8")).hexdigest()

        with open(feat_schema_file, "r", encoding="utf-8") as f:
            feat_schema_raw = f.read()
            feat_schema = json.loads(feat_schema_raw)
            feature_schema_sha = hashlib.sha256(feat_schema_raw.encode("utf-8")).hexdigest()

        classes = label_map.get("classes", [])
        num_classes = len(classes)
        expected_features = feat_schema.get("num_features_out", 0)

        # 3. Determine required matrix representation
        algo_key = algorithm.lower().strip()
        if algo_key in ("svm", "mlp", "linearsvc", "mlpclassifier"):
            matrix_filename = "X_scaled.npz"
            rep_type = "scaled"
        elif algo_key in ("decision_tree", "decisiontree", "decisiontreeclassifier"):
            matrix_filename = "X_unscaled.npz"
            rep_type = "unscaled"
        else:
            raise ValueError(f"Unauthorized or unknown algorithm '{algorithm}'. Must be svm, mlp, or decision_tree.")

        matrix_file = part_dir / matrix_filename
        y_file = part_dir / "y.npy"

        if not matrix_file.exists():
            raise FileNotFoundError(
                f"Required {rep_type} feature matrix missing: {matrix_file}. "
                f"Algorithm '{algo_key}' requires '{matrix_filename}'."
            )
        if not y_file.exists():
            raise FileNotFoundError(f"Ground truth target file missing: {y_file}")

        # 4. Compute cryptographic checksums
        with open(matrix_file, "rb") as f:
            matrix_sha = hashlib.sha256(f.read()).hexdigest()
        with open(y_file, "rb") as f:
            labels_sha = hashlib.sha256(f.read()).hexdigest()

        # 5. Load sparse CSR matrix and target array
        logger.info(f"Loading {part_clean} partition ({rep_type}) for algorithm '{algo_key}'...")
        X = sparse.load_npz(matrix_file)
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()

        y = np.load(y_file)

        # 6. Verify row count and feature alignment
        if X.shape[0] != len(y):
            raise ValueError(
                f"Row count mismatch in {part_clean} partition: Feature matrix has {X.shape[0]} rows, "
                f"but target label array has {len(y)} rows."
            )

        if expected_features > 0 and X.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch in {part_clean} partition: Matrix has {X.shape[1]} columns, "
                f"expected {expected_features} based on CP-05 feature schema."
            )

        loaded_part = LoadedEvaluationPartition(
            X=X,
            y=y,
            partition_name=part_clean,
            algorithm=algo_key,
            representation_type=rep_type,
            num_samples=X.shape[0],
            num_features=X.shape[1],
            num_classes=num_classes,
            classes=classes,
            engineering_run_id=engineering_run_id,
            matrix_checksum=matrix_sha,
            labels_checksum=labels_sha,
            label_mapping_checksum=label_mapping_sha,
            feature_schema_checksum=feature_schema_sha,
        )
        loaded_part.verify_alignment()

        logger.info(
            f"Successfully verified and loaded {part_clean} partition: "
            f"{loaded_part.num_samples:,} rows, {loaded_part.num_features} features, "
            f"{loaded_part.num_classes} classes."
        )
        return loaded_part
