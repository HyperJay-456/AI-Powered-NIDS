"""CP-05 Artifact Loader & Verification Engine (CP-06).

Discovers completed CP-05 engineering runs, verifies artifact integrity, and enforces the
strict anti-leakage rule: ONLY the training partition (X_train, y_train) is loaded.
Validation and test partitions are NEVER touched during model training.
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
class EngineeringRunIntegrityCheck:
    """Outcome of pre-training artifact integrity verification."""

    is_eligible: bool
    run_id: str
    rejection_reasons: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    total_raw_rows: int = 0
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    num_features: int = 0
    num_classes: int = 0
    classes: List[str] = field(default_factory=list)
    source_filenames: List[str] = field(default_factory=list)
    source_checksums: Dict[str, str] = field(default_factory=dict)
    has_scaled_representation: bool = False
    has_unscaled_representation: bool = False
    manifest_path: Optional[str] = None


@dataclass
class LoadedTrainingData:
    """Safely loaded training partition and associated provenance metadata."""

    X: sparse.csr_matrix
    y: np.ndarray
    algorithm: str
    representation_type: str  # "scaled" or "unscaled"
    num_samples: int
    num_features: int
    num_classes: int
    classes: List[str]
    feature_names: List[str]
    training_partition_checksum: str
    feature_schema_checksum: str
    label_mapping_checksum: str
    engineering_run_id: str
    source_filenames: List[str]
    source_checksums: Dict[str, str]


class CP05ArtifactLoader:
    """Enforces zero-leakage training partition loading and verification."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        """Initialize loader with directory paths."""
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.raw_dir = raw_dir or settings.raw_data_dir

    def list_eligible_engineering_runs(self) -> List[Dict[str, Any]]:
        """Discover and list all eligible completed CP-05 engineering runs."""
        if not self.processed_dir.exists():
            return []

        runs: List[Dict[str, Any]] = []
        for run_path in sorted(self.processed_dir.glob("eng_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_path.is_dir():
                continue

            check = self.verify_engineering_run_integrity(run_path.name)
            if check.is_eligible:
                runs.append({
                    "run_id": check.run_id,
                    "train_rows": check.train_rows,
                    "num_features": check.num_features,
                    "num_classes": check.num_classes,
                    "classes": check.classes,
                    "source_filenames": check.source_filenames,
                    "status": "Verified Eligible",
                    "path": str(run_path),
                })
        return runs

    def verify_engineering_run_integrity(self, run_id: str) -> EngineeringRunIntegrityCheck:
        """Verify the complete structural integrity and eligibility of a CP-05 run.

        Args:
            run_id: Unique directory name of the engineering run.

        Returns:
            EngineeringRunIntegrityCheck with validation results and rejection reasons if any.
        """
        run_dir = self.processed_dir / run_id
        reasons: List[str] = []
        warnings: List[str] = []

        if not run_dir.exists() or not run_dir.is_dir():
            return EngineeringRunIntegrityCheck(
                is_eligible=False,
                run_id=run_id,
                rejection_reasons=[f"Engineering run directory does not exist: {run_dir}"],
            )

        # 1. Verify engineering manifest
        manifest_file = run_dir / "metadata" / "engineering_manifest.json"
        if not manifest_file.exists():
            # Check alternative location
            alt_manifest = run_dir / "engineering_manifest.json"
            if alt_manifest.exists():
                manifest_file = alt_manifest
            else:
                reasons.append("Missing 'engineering_manifest.json' in run metadata.")

        manifest_data: Dict[str, Any] = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("status") != "Completed":
                    reasons.append(f"Engineering run status is '{manifest_data.get('status')}', must be 'Completed'.")
            except Exception as e:
                reasons.append(f"Failed to parse engineering manifest: {e}")

        # 2. Verify Training Partition Files
        train_dir = run_dir / "datasets" / "train"
        scaled_file = train_dir / "X_scaled.npz"
        unscaled_file = train_dir / "X_unscaled.npz"
        y_file = train_dir / "y.npy"

        has_scaled = scaled_file.exists() and scaled_file.stat().st_size > 0
        has_unscaled = unscaled_file.exists() and unscaled_file.stat().st_size > 0
        has_y = y_file.exists() and y_file.stat().st_size > 0

        if not has_scaled:
            reasons.append("Missing or empty scaled training feature matrix ('datasets/train/X_scaled.npz').")
        if not has_unscaled:
            reasons.append("Missing or empty unscaled training feature matrix ('datasets/train/X_unscaled.npz').")
        if not has_y:
            reasons.append("Missing or empty training target label array ('datasets/train/y.npy').")

        # 3. Check for at least 2 distinct classes in training partition
        num_classes = manifest_data.get("num_classes", 0)
        classes = manifest_data.get("classes", [])
        if has_y:
            try:
                y_arr = np.load(y_file, mmap_mode="r")
                unique_y = np.unique(y_arr)
                if len(unique_y) < 2:
                    reasons.append(
                        f"Degenerate training partition: Contains only {len(unique_y)} distinct class ({unique_y}). "
                        "Supervised ML training requires at least 2 distinct classes."
                    )
            except Exception as e:
                reasons.append(f"Failed to read training target array 'y.npy': {e}")

        # 4. Verify feature schema and label mapping
        meta_dir = run_dir / "metadata"
        schema_file = meta_dir / "feature_schema.json"
        if not schema_file.exists():
            # Check consolidated metadata.json
            master_meta = meta_dir / "metadata.json"
            if not master_meta.exists():
                reasons.append("Missing feature schema metadata ('feature_schema.json' or 'metadata.json').")

        label_map_file = meta_dir / "label_mapping.json"
        if not label_map_file.exists():
            master_meta = meta_dir / "metadata.json"
            if not master_meta.exists():
                reasons.append("Missing label mapping metadata ('label_mapping.json' or 'metadata.json').")

        # 5. Check preprocessor bundle
        preproc_dir = run_dir / "preprocessors"
        if not preproc_dir.exists() or not (preproc_dir / "label_processor.joblib").exists():
            warnings.append("Label processor artifact 'label_processor.joblib' missing from preprocessors/.")

        # Source files
        source_filenames = manifest_data.get("source_filenames", [])
        source_checksums = manifest_data.get("source_checksums", {})

        is_eligible = len(reasons) == 0

        return EngineeringRunIntegrityCheck(
            is_eligible=is_eligible,
            run_id=run_id,
            rejection_reasons=reasons,
            warning_messages=warnings,
            total_raw_rows=manifest_data.get("total_raw_rows", 0),
            train_rows=manifest_data.get("train_rows", 0),
            val_rows=manifest_data.get("val_rows", 0),
            test_rows=manifest_data.get("test_rows", 0),
            num_features=manifest_data.get("num_features_out", 0),
            num_classes=num_classes,
            classes=classes,
            source_filenames=source_filenames,
            source_checksums=source_checksums,
            has_scaled_representation=has_scaled,
            has_unscaled_representation=has_unscaled,
            manifest_path=str(manifest_file) if manifest_file.exists() else None,
        )

    def load_training_partition_only(
        self,
        run_id: str,
        algorithm: str,
    ) -> LoadedTrainingData:
        """STRICT ZERO-LEAKAGE: Load EXCLUSIVELY the training partition for model fitting.

        Validation and test sets remain strictly untouched on disk.

        Args:
            run_id: CP-05 engineering run ID.
            algorithm: Target algorithm ('svm', 'mlp', 'decision_tree').

        Returns:
            LoadedTrainingData containing training matrix, targets, and provenance hashes.
        """
        check = self.verify_engineering_run_integrity(run_id)
        if not check.is_eligible:
            raise ValueError(
                f"Engineering run '{run_id}' failed integrity verification: {'; '.join(check.rejection_reasons)}"
            )

        run_dir = self.processed_dir / run_id
        train_dir = run_dir / "datasets" / "train"

        # 1. Select the algorithm-appropriate representation
        algo = algorithm.lower().strip()
        if algo in ("svm", "mlp"):
            matrix_file = train_dir / "X_scaled.npz"
            representation = "scaled"
        elif algo == "decision_tree":
            matrix_file = train_dir / "X_unscaled.npz"
            representation = "unscaled"
        else:
            raise ValueError(f"Unauthorized algorithm '{algorithm}'. Must be 'svm', 'mlp', or 'decision_tree'.")

        if not matrix_file.exists():
            raise FileNotFoundError(f"Required training matrix file not found: {matrix_file}")

        y_file = train_dir / "y.npy"
        if not y_file.exists():
            raise FileNotFoundError(f"Training target file not found: {y_file}")

        # 2. Compute cryptographic hashes of training partition
        with open(matrix_file, "rb") as f:
            matrix_hash = hashlib.sha256(f.read()).hexdigest()

        # 3. Load sparse training matrix and targets
        X_train = sparse.load_npz(matrix_file)
        if not isinstance(X_train, sparse.csr_matrix):
            X_train = X_train.tocsr()

        y_train = np.load(y_file)

        # 4. Load metadata for provenance
        meta_dir = run_dir / "metadata"
        schema_file = meta_dir / "feature_schema.json"
        schema_hash = ""
        feature_names: List[str] = []

        if schema_file.exists():
            with open(schema_file, "rb") as f:
                schema_hash = hashlib.sha256(f.read()).hexdigest()
            with open(schema_file, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                feature_names = sdata.get("output_features", [])
        else:
            # Fallback to master metadata.json
            master_meta = meta_dir / "metadata.json"
            if master_meta.exists():
                with open(master_meta, "rb") as f:
                    schema_hash = hashlib.sha256(f.read()).hexdigest()
                with open(master_meta, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                    feature_names = mdata.get("feature_schema", {}).get("output_features", [])

        label_map_file = meta_dir / "label_mapping.json"
        label_hash = ""
        classes: List[str] = check.classes

        if label_map_file.exists():
            with open(label_map_file, "rb") as f:
                label_hash = hashlib.sha256(f.read()).hexdigest()
            with open(label_map_file, "r", encoding="utf-8") as f:
                ldata = json.load(f)
                if not classes:
                    classes = ldata.get("classes", [])

        logger.info(
            f"Successfully loaded CP-05 training partition for '{algo}' from '{run_id}': "
            f"{X_train.shape[0]:,} samples, {X_train.shape[1]:,} features, representation='{representation}'."
        )

        return LoadedTrainingData(
            X=X_train,
            y=y_train,
            algorithm=algo,
            representation_type=representation,
            num_samples=X_train.shape[0],
            num_features=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            classes=classes,
            feature_names=feature_names,
            training_partition_checksum=matrix_hash,
            feature_schema_checksum=schema_hash,
            label_mapping_checksum=label_hash,
            engineering_run_id=run_id,
            source_filenames=check.source_filenames,
            source_checksums=check.source_checksums,
        )
