"""Artifact Management & Versioned Preprocessing Persistence Module (CP-05).

Persists all engineered dataset partitions, sparse feature matrices, fitted
preprocessors, and auditable JSON metadata into versioned run directories:
  data/processed/<engineering_run_id>/
    ├── metadata/
    │   ├── engineering_manifest.json
    │   ├── source_files.json
    │   ├── feature_schema.json
    │   ├── label_mapping.json
    │   ├── preprocessing_config.json
    │   ├── row_accounting.json
    │   └── class_distribution.json
    ├── preprocessors/
    │   ├── scaled_preprocessor.joblib
    │   └── tree_preprocessor.joblib
    ├── datasets/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    └── reports/
        └── engineering_report.json

Enforces atomic writing, prevents overwriting past runs, and isolates failed runs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import uuid

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EngineeringRunRecord:
    """Metadata summary of a persisted dataset engineering run."""

    run_id: str
    created_at_utc: str
    status: str  # "Completed", "Failed", "In Progress"
    source_filenames: List[str]
    source_checksums: Dict[str, str]
    total_raw_rows: int
    supervised_rows: int
    unlabeled_rows: int
    quarantined_rows: int
    excluded_duplicate_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    num_features_in: int
    num_features_out: int
    num_classes: int
    classes: List[str]
    random_seed: int
    split_strategy: str
    imputation_strategy: str
    scaling_strategy: str
    duplicate_policy: str
    artifact_directory: str
    error_message: Optional[str] = None


class ArtifactManager:
    """Manages versioned storage and loading of dataset engineering artifacts."""

    def __init__(self, processed_dir: Optional[Path] = None):
        """Initialize artifact manager with base output directory."""
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_run_id() -> str:
        """Generate a timestamped, collision-resistant engineering run identifier."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_suffix = uuid.uuid4().hex[:6]
        return f"eng_run_{now_str}_{rand_suffix}"

    def list_runs(self) -> List[EngineeringRunRecord]:
        """List all completed engineering runs sorted by creation timestamp descending."""
        runs: List[EngineeringRunRecord] = []
        if not self.processed_dir.exists():
            return runs

        for entry in self.processed_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                manifest_file = entry / "metadata" / "engineering_manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            runs.append(EngineeringRunRecord(**data))
                    except Exception as e:
                        logger.warning(f"Could not parse run manifest at {manifest_file}: {e}")

        # Sort newest first
        runs.sort(key=lambda r: r.created_at_utc, reverse=True)
        return runs

    def get_run(self, run_id: str) -> Optional[EngineeringRunRecord]:
        """Retrieve run metadata record by identifier."""
        manifest_file = self.processed_dir / run_id / "metadata" / "engineering_manifest.json"
        if not manifest_file.exists():
            return None
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return EngineeringRunRecord(**data)
        except Exception as e:
            logger.error(f"Failed to load run {run_id}: {e}")
            return None

    def save_run_artifacts(
        self,
        run_record: EngineeringRunRecord,
        metadata_dict: Dict[str, Any],
        preprocessor_bundle: Dict[str, Any],
        datasets: Dict[str, Dict[str, Any]],
        report_dict: Dict[str, Any],
    ) -> Path:
        """Atomically persist an entire dataset engineering run to disk.

        Args:
            run_record: EngineeringRunRecord summary.
            metadata_dict: Mapping of metadata filenames to JSON-serializable dictionaries.
            preprocessor_bundle: Mapping of preprocessor names to serializable objects.
            datasets: Dict mapping 'train', 'validation', 'test' to matrices and arrays.
            report_dict: Full engineering report dictionary.

        Returns:
            Path to the committed versioned run directory.
        """
        final_dir = self.processed_dir / run_record.run_id
        if final_dir.exists():
            raise FileExistsError(f"Engineering run directory '{run_record.run_id}' already exists!")

        temp_dir = self.processed_dir / f".tmp_{run_record.run_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            # Create directory tree
            meta_dir = temp_dir / "metadata"
            prep_dir = temp_dir / "preprocessors"
            data_dir = temp_dir / "datasets"
            rep_dir = temp_dir / "reports"

            for d in (meta_dir, prep_dir, data_dir, rep_dir):
                d.mkdir(parents=True, exist_ok=True)

            # 1. Save metadata JSONs
            for filename, content in metadata_dict.items():
                target_json = meta_dir / f"{filename}.json"
                with open(target_json, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, default=str)

            # Master consolidated metadata.json
            with open(meta_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, default=str)

            # Save engineering manifest
            manifest_json = meta_dir / "engineering_manifest.json"
            with open(manifest_json, "w", encoding="utf-8") as f:
                json.dump(asdict(run_record), f, indent=2, default=str)

            # 2. Save preprocessors via joblib
            for prep_name, prep_obj in preprocessor_bundle.items():
                joblib_path = prep_dir / f"{prep_name}.joblib"
                joblib.dump(prep_obj, joblib_path, compress=3)

            # 3. Save datasets
            for split_name, split_data in datasets.items():
                split_dir = data_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                # Save Scaled representation
                if "X_scaled" in split_data and split_data["X_scaled"] is not None:
                    sparse.save_npz(split_dir / "X_scaled.npz", split_data["X_scaled"])

                # Save Unscaled representation (for Decision Tree)
                if "X_unscaled" in split_data and split_data["X_unscaled"] is not None:
                    sparse.save_npz(split_dir / "X_unscaled.npz", split_data["X_unscaled"])

                # Save targets
                if "y" in split_data and split_data["y"] is not None:
                    np.save(split_dir / "y.npy", split_data["y"])

                # Save provenance sidecar (JSON records)
                if "provenance" in split_data and split_data["provenance"] is not None:
                    prov_df: pd.DataFrame = split_data["provenance"]
                    prov_df.to_json(split_dir / "provenance.json", orient="records", date_format="iso")

            # 4. Save full engineering report
            with open(rep_dir / "engineering_report.json", "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, default=str)

            # 5. Commit atomically via rename
            os.replace(temp_dir, final_dir)
            logger.info(f"Atomically committed engineering run to {final_dir}")
            return final_dir

        except Exception as e:
            logger.error(f"Failed to persist engineering run {run_record.run_id}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
