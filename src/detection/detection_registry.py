"""Detection Registry & History Catalog (CP-08).

Provides discovery, cryptographic integrity verification, and retrieval for completed
intrusion detection inference runs stored in data/analyses/.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from config.settings import settings
from src.detection.detection_manifest import DetectionRunManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionRegistry:
    """Manages discovery and verification of historical intrusion detection runs."""

    def __init__(self, analyses_dir: Optional[Path] = None):
        """Initialize registry with analyses directory."""
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.analyses_dir.mkdir(parents=True, exist_ok=True)

    def list_detection_runs(
        self,
        include_test_fixtures: bool = False,
        verify_checksums: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all valid persisted detection runs in reverse chronological order."""
        if not self.analyses_dir.exists():
            return []

        runs: List[Dict[str, Any]] = []

        for run_dir in sorted(self.analyses_dir.glob("detect_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue

            manifest_file = run_dir / "run_manifest.json"
            if not manifest_file.exists():
                continue

            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter test fixtures
                if not include_test_fixtures:
                    if data.get("is_test_fixture", False) or data.get("run_type") in ("test", "fixture", "test_fixture"):
                        continue

                # Checksum verification
                if verify_checksums:
                    artifacts = data.get("output_artifacts", {})
                    pred_sha = artifacts.get("predictions.csv")
                    pred_file = run_dir / "predictions.csv"
                    if pred_sha and pred_file.exists():
                        with open(pred_file, "rb") as pf:
                            actual_sha = hashlib.sha256(pf.read()).hexdigest()
                        if actual_sha != pred_sha:
                            logger.warning(f"Detection run '{run_dir.name}' has corrupted predictions.csv checksum.")
                            continue

                runs.append({
                    "detection_run_id": data.get("detection_run_id", run_dir.name),
                    "created_at_utc": data.get("created_at_utc", ""),
                    "completed_at_utc": data.get("completed_at_utc", ""),
                    "status": data.get("status", "Completed"),
                    "source_dataset_id": data.get("source_dataset_id", "unknown"),
                    "source_filename": data.get("source_filename", "unknown"),
                    "selected_algorithms": data.get("selected_algorithms", []),
                    "engineering_run_id": data.get("engineering_run_id", "unknown"),
                    "input_rows": data.get("row_accounting", {}).get("input_rows", 0),
                    "predicted_rows": data.get("row_accounting", {}).get("successfully_predicted_rows", 0),
                    "has_ground_truth": data.get("has_ground_truth_target", False),
                    "is_test_fixture": data.get("is_test_fixture", False),
                    "path": str(run_dir),
                })
            except Exception as e:
                logger.debug(f"Failed to read detection manifest in {run_dir}: {e}")

        return runs

    def get_latest_run(self, include_test_fixtures: bool = False) -> Optional[Dict[str, Any]]:
        """Get summary of the latest completed detection run."""
        runs = self.list_detection_runs(include_test_fixtures=include_test_fixtures, verify_checksums=False)
        return runs[0] if runs else None

    def get_detection_manifest(self, detection_run_id: str) -> Optional[DetectionRunManifest]:
        """Load and reconstruct DetectionRunManifest for a specific run."""
        manifest_file = self.analyses_dir / detection_run_id / "run_manifest.json"
        if not manifest_file.exists():
            return None
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DetectionRunManifest.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load detection manifest for '{detection_run_id}': {e}")
            return None

    def load_predictions_df(self, detection_run_id: str) -> Optional[pd.DataFrame]:
        """Load predictions.csv safely with path traversal protection."""
        file_path = self.analyses_dir / detection_run_id / "predictions.csv"
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(self.analyses_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{file_path}'")
        if not file_path.exists():
            return None
        return pd.read_csv(file_path)

    def load_comparison_df(self, detection_run_id: str) -> Optional[pd.DataFrame]:
        """Load model_comparison.csv safely if present."""
        file_path = self.analyses_dir / detection_run_id / "model_comparison.csv"
        if not file_path.exists():
            return None
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(self.analyses_dir.resolve())):
            raise PermissionError(f"Security violation: path traversal detected for '{file_path}'")
        return pd.read_csv(file_path)
