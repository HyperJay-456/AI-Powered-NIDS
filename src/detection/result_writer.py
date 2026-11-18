"""Result Writer & Atomic Persistence Engine (CP-08).

Safely writes intrusion detection inference artifacts and manifests to disk in data/analyses/<run_id>/:
  - predictions.csv: per-row predictions and model-specific scores
  - model_comparison.csv: multi-model comparative records
  - run_manifest.json: auditable provenance and accounting manifest

STRICT INVARIANTS:
  1. Atomic directory commits via .tmp staging directory
  2. Complete row accounting invariant (Input = Predicted + Rejected + Skipped)
  3. Never overwrites prior analysis runs
  4. Cryptographic SHA-256 verification of all output files
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from config.settings import settings
from src.detection.detection_manifest import DetectionRunManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionResultWriter:
    """Manages atomic output storage, checksum calculation, and row accounting validation."""

    def __init__(self, analyses_dir: Optional[Path] = None):
        """Initialize writer with target analyses directory."""
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.analyses_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_detection_run_id() -> str:
        """Generate a deterministic, timestamped collision-resistant run ID."""
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:6]
        return f"detect_run_{now_str}_{token}"

    def write_results(
        self,
        manifest: DetectionRunManifest,
        predictions_df: pd.DataFrame,
        comparison_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Atomically persist prediction artifacts and master manifest.

        Args:
            manifest: DetectionRunManifest containing provenance and accounting.
            predictions_df: DataFrame of per-row predictions and scores.
            comparison_df: Optional multi-model side-by-side comparison DataFrame.

        Returns:
            Path to the committed output directory.

        Raises:
            ValueError: If row accounting invariant fails.
            FileExistsError: If target run directory already exists.
        """
        # Validate Row Accounting Invariant
        accounting = manifest.row_accounting
        if not accounting.accounting_invariant_holds:
            raise ValueError(
                f"ROW ACCOUNTING INVARIANT VIOLATION: Input rows ({accounting.input_rows}) != "
                f"Predicted ({accounting.successfully_predicted_rows}) + "
                f"Rejected ({accounting.rejected_rows}) + "
                f"Skipped ({accounting.skipped_rows})."
            )

        if len(predictions_df) != accounting.successfully_predicted_rows:
            raise ValueError(
                f"ROW ACCOUNTING MISMATCH: Output DataFrame length ({len(predictions_df)}) "
                f"does not match successfully predicted count ({accounting.successfully_predicted_rows})."
            )

        run_id = manifest.detection_run_id
        final_dir = self.analyses_dir / run_id
        if final_dir.exists():
            raise FileExistsError(f"Detection run directory already exists: {final_dir}")

        temp_dir = self.analyses_dir / f".tmp_{run_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_artifacts: Dict[str, str] = {}

            # 1. Write predictions.csv
            pred_file = temp_dir / "predictions.csv"
            predictions_df.to_csv(pred_file, index=False)
            with open(pred_file, "rb") as f:
                output_artifacts["predictions.csv"] = hashlib.sha256(f.read()).hexdigest()

            # 2. Write model_comparison.csv if multiple models
            if comparison_df is not None and not comparison_df.empty:
                comp_file = temp_dir / "model_comparison.csv"
                comparison_df.to_csv(comp_file, index=False)
                with open(comp_file, "rb") as f:
                    output_artifacts["model_comparison.csv"] = hashlib.sha256(f.read()).hexdigest()

            # 3. Update and write run_manifest.json
            manifest.output_artifacts = output_artifacts
            manifest_file = temp_dir / "run_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # 4. Atomic directory rename commit
            os.replace(temp_dir, final_dir)
            logger.info(f"Successfully persisted detection run to {final_dir}")
            return final_dir

        except Exception as e:
            logger.error(f"Failed to persist detection run '{run_id}': {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
