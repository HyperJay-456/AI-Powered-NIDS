"""Detection Run Discovery & Eligibility Verifier for Prediction Explorer (CP-09).

Scans data/analyses/ for completed intrusion detection inference runs and applies
a rigorous 9-point verification check to ensure only genuine, uncorrupted runs
are made available for interactive exploration and analyst review.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings
from src.detection.detection_manifest import DetectionRunManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EligibleRunRecord:
    """Detailed record of a verified detection run available for exploration."""

    detection_run_id: str
    created_at_utc: str
    completed_at_utc: str
    status: str  # "Completed", "Completed with Warnings", "Failed", "Incomplete", "Corrupted", "Test Fixture"
    is_eligible: bool
    source_dataset_id: str
    source_filename: str
    source_sha256: str
    selected_algorithms: List[str]
    input_rows: int
    predicted_rows: int
    has_ground_truth: bool
    is_test_fixture: bool
    run_type: str
    manifest_path: str
    predictions_path: str
    comparison_path: Optional[str]
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    @property
    def display_label(self) -> str:
        """Formatted string for UI dropdown selection."""
        algos = ", ".join(self.selected_algorithms)
        status_suffix = " (Warnings)" if self.status == "Completed with Warnings" else ""
        return f"{self.detection_run_id} | {self.source_filename} | [{algos}] | {self.predicted_rows:,} rows{status_suffix}"


class ExplorerRegistry:
    """Manages discovery, verification, and retrieval of detection runs for CP-09."""

    def __init__(self, analyses_dir: Optional[Path] = None):
        """Initialize registry with analyses directory."""
        self.analyses_dir = analyses_dir or settings.analyses_dir
        self.analyses_dir.mkdir(parents=True, exist_ok=True)

    def list_all_runs(
        self,
        include_test_fixtures: bool = False,
    ) -> List[EligibleRunRecord]:
        """Discover and verify all detection runs in data/analyses/.

        Returns runs in reverse chronological order (newest first).
        """
        if not self.analyses_dir.exists():
            return []

        records: List[EligibleRunRecord] = []

        for run_dir in sorted(
            self.analyses_dir.glob("detect_run_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if not run_dir.is_dir():
                continue

            record = self.verify_run(run_dir, include_test_fixtures=include_test_fixtures)
            if record is not None:
                records.append(record)

        return records

    def list_eligible_runs(
        self,
        include_test_fixtures: bool = False,
    ) -> List[EligibleRunRecord]:
        """List only runs that pass all eligibility criteria for exploration."""
        all_runs = self.list_all_runs(include_test_fixtures=include_test_fixtures)
        return [r for r in all_runs if r.is_eligible]

    def get_run(
        self,
        detection_run_id: str,
        include_test_fixtures: bool = False,
    ) -> Optional[EligibleRunRecord]:
        """Fetch and verify a specific detection run by ID."""
        run_dir = self.analyses_dir / detection_run_id
        if not run_dir.exists() or not run_dir.is_dir():
            return None
        return self.verify_run(run_dir, include_test_fixtures=include_test_fixtures)

    def verify_run(
        self,
        run_dir: Path,
        include_test_fixtures: bool = False,
    ) -> Optional[EligibleRunRecord]:
        """Apply 9-point verification check on a detection run directory."""
        run_id = run_dir.name
        manifest_file = run_dir / "run_manifest.json"
        predictions_file = run_dir / "predictions.csv"
        comparison_file = run_dir / "model_comparison.csv"

        rejection_reasons: List[str] = []
        warnings: List[str] = []

        # 1. Directory exists
        if not run_dir.exists():
            return None

        # 2. Manifest exists and is readable JSON
        if not manifest_file.exists():
            return EligibleRunRecord(
                detection_run_id=run_id,
                created_at_utc="",
                completed_at_utc="",
                status="Incomplete",
                is_eligible=False,
                source_dataset_id="unknown",
                source_filename="unknown",
                source_sha256="",
                selected_algorithms=[],
                input_rows=0,
                predicted_rows=0,
                has_ground_truth=False,
                is_test_fixture=False,
                run_type="unknown",
                manifest_path=str(manifest_file),
                predictions_path=str(predictions_file),
                comparison_path=None,
                rejection_reasons=["Missing run_manifest.json"],
            )

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return EligibleRunRecord(
                detection_run_id=run_id,
                created_at_utc="",
                completed_at_utc="",
                status="Corrupted",
                is_eligible=False,
                source_dataset_id="unknown",
                source_filename="unknown",
                source_sha256="",
                selected_algorithms=[],
                input_rows=0,
                predicted_rows=0,
                has_ground_truth=False,
                is_test_fixture=False,
                run_type="unknown",
                manifest_path=str(manifest_file),
                predictions_path=str(predictions_file),
                comparison_path=None,
                rejection_reasons=[f"Corrupted run_manifest.json: {e}"],
            )

        # Extract manifest fields
        status = data.get("status", "Unknown")
        manifest_warnings = data.get("warnings", [])
        if isinstance(manifest_warnings, list):
            warnings.extend(manifest_warnings)
        is_fixture = data.get("is_test_fixture", False) or data.get("run_type") in ("test", "fixture", "test_fixture")
        run_type = data.get("run_type", "production")
        source_dataset_id = data.get("source_dataset_id", "unknown")
        source_filename = data.get("source_filename", "unknown")
        source_sha256 = data.get("source_sha256", "")
        selected_algorithms = data.get("selected_algorithms", [])
        models_meta = data.get("models", {})
        accounting = data.get("row_accounting", {})
        input_rows = accounting.get("input_rows", 0)
        predicted_rows = accounting.get("successfully_predicted_rows", 0)
        has_gt = data.get("has_ground_truth_target", False)
        output_artifacts = data.get("output_artifacts", {})

        # 3. Status is eligible ("Completed" or "Completed with Warnings")
        if status not in ("Completed", "Completed with Warnings"):
            rejection_reasons.append(f"Ineligible run status: '{status}'")

        # 4. Production vs Test Fixture isolation
        if is_fixture and not include_test_fixtures:
            return None  # Excluded from production view

        # 5. Prediction artifact exists
        if not predictions_file.exists():
            rejection_reasons.append("Missing predictions.csv artifact")

        # 6. Prediction artifact SHA-256 matches manifest
        expected_sha = output_artifacts.get("predictions.csv")
        if expected_sha and predictions_file.exists():
            try:
                with open(predictions_file, "rb") as pf:
                    actual_sha = hashlib.sha256(pf.read()).hexdigest()
                if actual_sha != expected_sha:
                    rejection_reasons.append(f"Checksum mismatch on predictions.csv: {actual_sha} != {expected_sha}")
            except Exception as e:
                rejection_reasons.append(f"Error reading predictions.csv: {e}")
        elif not expected_sha and predictions_file.exists():
            warnings.append("predictions.csv present but not recorded in manifest output_artifacts")

        # 7. Model metadata available
        if not models_meta:
            rejection_reasons.append("No model inference metadata recorded in manifest")

        # 8. Source dataset provenance available
        if not source_filename or source_filename == "unknown":
            warnings.append("Source dataset provenance incomplete in manifest")

        # 9. Artifact schema supported
        if predictions_file.exists() and not rejection_reasons:
            try:
                # Read header only
                header_cols = pd.read_csv(predictions_file, nrows=0).columns.tolist()
                if "source_row_index" not in header_cols:
                    rejection_reasons.append("predictions.csv lacks required 'source_row_index' column")
            except Exception as e:
                rejection_reasons.append(f"Failed to inspect schema of predictions.csv: {e}")

        # Final status calculation
        is_eligible = (len(rejection_reasons) == 0)
        final_status = status
        if not is_eligible:
            if "Checksum mismatch" in " ".join(rejection_reasons):
                final_status = "Corrupted"
            elif "Missing" in " ".join(rejection_reasons):
                final_status = "Incomplete"
            else:
                final_status = "Failed"
        elif is_fixture:
            final_status = "Test Fixture"

        return EligibleRunRecord(
            detection_run_id=run_id,
            created_at_utc=data.get("created_at_utc", ""),
            completed_at_utc=data.get("completed_at_utc", ""),
            status=final_status,
            is_eligible=is_eligible,
            source_dataset_id=source_dataset_id,
            source_filename=source_filename,
            source_sha256=source_sha256,
            selected_algorithms=selected_algorithms,
            input_rows=input_rows,
            predicted_rows=predicted_rows,
            has_ground_truth=has_gt,
            is_test_fixture=is_fixture,
            run_type=run_type,
            manifest_path=str(manifest_file),
            predictions_path=str(predictions_file),
            comparison_path=str(comparison_file) if comparison_file.exists() else None,
            warnings=warnings,
            rejection_reasons=rejection_reasons,
        )
