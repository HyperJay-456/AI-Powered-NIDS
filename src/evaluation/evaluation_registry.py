"""Evaluation Run Registry & Persistence Engine (CP-07).

Manages atomic disk persistence, versioning, discovery, and inspection for evaluation runs:
Structure:
  models/evaluations/<evaluation_run_id>/
    ├── evaluation_manifest.json
    ├── validation/
    │   ├── aggregate_metrics.json
    │   ├── per_class_metrics.json
    │   ├── confusion_matrices.json
    │   ├── roc_analysis.json
    │   ├── pr_analysis.json
    │   └── security_metrics.json
    └── test/  (optional, populated upon gated final holdout evaluation)
        ├── aggregate_metrics.json
        ├── per_class_metrics.json
        ├── confusion_matrices.json
        ├── roc_analysis.json
        ├── pr_analysis.json
        └── security_metrics.json

Features:
- Enforces atomic directory commits via temporary staging directories (.tmp_eval_run_*).
- Strictly isolates test-fixture evaluation runs from production evaluation histories.
- Never overwrites past evaluation runs; generates timestamped unique identifiers.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

from config.settings import settings
from src.evaluation.evaluation_manifest import (
    AlgorithmEvaluationResult,
    CandidateModelFreezeRecord,
    EvaluationRunManifest,
    capture_evaluation_environment,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationRegistry:
    """Manages persistent evaluation manifests and metric artifacts."""

    def __init__(self, evaluations_dir: Optional[Path] = None):
        """Initialize registry with evaluations directory."""
        self.eval_dir = evaluations_dir or settings.evaluations_dir
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_evaluation_run_id() -> str:
        """Generate a deterministic timestamped unique ID for an evaluation session."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_token = uuid.uuid4().hex[:6]
        return f"eval_run_{timestamp}_{rand_token}"

    def save_evaluation_run(
        self,
        manifest: EvaluationRunManifest,
    ) -> Path:
        """Atomically persist an entire evaluation manifest and sub-partition artifacts.

        Args:
            manifest: EvaluationRunManifest containing validation and optional test results.

        Returns:
            Path to the committed evaluation directory.
        """
        run_id = manifest.evaluation_run_id
        final_dir = self.eval_dir / run_id

        # If already exists (e.g. updating with test holdout evaluation), update safely
        temp_dir = self.eval_dir / f".tmp_{run_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        # If final_dir exists, copy current content to temp_dir first
        if final_dir.exists():
            shutil.copytree(final_dir, temp_dir)
        else:
            temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Write partition subdirectories and metric files
            for part_name, results_dict in (
                ("validation", manifest.validation_results),
                ("test", manifest.test_results),
            ):
                if not results_dict:
                    continue

                part_dir = temp_dir / part_name
                part_dir.mkdir(parents=True, exist_ok=True)

                # Aggregate metrics for each algorithm
                agg_data = {
                    algo: res.aggregate_metrics.to_dict()
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(agg_data, f, indent=2)

                # Per-class metrics
                per_class_data = {
                    algo: [m.to_dict() for m in res.per_class_metrics]
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(per_class_data, f, indent=2)

                # Confusion matrices
                cm_data = {
                    algo: res.confusion_matrix.to_dict()
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
                    json.dump(cm_data, f, indent=2)

                # ROC analysis
                roc_data = {
                    algo: {
                        "macro_auc": res.macro_roc_auc,
                        "curves": {k: v.to_dict() for k, v in res.roc_analysis.items()},
                    }
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "roc_analysis.json", "w", encoding="utf-8") as f:
                    json.dump(roc_data, f, indent=2)

                # PR analysis
                pr_data = {
                    algo: {
                        "macro_ap": res.macro_average_precision,
                        "curves": {k: v.to_dict() for k, v in res.pr_analysis.items()},
                    }
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "pr_analysis.json", "w", encoding="utf-8") as f:
                    json.dump(pr_data, f, indent=2)

                # Security metrics
                sec_data = {
                    algo: res.security_metrics.to_dict()
                    for algo, res in results_dict.items()
                }
                with open(part_dir / "security_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(sec_data, f, indent=2)

            # 2. Save master evaluation manifest
            manifest_file = temp_dir / "evaluation_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # 3. Atomic commit
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            os.replace(temp_dir, final_dir)

            logger.info(f"Successfully persisted evaluation run to {final_dir}")
            return final_dir

        except Exception as e:
            logger.error(f"Failed to persist evaluation run '{run_id}': {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def list_evaluation_runs(
        self,
        include_test_fixtures: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all valid persisted evaluation runs in reverse chronological order."""
        if not self.eval_dir.exists():
            return []

        runs: List[Dict[str, Any]] = []

        for run_dir in sorted(self.eval_dir.glob("eval_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue

            manifest_file = run_dir / "evaluation_manifest.json"
            if not manifest_file.exists():
                continue

            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Exclude test fixtures unless explicitly requested
                if not include_test_fixtures:
                    if data.get("is_test_fixture", False):
                        continue

                runs.append({
                    "evaluation_run_id": data.get("evaluation_run_id", run_dir.name),
                    "engineering_run_id": data.get("engineering_run_id", "unknown"),
                    "training_session_id": data.get("training_session_id", "unknown"),
                    "status": data.get("status", "Completed"),
                    "created_at_utc": data.get("created_at_utc", ""),
                    "completed_at_utc": data.get("completed_at_utc", ""),
                    "evaluated_algorithms": data.get("evaluated_algorithms", []),
                    "partitions_evaluated": data.get("partitions_evaluated", []),
                    "test_gate_confirmed": data.get("test_gate_confirmed", False),
                    "is_test_fixture": data.get("is_test_fixture", False),
                    "path": str(run_dir),
                })
            except Exception as e:
                logger.debug(f"Failed to read evaluation manifest in {run_dir}: {e}")

        return runs

    def get_evaluation_manifest(self, evaluation_run_id: str) -> Optional[EvaluationRunManifest]:
        """Load and reconstruct an evaluation run manifest."""
        manifest_file = self.eval_dir / evaluation_run_id / "evaluation_manifest.json"
        if not manifest_file.exists():
            return None

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationRunManifest.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load evaluation manifest for '{evaluation_run_id}': {e}")
            return None

    # Persistent Test Holdout Locks
    LOCK_FILE_NAME: str = "test_holdout_locks.json"

    def _get_locks_file_path(self) -> Path:
        return self.eval_dir / self.LOCK_FILE_NAME

    def get_test_lock(self, engineering_run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve persistent test holdout lock for a CP-05 engineering run if one exists."""
        locks_file = self._get_locks_file_path()
        if locks_file.exists():
            try:
                with open(locks_file, "r", encoding="utf-8") as f:
                    locks_data = json.load(f)
                if engineering_run_id in locks_data:
                    return locks_data[engineering_run_id]
            except Exception as e:
                logger.debug(f"Failed to read test holdout locks file: {e}")

        # Fallback inspection: scan existing evaluation manifests for this engineering run
        for run_dir in self.eval_dir.glob("eval_run_*"):
            if not run_dir.is_dir():
                continue
            manifest_file = run_dir / "evaluation_manifest.json"
            if not manifest_file.exists():
                continue
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("engineering_run_id") == engineering_run_id:
                    if data.get("test_holdout_locked") or data.get("test_gate_confirmed") or "test" in data.get("partitions_evaluated", []):
                        return {
                            "engineering_run_id": engineering_run_id,
                            "evaluation_run_id": data.get("evaluation_run_id", run_dir.name),
                            "locked_at_utc": data.get("test_holdout_locked_at_utc") or data.get("test_gate_confirmed_at_utc") or data.get("completed_at_utc"),
                            "candidate_set_hash": data.get("candidate_set_hash"),
                            "candidate_checksums": {k: v.get("model_checksum") for k, v in data.get("frozen_candidates", {}).items()},
                            "primary_candidate": next((k for k, v in data.get("frozen_candidates", {}).items() if v.get("selected_as_primary")), "unknown"),
                            "is_test_fixture": data.get("is_test_fixture", False),
                        }
            except Exception:
                pass

        return None

    def acquire_test_lock(
        self,
        engineering_run_id: str,
        evaluation_run_id: str,
        candidate_checksums: Dict[str, str],
        primary_candidate: str,
        candidate_set_hash: str,
        is_test_fixture: bool = False,
    ) -> None:
        """Atomically persist a test holdout lock for an engineering run."""
        locks_file = self._get_locks_file_path()
        locks_data: Dict[str, Any] = {}
        if locks_file.exists():
            try:
                with open(locks_file, "r", encoding="utf-8") as f:
                    locks_data = json.load(f)
            except Exception:
                locks_data = {}

        existing = locks_data.get(engineering_run_id)
        if existing and existing.get("evaluation_run_id") != evaluation_run_id:
            raise PermissionError(
                f"TEST HOLDOUT GOVERNANCE CONFLICT: Engineering run '{engineering_run_id}' "
                f"is already locked by evaluation run '{existing.get('evaluation_run_id')}'."
            )

        locks_data[engineering_run_id] = {
            "engineering_run_id": engineering_run_id,
            "evaluation_run_id": evaluation_run_id,
            "candidate_checksums": candidate_checksums,
            "primary_candidate": primary_candidate,
            "candidate_set_hash": candidate_set_hash,
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_test_fixture": is_test_fixture,
        }

        # Write atomically
        temp_locks = self.eval_dir / f".tmp_{self.LOCK_FILE_NAME}"
        with open(temp_locks, "w", encoding="utf-8") as f:
            json.dump(locks_data, f, indent=2)
        os.replace(temp_locks, locks_file)
        logger.info(f"Persisted test holdout lock for '{engineering_run_id}' under evaluation run '{evaluation_run_id}'")
