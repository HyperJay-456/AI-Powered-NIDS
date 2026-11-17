"""Model Registry & Artifact Persistence Engine (CP-06).

Manages atomic disk persistence, versioning, metadata inspection, and cryptographic verification
for trained machine learning models in models/saved/.
Structure:
  models/saved/<training_run_id>/
    ├── training_manifest.json
    ├── svm/
    │   ├── model.joblib
    │   └── metadata.json
    ├── mlp/
    │   ├── model.joblib
    │   └── metadata.json
    └── decision_tree/
        ├── model.joblib
        └── metadata.json
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Dict, List, Optional, Tuple

import joblib

from config.settings import settings
from src.models.training_manifest import (
    AlgorithmTrainingManifest,
    MasterTrainingRunManifest,
    capture_training_environment,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifactRecord:
    """Summary of a registered model artifact."""

    training_run_id: str
    algorithm: str
    algorithm_display_name: str
    estimator_class: str
    status: str
    artifact_path: str
    model_file_sha256: str
    model_size_bytes: int
    training_duration_seconds: float
    training_samples: int
    feature_count: int
    class_count: int
    classes: List[str]
    engineering_run_id: str
    convergence_status: str
    warnings: List[str]
    created_at_utc: str
    run_type: str = "production"
    is_test_fixture: bool = False


class ModelRegistry:
    """Provides atomic persistence, discovery, and verification for trained models."""

    def __init__(self, saved_models_dir: Optional[Path] = None):
        """Initialize registry with models directory."""
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.saved_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_training_run_id() -> str:
        """Generate a deterministic timestamped unique ID for a training session."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_token = uuid.uuid4().hex[:6]
        return f"train_run_{timestamp}_{rand_token}"

    def save_training_session(
        self,
        training_run_id: str,
        engineering_run_id: str,
        results_by_algo: Dict[str, Any],  # algorithm -> TrainingResult
        training_manifests: Dict[str, AlgorithmTrainingManifest],
        started_at_utc: str,
        run_type: str = "production",
        is_test_fixture: bool = False,
    ) -> Path:
        """Atomically persist trained models, metadata, and master training manifest.

        Args:
            training_run_id: Unique session ID.
            engineering_run_id: CP-05 source run ID.
            results_by_algo: Mapping of algorithm keys to trainer results.
            training_manifests: Mapping of algorithm keys to individual manifests.
            started_at_utc: ISO timestamp when session began.
            run_type: Execution mode ('production' or 'test_fixture').
            is_test_fixture: Explicit boolean flag tagging test fixture runs.

        Returns:
            Path to the committed training run directory.
        """
        final_dir = self.saved_dir / training_run_id
        temp_dir = self.saved_dir / f".tmp_{training_run_id}"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc).isoformat()
        overall_status = "Completed"

        try:
            for algo_key, result in results_by_algo.items():
                if result.status == "Failed":
                    if overall_status == "Completed":
                        overall_status = "Partial Success"
                    continue

                algo_dir = temp_dir / algo_key
                algo_dir.mkdir(parents=True, exist_ok=True)

                # 1. Save fitted model artifact via joblib
                model_file = algo_dir / "model.joblib"
                joblib.dump(result.estimator, model_file, compress=3)

                # 2. Compute SHA-256 of the serialized model
                with open(model_file, "rb") as f:
                    model_sha256 = hashlib.sha256(f.read()).hexdigest()

                model_size = model_file.stat().st_size

                # Update individual manifest
                manifest = training_manifests.get(algo_key)
                if manifest:
                    manifest.artifact_checksum = model_sha256
                    manifest.artifact_path = f"models/saved/{training_run_id}/{algo_key}/model.joblib"
                    manifest.model_size_bytes = model_size
                    manifest.run_type = run_type
                    manifest.is_test_fixture = is_test_fixture

                    # Save metadata.json
                    meta_file = algo_dir / "metadata.json"
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(manifest.to_dict(), f, indent=2, default=str)

            # 3. Save master session training manifest
            master_manifest = MasterTrainingRunManifest(
                session_run_id=training_run_id,
                engineering_run_id=engineering_run_id,
                started_at_utc=started_at_utc,
                completed_at_utc=now_utc,
                overall_status=overall_status,
                trained_algorithms=list(results_by_algo.keys()),
                model_manifests=training_manifests,
                run_type=run_type,
                is_test_fixture=is_test_fixture,
            )

            manifest_file = temp_dir / "training_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(master_manifest.to_dict(), f, indent=2, default=str)

            # 4. Atomic directory rename
            os.replace(temp_dir, final_dir)
            logger.info(f"Atomically committed training run to {final_dir} (run_type={run_type})")
            return final_dir

        except Exception as e:
            logger.error(f"Failed to persist training run '{training_run_id}': {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def list_training_runs(self, include_test_fixtures: bool = False) -> List[Dict[str, Any]]:
        """List all valid persisted training runs in reverse chronological order."""
        if not self.saved_dir.exists():
            return []

        runs: List[Dict[str, Any]] = []
        for run_dir in sorted(self.saved_dir.glob("train_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue

            manifest_file = run_dir / "training_manifest.json"
            if not manifest_file.exists():
                continue

            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter out test fixtures unless explicitly requested
                if not include_test_fixtures:
                    if data.get("is_test_fixture", False) or data.get("run_type") in ("test", "fixture", "test_fixture"):
                        continue

                runs.append({
                    "session_run_id": data.get("session_run_id", run_dir.name),
                    "engineering_run_id": data.get("engineering_run_id", "unknown"),
                    "started_at_utc": data.get("started_at_utc", ""),
                    "completed_at_utc": data.get("completed_at_utc", ""),
                    "overall_status": data.get("overall_status", "Completed"),
                    "trained_algorithms": data.get("trained_algorithms", []),
                    "model_count": len(data.get("model_manifests", {})),
                    "path": str(run_dir),
                    "run_type": data.get("run_type", "production"),
                    "is_test_fixture": data.get("is_test_fixture", False),
                })
            except Exception as e:
                logger.debug(f"Failed to read training manifest in {run_dir}: {e}")

        return runs

    def list_all_trained_models(
        self,
        include_test_fixtures: bool = False,
        verify_sha256: bool = True,
    ) -> List[ModelArtifactRecord]:
        """Discover every registered individual model across all training runs."""
        models: List[ModelArtifactRecord] = []

        for run in self.list_training_runs(include_test_fixtures=include_test_fixtures):
            run_id = run["session_run_id"]
            run_dir = self.saved_dir / run_id

            for algo_dir in run_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_key = algo_dir.name
                model_file = algo_dir / "model.joblib"
                meta_file = algo_dir / "metadata.json"

                if not model_file.exists() or model_file.stat().st_size == 0:
                    continue

                meta: Dict[str, Any] = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass

                # Filter out test fixtures if requested
                if not include_test_fixtures:
                    if meta.get("is_test_fixture", False) or meta.get("run_type") in ("test", "fixture", "test_fixture"):
                        continue

                # Verify SHA-256 before marking model as available
                if verify_sha256:
                    if not self.verify_model_integrity(run_id, algo_key):
                        logger.warning(
                            f"Model integrity verification failed for '{run_id}/{algo_key}'. "
                            "Excluding corrupted or mismatched artifact from registry."
                        )
                        continue

                models.append(ModelArtifactRecord(
                    training_run_id=run_id,
                    algorithm=algo_key,
                    algorithm_display_name=meta.get("algorithm_display_name", algo_key.upper()),
                    estimator_class=meta.get("estimator_class", "unknown"),
                    status=meta.get("status", "Completed"),
                    artifact_path=str(model_file),
                    model_file_sha256=meta.get("artifact_checksum", ""),
                    model_size_bytes=model_file.stat().st_size,
                    training_duration_seconds=meta.get("training_duration_seconds", 0.0),
                    training_samples=meta.get("training_samples", 0),
                    feature_count=meta.get("feature_count", 0),
                    class_count=meta.get("class_count", 0),
                    classes=meta.get("classes", []),
                    engineering_run_id=meta.get("engineering_run_id", run.get("engineering_run_id", "")),
                    convergence_status=meta.get("convergence_status", "Converged"),
                    warnings=meta.get("warnings", []),
                    created_at_utc=meta.get("completed_at_utc", run.get("completed_at_utc", "")),
                    run_type=meta.get("run_type", run.get("run_type", "production")),
                    is_test_fixture=meta.get("is_test_fixture", run.get("is_test_fixture", False)),
                ))

        return models

    def get_latest_model(
        self,
        algorithm: str,
        include_test_fixtures: bool = False,
    ) -> Optional[ModelArtifactRecord]:
        """Retrieve the most recent verified model artifact for a specific algorithm."""
        algo = algorithm.lower().strip()
        all_models = self.list_all_trained_models(include_test_fixtures=include_test_fixtures, verify_sha256=True)
        matching = [m for m in all_models if m.algorithm == algo and m.status in ("Completed", "Completed with Warnings")]
        return matching[0] if matching else None

    def load_model(self, training_run_id: str, algorithm: str) -> Any:
        """Load fitted model estimator safely from the registry."""
        model_path = self.saved_dir / training_run_id / algorithm / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        # Security check: ensure path is strictly within saved_models_dir
        resolved = model_path.resolve()
        if not str(resolved).startswith(str(self.saved_dir.resolve())):
            raise SecurityError(f"Path traversal detected: {model_path}")

        return joblib.load(model_path)

    def verify_model_integrity(self, training_run_id: str, algorithm: str) -> bool:
        """Verify model file existence and SHA-256 match against metadata."""
        algo_dir = self.saved_dir / training_run_id / algorithm
        model_file = algo_dir / "model.joblib"
        meta_file = algo_dir / "metadata.json"

        if not model_file.exists() or not meta_file.exists():
            return False

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_sha = meta.get("artifact_checksum")

            with open(model_file, "rb") as f:
                computed_sha = hashlib.sha256(f.read()).hexdigest()

            return expected_sha == computed_sha
        except Exception:
            return False
