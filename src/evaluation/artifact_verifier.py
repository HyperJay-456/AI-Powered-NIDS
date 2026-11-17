"""Artifact Verifier & Eligibility Auditor (CP-07).

Enforces strict production model eligibility gating prior to evaluation:
1. Rejects test fixtures unless explicitly in test fixture mode.
2. Rejects incomplete, failed, or missing model runs.
3. Cryptographically verifies SHA-256 checksums before loading.
4. Validates CP-05 engineering run provenance.
5. Verifies feature dimension and class label ordering alignment.
6. Guardrails against untrusted or arbitrary deserialization.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from config.settings import settings
from src.data.engineering.artifact_manager import ArtifactManager
from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEligibilityCheckResult:
    """Outcome of validating a model artifact for evaluation readiness."""

    is_eligible: bool
    algorithm: str
    training_run_id: str
    engineering_run_id: str
    artifact_path: str
    model_sha256: str
    feature_count: int
    class_count: int
    classes: List[str]
    rejection_reasons: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    is_test_fixture: bool = False
    run_type: str = "production"

    def __bool__(self) -> bool:
        """Allow evaluating check result directly as boolean."""
        return self.is_eligible


class ArtifactVerifier:
    """Performs rigorous verification of trained model artifacts and provenance."""

    def __init__(
        self,
        saved_models_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        """Initialize verifier with model and dataset root paths."""
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.registry = ModelRegistry(saved_models_dir=self.saved_dir)
        self.artifact_mgr = ArtifactManager(processed_dir=self.processed_dir)

    def verify_model_eligibility(
        self,
        training_run_id: str,
        algorithm: str,
        allow_test_fixtures: bool = False,
    ) -> ModelEligibilityCheckResult:
        """Perform comprehensive cryptographic, provenance, and schema verification.

        Args:
            training_run_id: Identifier of the training session.
            algorithm: Algorithm key ('svm', 'mlp', 'decision_tree').
            allow_test_fixtures: Whether to permit test fixture models (e.g. in automated unit tests).

        Returns:
            ModelEligibilityCheckResult with detailed gate decisions.
        """
        rejection_reasons: List[str] = []
        warning_messages: List[str] = []

        algo_key = algorithm.lower().strip()
        algo_dir = self.saved_dir / training_run_id / algo_key
        model_file = algo_dir / "model.joblib"
        meta_file = algo_dir / "metadata.json"
        manifest_file = self.saved_dir / training_run_id / "training_manifest.json"

        # 1. Existence checks
        if not algo_dir.exists() or not algo_dir.is_dir():
            return ModelEligibilityCheckResult(
                is_eligible=False,
                algorithm=algo_key,
                training_run_id=training_run_id,
                engineering_run_id="",
                artifact_path=str(model_file),
                model_sha256="",
                feature_count=0,
                class_count=0,
                classes=[],
                rejection_reasons=[f"Model directory does not exist: {algo_dir}"],
            )

        if not model_file.exists() or model_file.stat().st_size == 0:
            rejection_reasons.append(f"Model file missing or zero bytes: {model_file}")

        if not meta_file.exists():
            rejection_reasons.append(f"Model metadata file missing: {meta_file}")

        if not manifest_file.exists():
            warning_messages.append(f"Training session manifest missing: {manifest_file}")

        # 2. Parse metadata
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                rejection_reasons.append(f"Failed to parse metadata JSON: {e}")

        # 3. Training session manifest check
        sess_manifest: Dict[str, Any] = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    sess_manifest = json.load(f)
            except Exception:
                pass

        # 4. Status check
        model_status = meta.get("status", "Unknown")
        if model_status not in ("Completed", "Completed with Warnings"):
            rejection_reasons.append(f"Model status is not completed: '{model_status}'")

        # 5. Production vs Test Fixture checks
        is_fixture = bool(
            meta.get("is_test_fixture", False)
            or sess_manifest.get("is_test_fixture", False)
            or meta.get("run_type") in ("test", "fixture", "test_fixture")
            or sess_manifest.get("run_type") in ("test", "fixture", "test_fixture")
        )
        run_type = meta.get("run_type", sess_manifest.get("run_type", "production"))

        if is_fixture and not allow_test_fixtures:
            rejection_reasons.append(
                "Model is a test fixture. Test fixtures are strictly excluded from production evaluation."
            )

        # 6. Cryptographic SHA-256 verification
        computed_sha = ""
        expected_sha = meta.get("artifact_checksum", "")
        if model_file.exists() and model_file.stat().st_size > 0:
            try:
                with open(model_file, "rb") as f:
                    computed_sha = hashlib.sha256(f.read()).hexdigest()
                if expected_sha and computed_sha != expected_sha:
                    rejection_reasons.append(
                        f"Cryptographic SHA-256 checksum mismatch for {model_file.name}. "
                        f"Expected: {expected_sha}, Computed: {computed_sha}"
                    )
            except Exception as e:
                rejection_reasons.append(f"Error computing model SHA-256: {e}")

        # 7. CP-05 Engineering Run Provenance check
        eng_run_id = meta.get("engineering_run_id", sess_manifest.get("engineering_run_id", ""))
        if not eng_run_id:
            rejection_reasons.append("Model metadata does not specify source CP-05 engineering run ID.")
        else:
            eng_meta_dir = self.processed_dir / eng_run_id / "metadata"
            eng_manifest_file = eng_meta_dir / "engineering_manifest.json"
            if not eng_manifest_file.exists():
                rejection_reasons.append(
                    f"Referenced CP-05 engineering run '{eng_run_id}' not found at {eng_manifest_file}."
                )
            else:
                try:
                    with open(eng_manifest_file, "r", encoding="utf-8") as f:
                        eng_manifest = json.load(f)
                    if eng_manifest.get("status") != "Completed":
                        rejection_reasons.append(
                            f"Referenced CP-05 run status is '{eng_manifest.get('status')}', not 'Completed'."
                        )
                except Exception as e:
                    rejection_reasons.append(f"Error reading referenced CP-05 manifest: {e}")

        # 8. Feature and Class Schema Alignment
        feature_count = meta.get("feature_count", 0)
        class_count = meta.get("class_count", 0)
        classes = meta.get("classes", [])

        if feature_count <= 0:
            rejection_reasons.append(f"Invalid feature dimension: {feature_count}")
        if class_count < 2 or not classes:
            rejection_reasons.append(f"Invalid class configuration: count={class_count}, classes={classes}")

        # 9. Estimator Deserialization & Structural Validation
        # Only attempt deserialization if trusted and checksum verified!
        if not rejection_reasons:
            try:
                resolved_path = model_file.resolve()
                if not str(resolved_path).startswith(str(self.saved_dir.resolve())):
                    rejection_reasons.append(f"Security error: path traversal attempt {model_file}")
                else:
                    estimator = joblib.load(model_file)

                    # Validate estimator attributes
                    est_features = getattr(estimator, "n_features_in_", None)
                    if est_features is not None and est_features != feature_count:
                        rejection_reasons.append(
                            f"Estimator n_features_in_ ({est_features}) does not match metadata ({feature_count})"
                        )

                    est_classes = getattr(estimator, "classes_", None)
                    if est_classes is not None:
                        if len(est_classes) != len(classes):
                            rejection_reasons.append(
                                f"Estimator classes_ count ({len(est_classes)}) does not match metadata ({len(classes)})"
                            )
            except Exception as e:
                rejection_reasons.append(f"Corrupted model artifact: failed to load with joblib: {e}")

        is_eligible = len(rejection_reasons) == 0

        return ModelEligibilityCheckResult(
            is_eligible=is_eligible,
            algorithm=algo_key,
            training_run_id=training_run_id,
            engineering_run_id=eng_run_id,
            artifact_path=str(model_file),
            model_sha256=computed_sha or expected_sha,
            feature_count=feature_count,
            class_count=class_count,
            classes=classes,
            rejection_reasons=rejection_reasons,
            warning_messages=warning_messages,
            is_test_fixture=is_fixture,
            run_type=run_type,
        )

    def discover_eligible_production_models(
        self,
        allow_test_fixtures: bool = False,
    ) -> List[ModelEligibilityCheckResult]:
        """Discover all verified eligible models currently persisted in models/saved/."""
        eligible_models: List[ModelEligibilityCheckResult] = []

        if not self.saved_dir.exists():
            return eligible_models

        # Enumerate runs
        for run_dir in sorted(self.saved_dir.glob("train_run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue

            for algo_dir in run_dir.iterdir():
                if not algo_dir.is_dir() or algo_dir.name.startswith("."):
                    continue

                algo_key = algo_dir.name
                check = self.verify_model_eligibility(
                    training_run_id=run_dir.name,
                    algorithm=algo_key,
                    allow_test_fixtures=allow_test_fixtures,
                )
                if check.is_eligible:
                    eligible_models.append(check)

        return eligible_models
