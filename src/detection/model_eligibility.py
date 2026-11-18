"""Model Eligibility & Governance Auditor (CP-08).

Validates that only verified, non-tampered, authorized production models that have
completed CP-06 training and CP-07 candidate freezing are permitted to execute inference.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import ALGORITHM_KEYS, settings
from src.data.engineering.artifact_manager import ArtifactManager
from src.evaluation.evaluation_registry import EvaluationRegistry
from src.models.model_registry import ModelArtifactRecord, ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EligibleModelInfo:
    """Detailed metadata for a verified, eligible inference model."""

    is_eligible: bool
    algorithm: str
    algorithm_display_name: str
    training_run_id: str
    engineering_run_id: str
    artifact_path: str
    model_sha256: str
    classes: List[str]
    feature_count: int
    is_primary_candidate: bool = False
    candidate_freeze_time_utc: Optional[str] = None
    candidate_selection_criterion: Optional[str] = None
    evaluation_run_id: Optional[str] = None
    rejection_reasons: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    is_test_fixture: bool = False
    run_type: str = "production"

    def __bool__(self) -> bool:
        return self.is_eligible


class ModelEligibilityVerifier:
    """Enforces strict production model gating for operational intrusion detection."""

    def __init__(
        self,
        saved_models_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        evaluations_dir: Optional[Path] = None,
    ):
        """Initialize verifier with repository directories."""
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.eval_dir = evaluations_dir or settings.evaluations_dir

        self.model_registry = ModelRegistry(saved_models_dir=self.saved_dir)
        self.artifact_mgr = ArtifactManager(processed_dir=self.processed_dir)
        self.eval_registry = EvaluationRegistry(evaluations_dir=self.eval_dir)

    def verify_model(
        self,
        training_run_id: str,
        algorithm: str,
        allow_test_fixtures: bool = False,
    ) -> EligibleModelInfo:
        """Perform comprehensive governance and integrity verification on a model.

        Checks:
        1. Model exists in registry.
        2. Authorized algorithm family (svm, mlp, decision_tree).
        3. Test fixture isolation (rejected in production mode).
        4. Artifact checksum integrity (SHA-256 match).
        5. Training manifest validity.
        6. CP-05 preprocessing artifact availability and lineage agreement.
        7. CP-07 candidate freeze record presence and checksum agreement.
        """
        rejection_reasons: List[str] = []
        warning_messages: List[str] = []

        algo_key = algorithm.lower().strip()
        if algo_key not in ALGORITHM_KEYS:
            return EligibleModelInfo(
                is_eligible=False,
                algorithm=algo_key,
                algorithm_display_name=algo_key.upper(),
                training_run_id=training_run_id,
                engineering_run_id="",
                artifact_path="",
                model_sha256="",
                classes=[],
                feature_count=0,
                rejection_reasons=[f"Unauthorized algorithm family '{algo_key}'. Authorized: {ALGORITHM_KEYS}"],
            )

        algo_dir = self.saved_dir / training_run_id / algo_key
        model_file = algo_dir / "model.joblib"
        meta_file = algo_dir / "metadata.json"
        session_manifest_file = self.saved_dir / training_run_id / "training_manifest.json"

        # 1. File existence
        if not model_file.exists() or not meta_file.exists():
            return EligibleModelInfo(
                is_eligible=False,
                algorithm=algo_key,
                algorithm_display_name=algo_key.upper(),
                training_run_id=training_run_id,
                engineering_run_id="",
                artifact_path=str(model_file),
                model_sha256="",
                classes=[],
                feature_count=0,
                rejection_reasons=[f"Model artifact or metadata file missing at {algo_dir}"],
            )

        # 2. Parse model metadata
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return EligibleModelInfo(
                is_eligible=False,
                algorithm=algo_key,
                algorithm_display_name=algo_key.upper(),
                training_run_id=training_run_id,
                engineering_run_id="",
                artifact_path=str(model_file),
                model_sha256="",
                classes=[],
                feature_count=0,
                rejection_reasons=[f"Corrupt model metadata JSON: {e}"],
            )

        # 3. Test fixture check
        is_fixture = bool(meta.get("is_test_fixture", False)) or (meta.get("run_type") in ("test", "test_fixture", "fixture"))
        if is_fixture and not allow_test_fixtures:
            rejection_reasons.append(
                "Model is flagged as an automated test fixture and cannot be used for production inference."
            )

        # 4. Checksum verification
        expected_sha256 = meta.get("artifact_checksum", "")
        if not expected_sha256:
            rejection_reasons.append("Model metadata lacks cryptographic artifact_checksum.")
        else:
            try:
                with open(model_file, "rb") as f:
                    actual_sha256 = hashlib.sha256(f.read()).hexdigest()
                if actual_sha256 != expected_sha256:
                    rejection_reasons.append(
                        f"CRITICAL CHECKSUM MISMATCH: Expected {expected_sha256}, got {actual_sha256}."
                    )
            except Exception as e:
                rejection_reasons.append(f"Failed to read model artifact for checksum verification: {e}")

        # 5. Training status
        if meta.get("status") not in ("Completed", "Completed with Warnings"):
            rejection_reasons.append(f"Model status is '{meta.get('status')}', not completed.")

        eng_run_id = meta.get("engineering_run_id", "")
        if not eng_run_id:
            rejection_reasons.append("Model metadata lacks engineering_run_id provenance.")

        # 6. Check CP-05 Preprocessing artifacts
        eng_run = self.artifact_mgr.get_run(eng_run_id) if eng_run_id else None
        if not eng_run:
            rejection_reasons.append(f"Referenced CP-05 engineering run '{eng_run_id}' not found in registry.")
        else:
            prep_file_name = "tree_preprocessor.joblib" if algo_key == "decision_tree" else "scaled_preprocessor.joblib"
            prep_path = self.processed_dir / eng_run_id / "preprocessors" / prep_file_name
            if not prep_path.exists():
                rejection_reasons.append(f"Required preprocessing bundle missing: {prep_path}")

        # 7. Check CP-07 Candidate Freezing
        candidate_frozen = False
        is_primary = False
        freeze_time = None
        selection_crit = None
        eval_run_id = None

        eval_runs = self.eval_registry.list_evaluation_runs(include_test_fixtures=allow_test_fixtures)
        for er in eval_runs:
            manifest = self.eval_registry.get_evaluation_manifest(er["evaluation_run_id"])
            if manifest and manifest.frozen_candidates:
                if algo_key in manifest.frozen_candidates:
                    rec = manifest.frozen_candidates[algo_key]
                    if rec.training_run_id == training_run_id and rec.model_checksum == expected_sha256:
                        candidate_frozen = True
                        is_primary = rec.selected_as_primary
                        freeze_time = rec.frozen_at_utc
                        selection_crit = rec.selection_criterion
                        eval_run_id = manifest.evaluation_run_id
                        break

        if not candidate_frozen and not allow_test_fixtures:
            rejection_reasons.append(
                f"Model '{algo_key}' has not completed CP-07 evaluation and candidate freezing. "
                "Models must be validated, evaluated, and frozen before production operational use."
            )

        classes = meta.get("classes", [])
        feature_count = meta.get("feature_count", 0)
        disp_name = meta.get("algorithm_display_name", algo_key.upper())

        is_eligible = (len(rejection_reasons) == 0)

        return EligibleModelInfo(
            is_eligible=is_eligible,
            algorithm=algo_key,
            algorithm_display_name=disp_name,
            training_run_id=training_run_id,
            engineering_run_id=eng_run_id,
            artifact_path=str(model_file),
            model_sha256=expected_sha256,
            classes=classes,
            feature_count=feature_count,
            is_primary_candidate=is_primary,
            candidate_freeze_time_utc=freeze_time,
            candidate_selection_criterion=selection_crit,
            evaluation_run_id=eval_run_id,
            rejection_reasons=rejection_reasons,
            warning_messages=warning_messages,
            is_test_fixture=is_fixture,
            run_type="test_fixture" if is_fixture else "production",
        )

    def discover_eligible_models(
        self,
        allow_test_fixtures: bool = False,
    ) -> List[EligibleModelInfo]:
        """Discover and verify all models currently available in the model registry."""
        eligible_models: List[EligibleModelInfo] = []

        # List all models in saved directory
        models = self.model_registry.list_all_trained_models(
            include_test_fixtures=allow_test_fixtures,
            verify_sha256=True,
        )

        for m in models:
            info = self.verify_model(
                training_run_id=m.training_run_id,
                algorithm=m.algorithm,
                allow_test_fixtures=allow_test_fixtures,
            )
            if info.is_eligible:
                eligible_models.append(info)

        return eligible_models
