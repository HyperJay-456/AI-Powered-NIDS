"""Evaluation Orchestrator & Protocol Gating Engine (CP-07).

Coordinates the end-to-end evaluation lifecycle with strict methodological guardrails:
1. Validation Evaluation:
   - Validates production model eligibility (SHA-256, schema, CP-05 provenance).
   - Enforces zero-retraining protocol (no model fitting or updating permitted).
   - Loads validation partition CSR matrices (scaled for SVM/MLP, unscaled for DecisionTree).
   - Executes resource-gated chunked batch inference.
   - Computes multiclass metrics, confusion matrices, ROC/PR curves, and security summaries.
   - Atomically records results in models/evaluations/<evaluation_run_id>/.

2. Candidate Freezing & Gating:
   - Locks candidate model artifacts, hyperparameters, and cryptographic checksums.
   - Records the selected primary candidate model and selection criteria based on validation.

3. Gated Final Test Holdout Evaluation:
   - Requires completed validation evaluation, frozen candidate artifacts, and explicit confirmation.
   - Re-verifies model checksums against frozen state before evaluation.
   - Evaluates strictly on the immutable test holdout partition.
   - Completely prohibits using test performance to tune, re-select, or modify models.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from src.evaluation.artifact_verifier import ArtifactVerifier
from src.evaluation.confusion_matrix import ConfusionMatrixEngine
from src.evaluation.evaluation_data_loader import EvaluationDataLoader
from src.evaluation.evaluation_manifest import (
    AlgorithmEvaluationResult,
    CandidateModelFreezeRecord,
    EvaluationRunManifest,
    capture_evaluation_environment,
    compute_candidate_set_hash,
)
from src.evaluation.evaluation_registry import EvaluationRegistry
from src.evaluation.inference_runner import InferenceRunner
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.precision_recall_analysis import PrecisionRecallAnalyzer
from src.evaluation.resource_manager import EvaluationResourceManager
from src.evaluation.roc_analysis import ROCAnalyzer
from src.evaluation.security_metrics import SecurityMetricsEngine
from src.models.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationOrchestrator:
    """Master workflow orchestrator for model evaluation, candidate freezing, and test holdout gating."""

    def __init__(
        self,
        saved_models_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        evaluations_dir: Optional[Path] = None,
    ):
        """Initialize orchestrator with subsystem directories."""
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.eval_dir = evaluations_dir or settings.evaluations_dir

        self.verifier = ArtifactVerifier(
            saved_models_dir=self.saved_dir,
            processed_dir=self.processed_dir,
        )
        self.data_loader = EvaluationDataLoader(processed_dir=self.processed_dir)
        self.model_registry = ModelRegistry(saved_models_dir=self.saved_dir)
        self.eval_registry = EvaluationRegistry(evaluations_dir=self.eval_dir)

    def run_validation_evaluation(
        self,
        training_run_id: str,
        algorithms: Optional[List[str]] = None,
        allow_test_fixtures: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> EvaluationRunManifest:
        """Execute validation evaluation across selected eligible models.

        Args:
            training_run_id: Training session identifier.
            algorithms: List of algorithm keys to evaluate (default: all eligible in session).
            allow_test_fixtures: Whether test fixtures are allowed (for unit testing).
            progress_callback: Optional progress reporter callback.

        Returns:
            EvaluationRunManifest populated with validation results.
        """
        # 1. Discover and verify eligible models
        run_dir = self.saved_dir / training_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Training run directory not found: {run_dir}")

        candidate_algos = algorithms or ["svm", "mlp", "decision_tree"]
        eligible_checks: Dict[str, Any] = {}

        for algo in candidate_algos:
            check = self.verifier.verify_model_eligibility(
                training_run_id=training_run_id,
                algorithm=algo,
                allow_test_fixtures=allow_test_fixtures,
            )
            if check.is_eligible:
                eligible_checks[algo] = check
            else:
                logger.warning(
                    f"Model '{training_run_id}/{algo}' not eligible for evaluation: "
                    f"{'; '.join(check.rejection_reasons)}"
                )

        if not eligible_checks:
            raise ValueError(
                f"No eligible models found in training run '{training_run_id}' for algorithms {candidate_algos}."
            )

        # 2. Extract common CP-05 engineering run ID
        first_check = next(iter(eligible_checks.values()))
        eng_run_id = first_check.engineering_run_id
        is_fixture = first_check.is_test_fixture

        eval_run_id = self.eval_registry.generate_evaluation_run_id()
        started_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Beginning validation evaluation '{eval_run_id}' for {len(eligible_checks)} models "
            f"against CP-05 engineering run '{eng_run_id}'."
        )

        validation_results: Dict[str, AlgorithmEvaluationResult] = {}
        model_checksums: Dict[str, str] = {}
        dataset_checksums: Dict[str, str] = {}

        total_steps = len(eligible_checks)
        current_step = 0

        # 3. Evaluate each eligible model sequentially
        for algo_key, check in eligible_checks.items():
            current_step += 1
            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"Loading validation partition for {algo_key.upper()} ({current_step}/{total_steps})...",
                )

            # Record model artifact checksum
            model_checksums[algo_key] = check.model_sha256

            # Load validation partition (ZERO-LEAKAGE: val only!)
            val_part = self.data_loader.load_partition(
                engineering_run_id=eng_run_id,
                partition="validation",
                algorithm=algo_key,
            )
            dataset_checksums["validation_matrix"] = val_part.matrix_checksum
            dataset_checksums["validation_labels"] = val_part.labels_checksum

            # Resource preflight
            preflight = EvaluationResourceManager.perform_preflight_check(
                num_samples=val_part.num_samples,
                num_features=val_part.num_features,
                num_classes=val_part.num_classes,
                target_path=self.eval_dir,
            )

            # Load model estimator safely
            estimator = self.model_registry.load_model(training_run_id, algo_key)

            # Batch inference
            runner = InferenceRunner(batch_size=preflight.batch_size)
            inf_res = runner.run_inference(
                estimator=estimator,
                algorithm=algo_key,
                X=val_part.X,
                y_true=val_part.y,
                classes=val_part.classes,
                progress_callback=progress_callback,
            )

            # Compute classification metrics
            agg_metrics = MetricsCalculator.calculate_aggregate_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
            )

            per_class_metrics = MetricsCalculator.calculate_per_class_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=val_part.classes,
            )

            # Compute confusion matrix
            cm_record = ConfusionMatrixEngine.compute_matrix(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=val_part.classes,
            )

            # ROC Analysis
            roc_results, macro_auc = ROCAnalyzer.compute_ovr_roc(
                y_true=inf_res.y_true,
                scores=inf_res.scores,
                classes=val_part.classes,
                score_type=inf_res.score_type,
            )

            # PR Analysis
            pr_results, macro_ap = PrecisionRecallAnalyzer.compute_ovr_pr(
                y_true=inf_res.y_true,
                scores=inf_res.scores,
                classes=val_part.classes,
                score_type=inf_res.score_type,
            )

            # Security Benign vs Attack analysis
            sec_metrics = SecurityMetricsEngine.compute_security_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=val_part.classes,
            )

            display_name = {
                "svm": "Support Vector Machine (LinearSVC)",
                "mlp": "Multilayer Perceptron (MLP)",
                "decision_tree": "Decision Tree (CART)",
            }.get(algo_key, algo_key.upper())

            algo_result = AlgorithmEvaluationResult(
                algorithm=algo_key,
                algorithm_display_name=display_name,
                training_run_id=training_run_id,
                model_checksum=check.model_sha256,
                partition="validation",
                evaluated_rows=inf_res.successfully_predicted_rows,
                failed_rows=inf_res.failed_rows,
                duration_seconds=inf_res.duration_seconds,
                ms_per_record=inf_res.ms_per_record,
                score_type=inf_res.score_type,
                aggregate_metrics=agg_metrics,
                per_class_metrics=per_class_metrics,
                confusion_matrix=cm_record,
                roc_analysis=roc_results,
                macro_roc_auc=macro_auc,
                pr_analysis=pr_results,
                macro_average_precision=macro_ap,
                security_metrics=sec_metrics,
                warnings=inf_res.warnings + preflight.warning_messages,
                error_message=None,
                is_test_fixture=is_fixture,
            )

            validation_results[algo_key] = algo_result

        completed_at = datetime.now(timezone.utc).isoformat()

        # Build master manifest
        manifest = EvaluationRunManifest(
            evaluation_run_id=eval_run_id,
            engineering_run_id=eng_run_id,
            training_session_id=training_run_id,
            status="Completed",
            created_at_utc=started_at,
            completed_at_utc=completed_at,
            evaluated_algorithms=list(validation_results.keys()),
            partitions_evaluated=["validation"],
            dataset_checksums=dataset_checksums,
            model_checksums=model_checksums,
            validation_results=validation_results,
            frozen_candidates={},
            test_gate_confirmed=False,
            test_results={},
            environment=capture_evaluation_environment(),
            is_test_fixture=is_fixture,
            notes=["Validation evaluation completed successfully."],
        )

        # Persist atomically
        self.eval_registry.save_evaluation_run(manifest)
        return manifest

    def freeze_candidate_models(
        self,
        evaluation_run_id: str,
        primary_candidate_algo: str,
        selection_criterion: str = "Highest validation macro F1-score with bounded false alarm rate",
    ) -> EvaluationRunManifest:
        """Lock candidate models and cryptographic checksums based on validation performance.

        Args:
            evaluation_run_id: ID of the completed validation evaluation run.
            primary_candidate_algo: Algorithm chosen as primary deployment candidate.
            selection_criterion: Objective documented justification for selection.

        Returns:
            Updated EvaluationRunManifest with frozen candidate records.
        """
        manifest = self.eval_registry.get_evaluation_manifest(evaluation_run_id)
        if not manifest:
            raise FileNotFoundError(f"Evaluation manifest '{evaluation_run_id}' not found.")

        # GOVERNANCE CHECK 1: Prevent changing candidate set after test evaluation has occurred
        if manifest.test_holdout_locked or manifest.test_gate_confirmed or manifest.test_results:
            raise PermissionError(
                f"GOVERNANCE VIOLATION: Candidate models for evaluation run '{evaluation_run_id}' "
                "cannot be modified or re-frozen because final test holdout evaluation has already been executed."
            )

        if not manifest.validation_results:
            raise ValueError("Cannot freeze candidates: No validation evaluation results present.")

        frozen_candidates: Dict[str, CandidateModelFreezeRecord] = {}
        frozen_time = datetime.now(timezone.utc).isoformat()

        primary_algo = primary_candidate_algo.lower().strip()
        if primary_algo not in manifest.validation_results:
            raise ValueError(
                f"Primary candidate '{primary_candidate_algo}' not found in evaluated validation results: "
                f"{list(manifest.validation_results.keys())}"
            )

        for algo_key, res in manifest.validation_results.items():
            is_primary = (algo_key == primary_algo)
            sec = res.security_metrics

            # Re-verify current model on disk to capture artifact path and verify current checksum
            check = self.verifier.verify_model_eligibility(
                training_run_id=res.training_run_id,
                algorithm=algo_key,
                allow_test_fixtures=manifest.is_test_fixture,
            )
            artifact_path = check.artifact_path if check else None

            frozen_candidates[algo_key] = CandidateModelFreezeRecord(
                algorithm=algo_key,
                algorithm_display_name=res.algorithm_display_name,
                training_run_id=res.training_run_id,
                model_checksum=res.model_checksum,
                frozen_at_utc=frozen_time,
                selected_as_primary=is_primary,
                selection_criterion=selection_criterion if is_primary else "Evaluated benchmark candidate",
                validation_macro_f1=res.aggregate_metrics.macro_f1,
                validation_accuracy=res.aggregate_metrics.accuracy,
                validation_far=sec.benign_false_alarm_rate if sec.is_available else None,
                artifact_path=artifact_path,
            )

        manifest.frozen_candidates = frozen_candidates
        manifest.candidate_set_hash = compute_candidate_set_hash(frozen_candidates)
        manifest.notes.append(
            f"Candidate models frozen at {frozen_time}. Primary candidate designated: '{primary_algo.upper()}'. "
            f"Candidate set hash: {manifest.candidate_set_hash[:16]}..."
        )

        self.eval_registry.save_evaluation_run(manifest)
        return manifest

    def run_gated_test_evaluation(
        self,
        evaluation_run_id: str,
        confirmation_token: str,
        notes: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> EvaluationRunManifest:
        """Evaluate frozen candidate models strictly on the held-out test partition.

        CRITICAL EVALUATION PROTOCOL:
        1. Candidates must have been frozen based on validation results prior to this call.
        2. Checks that current model artifact checksums match the frozen candidate checksums.
        3. Requires explicit confirmation token.
        4. Test performance is recorded as final holdout evaluation and CANNOT be used to tune models.

        Args:
            evaluation_run_id: Existing evaluation run ID.
            confirmation_token: Explicit confirmation string.
            notes: Optional user-supplied confirmation notes.
            progress_callback: Optional progress reporter.

        Returns:
            Updated EvaluationRunManifest with immutable test results.
        """
        if confirmation_token != "CONFIRM_TEST_EVALUATION":
            raise PermissionError(
                "Final test evaluation requires explicit confirmation. "
                "The test partition is a strictly isolated final holdout."
            )

        manifest = self.eval_registry.get_evaluation_manifest(evaluation_run_id)
        if not manifest:
            raise FileNotFoundError(f"Evaluation manifest '{evaluation_run_id}' not found.")

        # GOVERNANCE CHECK 1: Disallow re-running test evaluation on this evaluation run
        if manifest.test_holdout_locked or manifest.test_gate_confirmed or manifest.test_results:
            raise PermissionError(
                f"TEST HOLDOUT LOCKED: The final test holdout partition has already been evaluated for "
                f"run '{evaluation_run_id}' (locked at {manifest.test_holdout_locked_at_utc or manifest.test_gate_confirmed_at_utc or 'earlier'}). "
                "Test results are immutable and cannot be rerun."
            )

        # GOVERNANCE CHECK 2: Require complete candidate set frozen before test access
        if not manifest.frozen_candidates:
            raise PermissionError(
                "Evaluation protocol violation: Candidate models must be selected and frozen "
                "based on validation results before executing final test holdout evaluation."
            )

        # GOVERNANCE CHECK 3: Persistent engineering run lock (prevent candidate alteration & snooping)
        existing_lock = self.eval_registry.get_test_lock(manifest.engineering_run_id)
        if existing_lock:
            locked_eval_id = existing_lock.get("evaluation_run_id")
            locked_hash = existing_lock.get("candidate_set_hash")
            curr_hash = manifest.candidate_set_hash or compute_candidate_set_hash(manifest.frozen_candidates)

            if locked_eval_id != evaluation_run_id or locked_hash != curr_hash:
                raise PermissionError(
                    f"TEST HOLDOUT GOVERNANCE VIOLATION: The final test holdout partition for engineering run "
                    f"'{manifest.engineering_run_id}' is permanently locked by evaluation run '{locked_eval_id}'. "
                    "Changing the candidate set or rerunning the test partition on the same engineering split "
                    "is strictly prohibited to prevent data snooping and model selection based on test performance."
                )
            else:
                raise PermissionError(
                    f"TEST HOLDOUT LOCKED: The final test holdout partition for engineering run "
                    f"'{manifest.engineering_run_id}' has already been evaluated in run '{locked_eval_id}'. "
                    "The test partition is an immutable one-time holdout."
                )

        eng_run_id = manifest.engineering_run_id
        training_run_id = manifest.training_session_id or ""
        test_results: Dict[str, AlgorithmEvaluationResult] = {}

        total_algos = len(manifest.frozen_candidates)
        current_idx = 0

        # Evaluate each frozen candidate model
        for algo_key, freeze_rec in manifest.frozen_candidates.items():
            current_idx += 1
            if progress_callback:
                progress_callback(
                    current_idx,
                    total_algos,
                    f"Evaluating held-out test partition for {algo_key.upper()} ({current_idx}/{total_algos})...",
                )

            # Verify model artifact checksum against frozen candidate record
            check = self.verifier.verify_model_eligibility(
                training_run_id=freeze_rec.training_run_id,
                algorithm=algo_key,
                allow_test_fixtures=manifest.is_test_fixture,
            )
            if not check.is_eligible:
                raise ValueError(
                    f"CRITICAL CHECKSUM MISMATCH: Model '{algo_key}' failed eligibility check: {check.rejection_reasons}"
                )

            if check.model_sha256 != freeze_rec.model_checksum:
                raise ValueError(
                    f"CRITICAL CHECKSUM MISMATCH: Model '{algo_key}' artifact was modified after freezing! "
                    f"Frozen: {freeze_rec.model_checksum}, Current: {check.model_sha256}"
                )

            # Load test partition
            test_part = self.data_loader.load_partition(
                engineering_run_id=eng_run_id,
                partition="test",
                algorithm=algo_key,
            )
            manifest.dataset_checksums["test_matrix"] = test_part.matrix_checksum
            manifest.dataset_checksums["test_labels"] = test_part.labels_checksum

            # Resource preflight
            preflight = EvaluationResourceManager.perform_preflight_check(
                num_samples=test_part.num_samples,
                num_features=test_part.num_features,
                num_classes=test_part.num_classes,
                target_path=self.eval_dir,
            )

            # Load model estimator safely
            estimator = self.model_registry.load_model(freeze_rec.training_run_id, algo_key)

            # Run inference
            runner = InferenceRunner(batch_size=preflight.batch_size)
            inf_res = runner.run_inference(
                estimator=estimator,
                algorithm=algo_key,
                X=test_part.X,
                y_true=test_part.y,
                classes=test_part.classes,
                progress_callback=progress_callback,
            )

            # Metrics
            agg_metrics = MetricsCalculator.calculate_aggregate_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
            )

            per_class_metrics = MetricsCalculator.calculate_per_class_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=test_part.classes,
            )

            cm_record = ConfusionMatrixEngine.compute_matrix(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=test_part.classes,
            )

            roc_results, macro_auc = ROCAnalyzer.compute_ovr_roc(
                y_true=inf_res.y_true,
                scores=inf_res.scores,
                classes=test_part.classes,
                score_type=inf_res.score_type,
            )

            pr_results, macro_ap = PrecisionRecallAnalyzer.compute_ovr_pr(
                y_true=inf_res.y_true,
                scores=inf_res.scores,
                classes=test_part.classes,
                score_type=inf_res.score_type,
            )

            sec_metrics = SecurityMetricsEngine.compute_security_metrics(
                y_true=inf_res.y_true,
                y_pred=inf_res.y_pred,
                classes=test_part.classes,
            )

            algo_result = AlgorithmEvaluationResult(
                algorithm=algo_key,
                algorithm_display_name=freeze_rec.algorithm_display_name,
                training_run_id=freeze_rec.training_run_id,
                model_checksum=check.model_sha256,
                partition="test",
                evaluated_rows=inf_res.successfully_predicted_rows,
                failed_rows=inf_res.failed_rows,
                duration_seconds=inf_res.duration_seconds,
                ms_per_record=inf_res.ms_per_record,
                score_type=inf_res.score_type,
                aggregate_metrics=agg_metrics,
                per_class_metrics=per_class_metrics,
                confusion_matrix=cm_record,
                roc_analysis=roc_results,
                macro_roc_auc=macro_auc,
                pr_analysis=pr_results,
                macro_average_precision=macro_ap,
                security_metrics=sec_metrics,
                warnings=inf_res.warnings + preflight.warning_messages,
                error_message=None,
                is_test_fixture=manifest.is_test_fixture,
            )

            test_results[algo_key] = algo_result

        # Update manifest
        confirm_time = datetime.now(timezone.utc).isoformat()
        manifest.test_gate_confirmed = True
        manifest.test_gate_confirmed_at_utc = confirm_time
        manifest.test_holdout_locked = True
        manifest.test_holdout_locked_at_utc = confirm_time
        manifest.candidate_set_hash = manifest.candidate_set_hash or compute_candidate_set_hash(manifest.frozen_candidates)
        manifest.test_gate_confirmation_notes = notes
        manifest.test_results = test_results
        if "test" not in manifest.partitions_evaluated:
            manifest.partitions_evaluated.append("test")

        manifest.notes.append(
            f"Final test holdout evaluation executed at {confirm_time} across {len(test_results)} frozen models. "
            "Test holdout locked permanently."
        )

        # Persist test holdout lock in registry
        primary_cand = next((k for k, v in manifest.frozen_candidates.items() if v.selected_as_primary), "unknown")
        self.eval_registry.acquire_test_lock(
            engineering_run_id=manifest.engineering_run_id,
            evaluation_run_id=manifest.evaluation_run_id,
            candidate_checksums={k: v.model_checksum for k, v in manifest.frozen_candidates.items()},
            primary_candidate=primary_cand,
            candidate_set_hash=manifest.candidate_set_hash,
            is_test_fixture=manifest.is_test_fixture,
        )

        self.eval_registry.save_evaluation_run(manifest)
        return manifest
