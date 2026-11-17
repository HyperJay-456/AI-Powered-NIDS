"""Training Orchestrator Engine (CP-06).

Coordinates the sequential, resource-aware training of the three authorized algorithms:
  - Mode A: Individual Algorithm Training (SVM, MLP, or Decision Tree)
  - Mode B: Sequential Training of All Three Models (Strictly sequential, no concurrency)
Enforces:
  - Exclusive single-job execution (no simultaneous training processes)
  - Strict zero-leakage training partition loading (validation/test remain untouched)
  - Preflight resource checks before fitting
  - Independent algorithm failure isolation
  - Atomic persistence in the model registry
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from config.settings import settings
from src.models.algorithms.decision_tree_model import DecisionTreeTrainer, DecisionTreeTrainingResult
from src.models.algorithms.mlp_model import MLPTrainer, MLPTrainingResult
from src.models.algorithms.svm_model import SVMTrainer, SVMTrainingResult
from src.models.artifact_loader import CP05ArtifactLoader, LoadedTrainingData
from src.models.model_config import (
    AuthorizedAlgorithm,
    DecisionTreeTrainingConfig,
    MasterTrainingConfiguration,
    MLPTrainingConfig,
    SVMTrainingConfig,
)
from src.models.model_factory import ModelFactory
from src.models.model_registry import ModelRegistry
from src.models.resource_manager import ResourceManager
from src.models.training_manifest import (
    AlgorithmTrainingManifest,
    capture_training_environment,
)
from src.models.training_progress import (
    TrainingProgressTracker,
    TrainingProgressUpdate,
    TrainingStage,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """Master orchestrator for model training execution and registry persistence."""

    # Class-level lock ensuring only one training job executes at any given moment
    _job_lock = threading.Lock()
    _active_job_id: Optional[str] = None

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        saved_models_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ):
        """Initialize orchestrator with directories."""
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.saved_dir = saved_models_dir or settings.saved_models_dir
        self.raw_dir = raw_dir or settings.raw_data_dir

        self.loader = CP05ArtifactLoader(processed_dir=self.processed_dir, raw_dir=self.raw_dir)
        self.registry = ModelRegistry(saved_models_dir=self.saved_dir)

    @classmethod
    def is_training_active(cls) -> bool:
        """Check whether a training job is currently running."""
        return cls._active_job_id is not None

    @classmethod
    def get_active_job_id(cls) -> Optional[str]:
        """Return the ID of the active training job if any."""
        return cls._active_job_id

    def train_single_model(
        self,
        engineering_run_id: str,
        algorithm: str,
        config: Optional[Any] = None,
        progress_callback: Optional[Callable[[TrainingProgressUpdate], None]] = None,
    ) -> AlgorithmTrainingManifest:
        """Execute training for a single authorized algorithm.

        Args:
            engineering_run_id: CP-05 source run ID.
            algorithm: One of 'svm', 'mlp', 'decision_tree'.
            config: Algorithm-specific configuration object.
            progress_callback: Optional UI progress callback.

        Returns:
            AlgorithmTrainingManifest with complete training results and provenance.
        """
        algo_key = algorithm.lower().strip()
        if algo_key not in AuthorizedAlgorithm.list_authorized():
            raise ValueError(
                f"Unauthorized algorithm '{algorithm}'. Must be one of {AuthorizedAlgorithm.list_authorized()}."
            )

        if config is None:
            config = ModelFactory.get_default_config(algo_key)

        if not self._job_lock.acquire(blocking=False):
            active_id = self._active_job_id or "unknown"
            raise RuntimeError(
                f"A training job is already actively executing (Job ID: {active_id}). "
                "Simultaneous training jobs are prohibited to preserve hardware stability."
            )

        training_run_id = self.registry.generate_training_run_id()
        self.__class__._active_job_id = training_run_id
        start_time_utc = datetime.now(timezone.utc).isoformat()
        tracker = TrainingProgressTracker(algorithm=algo_key, callback=progress_callback)

        try:
            tracker.update(
                stage=TrainingStage.PREFLIGHT,
                message=f"Verifying integrity of CP-05 engineering run '{engineering_run_id}'...",
                progress_percentage=0.0,
            )

            # 1. Verify and load training partition exclusively
            data: LoadedTrainingData = self.loader.load_training_partition_only(
                run_id=engineering_run_id,
                algorithm=algo_key,
            )

            # 2. Resource Preflight
            preflight = ResourceManager.preflight_check(
                algorithm=algo_key,
                X=data.X,
                y=data.y,
                config=config,
            )

            if not preflight.is_safe:
                tracker.mark_failed(preflight.rejection_reason or "Resource preflight check failed.")
                raise MemoryError(preflight.rejection_reason)

            # 3. Instantiate trainer and execute fitting
            trainer = ModelFactory.create_trainer(algorithm=algo_key, config=config)
            result = trainer.train(
                X_train=data.X,
                y_train=data.y,
                classes=data.classes,
                progress_tracker=tracker,
            )

            now_utc = datetime.now(timezone.utc).isoformat()
            algo_manifest = AlgorithmTrainingManifest(
                training_run_id=training_run_id,
                algorithm=algo_key,
                algorithm_display_name=AuthorizedAlgorithm.get_display_name(algo_key),
                estimator_class=result.estimator.__class__.__module__ + "." + result.estimator.__class__.__name__,
                status=result.status,
                engineering_run_id=engineering_run_id,
                source_dataset_names=data.source_filenames,
                source_checksums=data.source_checksums,
                training_partition_checksum=data.training_partition_checksum,
                feature_schema_checksum=data.feature_schema_checksum,
                label_mapping_checksum=data.label_mapping_checksum,
                training_samples=data.num_samples,
                feature_count=data.num_features,
                class_count=data.num_classes,
                classes=data.classes,
                input_representation=f"{data.representation_type.title()} Sparse CSR",
                random_seed=getattr(config, "random_state", 42),
                hyperparameters=config.to_dict() if hasattr(config, "to_dict") else asdict(config),
                training_duration_seconds=result.duration_seconds,
                started_at_utc=start_time_utc,
                completed_at_utc=now_utc,
                convergence_status=getattr(result, "convergence_status", "Converged"),
                solver_iterations=getattr(result, "solver_iterations", None),
                diagnostics=result.diagnostics,
                warnings=result.warnings,
                error_message=result.error_message,
                preprocessing_artifact_references=[
                    f"data/processed/{engineering_run_id}/metadata/feature_schema.json",
                    f"data/processed/{engineering_run_id}/metadata/label_mapping.json",
                    f"data/processed/{engineering_run_id}/preprocessors/label_processor.joblib",
                ],
                environment=capture_training_environment(),
            )

            # 4. Atomically persist to registry if successful
            if result.status in ("Completed", "Completed with Warnings"):
                self.registry.save_training_session(
                    training_run_id=training_run_id,
                    engineering_run_id=engineering_run_id,
                    results_by_algo={algo_key: result},
                    training_manifests={algo_key: algo_manifest},
                    started_at_utc=start_time_utc,
                )

            return algo_manifest

        finally:
            self.__class__._active_job_id = None
            self._job_lock.release()

    def train_all_models_sequentially(
        self,
        engineering_run_id: str,
        master_config: Optional[MasterTrainingConfiguration] = None,
        progress_callback: Optional[Callable[[TrainingProgressUpdate], None]] = None,
    ) -> Dict[str, AlgorithmTrainingManifest]:
        """Train all three authorized algorithms sequentially.

        Sequence:
          1. Linear Support Vector Machine (LinearSVC)
          2. Multilayer Perceptron (MLPClassifier)
          3. Decision Tree (DecisionTreeClassifier)

        Preserves independent failure isolation: a failure in one estimator does NOT
        corrupt or abort previously succeeded models.

        Args:
            engineering_run_id: CP-05 source run ID.
            master_config: Configuration container for all algorithms.
            progress_callback: Optional UI progress callback.

        Returns:
            Dictionary mapping algorithm keys to their individual training manifests.
        """
        if not self._job_lock.acquire(blocking=False):
            active_id = self._active_job_id or "unknown"
            raise RuntimeError(
                f"A training job is already actively executing (Job ID: {active_id}). "
                "Simultaneous training jobs are prohibited to preserve hardware stability."
            )

        training_run_id = self.registry.generate_training_run_id()
        self.__class__._active_job_id = training_run_id
        session_start_time = datetime.now(timezone.utc).isoformat()

        cfg = master_config or MasterTrainingConfiguration(engineering_run_id=engineering_run_id)

        algorithms_sequence = [
            ("svm", cfg.svm_config),
            ("mlp", cfg.mlp_config),
            ("decision_tree", cfg.decision_tree_config),
        ]

        results_by_algo: Dict[str, Any] = {}
        manifests_by_algo: Dict[str, AlgorithmTrainingManifest] = {}

        try:
            for algo_key, algo_cfg in algorithms_sequence:
                tracker = TrainingProgressTracker(algorithm=algo_key, callback=progress_callback)
                algo_start_time = datetime.now(timezone.utc).isoformat()

                try:
                    tracker.update(
                        stage=TrainingStage.STARTING,
                        message=f"Starting sequential training for {AuthorizedAlgorithm.get_display_name(algo_key)}...",
                        progress_percentage=0.0,
                    )

                    # 1. Load algorithm-appropriate training partition exclusively
                    data = self.loader.load_training_partition_only(
                        run_id=engineering_run_id,
                        algorithm=algo_key,
                    )

                    # 2. Preflight check
                    preflight = ResourceManager.preflight_check(
                        algorithm=algo_key,
                        X=data.X,
                        y=data.y,
                        config=algo_cfg,
                        safety_margin_percent=cfg.safety_margin_percent,
                    )

                    if not preflight.is_safe:
                        tracker.mark_failed(preflight.rejection_reason or "Resource preflight check failed.")
                        continue

                    # 3. Fit trainer
                    trainer = ModelFactory.create_trainer(algorithm=algo_key, config=algo_cfg)
                    result = trainer.train(
                        X_train=data.X,
                        y_train=data.y,
                        classes=data.classes,
                        progress_tracker=tracker,
                    )

                    results_by_algo[algo_key] = result
                    algo_end_time = datetime.now(timezone.utc).isoformat()

                    manifest = AlgorithmTrainingManifest(
                        training_run_id=training_run_id,
                        algorithm=algo_key,
                        algorithm_display_name=AuthorizedAlgorithm.get_display_name(algo_key),
                        estimator_class=result.estimator.__class__.__module__ + "." + result.estimator.__class__.__name__,
                        status=result.status,
                        engineering_run_id=engineering_run_id,
                        source_dataset_names=data.source_filenames,
                        source_checksums=data.source_checksums,
                        training_partition_checksum=data.training_partition_checksum,
                        feature_schema_checksum=data.feature_schema_checksum,
                        label_mapping_checksum=data.label_mapping_checksum,
                        training_samples=data.num_samples,
                        feature_count=data.num_features,
                        class_count=data.num_classes,
                        classes=data.classes,
                        input_representation=f"{data.representation_type.title()} Sparse CSR",
                        random_seed=getattr(algo_cfg, "random_state", 42),
                        hyperparameters=algo_cfg.to_dict() if hasattr(algo_cfg, "to_dict") else asdict(algo_cfg),
                        training_duration_seconds=result.duration_seconds,
                        started_at_utc=algo_start_time,
                        completed_at_utc=algo_end_time,
                        convergence_status=getattr(result, "convergence_status", "Converged"),
                        solver_iterations=getattr(result, "solver_iterations", None),
                        diagnostics=result.diagnostics,
                        warnings=result.warnings,
                        error_message=result.error_message,
                        preprocessing_artifact_references=[
                            f"data/processed/{engineering_run_id}/metadata/feature_schema.json",
                            f"data/processed/{engineering_run_id}/metadata/label_mapping.json",
                            f"data/processed/{engineering_run_id}/preprocessors/label_processor.joblib",
                        ],
                        environment=capture_training_environment(),
                    )
                    manifests_by_algo[algo_key] = manifest

                except Exception as e:
                    logger.error(f"Error during sequential training of '{algo_key}': {e}")
                    tracker.mark_failed(str(e))

            # 4. Atomically persist all successfully fitted models
            if any(r.status in ("Completed", "Completed with Warnings") for r in results_by_algo.values()):
                self.registry.save_training_session(
                    training_run_id=training_run_id,
                    engineering_run_id=engineering_run_id,
                    results_by_algo=results_by_algo,
                    training_manifests=manifests_by_algo,
                    started_at_utc=session_start_time,
                )

            return manifests_by_algo

        finally:
            self.__class__._active_job_id = None
            self._job_lock.release()
