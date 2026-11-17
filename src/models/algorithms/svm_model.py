"""Linear Support Vector Machine Trainer (CP-06).

Implements scalable, sparse-compatible Linear SVM classification via sklearn.svm.LinearSVC.
Operates strictly on the CP-05 scaled sparse CSR training partition.
Captures convergence warnings, actual solver iterations, and execution duration.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from src.models.model_config import SVMTrainingConfig
from src.models.training_progress import TrainingProgressTracker, TrainingStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SVMTrainingResult:
    """Outcome of Linear SVM model training."""

    estimator: LinearSVC
    status: str  # "Completed", "Completed with Warnings", "Failed"
    convergence_status: str  # "Converged", "Iteration Limit Reached (Convergence Warning)", "Failed"
    solver_iterations: int
    duration_seconds: float
    num_samples: int
    num_features: int
    classes: List[str]
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class SVMTrainer:
    """Trains Linear Support Vector Machine on scaled sparse CSR representations."""

    def __init__(self, config: Optional[SVMTrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or SVMTrainingConfig()
        self.config.validate()

    def train(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        classes: Optional[List[str]] = None,
        progress_tracker: Optional[TrainingProgressTracker] = None,
    ) -> SVMTrainingResult:
        """Fit LinearSVC exclusively on training partition records.

        Args:
            X_train: CP-05 Scaled sparse CSR training feature matrix.
            y_train: Integer target labels array.
            classes: Optional human-readable class names.
            progress_tracker: Optional progress tracker for UI telemetry.

        Returns:
            SVMTrainingResult with fitted estimator, convergence telemetry, and diagnostics.
        """
        if not isinstance(X_train, sparse.csr_matrix):
            if sparse.issparse(X_train):
                X_train = X_train.tocsr()
            else:
                X_train = sparse.csr_matrix(X_train)

        n_samples, n_features = X_train.shape
        cfg = self.config

        if progress_tracker:
            progress_tracker.update(
                stage=TrainingStage.STARTING,
                message=f"Initializing LinearSVC with C={cfg.C}, max_iter={cfg.max_iter:,}...",
                progress_percentage=0.0,
            )

        # In scikit-learn LinearSVC, auto dual selects dual=False when n_samples > n_features
        estimator = LinearSVC(
            C=cfg.C,
            penalty=cfg.penalty,
            loss=cfg.loss,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            dual=cfg.dual,
        )

        captured_warnings: List[str] = []
        start_time = time.perf_counter()

        if progress_tracker:
            progress_tracker.update(
                stage=TrainingStage.TRAINING,
                message=f"Optimizing hyperplane margins across {n_samples:,} flows and {n_features} features...",
                progress_percentage=50.0,
            )

        try:
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X_train, y_train)

                for w in warn_list:
                    msg = str(w.message)
                    if issubclass(w.category, ConvergenceWarning):
                        captured_warnings.append(f"ConvergenceWarning: {msg}")
                    else:
                        captured_warnings.append(f"{w.category.__name__}: {msg}")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"LinearSVC fitting failed: {e}")
            if progress_tracker:
                progress_tracker.mark_failed(str(e))
            return SVMTrainingResult(
                estimator=estimator,
                status="Failed",
                convergence_status="Failed",
                solver_iterations=0,
                duration_seconds=round(elapsed, 2),
                num_samples=n_samples,
                num_features=n_features,
                classes=classes or [str(c) for c in np.unique(y_train)],
                error_message=str(e),
            )

        elapsed = time.perf_counter() - start_time

        # Extract solver iterations: LinearSVC stores n_iter_ (array of ints in multi-class)
        raw_iters = getattr(estimator, "n_iter_", None)
        if raw_iters is not None:
            if isinstance(raw_iters, np.ndarray):
                actual_iters = int(np.max(raw_iters))
            else:
                actual_iters = int(raw_iters)
        else:
            actual_iters = cfg.max_iter

        has_convergence_warning = any("convergence" in w.lower() for w in captured_warnings)

        if has_convergence_warning:
            status = "Completed with Warnings"
            convergence_status = "Iteration Limit Reached (Convergence Warning)"
            log_msg = f"LinearSVC finished with convergence warnings after {actual_iters:,} iterations ({elapsed:.2f}s)."
        else:
            status = "Completed"
            convergence_status = "Converged"
            log_msg = f"LinearSVC converged successfully in {actual_iters:,} iterations ({elapsed:.2f}s)."

        logger.info(log_msg)

        resolved_classes = classes or [str(c) for c in getattr(estimator, "classes_", np.unique(y_train))]

        diagnostics = {
            "solver_iterations": actual_iters,
            "max_iter_configured": cfg.max_iter,
            "regularization_C": cfg.C,
            "loss_function": cfg.loss,
            "penalty": cfg.penalty,
            "intercept_shape": list(getattr(estimator, "intercept_", np.array([])).shape),
            "coef_shape": list(getattr(estimator, "coef_", np.array([])).shape),
        }

        if progress_tracker:
            progress_tracker.mark_completed(
                message=log_msg,
                has_warnings=has_convergence_warning,
                diagnostics=diagnostics,
            )

        return SVMTrainingResult(
            estimator=estimator,
            status=status,
            convergence_status=convergence_status,
            solver_iterations=actual_iters,
            duration_seconds=round(elapsed, 2),
            num_samples=n_samples,
            num_features=n_features,
            classes=resolved_classes,
            warnings=captured_warnings,
            diagnostics=diagnostics,
        )
