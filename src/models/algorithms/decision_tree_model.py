"""Decision Tree Classifier Trainer (CP-06).

Implements recursive binary partitioning (CART) classification via sklearn.tree.DecisionTreeClassifier.
Operates strictly on the CP-05 unscaled feature representation to preserve physical network
flow thresholds (byte counts, packet lengths, flow durations) for transparent human inspection.
Captures actual tree depth, leaf count, total node count, and training duration.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional
import warnings

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier

from src.models.model_config import DecisionTreeTrainingConfig
from src.models.training_progress import TrainingProgressTracker, TrainingStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionTreeTrainingResult:
    """Outcome of Decision Tree model training."""

    estimator: DecisionTreeClassifier
    status: str  # "Completed", "Completed with Warnings", "Failed"
    actual_depth: int
    leaf_count: int
    node_count: int
    duration_seconds: float
    num_samples: int
    num_features: int
    classes: List[str]
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class DecisionTreeTrainer:
    """Trains Decision Tree on unscaled physical flow feature representations."""

    def __init__(self, config: Optional[DecisionTreeTrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or DecisionTreeTrainingConfig()
        self.config.validate()

    def train(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        classes: Optional[List[str]] = None,
        progress_tracker: Optional[TrainingProgressTracker] = None,
    ) -> DecisionTreeTrainingResult:
        """Fit DecisionTreeClassifier exclusively on unscaled training partition records.

        Args:
            X_train: CP-05 Unscaled sparse CSR training feature matrix.
            y_train: Integer target labels array.
            classes: Optional human-readable class names.
            progress_tracker: Optional progress tracker for UI telemetry.

        Returns:
            DecisionTreeTrainingResult with fitted tree, structural diagnostics, and duration.
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
                message=f"Initializing Decision Tree (criterion={cfg.criterion}, max_depth={cfg.max_depth})...",
                progress_percentage=0.0,
            )

        estimator = DecisionTreeClassifier(
            criterion=cfg.criterion,
            splitter=cfg.splitter,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            max_leaf_nodes=cfg.max_leaf_nodes,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
        )

        captured_warnings: List[str] = []
        start_time = time.perf_counter()

        if progress_tracker:
            progress_tracker.update(
                stage=TrainingStage.TRAINING,
                message=f"Constructing binary split nodes across {n_samples:,} flows (unscaled representation)...",
                progress_percentage=50.0,
            )

        try:
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always")
                estimator.fit(X_train, y_train)

                for w in warn_list:
                    captured_warnings.append(f"{w.category.__name__}: {w.message}")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"DecisionTreeClassifier fitting failed: {e}")
            if progress_tracker:
                progress_tracker.mark_failed(str(e))
            return DecisionTreeTrainingResult(
                estimator=estimator,
                status="Failed",
                actual_depth=0,
                leaf_count=0,
                node_count=0,
                duration_seconds=round(elapsed, 2),
                num_samples=n_samples,
                num_features=n_features,
                classes=classes or [str(c) for c in np.unique(y_train)],
                error_message=str(e),
            )

        elapsed = time.perf_counter() - start_time

        # Extract structural diagnostics from fitted tree
        actual_depth = int(estimator.get_depth())
        leaf_count = int(estimator.get_n_leaves())
        node_count = int(estimator.tree_.node_count)

        resolved_classes = classes or [str(c) for c in getattr(estimator, "classes_", np.unique(y_train))]

        diagnostics = {
            "actual_depth": actual_depth,
            "max_depth_configured": cfg.max_depth,
            "leaf_count": leaf_count,
            "node_count": node_count,
            "criterion": cfg.criterion,
            "splitter": cfg.splitter,
            "min_samples_split": cfg.min_samples_split,
            "min_samples_leaf": cfg.min_samples_leaf,
            "feature_importances_nonzero_count": int(np.sum(estimator.feature_importances_ > 0)),
        }

        has_warnings = len(captured_warnings) > 0
        status = "Completed with Warnings" if has_warnings else "Completed"
        log_msg = (
            f"Decision Tree trained in {elapsed:.2f}s: depth={actual_depth}, "
            f"leaves={leaf_count:,}, nodes={node_count:,}."
        )
        logger.info(log_msg)

        if progress_tracker:
            progress_tracker.mark_completed(
                message=log_msg,
                has_warnings=has_warnings,
                diagnostics=diagnostics,
            )

        return DecisionTreeTrainingResult(
            estimator=estimator,
            status=status,
            actual_depth=actual_depth,
            leaf_count=leaf_count,
            node_count=node_count,
            duration_seconds=round(elapsed, 2),
            num_samples=n_samples,
            num_features=n_features,
            classes=resolved_classes,
            warnings=captured_warnings,
            diagnostics=diagnostics,
        )
