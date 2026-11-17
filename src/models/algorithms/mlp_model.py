"""Multilayer Perceptron Neural Network Trainer (CP-06).

Implements feedforward artificial neural network classification via sklearn.neural_network.MLPClassifier.
Supports resource-aware incremental mini-batch training via partial_fit() directly on
scaled SciPy CSR sparse matrices without dense memory conversion.
Captures actual epoch/batch progression and training loss trajectories.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from src.models.model_config import MLPTrainingConfig
from src.models.training_progress import TrainingProgressTracker, TrainingStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLPTrainingResult:
    """Outcome of Multilayer Perceptron neural network model training."""

    estimator: MLPClassifier
    status: str  # "Completed", "Completed with Warnings", "Failed"
    convergence_status: str  # "Converged", "Iteration Limit Reached", "Loss Stagnated", "Failed"
    total_epochs_trained: int
    total_batches_processed: int
    loss_history: List[float]
    final_loss: float
    duration_seconds: float
    num_samples: int
    num_features: int
    classes: List[str]
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class MLPTrainer:
    """Trains Multilayer Perceptron on scaled sparse CSR representations."""

    def __init__(self, config: Optional[MLPTrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or MLPTrainingConfig()
        self.config.validate()

    def train(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        classes: Optional[List[str]] = None,
        progress_tracker: Optional[TrainingProgressTracker] = None,
    ) -> MLPTrainingResult:
        """Fit MLPClassifier exclusively on training partition records.

        Uses incremental mini-batch training with partial_fit() to bound memory consumption
        and provide authentic epoch-by-epoch loss tracking.

        Args:
            X_train: CP-05 Scaled sparse CSR training feature matrix.
            y_train: Integer target labels array.
            classes: Optional human-readable class names.
            progress_tracker: Optional progress tracker for UI telemetry.

        Returns:
            MLPTrainingResult with fitted estimator, loss history, and diagnostics.
        """
        if not isinstance(X_train, sparse.csr_matrix):
            if sparse.issparse(X_train):
                X_train = X_train.tocsr()
            else:
                X_train = sparse.csr_matrix(X_train)

        n_samples, n_features = X_train.shape
        cfg = self.config

        all_unique_classes = np.unique(y_train)
        n_classes = len(all_unique_classes)
        resolved_classes = classes or [str(c) for c in all_unique_classes]

        if progress_tracker:
            progress_tracker.update(
                stage=TrainingStage.STARTING,
                message=(
                    f"Initializing MLP ({cfg.hidden_layer_sizes}, {cfg.activation}, {cfg.solver}) "
                    f"for {n_samples:,} samples, batch_size={cfg.batch_size}..."
                ),
                progress_percentage=0.0,
            )

        estimator = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            activation=cfg.activation,
            solver=cfg.solver,
            alpha=cfg.alpha,
            batch_size=cfg.batch_size,
            learning_rate_init=cfg.learning_rate_init,
            max_iter=cfg.max_epochs,
            random_state=cfg.random_state,
            tol=cfg.tol,
            early_stopping=False,  # Strict CP-06 rule: no validation early stopping
        )

        captured_warnings: List[str] = []
        loss_history: List[float] = []
        start_time = time.perf_counter()

        # Incremental mini-batch training loop
        if cfg.use_incremental and n_samples > cfg.batch_size:
            total_batches_per_epoch = int(np.ceil(n_samples / float(cfg.batch_size)))
            total_batches = 0
            is_first_call = True
            consecutive_no_change = 0
            stopping_criterion_met = False
            total_epochs = 0

            try:
                for epoch in range(1, cfg.max_epochs + 1):
                    total_epochs = epoch
                    # Deterministic batch permutation per epoch
                    rng = np.random.default_rng(cfg.random_state + epoch)
                    permutation = rng.permutation(n_samples)

                    epoch_losses = []

                    for batch_idx in range(total_batches_per_epoch):
                        batch_start = batch_idx * cfg.batch_size
                        batch_end = min(n_samples, batch_start + cfg.batch_size)
                        indices = permutation[batch_start:batch_end]

                        # Safe sparse slicing
                        X_batch = X_train[indices]
                        y_batch = y_train[indices]

                        # First partial_fit must register all possible classes
                        if is_first_call:
                            estimator.partial_fit(X_batch, y_batch, classes=all_unique_classes)
                            is_first_call = False
                        else:
                            estimator.partial_fit(X_batch, y_batch)

                        total_batches += 1
                        if hasattr(estimator, "loss_") and estimator.loss_ is not None:
                            epoch_losses.append(float(estimator.loss_))

                    # End of epoch telemetry: calculate true average mini-batch loss
                    current_loss = float(np.mean(epoch_losses)) if epoch_losses else float(
                        getattr(estimator, "loss_", 0.0)
                    )
                    loss_history.append(current_loss)

                    # Mathematical stopping criterion verification: check loss improvement against tol
                    if len(loss_history) > 1:
                        loss_delta = loss_history[-2] - current_loss
                        if loss_delta < cfg.tol:
                            consecutive_no_change += 1
                        else:
                            consecutive_no_change = 0

                        if consecutive_no_change >= cfg.n_iter_no_change:
                            stopping_criterion_met = True
                            logger.info(
                                f"MLP convergence criterion satisfied at epoch {epoch}: "
                                f"loss delta < {cfg.tol} for {cfg.n_iter_no_change} consecutive epochs."
                            )

                    pct = (epoch / float(cfg.max_epochs)) * 100.0
                    if progress_tracker:
                        progress_tracker.update(
                            stage=TrainingStage.TRAINING,
                            message=f"Epoch {epoch}/{cfg.max_epochs} complete. Loss: {current_loss:.4f}",
                            progress_percentage=pct,
                            current_epoch=epoch,
                            total_epochs=cfg.max_epochs,
                            current_batch=total_batches,
                            total_batches=total_batches_per_epoch * cfg.max_epochs,
                            current_loss=current_loss,
                            diagnostics={"loss_history": list(loss_history)},
                        )

                    if stopping_criterion_met:
                        break

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"MLP incremental partial_fit failed: {e}")
                if progress_tracker:
                    progress_tracker.mark_failed(str(e))
                return MLPTrainingResult(
                    estimator=estimator,
                    status="Failed",
                    convergence_status="Failed",
                    total_epochs_trained=len(loss_history),
                    total_batches_processed=total_batches,
                    loss_history=loss_history,
                    final_loss=loss_history[-1] if loss_history else 0.0,
                    duration_seconds=round(elapsed, 2),
                    num_samples=n_samples,
                    num_features=n_features,
                    classes=resolved_classes,
                    error_message=str(e),
                )

            final_loss = loss_history[-1] if loss_history else 0.0

        else:
            # Full partition fit fallback for small datasets or explicit full fit
            if progress_tracker:
                progress_tracker.update(
                    stage=TrainingStage.TRAINING,
                    message=f"Training MLP across full partition ({n_samples:,} rows)...",
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

                total_epochs = getattr(estimator, "n_iter_", cfg.max_epochs)
                final_loss = float(getattr(estimator, "loss_", 0.0))
                loss_history = [float(l) for l in getattr(estimator, "loss_curve_", [final_loss])]
                total_batches = total_epochs

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"MLP full fit failed: {e}")
                if progress_tracker:
                    progress_tracker.mark_failed(str(e))
                return MLPTrainingResult(
                    estimator=estimator,
                    status="Failed",
                    convergence_status="Failed",
                    total_epochs_trained=0,
                    total_batches_processed=0,
                    loss_history=[],
                    final_loss=0.0,
                    duration_seconds=round(elapsed, 2),
                    num_samples=n_samples,
                    num_features=n_features,
                    classes=resolved_classes,
                    error_message=str(e),
                )

        elapsed = time.perf_counter() - start_time
        has_warnings = len(captured_warnings) > 0

        # Authentic convergence semantics derivation: distinguish stopping criterion vs epoch limit vs warning
        if cfg.use_incremental and n_samples > cfg.batch_size:
            if stopping_criterion_met:
                status = "Completed"
                convergence_status = "Converged (Loss Tolerance Met)"
            else:
                status = "Completed"
                convergence_status = "Max Epochs Reached without Convergence"
        else:
            if has_warnings:
                status = "Completed with Warnings"
                convergence_status = "Maximum Iterations Reached (Convergence Warning)"
            elif total_epochs < cfg.max_epochs:
                status = "Completed"
                convergence_status = "Converged (Loss Tolerance Met)"
            else:
                status = "Completed"
                convergence_status = "Max Epochs Reached without Convergence"
        log_msg = (
            f"MLP training completed in {elapsed:.2f}s across {total_epochs} epochs. "
            f"Final loss: {final_loss:.4f}."
        )
        logger.info(log_msg)

        diagnostics = {
            "hidden_layer_sizes": list(cfg.hidden_layer_sizes),
            "activation": cfg.activation,
            "solver": cfg.solver,
            "alpha": cfg.alpha,
            "batch_size": cfg.batch_size,
            "total_epochs": total_epochs,
            "total_batches": total_batches,
            "final_loss": round(final_loss, 5),
            "loss_history": [round(l, 5) for l in loss_history[-20:]],  # Last 20 for brevity
            "n_layers": getattr(estimator, "n_layers_", len(cfg.hidden_layer_sizes) + 2),
            "n_outputs": getattr(estimator, "n_outputs_", n_classes),
        }

        if progress_tracker:
            progress_tracker.mark_completed(
                message=log_msg,
                has_warnings=has_warnings,
                diagnostics=diagnostics,
            )

        return MLPTrainingResult(
            estimator=estimator,
            status=status,
            convergence_status=convergence_status,
            total_epochs_trained=total_epochs,
            total_batches_processed=total_batches,
            loss_history=loss_history,
            final_loss=round(final_loss, 5),
            duration_seconds=round(elapsed, 2),
            num_samples=n_samples,
            num_features=n_features,
            classes=resolved_classes,
            warnings=captured_warnings,
            diagnostics=diagnostics,
        )
