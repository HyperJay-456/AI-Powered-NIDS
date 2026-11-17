"""Training Progress Tracker & Diagnostic State Machine (CP-06).

Maintains structured progress and diagnostic state throughout algorithm training.
Enforces authentic progress reporting:
  - For MLP: Reports actual completed epochs, batches, and training losses.
  - For SVM and Decision Tree: Reports elapsed time and state transitions without inventing fake percentages.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import enum
import time
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingStage(str, enum.Enum):
    """Lifecycle stages of an algorithm training run."""

    QUEUED = "Queued"
    PREFLIGHT = "Resource Preflight"
    STARTING = "Starting"
    TRAINING = "Training"
    COMPLETED = "Completed"
    COMPLETED_WITH_WARNINGS = "Completed with Warnings"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class TrainingProgressUpdate:
    """Individual telemetry update emitted during training."""

    algorithm: str
    stage: TrainingStage
    progress_percentage: float  # 0.0 to 100.0 (or -1.0 if indeterminable)
    message: str
    elapsed_seconds: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_batch: Optional[int] = None
    total_batches: Optional[int] = None
    current_loss: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TrainingProgressTracker:
    """Tracks training progress and dispatches real-time telemetry to callbacks."""

    def __init__(
        self,
        algorithm: str,
        callback: Optional[Callable[[TrainingProgressUpdate], None]] = None,
    ):
        """Initialize tracker for an algorithm."""
        self.algorithm = algorithm
        self.callback = callback
        self.start_time: float = time.perf_counter()
        self.stage: TrainingStage = TrainingStage.QUEUED
        self.updates: List[TrainingProgressUpdate] = []

    def update(
        self,
        stage: TrainingStage,
        message: str,
        progress_percentage: float = -1.0,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        current_batch: Optional[int] = None,
        total_batches: Optional[int] = None,
        current_loss: Optional[float] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> TrainingProgressUpdate:
        """Emit a progress update and notify listener."""
        elapsed = time.perf_counter() - self.start_time
        self.stage = stage

        update = TrainingProgressUpdate(
            algorithm=self.algorithm,
            stage=stage,
            progress_percentage=progress_percentage,
            message=message,
            elapsed_seconds=round(elapsed, 2),
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            current_batch=current_batch,
            total_batches=total_batches,
            current_loss=current_loss,
            diagnostics=diagnostics or {},
        )
        self.updates.append(update)

        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.debug(f"Progress callback raised exception: {e}")

        return update

    def mark_completed(
        self,
        message: str = "Training completed successfully.",
        has_warnings: bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> TrainingProgressUpdate:
        """Transition state to Completed or Completed with Warnings."""
        stage = TrainingStage.COMPLETED_WITH_WARNINGS if has_warnings else TrainingStage.COMPLETED
        return self.update(
            stage=stage,
            message=message,
            progress_percentage=100.0,
            diagnostics=diagnostics,
        )

    def mark_failed(
        self,
        error_message: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> TrainingProgressUpdate:
        """Transition state to Failed with error details."""
        return self.update(
            stage=TrainingStage.FAILED,
            message=f"Training failed: {error_message}",
            progress_percentage=-1.0,
            diagnostics=diagnostics,
        )
