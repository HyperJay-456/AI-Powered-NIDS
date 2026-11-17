"""Batch-Based Inference Runner & Score Generator (CP-07).

Executes batch-based inference on sparse CSR feature matrices without dense expansion:
1. Batches input matrix to strictly bound working memory.
2. Extracts discrete predictions via estimator.predict().
3. Extracts raw decision scores or probabilities:
   - LinearSVC: decision_function() signed distance margins.
   - MLPClassifier: predict_proba() uncalibrated output probabilities.
   - DecisionTreeClassifier: predict_proba() leaf-node class distribution.
4. Aligns score columns with estimator.classes_ and canonical label mapping.
5. Measures total inference time and milliseconds per record.
6. Strictly verifies the row invariant:
   Total Rows = Successfully Predicted Rows + Failed Rows
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BatchInferenceResult:
    """Consolidated outcome of batch inference execution."""

    y_true: np.ndarray
    y_pred: np.ndarray
    scores: Optional[np.ndarray]  # Shape (n_samples, n_classes) or None
    score_type: str  # "decision_function", "predict_proba", or "none"
    total_rows: int
    successfully_predicted_rows: int
    failed_rows: int
    duration_seconds: float
    ms_per_record: float
    classes: List[str]
    estimator_classes: List[Any]
    warnings: List[str] = field(default_factory=list)


class InferenceRunner:
    """Executes chunked batch inference on sparse matrices while preserving memory bounds."""

    def __init__(self, batch_size: int = 10_000):
        """Initialize runner with batch size."""
        self.batch_size = max(100, batch_size)

    def run_inference(
        self,
        estimator: Any,
        algorithm: str,
        X: sparse.csr_matrix,
        y_true: np.ndarray,
        classes: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchInferenceResult:
        """Execute chunked prediction across all rows of the sparse feature matrix.

        Args:
            estimator: Fitted scikit-learn model object.
            algorithm: Algorithm key ('svm', 'mlp', 'decision_tree').
            X: Input sparse CSR feature matrix.
            y_true: True ground-truth label indices.
            classes: Canonical class names list from label mapping.
            progress_callback: Optional callback(processed, total, status_msg).

        Returns:
            BatchInferenceResult containing predictions, scores, timing, and row accounting.
        """
        algo_key = algorithm.lower().strip()
        n_samples = X.shape[0]
        n_classes = len(classes)
        warnings: List[str] = []

        if n_samples != len(y_true):
            raise ValueError(
                f"Row mismatch: X has {n_samples} rows, y_true has {len(y_true)} items."
            )

        # Inspect estimator capabilities
        has_decision_function = hasattr(estimator, "decision_function")
        has_predict_proba = hasattr(estimator, "predict_proba")
        est_classes = list(getattr(estimator, "classes_", range(n_classes)))

        score_type = "none"
        if algo_key in ("svm", "linearsvc"):
            score_type = "decision_function" if has_decision_function else "none"
        elif algo_key in ("mlp", "decision_tree", "decisiontree"):
            score_type = "predict_proba" if has_predict_proba else "none"

        # Preallocate prediction array
        y_pred = np.empty(n_samples, dtype=np.int64)
        scores: Optional[np.ndarray] = None
        if score_type != "none":
            scores = np.empty((n_samples, n_classes), dtype=np.float64)

        successfully_predicted = 0
        failed_rows = 0

        start_time = time.perf_counter()

        # Batch loop
        num_batches = int(np.ceil(n_samples / self.batch_size))
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_slice = slice(start_idx, end_idx)
            batch_count = end_idx - start_idx

            X_batch = X[batch_slice]

            try:
                # 1. Discrete class predictions
                pred_batch = estimator.predict(X_batch)
                y_pred[batch_slice] = pred_batch
                successfully_predicted += batch_count

                # 2. Continuous scores (decision function or probability)
                if score_type == "decision_function":
                    df_out = estimator.decision_function(X_batch)
                    if df_out.ndim == 1:
                        # Binary classification: decision_function returns 1D array of distance to hyperplane
                        # Column 0 = -score (class 0 margin), Column 1 = score (class 1 margin)
                        scores[batch_slice, 0] = -df_out
                        scores[batch_slice, 1] = df_out
                    else:
                        # Multiclass: shape (batch_count, n_classes)
                        if df_out.shape[1] == n_classes:
                            scores[batch_slice, :] = df_out
                        else:
                            # Map columns based on est_classes
                            for col_idx, est_cls in enumerate(est_classes):
                                if est_cls < n_classes:
                                    scores[batch_slice, est_cls] = df_out[:, col_idx]

                elif score_type == "predict_proba":
                    proba_out = estimator.predict_proba(X_batch)
                    if proba_out.shape[1] == n_classes:
                        scores[batch_slice, :] = proba_out
                    else:
                        for col_idx, est_cls in enumerate(est_classes):
                            if est_cls < n_classes:
                                scores[batch_slice, est_cls] = proba_out[:, col_idx]

            except Exception as e:
                logger.error(f"Inference failure on batch [{start_idx}:{end_idx}]: {e}")
                failed_rows += batch_count
                warnings.append(f"Batch [{start_idx}:{end_idx}] inference exception: {e}")

            if progress_callback:
                progress_callback(
                    end_idx,
                    n_samples,
                    f"Evaluating {algorithm.upper()}: {end_idx:,} / {n_samples:,} records processed"
                )

        duration = time.perf_counter() - start_time
        ms_per_record = (duration / n_samples * 1000.0) if n_samples > 0 else 0.0

        # Maintain critical row-accounting invariant:
        # Total Rows = Successfully Predicted Rows + Failed Rows
        if n_samples != (successfully_predicted + failed_rows):
            raise RuntimeError(
                f"Evaluation accounting invariant violated! Total={n_samples}, "
                f"Predicted={successfully_predicted}, Failed={failed_rows}."
            )

        if failed_rows > 0:
            warnings.append(f"{failed_rows:,} records failed during prediction pass.")

        return BatchInferenceResult(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            score_type=score_type,
            total_rows=n_samples,
            successfully_predicted_rows=successfully_predicted,
            failed_rows=failed_rows,
            duration_seconds=duration,
            ms_per_record=ms_per_record,
            classes=classes,
            estimator_classes=est_classes,
            warnings=warnings,
        )
