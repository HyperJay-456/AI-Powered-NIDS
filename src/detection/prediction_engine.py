"""Prediction Execution Engine (CP-08).

Executes batch model inference strictly across the three authorized algorithms:
  1. Support Vector Machine (LinearSVC)
  2. Multilayer Perceptron (MLPClassifier)
  3. Decision Tree (DecisionTreeClassifier)

STRICT INVARIANTS:
  1. Calls predict(), decision_function(), or predict_proba() only
  2. Never calls fit(), fit_transform(), or partial_fit()
  3. Preserves sparse matrix representation during inference
  4. Keeps execution sequential by default to prevent resource exhaustion
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPredictionEngine:
    """Executes model inference on preprocessed sparse matrix chunks."""

    @staticmethod
    def predict_chunk(
        estimator: Any,
        algorithm: str,
        X_chunk: sparse.csr_matrix,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Execute inference on a single sparse batch for an algorithm.

        Args:
            estimator: Fitted scikit-learn estimator instance.
            algorithm: Algorithm key ('svm', 'mlp', 'decision_tree').
            X_chunk: SciPy CSR sparse matrix.

        Returns:
            Tuple of:
              - y_pred: Integer class predictions array (shape N,)
              - scores: Raw score array (margins for LinearSVC, softmax/leaf probs for MLP/DT)
              - score_type: "decision_margin" or "uncalibrated_probabilities"
        """
        algo = algorithm.lower().strip()

        # 1. Prediction
        y_pred = estimator.predict(X_chunk)

        # 2. Extract model-specific raw score representation
        scores: Optional[np.ndarray] = None
        score_type: str = "unknown"

        if algo == "svm":
            # LinearSVC signed geometric margin distances
            if hasattr(estimator, "decision_function"):
                scores = estimator.decision_function(X_chunk)
            score_type = "decision_margin"

        elif algo in ("mlp", "decision_tree"):
            # Softmax outputs or leaf distributions
            if hasattr(estimator, "predict_proba"):
                scores = estimator.predict_proba(X_chunk)
            score_type = "uncalibrated_probabilities"

        return y_pred, scores, score_type
