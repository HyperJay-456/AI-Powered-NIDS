"""Score Interpreter & Semantic Transparency Module (CP-08).

Provides honest, model-specific interpretation of output scores without misrepresenting
uncalibrated decision margins or raw heuristic distributions as true probabilities.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DetectionScoreInterpreter:
    """Interprets raw model outputs and provides honest descriptions."""

    @staticmethod
    def get_score_metadata(algorithm: str) -> Dict[str, str]:
        """Return standardized descriptor for an algorithm's output scores."""
        algo = algorithm.lower().strip()
        if algo == "svm":
            return {
                "score_type": "decision_margin",
                "score_label": "Decision Margin",
                "score_unit": "signed distance",
                "description": (
                    "LinearSVC signed geometric distance to separating hyperplane. "
                    "Positive values indicate class affinity; not a calibrated probability."
                ),
            }
        elif algo == "mlp":
            return {
                "score_type": "uncalibrated_probabilities",
                "score_label": "Model Confidence Score",
                "score_unit": "softmax output [0.0 - 1.0]",
                "description": (
                    "MLPClassifier output layer softmax distribution. "
                    "Uncalibrated heuristic confidence score; not a guaranteed probability of attack."
                ),
            }
        elif algo == "decision_tree":
            return {
                "score_type": "uncalibrated_probabilities",
                "score_label": "Tree Leaf Frequency",
                "score_unit": "leaf fraction [0.0 - 1.0]",
                "description": (
                    "DecisionTreeClassifier fraction of training samples in terminating leaf node. "
                    "Uncalibrated empirical score; not a calibrated intrusion probability."
                ),
            }
        else:
            return {
                "score_type": "unknown",
                "score_label": "Model Score",
                "score_unit": "raw",
                "description": "Unspecified model output score.",
            }

    @staticmethod
    def extract_winning_scores(
        algorithm: str,
        scores: Optional[np.ndarray],
        predicted_indices: np.ndarray,
    ) -> List[Optional[float]]:
        """Extract winning score for each sample according to its predicted class.

        Args:
            algorithm: Algorithm key ('svm', 'mlp', 'decision_tree').
            scores: Raw score array (decision margins or probability distributions).
            predicted_indices: Array of predicted integer class indices.

        Returns:
            List of float scores for each sample (or None if unavailable).
        """
        if scores is None or len(scores) == 0:
            return [None] * len(predicted_indices)

        winning_scores: List[Optional[float]] = []
        n_samples = len(predicted_indices)

        if scores.ndim == 1:
            # Binary classification 1D margins
            for i in range(n_samples):
                winning_scores.append(float(scores[i]))
        elif scores.ndim == 2:
            # Multiclass OvR margins or probability matrix
            for i in range(n_samples):
                pred_idx = int(predicted_indices[i])
                if 0 <= pred_idx < scores.shape[1]:
                    winning_scores.append(float(scores[i, pred_idx]))
                else:
                    # Fallback to max score
                    winning_scores.append(float(np.max(scores[i])))
        else:
            winning_scores = [None] * n_samples

        return winning_scores

    @staticmethod
    def format_score_display(score: Optional[float], score_type: str) -> str:
        """Format score value for UI presentation with honest units."""
        if score is None:
            return "N/A"
        if score_type == "decision_margin":
            return f"{score:+.3f} margin"
        elif score_type == "uncalibrated_probabilities":
            return f"{score:.3f} score"
        return f"{score:.3f}"
