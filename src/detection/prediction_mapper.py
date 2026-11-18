"""Prediction Mapper & Attack Label Engine (CP-08).

Maps raw integer model prediction indices to verified human-readable attack classes
using the exact label mappings established in CP-05 and stored in model metadata.
"""

from typing import Dict, List, Optional, Union
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPredictionMapper:
    """Translates model prediction indices to attack classification labels."""

    def __init__(
        self,
        index_to_label: Optional[Dict[int, str]] = None,
        classes: Optional[List[str]] = None,
    ):
        """Initialize mapper with either an index-to-label mapping or class list.

        Args:
            index_to_label: Explicit dictionary mapping integer index to class string.
            classes: Ordered list of class strings where index corresponds to class order.
        """
        self.index_to_label: Dict[int, str] = {}

        if index_to_label:
            # Normalize keys to int
            self.index_to_label = {int(k): str(v) for k, v in index_to_label.items()}
        elif classes:
            self.index_to_label = {i: str(c) for i, c in enumerate(classes)}

        if not self.index_to_label:
            raise ValueError("DetectionPredictionMapper requires a non-empty class mapping.")

        self.label_to_index: Dict[str, int] = {v: k for k, v in self.index_to_label.items()}

    def map_predictions(self, y_pred_indices: Union[np.ndarray, List[int]]) -> List[str]:
        """Convert a batch of integer prediction indices to string labels.

        Args:
            y_pred_indices: Array or list of integer class predictions.

        Returns:
            List of human-readable class names.

        Raises:
            ValueError: If any prediction index is not present in the verified mapping.
        """
        labels: List[str] = []
        unmapped_indices = set()

        for idx in y_pred_indices:
            idx_int = int(idx)
            if idx_int in self.index_to_label:
                labels.append(self.index_to_label[idx_int])
            else:
                unmapped_indices.add(idx_int)

        if unmapped_indices:
            raise ValueError(
                f"Prediction mapping failed: unmapped class indices {sorted(unmapped_indices)} "
                f"not found in model label mapping: {self.index_to_label}"
            )

        return labels

    def map_single(self, index: int) -> str:
        """Map a single class index to its label."""
        idx_int = int(index)
        if idx_int not in self.index_to_label:
            raise ValueError(f"Unmapped prediction index {idx_int} (valid: {sorted(self.index_to_label.keys())})")
        return self.index_to_label[idx_int]

    @property
    def known_classes(self) -> List[str]:
        """List of all known class labels in index order."""
        return [self.index_to_label[i] for i in sorted(self.index_to_label.keys())]

    @staticmethod
    def is_benign(label: str) -> bool:
        """Determine if a class label represents benign baseline traffic."""
        return label.strip().upper() == "BENIGN"

    @classmethod
    def is_attack(cls, label: str) -> bool:
        """Determine if a class label represents an attack classification."""
        return not cls.is_benign(label)
