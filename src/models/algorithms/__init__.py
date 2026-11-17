"""Authorized Model Algorithm Implementations (CP-06).

Exports trainers and result dataclasses for the three approved NIDS algorithms:
  1. SVMTrainer (LinearSVC)
  2. MLPTrainer (MLPClassifier with incremental partial_fit)
  3. DecisionTreeTrainer (DecisionTreeClassifier on unscaled features)
"""

from src.models.algorithms.decision_tree_model import (
    DecisionTreeTrainer,
    DecisionTreeTrainingResult,
)
from src.models.algorithms.mlp_model import MLPTrainer, MLPTrainingResult
from src.models.algorithms.svm_model import SVMTrainer, SVMTrainingResult

__all__ = [
    "SVMTrainer",
    "SVMTrainingResult",
    "MLPTrainer",
    "MLPTrainingResult",
    "DecisionTreeTrainer",
    "DecisionTreeTrainingResult",
]
