"""Model Factory for Authorized Algorithms (CP-06).

Instantiates algorithm trainers and configurations strictly enforcing the project constraint
that exactly three algorithms are authorized:
  1. Support Vector Machine (svm)
  2. Multilayer Perceptron (mlp)
  3. Decision Tree (decision_tree)
"""

from typing import Any, Optional, Union

from src.models.algorithms.decision_tree_model import DecisionTreeTrainer
from src.models.algorithms.mlp_model import MLPTrainer
from src.models.algorithms.svm_model import SVMTrainer
from src.models.model_config import (
    AuthorizedAlgorithm,
    DecisionTreeTrainingConfig,
    MLPTrainingConfig,
    SVMTrainingConfig,
)


class ModelFactory:
    """Factory for creating authorized algorithm trainers and default configs."""

    @staticmethod
    def create_trainer(
        algorithm: Union[str, AuthorizedAlgorithm],
        config: Optional[Any] = None,
    ) -> Union[SVMTrainer, MLPTrainer, DecisionTreeTrainer]:
        """Create and return a configured algorithm trainer.

        Args:
            algorithm: One of 'svm', 'mlp', 'decision_tree'.
            config: Optional algorithm-specific configuration object.

        Returns:
            Configured trainer instance.

        Raises:
            ValueError: If an unauthorized algorithm key is provided.
        """
        algo = algorithm.value if isinstance(algorithm, AuthorizedAlgorithm) else str(algorithm).lower().strip()

        if algo == "svm":
            cfg = config if isinstance(config, SVMTrainingConfig) else SVMTrainingConfig()
            return SVMTrainer(config=cfg)

        elif algo == "mlp":
            cfg = config if isinstance(config, MLPTrainingConfig) else MLPTrainingConfig()
            return MLPTrainer(config=cfg)

        elif algo == "decision_tree":
            cfg = config if isinstance(config, DecisionTreeTrainingConfig) else DecisionTreeTrainingConfig()
            return DecisionTreeTrainer(config=cfg)

        else:
            authorized_list = AuthorizedAlgorithm.list_authorized()
            raise ValueError(
                f"Unauthorized algorithm '{algorithm}'. The NIDS architecture strictly authorizes "
                f"exactly three algorithms: {', '.join(authorized_list)}."
            )

    @staticmethod
    def get_default_config(
        algorithm: Union[str, AuthorizedAlgorithm],
    ) -> Union[SVMTrainingConfig, MLPTrainingConfig, DecisionTreeTrainingConfig]:
        """Return fresh default configuration for an authorized algorithm."""
        algo = algorithm.value if isinstance(algorithm, AuthorizedAlgorithm) else str(algorithm).lower().strip()

        if algo == "svm":
            return SVMTrainingConfig()
        elif algo == "mlp":
            return MLPTrainingConfig()
        elif algo == "decision_tree":
            return DecisionTreeTrainingConfig()
        else:
            raise ValueError(f"Unauthorized algorithm '{algorithm}'. Must be one of {AuthorizedAlgorithm.list_authorized()}")
