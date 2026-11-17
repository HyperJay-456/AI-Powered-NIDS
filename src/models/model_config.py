"""Model Configuration & Algorithm Specifications (CP-06).

Defines strongly-typed, validated hyperparameter specifications for the exactly three
authorized ML algorithms:
  1. Support Vector Machine (LinearSVC)
  2. Multilayer Perceptron (MLPClassifier)
  3. Decision Tree (DecisionTreeClassifier)
"""

from dataclasses import asdict, dataclass, field
import enum
from typing import Any, Dict, List, Optional, Tuple, Union


class AuthorizedAlgorithm(str, enum.Enum):
    """The exactly three authorized machine learning algorithms for the NIDS project."""

    SVM = "svm"
    MLP = "mlp"
    DECISION_TREE = "decision_tree"

    @classmethod
    def list_authorized(cls) -> List[str]:
        """Return list of string keys for all authorized algorithms."""
        return [algo.value for algo in cls]

    @classmethod
    def get_display_name(cls, algorithm: Union[str, "AuthorizedAlgorithm"]) -> str:
        """Return enterprise display name for an algorithm."""
        val = algorithm.value if isinstance(algorithm, AuthorizedAlgorithm) else str(algorithm).lower()
        mapping = {
            "svm": "Linear Support Vector Machine (LinearSVC)",
            "mlp": "Multilayer Perceptron (Neural Network)",
            "decision_tree": "Decision Tree Classifier (CART)",
        }
        return mapping.get(val, f"Unknown ({val})")


@dataclass
class SVMTrainingConfig:
    """Hyperparameters for Linear Support Vector Machine training.

    Uses sklearn.svm.LinearSVC for high scalability on sparse network flow datasets.
    """

    C: float = 1.0
    penalty: str = "l2"
    loss: str = "squared_hinge"
    tol: float = 1e-4
    max_iter: int = 2000
    class_weight: Optional[str] = None  # None or "balanced"
    random_state: int = 42
    dual: str = "auto"  # sklearn 1.3+ auto-selects dual based on n_samples > n_features

    def validate(self) -> None:
        """Validate hyperparameter boundaries."""
        if self.C <= 0:
            raise ValueError(f"SVM regularization parameter C must be positive, got {self.C}")
        if self.penalty not in ("l1", "l2"):
            raise ValueError(f"SVM penalty must be 'l1' or 'l2', got {self.penalty}")
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError(f"SVM loss must be 'hinge' or 'squared_hinge', got {self.loss}")
        if self.tol <= 0:
            raise ValueError(f"SVM tolerance must be positive, got {self.tol}")
        if self.max_iter < 100:
            raise ValueError(f"SVM max_iter must be at least 100, got {self.max_iter}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class MLPTrainingConfig:
    """Hyperparameters for Multilayer Perceptron (Neural Network) training.

    Supports resource-aware incremental mini-batch training via partial_fit()
    on sparse CSR feature matrices without dense expansion.
    """

    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"  # "relu", "tanh", "logistic"
    solver: str = "adam"  # "adam", "sgd"
    alpha: float = 0.0001  # L2 penalty
    batch_size: int = 256  # Mini-batch size for incremental partial_fit
    learning_rate_init: float = 0.001
    max_epochs: int = 20  # Number of passes over the training partition
    tol: float = 1e-4
    random_state: int = 42
    use_incremental: bool = True  # If True, processes in mini-batches via partial_fit

    def validate(self) -> None:
        """Validate hyperparameter boundaries."""
        if not self.hidden_layer_sizes or any(h <= 0 for h in self.hidden_layer_sizes):
            raise ValueError(f"MLP hidden_layer_sizes must contain positive integers, got {self.hidden_layer_sizes}")
        if self.activation not in ("relu", "tanh", "logistic"):
            raise ValueError(f"MLP activation must be 'relu', 'tanh', or 'logistic', got {self.activation}")
        if self.solver not in ("adam", "sgd"):
            raise ValueError(f"MLP solver must be 'adam' or 'sgd', got {self.solver}")
        if self.alpha < 0:
            raise ValueError(f"MLP alpha regularization must be non-negative, got {self.alpha}")
        if self.batch_size <= 0:
            raise ValueError(f"MLP batch_size must be positive, got {self.batch_size}")
        if self.learning_rate_init <= 0:
            raise ValueError(f"MLP learning_rate_init must be positive, got {self.learning_rate_init}")
        if self.max_epochs <= 0:
            raise ValueError(f"MLP max_epochs must be positive, got {self.max_epochs}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["hidden_layer_sizes"] = list(self.hidden_layer_sizes)
        return data


@dataclass
class DecisionTreeTrainingConfig:
    """Hyperparameters for Decision Tree (CART) classifier training.

    Trained on the unscaled feature representation with physical flow magnitudes.
    """

    criterion: str = "gini"  # "gini", "entropy", "log_loss"
    splitter: str = "best"  # "best", "random"
    max_depth: Optional[int] = 15  # Constrains tree depth to prevent runaway memory
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_leaf_nodes: Optional[int] = None
    class_weight: Optional[str] = None  # None or "balanced"
    random_state: int = 42

    def validate(self) -> None:
        """Validate hyperparameter boundaries."""
        if self.criterion not in ("gini", "entropy", "log_loss"):
            raise ValueError(f"Decision Tree criterion must be 'gini', 'entropy', or 'log_loss', got {self.criterion}")
        if self.splitter not in ("best", "random"):
            raise ValueError(f"Decision Tree splitter must be 'best' or 'random', got {self.splitter}")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(f"Decision Tree max_depth must be positive, got {self.max_depth}")
        if self.min_samples_split < 2:
            raise ValueError(f"Decision Tree min_samples_split must be >= 2, got {self.min_samples_split}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"Decision Tree min_samples_leaf must be >= 1, got {self.min_samples_leaf}")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes <= 1:
            raise ValueError(f"Decision Tree max_leaf_nodes must be > 1, got {self.max_leaf_nodes}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class MasterTrainingConfiguration:
    """Master configuration container for training single or all models."""

    engineering_run_id: str = ""
    target_algorithm: str = "all"  # "svm", "mlp", "decision_tree", or "all"
    svm_config: SVMTrainingConfig = field(default_factory=SVMTrainingConfig)
    mlp_config: MLPTrainingConfig = field(default_factory=MLPTrainingConfig)
    decision_tree_config: DecisionTreeTrainingConfig = field(default_factory=DecisionTreeTrainingConfig)
    safety_margin_percent: float = 75.0  # Max percentage of available RAM allowed

    def get_algorithm_config(self, algorithm: str) -> Any:
        """Retrieve algorithm-specific configuration object."""
        algo = algorithm.lower().strip()
        if algo == "svm":
            return self.svm_config
        elif algo == "mlp":
            return self.mlp_config
        elif algo == "decision_tree":
            return self.decision_tree_config
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Must be one of {AuthorizedAlgorithm.list_authorized()}")
