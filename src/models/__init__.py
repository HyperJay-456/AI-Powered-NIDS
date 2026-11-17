"""Model definitions, training pipelines, and serialization for SVM, MLP, and Decision Tree.

Checkpoint CP-06: ML Training Pipeline & Model Registry.
"""

from src.models.algorithms import (
    DecisionTreeTrainer,
    DecisionTreeTrainingResult,
    MLPTrainer,
    MLPTrainingResult,
    SVMTrainer,
    SVMTrainingResult,
)
from src.models.artifact_loader import (
    CP05ArtifactLoader,
    EngineeringRunIntegrityCheck,
    LoadedTrainingData,
)
from src.models.model_config import (
    AuthorizedAlgorithm,
    DecisionTreeTrainingConfig,
    MasterTrainingConfiguration,
    MLPTrainingConfig,
    SVMTrainingConfig,
)
from src.models.model_factory import ModelFactory
from src.models.model_registry import ModelArtifactRecord, ModelRegistry
from src.models.resource_manager import PreflightCheckResult, ResourceManager
from src.models.training_manifest import (
    AlgorithmTrainingManifest,
    MasterTrainingRunManifest,
    capture_training_environment,
)
from src.models.training_orchestrator import TrainingOrchestrator
from src.models.training_progress import (
    TrainingProgressTracker,
    TrainingProgressUpdate,
    TrainingStage,
)

__all__ = [
    "AuthorizedAlgorithm",
    "SVMTrainingConfig",
    "MLPTrainingConfig",
    "DecisionTreeTrainingConfig",
    "MasterTrainingConfiguration",
    "ResourceManager",
    "PreflightCheckResult",
    "CP05ArtifactLoader",
    "EngineeringRunIntegrityCheck",
    "LoadedTrainingData",
    "AlgorithmTrainingManifest",
    "MasterTrainingRunManifest",
    "capture_training_environment",
    "TrainingStage",
    "TrainingProgressUpdate",
    "TrainingProgressTracker",
    "SVMTrainer",
    "SVMTrainingResult",
    "MLPTrainer",
    "MLPTrainingResult",
    "DecisionTreeTrainer",
    "DecisionTreeTrainingResult",
    "ModelFactory",
    "ModelRegistry",
    "ModelArtifactRecord",
    "TrainingOrchestrator",
]

