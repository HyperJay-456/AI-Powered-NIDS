"""Model Evaluation Laboratory Subsystem (CP-07).

Provides zero-retraining, partition-segregated evaluation, benchmarking, and comparison
for the three authorized algorithm families:
1. Support Vector Machine (LinearSVC)
2. Multilayer Perceptron (MLPClassifier)
3. Decision Tree (DecisionTreeClassifier)
"""

from src.evaluation.artifact_verifier import ArtifactVerifier, ModelEligibilityCheckResult
from src.evaluation.confusion_matrix import ConfusionMatrixEngine
from src.evaluation.evaluation_data_loader import EvaluationDataLoader, LoadedEvaluationPartition
from src.evaluation.evaluation_manifest import (
    AggregateMetricsRecord,
    AlgorithmEvaluationResult,
    CandidateModelFreezeRecord,
    ConfusionMatrixRecord,
    EvaluationRunManifest,
    PerClassMetricRecord,
    PrecisionRecallMetricRecord,
    ROCMetricRecord,
    SecurityMetricRecord,
    capture_evaluation_environment,
)
from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.evaluation.evaluation_registry import EvaluationRegistry
from src.evaluation.inference_runner import BatchInferenceResult, InferenceRunner
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.model_comparator import ModelComparator
from src.evaluation.precision_recall_analysis import PrecisionRecallAnalyzer
from src.evaluation.resource_manager import EvaluationPreflightResult, EvaluationResourceManager
from src.evaluation.roc_analysis import ROCAnalyzer
from src.evaluation.security_metrics import SecurityMetricsEngine

__all__ = [
    "ArtifactVerifier",
    "ModelEligibilityCheckResult",
    "ConfusionMatrixEngine",
    "EvaluationDataLoader",
    "LoadedEvaluationPartition",
    "AggregateMetricsRecord",
    "AlgorithmEvaluationResult",
    "CandidateModelFreezeRecord",
    "ConfusionMatrixRecord",
    "EvaluationRunManifest",
    "PerClassMetricRecord",
    "PrecisionRecallMetricRecord",
    "ROCMetricRecord",
    "SecurityMetricRecord",
    "capture_evaluation_environment",
    "EvaluationOrchestrator",
    "EvaluationRegistry",
    "BatchInferenceResult",
    "InferenceRunner",
    "MetricsCalculator",
    "ModelComparator",
    "PrecisionRecallAnalyzer",
    "EvaluationPreflightResult",
    "EvaluationResourceManager",
    "ROCAnalyzer",
    "SecurityMetricsEngine",
]
