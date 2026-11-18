"""Intrusion Detection Engine Subsystem (CP-08).

Provides modular, local, offline, CSV-based network intrusion detection inference
using the three authorized algorithms:
  1. Support Vector Machine (LinearSVC)
  2. Multilayer Perceptron (MLPClassifier)
  3. Decision Tree (DecisionTreeClassifier)
"""

from src.detection.dataset_loader import DetectionDatasetLoader, VerifiedDatasetSource
from src.detection.detection_manifest import (
    DetectionRunManifest,
    DisagreementSummary,
    ModelInferenceMetadata,
    RowAccountingRecord,
    capture_detection_environment,
)
from src.detection.detection_registry import DetectionRegistry
from src.detection.disagreement_analyzer import DisagreementAnalyzer
from src.detection.inference_pipeline import DetectionInferencePipeline
from src.detection.input_validator import DetectionInputValidator, ValidatedChunkResult
from src.detection.model_eligibility import EligibleModelInfo, ModelEligibilityVerifier
from src.detection.prediction_engine import DetectionPredictionEngine
from src.detection.prediction_mapper import DetectionPredictionMapper
from src.detection.preprocessing_loader import DetectionPreprocessingLoader
from src.detection.resource_manager import DetectionPreflightResult, DetectionResourceManager
from src.detection.result_writer import DetectionResultWriter
from src.detection.score_interpreter import DetectionScoreInterpreter

__all__ = [
    "DetectionDatasetLoader",
    "VerifiedDatasetSource",
    "DetectionRunManifest",
    "DisagreementSummary",
    "ModelInferenceMetadata",
    "RowAccountingRecord",
    "capture_detection_environment",
    "DetectionRegistry",
    "DisagreementAnalyzer",
    "DetectionInferencePipeline",
    "DetectionInputValidator",
    "ValidatedChunkResult",
    "EligibleModelInfo",
    "ModelEligibilityVerifier",
    "DetectionPredictionEngine",
    "DetectionPredictionMapper",
    "DetectionPreprocessingLoader",
    "DetectionPreflightResult",
    "DetectionResourceManager",
    "DetectionResultWriter",
    "DetectionScoreInterpreter",
]
