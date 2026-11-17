"""Dataset Engineering and Preprocessing Pipeline Package (CP-05).

Provides verified source selection, schema compatibility reconciliation,
leakage-free imputation, duplicate group isolation, feature policy evaluation,
sparse one-hot encoding, dual-representation scaling (Scaled for SVM/MLP,
Unscaled for Decision Tree), ground-truth label encoding, class imbalance
analysis, reproducible splitting, atomic artifact persistence, and auditable
row accounting.
"""

from src.data.engineering.artifact_manager import (
    ArtifactManager,
    EngineeringRunRecord,
)
from src.data.engineering.dataset_registry import (
    DatasetIntegrityCheck,
    DatasetRegistry,
)
from src.data.engineering.dataset_splitter import (
    PartitionIndices,
    ReproducibleDatasetSplitter,
    SplittingAuditReport,
    SplittingStrategy,
)
from src.data.engineering.duplicate_policy import (
    ConflictingLabelGroup,
    DuplicateAuditReport,
    DuplicateHandlingPolicy,
    DuplicatePolicyManager,
    compute_feature_fingerprints,
)
from src.data.engineering.encoding import (
    CategoricalEncoderMetadata,
    LeakageFreeCategoricalEncoder,
)
from src.data.engineering.engineering_report import (
    EngineeringReportGenerator,
    PreprocessingAuditReport,
    RowAccountingReport,
)
from src.data.engineering.feature_engineering import FeatureSanitizer
from src.data.engineering.feature_policy import (
    ColumnRole,
    FeaturePolicyDecision,
    FeaturePolicyManager,
    SchemaPolicyManifest,
)
from src.data.engineering.label_processor import (
    ClassDistributionItem,
    ClassImbalanceReport,
    GroundTruthLabelProcessor,
    LabelMappingMetadata,
)
from src.data.engineering.missing_values import (
    ColumnMissingReport,
    LeakageFreeImputer,
    MissingValueAuditReport,
    analyze_missing_values,
)
from src.data.engineering.preprocessing_pipeline import (
    PreprocessingPipeline,
    PreprocessingRunConfiguration,
)
from src.data.engineering.scaling import (
    DualRepresentationScaler,
    ScalerStatistics,
)
from src.data.engineering.source_auditor import (
    SchemaAuditResult,
    SourceAuditor,
)

__all__ = [
    "ArtifactManager",
    "CategoricalEncoderMetadata",
    "ClassDistributionItem",
    "ClassImbalanceReport",
    "ColumnMissingReport",
    "ColumnRole",
    "ConflictingLabelGroup",
    "DatasetIntegrityCheck",
    "DatasetRegistry",
    "DualRepresentationScaler",
    "DuplicateAuditReport",
    "DuplicateHandlingPolicy",
    "DuplicatePolicyManager",
    "EngineeringReportGenerator",
    "EngineeringRunRecord",
    "FeaturePolicyDecision",
    "FeaturePolicyManager",
    "FeatureSanitizer",
    "GroundTruthLabelProcessor",
    "LabelMappingMetadata",
    "LeakageFreeCategoricalEncoder",
    "LeakageFreeImputer",
    "MissingValueAuditReport",
    "PartitionIndices",
    "PreprocessingAuditReport",
    "PreprocessingPipeline",
    "PreprocessingRunConfiguration",
    "ReproducibleDatasetSplitter",
    "RowAccountingReport",
    "ScalerStatistics",
    "SchemaAuditResult",
    "SchemaPolicyManifest",
    "SourceAuditor",
    "SplittingAuditReport",
    "SplittingStrategy",
    "analyze_missing_values",
    "compute_feature_fingerprints",
]
