"""Feature Policy & Leakage-Risk Assessment Module (CP-05).

Classifies schema columns into model features, identifiers with memorization/leakage risks,
ports, timestamps, and target labels. Establishes documented inclusion and exclusion policies
while preserving excluded columns in an audit provenance sidecar.
"""

from dataclasses import dataclass, field
import enum
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from config.settings import settings
from src.data.schema_inspector import normalize_column_name
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ColumnRole(str, enum.Enum):
    """Categorical role of a column in dataset engineering."""

    NUMERIC_FEATURE = "numeric_feature"
    CATEGORICAL_FEATURE = "categorical_feature"
    IDENTIFIER_EXCLUDED = "identifier_excluded"
    PORT_FIELD = "port_field"
    TIMESTAMP_EXCLUDED = "timestamp_excluded"
    GROUND_TRUTH_LABEL = "ground_truth_label"
    UNSUPPORTED = "unsupported"

    # Aliases for convenience
    IDENTIFIER = "identifier_excluded"
    PORT = "port_field"
    TIMESTAMP = "timestamp_excluded"
    LABEL = "ground_truth_label"


@dataclass
class FeaturePolicyDecision:
    """Documented policy decision for an individual column."""

    original_name: str
    normalized_name: str
    role: ColumnRole
    include_in_model: bool
    exclusion_reason: Optional[str] = None
    is_user_configurable: bool = True
    inferred_dtype: str = "unknown"
    notes: Optional[str] = None

    @property
    def column_name(self) -> str:
        """Alias for original column name."""
        return self.original_name

    @property
    def transformation_action(self) -> str:
        """Human-readable description of the preprocessing treatment."""
        if self.role == ColumnRole.NUMERIC_FEATURE or self.role == ColumnRole.PORT_FIELD:
            return "StandardScaler (SVM/MLP) & Unscaled (Decision Tree)"
        elif self.role == ColumnRole.CATEGORICAL_FEATURE:
            return "Sparse OneHotEncoder (handle_unknown='ignore')"
        elif self.role in (ColumnRole.IDENTIFIER_EXCLUDED, ColumnRole.TIMESTAMP_EXCLUDED):
            return "Quarantined into Provenance Sidecar"
        elif self.role == ColumnRole.GROUND_TRUTH_LABEL:
            return "Ground-Truth LabelEncoder (supervised target)"
        return "Unsupported / Excluded"

    @property
    def reason(self) -> str:
        """Justification for policy decision."""
        return self.exclusion_reason or self.notes or "Feature included in ML training matrix"


@dataclass
class SchemaPolicyManifest:
    """Complete collection of column decisions for an engineered dataset."""

    all_decisions: List[FeaturePolicyDecision]
    model_feature_columns: List[str]
    numeric_feature_columns: List[str]
    categorical_feature_columns: List[str]
    excluded_identifier_columns: List[str]
    target_label_column: Optional[str] = None
    total_input_columns: int = 0
    total_model_features: int = 0
    total_excluded_columns: int = 0


# Recognized identifier and leakage-prone column patterns
DEFAULT_IDENTIFIERS = {
    "flow_id",
    "flow id",
    "source_ip",
    "source ip",
    "src_ip",
    "destination_ip",
    "destination ip",
    "dst_ip",
    "external_ip",
}

DEFAULT_TIMESTAMPS = {
    "timestamp",
    "time",
    "date",
    "time_stamp",
}

DEFAULT_PORTS = {
    "source_port",
    "source port",
    "src_port",
    "destination_port",
    "destination port",
    "dst_port",
}


class FeaturePolicyManager:
    """Assesses columns, enforces leakage prevention, and manages feature definitions."""

    def __init__(self, include_destination_port: bool = True, include_source_port: bool = False):
        """Initialize policy manager with configurable port handling options.

        Args:
            include_destination_port: If True, Destination Port is retained as a discrete feature.
            include_source_port: If True, Source Port is retained (default False to reduce ephemeral port noise).
        """
        self.include_destination_port = include_destination_port
        self.include_source_port = include_source_port

    @classmethod
    def create_manifest(
        cls,
        columns: List[str],
        dtypes: Optional[Dict[str, str]] = None,
        label_column: Optional[str] = None,
    ) -> SchemaPolicyManifest:
        """Create policy manifest from list of column names."""
        mgr = cls()
        dtypes_dict = dtypes or {}
        return mgr.evaluate_schema(columns, dtypes_dict, label_column=label_column)

    @classmethod
    def classify_columns(cls, columns: List[str]) -> List[FeaturePolicyDecision]:
        """Classify list of columns and return decisions."""
        manifest = cls.create_manifest(columns)
        return manifest.all_decisions

    def evaluate_column(
        self,
        col_name: str,
        dtype_str: str,
        is_target_label: bool = False,
    ) -> FeaturePolicyDecision:
        """Evaluate a single column against cybersecurity feature and leakage policies.

        Args:
            col_name: Original column name.
            dtype_str: Inferred data type string.
            is_target_label: Whether this column is the ground-truth target.

        Returns:
            FeaturePolicyDecision with documented reasoning.
        """
        norm = normalize_column_name(col_name)

        # 1. Ground-Truth Target Label
        if is_target_label or norm in ("label", "target"):
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.GROUND_TRUTH_LABEL,
                include_in_model=False,
                exclusion_reason="Supervised target variable (extracted to label vector y).",
                is_user_configurable=False,
                inferred_dtype=dtype_str,
                notes="Ground-truth classification target.",
            )

        # 2. Network Identifiers (High memorization and leakage risk)
        if norm in DEFAULT_IDENTIFIERS:
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.IDENTIFIER_EXCLUDED,
                include_in_model=False,
                exclusion_reason="Identifier leakage risk: causes memorization rather than generalized threat patterns.",
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes="Preserved in provenance metadata sidecar.",
            )

        # 3. Timestamps (Temporal leakage risk)
        if norm in DEFAULT_TIMESTAMPS:
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.TIMESTAMP_EXCLUDED,
                include_in_model=False,
                exclusion_reason="Temporal leakage risk: models can overfit to specific capture time windows.",
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes="Preserved in provenance metadata sidecar.",
            )

        # 4. Port Fields
        if norm in DEFAULT_PORTS:
            if "destination" in norm or "dst" in norm:
                include = self.include_destination_port
                reason = None if include else "Destination port excluded by user configuration."
                notes = "Service identifier (HTTP 80, HTTPS 443, SSH 22, DNS 53, etc.)."
            else:
                include = self.include_source_port
                reason = None if include else "Source port excluded to reduce ephemeral port noise."
                notes = "Source client ephemeral port."

            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.PORT_FIELD,
                include_in_model=include,
                exclusion_reason=reason,
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes=notes,
            )

        # 5. Protocol Field
        if norm in ("protocol", "protocol_type"):
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.CATEGORICAL_FEATURE,
                include_in_model=True,
                exclusion_reason=None,
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes="Transport layer protocol (TCP, UDP, ICMP). Categorical encoding required.",
            )

        # 6. Numeric Flow Telemetry Features (CIC-IDS-2017 core features)
        is_numeric = (
            "int" in dtype_str
            or "float" in dtype_str
            or "double" in dtype_str
            or dtype_str == "unknown"
        )
        if is_numeric:
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.NUMERIC_FEATURE,
                include_in_model=True,
                exclusion_reason=None,
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes="Continuous or discrete flow metric.",
            )

        # 7. Non-numeric / Categorical features
        return FeaturePolicyDecision(
            original_name=col_name,
            normalized_name=norm,
            role=ColumnRole.CATEGORICAL_FEATURE,
            include_in_model=True,
            exclusion_reason=None,
            is_user_configurable=True,
            inferred_dtype=dtype_str,
            notes="Discrete categorical attribute.",
        )

    def evaluate_schema(
        self,
        columns: List[str],
        dtypes: Dict[str, str],
        label_column: Optional[str] = None,
    ) -> SchemaPolicyManifest:
        """Evaluate a full schema and generate a SchemaPolicyManifest.

        Args:
            columns: Ordered list of column names.
            dtypes: Mapping of column names to inferred data type strings.
            label_column: Name of target label column if present.

        Returns:
            SchemaPolicyManifest with all decisions and segregated lists.
        """
        decisions: List[FeaturePolicyDecision] = []
        model_features: List[str] = []
        numeric_features: List[str] = []
        categorical_features: List[str] = []
        excluded_identifiers: List[str] = []
        resolved_label: Optional[str] = None

        for col in columns:
            dtype_str = dtypes.get(col, "unknown")
            is_label = (col == label_column) or (normalize_column_name(col) == "label")
            dec = self.evaluate_column(col, dtype_str, is_target_label=is_label)
            decisions.append(dec)

            if dec.role == ColumnRole.GROUND_TRUTH_LABEL:
                resolved_label = col
            elif dec.include_in_model:
                model_features.append(col)
                if dec.role == ColumnRole.NUMERIC_FEATURE or dec.role == ColumnRole.PORT_FIELD:
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
            else:
                excluded_identifiers.append(col)

        return SchemaPolicyManifest(
            all_decisions=decisions,
            model_feature_columns=model_features,
            numeric_feature_columns=numeric_features,
            categorical_feature_columns=categorical_features,
            excluded_identifier_columns=excluded_identifiers,
            target_label_column=resolved_label,
            total_input_columns=len(columns),
            total_model_features=len(model_features),
            total_excluded_columns=len(excluded_identifiers),
        )
