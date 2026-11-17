"""Feature Policy & Leakage-Risk Assessment Module (CP-05).

Classifies schema columns into model features, identifiers with memorization/leakage risks,
ports, timestamps, and target labels. Establishes documented inclusion and exclusion policies
while preserving excluded columns in an audit provenance sidecar.
"""

from dataclasses import dataclass, field
import enum
from typing import Any, Dict, List, Optional, Set, Tuple

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to JSON-serializable dictionary."""
        return {
            "original_name": self.original_name,
            "normalized_name": self.normalized_name,
            "role": self.role.value,
            "include_in_model": self.include_in_model,
            "exclusion_reason": self.exclusion_reason,
            "is_user_configurable": self.is_user_configurable,
            "inferred_dtype": self.inferred_dtype,
            "notes": self.notes,
        }


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        return {
            "all_decisions": [d.to_dict() for d in self.all_decisions],
            "model_feature_columns": self.model_feature_columns,
            "numeric_feature_columns": self.numeric_feature_columns,
            "categorical_feature_columns": self.categorical_feature_columns,
            "excluded_identifier_columns": self.excluded_identifier_columns,
            "target_label_column": self.target_label_column,
            "total_input_columns": self.total_input_columns,
            "total_model_features": self.total_model_features,
            "total_excluded_columns": self.total_excluded_columns,
        }


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
    """Assesses columns, enforces leakage prevention, and manages feature definitions.

    Defaults:
        - Flow ID: exclude from model features.
        - Source IP: exclude by default, preserve in provenance.
        - Destination IP: exclude by default, preserve in provenance.
        - Raw Timestamp: exclude by default, preserve in provenance.
        - Source Port: configurable candidate feature (default False to minimize ephemeral noise).
        - Destination Port: configurable candidate feature (default True as service identifier).
        - Protocol: categorical feature (TCP/UDP/ICMP).
    """

    def __init__(
        self,
        include_destination_port: bool = True,
        include_source_port: bool = False,
        user_overrides: Optional[Dict[str, bool]] = None,
    ):
        """Initialize policy manager with configurable port and feature handling options.

        Args:
            include_destination_port: If True, Destination Port is retained as a candidate feature.
            include_source_port: If True, Source Port is retained as a candidate feature.
            user_overrides: Optional dictionary mapping column names to explicit inclusion booleans.
        """
        self.include_destination_port = include_destination_port
        self.include_source_port = include_source_port
        self.user_overrides: Dict[str, bool] = {
            normalize_column_name(k): v for k, v in (user_overrides or {}).items()
        }

    @classmethod
    def create_manifest(
        cls,
        columns: List[str],
        dtypes: Optional[Dict[str, str]] = None,
        label_column: Optional[str] = None,
        include_destination_port: bool = True,
        include_source_port: bool = False,
        user_overrides: Optional[Dict[str, bool]] = None,
    ) -> SchemaPolicyManifest:
        """Create policy manifest from list of column names with configurable options."""
        mgr = cls(
            include_destination_port=include_destination_port,
            include_source_port=include_source_port,
            user_overrides=user_overrides,
        )
        dtypes_dict = dtypes or {}
        return mgr.evaluate_schema(columns, dtypes_dict, label_column=label_column)

    @classmethod
    def classify_columns(
        cls,
        columns: List[str],
        include_destination_port: bool = True,
        include_source_port: bool = False,
        user_overrides: Optional[Dict[str, bool]] = None,
    ) -> List[FeaturePolicyDecision]:
        """Classify list of columns and return decisions."""
        manifest = cls.create_manifest(
            columns,
            include_destination_port=include_destination_port,
            include_source_port=include_source_port,
            user_overrides=user_overrides,
        )
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

        # 2. Network Identifiers (Memorization and host-tracking leakage risk)
        if norm in DEFAULT_IDENTIFIERS:
            is_flow_id = "flow" in norm and "id" in norm
            reason = (
                "Flow ID unique transaction identifier: excluded from model features to prevent memorization."
                if is_flow_id
                else "Host IP address identifier: excluded by default to prevent host memorization; preserved in provenance sidecar."
            )
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.IDENTIFIER_EXCLUDED,
                include_in_model=False,
                exclusion_reason=reason,
                is_user_configurable=not is_flow_id,
                inferred_dtype=dtype_str,
                notes="Preserved in provenance metadata sidecar." if not is_flow_id else "Unique flow transaction key.",
            )

        # 3. Timestamps (Temporal leakage risk)
        if norm in DEFAULT_TIMESTAMPS:
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.TIMESTAMP_EXCLUDED,
                include_in_model=False,
                exclusion_reason="Raw timestamp: excluded by default to prevent temporal lookahead/overfitting; preserved in provenance sidecar.",
                is_user_configurable=True,
                inferred_dtype=dtype_str,
                notes="Preserved in provenance metadata sidecar.",
            )

        # 4. Port Fields (Configurable candidate features)
        if norm in DEFAULT_PORTS:
            is_dst = "destination" in norm or "dst" in norm
            if norm in self.user_overrides:
                include = self.user_overrides[norm]
                reason = None if include else f"{'Destination' if is_dst else 'Source'} Port explicitly excluded by user configuration."
            elif is_dst:
                include = self.include_destination_port
                reason = None if include else "Destination Port excluded by user configuration."
            else:
                include = self.include_source_port
                reason = None if include else "Source Port excluded by default policy (configurable candidate feature)."

            notes = (
                "Destination service port candidate (e.g. HTTP 80, HTTPS 443, SSH 22, DNS 53)."
                if is_dst
                else "Source client port candidate (configurable candidate feature)."
            )

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

        # 5. Protocol Field (Categorical feature)
        if norm in ("protocol", "protocol_type"):
            include = self.user_overrides.get(norm, True)
            return FeaturePolicyDecision(
                original_name=col_name,
                normalized_name=norm,
                role=ColumnRole.CATEGORICAL_FEATURE,
                include_in_model=include,
                exclusion_reason=None if include else "Protocol excluded by user override.",
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
