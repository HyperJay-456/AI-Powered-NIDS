"""Input Validator & Ground-Truth Protection Engine (CP-08).

Validates raw input CSV chunk schemas against the exact feature schema required by the model,
strictly isolates and strips any ground-truth target label to prevent data leakage,
segregates network identifiers into provenance, and coerces numerical values safely.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidatedChunkResult:
    """Outcome of validating and preparing a raw CSV chunk for preprocessing."""

    features_df: pd.DataFrame
    ground_truth_series: Optional[pd.Series]
    provenance_df: pd.DataFrame
    valid_indices: List[int]
    rejected_indices: List[int]
    rejected_reasons: List[str] = field(default_factory=list)


class DetectionInputValidator:
    """Enforces schema alignment, numeric sanitization, and ground-truth label isolation."""

    def __init__(self, feature_schema: Dict[str, Any]):
        """Initialize validator with model's expected feature schema.

        Args:
            feature_schema: Dictionary loaded from engineering run's feature_schema.json.
        """
        self.feature_schema = feature_schema
        self.required_input_features: List[str] = feature_schema.get("input_features", [])
        self.numeric_features: Set[str] = set(feature_schema.get("numeric_features", []))
        self.categorical_features: Set[str] = set(feature_schema.get("categorical_features", []))
        self.excluded_identifiers: Set[str] = set(feature_schema.get("excluded_identifiers", []))

        # Build column normalization lookup (case-insensitive and trimmed whitespace)
        self.normalized_to_canonical: Dict[str, str] = {
            col.strip().lower(): col for col in self.required_input_features
        }

    def validate_schema_compatibility(self, sample_columns: List[str]) -> Tuple[bool, List[str]]:
        """Verify whether an incoming CSV contains all required feature columns.

        Returns:
            Tuple of (is_compatible: bool, missing_columns: List[str])
        """
        raw_normalized = {c.strip().lower(): c for c in sample_columns}
        missing_features: List[str] = []

        for req_col in self.required_input_features:
            norm_req = req_col.strip().lower()
            if norm_req not in raw_normalized:
                missing_features.append(req_col)

        is_compatible = (len(missing_features) == 0)
        return is_compatible, missing_features

    def process_chunk(
        self,
        chunk_df: pd.DataFrame,
        label_column_name: Optional[str] = None,
    ) -> ValidatedChunkResult:
        """Sanitize chunk, strip target labels, and assemble required model features in exact order.

        Args:
            chunk_df: Raw DataFrame chunk from CSV.
            label_column_name: Optional known ground-truth label column name.

        Returns:
            ValidatedChunkResult containing model-ready features and segregated ground-truth.
        """
        df = chunk_df.copy()

        # Map actual column names to canonical schema names (handling whitespace variance)
        actual_cols = list(df.columns)
        rename_map: Dict[str, str] = {}
        for col in actual_cols:
            norm = col.strip().lower()
            if norm in self.normalized_to_canonical:
                rename_map[col] = self.normalized_to_canonical[norm]

        if rename_map:
            df = df.rename(columns=rename_map)

        # 1. GROUND-TRUTH TARGET PROTECTION
        # Identify if any label column is present in the CSV
        target_series: Optional[pd.Series] = None
        label_col_found = None

        if label_column_name and label_column_name in df.columns:
            label_col_found = label_column_name
        else:
            # Check for common variants ("Label", " Label", "label", "attack_cat")
            for c in df.columns:
                if c.strip().lower() in ("label", "target", "class", "attack_cat"):
                    label_col_found = c
                    break

        if label_col_found and label_col_found in df.columns:
            # Strictly extract for reference only
            target_series = df[label_col_found].astype(str).str.strip().copy()
            # Drop from features immediately so model cannot see it
            df = df.drop(columns=[label_col_found], errors="ignore")

        # 2. SEPARATE NETWORK IDENTIFIERS INTO PROVENANCE
        # (Flow ID, Source IP, Destination IP, Timestamp)
        prov_cols_present = [c for c in actual_cols if c.strip().lower() in {id_c.strip().lower() for id_c in self.excluded_identifiers}]
        if prov_cols_present:
            provenance_df = chunk_df[prov_cols_present].copy()
            df = df.drop(columns=[c for c in prov_cols_present if c in df.columns], errors="ignore")
        else:
            provenance_df = pd.DataFrame(index=df.index)

        # 3. ASSEMBLE MODEL FEATURES IN EXACT CANONICAL ORDER
        features_dict: Dict[str, pd.Series] = {}
        rejected_indices: List[int] = []
        rejection_reasons: List[str] = []

        for req_col in self.required_input_features:
            if req_col not in df.columns:
                # Column is genuinely missing: fill with NaN to allow fitted imputer to handle if safe
                logger.warning(f"Required feature '{req_col}' missing from chunk; initializing as NaN.")
                features_dict[req_col] = pd.Series(np.nan, index=df.index, dtype=float)
                continue

            raw_col = df[req_col]

            if req_col in self.numeric_features:
                # Safe numeric conversion, converting inf / -inf to NaN
                # Legitimate numerical zero is carefully preserved
                s_num = pd.to_numeric(raw_col, errors="coerce")
                s_clean = s_num.replace([np.inf, -np.inf], np.nan)
                features_dict[req_col] = s_clean
            elif req_col in self.categorical_features:
                # Categorical conversion, replacing empty/nan with MISSING
                s_cat = raw_col.astype(str).str.strip()
                s_cat = s_cat.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING"})
                features_dict[req_col] = s_cat
            else:
                features_dict[req_col] = raw_col.copy()

        features_df = pd.DataFrame(features_dict, index=df.index)
        valid_indices = list(range(len(features_df)))

        return ValidatedChunkResult(
            features_df=features_df,
            ground_truth_series=target_series,
            provenance_df=provenance_df,
            valid_indices=valid_indices,
            rejected_indices=rejected_indices,
            rejected_reasons=rejection_reasons,
        )
