"""Feature Engineering & Sanitization Module (CP-05).

Executes conservative, documented feature transformations:
  - Coerces numeric features safely while replacing +inf / -inf with NaN
  - Preserves legitimate numerical zero values without distortion
  - Enforces deterministic, stable column ordering
  - Segregates excluded identifier columns into an audit provenance DataFrame
  - Separates the ground-truth target label
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.engineering.feature_policy import SchemaPolicyManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSanitizer:
    """Applies conservative feature sanitization and separates model inputs from provenance."""

    def __init__(self, policy_manifest: Optional[SchemaPolicyManifest] = None):
        """Initialize sanitizer with optional policy manifest."""
        self.policy_manifest = policy_manifest

    def sanitize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """Sanitize DataFrame and separate features from provenance sidecar.

        Args:
            df: Input raw DataFrame.

        Returns:
            Tuple of (features_df, provenance_sidecar_df, quarantined_indices).
        """
        from src.data.engineering.feature_policy import FeaturePolicyManager

        manifest = self.policy_manifest or FeaturePolicyManager.create_manifest(list(df.columns))
        features_df, target_series, prov_df = self.sanitize_and_split_features(df, manifest)
        return features_df, prov_df, []

    @staticmethod
    def sanitize_and_split_features(
        df: pd.DataFrame,
        schema_manifest: SchemaPolicyManifest,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        """Sanitize columns, separate model features, extract target, and create provenance sidecar.

        Args:
            df: Raw input DataFrame partition.
            schema_manifest: SchemaPolicyManifest with column classifications.

        Returns:
            Tuple of:
              - features_df: DataFrame containing only model features in stable column order
              - target_series: Series containing ground-truth labels (or None if missing)
              - provenance_df: DataFrame containing excluded identifiers, timestamps, and record indices
        """
        model_cols = schema_manifest.model_feature_columns
        num_cols = schema_manifest.numeric_feature_columns
        cat_cols = schema_manifest.categorical_feature_columns
        ident_cols = schema_manifest.excluded_identifier_columns
        label_col = schema_manifest.target_label_column

        # 1. Extract Target Label Series
        target_series: Optional[pd.Series] = None
        if label_col and label_col in df.columns:
            target_series = df[label_col].astype(str).str.strip().copy()

        # 2. Extract Provenance Sidecar (identifiers, timestamps, original row indices)
        prov_cols_present = [c for c in ident_cols if c in df.columns]
        if prov_cols_present:
            provenance_df = df[prov_cols_present].copy()
        else:
            provenance_df = pd.DataFrame(index=df.index)

        # 3. Build Model Features DataFrame in stable order
        features_dict: Dict[str, pd.Series] = {}

        for col in model_cols:
            if col not in df.columns:
                logger.warning(f"Expected feature column '{col}' missing from data; creating NaN series.")
                features_dict[col] = pd.Series(np.nan, index=df.index, dtype=float)
                continue

            raw_series = df[col]

            if col in num_cols:
                # Numeric feature: coerce to numeric, convert inf/-inf to nan
                # Do NOT convert 0 to nan!
                s_numeric = pd.to_numeric(raw_series, errors="coerce")
                s_clean = s_numeric.replace([np.inf, -np.inf], np.nan)
                features_dict[col] = s_clean
            elif col in cat_cols:
                # Categorical feature: trim whitespace, represent as string/object
                s_cat = raw_series.astype(str).str.strip()
                s_cat = s_cat.replace({"": "MISSING", "nan": "MISSING", "None": "MISSING"})
                features_dict[col] = s_cat
            else:
                features_dict[col] = raw_series.copy()

        features_df = pd.DataFrame(features_dict, index=df.index)

        return features_df, target_series, provenance_df
