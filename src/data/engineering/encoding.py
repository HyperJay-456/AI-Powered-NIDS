"""Categorical Encoding Module (CP-05).

Provides sparse, leakage-free categorical feature encoding via Scikit-learn's
OneHotEncoder:
  - Fits learned category vocabularies exclusively on the training partition
  - Handles unseen categories in validation/test without raising exceptions
  - Emits SciPy CSR sparse matrices to prevent dense memory expansion
  - Preserves stable feature name mappings for explainability
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CategoricalEncoderMetadata:
    """Summary of learned categorical encoding parameters."""

    categorical_columns: List[str]
    total_encoded_features: int
    category_vocabularies: Dict[str, List[str]]
    feature_names_out: List[str]


class LeakageFreeCategoricalEncoder:
    """One-hot encoder fitted strictly on the training partition with sparse output."""

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        handle_unknown: str = "ignore",
    ):
        """Initialize encoder with optional columns and safe unknown handling."""
        self.handle_unknown = handle_unknown
        self.encoder: Optional[OneHotEncoder] = None
        self.categorical_columns: List[str] = list(categorical_columns) if categorical_columns else []
        self.is_fitted: bool = False
        self.metadata_: Optional[CategoricalEncoderMetadata] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
    ) -> "LeakageFreeCategoricalEncoder":
        """Learn category vocabularies exclusively from training partition records.

        Args:
            train_df: Training DataFrame partition.
            categorical_columns: Optional columns to encode.

        Returns:
            Fitted encoder instance.
        """
        if categorical_columns is not None:
            self.categorical_columns = list(categorical_columns)
        elif not self.categorical_columns:
            self.categorical_columns = list(train_df.columns)

        if not self.categorical_columns:
            self.is_fitted = True
            self.metadata_ = CategoricalEncoderMetadata(
                categorical_columns=[],
                total_encoded_features=0,
                category_vocabularies={},
                feature_names_out=[],
            )
            return self

        # Extract categorical subset
        sub_df = train_df[self.categorical_columns].astype(str)

        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=True,
            dtype=np.float32,
        )
        self.encoder.fit(sub_df)
        self.is_fitted = True

        # Extract metadata
        vocabularies: Dict[str, List[str]] = {}
        for col_name, categories in zip(self.categorical_columns, self.encoder.categories_):
            vocabularies[col_name] = [str(c) for c in categories]

        feature_names = list(self.encoder.get_feature_names_out(self.categorical_columns))

        self.metadata_ = CategoricalEncoderMetadata(
            categorical_columns=self.categorical_columns,
            total_encoded_features=len(feature_names),
            category_vocabularies=vocabularies,
            feature_names_out=feature_names,
        )

        logger.info(
            f"Fitted OneHotEncoder on {len(self.categorical_columns)} categorical columns "
            f"-> {len(feature_names)} encoded features."
        )
        return self

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
    ) -> sparse.csr_matrix:
        """Fit on train and transform to sparse CSR matrix."""
        self.fit(train_df, categorical_columns)
        return self.transform(train_df)

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """Encode categorical columns of a partition into a SciPy CSR sparse matrix.

        Args:
            df: DataFrame partition to encode.

        Returns:
            SciPy CSR sparse matrix of encoded features (or empty CSR if no categorical cols).
        """
        if not self.is_fitted:
            raise RuntimeError("LeakageFreeCategoricalEncoder must be fitted on training data before transform().")

        if not self.categorical_columns or self.encoder is None:
            return sparse.csr_matrix((len(df), 0), dtype=np.float32)

        sub_df = df[self.categorical_columns].astype(str)
        res = self.encoder.transform(sub_df)
        if not isinstance(res, sparse.csr_matrix):
            res = sparse.csr_matrix(res, dtype=np.float32)

        return res
