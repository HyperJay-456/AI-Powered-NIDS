"""Numerical Scaling & Dual-Representation Generator Module (CP-05).

Provides dual representation pipelines tailored to the project's authorized algorithms:
  1. Scaled Representation (SVM & MLP): Uses StandardScaler(with_mean=False) fitted
     exclusively on the training partition. Preserves sparsity and prevents dense expansion.
  2. Unscaled Representation (Decision Tree): Retains original numerical magnitudes
     after median imputation, preserving tree-split interpretability.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalerStatistics:
    """Fitted scaling parameters learned strictly from the training partition."""

    numeric_columns: List[str]
    means: Dict[str, float]
    scales: Dict[str, float]
    variances: Dict[str, float]


class DualRepresentationScaler:
    """Manages scaling for SVM/MLP while preserving unscaled representations for Decision Tree."""

    def __init__(self, with_mean: bool = False):
        """Initialize scaler.

        Args:
            with_mean: Set to False to preserve sparse matrix compatibility and avoid dense expansion.
        """
        self.with_mean = with_mean
        self.scaler: Optional[StandardScaler] = None
        self.numeric_columns: List[str] = []
        self.is_fitted: bool = False
        self.stats_: Optional[ScalerStatistics] = None

    def fit(
        self,
        train_df: Any,
        numeric_columns: Optional[List[str]] = None,
    ) -> "DualRepresentationScaler":
        """Fit scaler parameters exclusively on the training partition.

        Args:
            train_df: Training DataFrame partition or array/sparse matrix.
            numeric_columns: Optional list of numeric feature column names.

        Returns:
            Fitted DualRepresentationScaler instance.
        """
        if isinstance(train_df, np.ndarray) or sparse.issparse(train_df):
            num_arr = train_df if isinstance(train_df, np.ndarray) else train_df.toarray()
            self.numeric_columns = [f"col_{i}" for i in range(num_arr.shape[1])]
            self.scaler = StandardScaler(with_mean=self.with_mean)
            self.scaler.fit(num_arr)
            self.is_fitted = True
            return self

        cols = numeric_columns if numeric_columns is not None else list(train_df.columns)
        self.numeric_columns = list(cols)

        if not self.numeric_columns:
            self.is_fitted = True
            self.stats_ = ScalerStatistics(
                numeric_columns=[],
                means={},
                scales={},
                variances={},
            )
            return self

        # Extract numeric array from training partition
        num_arr = train_df[self.numeric_columns].to_numpy(dtype=np.float32)

        self.scaler = StandardScaler(with_mean=self.with_mean)
        self.scaler.fit(num_arr)
        self.is_fitted = True

        # Extract parameters for metadata
        means = {}
        scales = {}
        variances = {}

        for idx, col in enumerate(self.numeric_columns):
            means[col] = float(self.scaler.mean_[idx]) if self.scaler.mean_ is not None else 0.0
            scales[col] = float(self.scaler.scale_[idx]) if self.scaler.scale_ is not None else 1.0
            variances[col] = float(self.scaler.var_[idx]) if self.scaler.var_ is not None else 1.0

        self.stats_ = ScalerStatistics(
            numeric_columns=self.numeric_columns,
            means=means,
            scales=scales,
            variances=variances,
        )

        logger.info(
            f"Fitted StandardScaler(with_mean={self.with_mean}) on {len(self.numeric_columns)} numeric features."
        )
        return self

    def transform_scaled(
        self,
        df: pd.DataFrame,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> sparse.csr_matrix:
        """Produce the SCALED representation (for SVM and MLP).

        Combines scaled numeric features with one-hot encoded categorical features
        into a single SciPy CSR sparse matrix.

        Args:
            df: DataFrame partition containing numeric features.
            cat_sparse_matrix: Optional sparse matrix of encoded categorical features.

        Returns:
            SciPy CSR sparse matrix of combined scaled features.
        """
        if not self.is_fitted:
            raise RuntimeError("DualRepresentationScaler must be fitted on training data before transform.")

        num_sparse: sparse.csr_matrix
        if self.numeric_columns and self.scaler is not None:
            num_arr = df[self.numeric_columns].to_numpy(dtype=np.float32)
            scaled_arr = self.scaler.transform(num_arr)
            num_sparse = sparse.csr_matrix(scaled_arr, dtype=np.float32)
        else:
            num_sparse = sparse.csr_matrix((len(df), 0), dtype=np.float32)

        if cat_sparse_matrix is not None and cat_sparse_matrix.shape[1] > 0:
            return sparse.hstack([num_sparse, cat_sparse_matrix], format="csr", dtype=np.float32)

        return num_sparse

    @property
    def fitted_scaler_(self) -> Optional[StandardScaler]:
        """Access the underlying fitted Scikit-learn StandardScaler."""
        return self.scaler

    def transform(
        self,
        data: Any,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Transform into both scaled and unscaled representations."""
        if isinstance(data, np.ndarray) or sparse.issparse(data):
            num_arr = data if isinstance(data, np.ndarray) else data.toarray()
            scaled_arr = self.scaler.transform(num_arr) if self.scaler is not None else num_arr
            s_csr = sparse.csr_matrix(scaled_arr, dtype=np.float32)
            u_csr = sparse.csr_matrix(num_arr, dtype=np.float32)
            return s_csr, u_csr

        s_csr = self.transform_scaled(data, cat_sparse_matrix)
        u_csr = self.transform_unscaled(data, cat_sparse_matrix)
        return s_csr, u_csr

    def fit_transform(
        self,
        data: Any,
        numeric_columns: Optional[List[str]] = None,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Fit on data and return (scaled_csr, unscaled_csr)."""
        self.fit(data, numeric_columns)
        return self.transform(data, cat_sparse_matrix)

    def transform_unscaled(
        self,
        df: pd.DataFrame,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> sparse.csr_matrix:
        """Produce the UNSCALED representation (for Decision Tree).

        Combines raw (imputed) numeric features with one-hot encoded categorical features
        without standard scaling.

        Args:
            df: DataFrame partition containing numeric features.
            cat_sparse_matrix: Optional sparse matrix of encoded categorical features.

        Returns:
            SciPy CSR sparse matrix of combined unscaled features.
        """
        num_sparse: sparse.csr_matrix
        if self.numeric_columns:
            num_arr = df[self.numeric_columns].to_numpy(dtype=np.float32)
            num_sparse = sparse.csr_matrix(num_arr, dtype=np.float32)
        else:
            num_sparse = sparse.csr_matrix((len(df), 0), dtype=np.float32)

        if cat_sparse_matrix is not None and cat_sparse_matrix.shape[1] > 0:
            return sparse.hstack([num_sparse, cat_sparse_matrix], format="csr", dtype=np.float32)

        return num_sparse

    @property
    def fitted_scaler_(self) -> Optional[StandardScaler]:
        """Access the underlying fitted Scikit-learn StandardScaler."""
        return self.scaler

    def transform(
        self,
        data: Any,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Transform into both scaled and unscaled representations."""
        if isinstance(data, np.ndarray) or sparse.issparse(data):
            num_arr = data if isinstance(data, np.ndarray) else data.toarray()
            scaled_arr = self.scaler.transform(num_arr) if self.scaler is not None else num_arr
            s_csr = sparse.csr_matrix(scaled_arr, dtype=np.float32)
            u_csr = sparse.csr_matrix(num_arr, dtype=np.float32)
            return s_csr, u_csr

        s_csr = self.transform_scaled(data, cat_sparse_matrix)
        u_csr = self.transform_unscaled(data, cat_sparse_matrix)
        return s_csr, u_csr

    def fit_transform(
        self,
        data: Any,
        numeric_columns: Optional[List[str]] = None,
        cat_sparse_matrix: Optional[sparse.csr_matrix] = None,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Fit on data and return (scaled_csr, unscaled_csr)."""
        self.fit(data, numeric_columns)
        return self.transform(data, cat_sparse_matrix)
