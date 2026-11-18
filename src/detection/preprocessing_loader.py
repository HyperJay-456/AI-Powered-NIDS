"""Preprocessing Loader & Transformation Engine (CP-08).

Reuses the exact, immutable fitted preprocessing transformers from CP-05:
  - Median Imputers (fitted on training partition)
  - OneHot Categorical Encoders (fitted on training partition, unknown categories ignored)
  - StandardScalers (with_mean=False, fitted on training partition for SVM/MLP)
  - Unscaled representation for Decision Tree

STRICT INVARIANTS:
  1. Never calls fit() or fit_transform()
  2. Preserves exact training feature ordering and dimensional alignment
  3. Emits SciPy CSR sparse matrices to avoid dense RAM expansion
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPreprocessingLoader:
    """Loads and applies fitted CP-05 preprocessing pipelines without refitting."""

    def __init__(self, engineering_run_id: str, processed_dir: Optional[Path] = None):
        """Initialize loader for a specific engineering run.

        Args:
            engineering_run_id: CP-05 versioned run ID.
            processed_dir: Directory containing processed engineering artifacts.
        """
        self.engineering_run_id = engineering_run_id
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.run_dir = self.processed_dir / engineering_run_id

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Engineering run directory not found: {self.run_dir}")

        self.metadata_dir = self.run_dir / "metadata"
        self.prep_dir = self.run_dir / "preprocessors"

        # Load schemas and mappings
        self.feature_schema = self._load_json(self.metadata_dir / "feature_schema.json")
        self.label_mapping = self._load_json(self.metadata_dir / "label_mapping.json")

        self.scaled_bundle: Optional[Dict[str, Any]] = None
        self.tree_bundle: Optional[Dict[str, Any]] = None

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required metadata JSON missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_preprocessor_bundle(self, algorithm: str) -> Dict[str, Any]:
        """Load the exact fitted preprocessing bundle for an algorithm."""
        algo = algorithm.lower().strip()
        if algo in ("svm", "mlp"):
            if self.scaled_bundle is None:
                path = self.prep_dir / "scaled_preprocessor.joblib"
                if not path.exists():
                    raise FileNotFoundError(f"Scaled preprocessor artifact missing: {path}")
                self.scaled_bundle = joblib.load(path)
            return self.scaled_bundle
        elif algo == "decision_tree":
            if self.tree_bundle is None:
                path = self.prep_dir / "tree_preprocessor.joblib"
                if not path.exists():
                    # Fallback to scaled preprocessor bundle which has unscaled transformation support
                    scaled_path = self.prep_dir / "scaled_preprocessor.joblib"
                    if scaled_path.exists():
                        self.tree_bundle = joblib.load(scaled_path)
                    else:
                        raise FileNotFoundError(f"Tree preprocessor artifact missing: {path}")
                else:
                    self.tree_bundle = joblib.load(path)
            return self.tree_bundle
        else:
            raise ValueError(f"Unsupported algorithm '{algo}'. Authorized: ('svm', 'mlp', 'decision_tree')")

    def transform(self, features_df: pd.DataFrame, algorithm: str) -> sparse.csr_matrix:
        """Apply the saved fitted preprocessing pipeline to features DataFrame.

        Never invokes fit() or alters saved parameters.
        Returns SciPy CSR sparse matrix matching model's expected input dimension.
        """
        algo = algorithm.lower().strip()
        bundle = self.load_preprocessor_bundle(algo)

        imputer = bundle.get("imputer")
        cat_encoder = bundle.get("categorical_encoder")
        scaler = bundle.get("scaler")
        expected_output_features: List[str] = bundle.get("output_features", [])
        num_cols: List[str] = bundle.get("numeric_columns", [])
        cat_cols: List[str] = bundle.get("categorical_columns", [])

        if imputer is None:
            raise RuntimeError(f"Fitted imputer missing from preprocessor bundle for '{algo}'.")

        # 1. Leakage-free imputation using training-learned medians & modes
        df_imputed = imputer.transform(features_df)

        # 2. Categorical one-hot encoding using training-learned vocabularies
        cat_sparse = None
        if cat_encoder is not None and cat_cols:
            cat_sparse = cat_encoder.transform(df_imputed)

        # 3. Algorithm-specific scaling
        if algo in ("svm", "mlp"):
            # SCALED representation (StandardScaler with_mean=False)
            if scaler is None:
                raise RuntimeError(f"Fitted scaler missing from scaled preprocessor bundle for '{algo}'.")
            X_csr = scaler.transform_scaled(df_imputed, cat_sparse)
        else:
            # UNSCALED representation (for Decision Tree)
            if scaler is not None and hasattr(scaler, "transform_unscaled"):
                X_csr = scaler.transform_unscaled(df_imputed, cat_sparse)
            else:
                # Direct unscaled assembly: raw numeric CSR + categorical sparse
                if num_cols:
                    num_arr = df_imputed[num_cols].to_numpy(dtype=np.float32)
                    num_csr = sparse.csr_matrix(num_arr, dtype=np.float32)
                else:
                    num_csr = sparse.csr_matrix((len(df_imputed), 0), dtype=np.float32)

                if cat_sparse is not None and cat_sparse.shape[1] > 0:
                    X_csr = sparse.hstack([num_csr, cat_sparse], format="csr", dtype=np.float32)
                else:
                    X_csr = num_csr

        # 4. Strict Dimensional Validation
        if expected_output_features and X_csr.shape[1] != len(expected_output_features):
            raise ValueError(
                f"Feature dimension mismatch for '{algo}': transformed matrix has {X_csr.shape[1]} features, "
                f"but model pipeline expects {len(expected_output_features)} features ({expected_output_features[:5]}...)."
            )

        return X_csr
