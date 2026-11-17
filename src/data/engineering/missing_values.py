"""Missing Values Analysis & Leakage-Free Imputation Module (CP-05).

Provides complete missing-value identification (empty strings, whitespace-only,
NaN, null, +inf, -inf, entire columns), preserves legitimate numerical zeros,
fits imputation statistics strictly on the training partition, and enforces the
strict rule that target labels are never imputed.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnMissingReport:
    """Missing value analysis for an individual column."""

    column_name: str
    dtype: str
    total_rows: int
    missing_cells: int
    missing_percentage: float
    infinite_cells: int
    empty_string_cells: int
    is_entirely_missing: bool
    proposed_strategy: str


@dataclass
class MissingValueAuditReport:
    """Overall missing value profile across all columns of a dataset."""

    total_rows: int
    total_columns: int
    rows_with_any_missing: int
    percentage_rows_affected: float
    columns_with_missing: List[ColumnMissingReport]
    clean_columns_count: int
    entirely_missing_columns: List[str]
    has_infinite_values: bool
    total_infinite_cells: int

    @property
    def total_missing_cells(self) -> int:
        """Total missing cells across all columns."""
        return sum(c.missing_cells for c in self.columns_with_missing)

    @property
    def num_columns_with_missing(self) -> int:
        """Total columns that have missing cells."""
        return len(self.columns_with_missing)

    @property
    def missing_cells_per_column(self) -> Dict[str, int]:
        """Mapping from column name to count of missing cells."""
        return {c.column_name: c.missing_cells for c in self.columns_with_missing}

    @property
    def max_column_missing_pct(self) -> float:
        """Maximum missing percentage in any single column."""
        return max((c.missing_percentage for c in self.columns_with_missing), default=0.0)


def analyze_missing_values(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
) -> MissingValueAuditReport:
    """Perform comprehensive missing value and non-finite numeric analysis on a DataFrame.

    Distinguishes:
      - Legitimate numeric zero (retained as valid numeric data)
      - Empty strings and whitespace-only strings
      - NaN / None / pd.NA
      - Positive and negative infinity (+inf, -inf)

    Args:
        df: Input DataFrame to analyze.
        label_col: Optional target label column name.

    Returns:
        MissingValueAuditReport with column-by-column breakdown.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0:
        return MissingValueAuditReport(
            total_rows=0,
            total_columns=total_cols,
            rows_with_any_missing=0,
            percentage_rows_affected=0.0,
            columns_with_missing=[],
            clean_columns_count=total_cols,
            entirely_missing_columns=[],
            has_infinite_values=False,
            total_infinite_cells=0,
        )

    # Boolean mask tracking any missing or infinite value per row
    row_has_issue = np.zeros(total_rows, dtype=bool)
    col_reports: List[ColumnMissingReport] = []
    entirely_missing_cols: List[str] = []
    total_inf_count = 0

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        # 1. Standard null/NaN check
        null_mask = series.isna()

        # 2. Empty string or whitespace-only check for object/string columns
        empty_str_mask = np.zeros(total_rows, dtype=bool)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check non-null elements for whitespace/empty
            non_null_vals = series.dropna().astype(str)
            whitespace_idx = non_null_vals[non_null_vals.str.strip() == ""].index
            empty_str_mask[whitespace_idx] = True

        # 3. Numeric non-finite check (+inf, -inf)
        inf_mask = np.zeros(total_rows, dtype=bool)
        if pd.api.types.is_numeric_dtype(series):
            # Do NOT confuse 0 with missing! np.isinf checks strictly inf/-inf
            inf_mask = np.isinf(series.to_numpy(dtype=float, na_value=0.0))

        # Combined missing mask for this column
        combined_missing = null_mask | empty_str_mask | inf_mask
        missing_count = int(combined_missing.sum())
        inf_count = int(inf_mask.sum())
        empty_str_count = int(empty_str_mask.sum())

        total_inf_count += inf_count
        row_has_issue |= combined_missing

        missing_pct = round((missing_count / total_rows * 100), 2)
        is_entirely_missing = missing_count == total_rows

        if is_entirely_missing:
            entirely_missing_cols.append(col)

        # Proposed strategy
        if col == label_col:
            strategy = "Quarantine / Exclude from Supervised Partitions (Labels Never Imputed)"
        elif is_entirely_missing:
            strategy = "Flagged: Column Entirely Missing (Requires Policy Decision)"
        elif pd.api.types.is_numeric_dtype(series):
            strategy = "Median Imputation (Fitted on Training Split Only)"
        else:
            strategy = "Most-Frequent Categorical Imputation (Fitted on Training Split Only)"

        if missing_count > 0:
            col_reports.append(
                ColumnMissingReport(
                    column_name=col,
                    dtype=dtype_str,
                    total_rows=total_rows,
                    missing_cells=missing_count,
                    missing_percentage=missing_pct,
                    infinite_cells=inf_count,
                    empty_string_cells=empty_str_count,
                    is_entirely_missing=is_entirely_missing,
                    proposed_strategy=strategy,
                )
            )

    rows_affected = int(row_has_issue.sum())
    pct_rows = round((rows_affected / total_rows * 100), 2)
    clean_count = total_cols - len(col_reports)

    return MissingValueAuditReport(
        total_rows=total_rows,
        total_columns=total_cols,
        rows_with_any_missing=rows_affected,
        percentage_rows_affected=pct_rows,
        columns_with_missing=col_reports,
        clean_columns_count=clean_count,
        entirely_missing_columns=entirely_missing_cols,
        has_infinite_values=total_inf_count > 0,
        total_infinite_cells=total_inf_count,
    )


class LeakageFreeImputer:
    """Imputer that learns replacement statistics strictly on the training partition.

    Supports:
      - Median imputation for numeric columns (robust against outliers)
      - Most-frequent imputation for categorical columns
      - Replacing +inf and -inf with NaN before numeric imputation
      - Strictly never imputes ground-truth target labels
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ):
        """Initialize imputer strategies."""
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> "LeakageFreeImputer":
        """Compute imputation parameters exclusively on the training DataFrame.

        Validation and test data MUST NOT be passed to fit().

        Args:
            train_df: Training partition DataFrame.
            numeric_columns: Optional list of numeric column names.
            categorical_columns: Optional list of categorical column names.

        Returns:
            Fitted LeakageFreeImputer instance.
        """
        self.numeric_medians_ = {}
        self.categorical_modes_ = {}

        if numeric_columns is None:
            numeric_columns = list(train_df.select_dtypes(include=[np.number]).columns)
        if categorical_columns is None:
            categorical_columns = list(train_df.select_dtypes(exclude=[np.number]).columns)

        # 1. Fit numeric medians (converting inf/-inf to nan for calculation)
        for col in numeric_columns:
            if col in train_df.columns:
                series = train_df[col].replace([np.inf, -np.inf], np.nan)
                valid_numeric = series.dropna()
                if len(valid_numeric) > 0:
                    if self.numeric_strategy == "median":
                        self.numeric_medians_[col] = float(valid_numeric.median())
                    else:
                        self.numeric_medians_[col] = float(valid_numeric.mean())
                else:
                    logger.warning(f"Training column '{col}' has 0 valid values; default median set to 0.0.")
                    self.numeric_medians_[col] = 0.0

        # 2. Fit categorical modes
        for col in categorical_columns:
            if col in train_df.columns:
                series = train_df[col].dropna()
                # Exclude empty strings
                str_series = series.astype(str)
                non_empty = str_series[str_series.str.strip() != ""]
                if len(non_empty) > 0:
                    mode_val = non_empty.mode()
                    self.categorical_modes_[col] = mode_val.iloc[0] if len(mode_val) > 0 else "MISSING"
                else:
                    self.categorical_modes_[col] = "MISSING"

        self.is_fitted = True
        return self

    @property
    def column_statistics_(self) -> Dict[str, Any]:
        """Consolidated mapping of learned training column statistics."""
        return {**self.numeric_medians_, **self.categorical_modes_}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned training imputation parameters to a DataFrame partition.

        Used identically on train, validation, and test partitions.

        Args:
            df: Input DataFrame partition to impute.

        Returns:
            Clean, imputed DataFrame copy.
        """
        if not self.is_fitted:
            raise RuntimeError("LeakageFreeImputer must be fitted on training data before transform().")

        df_out = df.copy()

        # Numeric transformation
        for col, median_val in self.numeric_medians_.items():
            if col in df_out.columns:
                # Replace inf/-inf with median
                series = df_out[col].replace([np.inf, -np.inf], np.nan)
                df_out[col] = series.fillna(median_val)

        # Categorical transformation
        for col, mode_val in self.categorical_modes_.items():
            if col in df_out.columns:
                series = df_out[col].copy()
                # Replace nulls and whitespace-only with mode
                null_or_empty = series.isna() | (series.astype(str).str.strip() == "")
                df_out.loc[null_or_empty, col] = mode_val

        return df_out
