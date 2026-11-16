"""Schema Inspection and Column Classification Module.

Provides column header normalization, collision detection, ground-truth label
recognition, and attribute type classification.
"""

from collections import Counter
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.ingestion_manifest import ColumnMetadata


def normalize_column_name(col: str) -> str:
    """Normalize column header by stripping whitespace and lowercasing."""
    if not isinstance(col, str):
        col = str(col)
    return col.strip().lower()


def detect_label_column(columns: List[str]) -> Optional[str]:
    """Identify ground-truth Label column using normalized case-insensitive matching.

    Matches 'Label', ' Label', 'label', 'class', etc., prioritizing 'label'.

    Args:
        columns: List of original column names.

    Returns:
        Exact original column name matching label criteria, or None.
    """
    # First pass: direct match on normalized 'label'
    for col in columns:
        if normalize_column_name(col) == "label":
            return col

    # Second pass: check for standard variants
    for col in columns:
        norm = normalize_column_name(col)
        if norm in ("labels", "attack_cat", "attack_category", "class"):
            return col

    return None


def detect_duplicate_columns(columns: List[str]) -> Tuple[List[str], List[str]]:
    """Detect raw duplicate headers and collisions caused by whitespace/case normalization.

    Args:
        columns: List of raw column names.

    Returns:
        Tuple of (exact_duplicates, normalized_duplicates).
    """
    raw_counts = Counter(columns)
    exact_duplicates = [col for col, count in raw_counts.items() if count > 1]

    normalized_map: Dict[str, List[str]] = {}
    for col in columns:
        norm = normalize_column_name(col)
        normalized_map.setdefault(norm, []).append(col)

    normalized_duplicates = []
    for norm, orig_list in normalized_map.items():
        if len(orig_list) > 1 and norm not in [normalize_column_name(c) for c in exact_duplicates]:
            normalized_duplicates.extend(orig_list)

    return exact_duplicates, normalized_duplicates


def classify_column(
    col_name: str,
    dtype_str: str,
    is_label: bool,
    non_null_count: int,
    unique_count: Optional[int] = None,
) -> str:
    """Classify a column into 'Label', 'Numeric', 'Categorical', or 'Mixed'.

    Args:
        col_name: Original column name.
        dtype_str: Inferred pandas dtype string.
        is_label: Whether this is the detected ground-truth label column.
        non_null_count: Count of valid non-null records.
        unique_count: Optional unique value count.

    Returns:
        Classification string.
    """
    if is_label:
        return "Label"

    dtype_lower = dtype_str.lower()

    if any(n in dtype_lower for n in ("float", "int", "numeric", "double")):
        return "Numeric"

    if "bool" in dtype_lower:
        return "Categorical"

    # For object/string columns
    if unique_count is not None and non_null_count > 0:
        cardinality_ratio = unique_count / non_null_count
        if unique_count <= 50 or cardinality_ratio < 0.05:
            return "Categorical"

    return "Categorical" if "object" in dtype_lower or "string" in dtype_lower else "Mixed"
