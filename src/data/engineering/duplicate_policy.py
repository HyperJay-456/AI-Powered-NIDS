"""Duplicate Data Policy & Partition-Leakage Isolation Module (CP-05).

Provides configurable handling for duplicate flow records:
  1. GROUP_ISOLATED (Recommended Default): Retains duplicates, but groups
     identical feature records into the same partition (train, validation,
     or test) to prevent cross-partition data leakage.
  2. EXCLUDE: Excludes duplicate feature records from the model-ready dataset,
     preserving only the primary occurrence.
  3. RETAIN_ALL: Preserves all records without group-boundary constraints.

Detects and reports conflicting-label groups (identical features with contradictory
ground-truth labels) without silently mutating data.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import enum
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.data.data_quality import serialize_row_length_safe
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateHandlingPolicy(str, enum.Enum):
    """Configurable duplicate handling policy options."""

    GROUP_ISOLATED = "group_isolated"  # Retain duplicates, group into same partition (Default)
    EXCLUDE = "exclude"                # Exclude duplicate occurrences from supervised dataset
    RETAIN_ALL = "retain_all"          # Keep all duplicates unconstrained (warn of leakage)


@dataclass
class ConflictingLabelGroup:
    """Group of identical feature records with divergent ground-truth labels."""

    feature_hash: str
    record_count: int
    conflicting_labels: Dict[str, int]
    sample_indices: List[int]


@dataclass
class DuplicateAuditReport:
    """Summary of duplicate record analysis and policy execution."""

    policy_applied: str
    total_records: int
    unique_feature_groups: int
    exact_duplicate_rows: int
    duplicate_percentage: float
    conflicting_label_groups_count: int
    conflicting_label_records_count: int
    conflicting_groups: List[ConflictingLabelGroup]
    records_retained: int
    records_excluded: int
    notes: List[str] = field(default_factory=list)

    @property
    def total_groups(self) -> int:
        """Alias for count of unique feature groups."""
        return self.unique_feature_groups


def compute_feature_fingerprints(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> List[str]:
    """Compute deterministic SHA-256 hex digests for each row across feature columns.

    Uses serialize_row_length_safe to prevent delimiter collisions.

    Args:
        df: DataFrame containing records.
        feature_columns: List of columns defining the feature space.

    Returns:
        List of 64-character SHA-256 hex digest strings corresponding to each row.
    """
    subset_df = df[feature_columns]
    fingerprints: List[str] = []

    for row in subset_df.itertuples(index=False, name=None):
        row_bytes = serialize_row_length_safe(row)
        fp_hex = hashlib.sha256(row_bytes).hexdigest()
        fingerprints.append(fp_hex)

    return fingerprints


class DuplicatePolicyManager:
    """Manages duplicate detection, conflicting label auditing, and partition grouping."""

    def __init__(self, policy: DuplicateHandlingPolicy = DuplicateHandlingPolicy.GROUP_ISOLATED):
        """Initialize with duplicate handling policy."""
        self.policy = policy

    def audit_duplicates(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str] = None,
    ) -> DuplicateAuditReport:
        """Audit duplicates without mutating data."""
        _, _, report = self.audit_and_apply_policy(df, feature_cols, label_col, self.policy)
        return report

    def generate_group_ids(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Generate deterministic integer group IDs for group-isolated splitting."""
        import numpy as np

        fps = compute_feature_fingerprints(df, feature_cols)
        unique_fps = {fp: i for i, fp in enumerate(sorted(set(fps)))}
        return np.array([unique_fps[fp] for fp in fps], dtype=np.int64)

    def apply_policy(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Apply policy and return (filtered_df, excluded_indices)."""
        out_df, _, report = self.audit_and_apply_policy(df, feature_cols, label_col, self.policy)
        if self.policy == DuplicateHandlingPolicy.EXCLUDE:
            excluded_indices = [i for i in range(len(df)) if i not in out_df.index]
        else:
            excluded_indices = []
        return out_df, excluded_indices

    @staticmethod
    def audit_and_apply_policy(
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: Optional[str] = None,
        policy: DuplicateHandlingPolicy = DuplicateHandlingPolicy.GROUP_ISOLATED,
    ) -> Tuple[pd.DataFrame, List[str], DuplicateAuditReport]:
        """Audit duplicate records, detect label conflicts, and apply the selected policy.

        Args:
            df: Full DataFrame containing features and optional label.
            feature_columns: Columns participating in feature representation.
            label_column: Name of target label column if present.
            policy: DuplicateHandlingPolicy to apply.

        Returns:
            Tuple of:
              - Processed DataFrame (filtered if policy is EXCLUDE, untouched otherwise)
              - List of group identifiers (fingerprints) matching the output DataFrame rows
              - DuplicateAuditReport with complete metrics
        """
        total_rows = len(df)
        if total_rows == 0:
            return df, [], DuplicateAuditReport(
                policy_applied=policy.value,
                total_records=0,
                unique_feature_groups=0,
                exact_duplicate_rows=0,
                duplicate_percentage=0.0,
                conflicting_label_groups_count=0,
                conflicting_label_records_count=0,
                conflicting_groups=[],
                records_retained=0,
                records_excluded=0,
                notes=["Input dataset is empty."],
            )

        # 1. Compute deterministic feature fingerprints
        fingerprints = compute_feature_fingerprints(df, feature_columns)

        # 2. Map fingerprints to row indices and labels
        fp_to_indices = defaultdict(list)
        fp_to_labels = defaultdict(list)

        has_label = label_column is not None and label_column in df.columns
        labels = df[label_column].astype(str).values if has_label else [None] * total_rows

        for idx, (fp, lbl) in enumerate(zip(fingerprints, labels)):
            fp_to_indices[fp].append(idx)
            if has_label and lbl is not None:
                fp_to_labels[fp].append(lbl)

        unique_groups = len(fp_to_indices)
        duplicate_count = total_rows - unique_groups
        duplicate_pct = round((duplicate_count / total_rows * 100), 2)

        # 3. Detect conflicting label groups
        conflicting_groups: List[ConflictingLabelGroup] = []
        conflicting_record_count = 0

        if has_label:
            for fp, group_labels in fp_to_labels.items():
                distinct_labels = set(group_labels)
                if len(distinct_labels) > 1:
                    label_counts = {lbl: group_labels.count(lbl) for lbl in distinct_labels}
                    indices = fp_to_indices[fp]
                    conflicting_record_count += len(indices)
                    conflicting_groups.append(
                        ConflictingLabelGroup(
                            feature_hash=fp[:12] + "...",
                            record_count=len(indices),
                            conflicting_labels=label_counts,
                            sample_indices=indices[:5],
                        )
                    )

        # 4. Apply selected policy
        notes: List[str] = []

        if policy == DuplicateHandlingPolicy.EXCLUDE:
            # Keep only first occurrence of each unique feature fingerprint
            first_indices = [indices[0] for indices in fp_to_indices.values()]
            first_indices.sort()
            out_df = df.iloc[first_indices].reset_index(drop=True)
            out_groups = [fingerprints[i] for i in first_indices]
            records_retained = len(out_df)
            records_excluded = total_rows - records_retained
            notes.append(
                f"Excluded {records_excluded:,} duplicate rows from the model-ready dataset. "
                f"Retained {records_retained:,} unique feature rows."
            )

        elif policy == DuplicateHandlingPolicy.GROUP_ISOLATED:
            # Retain all records; return group fingerprints for group-aware splitting
            out_df = df.copy()
            out_groups = fingerprints
            records_retained = total_rows
            records_excluded = 0
            notes.append(
                f"Retained all {total_rows:,} records (including {duplicate_count:,} duplicates). "
                "Feature group identifiers assigned to prevent cross-partition leakage."
            )

        else:  # RETAIN_ALL
            out_df = df.copy()
            out_groups = fingerprints
            records_retained = total_rows
            records_excluded = 0
            notes.append(
                f"Retained all {total_rows:,} records unconditionally. "
                "Warning: Identical feature records may cross train/validation/test partitions."
            )

        if conflicting_groups:
            notes.append(
                f"Detected {len(conflicting_groups):,} feature groups ({conflicting_record_count:,} records) "
                "with conflicting ground-truth labels. Retained without modification for audit visibility."
            )

        report = DuplicateAuditReport(
            policy_applied=policy.value,
            total_records=total_rows,
            unique_feature_groups=unique_groups,
            exact_duplicate_rows=duplicate_count,
            duplicate_percentage=duplicate_pct,
            conflicting_label_groups_count=len(conflicting_groups),
            conflicting_label_records_count=conflicting_record_count,
            conflicting_groups=conflicting_groups,
            records_retained=records_retained,
            records_excluded=records_excluded,
            notes=notes,
        )

        return out_df, out_groups, report
