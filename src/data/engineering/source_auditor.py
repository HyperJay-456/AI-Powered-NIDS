"""Source Auditor & Schema Compatibility Reconciliation Module (CP-05).

Evaluates schema compatibility across multiple candidate source datasets,
detects column differences, validates target label consistency, and enforces
safe schema matching before dataset engineering begins.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config.settings import settings
from src.data.ingestion_manifest import DatasetRecord
from src.data.schema_inspector import normalize_column_name
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaAuditResult:
    """Outcome of multi-dataset schema compatibility analysis."""

    is_compatible: bool
    selected_dataset_ids: List[str]
    selected_filenames: List[str]
    common_columns: List[str]
    common_normalized_columns: List[str]
    conflicting_columns: Dict[str, List[str]] = field(default_factory=dict)
    missing_columns_by_file: Dict[str, List[str]] = field(default_factory=dict)
    label_column: Optional[str] = None
    has_consistent_label: bool = False
    total_source_rows: int = 0
    total_columns_count: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def incompatible_reasons(self) -> List[str]:
        """Convenience alias for rejection reasons."""
        return self.rejection_reasons


class SourceAuditor:
    """Audits and reconciles schemas across multiple candidate CSV datasets."""

    @staticmethod
    def audit_datasets(records: List[DatasetRecord]) -> SchemaAuditResult:
        """Audit compatibility across a collection of candidate dataset records.

        Args:
            records: List of DatasetRecord instances to reconcile.

        Returns:
            SchemaAuditResult with compatibility verdict, column details, and warnings.
        """
        if not records:
            return SchemaAuditResult(
                is_compatible=False,
                selected_dataset_ids=[],
                selected_filenames=[],
                common_columns=[],
                common_normalized_columns=[],
                rejection_reasons=["No datasets selected for engineering."],
            )

        dataset_ids = [r.dataset_id for r in records]
        filenames = [r.safe_filename for r in records]
        total_rows = sum(r.valid_rows for r in records)

        # Single dataset case: always self-compatible
        if len(records) == 1:
            r = records[0]
            col_names = [c.name for c in r.columns] if r.columns else []
            norm_cols = [c.normalized_name for c in r.columns] if r.columns else []
            label_col = r.label_column_name if r.has_ground_truth_label else None

            warnings = list(r.validation_warnings) if r.validation_warnings else []
            if not label_col:
                warnings.append("Selected dataset has no detected ground-truth 'Label' column.")

            return SchemaAuditResult(
                is_compatible=True,
                selected_dataset_ids=dataset_ids,
                selected_filenames=filenames,
                common_columns=col_names,
                common_normalized_columns=norm_cols,
                conflicting_columns={},
                missing_columns_by_file={},
                label_column=label_col,
                has_consistent_label=bool(label_col),
                total_source_rows=total_rows,
                total_columns_count=len(col_names),
                rejection_reasons=[],
                warnings=warnings,
            )

        # Multi-dataset reconciliation
        # 1. Compare normalized column sets
        file_norm_cols: Dict[str, Set[str]] = {}
        file_orig_cols: Dict[str, List[str]] = {}
        file_labels: Dict[str, Optional[str]] = {}

        for r in records:
            orig = [c.name for c in r.columns] if r.columns else []
            norm = {c.normalized_name for c in r.columns} if r.columns else set()
            file_orig_cols[r.safe_filename] = orig
            file_norm_cols[r.safe_filename] = norm
            file_labels[r.safe_filename] = r.label_column_name if r.has_ground_truth_label else None

        # Intersection of all normalized columns
        common_norm_set = set.intersection(*file_norm_cols.values())
        all_norm_set = set.union(*file_norm_cols.values())

        # Check for discrepancies
        missing_by_file: Dict[str, List[str]] = {}
        for fname, col_set in file_norm_cols.items():
            missing = sorted(list(all_norm_set - col_set))
            if missing:
                missing_by_file[fname] = missing

        rejection_reasons: List[str] = []
        warnings: List[str] = []

        if missing_by_file:
            all_missing_cols = sorted(list({c for cols in missing_by_file.values() for c in cols}))
            rejection_reasons.append(
                f"Schema mismatch detected across {len(records)} selected files. "
                "All files must share identical feature sets to be combined. "
                f"Missing or mismatched columns: {', '.join(all_missing_cols)}."
            )

        # Check label consistency across files
        labels_present = set(file_labels.values())
        if None in labels_present and len(labels_present) > 1:
            rejection_reasons.append(
                "Inconsistent label availability: Some selected datasets have a ground-truth "
                "label column while others do not."
            )
            has_consistent_label = False
            resolved_label = None
        elif None in labels_present and len(labels_present) == 1:
            warnings.append("None of the selected datasets contain a ground-truth label column.")
            has_consistent_label = False
            resolved_label = None
        else:
            # All have a label column; check normalized identity
            norm_labels = {normalize_column_name(l) for l in labels_present if l}
            if len(norm_labels) > 1:
                rejection_reasons.append(
                    f"Conflicting target label column names detected: {sorted(list(norm_labels))}."
                )
                has_consistent_label = False
                resolved_label = None
            else:
                has_consistent_label = True
                resolved_label = records[0].label_column_name

        # Establish common column names in reference order from first file
        ref_cols = file_orig_cols[records[0].safe_filename]
        common_cols = [c for c in ref_cols if normalize_column_name(c) in common_norm_set]
        common_norm_list = [normalize_column_name(c) for c in common_cols]

        is_compatible = len(rejection_reasons) == 0

        return SchemaAuditResult(
            is_compatible=is_compatible,
            selected_dataset_ids=dataset_ids,
            selected_filenames=filenames,
            common_columns=common_cols,
            common_normalized_columns=common_norm_list,
            conflicting_columns={},
            missing_columns_by_file=missing_by_file,
            label_column=resolved_label,
            has_consistent_label=has_consistent_label,
            total_source_rows=total_rows,
            total_columns_count=len(common_cols),
            rejection_reasons=rejection_reasons,
            warnings=warnings,
        )

    def audit_sources(self, file_paths: List[Path]) -> SchemaAuditResult:
        """Audit schemas directly from raw CSV file paths."""
        import pandas as pd
        from src.data.ingestion_manifest import ColumnMetadata, DatasetRecord

        records: List[DatasetRecord] = []
        for p in file_paths:
            path_obj = Path(p)
            df_head = pd.read_csv(path_obj, nrows=1)
            cols = [
                ColumnMetadata(
                    name=c,
                    normalized_name=normalize_column_name(c),
                    dtype=str(df_head[c].dtype),
                )
                for c in df_head.columns
            ]
            has_label = any(normalize_column_name(c) == "label" for c in df_head.columns)
            label_name = next((c for c in df_head.columns if normalize_column_name(c) == "label"), None)
            rec = DatasetRecord(
                dataset_id=path_obj.stem,
                original_filename=path_obj.name,
                safe_filename=path_obj.name,
                file_path=f"data/raw/{path_obj.name}",
                file_size_bytes=path_obj.stat().st_size if path_obj.exists() else 0,
                file_size_formatted="0 MB",
                sha256_checksum="",
                import_timestamp="2026-11-17T00:00:00Z",
                total_rows=1,
                valid_rows=1,
                invalid_rows=0,
                duplicate_rows=0,
                numeric_anomaly_count=0,
                total_columns=len(cols),
                columns=cols,
                has_ground_truth_label=has_label,
                label_column_name=label_name,
                label_distribution={},
                validation_status="Validated",
                validation_errors=[],
                validation_warnings=[],
            )
            records.append(rec)
        return self.audit_datasets(records)
