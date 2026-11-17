"""Engineering Audit Report Generator Module (CP-05).

Produces transparent, auditable row-accounting reports and data-engineering
summaries to ensure zero silent row loss and complete data provenance:
  Total Raw Rows = Supervised Included + Unlabeled Pool + Quarantined + Duplicate Excluded
  Supervised Included = Train Partition + Validation Partition + Test Partition
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.engineering.artifact_manager import EngineeringRunRecord
from src.data.engineering.duplicate_policy import DuplicateAuditReport
from src.data.engineering.feature_policy import SchemaPolicyManifest
from src.data.engineering.label_processor import ClassImbalanceReport
from src.data.engineering.missing_values import MissingValueAuditReport
from src.data.engineering.source_auditor import SchemaAuditResult


@dataclass
class RowAccountingReport:
    """Rigorous row disposition audit ensuring no record is silently discarded."""

    total_raw_rows: int
    supervised_rows: int
    unlabeled_pool_rows: int
    quarantined_invalid_rows: int
    excluded_duplicate_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    invariant_raw_reconciled: bool
    invariant_supervised_reconciled: bool
    disposition_breakdown: Dict[str, int]
    notes: List[str] = field(default_factory=list)


@dataclass
class PreprocessingAuditReport:
    """Consolidated master audit report for an engineering run."""

    run_id: str
    timestamp_utc: str
    status: str
    source_files_count: int
    source_filenames: List[str]
    row_accounting: RowAccountingReport
    missing_values_audit: Dict[str, Any]
    duplicate_audit: Dict[str, Any]
    feature_policy_audit: Dict[str, Any]
    class_imbalance_audit: Dict[str, Any]
    splitting_audit: Dict[str, Any]
    output_feature_count: int
    output_feature_names: List[str]
    dual_representations_created: List[str]  # ["Scaled (SVM/MLP)", "Unscaled (Decision Tree)"]
    data_leakage_prevented: bool
    leakage_safeguard_notes: List[str]
    artifact_directory: str
    warnings: List[str] = field(default_factory=list)

    @property
    def total_raw_rows(self) -> int:
        return self.row_accounting.total_raw_rows

    @property
    def supervised_rows(self) -> int:
        return self.row_accounting.supervised_rows

    @property
    def train_rows(self) -> int:
        return self.row_accounting.train_rows

    @property
    def val_rows(self) -> int:
        return self.row_accounting.val_rows

    @property
    def test_rows(self) -> int:
        return self.row_accounting.test_rows

    @property
    def num_classes(self) -> int:
        return self.class_imbalance_audit.get("num_classes", 0)

    @property
    def classes(self) -> List[str]:
        return [c.get("class_name", "") for c in self.class_imbalance_audit.get("class_distributions", [])]

    @property
    def num_features_out(self) -> int:
        return self.output_feature_count


class EngineeringReportGenerator:
    """Assembles structured audit reports from pipeline execution results."""

    @staticmethod
    def generate_row_accounting(
        total_raw: int,
        supervised: int,
        unlabeled: int,
        quarantined: int,
        excluded_dups: int,
        train_count: int,
        val_count: int,
        test_count: int,
    ) -> RowAccountingReport:
        """Verify and construct a strict row-accounting audit.

        Invariants:
          1. total_raw == supervised + unlabeled + quarantined + excluded_dups
          2. supervised == train_count + val_count + test_count
        """
        raw_sum = supervised + unlabeled + quarantined + excluded_dups
        supervised_sum = train_count + val_count + test_count

        raw_reconciled = (raw_sum == total_raw)
        sup_reconciled = (supervised_sum == supervised)

        notes: List[str] = []
        if not raw_reconciled:
            notes.append(
                f"Row Accounting Error: Raw rows ({total_raw}) != sum of dispositions ({raw_sum}). "
                f"Difference: {total_raw - raw_sum} rows unaccounted!"
            )
        else:
            notes.append("Raw row disposition invariant verified: 100% of input records accounted for.")

        if not sup_reconciled:
            notes.append(
                f"Partition Accounting Error: Supervised rows ({supervised}) != "
                f"train + val + test ({supervised_sum})."
            )
        else:
            notes.append("Partition invariant verified: Supervised rows match partition sum exactly.")

        breakdown = {
            "Included in Supervised Dataset": supervised,
            "Segregated to Unlabeled Pool (Missing Labels)": unlabeled,
            "Quarantined (Structural Defects)": quarantined,
            "Excluded (Duplicate Policy)": excluded_dups,
        }

        return RowAccountingReport(
            total_raw_rows=total_raw,
            supervised_rows=supervised,
            unlabeled_pool_rows=unlabeled,
            quarantined_invalid_rows=quarantined,
            excluded_duplicate_rows=excluded_dups,
            train_rows=train_count,
            val_rows=val_count,
            test_rows=test_count,
            invariant_raw_reconciled=raw_reconciled,
            invariant_supervised_reconciled=sup_reconciled,
            disposition_breakdown=breakdown,
            notes=notes,
        )

    @classmethod
    def assemble_report(
        cls,
        run_record: EngineeringRunRecord,
        row_accounting: RowAccountingReport,
        missing_audit: MissingValueAuditReport,
        dup_audit: DuplicateAuditReport,
        schema_manifest: SchemaPolicyManifest,
        imbalance_audit: ClassImbalanceReport,
        splitting_audit: Dict[str, Any],
        output_feature_names: List[str],
        artifact_dir: str,
        warnings: Optional[List[str]] = None,
    ) -> PreprocessingAuditReport:
        """Construct the master PreprocessingAuditReport."""
        warnings_list = list(warnings or [])
        warnings_list.extend(missing_audit.has_infinite_values and ["Infinite flow rate values detected and cleaned."] or [])
        warnings_list.extend(imbalance_audit.warnings)

        leakage_safeguards = [
            "Dataset partitioning occurred BEFORE fitting any imputation, scaling, or encoding parameters.",
            "Numeric medians fitted exclusively on training partition records.",
            "StandardScaler parameters (scale and variance) fitted exclusively on training partition records.",
            "OneHotEncoder category vocabularies fitted exclusively on training partition records.",
            "Group-isolated duplicate policy prevents identical feature records from spanning train/val/test partitions.",
            "Identifiers and timestamps excluded from model feature matrix to prevent memorization leakage.",
        ]

        return PreprocessingAuditReport(
            run_id=run_record.run_id,
            timestamp_utc=run_record.created_at_utc,
            status=run_record.status,
            source_files_count=len(run_record.source_filenames),
            source_filenames=run_record.source_filenames,
            row_accounting=row_accounting,
            missing_values_audit=asdict(missing_audit),
            duplicate_audit=asdict(dup_audit),
            feature_policy_audit=asdict(schema_manifest),
            class_imbalance_audit=asdict(imbalance_audit),
            splitting_audit=splitting_audit,
            output_feature_count=len(output_feature_names),
            output_feature_names=output_feature_names,
            dual_representations_created=[
                "Scaled Representation (StandardScaler without mean, sparse CSR for SVM & MLP)",
                "Unscaled Representation (Imputed original magnitude, sparse CSR for Decision Tree)",
            ],
            data_leakage_prevented=True,
            leakage_safeguard_notes=leakage_safeguards,
            artifact_directory=artifact_dir,
            warnings=warnings_list,
        )
