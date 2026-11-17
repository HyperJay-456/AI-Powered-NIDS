"""Preprocessing & Dataset Engineering Pipeline Orchestrator (CP-05).

Coordinates the end-to-end transformation of validated network flow CSV datasets
into reproducible, model-ready train, validation, and test partitions:
  - Enforces raw dataset immutability (read-only access to data/raw/)
  - Enforces strict row accounting (zero silent row loss)
  - Enforces zero data leakage (splitting occurs before fitting any preprocessor)
  - Generates dual representations: Scaled (SVM/MLP) and Unscaled (Decision Tree)
  - Persists atomic, versioned artifacts to data/processed/<engineering_run_id>/
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from config.settings import settings
from src.data.engineering.artifact_manager import (
    ArtifactManager,
    EngineeringRunRecord,
)
from src.data.engineering.dataset_registry import DatasetRegistry
from src.data.engineering.dataset_splitter import (
    PartitionIndices,
    ReproducibleDatasetSplitter,
    SplittingStrategy,
)
from src.data.engineering.duplicate_policy import (
    DuplicateHandlingPolicy,
    DuplicatePolicyManager,
)
from src.data.engineering.encoding import LeakageFreeCategoricalEncoder
from src.data.engineering.engineering_report import (
    EngineeringReportGenerator,
    PreprocessingAuditReport,
    RowAccountingReport,
)
from src.data.engineering.feature_engineering import FeatureSanitizer
from src.data.engineering.feature_policy import (
    FeaturePolicyManager,
    SchemaPolicyManifest,
)
from src.data.engineering.label_processor import GroundTruthLabelProcessor
from src.data.engineering.missing_values import (
    LeakageFreeImputer,
    analyze_missing_values,
)
from src.data.engineering.scaling import DualRepresentationScaler
from src.data.engineering.source_auditor import SchemaAuditResult, SourceAuditor
from src.data.ingestion_manifest import DatasetRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingRunConfiguration:
    """Configurable parameters for a dataset engineering execution run."""

    source_filenames: List[str] = field(default_factory=list)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    duplicate_policy: DuplicateHandlingPolicy = DuplicateHandlingPolicy.GROUP_ISOLATED
    splitting_strategy: SplittingStrategy = SplittingStrategy.STRATIFIED
    include_destination_port: bool = True
    include_source_port: bool = False
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    numeric_imputer_strategy: Optional[str] = None
    categorical_imputer_strategy: Optional[str] = None
    scaler_with_mean: bool = False  # False preserves sparse matrix efficiency for SVM/MLP
    enable_dual_scaling: bool = True
    chunk_size: int = 50_000

    def __post_init__(self) -> None:
        if self.numeric_imputer_strategy:
            self.numeric_imputation_strategy = self.numeric_imputer_strategy
        if self.categorical_imputer_strategy:
            self.categorical_imputation_strategy = self.categorical_imputer_strategy


class PreprocessingPipeline:
    """Master orchestrator for the Dataset Engineering and Preprocessing subsystem."""

    def __init__(
        self,
        config: Optional[PreprocessingRunConfiguration] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        metadata_dir: Optional[Path] = None,
    ):
        """Initialize pipeline with storage directories."""
        self.config = config or PreprocessingRunConfiguration()
        self.raw_dir = raw_dir or settings.raw_data_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.metadata_dir = metadata_dir or settings.metadata_dir

        self.registry = DatasetRegistry(raw_dir=self.raw_dir, metadata_dir=self.metadata_dir)
        self.artifact_mgr = ArtifactManager(processed_dir=self.processed_dir)

    def execute_pipeline(
        self,
        selected_dataset_ids: Optional[List[str]] = None,
        config: Optional[PreprocessingRunConfiguration] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> PreprocessingAuditReport:
        """Execute full dataset engineering pipeline on selected cataloged datasets.

        Args:
            selected_dataset_ids: Optional list of dataset IDs from the CP-04 ingestion manifest.
            config: Optional PreprocessingRunConfiguration. Defaults to project standard.
            progress_callback: Optional callback receiving progress updates.

        Returns:
            PreprocessingAuditReport with complete metrics, row dispositions, and artifact paths.
        """
        cfg = config or self.config
        run_id = ArtifactManager.generate_run_id()
        now_utc = datetime.now(timezone.utc).isoformat()

        def notify(step: int, total: int, stage_name: str, message: str) -> None:
            if not progress_callback:
                return
            pct = step / float(total)
            try:
                progress_callback(step, total, stage_name, message)
            except TypeError:
                try:
                    progress_callback(pct, message)
                except Exception:
                    pass

        target_ids = selected_dataset_ids if selected_dataset_ids is not None else cfg.source_filenames
        if not target_ids:
            target_ids = [s.safe_filename for s in self.registry.get_eligible_sources()]

        logger.info(f"Starting Dataset Engineering run '{run_id}' on {len(target_ids)} dataset(s).")
        notify(1, 8, "Source Verification", "Auditing selected source datasets and checking disk integrity...")

        # ----------------------------------------------------------------------
        # STAGE 1: Source Dataset Integrity & Eligibility Audit
        # ----------------------------------------------------------------------
        records: List[DatasetRecord] = []
        source_checksums: Dict[str, str] = {}

        for ds_id in target_ids:
            rec = self.registry.get_record(ds_id)
            if not rec:
                for r in self.registry.list_all_records():
                    if r.safe_filename == ds_id or r.original_filename == ds_id:
                        rec = r
                        break
            if not rec:
                fpath = self.raw_dir / ds_id
                if fpath.exists():
                    from src.data.ingestion_manifest import ColumnMetadata
                    df_head = pd.read_csv(fpath, nrows=5)
                    full_len = sum(1 for _ in open(fpath, "r", encoding="utf-8")) - 1
                    hasher = hashlib.sha256(fpath.read_bytes())
                    rec = DatasetRecord(
                        dataset_id=fpath.stem,
                        original_filename=fpath.name,
                        safe_filename=fpath.name,
                        file_path=f"data/raw/{fpath.name}",
                        file_size_bytes=fpath.stat().st_size,
                        file_size_formatted=f"{fpath.stat().st_size / (1024*1024):.2f} MB",
                        sha256_checksum=hasher.hexdigest(),
                        import_timestamp=now_utc,
                        total_rows=full_len,
                        valid_rows=full_len,
                        invalid_rows=0,
                        duplicate_rows=0,
                        numeric_anomaly_count=0,
                        total_columns=len(df_head.columns),
                        columns=[ColumnMetadata(name=c, normalized_name=c.lower().strip(), dtype=str(df_head[c].dtype)) for c in df_head.columns],
                        has_ground_truth_label="Label" in df_head.columns,
                        label_column_name="Label" if "Label" in df_head.columns else None,
                        label_distribution={},
                        validation_status="Validated",
                        validation_errors=[],
                        validation_warnings=[],
                    )

            if not rec:
                raise ValueError(f"Dataset ID '{ds_id}' not found in ingestion manifest or disk.")

            check = self.registry.verify_source_integrity(rec)
            if not check.is_eligible:
                raise ValueError(f"Dataset '{rec.safe_filename}' is not eligible: {check.rejection_reason}")

            records.append(rec)
            source_checksums[rec.safe_filename] = rec.sha256_checksum

        # ----------------------------------------------------------------------
        # STAGE 2: Multi-Dataset Schema Reconciliation
        # ----------------------------------------------------------------------
        notify(2, 8, "Schema Reconciliation", "Reconciling schema compatibility across sources...")

        schema_audit = SourceAuditor.audit_datasets(records)
        if not schema_audit.is_compatible:
            raise ValueError(f"Schema compatibility error: {'; '.join(schema_audit.rejection_reasons)}")

        if not schema_audit.has_consistent_label or not schema_audit.label_column:
            raise ValueError("Supervised dataset engineering requires a ground-truth 'Label' column.")

        # ----------------------------------------------------------------------
        # STAGE 3: Load Raw Datasets (Immutable Read) & Row Accounting Init
        # ----------------------------------------------------------------------
        notify(3, 8, "Telemetry Ingestion", "Reading raw flow telemetry chunks...")

        dfs: List[pd.DataFrame] = []
        total_raw_rows = 0

        for rec in records:
            file_path = self.raw_dir / rec.safe_filename
            # Read in bounded chunks to manage memory
            chunks: List[pd.DataFrame] = []
            for chunk in pd.read_csv(
                file_path,
                chunksize=cfg.chunk_size,
                encoding=rec.encoding or "utf-8",
                low_memory=False,
            ):
                chunks.append(chunk)

            file_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            total_raw_rows += len(file_df)
            dfs.append(file_df)

        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # ----------------------------------------------------------------------
        # STAGE 4: Row Accounting & Missing Label Segregation
        # ----------------------------------------------------------------------
        notify(4, 8, "Row Accounting", "Classifying row dispositions and segregating missing labels...")

        label_col = schema_audit.label_column
        raw_labels = combined_df[label_col].astype(str).str.strip()

        # Identify missing or null labels
        missing_label_mask = (
            combined_df[label_col].isna()
            | (raw_labels == "")
            | (raw_labels.str.lower() == "nan")
            | (raw_labels.str.lower() == "null")
        )

        unlabeled_count = int(missing_label_mask.sum())
        quarantined_count = 0  # Completed validated datasets from CP-04 have 0 structural errors

        # Supervised candidate subset
        supervised_df = combined_df[~missing_label_mask].reset_index(drop=True)
        unlabeled_df = combined_df[missing_label_mask].reset_index(drop=True)

        # ----------------------------------------------------------------------
        # STAGE 5: Missing Values & Anomaly Analysis
        # ----------------------------------------------------------------------
        notify(4, 8, "Missing Values", "Auditing missing cells and non-finite numeric rates...")

        missing_audit = analyze_missing_values(supervised_df, label_col=label_col)

        # ----------------------------------------------------------------------
        # STAGE 6: Feature Policy & Leakage-Risk Evaluation
        # ----------------------------------------------------------------------
        notify(5, 8, "Feature Policy", "Evaluating column roles and excluding network identifiers...")

        col_dtypes = {c: str(supervised_df[c].dtype) for c in supervised_df.columns}
        policy_mgr = FeaturePolicyManager(
            include_destination_port=cfg.include_destination_port,
            include_source_port=cfg.include_source_port,
        )
        schema_manifest = policy_mgr.evaluate_schema(
            columns=list(supervised_df.columns),
            dtypes=col_dtypes,
            label_column=label_col,
        )

        # ----------------------------------------------------------------------
        # STAGE 7: Feature Sanitization & Duplicate Policy
        # ----------------------------------------------------------------------
        notify(5, 8, "Feature Sanitization", "Applying duplicate isolation policy and computing fingerprints...")

        features_raw, target_series, provenance_raw = FeatureSanitizer.sanitize_and_split_features(
            supervised_df,
            schema_manifest,
        )

        # Apply duplicate policy
        (
            features_clean,
            group_fingerprints,
            dup_audit,
        ) = DuplicatePolicyManager.audit_and_apply_policy(
            features_raw,
            feature_columns=schema_manifest.model_feature_columns,
            label_column=None,
            policy=cfg.duplicate_policy,
        )

        # Synchronize target and provenance after duplicate policy
        target_clean = target_series.iloc[features_clean.index].reset_index(drop=True)
        provenance_clean = provenance_raw.iloc[features_clean.index].reset_index(drop=True)
        features_clean = features_clean.reset_index(drop=True)

        excluded_dup_rows = dup_audit.records_excluded
        supervised_final_count = len(features_clean)

        # ----------------------------------------------------------------------
        # STAGE 8: Ground-Truth Label Encoding & Imbalance Analysis
        # ----------------------------------------------------------------------
        notify(6, 8, "Label Encoding", "Encoding ground-truth classes and analyzing imbalance...")

        label_proc = GroundTruthLabelProcessor()
        label_proc.fit(target_clean, column_name=label_col)
        y_all = label_proc.transform(target_clean)
        imbalance_audit = label_proc.analyze_imbalance(target_clean)

        # ----------------------------------------------------------------------
        # STAGE 9: STRICT DATA LEAKAGE PREVENTION — Dataset Partitioning
        # ----------------------------------------------------------------------
        notify(7, 8, "Partition Splitting", "Partitioning Train (70%), Val (15%), Test (15%) before preprocessor fitting...")

        splitter = ReproducibleDatasetSplitter(
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            random_seed=cfg.random_seed,
        )

        split_groups = group_fingerprints if cfg.duplicate_policy == DuplicateHandlingPolicy.GROUP_ISOLATED else None
        split_indices, splitting_audit = splitter.split(
            y_all,
            groups=split_groups,
            strategy=cfg.splitting_strategy,
        )

        train_idx = split_indices.train_indices
        val_idx = split_indices.val_indices
        test_idx = split_indices.test_indices

        # Partition feature DataFrames and targets
        X_train_df = features_clean.iloc[train_idx].reset_index(drop=True)
        X_val_df = features_clean.iloc[val_idx].reset_index(drop=True)
        X_test_df = features_clean.iloc[test_idx].reset_index(drop=True)

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]

        prov_train = provenance_clean.iloc[train_idx].reset_index(drop=True)
        prov_val = provenance_clean.iloc[val_idx].reset_index(drop=True)
        prov_test = provenance_clean.iloc[test_idx].reset_index(drop=True)

        # ----------------------------------------------------------------------
        # STAGE 10: Fit Preprocessors EXCLUSIVELY on Training Partition
        # ----------------------------------------------------------------------
        notify(8, 8, "Preprocessor Fitting", "Fitting imputers, encoders, and scalers strictly on training records...")

        num_cols = schema_manifest.numeric_feature_columns
        cat_cols = schema_manifest.categorical_feature_columns

        # 1. Leakage-free Imputer (learned on train only)
        imputer = LeakageFreeImputer(
            numeric_strategy=cfg.numeric_imputation_strategy,
            categorical_strategy=cfg.categorical_imputation_strategy,
        )
        imputer.fit(X_train_df, numeric_columns=num_cols, categorical_columns=cat_cols)

        X_train_imp = imputer.transform(X_train_df)
        X_val_imp = imputer.transform(X_val_df)
        X_test_imp = imputer.transform(X_test_df)

        # 2. Categorical OneHotEncoder (learned on train only)
        cat_encoder = LeakageFreeCategoricalEncoder()
        cat_encoder.fit(X_train_imp, categorical_columns=cat_cols)

        cat_sparse_train = cat_encoder.transform(X_train_imp)
        cat_sparse_val = cat_encoder.transform(X_val_imp)
        cat_sparse_test = cat_encoder.transform(X_test_imp)

        # 3. Scaler & Dual Representations (learned on train only)
        scaler = DualRepresentationScaler(with_mean=cfg.scaler_with_mean)
        scaler.fit(X_train_imp, numeric_columns=num_cols)

        # Generate Scaled Representation (for SVM & MLP)
        X_train_scaled = scaler.transform_scaled(X_train_imp, cat_sparse_train)
        X_val_scaled = scaler.transform_scaled(X_val_imp, cat_sparse_val)
        X_test_scaled = scaler.transform_scaled(X_test_imp, cat_sparse_test)

        # Generate Unscaled Representation (for Decision Tree)
        X_train_unscaled = scaler.transform_unscaled(X_train_imp, cat_sparse_train)
        X_val_unscaled = scaler.transform_unscaled(X_val_imp, cat_sparse_val)
        X_test_unscaled = scaler.transform_unscaled(X_test_imp, cat_sparse_test)

        # Assembled feature names in exact column order
        encoded_cat_names = cat_encoder.metadata_.feature_names_out if cat_encoder.metadata_ else []
        output_feature_names = num_cols + encoded_cat_names

        # ----------------------------------------------------------------------
        # STAGE 11: Master Row Accounting & Audit Assembly
        # ----------------------------------------------------------------------
        notify(8, 8, "Row Accounting Audit", "Compiling row-accounting audit and provenance logs...")

        row_accounting = EngineeringReportGenerator.generate_row_accounting(
            total_raw=total_raw_rows,
            supervised=supervised_final_count,
            unlabeled=unlabeled_count,
            quarantined=quarantined_count,
            excluded_dups=excluded_dup_rows,
            train_count=len(train_idx),
            val_count=len(val_idx),
            test_count=len(test_idx),
        )

        run_record = EngineeringRunRecord(
            run_id=run_id,
            created_at_utc=now_utc,
            status="Completed",
            source_filenames=[r.safe_filename for r in records],
            source_checksums=source_checksums,
            total_raw_rows=total_raw_rows,
            supervised_rows=supervised_final_count,
            unlabeled_rows=unlabeled_count,
            quarantined_rows=quarantined_count,
            excluded_duplicate_rows=excluded_dup_rows,
            train_rows=len(train_idx),
            val_rows=len(val_idx),
            test_rows=len(test_idx),
            num_features_in=len(schema_manifest.model_feature_columns),
            num_features_out=len(output_feature_names),
            num_classes=label_proc.mapping_.total_classes if label_proc.mapping_ else 0,
            classes=label_proc.mapping_.classes if label_proc.mapping_ else [],
            random_seed=cfg.random_seed,
            split_strategy=cfg.splitting_strategy.value,
            imputation_strategy=cfg.numeric_imputation_strategy,
            scaling_strategy="StandardScaler (with_mean=False, sparse)",
            duplicate_policy=cfg.duplicate_policy.value,
            artifact_directory=str(self.processed_dir / run_id),
            error_message=None,
        )

        # ----------------------------------------------------------------------
        # STAGE 12: Atomic Artifact Persistence
        # ----------------------------------------------------------------------
        notify(8, 8, "Artifact Persistence", f"Atomically persisting artifacts to {run_id}...")

        metadata_dict = {
            "source_files": [
                {
                    "filename": r.safe_filename,
                    "sha256": r.sha256_checksum,
                    "valid_rows": r.valid_rows,
                    "columns": r.total_columns,
                }
                for r in records
            ],
            "feature_schema": {
                "input_features": schema_manifest.model_feature_columns,
                "numeric_features": schema_manifest.numeric_feature_columns,
                "categorical_features": schema_manifest.categorical_feature_columns,
                "excluded_identifiers": schema_manifest.excluded_identifier_columns,
                "output_features": output_feature_names,
                "total_output_features": len(output_feature_names),
            },
            "label_mapping": asdict(label_proc.mapping_) if label_proc.mapping_ else {},
            "preprocessing_config": {
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "random_seed": cfg.random_seed,
                "duplicate_policy": cfg.duplicate_policy.value,
                "splitting_strategy": cfg.splitting_strategy.value,
                "imputation_strategy": cfg.numeric_imputation_strategy,
                "scaling_strategy": "standard",
            },
            "row_accounting": asdict(row_accounting),
            "class_distribution": asdict(imbalance_audit),
        }

        preprocessor_bundle = {
            "scaled_preprocessor": {
                "imputer": imputer,
                "categorical_encoder": cat_encoder,
                "scaler": scaler,
                "output_features": output_feature_names,
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
            },
            "tree_preprocessor": {
                "imputer": imputer,
                "categorical_encoder": cat_encoder,
                "scaler": None,  # Unscaled for Decision Tree
                "output_features": output_feature_names,
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
            },
            "label_processor": label_proc,
        }

        datasets = {
            "train": {
                "X_scaled": X_train_scaled,
                "X_unscaled": X_train_unscaled,
                "y": y_train,
                "provenance": prov_train,
            },
            "validation": {
                "X_scaled": X_val_scaled,
                "X_unscaled": X_val_unscaled,
                "y": y_val,
                "provenance": prov_val,
            },
            "test": {
                "X_scaled": X_test_scaled,
                "X_unscaled": X_test_unscaled,
                "y": y_test,
                "provenance": prov_test,
            },
        }

        master_report = EngineeringReportGenerator.assemble_report(
            run_record=run_record,
            row_accounting=row_accounting,
            missing_audit=missing_audit,
            dup_audit=dup_audit,
            schema_manifest=schema_manifest,
            imbalance_audit=imbalance_audit,
            splitting_audit=asdict(splitting_audit),
            output_feature_names=output_feature_names,
            artifact_dir=str(self.processed_dir / run_id),
        )

        self.artifact_mgr.save_run_artifacts(
            run_record=run_record,
            metadata_dict=metadata_dict,
            preprocessor_bundle=preprocessor_bundle,
            datasets=datasets,
            report_dict=asdict(master_report),
        )

        notify(8, 8, "Completed", f"Dataset Engineering run '{run_id}' completed successfully!")

        logger.info(
            f"Dataset Engineering run '{run_id}' completed: {row_accounting.supervised_rows:,} supervised rows "
            f"({row_accounting.train_rows:,} train, {row_accounting.val_rows:,} val, {row_accounting.test_rows:,} test), "
            f"{len(output_feature_names)} features, {run_record.num_classes} classes."
        )

        return master_report
