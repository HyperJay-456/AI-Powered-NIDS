"""Comprehensive Automated Test Suite for CP-05 Dataset Engineering & Preprocessing Pipeline.

Tests all 12 modules under src/data/engineering/ and their integration with settings,
status checking, and Streamlit UI components, strictly enforcing all zero-leakage,
dual-scaling, group-isolation, and row-accounting invariants.
"""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from config.settings import PreprocessingConfig, settings
from dashboard.status_checker import (
    DatasetEngineeringStatus,
    check_dataset_engineering_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)
from src.data.engineering.artifact_manager import (
    ArtifactManager,
    EngineeringRunRecord,
)
from src.data.engineering.dataset_registry import (
    DatasetIntegrityCheck,
    DatasetRegistry,
)
from src.data.engineering.dataset_splitter import (
    PartitionIndices,
    ReproducibleDatasetSplitter,
    SplittingStrategy,
)
from src.data.engineering.duplicate_policy import (
    DuplicateHandlingPolicy,
    DuplicatePolicyManager,
    compute_feature_fingerprints,
)
from src.data.engineering.encoding import LeakageFreeCategoricalEncoder
from src.data.engineering.engineering_report import (
    EngineeringReportGenerator,
    RowAccountingReport,
)
from src.data.engineering.feature_engineering import FeatureSanitizer
from src.data.engineering.feature_policy import (
    ColumnRole,
    FeaturePolicyManager,
)
from src.data.engineering.label_processor import GroundTruthLabelProcessor
from src.data.engineering.missing_values import (
    LeakageFreeImputer,
    analyze_missing_values,
)
from src.data.engineering.preprocessing_pipeline import (
    PreprocessingPipeline,
    PreprocessingRunConfiguration,
)
from src.data.engineering.scaling import DualRepresentationScaler
from src.data.engineering.source_auditor import SourceAuditor
from src.data.ingestion_manifest import IngestionManifest


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_flow_dataframe() -> pd.DataFrame:
    """Create a realistic synthetic flow dataframe with identifiers, numerics, categoricals, and labels."""
    np.random.seed(42)
    n_rows = 120

    data = {
        "Flow ID": [f"192.168.1.{i}-10.0.0.1-443-{i}-6" for i in range(n_rows)],
        "Source IP": [f"192.168.1.{i % 20}" for i in range(n_rows)],
        "Destination IP": ["10.0.0.1"] * n_rows,
        "Timestamp": ["07/07/2017 8:55:12 AM"] * n_rows,
        "Source Port": [1024 + (i % 50) for i in range(n_rows)],
        "Destination Port": [443 if i % 2 == 0 else 80 for i in range(n_rows)],
        "Protocol": ["TCP" if i % 3 != 0 else "UDP" for i in range(n_rows)],
        "Flow Duration": [float(i * 100) for i in range(n_rows)],
        "Total Fwd Packets": [float(i + 1) for i in range(n_rows)],
        "Total Backward Packets": [float(i * 2) for i in range(n_rows)],
        "Flow Bytes/s": [float(i * 10.5) for i in range(n_rows)],
        "Flow Packets/s": [float(i * 1.5) for i in range(n_rows)],
        "Fwd Packet Length Mean": [float(50.0 + i) for i in range(n_rows)],
        "Label": ["BENIGN" if i % 3 != 0 else "DDoS" for i in range(n_rows)],
    }
    # Introduce intentional anomalies:
    # 1. Non-finite values
    data["Flow Bytes/s"][5] = np.inf
    data["Flow Packets/s"][10] = -np.inf
    data["Flow Duration"][15] = np.nan
    # 2. Duplicate rows (indices 20 and 21)
    data["Flow Duration"][21] = data["Flow Duration"][20]
    data["Total Fwd Packets"][21] = data["Total Fwd Packets"][20]
    data["Total Backward Packets"][21] = data["Total Backward Packets"][20]
    data["Flow Bytes/s"][21] = data["Flow Bytes/s"][20]
    data["Flow Packets/s"][21] = data["Flow Packets/s"][20]
    data["Fwd Packet Length Mean"][21] = data["Fwd Packet Length Mean"][20]
    data["Protocol"][21] = data["Protocol"][20]
    data["Label"][21] = data["Label"][20]
    # 3. Explicit zero values (must be preserved!)
    data["Total Backward Packets"][0] = 0.0

    return pd.DataFrame(data)


@pytest.fixture
def temp_project_dir():
    """Create isolated temporary directory tree for testing artifacts and datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        raw_dir = root / "data" / "raw"
        processed_dir = root / "data" / "processed"
        meta_dir = root / "data" / "metadata"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        yield {
            "root": root,
            "raw": raw_dir,
            "processed": processed_dir,
            "meta": meta_dir,
        }


# ==============================================================================
# 1. DATASET REGISTRY & INTEGRITY TESTS
# ==============================================================================


def test_dataset_registry_eligible_sources(temp_project_dir, sample_flow_dataframe):
    """Test registry identification of validated sources against manifest and disk."""
    raw_dir = temp_project_dir["raw"]
    meta_dir = temp_project_dir["meta"]

    csv_path = raw_dir / "test_flow.csv"
    sample_flow_dataframe.to_csv(csv_path, index=False)

    import hashlib
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    manifest_file = meta_dir / "ingestion_manifest.json"
    manifest_data = {
        "test_flow.csv": {
            "filename": "test_flow.csv",
            "file_size_bytes": csv_path.stat().st_size,
            "sha256_checksum": sha,
            "total_rows": len(sample_flow_dataframe),
            "column_count": len(sample_flow_dataframe.columns),
            "validation_status": "Validated",
            "is_valid": True,
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    registry = DatasetRegistry(raw_dir=raw_dir, metadata_dir=meta_dir)
    sources = registry.get_eligible_sources()

    assert len(sources) == 1
    assert sources[0].filename == "test_flow.csv"
    assert sources[0].sha256_checksum == sha
    assert registry.verify_source_integrity("test_flow.csv") is True


def test_dataset_registry_sha256_corruption_detection(temp_project_dir, sample_flow_dataframe):
    """Test registry detects byte corruption via live SHA-256 calculation."""
    raw_dir = temp_project_dir["raw"]
    meta_dir = temp_project_dir["meta"]

    csv_path = raw_dir / "corrupted.csv"
    sample_flow_dataframe.to_csv(csv_path, index=False)

    manifest_file = meta_dir / "ingestion_manifest.json"
    manifest_data = {
        "corrupted.csv": {
            "filename": "corrupted.csv",
            "file_size_bytes": csv_path.stat().st_size,
            "sha256_checksum": "fake_checksum_00000000000000000000000000000000000000000000000000",
            "total_rows": len(sample_flow_dataframe),
            "column_count": len(sample_flow_dataframe.columns),
            "validation_status": "Validated",
            "is_valid": True,
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    registry = DatasetRegistry(raw_dir=raw_dir, metadata_dir=meta_dir)
    assert registry.verify_source_integrity("corrupted.csv") is False


# ==============================================================================
# 2. SOURCE AUDITOR SCHEMA RECONCILIATION TESTS
# ==============================================================================


def test_source_auditor_schema_matching(temp_project_dir, sample_flow_dataframe):
    """Test schema reconciliation on identical schemas."""
    raw_dir = temp_project_dir["raw"]
    p1 = raw_dir / "file1.csv"
    p2 = raw_dir / "file2.csv"
    sample_flow_dataframe.to_csv(p1, index=False)
    sample_flow_dataframe.to_csv(p2, index=False)

    auditor = SourceAuditor()
    res = auditor.audit_sources([p1, p2])

    assert res.is_compatible is True
    assert len(res.common_columns) == len(sample_flow_dataframe.columns)
    assert len(res.incompatible_reasons) == 0


def test_source_auditor_schema_mismatch(temp_project_dir, sample_flow_dataframe):
    """Test schema reconciliation detects and reports missing columns."""
    raw_dir = temp_project_dir["raw"]
    p1 = raw_dir / "full.csv"
    p2 = raw_dir / "missing_col.csv"
    sample_flow_dataframe.to_csv(p1, index=False)

    df_subset = sample_flow_dataframe.drop(columns=["Flow Duration"])
    df_subset.to_csv(p2, index=False)

    auditor = SourceAuditor()
    res = auditor.audit_sources([p1, p2])

    assert res.is_compatible is False
    assert any("flow duration" in r.lower() for r in res.incompatible_reasons)


# ==============================================================================
# 3. MISSING VALUES & ANOMALIES TESTS
# ==============================================================================


def test_missing_values_detection_and_infinity(sample_flow_dataframe):
    """Test detection of NaN, None, and positive/negative infinity."""
    report = analyze_missing_values(sample_flow_dataframe)

    assert report.total_infinite_cells == 2  # inf and -inf
    assert report.num_columns_with_missing > 0
    assert "Flow Duration" in report.missing_cells_per_column


def test_zero_preservation_in_imputation():
    """Verify explicit zero values are preserved and NOT converted to NaN or imputed."""
    df_train = pd.DataFrame({
        "feature_a": [0.0, 0.0, 10.0, 20.0, np.nan],
        "feature_b": [5.0, 5.0, 5.0, 5.0, 5.0],
    })
    imputer = LeakageFreeImputer(numeric_strategy="median")
    imputer.fit(df_train)
    res = imputer.transform(df_train)

    # Median of [0.0, 0.0, 10.0, 20.0] is 5.0
    assert res["feature_a"].iloc[0] == 0.0
    assert res["feature_a"].iloc[1] == 0.0
    assert res["feature_a"].iloc[4] == 5.0  # Imputed value


def test_leakage_free_imputer_train_only():
    """Verify imputer statistics are calculated strictly on training partition."""
    df_train = pd.DataFrame({"val": [10.0, 20.0, 30.0]})  # median = 20.0
    df_val = pd.DataFrame({"val": [np.nan, 1000000.0]})    # massive outlier in val

    imputer = LeakageFreeImputer(numeric_strategy="median")
    imputer.fit(df_train)
    val_transformed = imputer.transform(df_val)

    # Imputed value in validation must be train median (20.0), unaffected by 1,000,000.0
    assert val_transformed["val"].iloc[0] == 20.0


# ==============================================================================
# 4. FEATURE POLICY & IDENTIFIER EXCLUSION TESTS
# ==============================================================================


def test_feature_policy_column_classification(sample_flow_dataframe):
    """Verify network identifiers, ports, timestamps, numerics, and labels are correctly assigned roles."""
    decisions = FeaturePolicyManager.classify_columns(list(sample_flow_dataframe.columns))
    role_map = {d.column_name: d.role for d in decisions}

    assert role_map["Flow ID"] == ColumnRole.IDENTIFIER
    assert role_map["Source IP"] == ColumnRole.IDENTIFIER
    assert role_map["Destination IP"] == ColumnRole.IDENTIFIER
    assert role_map["Timestamp"] == ColumnRole.TIMESTAMP
    assert role_map["Source Port"] == ColumnRole.PORT
    assert role_map["Destination Port"] == ColumnRole.PORT
    assert role_map["Protocol"] == ColumnRole.CATEGORICAL_FEATURE
    assert role_map["Flow Duration"] == ColumnRole.NUMERIC_FEATURE
    assert role_map["Label"] == ColumnRole.LABEL


def test_feature_sanitizer_provenance_sidecar(sample_flow_dataframe):
    """Verify FeatureSanitizer strips identifiers into provenance sidecar and retains model features."""
    manifest = FeaturePolicyManager.create_manifest(list(sample_flow_dataframe.columns))
    sanitizer = FeatureSanitizer(policy_manifest=manifest)

    cleaned_df, sidecar_df, quarantined_idx = sanitizer.sanitize(sample_flow_dataframe)

    assert "Flow ID" not in cleaned_df.columns
    assert "Source IP" not in cleaned_df.columns
    assert "Destination IP" not in cleaned_df.columns
    assert "Timestamp" not in cleaned_df.columns
    assert "Flow ID" in sidecar_df.columns
    assert "Source IP" in sidecar_df.columns
    assert len(sidecar_df) == len(sample_flow_dataframe)


# ==============================================================================
# 5. DUPLICATE HANDLING & GROUP ISOLATION TESTS
# ==============================================================================


def test_duplicate_policy_group_isolated(sample_flow_dataframe):
    """Verify duplicate records are assigned to the exact same partition under group isolation."""
    # Ensure rows 20 and 21 are exact duplicate features
    mgr = DuplicatePolicyManager(policy=DuplicateHandlingPolicy.GROUP_ISOLATED)
    feature_cols = [c for c in sample_flow_dataframe.columns if c != "Label"]
    report = mgr.audit_duplicates(sample_flow_dataframe, feature_cols=feature_cols, label_col="Label")

    assert report.total_groups > 0
    splitter = ReproducibleDatasetSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        strategy=SplittingStrategy.GROUP_ISOLATED,
    )

    group_ids = mgr.generate_group_ids(sample_flow_dataframe, feature_cols=feature_cols)
    partitions, split_report = splitter.split(
        df=sample_flow_dataframe,
        target_series=sample_flow_dataframe["Label"],
        groups=group_ids,
    )

    # Check that rows with same group_id never land in different partitions
    train_groups = set(group_ids[partitions.train_indices])
    val_groups = set(group_ids[partitions.val_indices])
    test_groups = set(group_ids[partitions.test_indices])

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_duplicate_policy_exclude():
    """Verify EXCLUDE duplicate policy drops duplicate rows."""
    df = pd.DataFrame({
        "feat1": [1, 2, 2, 3],
        "feat2": [10, 20, 20, 30],
        "Label": ["A", "B", "B", "C"],
    })
    mgr = DuplicatePolicyManager(policy=DuplicateHandlingPolicy.EXCLUDE)
    filtered_df, excluded_idx = mgr.apply_policy(df, feature_cols=["feat1", "feat2"])

    assert len(filtered_df) == 3
    assert len(excluded_idx) == 1


# ==============================================================================
# 6. SPARSE CATEGORICAL ENCODING & UNSEEN CATEGORIES
# ==============================================================================


def test_categorical_encoder_train_fitting_and_unseen():
    """Verify encoder fits on train only and ignores unseen categories without crashing."""
    train_df = pd.DataFrame({"protocol": ["TCP", "UDP", "TCP"]})
    test_df = pd.DataFrame({"protocol": ["TCP", "ICMP", "UDP"]})  # 'ICMP' is unseen

    encoder = LeakageFreeCategoricalEncoder(categorical_columns=["protocol"])
    train_csr = encoder.fit_transform(train_df)
    test_csr = encoder.transform(test_df)

    assert sparse.issparse(train_csr)
    assert sparse.issparse(test_csr)
    assert test_csr.shape[0] == 3
    assert test_csr.shape[1] == train_csr.shape[1]


# ==============================================================================
# 7. DUAL REPRESENTATION SCALING TESTS
# ==============================================================================


def test_dual_representation_scaler():
    """Verify DualRepresentationScaler produces scaled (with_mean=False) and unscaled matrices."""
    X_num = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]])
    scaler = DualRepresentationScaler()
    scaled_csr, unscaled_csr = scaler.fit_transform(X_num)

    assert sparse.issparse(scaled_csr)
    assert sparse.issparse(unscaled_csr)
    assert scaled_csr.shape == unscaled_csr.shape == (3, 2)

    # Scaled values should have unit variance
    dense_scaled = scaled_csr.toarray()
    assert np.allclose(np.std(dense_scaled, axis=0), [1.0, 1.0], atol=1e-4)
    # Unscaled values must match original magnitudes exactly
    assert np.array_equal(unscaled_csr.toarray(), X_num)


# ==============================================================================
# 8. DATA LEAKAGE PREVENTION PROOF
# ==============================================================================


def test_data_leakage_prevention_proof():
    """Prove that injecting massive anomalies into validation/test does not affect training statistics."""
    train_data = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    val_data = pd.DataFrame({"f1": [999999999.0]})  # Extreme outlier

    imputer = LeakageFreeImputer(numeric_strategy="median")
    scaler = DualRepresentationScaler()

    # Fit solely on training
    imputer.fit(train_data)
    imputed_train = imputer.transform(train_data)
    scaled_train, _ = scaler.fit_transform(imputed_train)

    train_median_before = imputer.column_statistics_["f1"]
    train_scale_before = scaler.fitted_scaler_.scale_[0]

    # Transform validation
    imputed_val = imputer.transform(val_data)
    scaled_val, _ = scaler.transform(imputed_val)

    # Verify train statistics are 100% invariant
    assert imputer.column_statistics_["f1"] == train_median_before == 3.0
    assert scaler.fitted_scaler_.scale_[0] == train_scale_before


# ==============================================================================
# 9. GROUND TRUTH LABEL ENCODING & CLASS IMBALANCE
# ==============================================================================


def test_ground_truth_label_processor():
    """Verify deterministic label mapping, inverse mapping, and class imbalance metrics without SMOTE."""
    labels = pd.Series(["BENIGN", "DDoS", "BENIGN", "PortScan", "BENIGN", None])

    processor = GroundTruthLabelProcessor()
    encoded, valid_mask = processor.fit_transform(labels)

    assert processor.num_classes == 3
    assert valid_mask.sum() == 5  # None row is masked out (not imputed)
    assert len(encoded) == 5

    # Reverse lookup
    inv = processor.inverse_transform(encoded)
    assert list(inv) == ["BENIGN", "DDoS", "BENIGN", "PortScan", "BENIGN"]

    # Imbalance report
    imb_report = processor.compute_imbalance_report(labels.dropna())
    assert imb_report.majority_class == "BENIGN"
    assert imb_report.majority_count == 3
    assert imb_report.imbalance_ratio > 1.0


# ==============================================================================
# 10. ROW ACCOUNTING INVARIANTS & AUDIT REPORT
# ==============================================================================


def test_row_accounting_invariants():
    """Verify strict row accounting equations hold exactly with zero discrepancy."""
    report = EngineeringReportGenerator.generate_row_accounting(
        total_raw=1000,
        supervised=800,
        unlabeled=100,
        quarantined=50,
        excluded_dups=50,
        train_count=560,
        val_count=120,
        test_count=120,
    )

    assert report.invariant_raw_reconciled is True
    assert report.invariant_supervised_reconciled is True
    assert report.total_raw_rows == 1000
    assert report.supervised_rows == 800


def test_row_accounting_discrepancy_detection():
    """Verify discrepancy is flagged if row accounting equation does not balance."""
    report = EngineeringReportGenerator.generate_row_accounting(
        total_raw=1000,
        supervised=800,
        unlabeled=100,
        quarantined=50,
        excluded_dups=40,  # 10 rows missing!
        train_count=560,
        val_count=120,
        test_count=120,
    )

    assert report.invariant_raw_reconciled is False
    assert any("Row Accounting Error" in n for n in report.notes)


# ==============================================================================
# 11. ATOMIC ARTIFACT PERSISTENCE & LOADING TESTS
# ==============================================================================


def test_artifact_manager_atomic_save_and_load(temp_project_dir):
    """Test atomic serialization of run metadata, preprocessors, and datasets."""
    processed_dir = temp_project_dir["processed"]
    manager = ArtifactManager(processed_dir=processed_dir)

    run_id = ArtifactManager.generate_run_id()
    record = EngineeringRunRecord(
        run_id=run_id,
        created_at_utc="2026-11-17T12:00:00Z",
        status="Completed",
        source_filenames=["flow.csv"],
        source_checksums={"flow.csv": "abc123sha"},
        total_raw_rows=100,
        supervised_rows=90,
        unlabeled_rows=5,
        quarantined_rows=5,
        excluded_duplicate_rows=0,
        train_rows=63,
        val_rows=14,
        test_rows=13,
        num_features_in=15,
        num_features_out=18,
        num_classes=2,
        classes=["BENIGN", "Attack"],
        random_seed=42,
        split_strategy="group_isolated",
        imputation_strategy="median",
        scaling_strategy="dual_standard",
        duplicate_policy="group_isolated",
        artifact_directory=str(processed_dir / run_id),
    )

    metadata_dict = {"feature_schema.json": {"num_features": 18}}
    preprocessors = {"scaled_preprocessor.joblib": {"dummy": "scaler"}}
    datasets = {
        "train": {
            "scaled": sparse.csr_matrix(np.zeros((63, 18))),
            "unscaled": sparse.csr_matrix(np.zeros((63, 18))),
            "labels": np.zeros(63, dtype=int),
        },
        "validation": {
            "scaled": sparse.csr_matrix(np.zeros((14, 18))),
            "unscaled": sparse.csr_matrix(np.zeros((14, 18))),
            "labels": np.zeros(14, dtype=int),
        },
        "test": {
            "scaled": sparse.csr_matrix(np.zeros((13, 18))),
            "unscaled": sparse.csr_matrix(np.zeros((13, 18))),
            "labels": np.zeros(13, dtype=int),
        },
    }
    report_dict = {"run_id": run_id}

    saved_path = manager.save_run_artifacts(
        run_record=record,
        metadata_dict=metadata_dict,
        preprocessor_bundle=preprocessors,
        datasets=datasets,
        report_dict=report_dict,
    )

    assert saved_path.exists()
    runs = manager.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id
    assert runs[0].supervised_rows == 90


# ==============================================================================
# 12. FULL END-TO-END PREPROCESSING PIPELINE INTEGRATION
# ==============================================================================


def test_end_to_end_preprocessing_pipeline(temp_project_dir, sample_flow_dataframe):
    """Test complete end-to-end dataset engineering pipeline execution and verification."""
    raw_dir = temp_project_dir["raw"]
    processed_dir = temp_project_dir["processed"]
    meta_dir = temp_project_dir["meta"]

    # Save CSV
    csv_name = "test_traffic.csv"
    csv_path = raw_dir / csv_name
    sample_flow_dataframe.to_csv(csv_path, index=False)

    import hashlib
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # Manifest
    manifest_file = meta_dir / "ingestion_manifest.json"
    manifest_data = {
        csv_name: {
            "filename": csv_name,
            "file_size_bytes": csv_path.stat().st_size,
            "sha256_checksum": sha,
            "total_rows": len(sample_flow_dataframe),
            "column_count": len(sample_flow_dataframe.columns),
            "validation_status": "Validated",
            "is_valid": True,
            "has_ground_truth_label": True,
            "label_column_name": "Label",
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    config = PreprocessingRunConfiguration(
        source_filenames=[csv_name],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        duplicate_policy=DuplicateHandlingPolicy.GROUP_ISOLATED,
        numeric_imputer_strategy="median",
        categorical_imputer_strategy="most_frequent",
        enable_dual_scaling=True,
    )

    pipeline = PreprocessingPipeline(
        config=config,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        metadata_dir=meta_dir,
    )

    steps_logged = []

    def on_progress(step, total, name, msg):
        steps_logged.append((step, name))

    run_record = pipeline.execute_pipeline(progress_callback=on_progress)

    assert run_record.status == "Completed"
    assert run_record.total_raw_rows == len(sample_flow_dataframe)
    assert run_record.supervised_rows > 0
    assert run_record.train_rows + run_record.val_rows + run_record.test_rows == run_record.supervised_rows
    assert len(steps_logged) >= 8

    # Verify status checker picks up completed run
    eng_status = check_dataset_engineering_status(processed_dir=processed_dir)
    assert eng_status.is_completed is True
    assert eng_status.state == "Completed"
    assert eng_status.run_count == 1
    assert eng_status.train_rows == run_record.train_rows
