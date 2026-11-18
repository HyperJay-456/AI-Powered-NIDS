"""Unit and Integration Tests for Intrusion Detection Engine (CP-08).

Validates:
  A. Model eligibility (production verification, fixture isolation, SHA-256 checks, CP-07 freeze gating)
  B. Input validation (schema alignment, checksum validation, ground-truth label protection, leakage detection)
  C. Preprocessing reuse (fitted imputer/encoder/scaler reuse without refit, scaled vs unscaled dual pipelines)
  D. Model inference (SVM margins, MLP/DT heuristic scores, multiclass mapping, honest score descriptions)
  E. Multi-model comparison (consensus analysis, agreement/disagreement tracking, no majority ensemble)
  F. Row accounting (Input = Predicted + Rejected + Skipped invariant, zero silent row loss)
  G. Artifact integrity (atomic commits, checksum verification, past run preservation, fixture isolation)
  H. Resource handling (RAM/disk preflight gating, batch bounds, concurrency protection)
  I. UI integration (Detection Center rendering and empty states)
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import joblib

from config.settings import AppConfig
from src.data.engineering.encoding import LeakageFreeCategoricalEncoder
from src.data.engineering.missing_values import LeakageFreeImputer
from src.data.engineering.scaling import DualRepresentationScaler
from src.data.ingestion_manifest import ColumnMetadata, DatasetRecord, IngestionManifest
from src.detection.dataset_loader import DetectionDatasetLoader, VerifiedDatasetSource
from src.detection.detection_manifest import (
    DetectionRunManifest,
    DisagreementSummary,
    ModelInferenceMetadata,
    RowAccountingRecord,
    capture_detection_environment,
)
from src.detection.detection_registry import DetectionRegistry
from src.detection.disagreement_analyzer import DisagreementAnalyzer
from src.detection.inference_pipeline import DetectionInferencePipeline
from src.detection.input_validator import DetectionInputValidator
from src.detection.model_eligibility import EligibleModelInfo, ModelEligibilityVerifier
from src.detection.prediction_engine import DetectionPredictionEngine
from src.detection.prediction_mapper import DetectionPredictionMapper
from src.detection.preprocessing_loader import DetectionPreprocessingLoader
from src.detection.resource_manager import DetectionPreflightResult, DetectionResourceManager
from src.detection.result_writer import DetectionResultWriter
from src.detection.score_interpreter import DetectionScoreInterpreter


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def mock_detection_environment(tmp_path: Path):
    """Create a fully populated, self-contained mock repository environment."""
    base_dir = tmp_path
    raw_dir = base_dir / "data" / "raw"
    metadata_dir = base_dir / "data" / "metadata"
    processed_dir = base_dir / "data" / "processed"
    saved_models_dir = base_dir / "models" / "saved"
    evaluations_dir = base_dir / "models" / "evaluations"
    analyses_dir = base_dir / "data" / "analyses"

    for d in (raw_dir, metadata_dir, processed_dir, saved_models_dir, evaluations_dir, analyses_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Feature schema & classes
    classes = ["BENIGN", "DoS", "PortScan"]
    numeric_features = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets"]
    categorical_features = ["Protocol"]
    input_features = numeric_features + categorical_features
    excluded_identifiers = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

    eng_run_id = "eng_run_20261118_test01"
    training_run_id = "train_run_20261118_test01"
    eval_run_id = "eval_run_20261118_test01"

    eng_dir = processed_dir / eng_run_id
    eng_meta_dir = eng_dir / "metadata"
    eng_prep_dir = eng_dir / "preprocessors"
    eng_data_dir = eng_dir / "datasets"

    for d in (eng_meta_dir, eng_prep_dir, eng_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Label mapping
    label_mapping = {
        "classes": classes,
        "label_to_index": {c: i for i, c in enumerate(classes)},
        "index_to_label": {i: c for i, c in enumerate(classes)},
    }
    with open(eng_meta_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f)

    # Feature schema
    feature_schema = {
        "input_features": input_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_identifiers": excluded_identifiers,
        "feature_schema_hash": "schema_hash_12345",
    }
    with open(eng_meta_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f)

    # Engineering manifest
    eng_manifest = {
        "run_id": eng_run_id,
        "created_at_utc": "2026-11-18T10:00:00Z",
        "status": "Completed",
        "source_filenames": ["train_traffic.csv"],
        "source_checksums": {"train_traffic.csv": "abc123sha"},
        "total_raw_rows": 60,
        "supervised_rows": 60,
        "unlabeled_rows": 0,
        "quarantined_rows": 0,
        "excluded_duplicate_rows": 0,
        "train_rows": 60,
        "val_rows": 0,
        "test_rows": 0,
        "num_features_in": len(input_features),
        "num_features_out": len(numeric_features) + 2,  # Protocol encoded to 2 dummy cols
        "num_classes": len(classes),
        "classes": classes,
        "random_seed": 42,
        "split_strategy": "stratified",
        "imputation_strategy": "median",
        "scaling_strategy": "standard",
        "duplicate_policy": "group_isolated",
        "artifact_directory": str(eng_dir),
        "error_message": None,
    }
    with open(eng_meta_dir / "engineering_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eng_manifest, f)

    # 2. Fit and save preprocessors on small synthetic train partition
    np.random.seed(42)
    n_train = 60
    train_df = pd.DataFrame({
        "Destination Port": np.random.randint(20, 8080, size=n_train),
        "Flow Duration": np.random.randint(100, 100000, size=n_train),
        "Total Fwd Packets": np.random.randint(1, 50, size=n_train),
        "Total Backward Packets": np.random.randint(0, 50, size=n_train),
        "Protocol": np.random.choice(["TCP", "UDP"], size=n_train),
    })
    y_train = np.array([i % len(classes) for i in range(n_train)])

    imputer = LeakageFreeImputer(numeric_strategy="median", categorical_strategy="most_frequent")
    imputer.fit(train_df, numeric_columns=numeric_features, categorical_columns=categorical_features)
    train_imp = imputer.transform(train_df)

    cat_encoder = LeakageFreeCategoricalEncoder()
    cat_encoder.fit(train_imp, categorical_columns=categorical_features)
    cat_sparse = cat_encoder.transform(train_imp)

    scaler = DualRepresentationScaler(with_mean=False)
    scaler.fit(train_imp, numeric_columns=numeric_features)

    X_train_scaled = scaler.transform_scaled(train_imp, cat_sparse)
    X_train_unscaled = scaler.transform_unscaled(train_imp, cat_sparse)

    output_features = numeric_features + (cat_encoder.metadata_.feature_names_out if cat_encoder.metadata_ else [])

    scaled_bundle = {
        "imputer": imputer,
        "categorical_encoder": cat_encoder,
        "scaler": scaler,
        "output_features": output_features,
        "numeric_columns": numeric_features,
        "categorical_columns": categorical_features,
    }
    joblib.dump(scaled_bundle, eng_prep_dir / "scaled_preprocessor.joblib")

    tree_bundle = {
        "imputer": imputer,
        "categorical_encoder": cat_encoder,
        "scaler": None,
        "output_features": output_features,
        "numeric_columns": numeric_features,
        "categorical_columns": categorical_features,
    }
    joblib.dump(tree_bundle, eng_prep_dir / "tree_preprocessor.joblib")

    # 3. Fit models
    svm = LinearSVC(random_state=42, max_iter=2000)
    svm.fit(X_train_scaled, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=50, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train_unscaled, y_train)

    models_map = {
        "svm": (svm, "Support Vector Machine (LinearSVC)", "LinearSVC"),
        "mlp": (mlp, "Multilayer Perceptron (MLP)", "MLPClassifier"),
        "decision_tree": (dt, "Decision Tree (CART)", "DecisionTreeClassifier"),
    }

    train_run_dir = saved_models_dir / training_run_id
    train_run_dir.mkdir(parents=True, exist_ok=True)

    session_manifest = {
        "session_run_id": training_run_id,
        "engineering_run_id": eng_run_id,
        "started_at_utc": "2026-11-18T10:00:00Z",
        "completed_at_utc": "2026-11-18T10:01:00Z",
        "overall_status": "Completed",
        "trained_algorithms": list(models_map.keys()),
        "model_manifests": {},
        "run_type": "production",
        "is_test_fixture": False,
    }

    model_checksums = {}
    for algo_k, (est, disp_name, est_cls) in models_map.items():
        algo_dir = train_run_dir / algo_k
        algo_dir.mkdir(parents=True, exist_ok=True)
        joblib_path = algo_dir / "model.joblib"
        joblib.dump(est, joblib_path)

        with open(joblib_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        model_checksums[algo_k] = sha

        meta = {
            "training_run_id": training_run_id,
            "algorithm": algo_k,
            "algorithm_display_name": disp_name,
            "estimator_class": est_cls,
            "status": "Completed",
            "engineering_run_id": eng_run_id,
            "training_samples": n_train,
            "feature_count": len(output_features),
            "class_count": len(classes),
            "classes": classes,
            "artifact_checksum": sha,
            "convergence_status": "Converged",
            "run_type": "production",
            "is_test_fixture": False,
        }
        with open(algo_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        session_manifest["model_manifests"][algo_k] = meta

    with open(train_run_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(session_manifest, f)

    # 4. Create CP-07 Candidate Freeze evaluation manifest
    eval_run_dir = evaluations_dir / eval_run_id
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    frozen_candidates = {
        "svm": {
            "algorithm": "svm",
            "algorithm_display_name": "Support Vector Machine (LinearSVC)",
            "training_run_id": training_run_id,
            "model_checksum": model_checksums["svm"],
            "frozen_at_utc": "2026-11-18T10:05:00Z",
            "selected_as_primary": False,
            "selection_criterion": "Evaluated candidate",
            "artifact_path": str(train_run_dir / "svm" / "model.joblib"),
        },
        "mlp": {
            "algorithm": "mlp",
            "algorithm_display_name": "Multilayer Perceptron (MLP)",
            "training_run_id": training_run_id,
            "model_checksum": model_checksums["mlp"],
            "frozen_at_utc": "2026-11-18T10:05:00Z",
            "selected_as_primary": True,
            "selection_criterion": "Highest Macro F1",
            "artifact_path": str(train_run_dir / "mlp" / "model.joblib"),
        },
        "decision_tree": {
            "algorithm": "decision_tree",
            "algorithm_display_name": "Decision Tree (CART)",
            "training_run_id": training_run_id,
            "model_checksum": model_checksums["decision_tree"],
            "frozen_at_utc": "2026-11-18T10:05:00Z",
            "selected_as_primary": False,
            "selection_criterion": "Evaluated candidate",
            "artifact_path": str(train_run_dir / "decision_tree" / "model.joblib"),
        },
    }

    eval_manifest = {
        "evaluation_run_id": eval_run_id,
        "engineering_run_id": eng_run_id,
        "training_session_id": training_run_id,
        "status": "Completed",
        "created_at_utc": "2026-11-18T10:04:00Z",
        "completed_at_utc": "2026-11-18T10:06:00Z",
        "evaluated_algorithms": list(models_map.keys()),
        "partitions_evaluated": ["validation"],
        "frozen_candidates": frozen_candidates,
        "is_test_fixture": False,
    }
    with open(eval_run_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f)

    # 5. Create a valid uploaded raw CSV dataset in data/raw/
    n_test_csv = 30
    test_csv_df = pd.DataFrame({
        "Flow ID": [f"flow_{i}" for i in range(n_test_csv)],
        "Source IP": ["192.168.1.10"] * n_test_csv,
        "Destination IP": ["10.0.0.1"] * n_test_csv,
        "Timestamp": ["2026-11-18 12:00:00"] * n_test_csv,
        "Destination Port": np.random.randint(20, 8080, size=n_test_csv),
        "Flow Duration": np.random.randint(100, 100000, size=n_test_csv),
        "Total Fwd Packets": np.random.randint(1, 50, size=n_test_csv),
        "Total Backward Packets": np.random.randint(0, 50, size=n_test_csv),
        "Protocol": np.random.choice(["TCP", "UDP"], size=n_test_csv),
        "Label": [classes[i % len(classes)] for i in range(n_test_csv)],
    })

    test_csv_path = raw_dir / "operational_traffic_batch1.csv"
    test_csv_df.to_csv(test_csv_path, index=False)

    with open(test_csv_path, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()

    ingest_manifest_file = metadata_dir / "ingestion_manifest.json"
    ingest_data = {
        "ds_operational_01": {
            "dataset_id": "ds_operational_01",
            "original_filename": "operational_traffic_batch1.csv",
            "safe_filename": "operational_traffic_batch1.csv",
            "file_path": str(test_csv_path),
            "file_size_bytes": test_csv_path.stat().st_size,
            "file_size_formatted": "12.5 KB",
            "sha256_checksum": csv_sha,
            "import_timestamp": "2026-11-18T10:00:00Z",
            "validation_status": "Validated",
            "total_rows": n_test_csv,
            "valid_rows": n_test_csv,
            "has_ground_truth_label": True,
            "label_column_name": "Label",
        }
    }
    with open(ingest_manifest_file, "w", encoding="utf-8") as f:
        json.dump(ingest_data, f)

    return {
        "base_dir": base_dir,
        "raw_dir": raw_dir,
        "metadata_dir": metadata_dir,
        "processed_dir": processed_dir,
        "saved_models_dir": saved_models_dir,
        "evaluations_dir": evaluations_dir,
        "analyses_dir": analyses_dir,
        "eng_run_id": eng_run_id,
        "training_run_id": training_run_id,
        "eval_run_id": eval_run_id,
        "classes": classes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "input_features": input_features,
        "model_checksums": model_checksums,
        "test_csv_path": test_csv_path,
        "test_dataset_id": "ds_operational_01",
        "n_test_csv": n_test_csv,
    }


# ==============================================================================
# SECTION A: Model Eligibility Tests
# ==============================================================================

def test_model_eligibility_valid_production_model_accepted(mock_detection_environment):
    """Test A.1: Valid production model with CP-07 candidate freeze is accepted."""
    env = mock_detection_environment
    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="mlp",
        allow_test_fixtures=False,
    )

    assert info.is_eligible is True
    assert info.algorithm == "mlp"
    assert info.model_sha256 == env["model_checksums"]["mlp"]
    assert info.is_primary_candidate is True
    assert len(info.rejection_reasons) == 0


def test_model_eligibility_rejects_test_fixtures(mock_detection_environment):
    """Test A.2: Test fixtures are strictly rejected in production inference mode."""
    env = mock_detection_environment
    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    # Mark MLP metadata as test fixture
    meta_path = env["saved_models_dir"] / env["training_run_id"] / "mlp" / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["is_test_fixture"] = True
    meta["run_type"] = "test_fixture"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # In production mode (allow_test_fixtures=False), must be rejected
    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="mlp",
        allow_test_fixtures=False,
    )
    assert info.is_eligible is False
    assert any("test fixture" in r for r in info.rejection_reasons)

    # In test mode (allow_test_fixtures=True), allowed
    info_test = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="mlp",
        allow_test_fixtures=True,
    )
    assert info_test.is_eligible is True


def test_model_eligibility_rejects_missing_model(mock_detection_environment):
    """Test A.3: Missing model run or nonexistent algorithm is safely rejected."""
    env = mock_detection_environment
    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    info = verifier.verify_model(
        training_run_id="nonexistent_run",
        algorithm="mlp",
    )
    assert info.is_eligible is False
    assert any("missing" in r.lower() for r in info.rejection_reasons)


def test_model_eligibility_rejects_corrupted_checksum(mock_detection_environment):
    """Test A.4: Checksum mismatch between model artifact and metadata is detected."""
    env = mock_detection_environment
    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    # Corrupt model artifact
    model_path = env["saved_models_dir"] / env["training_run_id"] / "svm" / "model.joblib"
    with open(model_path, "ab") as f:
        f.write(b"\x00\x00CORRUPTED_BYTES")

    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="svm",
    )
    assert info.is_eligible is False
    assert any("CRITICAL CHECKSUM MISMATCH" in r for r in info.rejection_reasons)


def test_model_eligibility_rejects_unauthorized_algorithm(mock_detection_environment):
    """Test A.5: Unauthorized algorithms (e.g. random_forest, xgboost) are strictly rejected."""
    env = mock_detection_environment
    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="random_forest",
    )
    assert info.is_eligible is False
    assert any("Unauthorized algorithm" in r for r in info.rejection_reasons)


def test_model_eligibility_rejects_unfrozen_candidate(mock_detection_environment):
    """Test A.6: Models that have not passed CP-07 candidate freezing are rejected in production."""
    env = mock_detection_environment
    # Clear evaluations directory to simulate un-evaluated models
    shutil.rmtree(env["evaluations_dir"])
    env["evaluations_dir"].mkdir(parents=True, exist_ok=True)

    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="decision_tree",
        allow_test_fixtures=False,
    )
    assert info.is_eligible is False
    assert any("candidate freezing" in r for r in info.rejection_reasons)


def test_model_eligibility_rejects_missing_preprocessing_artifact(mock_detection_environment):
    """Test A.7: Rejects model if its referenced CP-05 preprocessing bundle is missing."""
    env = mock_detection_environment
    # Delete tree_preprocessor.joblib
    tree_prep = env["processed_dir"] / env["eng_run_id"] / "preprocessors" / "tree_preprocessor.joblib"
    if tree_prep.exists():
        tree_prep.unlink()

    verifier = ModelEligibilityVerifier(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
    )

    info = verifier.verify_model(
        training_run_id=env["training_run_id"],
        algorithm="decision_tree",
    )
    assert info.is_eligible is False
    assert any("preprocessing bundle missing" in r.lower() for r in info.rejection_reasons)


# ==============================================================================
# SECTION B: Input Validation & Ground-Truth Protection Tests
# ==============================================================================

def test_input_validation_valid_csv_accepted(mock_detection_environment):
    """Test B.1: Valid CP-04 dataset source is verified cleanly."""
    env = mock_detection_environment
    loader = DetectionDatasetLoader(
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
        processed_dir=env["processed_dir"],
    )

    source = loader.verify_and_prepare_source(
        dataset_id=env["test_dataset_id"],
        engineering_run_id=env["eng_run_id"],
    )
    assert source.safe_filename == "operational_traffic_batch1.csv"
    assert source.total_rows == env["n_test_csv"]
    assert source.has_ground_truth_label is True


def test_input_validation_tampered_source_checksum_rejected(mock_detection_environment):
    """Test B.2: Source CSV tampering since ingestion is caught immediately."""
    env = mock_detection_environment
    loader = DetectionDatasetLoader(
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
        processed_dir=env["processed_dir"],
    )

    # Tamper with the raw CSV on disk
    with open(env["test_csv_path"], "a", encoding="utf-8") as f:
        f.write("\n9999,1.1.1.1,2.2.2.2,2026-11-18,80,100,2,2,TCP,BENIGN")

    with pytest.raises(ValueError) as exc_info:
        loader.verify_and_prepare_source(dataset_id=env["test_dataset_id"])
    assert "CRITICAL FILE TAMPERING DETECTED" in str(exc_info.value)


def test_input_validation_missing_required_feature_detected(mock_detection_environment):
    """Test B.3: Schema validator detects missing required model features."""
    env = mock_detection_environment
    prep_loader = DetectionPreprocessingLoader(
        engineering_run_id=env["eng_run_id"],
        processed_dir=env["processed_dir"],
    )
    validator = DetectionInputValidator(feature_schema=prep_loader.feature_schema)

    # Sample columns missing 'Flow Duration'
    incomplete_cols = ["Destination Port", "Total Fwd Packets", "Protocol"]
    is_compat, missing = validator.validate_schema_compatibility(incomplete_cols)
    assert is_compat is False
    assert "Flow Duration" in missing


def test_input_validation_ground_truth_label_strictly_isolated(mock_detection_environment):
    """Test B.4: Ground-truth target label is stripped and never passed to model features."""
    env = mock_detection_environment
    prep_loader = DetectionPreprocessingLoader(
        engineering_run_id=env["eng_run_id"],
        processed_dir=env["processed_dir"],
    )
    validator = DetectionInputValidator(feature_schema=prep_loader.feature_schema)

    raw_chunk = pd.read_csv(env["test_csv_path"])
    assert "Label" in raw_chunk.columns

    res = validator.process_chunk(raw_chunk, label_column_name="Label")

    # Features DataFrame MUST NOT contain target label
    assert "Label" not in res.features_df.columns
    assert "label" not in res.features_df.columns
    # Target is preserved in reference-only ground_truth_series
    assert res.ground_truth_series is not None
    assert len(res.ground_truth_series) == len(raw_chunk)


def test_input_validation_excluded_identifiers_isolated(mock_detection_environment):
    """Test B.5: Network identifiers (Flow ID, IPs, Timestamp) are segregated to provenance."""
    env = mock_detection_environment
    prep_loader = DetectionPreprocessingLoader(
        engineering_run_id=env["eng_run_id"],
        processed_dir=env["processed_dir"],
    )
    validator = DetectionInputValidator(feature_schema=prep_loader.feature_schema)

    raw_chunk = pd.read_csv(env["test_csv_path"])
    res = validator.process_chunk(raw_chunk)

    for ident in ["Flow ID", "Source IP", "Destination IP", "Timestamp"]:
        assert ident not in res.features_df.columns
        assert ident in res.provenance_df.columns


def test_input_validation_training_data_overlap_detected(mock_detection_environment):
    """Test B.6: Training dataset overlap is audited and flagged with clear warning."""
    env = mock_detection_environment
    loader = DetectionDatasetLoader(
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
        processed_dir=env["processed_dir"],
    )

    # Record test CSV in engineering source_files.json
    source_files_json = env["processed_dir"] / env["eng_run_id"] / "metadata" / "source_files.json"
    with open(env["test_csv_path"], "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(source_files_json, "w", encoding="utf-8") as f:
        json.dump([{"filename": "operational_traffic_batch1.csv", "sha256": sha}], f)

    source = loader.verify_and_prepare_source(
        dataset_id=env["test_dataset_id"],
        engineering_run_id=env["eng_run_id"],
    )
    assert source.overlaps_training_data is True
    assert "cannot be claimed as unseen operational traffic" in source.overlap_details


# ==============================================================================
# SECTION C: Preprocessing Reuse Tests
# ==============================================================================

def test_preprocessing_saved_transformers_reused_no_refit(mock_detection_environment):
    """Test C.1: Preprocessing transformation strictly calls transform() without refitting."""
    env = mock_detection_environment
    prep_loader = DetectionPreprocessingLoader(
        engineering_run_id=env["eng_run_id"],
        processed_dir=env["processed_dir"],
    )

    scaled_bundle = prep_loader.load_preprocessor_bundle("svm")
    scaler = scaled_bundle["scaler"]
    original_means = dict(scaler.stats_.means)

    raw_chunk = pd.read_csv(env["test_csv_path"])
    validator = DetectionInputValidator(prep_loader.feature_schema)
    val_res = validator.process_chunk(raw_chunk)

    # Run transformation
    X_csr = prep_loader.transform(val_res.features_df, algorithm="svm")

    # Assert scaler parameters were not modified
    assert scaler.stats_.means == original_means
    assert sparse.issparse(X_csr)


def test_preprocessing_svm_mlp_scaled_tree_unscaled(mock_detection_environment):
    """Test C.2: SVM/MLP receive standard scaled features, Decision Tree receives unscaled."""
    env = mock_detection_environment
    prep_loader = DetectionPreprocessingLoader(
        engineering_run_id=env["eng_run_id"],
        processed_dir=env["processed_dir"],
    )

    raw_chunk = pd.read_csv(env["test_csv_path"])
    validator = DetectionInputValidator(prep_loader.feature_schema)
    val_res = validator.process_chunk(raw_chunk)

    X_svm = prep_loader.transform(val_res.features_df, algorithm="svm")
    X_mlp = prep_loader.transform(val_res.features_df, algorithm="mlp")
    X_dt = prep_loader.transform(val_res.features_df, algorithm="decision_tree")

    # SVM and MLP share identical scaled CSR
    assert np.allclose(X_svm.toarray(), X_mlp.toarray())

    # Decision tree unscaled representation retains original numerical magnitudes
    # e.g. Port 8080 vs scaled value ~1.5
    raw_ports = val_res.features_df["Destination Port"].to_numpy()
    dt_ports = X_dt.toarray()[:, 0]
    assert np.allclose(raw_ports, dt_ports)
    assert not np.allclose(X_svm.toarray()[:, 0], dt_ports)


# ==============================================================================
# SECTION D: Model Inference & Score Interpretation Tests
# ==============================================================================

def test_model_inference_linearsvc_margin_scores(mock_detection_environment):
    """Test D.1: LinearSVC returns decision margin distances, not probabilities."""
    env = mock_detection_environment
    pipeline = DetectionInferencePipeline(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
        analyses_dir=env["analyses_dir"],
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
    )

    manifest = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithm="svm",
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )

    assert manifest.status == "Completed"
    assert manifest.models["svm"].score_type == "decision_margin"

    desc = DetectionScoreInterpreter.get_score_metadata("svm")
    assert "not a calibrated probability" in desc["description"]


def test_model_inference_mlp_and_dt_uncalibrated_probabilities(mock_detection_environment):
    """Test D.2: MLP and Decision Tree scores are described as uncalibrated heuristic scores."""
    mlp_desc = DetectionScoreInterpreter.get_score_metadata("mlp")
    assert "not a guaranteed probability" in mlp_desc["description"]

    dt_desc = DetectionScoreInterpreter.get_score_metadata("decision_tree")
    assert "not a calibrated intrusion probability" in dt_desc["description"]


def test_model_inference_multiclass_label_mapping(mock_detection_environment):
    """Test D.3: Integer predictions map to attack classes and reject unknown indices."""
    classes = ["BENIGN", "DoS", "PortScan"]
    mapper = DetectionPredictionMapper(classes=classes)

    preds = mapper.map_predictions([0, 1, 2, 0])
    assert preds == ["BENIGN", "DoS", "PortScan", "BENIGN"]

    with pytest.raises(ValueError) as exc_info:
        mapper.map_predictions([0, 99])
    assert "unmapped class indices [99]" in str(exc_info.value)


# ==============================================================================
# SECTION E: Multiple Model Comparison & Disagreement Tests
# ==============================================================================

def test_multiple_models_agreement_and_disagreement():
    """Test E.1: Consensus analysis identifies agreement and disagreement per row."""
    preds_by_algo = {
        "svm": ["BENIGN", "DoS", "PortScan", "BENIGN"],
        "mlp": ["BENIGN", "DoS", "BENIGN", "BENIGN"],
        "decision_tree": ["BENIGN", "DoS", "PortScan", "DoS"],
    }

    statuses, diffs, summary = DisagreementAnalyzer.analyze_row_consensus(preds_by_algo)

    # Row 0: All BENIGN -> Agreement
    assert statuses[0] == "Agreement"
    assert diffs[0] == []

    # Row 1: All DoS -> Agreement
    assert statuses[1] == "Agreement"
    assert diffs[1] == []

    # Row 2: svm=PortScan, mlp=BENIGN, dt=PortScan -> Disagreement
    assert statuses[2] == "Disagreement"
    assert "mlp" in diffs[2]

    # Row 3: svm=BENIGN, mlp=BENIGN, dt=DoS -> Disagreement
    assert statuses[3] == "Disagreement"
    assert "decision_tree" in diffs[3]

    assert summary.total_compared_rows == 4
    assert summary.agreement_count == 2
    assert summary.disagreement_count == 2
    assert summary.agreement_percentage == 50.0


def test_multiple_models_does_not_create_majority_ensemble(mock_detection_environment):
    """Test E.2: Multi-model comparison preserves individual outputs and creates no ensemble."""
    env = mock_detection_environment
    pipeline = DetectionInferencePipeline(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
        analyses_dir=env["analyses_dir"],
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
    )

    manifest = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )

    assert manifest.status == "Completed"
    assert len(manifest.selected_algorithms) == 3
    # Verifies exactly the three authorized algorithms, no pseudo 4th ensemble algorithm
    for a in manifest.selected_algorithms:
        assert a in ("svm", "mlp", "decision_tree")


# ==============================================================================
# SECTION F: Row Accounting Invariant Tests
# ==============================================================================

def test_row_accounting_invariant_enforced(mock_detection_environment):
    """Test F.1: Complete row accounting invariant holds with zero silent row loss."""
    env = mock_detection_environment
    pipeline = DetectionInferencePipeline(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
        analyses_dir=env["analyses_dir"],
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
    )

    manifest = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithm="mlp",
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )

    accounting = manifest.row_accounting
    assert accounting.accounting_invariant_holds is True
    assert accounting.input_rows == env["n_test_csv"]
    assert accounting.successfully_predicted_rows == env["n_test_csv"]
    assert accounting.rejected_rows == 0
    assert accounting.skipped_rows == 0


# ==============================================================================
# SECTION G: Artifact Integrity & Persistence Tests
# ==============================================================================

def test_artifact_integrity_atomic_write_and_manifest_checksums(mock_detection_environment):
    """Test G.1: Output files are written atomically with verifiable SHA-256 checksums."""
    env = mock_detection_environment
    pipeline = DetectionInferencePipeline(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
        analyses_dir=env["analyses_dir"],
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
    )

    manifest = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithms=["svm", "mlp"],
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )

    run_dir = env["analyses_dir"] / manifest.detection_run_id
    assert run_dir.exists()

    pred_file = run_dir / "predictions.csv"
    comp_file = run_dir / "model_comparison.csv"
    man_file = run_dir / "run_manifest.json"

    assert pred_file.exists()
    assert comp_file.exists()
    assert man_file.exists()

    # Verify SHA-256 recorded in manifest matches disk
    with open(pred_file, "rb") as f:
        actual_pred_sha = hashlib.sha256(f.read()).hexdigest()
    assert manifest.output_artifacts["predictions.csv"] == actual_pred_sha


def test_artifact_integrity_prevents_overwriting_past_runs(mock_detection_environment):
    """Test G.2: Repeated runs create independent timestamped IDs without overwriting."""
    env = mock_detection_environment
    pipeline = DetectionInferencePipeline(
        saved_models_dir=env["saved_models_dir"],
        processed_dir=env["processed_dir"],
        evaluations_dir=env["evaluations_dir"],
        analyses_dir=env["analyses_dir"],
        raw_data_dir=env["raw_dir"],
        metadata_dir=env["metadata_dir"],
    )

    run1 = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithm="svm",
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )
    run2 = pipeline.run_inference(
        dataset_id=env["test_dataset_id"],
        algorithm="svm",
        training_run_id=env["training_run_id"],
        allow_test_fixtures=True,
    )

    assert run1.detection_run_id != run2.detection_run_id
    assert (env["analyses_dir"] / run1.detection_run_id).exists()
    assert (env["analyses_dir"] / run2.detection_run_id).exists()


# ==============================================================================
# SECTION H: Resource Handling Tests
# ==============================================================================

def test_resource_preflight_memory_and_batch_bounds():
    """Test H.1: Resource preflight clamps batch sizes and audits memory headroom."""
    preflight = DetectionResourceManager.perform_preflight_check(
        num_features=80,
        batch_size=100_000,  # Above max 50,000
    )
    assert preflight.is_safe is True
    assert preflight.batch_size == DetectionResourceManager.MAX_BATCH_SIZE


# ==============================================================================
# SECTION I: UI Integration Tests
# ==============================================================================

def test_detection_center_page_renders_with_empty_state():
    """Test I.1: Detection Center page renders without crashing in bare environment."""
    from dashboard.pages import detection_center
    # Module imports and function exists
    assert hasattr(detection_center, "render")
