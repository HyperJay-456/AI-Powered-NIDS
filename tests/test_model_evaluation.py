"""Unit and Integration Tests for Model Evaluation Laboratory (CP-07).

Validates:
1. Production model artifact eligibility, checksum verification, and test-fixture isolation.
2. Zero-leakage data isolation (validation/test only, strictly forbidding training data).
3. Multiclass classification metrics, per-class error rates, and zero-division handling.
4. Confusion matrix row-normalization and sample-count conservation.
5. One-vs-rest ROC and Precision-Recall analysis (LinearSVC margin scores vs MLP/DT probabilities).
6. Security-oriented Benign vs Attack summary metrics.
7. Multi-model comparison and generalization gap analysis.
8. Protocol gating: Candidate model freezing and gated final holdout test evaluation.
9. Sparse matrix batch inference preservation and row-accounting invariant.
10. Evaluation registry atomic persistence and discovery.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, List
import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import joblib

from config.settings import AppConfig
from src.evaluation.artifact_verifier import ArtifactVerifier, ModelEligibilityCheckResult
from src.evaluation.confusion_matrix import ConfusionMatrixEngine
from src.evaluation.evaluation_data_loader import EvaluationDataLoader, LoadedEvaluationPartition
from src.evaluation.evaluation_manifest import (
    AggregateMetricsRecord,
    AlgorithmEvaluationResult,
    CandidateModelFreezeRecord,
    ConfusionMatrixRecord,
    EvaluationRunManifest,
    PerClassMetricRecord,
    PrecisionRecallMetricRecord,
    ROCMetricRecord,
    SecurityMetricRecord,
    capture_evaluation_environment,
)
from src.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.evaluation.evaluation_registry import EvaluationRegistry
from src.evaluation.inference_runner import BatchInferenceResult, InferenceRunner
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.model_comparator import ModelComparator
from src.evaluation.precision_recall_analysis import PrecisionRecallAnalyzer
from src.evaluation.resource_manager import EvaluationPreflightResult, EvaluationResourceManager
from src.evaluation.roc_analysis import ROCAnalyzer
from src.evaluation.security_metrics import SecurityMetricsEngine


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def mock_dataset_env(tmp_path: Path):
    """Create a mock CP-05 dataset environment with validation and test partitions."""
    processed_dir = tmp_path / "processed"
    eng_run_id = "eng_run_20261117_test01"
    run_dir = processed_dir / eng_run_id
    meta_dir = run_dir / "metadata"
    data_dir = run_dir / "datasets"

    for d in (meta_dir, data_dir / "train", data_dir / "validation", data_dir / "test"):
        d.mkdir(parents=True, exist_ok=True)

    classes = ["BENIGN", "DoS", "PortScan"]
    n_features = 8
    n_samples = 60

    # Label mapping
    label_mapping = {
        "classes": classes,
        "label_to_index": {c: i for i, c in enumerate(classes)},
        "index_to_label": {i: c for i, c in enumerate(classes)},
    }
    with open(meta_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f)

    # Feature schema
    feature_schema = {
        "num_features_out": n_features,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }
    with open(meta_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f)

    # Engineering manifest
    eng_manifest = {
        "run_id": eng_run_id,
        "status": "Completed",
        "num_features_out": n_features,
        "classes": classes,
    }
    with open(meta_dir / "engineering_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eng_manifest, f)

    # Generate synthetic CSR matrices and targets
    np.random.seed(42)
    for split_name in ("train", "validation", "test"):
        split_dir = data_dir / split_name
        # Targets: balanced across 3 classes
        y = np.array([i % len(classes) for i in range(n_samples)], dtype=np.int64)
        np.save(split_dir / "y.npy", y)

        # Scaled matrix
        dense_scaled = np.random.randn(n_samples, n_features).astype(np.float32)
        X_scaled = sparse.csr_matrix(dense_scaled)
        sparse.save_npz(split_dir / "X_scaled.npz", X_scaled)

        # Unscaled matrix
        dense_unscaled = np.random.uniform(0, 100, (n_samples, n_features)).astype(np.float32)
        X_unscaled = sparse.csr_matrix(dense_unscaled)
        sparse.save_npz(split_dir / "X_unscaled.npz", X_unscaled)

    return {
        "processed_dir": processed_dir,
        "eng_run_id": eng_run_id,
        "classes": classes,
        "n_features": n_features,
        "n_samples": n_samples,
    }


@pytest.fixture
def mock_trained_models(tmp_path: Path, mock_dataset_env: Dict):
    """Create mock CP-06 trained models in models/saved/."""
    saved_dir = tmp_path / "models" / "saved"
    training_run_id = "train_run_20261117_test01"
    run_dir = saved_dir / training_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    eng_run_id = mock_dataset_env["eng_run_id"]
    classes = mock_dataset_env["classes"]
    n_features = mock_dataset_env["n_features"]

    # Train tiny models on random data
    X_train = np.random.randn(60, n_features)
    y_train = np.array([i % len(classes) for i in range(60)])

    # 1. LinearSVC
    svm = LinearSVC(random_state=42, max_iter=2000)
    svm.fit(X_train, y_train)

    # 2. MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    mlp.fit(X_train, y_train)

    # 3. DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)

    models_map = {
        "svm": (svm, "Support Vector Machine (LinearSVC)", "LinearSVC"),
        "mlp": (mlp, "Multilayer Perceptron (MLP)", "MLPClassifier"),
        "decision_tree": (dt, "Decision Tree (CART)", "DecisionTreeClassifier"),
    }

    session_manifest = {
        "session_run_id": training_run_id,
        "engineering_run_id": eng_run_id,
        "started_at_utc": "2026-11-17T12:00:00Z",
        "completed_at_utc": "2026-11-17T12:01:00Z",
        "overall_status": "Completed",
        "trained_algorithms": list(models_map.keys()),
        "model_manifests": {},
        "run_type": "production",
        "is_test_fixture": False,
    }

    for algo_k, (est, disp_name, est_cls) in models_map.items():
        algo_dir = run_dir / algo_k
        algo_dir.mkdir(parents=True, exist_ok=True)

        joblib_path = algo_dir / "model.joblib"
        joblib.dump(est, joblib_path)

        with open(joblib_path, "rb") as f:
            sha256_val = hashlib.sha256(f.read()).hexdigest()

        meta = {
            "training_run_id": training_run_id,
            "algorithm": algo_k,
            "algorithm_display_name": disp_name,
            "estimator_class": est_cls,
            "status": "Completed",
            "engineering_run_id": eng_run_id,
            "training_samples": 60,
            "feature_count": n_features,
            "class_count": len(classes),
            "classes": classes,
            "artifact_checksum": sha256_val,
            "convergence_status": "Converged",
            "run_type": "production",
            "is_test_fixture": False,
        }
        with open(algo_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        session_manifest["model_manifests"][algo_k] = meta

    with open(run_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(session_manifest, f)

    return {
        "saved_dir": saved_dir,
        "training_run_id": training_run_id,
        "eng_run_id": eng_run_id,
        "classes": classes,
        "evaluations_dir": tmp_path / "models" / "evaluations",
    }


# ==============================================================================
# Test Suite
# ==============================================================================

def test_artifact_eligibility_production_models(mock_trained_models, mock_dataset_env):
    """Test 1: Only verified production models are eligible for evaluation."""
    verifier = ArtifactVerifier(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
    )

    eligible = verifier.discover_eligible_production_models(allow_test_fixtures=False)
    assert len(eligible) == 3
    algos = {m.algorithm for m in eligible}
    assert algos == {"svm", "mlp", "decision_tree"}

    for m in eligible:
        assert m.is_eligible is True
        assert m.run_type == "production"
        assert m.is_test_fixture is False
        assert len(m.model_sha256) == 64


def test_artifact_eligibility_rejects_test_fixtures(mock_trained_models, mock_dataset_env):
    """Test 2: Test fixture models are strictly rejected in production mode."""
    # Tag svm as a test fixture
    svm_meta_file = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "svm" / "metadata.json"
    with open(svm_meta_file, "r") as f:
        data = json.load(f)
    data["is_test_fixture"] = True
    data["run_type"] = "test_fixture"
    with open(svm_meta_file, "w") as f:
        json.dump(data, f)

    verifier = ArtifactVerifier(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
    )

    check = verifier.verify_model_eligibility(
        mock_trained_models["training_run_id"],
        "svm",
        allow_test_fixtures=False,
    )
    assert check.is_eligible is False
    assert any("test fixture" in r.lower() for r in check.rejection_reasons)


def test_artifact_eligibility_rejects_checksum_tampering(mock_trained_models, mock_dataset_env):
    """Test 3: Corrupted or tampered model artifacts are rejected with checksum mismatch."""
    mlp_file = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "mlp" / "model.joblib"
    # Tamper with file content
    with open(mlp_file, "ab") as f:
        f.write(b"CORRUPTED_BYTES")

    verifier = ArtifactVerifier(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
    )

    check = verifier.verify_model_eligibility(
        mock_trained_models["training_run_id"],
        "mlp",
        allow_test_fixtures=False,
    )
    assert check.is_eligible is False
    assert any("checksum mismatch" in r.lower() for r in check.rejection_reasons)


def test_zero_leakage_data_loader_forbids_training_partition(mock_dataset_env):
    """Test 4: Attempting to load the training partition raises PermissionError."""
    loader = EvaluationDataLoader(processed_dir=mock_dataset_env["processed_dir"])

    with pytest.raises(PermissionError) as exc_info:
        loader.load_partition(
            engineering_run_id=mock_dataset_env["eng_run_id"],
            partition="train",
            algorithm="svm",
        )
    assert "DATA LEAKAGE VIOLATION" in str(exc_info.value)


def test_data_loader_loads_validation_and_test_cleanly(mock_dataset_env):
    """Test 5: Validation and test partitions load with correct representations and checksums."""
    loader = EvaluationDataLoader(processed_dir=mock_dataset_env["processed_dir"])

    # LinearSVC requires scaled matrix
    val_svm = loader.load_partition(
        engineering_run_id=mock_dataset_env["eng_run_id"],
        partition="validation",
        algorithm="svm",
    )
    assert val_svm.partition_name == "validation"
    assert val_svm.representation_type == "scaled"
    assert val_svm.num_samples == 60
    assert val_svm.num_features == 8
    assert sparse.isspmatrix_csr(val_svm.X)

    # DecisionTree requires unscaled matrix
    val_dt = loader.load_partition(
        engineering_run_id=mock_dataset_env["eng_run_id"],
        partition="validation",
        algorithm="decision_tree",
    )
    assert val_dt.partition_name == "validation"
    assert val_dt.representation_type == "unscaled"


def test_sparse_batch_inference_runner_and_accounting_invariant(mock_trained_models, mock_dataset_env):
    """Test 6: Inference preserves CSR matrices and satisfies total = predicted + failed invariant."""
    loader = EvaluationDataLoader(processed_dir=mock_dataset_env["processed_dir"])
    val_part = loader.load_partition(
        mock_dataset_env["eng_run_id"],
        "validation",
        "mlp",
    )

    model_file = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "mlp" / "model.joblib"
    estimator = joblib.load(model_file)

    runner = InferenceRunner(batch_size=20)
    inf_res = runner.run_inference(
        estimator=estimator,
        algorithm="mlp",
        X=val_part.X,
        y_true=val_part.y,
        classes=val_part.classes,
    )

    assert inf_res.total_rows == 60
    assert inf_res.successfully_predicted_rows == 60
    assert inf_res.failed_rows == 0
    assert inf_res.total_rows == (inf_res.successfully_predicted_rows + inf_res.failed_rows)
    assert inf_res.score_type == "predict_proba"
    assert inf_res.scores.shape == (60, 3)
    assert inf_res.duration_seconds >= 0.0
    assert inf_res.ms_per_record >= 0.0


def test_linearsvc_decision_function_scores(mock_trained_models, mock_dataset_env):
    """Test 7: LinearSVC returns signed distance margins via decision_function without probability labeling."""
    loader = EvaluationDataLoader(processed_dir=mock_dataset_env["processed_dir"])
    val_part = loader.load_partition(
        mock_dataset_env["eng_run_id"],
        "validation",
        "svm",
    )

    model_file = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "svm" / "model.joblib"
    estimator = joblib.load(model_file)

    runner = InferenceRunner(batch_size=25)
    inf_res = runner.run_inference(
        estimator=estimator,
        algorithm="svm",
        X=val_part.X,
        y_true=val_part.y,
        classes=val_part.classes,
    )

    assert inf_res.score_type == "decision_function"
    assert inf_res.scores.shape == (60, 3)
    # Decision scores can be negative (signed margin distance)
    assert np.any(inf_res.scores < 0.0) or np.any(inf_res.scores > 0.0)


def test_classification_metrics_calculator_multiclass():
    """Test 8: Computes multiclass accuracy, balanced accuracy, macro/weighted F1, MCC, and Kappa."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2])

    agg = MetricsCalculator.calculate_aggregate_metrics(y_true, y_pred)
    assert 0.0 <= agg.accuracy <= 1.0
    assert 0.0 <= agg.balanced_accuracy <= 1.0
    assert 0.0 <= agg.macro_f1 <= 1.0
    assert 0.0 <= agg.weighted_f1 <= 1.0
    assert -1.0 <= agg.matthews_corrcoef <= 1.0
    assert -1.0 <= agg.cohen_kappa <= 1.0


def test_per_class_metrics_and_zero_division():
    """Test 9: Per-class metrics correctly calculate support, error rates, and handle zero predictions."""
    classes = ["BENIGN", "DoS", "PortScan", "ZeroSupportClass"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])  # ZeroSupportClass never occurs, PortScan has 1 mispredicted

    per_class = MetricsCalculator.calculate_per_class_metrics(y_true, y_pred, classes)
    assert len(per_class) == 4

    zero_supp = per_class[3]
    assert zero_supp.class_name == "ZeroSupportClass"
    assert zero_supp.support == 0
    assert zero_supp.precision == 0.0
    assert zero_supp.recall == 0.0
    assert zero_supp.f1_score == 0.0
    assert "Zero ground-truth support" in zero_supp.notes


def test_confusion_matrix_accounting_and_normalization():
    """Test 10: Confusion matrix preserves total count and row-normalizes correctly."""
    classes = ["BENIGN", "DoS", "PortScan"]
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])

    cm = ConfusionMatrixEngine.compute_matrix(y_true, y_pred, classes)
    assert cm.total_samples == 6

    raw_sum = sum(sum(row) for row in cm.raw_matrix)
    assert raw_sum == 6

    # Row 0 has support 3 (two predicted 0, one predicted 1)
    # Row 0 normalized percentages should sum to 100%
    assert round(sum(cm.normalized_matrix[0]), 1) == 100.0


def test_one_vs_rest_roc_and_missing_classes():
    """Test 11: One-vs-rest ROC-AUC handles continuous scores and flags missing classes."""
    classes = ["BENIGN", "DoS", "RareAttack"]
    y_true = np.array([0, 0, 1, 1, 0, 1])  # RareAttack has 0 samples
    scores = np.array([
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.2, 0.8, 0.0],
        [0.3, 0.7, 0.0],
        [0.9, 0.1, 0.0],
        [0.4, 0.6, 0.0],
    ])

    roc_dict, macro_auc = ROCAnalyzer.compute_ovr_roc(y_true, scores, classes)
    assert roc_dict["BENIGN"].is_available is True
    assert roc_dict["BENIGN"].auc is not None

    assert roc_dict["RareAttack"].is_available is False
    assert "No positive ground-truth samples" in roc_dict["RareAttack"].unavailable_reason
    assert macro_auc is not None  # Computed across estimable classes


def test_one_vs_rest_precision_recall_and_average_precision():
    """Test 12: PR analyzer computes curves, Average Precision, and handles unrepresented classes."""
    classes = ["BENIGN", "DoS"]
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])

    pr_dict, macro_ap = PrecisionRecallAnalyzer.compute_ovr_pr(y_true, scores, classes)
    assert pr_dict["BENIGN"].is_available is True
    assert pr_dict["BENIGN"].average_precision is not None
    assert pr_dict["DoS"].is_available is True
    assert macro_ap is not None


def test_security_metrics_benign_vs_attack():
    """Test 13: Security metrics compute False Alarm Rate, Detection Recall, and Specificity."""
    classes = ["BENIGN", "DoS", "PortScan"]
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])  # 4 benign, 4 attacks
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 0])  # 1 benign predicted as DoS (FA), 1 PortScan missed as benign (FN)

    sec = SecurityMetricsEngine.compute_security_metrics(y_true, y_pred, classes)
    assert sec.is_available is True
    assert sec.benign_label == "BENIGN"
    assert sec.benign_records_count == 4
    assert sec.attack_records_count == 4

    # Benign False Alarm Rate: 1 / 4 = 0.25
    assert sec.benign_false_alarm_rate == 0.25
    # Benign Specificity: 3 / 4 = 0.75
    assert sec.benign_specificity == 0.75
    # Attack Detection Recall: 3 / 4 = 0.75
    assert sec.attack_detection_recall == 0.75
    # Attack False Negative Rate: 1 / 4 = 0.25
    assert sec.attack_false_negative_rate == 0.25


def test_security_metrics_handles_missing_benign_class():
    """Test 14: Security view gracefully marks unavailable if no BENIGN label exists."""
    classes = ["AttackTypeA", "AttackTypeB"]
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    sec = SecurityMetricsEngine.compute_security_metrics(y_true, y_pred, classes)
    assert sec.is_available is False
    assert "No identifiable BENIGN class" in sec.unavailable_reason


def test_model_comparator_dataframes(mock_trained_models, mock_dataset_env):
    """Test 15: ModelComparator builds valid DataFrames for comparison and generalization gap."""
    loader = EvaluationDataLoader(processed_dir=mock_dataset_env["processed_dir"])
    val_part = loader.load_partition(mock_dataset_env["eng_run_id"], "validation", "mlp")

    model_file = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "mlp" / "model.joblib"
    est = joblib.load(model_file)
    runner = InferenceRunner(batch_size=50)
    inf = runner.run_inference(est, "mlp", val_part.X, val_part.y, val_part.classes)

    agg = MetricsCalculator.calculate_aggregate_metrics(inf.y_true, inf.y_pred)
    per_class = MetricsCalculator.calculate_per_class_metrics(inf.y_true, inf.y_pred, val_part.classes)
    cm = ConfusionMatrixEngine.compute_matrix(inf.y_true, inf.y_pred, val_part.classes)
    roc_map, macro_auc = ROCAnalyzer.compute_ovr_roc(inf.y_true, inf.scores, val_part.classes)
    pr_map, macro_ap = PrecisionRecallAnalyzer.compute_ovr_pr(inf.y_true, inf.scores, val_part.classes)
    sec = SecurityMetricsEngine.compute_security_metrics(inf.y_true, inf.y_pred, val_part.classes)

    res = AlgorithmEvaluationResult(
        algorithm="mlp",
        algorithm_display_name="Multilayer Perceptron (MLP)",
        training_run_id=mock_trained_models["training_run_id"],
        model_checksum="dummy_sha256",
        partition="validation",
        evaluated_rows=60,
        failed_rows=0,
        duration_seconds=0.1,
        ms_per_record=1.5,
        score_type="predict_proba",
        aggregate_metrics=agg,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        roc_analysis=roc_map,
        macro_roc_auc=macro_auc,
        pr_analysis=pr_map,
        macro_average_precision=macro_ap,
        security_metrics=sec,
    )

    results_by_algo = {"mlp": res}
    val_df = ModelComparator.build_validation_comparison_dataframe(results_by_algo)
    assert not val_df.empty
    assert "Algorithm" in val_df.columns
    assert "Accuracy" in val_df.columns

    gap_df = ModelComparator.build_generalization_gap_dataframe(results_by_algo, results_by_algo)
    assert not gap_df.empty
    assert "Acc Gap (Delta)" in gap_df.columns


def test_full_orchestrator_validation_freeze_and_gated_test(mock_trained_models, mock_dataset_env):
    """Test 16: Complete end-to-end evaluation flow: validation -> freeze candidates -> gated test."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    # 1. Run Validation Evaluation
    val_manifest = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )

    assert val_manifest.status == "Completed"
    assert "validation" in val_manifest.partitions_evaluated
    assert len(val_manifest.validation_results) == 3
    assert val_manifest.test_gate_confirmed is False

    # 2. Attempting Test Evaluation before freezing candidates MUST FAIL
    with pytest.raises(PermissionError) as exc_info:
        orch.run_gated_test_evaluation(
            evaluation_run_id=val_manifest.evaluation_run_id,
            confirmation_token="CONFIRM_TEST_EVALUATION",
        )
    assert "Candidate models must be selected and frozen" in str(exc_info.value)

    # 3. Freeze Candidates
    frozen_manifest = orch.freeze_candidate_models(
        evaluation_run_id=val_manifest.evaluation_run_id,
        primary_candidate_algo="mlp",
        selection_criterion="Superior macro F1 on validation",
    )
    assert len(frozen_manifest.frozen_candidates) == 3
    assert frozen_manifest.frozen_candidates["mlp"].selected_as_primary is True

    # 4. Attempting Test Evaluation with invalid token MUST FAIL
    with pytest.raises(PermissionError) as exc_token:
        orch.run_gated_test_evaluation(
            evaluation_run_id=val_manifest.evaluation_run_id,
            confirmation_token="INVALID_TOKEN",
        )
    assert "explicit confirmation" in str(exc_token.value)

    # 5. Gated Test Evaluation with valid token
    test_manifest = orch.run_gated_test_evaluation(
        evaluation_run_id=val_manifest.evaluation_run_id,
        confirmation_token="CONFIRM_TEST_EVALUATION",
        notes="Automated test run confirmation.",
    )

    assert test_manifest.test_gate_confirmed is True
    assert "test" in test_manifest.partitions_evaluated
    assert len(test_manifest.test_results) == 3

    # Check persistence in registry
    reg = EvaluationRegistry(evaluations_dir=mock_trained_models["evaluations_dir"])
    runs = reg.list_evaluation_runs()
    assert len(runs) == 1
    assert runs[0]["evaluation_run_id"] == val_manifest.evaluation_run_id
    assert runs[0]["test_gate_confirmed"] is True


def test_resource_preflight_gating():
    """Test 17: Resource manager correctly checks RAM, disk, and exact curve practicality."""
    preflight = EvaluationResourceManager.perform_preflight_check(
        num_samples=100_000,
        num_features=78,
        num_classes=15,
        sparse_density=0.15,
    )
    assert preflight.is_safe is True
    assert preflight.batch_size > 0
    assert preflight.estimated_batch_memory_bytes > 0
    assert preflight.estimated_scores_memory_bytes > 0


# ==============================================================================
# CP-07 Governance & Test Holdout Audit Tests
# ==============================================================================

def test_audit_governance_repeated_test_evaluation_locked(mock_trained_models, mock_dataset_env):
    """Audit 1: Prevent repeated test holdout evaluation attempts on the same run."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    # Validation -> Freeze -> Gated Test
    val_manifest = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(
        evaluation_run_id=val_manifest.evaluation_run_id,
        primary_candidate_algo="mlp",
    )
    first_test_manifest = orch.run_gated_test_evaluation(
        evaluation_run_id=val_manifest.evaluation_run_id,
        confirmation_token="CONFIRM_TEST_EVALUATION",
    )
    assert first_test_manifest.test_holdout_locked is True
    assert first_test_manifest.test_holdout_locked_at_utc is not None
    assert first_test_manifest.candidate_set_hash is not None

    # Immediate second attempt on the same evaluation run MUST FAIL with PermissionError
    with pytest.raises(PermissionError) as exc_info:
        orch.run_gated_test_evaluation(
            evaluation_run_id=val_manifest.evaluation_run_id,
            confirmation_token="CONFIRM_TEST_EVALUATION",
        )
    assert "TEST HOLDOUT LOCKED" in str(exc_info.value)


def test_audit_governance_changing_candidate_set_after_test_rejected(mock_trained_models, mock_dataset_env):
    """Audit 2: Prevent modifying candidate set once test evaluation has occurred."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    val_manifest = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(
        evaluation_run_id=val_manifest.evaluation_run_id,
        primary_candidate_algo="mlp",
    )
    orch.run_gated_test_evaluation(
        evaluation_run_id=val_manifest.evaluation_run_id,
        confirmation_token="CONFIRM_TEST_EVALUATION",
    )

    # Attempt to re-freeze candidate models with a different primary candidate MUST FAIL
    with pytest.raises(PermissionError) as exc_info:
        orch.freeze_candidate_models(
            evaluation_run_id=val_manifest.evaluation_run_id,
            primary_candidate_algo="svm",
        )
    assert "GOVERNANCE VIOLATION" in str(exc_info.value)
    assert "cannot be modified or re-frozen" in str(exc_info.value)


def test_audit_governance_different_candidate_set_on_locked_engineering_run_rejected(mock_trained_models, mock_dataset_env):
    """Audit 3: Prevent changing candidate set and rerunning the test partition on same split."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    # Session 1: Locks the engineering run test holdout
    val1 = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(val1.evaluation_run_id, primary_candidate_algo="mlp")
    orch.run_gated_test_evaluation(val1.evaluation_run_id, confirmation_token="CONFIRM_TEST_EVALUATION")

    # Session 2: Fresh evaluation run on the SAME engineering run with a DIFFERENT primary candidate
    val2 = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )
    # Freeze with primary "decision_tree" instead of "mlp"
    orch.freeze_candidate_models(val2.evaluation_run_id, primary_candidate_algo="decision_tree")

    # Attempt to evaluate test partition with altered candidate selection MUST FAIL
    with pytest.raises(PermissionError) as exc_info:
        orch.run_gated_test_evaluation(val2.evaluation_run_id, confirmation_token="CONFIRM_TEST_EVALUATION")
    assert "TEST HOLDOUT GOVERNANCE VIOLATION" in str(exc_info.value)
    assert "Changing the candidate set or rerunning the test partition" in str(exc_info.value)


def test_audit_governance_candidate_artifact_checksum_change_rejected(mock_trained_models, mock_dataset_env):
    """Audit 4: Detect and reject candidate artifact tampering between freeze and test evaluation."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    val = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp", "decision_tree"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(val.evaluation_run_id, primary_candidate_algo="decision_tree")

    # Tamper with the decision_tree model artifact on disk after freeze
    dt_model_path = mock_trained_models["saved_dir"] / mock_trained_models["training_run_id"] / "decision_tree" / "model.joblib"
    assert dt_model_path.exists()
    with open(dt_model_path, "ab") as f:
        f.write(b"\x00\x00CORRUPT_CHECKSUM_PAYLOAD")

    # Gated test evaluation must detect mismatch and reject execution
    with pytest.raises(ValueError) as exc_info:
        orch.run_gated_test_evaluation(val.evaluation_run_id, confirmation_token="CONFIRM_TEST_EVALUATION")
    assert "CRITICAL CHECKSUM MISMATCH" in str(exc_info.value)


def test_audit_governance_persistent_lock_file_and_manifest_recovery(mock_trained_models, mock_dataset_env):
    """Audit 5: Verify persistent lock file persistence and fallback recovery from manifests."""
    orch = EvaluationOrchestrator(
        saved_models_dir=mock_trained_models["saved_dir"],
        processed_dir=mock_dataset_env["processed_dir"],
        evaluations_dir=mock_trained_models["evaluations_dir"],
    )

    val = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(val.evaluation_run_id, primary_candidate_algo="svm")
    orch.run_gated_test_evaluation(val.evaluation_run_id, confirmation_token="CONFIRM_TEST_EVALUATION")

    # Check lock file exists
    locks_file = mock_trained_models["evaluations_dir"] / "test_holdout_locks.json"
    assert locks_file.exists()
    reg = orch.eval_registry
    lock_info = reg.get_test_lock(mock_dataset_env["eng_run_id"])
    assert lock_info is not None
    assert lock_info["evaluation_run_id"] == val.evaluation_run_id
    assert lock_info["primary_candidate"] == "svm"

    # Delete the lock file to simulate accidental deletion
    locks_file.unlink()
    assert not locks_file.exists()

    # Fallback inspection of saved manifests MUST recover the lock
    recovered_lock = reg.get_test_lock(mock_dataset_env["eng_run_id"])
    assert recovered_lock is not None
    assert recovered_lock["evaluation_run_id"] == val.evaluation_run_id

    # And attempting a new test run on that split must STILL be blocked
    val_new = orch.run_validation_evaluation(
        training_run_id=mock_trained_models["training_run_id"],
        algorithms=["svm", "mlp"],
        allow_test_fixtures=False,
    )
    orch.freeze_candidate_models(val_new.evaluation_run_id, primary_candidate_algo="mlp")
    with pytest.raises(PermissionError) as exc_blocked:
        orch.run_gated_test_evaluation(val_new.evaluation_run_id, confirmation_token="CONFIRM_TEST_EVALUATION")
    assert "TEST HOLDOUT GOVERNANCE VIOLATION" in str(exc_blocked.value)


def test_audit_governance_default_batch_size_safeguards():
    """Audit 6: Verify default batch size is 10,000 and resource-aware safeguards work."""
    from config.settings import settings

    assert settings.evaluation.default_batch_size == 10_000
    assert EvaluationResourceManager.DEFAULT_BATCH_SIZE == 10_000
    runner = InferenceRunner()
    assert runner.batch_size == 10_000

    # Resource preflight under nominal conditions maintains 10,000 batch size
    preflight = EvaluationResourceManager.perform_preflight_check(
        num_samples=50_000,
        num_features=80,
        num_classes=15,
        sparse_density=0.1,
    )
    assert preflight.batch_size == 10_000
    assert preflight.is_safe is True
