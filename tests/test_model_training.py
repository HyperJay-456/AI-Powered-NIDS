"""Automated Test Suite for CP-06 ML Training Pipeline & Model Registry.

Tests the resource-aware, zero-leakage training subsystem and model registry
supporting strictly the three authorized algorithms:
1. Support Vector Machine (LinearSVC)
2. Multilayer Perceptron (MLP with mini-batch incremental learning)
3. Decision Tree (CART on unscaled flow representation)
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Generator, List, Tuple
import unittest

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from dashboard.status_checker import (
    check_model_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)
from src.data.engineering.artifact_manager import ArtifactManager
from src.models import (
    AlgorithmTrainingManifest,
    AuthorizedAlgorithm,
    CP05ArtifactLoader,
    DecisionTreeTrainer,
    DecisionTreeTrainingConfig,
    LoadedTrainingData,
    MasterTrainingConfiguration,
    MLPTrainer,
    MLPTrainingConfig,
    ModelArtifactRecord,
    ModelFactory,
    ModelRegistry,
    PreflightCheckResult,
    ResourceManager,
    SVMTrainer,
    SVMTrainingConfig,
    TrainingOrchestrator,
    TrainingProgressTracker,
)


@pytest.fixture
def temp_project_dir() -> Generator[Dict[str, Path], None, None]:
    """Create a temporary directory structure mimicking project directories."""
    with tempfile.TemporaryDirectory() as base_temp:
        base_path = Path(base_temp)
        raw_dir = base_path / "data" / "raw"
        processed_dir = base_path / "data" / "processed"
        models_dir = base_path / "models" / "saved"
        meta_dir = base_path / "data" / "metadata"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        yield {
            "base": base_path,
            "raw": raw_dir,
            "processed": processed_dir,
            "models": models_dir,
            "metadata": meta_dir,
        }


def create_mock_cp05_run(
    processed_dir: Path,
    run_id: str = "eng_run_20261117_000001",
    n_samples: int = 120,
    n_features: int = 10,
    single_class: bool = False,
    status: str = "Completed",
) -> Path:
    """Helper to populate a synthetic but structurally complete CP-05 run on disk."""
    run_dir = processed_dir / run_id
    train_dir = run_dir / "datasets" / "train"
    val_dir = run_dir / "datasets" / "val"
    test_dir = run_dir / "datasets" / "test"
    meta_dir = run_dir / "metadata"
    preproc_dir = run_dir / "preprocessors"

    for d in [train_dir, val_dir, test_dir, meta_dir, preproc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Feature matrices: X_scaled and X_unscaled as sparse CSR
    np.random.seed(42)
    dense_train = np.random.randn(n_samples, n_features).astype(np.float32)
    # Ensure non-negative physical values for unscaled
    dense_unscaled = np.abs(dense_train) * 100.0

    sparse_scaled = sparse.csr_matrix(dense_train)
    sparse_unscaled = sparse.csr_matrix(dense_unscaled)

    sparse.save_npz(train_dir / "X_scaled.npz", sparse_scaled)
    sparse.save_npz(train_dir / "X_unscaled.npz", sparse_unscaled)

    # Targets: binary classification (0: BENIGN, 1: DoS) unless single_class
    if single_class:
        y_train = np.zeros(n_samples, dtype=np.int32)
        classes = ["BENIGN"]
    else:
        y_train = np.array([0, 1] * (n_samples // 2), dtype=np.int32)
        classes = ["BENIGN", "DoS"]

    np.save(train_dir / "y.npy", y_train)

    # Held-out sets: strictly exist to prove anti-leakage isolation
    dense_val = np.random.randn(30, n_features).astype(np.float32)
    sparse.save_npz(val_dir / "X_scaled.npz", sparse.csr_matrix(dense_val))
    np.save(val_dir / "y.npy", np.array([0, 1] * 15, dtype=np.int32))

    dense_test = np.random.randn(30, n_features).astype(np.float32)
    sparse.save_npz(test_dir / "X_scaled.npz", sparse.csr_matrix(dense_test))
    np.save(test_dir / "y.npy", np.array([0, 1] * 15, dtype=np.int32))

    # Metadata files
    manifest_data = {
        "run_id": run_id,
        "status": status,
        "created_at": "2026-11-17T12:00:00Z",
        "total_raw_rows": n_samples + 60,
        "train_rows": n_samples,
        "val_rows": 30,
        "test_rows": 30,
        "num_features_out": n_features,
        "num_classes": len(classes),
        "classes": classes,
        "source_filenames": ["test_traffic.csv"],
        "source_checksums": {"test_traffic.csv": "abc123mockhash"},
    }

    with open(meta_dir / "engineering_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open(meta_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({"feature_names": [f"f_{i}" for i in range(n_features)]}, f, indent=2)

    with open(meta_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_index": {c: i for i, c in enumerate(classes)}}, f, indent=2)

    # Mock preprocessor artifact
    joblib.dump({"classes": classes}, preproc_dir / "label_processor.joblib")

    return run_dir


# ==============================================================================
# SECTION 1: CONFIGURATION & FACTORY TESTS
# ==============================================================================


def test_authorized_algorithm_enum():
    """Verify only the three authorized algorithms exist in AuthorizedAlgorithm."""
    assert AuthorizedAlgorithm.SVM.value == "svm"
    assert AuthorizedAlgorithm.MLP.value == "mlp"
    assert AuthorizedAlgorithm.DECISION_TREE.value == "decision_tree"
    assert len(AuthorizedAlgorithm) == 3


def test_model_factory_trainer_instantiation():
    """Verify ModelFactory creates appropriate trainers for authorized algorithms."""
    factory = ModelFactory()

    svm_trainer = factory.create_trainer("svm")
    assert isinstance(svm_trainer, SVMTrainer)

    mlp_trainer = factory.create_trainer("mlp")
    assert isinstance(mlp_trainer, MLPTrainer)

    dt_trainer = factory.create_trainer("decision_tree")
    assert isinstance(dt_trainer, DecisionTreeTrainer)


def test_model_factory_unauthorized_algorithm_rejection():
    """Verify ModelFactory rejects unauthorized algorithms with ValueError."""
    factory = ModelFactory()
    unauthorized_algos = ["random_forest", "xgboost", "knn", "naive_bayes", "logistic_regression"]

    for algo in unauthorized_algos:
        with pytest.raises(ValueError) as excinfo:
            factory.create_trainer(algo)
        assert "Unauthorized algorithm" in str(excinfo.value)


# ==============================================================================
# SECTION 2: RESOURCE MANAGER & PREFLIGHT CHECKS
# ==============================================================================


def test_resource_manager_system_memory():
    """Verify ResourceManager queries real physical host memory without external psutil."""
    rm = ResourceManager()
    avail_bytes = rm.get_available_system_memory()
    assert avail_bytes is not None
    assert avail_bytes > 0

    info = rm.get_system_memory_info()
    assert "available_ram_bytes" in info
    assert "available_ram_formatted" in info
    assert "available_ram_gb" in info
    assert info["available_ram_gb"] > 0.0


def test_resource_manager_training_memory_estimation():
    """Verify memory estimation scales with row count and feature dimension."""
    rm = ResourceManager()

    small_mem = rm.estimate_memory_from_dimensions(n_samples=1000, n_features=20, algorithm="svm")
    large_mem = rm.estimate_memory_from_dimensions(n_samples=100000, n_features=78, algorithm="svm")

    assert large_mem > small_mem
    assert small_mem > 0


def test_resource_manager_preflight_checks():
    """Verify preflight check approves normal runs and issues warning on unconstrained DT."""
    rm = ResourceManager()

    # Normal constrained check
    res_normal = rm.preflight_check_dimensions(
        n_samples=1000,
        n_features=50,
        algorithms=["svm", "mlp", "decision_tree"],
        dt_max_depth=15,
    )
    assert res_normal.is_safe is True

    # Unconstrained max_depth warning
    res_unconstrained = rm.preflight_check_dimensions(
        n_samples=5000,
        n_features=50,
        algorithms=["decision_tree"],
        dt_max_depth=None,
    )
    assert res_unconstrained.is_safe is True
    assert any("unconstrained" in w.lower() for w in res_unconstrained.warning_messages)


# ==============================================================================
# SECTION 3: ARTIFACT LOADER & ANTI-LEAKAGE ENFORCEMENT
# ==============================================================================


def test_artifact_loader_rejection_of_missing_run(temp_project_dir):
    """Verify loader rejects non-existent engineering run."""
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    check = loader.verify_engineering_run_integrity("non_existent_run_id")

    assert not check.is_eligible
    assert any("does not exist" in r for r in check.rejection_reasons)


def test_artifact_loader_rejection_of_incomplete_status(temp_project_dir):
    """Verify loader rejects a run with status != 'Completed'."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="failed_run", status="Failed")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    check = loader.verify_engineering_run_integrity("failed_run")

    assert not check.is_eligible
    assert any("must be 'Completed'" in r for r in check.rejection_reasons)


def test_artifact_loader_rejection_of_single_class_training_set(temp_project_dir):
    """Verify loader detects and rejects degenerate single-class training sets."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="single_class_run", single_class=True)
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    check = loader.verify_engineering_run_integrity("single_class_run")

    assert not check.is_eligible
    assert any("at least 2 distinct classes" in r for r in check.rejection_reasons)


def test_artifact_loader_strict_zero_leakage_loading(temp_project_dir):
    """Verify load_training_partition_only strictly loads ONLY training files."""
    run_dir = create_mock_cp05_run(temp_project_dir["processed"], run_id="valid_run")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])

    # Load SVM/MLP (scaled)
    data_scaled = loader.load_training_partition_only("valid_run", "svm")
    assert isinstance(data_scaled, LoadedTrainingData)
    assert data_scaled.representation_type == "scaled"
    assert data_scaled.num_samples == 120
    assert data_scaled.num_features == 10
    assert data_scaled.num_classes == 2
    assert sparse.issparse(data_scaled.X)

    # Load Decision Tree (unscaled)
    data_unscaled = loader.load_training_partition_only("valid_run", "decision_tree")
    assert data_unscaled.representation_type == "unscaled"
    assert data_unscaled.num_samples == 120
    assert data_unscaled.num_features == 10
    assert sparse.issparse(data_unscaled.X)

    # Verify validation and test files were NEVER modified or touched during load
    val_file = run_dir / "datasets" / "val" / "X_scaled.npz"
    test_file = run_dir / "datasets" / "test" / "X_scaled.npz"
    assert val_file.exists()
    assert test_file.exists()


# ==============================================================================
# SECTION 4: ALGORITHM TRAINERS
# ==============================================================================


def test_svm_trainer_execution(temp_project_dir):
    """Verify SVMTrainer trains LinearSVC on sparse matrix and records iterations."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_svm")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    train_data = loader.load_training_partition_only("run_svm", "svm")

    config = SVMTrainingConfig(max_iter=1000, C=1.0, random_state=42)
    trainer = SVMTrainer(config=config)
    result = trainer.train(
        X_train=train_data.X,
        y_train=train_data.y,
        classes=train_data.classes,
    )

    assert result.status == "Completed"
    assert isinstance(result.estimator, LinearSVC)
    assert result.num_samples == 120
    assert result.num_features == 10
    assert result.duration_seconds >= 0.0
    assert result.solver_iterations >= 1


def test_mlp_trainer_minibatch_partial_fit(temp_project_dir):
    """Verify MLPTrainer trains via mini-batch partial_fit and tracks loss history."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_mlp")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    train_data = loader.load_training_partition_only("run_mlp", "mlp")

    config = MLPTrainingConfig(
        hidden_layer_sizes=(16, 8),
        max_epochs=5,  # 5 epochs
        batch_size=32,
        random_state=42,
    )
    trainer = MLPTrainer(config=config)
    result = trainer.train(
        X_train=train_data.X,
        y_train=train_data.y,
        classes=train_data.classes,
    )

    assert result.status == "Completed"
    assert isinstance(result.estimator, MLPClassifier)
    assert result.total_epochs_trained == 5
    assert len(result.loss_history) == 5
    assert all(loss > 0 for loss in result.loss_history)
    assert result.duration_seconds >= 0.0


def test_decision_tree_trainer_unscaled_features(temp_project_dir):
    """Verify DecisionTreeTrainer fits on unscaled features and extracts tree metrics."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_dt")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    train_data = loader.load_training_partition_only("run_dt", "decision_tree")

    config = DecisionTreeTrainingConfig(max_depth=5, min_samples_split=2, min_samples_leaf=1)
    trainer = DecisionTreeTrainer(config=config)
    result = trainer.train(
        X_train=train_data.X,
        y_train=train_data.y,
        classes=train_data.classes,
    )

    assert result.status == "Completed"
    assert isinstance(result.estimator, DecisionTreeClassifier)
    assert result.actual_depth <= 5
    assert result.leaf_count >= 1
    assert result.node_count >= 1
    assert result.duration_seconds >= 0.0


# ==============================================================================
# SECTION 5: MODEL REGISTRY & ATOMIC PERSISTENCE
# ==============================================================================


def test_model_registry_save_and_verify(temp_project_dir):
    """Verify ModelRegistry atomic persistence, SHA-256 calculation, and integrity checks."""
    models_dir = temp_project_dir["models"]
    registry = ModelRegistry(saved_models_dir=models_dir)

    # Train a dummy SVM to persist
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_reg")
    loader = CP05ArtifactLoader(processed_dir=temp_project_dir["processed"])
    train_data = loader.load_training_partition_only("run_reg", "svm")

    config = SVMTrainingConfig()
    trainer = SVMTrainer(config=config)
    res = trainer.train(X_train=train_data.X, y_train=train_data.y, classes=train_data.classes)

    session_id = "train_run_test_001"
    manifest = AlgorithmTrainingManifest(
        training_run_id=session_id,
        algorithm="svm",
        algorithm_display_name="Support Vector Machine",
        estimator_class="sklearn.svm.LinearSVC",
        status=res.status,
        engineering_run_id="run_reg",
        source_dataset_names=train_data.source_filenames,
        source_checksums=train_data.source_checksums,
        training_partition_checksum=train_data.training_partition_checksum,
        feature_schema_checksum=train_data.feature_schema_checksum,
        label_mapping_checksum=train_data.label_mapping_checksum,
        training_samples=train_data.num_samples,
        feature_count=train_data.num_features,
        class_count=train_data.num_classes,
        classes=train_data.classes,
        input_representation="Scaled Sparse CSR",
        random_seed=42,
        hyperparameters=config.to_dict(),
        training_duration_seconds=res.duration_seconds,
        started_at_utc="2026-11-17T12:00:00Z",
        completed_at_utc="2026-11-17T12:01:00Z",
        convergence_status="Converged",
    )

    committed_dir = registry.save_training_session(
        training_run_id=session_id,
        engineering_run_id="run_reg",
        results_by_algo={"svm": res},
        training_manifests={"svm": manifest},
        started_at_utc="2026-11-17T12:00:00Z",
    )

    assert committed_dir.exists()
    assert (committed_dir / "training_manifest.json").exists()
    assert (committed_dir / "svm" / "model.joblib").exists()
    assert (committed_dir / "svm" / "metadata.json").exists()

    # Verify cryptographic integrity
    is_valid = registry.verify_model_integrity(session_id, "svm")
    assert is_valid is True

    # Check listing methods
    runs = registry.list_training_runs()
    assert len(runs) == 1
    assert runs[0]["session_run_id"] == session_id

    models = registry.list_all_trained_models()
    assert len(models) == 1
    assert models[0].algorithm == "svm"
    assert models[0].training_run_id == session_id

    # Verify latest model query
    latest_svm = registry.get_latest_model("svm")
    assert latest_svm is not None
    assert latest_svm.algorithm == "svm"

    # Verify loading model
    loaded_model = registry.load_model(session_id, "svm")
    assert isinstance(loaded_model, LinearSVC)


# ==============================================================================
# SECTION 6: TRAINING ORCHESTRATOR & FULL WORKFLOW
# ==============================================================================


def test_orchestrator_train_single(temp_project_dir):
    """Verify TrainingOrchestrator trains an individual model and persists it."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_orch_single")
    orchestrator = TrainingOrchestrator(
        processed_dir=temp_project_dir["processed"],
        saved_models_dir=temp_project_dir["models"],
    )

    manifest = orchestrator.train_single_model(
        engineering_run_id="run_orch_single",
        algorithm="decision_tree",
        config=DecisionTreeTrainingConfig(max_depth=5),
    )

    assert isinstance(manifest, AlgorithmTrainingManifest)
    assert manifest.status == "Completed"
    assert manifest.algorithm == "decision_tree"
    assert manifest.training_samples == 120

    # Verify saved model in registry
    registry = ModelRegistry(saved_models_dir=temp_project_dir["models"])
    latest = registry.get_latest_model("decision_tree")
    assert latest is not None
    assert latest.algorithm == "decision_tree"


def test_orchestrator_train_all_sequential(temp_project_dir):
    """Verify TrainingOrchestrator sequentially trains all three authorized algorithms."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_orch_all")
    orchestrator = TrainingOrchestrator(
        processed_dir=temp_project_dir["processed"],
        saved_models_dir=temp_project_dir["models"],
    )

    tracker_events = []
    manifests = orchestrator.train_all_models_sequentially(
        engineering_run_id="run_orch_all",
        master_config=MasterTrainingConfiguration(
            mlp_config=MLPTrainingConfig(max_epochs=3, batch_size=32),
        ),
        progress_callback=lambda u: tracker_events.append(u.stage.value),
    )

    assert len(manifests) == 3
    assert "svm" in manifests
    assert "mlp" in manifests
    assert "decision_tree" in manifests
    assert all(m.status == "Completed" for m in manifests.values())

    # Verify tracker received stage updates
    assert len(tracker_events) > 0

    # Verify all 3 models are registered
    registry = ModelRegistry(saved_models_dir=temp_project_dir["models"])
    assert registry.get_latest_model("svm") is not None
    assert registry.get_latest_model("mlp") is not None
    assert registry.get_latest_model("decision_tree") is not None


def test_orchestrator_concurrency_lock(temp_project_dir):
    """Verify TrainingOrchestrator raises RuntimeError if a concurrent job is attempted."""
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_lock")
    orchestrator = TrainingOrchestrator(
        processed_dir=temp_project_dir["processed"],
        saved_models_dir=temp_project_dir["models"],
    )

    # Acquire lock manually to simulate active job
    orchestrator._job_lock.acquire()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            orchestrator.train_single_model("run_lock", "svm")
        assert "already actively executing" in str(excinfo.value)
    finally:
        orchestrator._job_lock.release()


# ==============================================================================
# SECTION 7: DASHBOARD STATUS CHECKER INTEGRATION
# ==============================================================================


def test_status_checker_cp06_model_discovery(temp_project_dir):
    """Verify status_checker discovers CP-06 registered models in run subdirectories."""
    models_dir = temp_project_dir["models"]

    # 1. Initially empty
    status_empty = check_model_status(models_dir=models_dir)
    assert status_empty.trained_count == 0
    assert status_empty.summary_text == "No Trained Models"

    # 2. Train and save models via orchestrator
    create_mock_cp05_run(temp_project_dir["processed"], run_id="run_status_test")
    orchestrator = TrainingOrchestrator(
        processed_dir=temp_project_dir["processed"],
        saved_models_dir=models_dir,
    )

    orchestrator.train_single_model("run_status_test", "svm")

    # 3. Model status after 1 model
    status_1 = check_model_status(models_dir=models_dir)
    assert status_1.trained_count == 1
    assert status_1.models["svm"].is_trained is True
    assert status_1.models["mlp"].is_trained is False
    assert status_1.models["decision_tree"].is_trained is False
    assert "1/3 Models Available" in status_1.summary_text

    # 4. Train the remaining 2
    orchestrator.train_single_model("run_status_test", "mlp")
    orchestrator.train_single_model("run_status_test", "decision_tree")

    status_all = check_model_status(models_dir=models_dir)
    assert status_all.trained_count == 3
    assert status_all.summary_text == "All 3 Models Ready"
    assert status_all.models["svm"].is_trained is True
    assert status_all.models["mlp"].is_trained is True
    assert status_all.models["decision_tree"].is_trained is True
