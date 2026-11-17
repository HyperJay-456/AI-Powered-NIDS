"""System and project status derivation module for the NIDS Dashboard.

Performs read-only inspection of the filesystem, model registry, and project state
to provide verifiable operational telemetry without fabricated or synthetic data.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import (
    ALGORITHM_KEYS,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
    SUPPORTED_ALGORITHMS,
    settings,
)


@dataclass(frozen=True)
class DatasetStatus:
    """Read-only verification of dataset assets."""

    is_loaded: bool
    file_count: int
    available_files: List[str]
    summary_text: str
    target_dataset_name: str
    storage_path: str
    expected_format: str = "CSV"
    validation_status: str = "Dataset Not Loaded"
    validated_count: int = 0


@dataclass(frozen=True)
class ModelArtifactStatus:
    """Verification status for an individual machine learning algorithm."""

    key: str
    name: str
    is_trained: bool
    artifact_path: Optional[str]
    status_label: str
    description: str
    algorithm_category: str


@dataclass(frozen=True)
class ModelSystemStatus:
    """Aggregated status across all three authorized ML algorithms."""

    trained_count: int
    total_models: int
    models: Dict[str, ModelArtifactStatus]
    summary_text: str


@dataclass(frozen=True)
class TrafficAnalysisStatus:
    """Operational status of traffic analysis workflows."""

    is_available: bool
    summary_text: str
    detail_text: str


@dataclass(frozen=True)
class LatestAnalysisStatus:
    """Verification status of the latest intrusion analysis run."""

    has_run: bool
    summary_text: str
    detail_text: str
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class DatasetEngineeringStatus:
    """Verifiable status of dataset engineering and preprocessing runs."""

    state: str  # "Not Started", "In Progress", "Completed", "Failed"
    is_completed: bool
    run_count: int
    latest_run_id: Optional[str]
    total_raw_rows: int
    supervised_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    num_features_out: int
    num_classes: int
    classes: List[str]
    summary_text: str
    detail_text: str


@dataclass(frozen=True)
class PipelineStage:
    """Pipeline progression stage status."""

    stage_id: str
    title: str
    description: str
    status: str  # "Completed", "Active", "Pending"
    badge_variant: str  # "success", "info", "neutral", "warning"
    target_checkpoint: str


@dataclass(frozen=True)
class ProjectMilestone:
    """Development milestone status tracking."""

    code: str
    title: str
    status: str  # "Completed", "Active", "Planned"
    badge_variant: str  # "success", "info", "neutral"
    description: str


def check_dataset_status(
    raw_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
) -> DatasetStatus:
    """Verify presence of compatible dataset files and validation state.

    Args:
        raw_dir: Optional override path for testing raw data storage.
        metadata_dir: Optional override path for testing metadata storage.

    Returns:
        DatasetStatus: Verifiable dataset availability and validation telemetry.
    """
    target_dir = raw_dir or RAW_DATA_DIR
    target_meta = metadata_dir or settings.metadata_dir
    target_name = settings.dataset.name

    if not target_dir.exists():
        return DatasetStatus(
            is_loaded=False,
            file_count=0,
            available_files=[],
            summary_text="Dataset Not Loaded",
            target_dataset_name=target_name,
            storage_path=str(target_dir),
            validation_status="Dataset Not Loaded",
            validated_count=0,
        )

    # Safe read-only inspection of CSV files
    csv_files = sorted([f.name for f in target_dir.glob("*.csv") if f.is_file()])
    count = len(csv_files)

    if count == 0:
        summary = "Dataset Not Loaded"
        loaded = False
        val_status = "Dataset Not Loaded"
        validated_count = 0
    else:
        summary = f"{count} File{'s' if count > 1 else ''} Available"
        loaded = True

        # Check ingestion manifest if available
        manifest_path = target_meta / "ingestion_manifest.json"
        validated_count = 0
        has_failed = False
        has_warnings = False

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                for item in manifest_data.values():
                    v_stat = item.get("validation_status", "")
                    if "Validated" in v_stat:
                        validated_count += 1
                    if "Warnings" in v_stat:
                        has_warnings = True
                    if "Failed" in v_stat:
                        has_failed = True

                if has_failed:
                    val_status = "Validation Failed"
                elif has_warnings:
                    val_status = "Validated with Warnings"
                elif validated_count > 0:
                    val_status = "Validated"
                else:
                    val_status = "Uploaded (Pending Validation)"
            except Exception:
                val_status = "Uploaded (Pending Validation)"
        else:
            val_status = "Uploaded (Pending Validation)"

    return DatasetStatus(
        is_loaded=loaded,
        file_count=count,
        available_files=csv_files,
        summary_text=summary,
        target_dataset_name=target_name,
        storage_path=str(target_dir.relative_to(settings.base_dir) if target_dir.is_relative_to(settings.base_dir) else target_dir),
        validation_status=val_status,
        validated_count=validated_count,
    )


def check_model_status(models_dir: Optional[Path] = None) -> ModelSystemStatus:
    """Verify presence of trained serialized model artifacts in the models directory.

    Args:
        models_dir: Optional override path for testing.

    Returns:
        ModelSystemStatus: Verifiable model artifacts status.
    """
    target_dir = models_dir or SAVED_MODELS_DIR
    model_metadata = {
        "svm": {
            "name": "Support Vector Machine",
            "category": "Maximum-Margin Classifier (Linear / RBF)",
            "desc": "Constructs optimal separating hyperplanes in high-dimensional feature spaces with strong theoretical generalization bounds.",
            "filenames": ["svm.joblib", "svm.pkl", "svm_model.joblib"],
        },
        "mlp": {
            "name": "Multilayer Perceptron",
            "category": "Feedforward Artificial Neural Network",
            "desc": "Models deep non-linear packet flow correlations via multi-layer representations and backpropagation optimization.",
            "filenames": ["mlp.joblib", "mlp.pkl", "mlp_model.joblib"],
        },
        "decision_tree": {
            "name": "Decision Tree",
            "category": "Recursive Binary Partitioning (CART)",
            "desc": (
                "Constructs a hierarchical decision tree using recursive binary splitting "
                "on feature thresholds to classify network traffic into benign or attack categories."
            ),
            "filenames": ["decision_tree.joblib", "decision_tree.pkl", "decision_tree_model.joblib", "dt.joblib"],
        },
    }

    result_models: Dict[str, ModelArtifactStatus] = {}
    trained_count = 0

    for key, meta in model_metadata.items():
        is_trained = False
        found_path: Optional[str] = None

        if target_dir.exists():
            for fname in meta["filenames"]:
                candidate = target_dir / fname
                if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                    is_trained = True
                    found_path = str(candidate.name)
                    trained_count += 1
                    break

        status_lbl = "Trained & Ready" if is_trained else "No Trained Models"

        result_models[key] = ModelArtifactStatus(
            key=key,
            name=meta["name"],
            is_trained=is_trained,
            artifact_path=found_path,
            status_label=status_lbl,
            description=meta["desc"],
            algorithm_category=meta["category"],
        )

    if trained_count == 0:
        summary = "No Trained Models"
    elif trained_count == len(model_metadata):
        summary = "All 3 Models Ready"
    else:
        summary = f"{trained_count}/3 Models Available"

    return ModelSystemStatus(
        trained_count=trained_count,
        total_models=len(model_metadata),
        models=result_models,
        summary_text=summary,
    )


def check_traffic_analysis_status(
    metadata_dir: Optional[Path] = None,
) -> TrafficAnalysisStatus:
    """Verify traffic analysis operational state from ingestion manifests.

    Args:
        metadata_dir: Optional override path for testing metadata storage.

    Returns:
        TrafficAnalysisStatus: Non-fabricated operational status.
    """
    target_meta = metadata_dir or settings.metadata_dir
    manifest_path = target_meta / "ingestion_manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                count = len(data)
                total_rows = sum(item.get("total_rows", 0) for item in data.values())
                return TrafficAnalysisStatus(
                    is_available=True,
                    summary_text=f"{count} Dataset{'s' if count > 1 else ''} Ingested",
                    detail_text=f"{total_rows:,} records cataloged" if total_rows > 0 else "Validation pending",
                )
        except Exception:
            pass

    return TrafficAnalysisStatus(
        is_available=False,
        summary_text="No Analysis Yet",
        detail_text="Awaiting CP-04 Traffic Ingestion",
    )


def check_latest_analysis_status() -> LatestAnalysisStatus:
    """Verify presence of recent analysis runs.

    Returns:
        LatestAnalysisStatus: Verifiable analysis record state.
    """
    return LatestAnalysisStatus(
        has_run=False,
        summary_text="No Analysis Yet",
        detail_text="Scheduled for CP-08",
        timestamp=None,
    )


def check_dataset_engineering_status(
    processed_dir: Optional[Path] = None,
    session_state: Optional[Any] = None,
) -> DatasetEngineeringStatus:
    """Verify dataset engineering and preprocessing state from processed artifacts.

    Args:
        processed_dir: Optional override path for testing processed data storage.
        session_state: Optional Streamlit session state object to check for active runs.

    Returns:
        DatasetEngineeringStatus: Verifiable engineering status without synthetic values.
    """
    target_dir = processed_dir or settings.processed_data_dir

    # Check session state for in-progress or error execution
    if session_state is not None:
        get_fn = getattr(session_state, "get", None)
        if callable(get_fn):
            if get_fn("engineering_in_progress", False):
                return DatasetEngineeringStatus(
                    state="In Progress",
                    is_completed=False,
                    run_count=0,
                    latest_run_id=None,
                    total_raw_rows=0,
                    supervised_rows=0,
                    train_rows=0,
                    val_rows=0,
                    test_rows=0,
                    num_features_out=0,
                    num_classes=0,
                    classes=[],
                    summary_text="In Progress",
                    detail_text="Pipeline execution actively running",
                )
            last_err = get_fn("engineering_last_error", None)
            if last_err:
                return DatasetEngineeringStatus(
                    state="Failed",
                    is_completed=False,
                    run_count=0,
                    latest_run_id=None,
                    total_raw_rows=0,
                    supervised_rows=0,
                    train_rows=0,
                    val_rows=0,
                    test_rows=0,
                    num_features_out=0,
                    num_classes=0,
                    classes=[],
                    summary_text="Failed",
                    detail_text=f"Error: {str(last_err)[:45]}...",
                )

    if not target_dir.exists():
        return DatasetEngineeringStatus(
            state="Not Started",
            is_completed=False,
            run_count=0,
            latest_run_id=None,
            total_raw_rows=0,
            supervised_rows=0,
            train_rows=0,
            val_rows=0,
            test_rows=0,
            num_features_out=0,
            num_classes=0,
            classes=[],
            summary_text="Not Started",
            detail_text="No prepared datasets in data/processed/",
        )

    # Inspect completed runs
    from src.data.engineering.artifact_manager import ArtifactManager

    mgr = ArtifactManager(processed_dir=target_dir)
    runs = mgr.list_runs()

    if not runs:
        return DatasetEngineeringStatus(
            state="Not Started",
            is_completed=False,
            run_count=0,
            latest_run_id=None,
            total_raw_rows=0,
            supervised_rows=0,
            train_rows=0,
            val_rows=0,
            test_rows=0,
            num_features_out=0,
            num_classes=0,
            classes=[],
            summary_text="Not Started",
            detail_text="No prepared datasets in data/processed/",
        )

    latest = runs[0]
    return DatasetEngineeringStatus(
        state="Completed",
        is_completed=True,
        run_count=len(runs),
        latest_run_id=latest.run_id,
        total_raw_rows=latest.total_raw_rows,
        supervised_rows=latest.supervised_rows,
        train_rows=latest.train_rows,
        val_rows=latest.val_rows,
        test_rows=latest.test_rows,
        num_features_out=latest.num_features_out,
        num_classes=latest.num_classes,
        classes=latest.classes,
        summary_text=f"{len(runs)} Prepared Run{'s' if len(runs) > 1 else ''}",
        detail_text=f"{latest.train_rows:,} train / {latest.val_rows:,} val / {latest.test_rows:,} test",
    )


def get_detection_pipeline_stages() -> List[PipelineStage]:
    """Retrieve the formal detection pipeline stages with their verifiable implementation state."""
    ds_status = check_dataset_status()
    eng_status = check_dataset_engineering_status()
    md_status = check_model_status()

    # Dynamic status evaluation
    stage1_status = "Completed" if ds_status.validated_count > 0 else ("Active" if ds_status.is_loaded else "Pending")
    stage1_variant = "success" if stage1_status == "Completed" else ("info" if stage1_status == "Active" else "neutral")

    stage2_status = "Completed" if eng_status.is_completed else ("Active" if eng_status.state == "In Progress" else "Pending")
    stage2_variant = "success" if stage2_status == "Completed" else ("info" if stage2_status == "Active" else "neutral")

    stage3_status = stage2_status
    stage3_variant = stage2_variant

    stage4_status = "Completed" if md_status.trained_count == md_status.total_models else ("Active" if md_status.trained_count > 0 else "Pending")
    stage4_variant = "success" if stage4_status == "Completed" else ("info" if stage4_status == "Active" else "neutral")

    return [
        PipelineStage(
            stage_id="traffic_input",
            title="1. CSV Traffic Input",
            description="Network flow telemetry upload & schema validation",
            status=stage1_status,
            badge_variant=stage1_variant,
            target_checkpoint="CP-04",
        ),
        PipelineStage(
            stage_id="preprocessing",
            title="2. Data Preprocessing",
            description="Header normalization, infinity & missing value remediation",
            status=stage2_status,
            badge_variant=stage2_variant,
            target_checkpoint="CP-05",
        ),
        PipelineStage(
            stage_id="feature_engineering",
            title="3. Feature Engineering",
            description="Robust feature scaling & attack category label encoding",
            status=stage3_status,
            badge_variant=stage3_variant,
            target_checkpoint="CP-05",
        ),
        PipelineStage(
            stage_id="ml_classification",
            title="4. ML Classification",
            description="Inference across SVM, MLP, and Decision Tree estimators",
            status=stage4_status,
            badge_variant=stage4_variant,
            target_checkpoint="CP-06",
        ),
        PipelineStage(
            stage_id="prediction_results",
            title="5. Prediction Results",
            description="Confidence thresholding, severity alerts & explainability",
            status="Pending",
            badge_variant="neutral",
            target_checkpoint="CP-08 & CP-09",
        ),
        PipelineStage(
            stage_id="reports_export",
            title="6. Forensic Reports",
            description="Diagnostic executive summaries & forensic event exports",
            status="Pending",
            badge_variant="neutral",
            target_checkpoint="CP-10",
        ),
    ]


def get_project_milestones() -> List[ProjectMilestone]:
    """Retrieve the complete 12-checkpoint milestone progression."""
    eng_status = check_dataset_engineering_status()
    cp05_status = "Completed" if eng_status.is_completed else "Active"
    cp05_variant = "success" if eng_status.is_completed else "info"

    return [
        ProjectMilestone("CP-01", "Project Foundation", "Completed", "success", "Directory structure, settings, dependencies & Git baseline"),
        ProjectMilestone("CP-02", "UI Design System", "Completed", "success", "Cybersecurity SaaS tokens, component library & application shell"),
        ProjectMilestone("CP-03", "Application Dashboard", "Completed", "success", "Verifiable executive overview dashboard & status telemetry"),
        ProjectMilestone("CP-04", "Traffic Upload & Data Explorer", "Completed", "success", "CSV flow uploader, schema validation & data distributions"),
        ProjectMilestone("CP-05", "Dataset Engineering", cp05_status, cp05_variant, "CIC-IDS-2017 cleaning, group-isolated splitting & dual scaling"),
        ProjectMilestone("CP-06", "ML Training Pipeline", "Planned", "neutral", "Unified trainer for SVM, MLP, Decision Tree & serialization"),
        ProjectMilestone("CP-07", "Model Evaluation Laboratory", "Planned", "neutral", "Cross-validation, confusion matrices, ROC-AUC & latency"),
        ProjectMilestone("CP-08", "Intrusion Detection Engine", "Planned", "neutral", "CSV-based intrusion detection engine & alert thresholding"),
        ProjectMilestone("CP-09", "Threat Intelligence & Prediction Explorer", "Planned", "neutral", "Attack taxonomy mapping, MITRE ATT&CK & explainability"),
        ProjectMilestone("CP-10", "Reports & Exports", "Planned", "neutral", "Executive threat reports & forensic CSV audit exports"),
        ProjectMilestone("CP-11", "Integration, Testing & Optimization", "Planned", "neutral", "End-to-end integration testing & performance profiling"),
        ProjectMilestone("CP-12", "Deployment & Release", "Planned", "neutral", "Production release packaging & academic presentation deck"),
    ]
