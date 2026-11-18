"""Overview Page: Executive cybersecurity dashboard and project state."""

from typing import Dict, List
import streamlit as st

from config.settings import settings
from dashboard.components.actions import styled_table
from dashboard.components.badges import badge_html, render_badge
from dashboard.components.cards import metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import info_panel
from dashboard.status_checker import (
    DatasetEngineeringStatus,
    DatasetStatus,
    ModelEvaluationStatus,
    ModelSystemStatus,
    check_dataset_engineering_status,
    check_dataset_status,
    check_latest_analysis_status,
    check_model_evaluation_status,
    check_model_status,
    check_traffic_analysis_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)


def _render_header() -> None:
    """Render Section A: Dashboard Header."""
    eng_status = check_dataset_engineering_status(session_state=st.session_state)
    md_status = check_model_status()
    eval_status = check_model_evaluation_status()
    det_status = check_latest_analysis_status()

    if det_status.has_run:
        cp_badge = ("Release v1.0.0 Active", "success")
    elif eval_status.is_completed:
        cp_badge = ("CP-08 Ready", "success")
    elif md_status.trained_count > 0:
        cp_badge = ("CP-07 Active", "info")
    elif eng_status.is_completed:
        cp_badge = ("CP-06 Active", "info")
    else:
        cp_badge = ("CP-05 Active", "info")

    page_header(
        title="Network Security Overview",
        subtitle=(
            "AI-Powered Network Intrusion Detection System designed for CSV-based intrusion detection analysis "
            "against CIC-IDS-2017 benchmark telemetry using SVM, MLP, and Decision Tree architectures."
        ),
        icon="🛡️",
        badges=[
            ("Mode: Local / Offline", "neutral"),
            ("v1.0.0 Production", "success"),
            cp_badge,
        ],
    )


def _render_quickstart_section(
    dataset_status: DatasetStatus,
    model_status: ModelSystemStatus,
) -> None:
    """Render Section: Quickstart Benchmark & One-Click Pipeline Initializer."""
    from src.data.demo_dataset import (
        clear_demo_dataset,
        install_benchmark_sample_dataset,
        is_demo_installed,
        is_pipeline_initialized,
        run_demo_pipeline,
    )

    section_header(
        title="Quickstart & Benchmark Pipeline",
        description="One-click dataset loader and automated pipeline initializer for immediate testing without external multi-gigabyte downloads:",
    )

    demo_installed = is_demo_installed()
    pipeline_ready = is_pipeline_initialized()

    col_info, col_actions = st.columns([3, 2])
    with col_info:
        st.markdown(
            f"""
            <div class="nids-card" style="margin-bottom: 0;">
                <div class="nids-card-title">⚡ Instant Onboarding Controls</div>
                <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.6;">
                    <div><strong>Dataset Status:</strong> {"✅ Benchmark Sample Installed (1,500 rows, 4 classes)" if demo_installed else "⚪ Clean / Empty (Ready to load)"}</div>
                    <div><strong>Pipeline Status:</strong> {"✅ Pipeline Active & Models Trained" if pipeline_ready else "⚪ Pipeline Uninitialized"}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.8125rem; color: var(--color-text-muted);">
                        Click <strong>Run Full Pipeline</strong> to generate sample telemetry, execute preprocessing, train the 3 models, evaluate test holdout, and perform batch detection in ~2 seconds.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_actions:
        st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("⚡ Load Sample CSV", key="btn_load_sample_csv", use_container_width=True):
                with st.spinner("Generating and validating benchmark sample CSV..."):
                    csv_path, rec = install_benchmark_sample_dataset()
                    st.toast(f"Sample dataset loaded: {rec.safe_filename}", icon="✅")
                    st.rerun()

        with c_btn2:
            if st.button("🚀 Run Full Pipeline", key="btn_run_full_pipeline", use_container_width=True, type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_prog(msg: str, frac: float) -> None:
                    progress_bar.progress(frac)
                    status_text.caption(f"⚙️ {msg}")

                try:
                    res = run_demo_pipeline(progress_callback=update_prog)
                    st.toast(f"Pipeline complete! Classified {res['predicted_rows']} flows.", icon="🚀")
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline initialization failed: {e}")

        if demo_installed or pipeline_ready or dataset_status.is_loaded or model_status.trained_count > 0:
            if st.button("🧹 Reset to Clean State", key="btn_reset_clean_state", use_container_width=True):
                with st.spinner("Restoring clean repository state..."):
                    res = clear_demo_dataset()
                    st.toast("System restored to clean state.", icon="🧹")
                    st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def _render_executive_summary_cards(
    dataset_status: DatasetStatus,
    model_status: ModelSystemStatus,
) -> None:
    """Render Section B: Executive Summary Cards with verifiable state."""
    col1, col2, col3, col4 = st.columns(4)

    # 1. Dataset Readiness
    with col1:
        dataset_color = None if dataset_status.is_loaded else "var(--color-status-warning)"
        metric_card(
            label="Dataset Readiness",
            value=dataset_status.summary_text,
            subtitle=f"Target: {dataset_status.target_dataset_name}",
            status_color=dataset_color,
        )

    # 2. Available Trained Models
    with col2:
        model_color = None if model_status.trained_count > 0 else "var(--color-text-secondary)"
        metric_card(
            label="Available Trained Models",
            value=model_status.summary_text,
            subtitle="SVM, MLP, Decision Tree",
            status_color=model_color,
        )

    # 3. Traffic Analysis Status
    with col3:
        traffic_status = check_traffic_analysis_status()
        metric_card(
            label="Traffic Analysis Status",
            value=traffic_status.summary_text,
            subtitle=traffic_status.detail_text,
            status_color="var(--color-text-secondary)",
        )

    # 4. Dataset Engineering Status (CP-05)
    with col4:
        eng_status = check_dataset_engineering_status(session_state=st.session_state)
        status_color = "var(--color-status-success)" if eng_status.is_completed else (
            "var(--color-status-warning)" if eng_status.state == "Failed" else "var(--color-text-secondary)"
        )
        metric_card(
            label="Dataset Engineering",
            value=eng_status.summary_text,
            subtitle=eng_status.detail_text,
            status_color=status_color,
        )


def _render_dataset_readiness_section(dataset_status: DatasetStatus) -> None:
    """Render Section C: Dataset Readiness Section."""
    section_header(
        title="Dataset Readiness & Telemetry Source",
        description="Verifiable state of benchmark dataset storage and expected input files:",
    )

    col_meta, col_status = st.columns([1, 1])
    with col_meta:
        st.markdown(
            f"""
            <div class="nids-card" style="margin-bottom: 0;">
                <div class="nids-card-title">Specification Parameters</div>
                <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.8;">
                    <div><strong>Target Corpus:</strong> {dataset_status.target_dataset_name}</div>
                    <div><strong>Expected Input:</strong> {dataset_status.expected_format} (Flow Telemetry CSV)</div>
                    <div><strong>Local Storage Target:</strong> <code>{dataset_status.storage_path}</code></div>
                    <div><strong>Expected Feature Dimension:</strong> ~78 Statistical Attributes (Awaiting CSV Validation)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        badge_variant = "success" if dataset_status.is_loaded else "warning"
        status_label = dataset_status.validation_status if dataset_status.is_loaded else dataset_status.summary_text
        badge_markup = badge_html(status_label, variant=badge_variant)
        st.markdown(
            f"""
            <div class="nids-card" style="margin-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div class="nids-card-title" style="margin: 0;">Storage State</div>
                    {badge_markup}
                </div>
                <div style="font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.6;">
                    <div><strong>Detected Files:</strong> {dataset_status.file_count} compatible CSVs</div>
                    <div><strong>Directory Status:</strong> Initialized (Read-only verified)</div>
                    <div><strong>Validation Status:</strong> {dataset_status.validation_status}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if not dataset_status.is_loaded:
        info_panel(
            title="Dataset Availability Notice",
            message=(
                f"No compatible CSV files are currently detected in the local repository storage directory ('{dataset_status.storage_path}'). "
                "Users may upload CIC-IDS-2017 CSV flow files through the 'Traffic Analysis' navigation tab. "
                "Uploaded files will be validated and cataloged in 'data/raw/' with original byte integrity."
            ),
            icon="ℹ️",
        )

    # Verification checklist of expected raw flow files
    st.markdown(
        "<div style='font-size: 0.8125rem; font-weight: 600; text-transform: uppercase; "
        "letter-spacing: 0.05em; color: var(--color-text-secondary); margin-bottom: 0.25rem;'>"
        "Expected Benchmark Flow Files (Kaggle Specification — Awaiting Validation):</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 0.75rem;'>"
        "Note: The 8 filenames and ~78 feature dimension reflect the standard CIC-IDS-2017 Kaggle distribution specification. "
        "Exact filenames and column schemas are treated as expected specifications awaiting validation upon CSV ingestion in CP-04/05."
        "</p>",
        unsafe_allow_html=True,
    )

    file_checklist = []
    for fname in settings.dataset.raw_files:
        is_present = fname in dataset_status.available_files
        file_checklist.append({
            "Expected Flow File": fname,
            "Presence Status": "Available" if is_present else "Not Present",
            "Target Path": f"data/raw/{fname}",
            "Validation Stage": "Awaiting Validation (CP-04/05)",
        })
    styled_table(file_checklist)


def _render_ml_model_overview_section(model_status: ModelSystemStatus) -> None:
    """Render Section D: Machine Learning Model Overview (Strictly 3 authorized algorithms)."""
    section_header(
        title="Machine Learning Model Architectures",
        description="Status of the three authorized machine learning models (strictly SVM, MLP, Decision Tree) implemented in CP-06:",
    )

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    algorithm_keys = ["svm", "mlp", "decision_tree"]

    for idx, key in enumerate(algorithm_keys):
        model = model_status.models[key]
        badge_variant = "success" if model.is_trained else "neutral"
        status_badge = badge_html(model.status_label, variant=badge_variant)

        with columns[idx]:
            st.markdown(
                f"""
                <div class="nids-model-card">
                    <div>
                        <div class="nids-model-header">
                            <div class="nids-model-category">{model.algorithm_category}</div>
                            <h4 class="nids-model-title">{model.name}</h4>
                            <div class="nids-model-status-row">{status_badge}</div>
                        </div>
                        <div class="nids-model-desc">{model.description}</div>
                    </div>
                    <div class="nids-model-footer">
                        <span style="color: var(--color-text-muted);">Checkpoint: CP-06</span>
                        <span style="color: var(--color-accent-primary); font-weight: 500;">
                            {model.artifact_path or "No Artifact"}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_detection_pipeline_section() -> None:
    """Render Section E: Detection Pipeline Overview."""
    section_header(
        title="End-to-End Detection Pipeline",
        description="Planned operational workflow connecting raw flow CSVs to forensic threat reporting:",
    )

    stages = get_detection_pipeline_stages()

    st.markdown('<div class="nids-pipeline-grid">', unsafe_allow_html=True)
    cols = st.columns(len(stages))
    for idx, stage in enumerate(stages):
        with cols[idx]:
            badge = badge_html(stage.status, variant=stage.badge_variant, show_dot=False)
            st.markdown(
                f"""
                <div class="nids-pipeline-step">
                    <div class="nids-pipeline-step-header">
                        <span style="font-size: 0.6875rem; font-weight: 700; color: var(--color-accent-primary); text-transform: uppercase;">
                            {stage.target_checkpoint}
                        </span>
                        {badge}
                    </div>
                    <h5 class="nids-pipeline-step-title">{stage.title}</h5>
                    <p class="nids-pipeline-step-desc">{stage.description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_project_progress_section() -> None:
    """Render Section F: Project Progress Section (12 Milestones)."""
    section_header(
        title="Project Development Milestones",
        description="12-checkpoint academic engineering roadmap and implementation state:",
    )

    milestones = get_project_milestones()

    # Render in 4 columns of 3 rows
    for row_idx in range(0, len(milestones), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            m_idx = row_idx + col_idx
            if m_idx < len(milestones):
                m = milestones[m_idx]
                with cols[col_idx]:
                    active_class = " active" if m.status == "Active" else ""
                    badge = badge_html(m.status, variant=m.badge_variant, show_dot=(m.status != "Planned"))
                    st.markdown(
                        f"""
                        <div class="nids-milestone-item{active_class}" style="margin-bottom: 0.75rem;">
                            <div class="nids-milestone-header">
                                <span class="nids-milestone-code">{m.code}: {m.title}</span>
                                {badge}
                            </div>
                            <div class="nids-milestone-desc">{m.description}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def _render_getting_started_section() -> None:
    """Render Section H: Getting Started Section."""
    from dashboard.navigation import set_active_route

    section_header(
        title="Getting Started Workflow",
        description="Sequential workflow to execute end-to-end CSV-based intrusion detection analysis on network flows:",
    )

    steps = [
        (
            "Prepare CIC-IDS-2017 Dataset Files",
            "Prepare the CIC-IDS-2017 benchmark CSV files from official or repository sources for traffic analysis. Alternatively, use the Quickstart loader above for instant verification.",
            "Prerequisite",
            "traffic_analysis",
            "Traffic Analysis",
        ),
        (
            "Upload Network Traffic via Traffic Analysis",
            "CSV upload and automated schema validation through the Traffic Analysis module in Checkpoint CP-04.",
            "Ready (CP-04)",
            "traffic_analysis",
            "Upload CSVs",
        ),
        (
            "Execute Dataset Engineering",
            "Clean missing values, remediate infinite flow rates, and apply robust scaling to flow feature matrices in CP-05.",
            "Ready (CP-05)",
            "traffic_analysis",
            "Engineering Pipeline",
        ),
        (
            "Train Authorized ML Models",
            "Train Support Vector Machine, Multilayer Perceptron, and Decision Tree estimators on validated training splits in CP-06.",
            "Ready (CP-06)",
            "model_laboratory",
            "Model Laboratory",
        ),
        (
            "Benchmark & Evaluate Performance",
            "Review 5-fold cross-validation metrics, confusion matrices, ROC-AUC curves, and empirical inference timing in CP-07.",
            "Ready (CP-07)",
            "model_laboratory",
            "Evaluation Lab",
        ),
        (
            "CSV-Based Intrusion Detection Analysis",
            "Execute batch or sample flow classification on uploaded network traffic CSVs using trained models in CP-08.",
            "Ready (CP-08)",
            "detection_center",
            "Detection Center",
        ),
        (
            "Analyze Threat Predictions & Export",
            "Inspect feature discriminators, review threat intelligence taxonomies, and export forensic audit logs in CP-09 to CP-11.",
            "Ready (CP-09 to CP-11)",
            "prediction_explorer",
            "Prediction Explorer",
        ),
    ]

    st.markdown('<div class="nids-step-list">', unsafe_allow_html=True)
    for idx, (title, desc, status_text, target_route, btn_label) in enumerate(steps, start=1):
        is_prereq = "Prerequisite" in status_text
        badge_variant = "info" if is_prereq else "success"
        badge = badge_html(status_text, variant=badge_variant, show_dot=False)

        col_text, col_nav = st.columns([5, 1])
        with col_text:
            st.markdown(
                f"""
                <div class="nids-step-item" style="margin-bottom: 0.5rem;">
                    <div class="nids-step-left">
                        <div class="nids-step-num">{idx}</div>
                        <div>
                            <div class="nids-step-text" style="font-weight: 600;">{title}</div>
                            <div style="font-size: 0.8125rem; color: var(--color-text-secondary); margin-top: 0.125rem;">{desc}</div>
                        </div>
                    </div>
                    <div>{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_nav:
            st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
            if st.button("Go ➔", key=f"btn_step_nav_{idx}", help=f"Navigate to {btn_label}"):
                set_active_route(target_route)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def _render_preprocessed_datasets_section(eng_status: DatasetEngineeringStatus) -> None:
    """Render Section H: Preprocessed Dataset Pipeline & Engineering State (CP-05)."""
    section_header(
        title="Preprocessed Datasets & ML Partitions",
        description="Verifiable state of model-ready partitions and preprocessors in data/processed/:",
    )

    if eng_status.is_completed:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                label="Prepared Runs",
                value=f"{eng_status.run_count} Run{'s' if eng_status.run_count != 1 else ''}",
                subtitle=f"ID: {eng_status.latest_run_id[:16]}...",
                status_color="var(--color-status-success)",
            )
        with c2:
            metric_card(
                label="Supervised Samples",
                value=f"{eng_status.supervised_rows:,}",
                subtitle=f"{eng_status.train_rows:,} Tr / {eng_status.val_rows:,} Val / {eng_status.test_rows:,} Te",
                status_color="var(--color-accent-primary)",
            )
        with c3:
            metric_card(
                label="Target Dimensions",
                value=f"{eng_status.supervised_rows:,} × {eng_status.num_features_out}",
                subtitle="Dual scaled & encoded",
                status_color="var(--color-accent-primary)",
            )
        with c4:
            classes_preview = ", ".join(eng_status.classes[:2]) + ("..." if len(eng_status.classes) > 2 else "")
            metric_card(
                label="Identified Classes",
                value=f"{eng_status.num_classes} Classes",
                subtitle=classes_preview,
                status_color="var(--color-status-success)",
            )
    else:
        status_msg = (
            "No preprocessed dataset artifacts are currently available in 'data/processed/'. "
            "Execute the dataset engineering pipeline in Traffic Analysis to partition, clean, and dual-scale flow datasets for ML training."
            if eng_status.state == "Not Started" else
            ("The dataset engineering pipeline is currently running..." if eng_status.state == "In Progress" else f"Last engineering run failed: {eng_status.detail_text}")
        )
        info_panel(
            title=f"Dataset Engineering Status: {eng_status.state}",
            message=status_msg,
            icon="⚙️",
        )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Open Dataset Engineering Pipeline ⚙️", key="btn_nav_preprocessing"):
            from dashboard.navigation import set_active_route

            st.session_state["traffic_analysis_tab_index"] = 1
            set_active_route("traffic_analysis")
            st.rerun()


def render() -> None:
    """Render the master Overview Dashboard viewport."""
    # 1. Fetch verifiable system state
    dataset_status = check_dataset_status()
    model_status = check_model_status()
    eng_status = check_dataset_engineering_status(session_state=st.session_state)

    # 2. Section A: Dashboard Header
    _render_header()

    # 3. Section B: Executive Summary Cards
    _render_executive_summary_cards(dataset_status, model_status)

    # 4. Section: Quickstart Benchmark & Pipeline Controls
    _render_quickstart_section(dataset_status, model_status)

    # 5. Section C: Dataset Readiness Section
    _render_dataset_readiness_section(dataset_status)

    # 5. Section D: Preprocessed Datasets & ML Partitions (CP-05)
    _render_preprocessed_datasets_section(eng_status)

    # 6. Section E: Machine Learning Model Overview
    _render_ml_model_overview_section(model_status)

    # 7. Section F: Detection Pipeline Overview
    _render_detection_pipeline_section()

    # 8. Section G: Project Progress Section
    _render_project_progress_section()

    # 9. Section H: Getting Started Section
    _render_getting_started_section()
