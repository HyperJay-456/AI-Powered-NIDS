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
    DatasetStatus,
    ModelSystemStatus,
    check_dataset_status,
    check_latest_analysis_status,
    check_model_status,
    check_traffic_analysis_status,
    get_detection_pipeline_stages,
    get_project_milestones,
)


def _render_header() -> None:
    """Render Section A: Dashboard Header."""
    page_header(
        title="Network Security Overview",
        subtitle=(
            "AI-Powered Network Intrusion Detection System analyzing network flow telemetry "
            "against CIC-IDS-2017 attack signatures using SVM, MLP, and Decision Tree architectures."
        ),
        icon="🛡️",
        badges=[
            ("Mode: Local / Offline", "neutral"),
            ("CP-03 Active", "info"),
        ],
    )


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

    # 4. Latest Analysis
    with col4:
        latest_status = check_latest_analysis_status()
        metric_card(
            label="Latest Analysis",
            value=latest_status.summary_text,
            subtitle=latest_status.detail_text,
            status_color="var(--color-text-secondary)",
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
                    <div><strong>Expected Input:</strong> {dataset_status.expected_format} (Flow Telemetry)</div>
                    <div><strong>Storage Location:</strong> <code>{dataset_status.storage_path}</code></div>
                    <div><strong>Feature Dimension:</strong> 78 Statistical Attributes</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        badge_variant = "success" if dataset_status.is_loaded else "warning"
        badge_markup = badge_html(dataset_status.summary_text, variant=badge_variant)
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
                    <div><strong>Ingestion Status:</strong> Awaiting CP-05 Pipeline</div>
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
                f"No compatible CSV files were detected in '{dataset_status.storage_path}'. "
                f"The system expects raw CIC-IDS-2017 network traffic CSVs in this directory. "
                "Dataset files will be processed and validated during Checkpoint CP-05 (Dataset Engineering)."
            ),
            icon="ℹ️",
        )

    # Verification checklist of expected raw flow files
    st.markdown(
        "<div style='font-size: 0.8125rem; font-weight: 600; text-transform: uppercase; "
        "letter-spacing: 0.05em; color: var(--color-text-secondary); margin-bottom: 0.5rem;'>"
        "Expected Benchmark Flow Files:</div>",
        unsafe_allow_html=True,
    )

    file_checklist = []
    for fname in settings.dataset.raw_files:
        is_present = fname in dataset_status.available_files
        file_checklist.append({
            "Expected Flow File": fname,
            "Presence Status": "Available" if is_present else "Not Present",
            "Target Path": f"data/raw/{fname}",
            "Pipeline Stage": "CP-04 / CP-05",
        })
    styled_table(file_checklist)


def _render_ml_model_overview_section(model_status: ModelSystemStatus) -> None:
    """Render Section D: Machine Learning Model Overview (Strictly 3 authorized algorithms)."""
    section_header(
        title="Machine Learning Model Architectures",
        description="Status of the three authorized machine learning models (strictly SVM, MLP, Decision Tree):",
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
        description="Operational workflow connecting raw flow CSVs to forensic threat reporting:",
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
    """Render Section G: Getting Started Section."""
    section_header(
        title="Getting Started Workflow",
        description="Sequential workflow to execute end-to-end intrusion detection on network flows:",
    )

    steps = [
        ("Prepare Network Traffic CSV", "Obtain benchmark network flow records formatted according to the CIC-IDS-2017 feature specification.", "Prerequisite"),
        ("Upload Network Traffic File", "Ingest and validate the CSV flow records through the Traffic Upload interface.", "Upcoming: CP-04"),
        ("Execute Dataset Engineering", "Clean missing values, remediate infinite flow rates, and apply robust scaling.", "Upcoming: CP-05"),
        ("Train Authorized ML Models", "Train Support Vector Machine, Multilayer Perceptron, and Decision Tree estimators.", "Upcoming: CP-06"),
        ("Benchmark & Evaluate Performance", "Review 5-fold cross-validation metrics, confusion matrices, ROC-AUC, and latency.", "Upcoming: CP-07"),
        ("Run Intrusion Detection Engine", "Feed single flows or micro-batches into trained models for real-time threat categorization.", "Upcoming: CP-08"),
        ("Analyze Threat Predictions & Export", "Inspect feature discriminators, review threat intelligence, and export forensic logs.", "Upcoming: CP-09 & CP-10"),
    ]

    st.markdown('<div class="nids-step-list">', unsafe_allow_html=True)
    for idx, (title, desc, status_text) in enumerate(steps, start=1):
        is_prereq = "Prerequisite" in status_text
        badge_variant = "info" if is_prereq else "neutral"
        badge = badge_html(status_text, variant=badge_variant, show_dot=False)

        st.markdown(
            f"""
            <div class="nids-step-item">
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
    st.markdown('</div>', unsafe_allow_html=True)


def render() -> None:
    """Render the master Overview Dashboard viewport."""
    # 1. Fetch verifiable system state
    dataset_status = check_dataset_status()
    model_status = check_model_status()

    # 2. Section A: Dashboard Header
    _render_header()

    # 3. Section B: Executive Summary Cards
    _render_executive_summary_cards(dataset_status, model_status)

    # 4. Section C: Dataset Readiness Section
    _render_dataset_readiness_section(dataset_status)

    # 5. Section D: Machine Learning Model Overview
    _render_ml_model_overview_section(model_status)

    # 6. Section E: Detection Pipeline Overview
    _render_detection_pipeline_section()

    # 7. Section F: Project Progress Section
    _render_project_progress_section()

    # 8. Section G: Getting Started Section
    _render_getting_started_section()
