"""Traffic Analysis & Data Explorer Page.

Provides professional multi-file CSV network flow ingestion, bounded-memory
validation, schema inspection, data quality diagnostics, ground-truth label
exploration, and interactive telemetry exploration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from dashboard.components.actions import styled_table
from dashboard.components.badges import badge_html
from dashboard.components.cards import metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import (
    error_panel,
    info_panel,
    success_panel,
    warning_panel,
)
from src.data.csv_ingestion import (
    check_batch_upload_limits,
    format_file_size,
    save_uploaded_csv,
)
from src.data.csv_validator import load_preview_cache, validate_csv_dataset
from src.data.engineering import (
    ArtifactManager,
    ColumnRole,
    DatasetRegistry,
    DuplicateHandlingPolicy,
    EngineeringRunRecord,
    FeaturePolicyManager,
    PreprocessingPipeline,
    PreprocessingRunConfiguration,
    SourceAuditor,
    analyze_missing_values,
)
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Traffic Analysis & Data Explorer view with Ingestion and Preprocessing tabs."""
    manifest = IngestionManifest()
    datasets = manifest.list_datasets()
    ingested_count = len(datasets)

    _render_header(ingested_count)

    tab_ingest, tab_engineer = st.tabs([
        "📥 Traffic Ingestion & Explorer",
        "⚙️ Dataset Engineering & Preprocessing",
    ])

    with tab_ingest:
        _render_ingestion_tab(manifest, datasets)

    with tab_engineer:
        _render_engineering_tab(manifest, datasets)


def _render_ingestion_tab(manifest: IngestionManifest, datasets: List[DatasetRecord]) -> None:
    """Render Tab 1: Traffic Ingestion & Explorer (CP-04 capabilities)."""
    _render_upload_section(manifest)

    # Reload datasets after possible import
    datasets = manifest.list_datasets()
    if not datasets:
        _render_empty_state()
        return

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    selected_record = _render_dataset_library(datasets)

    if selected_record:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_dataset_summary(selected_record)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_dataset_preview(selected_record)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_column_explorer(selected_record)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_data_quality_section(selected_record)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_ground_truth_distribution(selected_record)


def _render_header(ingested_count: int) -> None:
    """Render Section A: Page Header."""
    status_text = f"{ingested_count} Dataset{'s' if ingested_count != 1 else ''} Ingested"
    status_variant = "success" if ingested_count > 0 else "neutral"

    page_header(
        title="Traffic Analysis & Telemetry Explorer",
        subtitle="Ingest, validate, and inspect network flow telemetry datasets with cryptographic byte integrity.",
        icon="📊",
        badges=[
            ("Flow Ingestion: Active", "info"),
            ("Format: Flow CSV", "neutral"),
            (status_text, status_variant),
        ],
    )


def _render_upload_section(manifest: IngestionManifest) -> None:
    """Render Section B: CSV Upload Interface with explicit action."""
    section_header(
        title="Import Network Flow Telemetry",
        description="Select one or multiple CIC-IDS-2017 benchmark CSV flow files for local storage and validation:",
    )

    st.markdown(
        f"""
        <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 0.75rem; line-height: 1.5;">
            <span>Configured file size limit: <strong>{settings.ingestion.max_upload_size_mb} MB</strong> per file</span>
            <span style="margin: 0 0.5rem;">•</span>
            <span>Batch RAM buffer threshold: <strong>{settings.ingestion.max_batch_upload_size_mb:,} MB</strong></span>
            <span style="margin: 0 0.5rem;">•</span>
            <span>Local storage target: <code>data/raw/</code></span>
            <span style="margin: 0 0.5rem;">•</span>
            <span>Batch validation chunk size: <strong>{settings.ingestion.default_chunk_size:,} rows</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="Upload Network Flow CSV Files",
        type=["csv"],
        accept_multiple_files=True,
        help="Select one or multiple CSV flow files. Only .csv format is supported.",
        key="traffic_csv_uploader",
    )

    col_help, col_sample_btn = st.columns([3, 1])
    with col_help:
        st.caption("Don't have a multi-gigabyte CIC-IDS-2017 flow CSV? Load the standard 1,500-flow benchmark sample with one click:")
    with col_sample_btn:
        if st.button("⚡ Load Benchmark Sample", key="btn_traffic_load_sample", use_container_width=True):
            from src.data.demo_dataset import install_benchmark_sample_dataset
            with st.spinner("Generating and validating benchmark sample CSV..."):
                install_benchmark_sample_dataset()
                st.toast("Benchmark sample installed successfully!", icon="✅")
                st.rerun()

    if uploaded_files:
        st.markdown(
            "<div style='font-size: 0.8125rem; font-weight: 600; text-transform: uppercase; "
            "letter-spacing: 0.05em; color: var(--color-text-secondary); margin: 0.75rem 0 0.375rem 0;'>"
            "Selected Files Staged for Import:</div>",
            unsafe_allow_html=True,
        )

        staged_data = [
            {
                "Filename": f.name,
                "File Size": format_file_size(getattr(f, "size", 0) or 0),
                "Format": "CSV",
                "Action Required": "Click 'Import & Validate Files' below",
            }
            for f in uploaded_files
        ]
        styled_table(staged_data)

        # Evaluate aggregate batch threshold post-upload
        batch_valid, batch_bytes, batch_msg = check_batch_upload_limits(uploaded_files)

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: var(--color-text-muted); margin: 0.5rem 0 0.75rem 0; line-height: 1.4;">
                <em>Upload Memory Note:</em> Streamlit buffers uploaded files in web process memory upon receipt.
                The aggregate limit ({settings.ingestion.max_batch_upload_size_mb:,} MB) prevents downstream storage
                processing of oversized batches, but does not prevent Streamlit's initial multipart upload memory buffer.
                Staged files are ingested sequentially into local storage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not batch_valid:
            error_panel(
                title="Aggregate Batch Limit Exceeded",
                message=batch_msg,
            )

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            import_clicked = st.button(
                "🚀 Import & Validate Files",
                type="primary",
                use_container_width=True,
                disabled=not batch_valid,
                key="btn_import_validate",
            )

        if import_clicked:
            progress_bar = st.progress(0, text="Initializing ingestion pipeline...")
            total_files = len(uploaded_files)

            for idx, file_obj in enumerate(uploaded_files, start=1):
                base_pct = (idx - 1) / total_files
                progress_bar.progress(
                    base_pct,
                    text=f"[{idx}/{total_files}] Storing and verifying '{file_obj.name}'...",
                )

                # 1. Atomic local storage & deduplication check
                ingest_res = save_uploaded_csv(file_obj)

                if not ingest_res.success:
                    error_panel(
                        title=f"Ingestion Rejected: {file_obj.name}",
                        message=ingest_res.message,
                    )
                    continue

                if ingest_res.is_duplicate:
                    info_panel(
                        title=f"Duplicate Skipped: {file_obj.name}",
                        message=ingest_res.message,
                    )

                # 2. Bounded-Memory Chunk Validation
                def val_progress(pct: float, msg: str) -> None:
                    overall = base_pct + (pct / total_files)
                    progress_bar.progress(
                        min(overall, 1.0),
                        text=f"[{idx}/{total_files}] {file_obj.name}: {msg}",
                    )

                if ingest_res.file_path and ingest_res.dataset_id:
                    val_record = validate_csv_dataset(
                        file_path=ingest_res.file_path,
                        dataset_id=ingest_res.dataset_id,
                        progress_callback=val_progress,
                    )

                    if val_record.validation_status == "Validated":
                        st.toast(f"✅ {file_obj.name}: Validated successfully ({val_record.total_rows:,} rows)")
                    elif val_record.validation_status == "Validated with Warnings":
                        st.toast(f"⚠️ {file_obj.name}: Validated with {len(val_record.validation_warnings)} warnings")
                    else:
                        st.toast(f"❌ {file_obj.name}: Validation failed")

            progress_bar.progress(1.0, text="Batch ingestion and validation completed.")
            st.rerun()


def _render_empty_state() -> None:
    """Render empty state when no datasets exist."""
    info_panel(
        title="No Datasets Currently Cataloged",
        message=(
            "No network traffic CSV files are currently in local storage. "
            "Use the file uploader above to import CIC-IDS-2017 benchmark CSV datasets. "
            "Uploaded files are safely preserved in 'data/raw/' with original byte integrity."
        ),
        icon="📂",
    )


def _render_dataset_library(records: List[DatasetRecord]) -> Optional[DatasetRecord]:
    """Render Section F: Uploaded Dataset Library and selector."""
    section_header(
        title="Uploaded Dataset Library",
        description="Inspect cataloged benchmark datasets and select a target file for deep exploration:",
    )

    # Format table of all cataloged datasets
    library_data = []
    for r in records:
        library_data.append({
            "Stored Name": r.safe_filename,
            "Original Name": r.original_filename,
            "Size": r.file_size_formatted,
            "Total Rows": f"{r.total_rows:,}" if r.total_rows else "0",
            "Columns": str(r.total_columns),
            "Label Present": "Yes" if r.has_ground_truth_label else "No",
            "Validation Status": r.validation_status,
            "Imported At": r.import_timestamp[:19].replace("T", " ") if r.import_timestamp else "N/A",
        })
    styled_table(library_data)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Dataset selection dropdown
    options = {r.dataset_id: f"{r.safe_filename} ({r.file_size_formatted} — {r.total_rows:,} rows — {r.validation_status})" for r in records}
    selected_id = st.selectbox(
        "Active Dataset for Telemetry Exploration:",
        options=list(options.keys()),
        format_func=lambda k: options[k],
        key="active_dataset_select",
    )

    for r in records:
        if r.dataset_id == selected_id:
            return r
    return records[0] if records else None


def _render_dataset_summary(record: DatasetRecord) -> None:
    """Render Section A of Explorer: Dataset Summary."""
    section_header(
        title=f"Dataset Metrics: {record.safe_filename}",
        description=f"Structural verification and quality accounting (Original: {record.original_filename}):",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            label="File Size & Checksum",
            value=record.file_size_formatted,
            subtitle=f"SHA: {record.sha256_checksum[:8]}...",
        )

    with col2:
        metric_card(
            label="Total Dimensions",
            value=f"{record.total_rows:,}",
            subtitle=f"{record.total_columns} Columns ({record.encoding})",
        )

    with col3:
        status_color = "var(--color-status-success)" if record.invalid_rows == 0 else "var(--color-status-error)"
        metric_card(
            label="Row Accounting",
            value=f"{record.valid_rows:,} Valid",
            subtitle=f"{record.invalid_rows} Structurally Invalid",
            status_color=status_color,
        )

    with col4:
        label_text = "Label Present" if record.has_ground_truth_label else "No Label"
        label_color = "var(--color-status-success)" if record.has_ground_truth_label else "var(--color-status-warning)"
        metric_card(
            label="Ground-Truth Label",
            value=label_text,
            subtitle=record.label_column_name or "Awaiting Ground Truth",
            status_color=label_color,
        )

    # Invariant Equation Callout
    inv_held = (record.total_rows == (record.valid_rows + record.invalid_rows))
    st.markdown(
        f"""
        <div class="nids-card" style="margin-top: 0.75rem; border-color: {'var(--color-border-default)' if inv_held else 'var(--color-status-error)'};">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span style="font-weight: 700; color: var(--color-text-primary); font-size: 0.875rem;">
                        Row-Accounting Invariant:
                    </span>
                    <code style="font-size: 0.875rem; color: var(--color-accent-primary);">
                        Total Rows ({record.total_rows:,}) = Structurally Valid ({record.valid_rows:,}) + Invalid ({record.invalid_rows:,})
                    </code>
                </div>
                <div>
                    {badge_html("Invariant Verified", variant="success" if inv_held else "error")}
                    {badge_html(record.validation_status, variant="success" if record.validation_status == "Validated" else ("warning" if "Warnings" in record.validation_status else "error"))}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_preview(record: DatasetRecord) -> None:
    """Render Section B of Explorer: Dataset Preview."""
    section_header(
        title="Dataset Telemetry Preview",
        description="Inspect sample flow records from the ingested file:",
    )

    preview_df = load_preview_cache(record.dataset_id)
    if preview_df is None or preview_df.empty:
        # Fallback: attempt to read head directly if cache missing
        try:
            target_path = Path(record.file_path)
            if target_path.exists():
                preview_df = pd.read_csv(target_path, nrows=50, encoding=record.encoding or "utf-8")
        except Exception as e:
            logger.warning(f"Failed to read preview from file {record.file_path}: {e}")

    if preview_df is None or preview_df.empty:
        info_panel("Preview Unavailable", "No preview records cached for this file.")
        return

    # Preview controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        row_count = st.selectbox(
            "Display Rows:",
            options=[10, 25, 50, 100],
            index=1,
            key=f"preview_rows_{record.dataset_id}",
        )
    with ctrl_col2:
        search_term = st.text_input(
            "Filter Preview:",
            placeholder="Search keywords in preview...",
            key=f"preview_search_{record.dataset_id}",
        )

    # Apply search filter across preview only
    display_df = preview_df.head(row_count)
    if search_term:
        mask = display_df.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False, regex=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.markdown(
        f"<div style='font-size: 0.75rem; color: var(--color-text-muted); margin-bottom: 0.5rem;'>"
        f"Displaying {len(display_df)} sample records. "
        "This preview represents a bounded-memory local window and does not alter full-dataset metrics.</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(display_df, use_container_width=True)


def _render_column_explorer(record: DatasetRecord) -> None:
    """Render Section C of Explorer: Column Explorer."""
    section_header(
        title="Schema & Column Explorer",
        description=f"Inferred data types, completeness, and classifications across all {len(record.columns)} attributes:",
    )

    if not record.columns:
        info_panel("Column Schema Not Available", "No column metadata has been recorded for this dataset.")
        return

    # Filter by classification
    categories = ["All", "Numeric", "Categorical", "Label", "Mixed"]
    col_filter, _ = st.columns([1, 3])
    with col_filter:
        sel_cat = st.selectbox("Filter by Attribute Type:", options=categories, key=f"cat_filter_{record.dataset_id}")

    col_data = []
    for c in record.columns:
        if sel_cat != "All" and c.classification != sel_cat:
            continue
        col_data.append({
            "Column Name": c.name,
            "Classification": c.classification,
            "Inferred Type": c.dtype,
            "Non-Null Count": f"{c.non_null_count:,}",
            "Missing Count": f"{c.missing_count:,}",
            "Missing %": f"{c.missing_percentage:.2f}%",
            "Non-Finite (Inf)": str(c.numeric_anomalies),
        })

    styled_table(col_data)


def _render_data_quality_section(record: DatasetRecord) -> None:
    """Render Section D of Explorer: Data Quality Overview."""
    section_header(
        title="Data Quality & Integrity Diagnostics",
        description="Comprehensive audit of missing cells, duplicate records, and numeric flow anomalies:",
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        metric_card(
            label="Missing-Value Rows",
            value=f"{record.missing_value_rows:,}",
            subtitle=f"{((record.missing_value_rows / record.total_rows * 100) if record.total_rows else 0):.2f}% of Dataset",
            status_color="var(--color-status-warning)" if record.missing_value_rows > 0 else None,
        )
    with q2:
        metric_card(
            label="Exact Duplicate Rows",
            value=f"{record.duplicate_rows:,}",
            subtitle="SHA-256 SQLite Bounded Index",
            status_color="var(--color-status-warning)" if record.duplicate_rows > 0 else None,
        )
    with q3:
        metric_card(
            label="Numeric Flow Anomalies",
            value=f"{record.numeric_anomaly_count:,}",
            subtitle="Infinite Values (inf / -inf)",
            status_color="var(--color-status-warning)" if record.numeric_anomaly_count > 0 else None,
        )
    with q4:
        metric_card(
            label="Structural Invalid Rows",
            value=f"{record.invalid_rows:,}",
            subtitle="Malformed CSV Records",
            status_color="var(--color-status-error)" if record.invalid_rows > 0 else None,
        )

    # Warnings / Errors panels
    if record.validation_errors:
        error_panel(
            title="Validation Errors Encountered",
            message=" • ".join(record.validation_errors),
        )

    if record.validation_warnings:
        warning_panel(
            title="Data Quality Notices & Warnings",
            message=" • ".join(record.validation_warnings),
        )


def _render_ground_truth_distribution(record: DatasetRecord) -> None:
    """Render Section E of Explorer: Ground-Truth Distribution."""
    section_header(
        title="Ground-Truth Label Distribution",
        description="Actual attack family distributions from benchmark labels (not model predictions):",
    )

    if not record.has_ground_truth_label or not record.label_distribution:
        info_panel(
            title="No Ground-Truth Labels Found",
            message=(
                "No ground-truth 'Label' column was identified in this dataset. "
                "Supervised model evaluation metrics cannot be derived from this file, "
                "but the data remains fully available for exploratory feature analysis."
            ),
            icon="ℹ️",
        )
        return

    # Format distribution dataframe
    total_labeled = sum(record.label_distribution.values())
    dist_items = []
    for label, count in sorted(record.label_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_labeled * 100) if total_labeled > 0 else 0
        dist_items.append({
            "Traffic Class": label,
            "Record Count": count,
            "Proportion": f"{pct:.2f}%",
            "Percentage": pct,
        })

    dist_df = pd.DataFrame(dist_items)

    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        fig = px.bar(
            dist_df,
            x="Record Count",
            y="Traffic Class",
            orientation="h",
            text="Proportion",
            color_discrete_sequence=["#6385FF"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=20, t=10, b=20),
            xaxis=dict(showgrid=True, gridcolor="#1E293B"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#F1F5F9", size=12),
            height=max(260, len(dist_df) * 32),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        table_data = [
            {
                "Traffic Class": row["Traffic Class"],
                "Record Count": f"{row['Record Count']:,}",
                "Proportion": row["Proportion"],
            }
            for _, row in dist_df.iterrows()
        ]
        styled_table(table_data)


# ==============================================================================
# TAB 2: DATASET ENGINEERING & PREPROCESSING (CP-05 WORKFLOW)
# ==============================================================================


def _render_engineering_tab(manifest: IngestionManifest, datasets: List[DatasetRecord]) -> None:
    """Render Tab 2: Dataset Engineering & Preprocessing (CP-05 Pipeline)."""
    section_header(
        title="Dataset Engineering & Preprocessing Pipeline",
        description="Transform validated raw flow CSVs into reproducible, clean, dual-scaled partitions for ML training.",
    )

    registry = DatasetRegistry()
    eligible_sources = registry.get_eligible_sources()

    if not eligible_sources:
        info_panel(
            title="No Validated Datasets Available for Preprocessing",
            message=(
                "No eligible flow datasets were found in 'data/raw/'. "
                "Switch to the 'Traffic Ingestion & Explorer' tab above to upload and validate CSV datasets first."
            ),
            icon="ℹ️",
        )
        artifact_mgr = ArtifactManager()
        past_runs = artifact_mgr.list_runs()
        if past_runs:
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            _render_previous_runs_section(past_runs)
        return

    # Stage 1: Eligible Source Dataset Selection
    selected_filenames = _render_source_selection_stage(eligible_sources, registry)

    if not selected_filenames:
        info_panel(
            title="Select at Least One Source Dataset",
            message="Please select one or more validated datasets from the selector above to configure the preprocessing pipeline.",
            icon="👉",
        )
        return

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Stage 2: Missing Values & Data Quality Telemetry
    _render_data_quality_telemetry_stage(selected_filenames)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Stage 3: Feature Preparation & Policy Manifest
    _render_feature_policy_stage(selected_filenames)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Stage 4: Preprocessing Configuration
    config = _render_configuration_stage(selected_filenames)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Stage 5: Execution Controls & Progress Telemetry
    _render_execution_controls_stage(config)

    # Stage 6: Engineering Results & Row Accounting
    artifact_mgr = ArtifactManager()
    past_runs = artifact_mgr.list_runs()
    if past_runs:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        active_run_id = st.session_state.get("active_engineering_run_id", past_runs[0].run_id)
        active_run = next((r for r in past_runs if r.run_id == active_run_id), past_runs[0])
        _render_engineering_results_stage(active_run, artifact_mgr)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        _render_previous_runs_section(past_runs)


def _render_source_selection_stage(
    eligible_sources: List[Any],
    registry: DatasetRegistry,
) -> List[str]:
    """Stage 1: Eligible Source Dataset Selection with SHA-256 disk integrity."""
    section_header(
        title="Stage 1: Source Dataset Selection & Integrity Verification",
        description="Select verified benchmark flow files from data/raw/ to include in the ML preprocessing pipeline:",
    )

    available_names = [s.filename for s in eligible_sources]
    default_selected = [available_names[0]] if available_names else []

    selected_files = st.multiselect(
        label="Select Input Flow Datasets:",
        options=available_names,
        default=default_selected,
        help="Choose one or more validated datasets. Multi-file selections will be audited for schema compatibility.",
        key="selected_engineering_sources",
    )

    if not selected_files:
        return []

    # Display verification table
    table_rows = []
    for s in eligible_sources:
        if s.filename in selected_files:
            integrity_ok = registry.verify_source_integrity(s.filename)
            status_text = "Verified Checksum Match" if integrity_ok else "Integrity Mismatch"
            badge_var = "success" if integrity_ok else "error"
            table_rows.append({
                "Source Filename": s.filename,
                "Valid Rows": f"{s.valid_rows:,}",
                "Columns": s.total_columns,
                "SHA-256 Fingerprint": f"{s.sha256_checksum[:12]}...{s.sha256_checksum[-8:]}",
                "Disk Integrity": badge_html(status_text, variant=badge_var, show_dot=False),
            })
    styled_table(table_rows)

    # Multi-file schema audit
    if len(selected_files) > 1:
        auditor = SourceAuditor()
        file_paths = [settings.raw_data_dir / fn for fn in selected_files]
        audit_res = auditor.audit_sources(file_paths)
        if audit_res.is_compatible:
            success_panel(
                title="Multi-Source Schema Compatibility Confirmed",
                message=f"All {len(selected_files)} selected datasets share compatible schemas with {len(audit_res.common_columns)} aligned flow attributes.",
            )
        else:
            warning_panel(
                title="Schema Divergence Detected",
                message=f"Selected files have differing column schemas: {', '.join(audit_res.incompatible_reasons)}. Preprocessing will reconcile matching features and quarantine divergences.",
            )

    return selected_files


def _render_data_quality_telemetry_stage(selected_filenames: List[str]) -> None:
    """Stage 2: Missing Values & Data Quality Telemetry."""
    section_header(
        title="Stage 2: Data Quality & Missing Value Diagnostics",
        description="Evaluate missing, infinite, and anomalous flow telemetry before partition splitting:",
    )

    first_fn = selected_filenames[0]
    preview_df = load_preview_cache(first_fn)

    if preview_df is None:
        try:
            preview_df = pd.read_csv(settings.raw_data_dir / first_fn, nrows=1000)
        except Exception:
            preview_df = None

    if preview_df is not None:
        missing_report = analyze_missing_values(preview_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                label="Missing Cells",
                value=f"{missing_report.total_missing_cells:,}",
                subtitle="NaN / null values in sample",
                status_color="var(--color-status-warning)" if missing_report.total_missing_cells > 0 else None,
            )
        with c2:
            metric_card(
                label="Columns with Missing",
                value=str(missing_report.num_columns_with_missing),
                subtitle=f"Out of {len(preview_df.columns)} features",
                status_color="var(--color-status-warning)" if missing_report.num_columns_with_missing > 0 else None,
            )
        with c3:
            metric_card(
                label="Infinite Cells (inf / -inf)",
                value=f"{missing_report.total_infinite_cells:,}",
                subtitle="Remediated to NaN before imputation",
                status_color="var(--color-status-warning)" if missing_report.total_infinite_cells > 0 else None,
            )
        with c4:
            metric_card(
                label="Max Missing in Single Column",
                value=f"{missing_report.max_column_missing_pct:.2f}%",
                subtitle="Highest missing attribute %",
                status_color="var(--color-status-warning)" if missing_report.max_column_missing_pct > 0 else None,
            )

        if missing_report.missing_cells_per_column:
            top_missing = sorted(
                missing_report.missing_cells_per_column.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
            m_df = pd.DataFrame([{"Feature": k, "Missing Count": v} for k, v in top_missing])
            fig = px.bar(
                m_df,
                x="Missing Count",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#8B5CF6"],
                title="Top Features by Missing Value Count",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=20, t=30, b=20),
                xaxis=dict(showgrid=True, gridcolor="#1E293B"),
                yaxis=dict(autorange="reversed"),
                font=dict(color="#F1F5F9", size=11),
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)

    info_panel(
        title="Target Label Separation Invariant",
        message=(
            "Target labels ('Label') will NEVER be imputed. Any sample with a missing, corrupted, or non-finite "
            "ground-truth label is strictly excluded from supervised partitions to prevent synthetic label bias. "
            "Feature missing values and infinities are imputed solely using training partition statistics."
        ),
        icon="🛡️",
    )


def _render_feature_policy_stage(selected_filenames: List[str]) -> None:
    """Stage 3: Feature Preparation & Policy Manifest."""
    section_header(
        title="Stage 3: Feature Policy & Identifier Exclusion",
        description="Deterministic classification of telemetry attributes to prevent feature leakage and memorization:",
    )

    first_fn = selected_filenames[0]
    preview_df = load_preview_cache(first_fn)
    if preview_df is None:
        try:
            preview_df = pd.read_csv(settings.raw_data_dir / first_fn, nrows=10)
        except Exception:
            return

    decisions = FeaturePolicyManager.classify_columns(list(preview_df.columns))

    numeric_count = sum(1 for d in decisions if d.role == ColumnRole.NUMERIC_FEATURE)
    cat_count = sum(1 for d in decisions if d.role == ColumnRole.CATEGORICAL_FEATURE)
    port_count = sum(1 for d in decisions if d.role == ColumnRole.PORT_FIELD)
    ident_count = sum(1 for d in decisions if d.role in (ColumnRole.IDENTIFIER_EXCLUDED, ColumnRole.TIMESTAMP_EXCLUDED))
    label_count = sum(1 for d in decisions if d.role == ColumnRole.GROUND_TRUTH_LABEL)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card(label="Numeric Features", value=str(numeric_count), subtitle="Standard / Unscaled Dual Scaling")
    with c2:
        metric_card(label="Categorical Features", value=str(cat_count), subtitle="Leakage-Free OneHotEncoder")
    with c3:
        metric_card(label="Candidate Port Fields", value=str(port_count), subtitle="Configurable Candidate Features", status_color="var(--color-accent-primary)")
    with c4:
        metric_card(label="Quarantined Identifiers", value=str(ident_count), subtitle="Isolated into Provenance Sidecar", status_color="var(--color-status-warning)")
    with c5:
        metric_card(label="Target Label Column", value=str(label_count), subtitle="Ground-Truth LabelEncoder", status_color="var(--color-status-success)")

    info_panel(
        title="Network Feature Policy & Identifier Isolation",
        message=(
            "To prevent machine learning models from memorizing specific host IP addresses or temporal flow timestamps, "
            "network identifiers ('Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'External IP') are excluded from model features "
            "by default and preserved in a provenance sidecar for forensic audits. Port fields ('Destination Port', 'Source Port') "
            "are treated as configurable candidate features with explicit inclusion/exclusion controls in Stage 4 below."
        ),
        icon="🔒",
    )

    with st.expander("📋 Inspect Full Column Classification Manifest"):
        manifest_data = [
            {
                "Column Name": d.column_name,
                "Assigned Role": d.role.value.replace("_", " ").title(),
                "Preprocessing Action": d.transformation_action,
                "Justification": d.reason,
            }
            for d in decisions
        ]
        styled_table(manifest_data)


def _render_configuration_stage(selected_filenames: List[str]) -> PreprocessingRunConfiguration:
    """Stage 4: Preprocessing Configuration."""
    section_header(
        title="Stage 4: Pipeline Configuration & Partition Strategy",
        description="Configure partition split ratios, random seed, duplicate handling, and imputation strategies:",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        train_pct = st.slider("Training Partition Ratio", min_value=50, max_value=85, value=70, step=5, format="%d%%")
        train_ratio = train_pct / 100.0
    with col2:
        val_pct = st.slider("Validation Partition Ratio", min_value=5, max_value=25, value=15, step=5, format="%d%%")
        val_ratio = val_pct / 100.0
    with col3:
        test_ratio = round(1.0 - train_ratio - val_ratio, 2)
        st.markdown(
            f"""
            <div class="nids-card" style="padding: 0.75rem; margin-top: 1.5rem; text-align: center;">
                <div style="font-size: 0.75rem; color: var(--color-text-secondary);">Test Partition (Derived)</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: var(--color-accent-primary);">{int(test_ratio * 100)}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        random_seed = st.number_input("Deterministic Random Seed", min_value=0, max_value=999999, value=42, step=1)
    with col5:
        dup_policy_choice = st.selectbox(
            "Duplicate Record Policy",
            options=[
                DuplicateHandlingPolicy.GROUP_ISOLATED.value,
                DuplicateHandlingPolicy.EXCLUDE.value,
                DuplicateHandlingPolicy.RETAIN_ALL.value,
            ],
            index=0,
            format_func=lambda x: {
                DuplicateHandlingPolicy.GROUP_ISOLATED.value: "Group-Isolated (Preserve frequency without leakage)",
                DuplicateHandlingPolicy.EXCLUDE.value: "Exclude All Duplicates (Strict unique rows)",
                DuplicateHandlingPolicy.RETAIN_ALL.value: "Retain All (Permit duplicate rows)",
            }[x],
        )
    with col6:
        num_imputer = st.selectbox("Numeric Imputer Strategy", options=["median", "mean"], index=0)

    col7, col8 = st.columns(2)
    with col7:
        cat_imputer = st.selectbox("Categorical Imputer Strategy", options=["most_frequent", "constant"], index=0)
    with col8:
        st.markdown(
            """
            <div class="nids-card" style="padding: 0.75rem; margin-top: 1.5rem;">
                <div style="font-size: 0.75rem; color: var(--color-text-secondary);">Dual Feature Scaling</div>
                <div style="font-size: 0.875rem; font-weight: 600; color: var(--color-status-success);">
                    ✓ Standard Scaler (SVM/MLP) &amp; Unscaled (Decision Tree)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col9, col10 = st.columns(2)
    with col9:
        inc_dst_port = st.checkbox(
            "Include Destination Port (Service port candidate)",
            value=True,
            help="Retains Destination Port as a discrete candidate feature (e.g. 80, 443, 22, 53).",
            key="cfg_inc_dst_port",
        )
    with col10:
        inc_src_port = st.checkbox(
            "Include Source Port (Client port candidate)",
            value=False,
            help="Retains Source Port as a candidate feature (disabled by default to reduce ephemeral client port noise).",
            key="cfg_inc_src_port",
        )

    return PreprocessingRunConfiguration(
        source_filenames=selected_filenames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=int(random_seed),
        duplicate_policy=DuplicateHandlingPolicy(dup_policy_choice),
        include_destination_port=inc_dst_port,
        include_source_port=inc_src_port,
        numeric_imputer_strategy=num_imputer,
        categorical_imputer_strategy=cat_imputer,
        enable_dual_scaling=True,
    )


def _render_execution_controls_stage(config: PreprocessingRunConfiguration) -> None:
    """Stage 5: Execution Controls & Progress Telemetry."""
    section_header(
        title="Stage 5: Pipeline Execution Controls",
        description="Execute deterministic dataset engineering across the selected source files:",
    )

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run_clicked = st.button("🚀 Start Dataset Engineering Pipeline", key="btn_run_pipeline", type="primary")

    if run_clicked:
        st.session_state["engineering_in_progress"] = True
        st.session_state["engineering_last_error"] = None

        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def on_progress(step: int, total_steps: int, stage_name: str, message: str) -> None:
            pct = min(1.0, max(0.0, step / float(total_steps)))
            progress_bar.progress(pct)
            status_box.markdown(f"**Stage {step}/{total_steps} — {stage_name}:** {message}")

        try:
            pipeline = PreprocessingPipeline(config=config)
            run_record = pipeline.execute_pipeline(progress_callback=on_progress)

            st.session_state["engineering_in_progress"] = False
            st.session_state["active_engineering_run_id"] = run_record.run_id
            success_panel(
                title="Dataset Preprocessing Successfully Completed!",
                message=f"Persisted model-ready partitions for run '{run_record.run_id}' with {run_record.supervised_rows:,} samples.",
            )
            st.rerun()
        except Exception as e:
            st.session_state["engineering_in_progress"] = False
            st.session_state["engineering_last_error"] = str(e)
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            error_panel(
                title="Dataset Engineering Pipeline Failed",
                message=f"Error encountered during execution: {e}. Please check raw CSV data integrity or schema compatibility.",
            )


def _render_engineering_results_stage(
    active_run: EngineeringRunRecord,
    artifact_mgr: ArtifactManager,
) -> None:
    """Stage 6: Engineering Results & Strict Row Accounting."""
    section_header(
        title="Stage 6: Engineering Results & Row Accounting",
        description=f"Audited metrics and verified row accounting for active run: {active_run.run_id}",
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card(label="Total Raw Rows", value=f"{active_run.total_raw_rows:,}")
    with c2:
        metric_card(label="Training Partition", value=f"{active_run.train_rows:,}", status_color="var(--color-accent-primary)")
    with c3:
        metric_card(label="Validation Partition", value=f"{active_run.val_rows:,}", status_color="var(--color-accent-primary)")
    with c4:
        metric_card(label="Test Partition", value=f"{active_run.test_rows:,}", status_color="var(--color-accent-primary)")
    with c5:
        metric_card(label="Model Matrix Dim", value=f"{active_run.supervised_rows:,} × {active_run.num_features_out}")
    with c6:
        metric_card(label="Identified Classes", value=str(active_run.num_classes), status_color="var(--color-status-success)")

    eq1_hold = (active_run.total_raw_rows == (
        active_run.supervised_rows +
        active_run.unlabeled_rows +
        active_run.quarantined_rows +
        active_run.excluded_duplicate_rows
    ))
    eq2_hold = (active_run.supervised_rows == (
        active_run.train_rows + active_run.val_rows + active_run.test_rows
    ))
    all_eq_hold = eq1_hold and eq2_hold

    accounting_badge = badge_html("Row-Accounting Verified ✓", variant="success") if all_eq_hold else badge_html("Accounting Discrepancy", variant="error")

    st.markdown(
        f"""
        <div class="nids-card" style="margin-top: 1rem; border-left: 4px solid var(--color-status-success);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div class="nids-card-title" style="margin: 0;">Strict Row-Accounting Audit Ledger</div>
                <div>{accounting_badge}</div>
            </div>
            <div style="font-size: 0.8125rem; color: var(--color-text-secondary); line-height: 1.8;">
                <div>
                    <strong>Equation 1 (Universe Accounting):</strong>
                    <code>Total Raw ({active_run.total_raw_rows:,}) = Supervised ({active_run.supervised_rows:,}) + Unlabeled ({active_run.unlabeled_rows:,}) + Quarantined ({active_run.quarantined_rows:,}) + Excluded Duplicates ({active_run.excluded_duplicate_rows:,})</code>
                </div>
                <div>
                    <strong>Equation 2 (Partition Accounting):</strong>
                    <code>Supervised ({active_run.supervised_rows:,}) = Train ({active_run.train_rows:,}) + Validation ({active_run.val_rows:,}) + Test ({active_run.test_rows:,})</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🛡️ Data Leakage Controls & Boundary Assessment (Tested vs. Unverified)"):
        st.markdown(
            """
            <div style="font-size: 0.8125rem; line-height: 1.7; color: var(--color-text-secondary);">
                <div style="font-weight: 600; color: var(--color-status-success); margin-bottom: 0.25rem;">
                    ✓ Tested &amp; Verified Leakage Controls (CP-05 Scope):
                </div>
                <ul style="margin-top: 0; padding-left: 1.25rem;">
                    <li><strong>Training-Only Preprocessing Fitting:</strong> All imputation medians, OneHotEncoder vocabularies, and StandardScaler parameters are learned exclusively from the training split (70%) after partitioning.</li>
                    <li><strong>Group-Isolated Duplicates:</strong> Exact duplicate flow samples are assigned unified group hashes and retained within the same partition, preventing identical sample leakage between train and val/test.</li>
                    <li><strong>Direct Identifier Exclusion:</strong> Flow ID, Source IP, Destination IP, and Timestamps are excluded from model features and preserved in a separate provenance sidecar.</li>
                    <li><strong>Candidate Port Flexibility:</strong> Source Port and Destination Port are treated as configurable candidate features with explicit user-controlled inclusion.</li>
                    <li><strong>High-Cardinality Capping:</strong> Categorical features are bounded with <code>max_categories</code> to prevent uncontrolled one-hot expansion.</li>
                </ul>
                <div style="font-weight: 600; color: var(--color-status-warning); margin-top: 0.75rem; margin-bottom: 0.25rem;">
                    ⚠️ Known Leakage Risks Outside CP-05 Verification Scope:
                </div>
                <ul style="margin-top: 0; padding-left: 1.25rem;">
                    <li><strong>Temporal Auto-Correlation:</strong> Flows are partitioned independently of arrival timestamp; temporal lookahead across multi-hour traffic bursts is not eliminated.</li>
                    <li><strong>Near-Duplicate Flows:</strong> Exact SHA-256 grouping isolates identical rows, but near-duplicate flows with jittered timestamps or slightly altered packet counts are not linked.</li>
                    <li><strong>Cross-Capture Host Subnets:</strong> Multi-day host IP or subnet overlap across separate raw CSV files is not audited across days.</li>
                    <li><strong>Downstream Modeling Leakage:</strong> Model hyperparameter selection and cross-validation leakage controls are deferred to CP-06 and CP-07.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    dist_file = artifact_mgr.processed_dir / active_run.run_id / "metadata" / "class_distribution.json"
    if dist_file.exists():
        try:
            with open(dist_file, "r", encoding="utf-8") as f:
                class_data = json.load(f)

            st.markdown(
                "<div style='font-size: 0.875rem; font-weight: 600; color: var(--color-text-primary); margin: 1.25rem 0 0.5rem 0;'>"
                "Partitioned Class Distribution & Imbalance Telemetry:</div>",
                unsafe_allow_html=True,
            )

            chart_rows = []
            table_rows = []
            for item in class_data:
                c_name = item.get("class_name", "")
                c_idx = item.get("class_index", 0)
                tot = item.get("total_count", 0)
                tr = item.get("train_count", 0)
                va = item.get("val_count", 0)
                te = item.get("test_count", 0)
                pct = item.get("total_percentage", 0.0)

                chart_rows.append({"Class": c_name, "Partition": "Train", "Count": tr})
                chart_rows.append({"Class": c_name, "Partition": "Validation", "Count": va})
                chart_rows.append({"Class": c_name, "Partition": "Test", "Count": te})

                table_rows.append({
                    "Class Name": c_name,
                    "Encoded Index": c_idx,
                    "Total Count": f"{tot:,}",
                    "Proportion": f"{pct:.2f}%",
                    "Train Count": f"{tr:,}",
                    "Val Count": f"{va:,}",
                    "Test Count": f"{te:,}",
                })

            col_chart, col_tbl = st.columns([3, 2])
            with col_chart:
                c_df = pd.DataFrame(chart_rows)
                fig = px.bar(
                    c_df,
                    x="Count",
                    y="Class",
                    color="Partition",
                    barmode="group",
                    orientation="h",
                    color_discrete_map={"Train": "#6385FF", "Validation": "#8B5CF6", "Test": "#10B981"},
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=20, t=20, b=20),
                    xaxis=dict(showgrid=True, gridcolor="#1E293B"),
                    yaxis=dict(autorange="reversed"),
                    font=dict(color="#F1F5F9", size=11),
                    height=max(260, len(class_data) * 36),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_tbl:
                styled_table(table_rows)
        except Exception as e:
            logger.warning(f"Could not load class distribution: {e}")


def _render_previous_runs_section(past_runs: List[EngineeringRunRecord]) -> None:
    """Stage 7: Previous Engineering Runs Explorer."""
    section_header(
        title="Stage 7: Versioned Engineering Runs Explorer",
        description="Inspect persisted preprocessing runs and designate the active dataset for CP-06 ML training:",
    )

    run_options = {r.run_id: f"{r.run_id} ({r.created_at_utc[:19]} UTC — {r.supervised_rows:,} samples)" for r in past_runs}
    current_active = st.session_state.get("active_engineering_run_id", past_runs[0].run_id)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_run_id = st.selectbox(
            "Select Run to Inspect:",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x],
            index=list(run_options.keys()).index(current_active) if current_active in run_options else 0,
            key="prev_run_selector",
        )
    with col_btn:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("Set as Active for CP-06", key="btn_set_active"):
            st.session_state["active_engineering_run_id"] = selected_run_id
            st.success(f"✓ Run '{selected_run_id}' designated as active for CP-06 ML training.")

    runs_table = [
        {
            "Run ID": r.run_id,
            "Created UTC": r.created_at_utc[:19],
            "Raw Rows": f"{r.total_raw_rows:,}",
            "Supervised Rows": f"{r.supervised_rows:,}",
            "Train / Val / Test": f"{r.train_rows:,} / {r.val_rows:,} / {r.test_rows:,}",
            "Features": f"{r.num_features_in} → {r.num_features_out}",
            "Classes": r.num_classes,
            "Policy": r.duplicate_policy,
            "Status": badge_html(r.status, variant="success" if r.status == "Completed" else "error", show_dot=False),
        }
        for r in past_runs
    ]
    styled_table(runs_table)
