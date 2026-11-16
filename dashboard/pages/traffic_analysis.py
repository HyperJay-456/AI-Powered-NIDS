"""Traffic Analysis & Data Explorer Page.

Provides professional multi-file CSV network flow ingestion, bounded-memory
validation, schema inspection, data quality diagnostics, ground-truth label
exploration, and interactive telemetry exploration.
"""

from pathlib import Path
from typing import Dict, List, Optional

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
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Traffic Analysis & Data Explorer view."""
    manifest = IngestionManifest()
    datasets = manifest.list_datasets()
    ingested_count = len(datasets)

    _render_header(ingested_count)
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
        title="Traffic Analysis & Data Explorer",
        subtitle="Upload, validate and explore network flow CSV datasets for cybersecurity analysis.",
        icon="📊",
        badges=[
            ("Mode: Local / Offline", "neutral"),
            ("Dataset Source: User-Uploaded", "neutral"),
            ("Supported Format: CSV", "neutral"),
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
