"""Detection Center Page: Operational CSV-based intrusion detection analysis and triage console (CP-08).

Provides:
- Section A: Verifiable Detection Overview & eligible model status
- Section B: Validated CP-04 dataset selection with training data overlap auditing
- Section C: Single-model or multi-model comparative selection (SVM, MLP, Decision Tree)
- Section D: Resource-aware configuration & RAM preflight
- Section E: Chunked batch inference with interactive progress
- Section F: Comprehensive detection results, class distributions, and consensus analysis
- Section G: Exportable prediction artifacts and historical run auditing
"""

from datetime import datetime
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
from src.data.ingestion_manifest import DatasetRecord, IngestionManifest
from src.detection.dataset_loader import DetectionDatasetLoader, VerifiedDatasetSource
from src.detection.detection_manifest import DetectionRunManifest
from src.detection.detection_registry import DetectionRegistry
from src.detection.inference_pipeline import DetectionInferencePipeline
from src.detection.model_eligibility import EligibleModelInfo, ModelEligibilityVerifier
from src.detection.resource_manager import DetectionResourceManager
from src.detection.score_interpreter import DetectionScoreInterpreter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Intrusion Detection Center operational console."""
    verifier = ModelEligibilityVerifier()
    eligible_models = verifier.discover_eligible_models(allow_test_fixtures=False)

    det_registry = DetectionRegistry()
    detection_runs = det_registry.list_detection_runs(include_test_fixtures=False, verify_checksums=False)

    # Page Header
    badge_variant = "success" if detection_runs else ("info" if eligible_models else "neutral")
    badge_text = f"{len(detection_runs)} Analysis Runs" if detection_runs else (f"{len(eligible_models)} Models Ready" if eligible_models else "Awaiting Models")

    page_header(
        title="Intrusion Detection Center",
        subtitle="Offline CSV-based network traffic inference, model consensus triage, and forensic analysis",
        icon="⚡",
        badge_text=badge_text,
        badge_variant=badge_variant,
    )

    tab_console, tab_history = st.tabs(["⚡ Live Detection Console", "📦 Detection Run History & Audit"])

    with tab_console:
        _render_detection_console(eligible_models, detection_runs)

    with tab_history:
        _render_detection_history(det_registry, detection_runs)


def _render_detection_console(
    eligible_models: List[EligibleModelInfo],
    detection_runs: List[Dict[str, Any]],
) -> None:
    """Render the primary operational inference console."""
    # --------------------------------------------------------------------------
    # SECTION A: Detection Overview
    # --------------------------------------------------------------------------
    section_header(
        title="Detection System Telemetry",
        description="Real-time operational readiness derived from verified model and analysis registries:",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            label="Eligible Production Models",
            value=f"{len(eligible_models)} / 3",
            subtitle="Authorized Algorithms",
        )

    with col2:
        metric_card(
            label="Completed Analyses",
            value=f"{len(detection_runs)}",
            subtitle="Historical Runs",
        )

    with col3:
        latest_run = detection_runs[0] if detection_runs else None
        latest_id = latest_run["detection_run_id"][:18] + "..." if latest_run else "None"
        metric_card(
            label="Latest Analysis Run",
            value=latest_id,
            subtitle=latest_run["created_at_utc"][:10] if latest_run else "No runs yet",
        )

    with col4:
        total_classified_rows = sum(r.get("predicted_rows", 0) for r in detection_runs)
        metric_card(
            label="Flows Analyzed",
            value=f"{total_classified_rows:,}",
            subtitle="Total Records",
        )

    # Empty State Callout when no eligible models exist
    if not eligible_models:
        warning_panel(
            title="No Eligible Production Model Available",
            message=(
                "No eligible production model is currently available. Train a model and complete the required "
                "evaluation and candidate-freeze steps first. "
                "Models must complete CP-06 training and CP-07 validation gating before operational deployment."
            ),
        )
        return

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION B: Dataset Selection
    # --------------------------------------------------------------------------
    section_header(
        title="Input Dataset Selection",
        description="Select a network flow CSV dataset previously uploaded and validated via CP-04:",
    )

    dataset_loader = DetectionDatasetLoader()
    available_datasets = dataset_loader.list_available_datasets()
    validated_datasets = [d for d in available_datasets if d.validation_status in ("Validated", "Validated with Warnings")]

    if not validated_datasets:
        info_panel(
            title="No Validated Datasets Available",
            message=(
                "No validated network traffic CSV files found in data/raw/. "
                "Please upload and validate datasets through the 'Traffic Upload & Data Explorer' page first."
            ),
        )
        return

    ds_options = {f"{d.safe_filename} ({d.file_size_formatted}, {d.valid_rows:,} rows)": d.dataset_id for d in validated_datasets}
    selected_ds_label = st.selectbox(
        "Choose Target Flow Dataset:",
        options=list(ds_options.keys()),
        index=0,
        key="sel_detection_dataset",
    )
    selected_dataset_id = ds_options[selected_ds_label]
    selected_rec = next(d for d in validated_datasets if d.dataset_id == selected_dataset_id)

    # Dataset details panel
    ds_col1, ds_col2, ds_col3 = st.columns(3)
    with ds_col1:
        st.markdown(f"**Dataset ID:** `{selected_rec.dataset_id}`")
        st.markdown(f"**Filename:** `{selected_rec.safe_filename}`")
    with ds_col2:
        st.markdown(f"**Size:** `{selected_rec.file_size_formatted}`")
        st.markdown(f"**Record Count:** `{selected_rec.valid_rows:,} rows`")
    with ds_col3:
        lbl_status = "Present (Reference Only)" if selected_rec.has_ground_truth_label else "Absent (Unlabeled)"
        st.markdown(f"**Ground-Truth Label:** `{lbl_status}`")
        st.markdown(f"**Validation Status:** `{selected_rec.validation_status}`")

    # Check for training data overlap (leakage warning)
    common_eng_id = eligible_models[0].engineering_run_id
    try:
        source_verified = dataset_loader.verify_and_prepare_source(selected_dataset_id, engineering_run_id=common_eng_id)
        if source_verified.overlaps_training_data:
            warning_panel(
                title="Training Data Overlap Notice ⚠️",
                message=(
                    f"The selected dataset '{selected_rec.safe_filename}' was an input to the training/evaluation "
                    f"dataset engineering run '{common_eng_id}'. This traffic overlaps with training data "
                    "and cannot be represented as unseen operational network traffic."
                ),
            )
    except Exception as e:
        error_panel(title="Dataset Integrity Error", message=f"Failed to verify dataset source: {e}")
        return

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION C: Model Selection
    # --------------------------------------------------------------------------
    section_header(
        title="Detection Model Selection",
        description="Select an individual algorithm or execute a multi-model comparative analysis:",
    )

    mode = st.radio(
        "Execution Mode:",
        options=["Single Model Inference", "Compare All Eligible Models"],
        index=1 if len(eligible_models) > 1 else 0,
        horizontal=True,
        key="rad_detection_mode",
    )

    selected_algos: List[str] = []
    algo_display_map = {m.algorithm: m.algorithm_display_name for m in eligible_models}

    if mode == "Single Model Inference":
        model_choices = {f"{m.algorithm_display_name} ({m.algorithm.upper()})": m.algorithm for m in eligible_models}
        sel_choice = st.selectbox(
            "Select Deployment Model:",
            options=list(model_choices.keys()),
            key="sel_single_model",
        )
        selected_algos = [model_choices[sel_choice]]
    else:
        # Multi-model comparison
        default_selected = [m.algorithm for m in eligible_models]
        selected_algos = st.multiselect(
            "Select Models to Compare:",
            options=[m.algorithm for m in eligible_models],
            default=default_selected,
            format_func=lambda x: f"{algo_display_map.get(x, x.upper())} ({x.upper()})",
            key="ms_multi_models",
        )

        if len(selected_algos) < 2:
            warning_panel(
                title="Comparative Mode Requires Multiple Models",
                message="Please select at least two models to perform a comparative consensus and disagreement analysis.",
            )

        info_panel(
            title="Multi-Model Consensus Architecture",
            message=(
                "Each selected model runs independently on the exact same flow telemetry without majority voting or ensembles. "
                "Row-level agreement and divergence are reported. Disagreements indicate model prediction uncertainty."
            ),
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION D: Configuration & Resource Preflight
    # --------------------------------------------------------------------------
    section_header(
        title="Inference Engine Configuration",
        description="Resource-aware batch processing parameters and hardware safety boundaries:",
    )

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        batch_size = st.slider(
            "Processing Batch Size (Rows per Chunk):",
            min_value=DetectionResourceManager.MIN_BATCH_SIZE,
            max_value=DetectionResourceManager.MAX_BATCH_SIZE,
            value=DetectionResourceManager.DEFAULT_BATCH_SIZE,
            step=1_000,
            help="Chunk size for reading CSV, applying sparse feature scaling, and running vector predictions.",
            key="slider_batch_size",
        )

    with cfg_col2:
        preflight = DetectionResourceManager.perform_preflight_check(
            num_features=eligible_models[0].feature_count or 80,
            batch_size=batch_size,
            num_models=len(selected_algos) or 1,
            target_path=settings.analyses_dir,
        )
        avail_ram_mb = f"{preflight.available_ram_bytes / (1024*1024):.1f} MB" if preflight.available_ram_bytes else "Detected"
        avail_disk_mb = f"{preflight.available_disk_bytes / (1024*1024):.1f} MB"
        chunk_ram_mb = f"{preflight.estimated_chunk_ram_bytes / (1024*1024):.1f} MB"

        st.markdown(f"**Available RAM:** `{avail_ram_mb}` | **Free Disk:** `{avail_disk_mb}`")
        st.markdown(f"**Estimated Chunk RAM:** `{chunk_ram_mb}`")
        if preflight.warning_messages:
            for w in preflight.warning_messages:
                st.caption(f"⚠️ {w}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION E: Run Detection
    # --------------------------------------------------------------------------
    section_header(
        title="Execute Intrusion Detection",
        description="Run batch model classification and consensus analysis on the selected traffic:",
    )

    btn_disabled = (len(selected_algos) == 0) or (mode == "Compare All Eligible Models" and len(selected_algos) < 2)

    if st.button("⚡ Run Intrusion Detection", type="primary", disabled=btn_disabled, key="btn_run_detection"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int, msg: str) -> None:
            pct = min(1.0, current / max(1, total))
            progress_bar.progress(pct)
            status_text.markdown(f"**Status:** {msg}")

        try:
            pipeline = DetectionInferencePipeline()
            manifest = pipeline.run_inference(
                dataset_id=selected_dataset_id,
                algorithms=selected_algos,
                batch_size=batch_size,
                allow_test_fixtures=False,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.markdown("**Status:** ✅ Inference completed successfully!")

            success_panel(
                title=f"Detection Run '{manifest.detection_run_id}' Completed! 🛡️",
                message=(
                    f"Successfully classified {manifest.row_accounting.successfully_predicted_rows:,} network flows "
                    f"across {len(manifest.selected_algorithms)} model(s). "
                    f"Results committed atomically to data/analyses/{manifest.detection_run_id}/."
                ),
            )
            st.session_state["active_detection_run_id"] = manifest.detection_run_id

        except Exception as e:
            logger.error(f"Inference run failed: {e}", exc_info=True)
            error_panel(title="Inference Execution Failed", message=str(e))

    # --------------------------------------------------------------------------
    # SECTION F: Detection Results
    # --------------------------------------------------------------------------
    active_run_id = st.session_state.get("active_detection_run_id")
    if not active_run_id and detection_runs:
        active_run_id = detection_runs[0]["detection_run_id"]

    if active_run_id:
        st.markdown("---")
        _render_detection_results(active_run_id)


def _render_detection_results(detection_run_id: str) -> None:
    """Render comprehensive detection results, class distributions, and consensus tables."""
    det_registry = DetectionRegistry()
    manifest = det_registry.get_detection_manifest(detection_run_id)
    if not manifest:
        warning_panel(title="Manifest Not Found", message=f"Could not load manifest for '{detection_run_id}'.")
        return

    predictions_df = det_registry.load_predictions_df(detection_run_id)
    if predictions_df is None or predictions_df.empty:
        warning_panel(title="Predictions Artifact Missing", message=f"No predictions.csv found for '{detection_run_id}'.")
        return

    section_header(
        title=f"Detection Results — Run `{detection_run_id}`",
        description=f"Classified {manifest.row_accounting.successfully_predicted_rows:,} network flow records against source `{manifest.source_filename}`:",
    )

    # 1. Metric Summary Cards
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    accounting = manifest.row_accounting
    with res_col1:
        metric_card(
            label="Total Input Records",
            value=f"{accounting.input_rows:,}",
            subtitle="Flows Ingested",
        )

    with res_col2:
        metric_card(
            label="Classified Predictions",
            value=f"{accounting.successfully_predicted_rows:,}",
            subtitle="Successful",
        )

    with res_col3:
        metric_card(
            label="Rejected / Skipped",
            value=f"{accounting.rejected_rows + accounting.skipped_rows:,}",
            subtitle="Records",
        )

    with res_col4:
        models_count = len(manifest.selected_algorithms)
        metric_card(
            label="Models Evaluated",
            value=f"{models_count}",
            subtitle="Independent Models",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Predicted Class Distribution Charts
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("#### Predicted Class Distribution")
        # Build bar chart of predictions
        chart_data = []
        for algo, dist in manifest.class_distribution.items():
            disp = manifest.models.get(algo, None)
            disp_name = disp.algorithm_display_name if disp else algo.upper()
            for cls_name, count in dist.items():
                chart_data.append({"Algorithm": disp_name, "Predicted Class": cls_name, "Record Count": count})

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(
                df_chart,
                x="Predicted Class",
                y="Record Count",
                color="Algorithm",
                barmode="group",
                title="Predicted Flow Classes by Model",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0E0"},
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("#### Attack vs Benign Ratio")
        # Compute benign vs attack for primary model
        primary_algo = manifest.selected_algorithms[0]
        dist = manifest.class_distribution.get(primary_algo, {})
        benign_cnt = dist.get("BENIGN", 0)
        attack_cnt = sum(cnt for cls_n, cnt in dist.items() if cls_n != "BENIGN")

        pie_df = pd.DataFrame([
            {"Category": "Benign Baseline", "Count": benign_cnt},
            {"Category": "Classified Attacks", "Count": attack_cnt},
        ])
        fig_pie = px.pie(
            pie_df,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={"Benign Baseline": "#22C55E", "Classified Attacks": "#EF4444"},
            hole=0.45,
            title=f"Classification Ratio ({primary_algo.upper()})",
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E0E0E0"})
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Model Consensus & Disagreement Section (when multiple models)
    if manifest.disagreement_summary and len(manifest.selected_algorithms) > 1:
        st.markdown("---")
        section_header(
            title="Model Consensus & Disagreement Analysis",
            description="Comparing independent predictions across selected models for every network flow record:",
        )

        d_sum = manifest.disagreement_summary
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        with d_col1:
            metric_card(
                label="Full Agreement",
                value=f"{d_sum.agreement_count:,}",
                subtitle=f"{d_sum.agreement_percentage:.1f}% Consensus",
            )

        with d_col2:
            metric_card(
                label="Model Disagreement",
                value=f"{d_sum.disagreement_count:,}",
                subtitle=f"{d_sum.disagreement_percentage:.1f}% Divergence",
            )

        with d_col3:
            top_pairwise = list(d_sum.pairwise_agreements.items())[0] if d_sum.pairwise_agreements else ("N/A", 0.0)
            metric_card(
                label="Top Pairwise Match",
                value=f"{top_pairwise[1]:.1f}%",
                subtitle=top_pairwise[0].replace("_vs_", " vs "),
            )

        with d_col4:
            metric_card(
                label="Triage Indicator",
                value=f"{d_sum.disagreement_count:,}",
                subtitle="Review Warranted",
            )

        info_panel(
            title="Consensus Interpretation Policy",
            message=d_sum.disagreement_disclaimer,
        )

        if d_sum.top_disagreements:
            st.markdown("##### Frequent Disagreement Patterns:")
            st.dataframe(
                pd.DataFrame(d_sum.top_disagreements).rename(columns={"pattern": "Model Predictions Pattern", "count": "Occurrences", "percentage": "Frequency (%)"}),
                use_container_width=True,
                hide_index=True,
            )

    # 4. Paginated Predictions Table
    st.markdown("---")
    st.markdown("#### Classified Records Explorer")

    # Table filters
    filter_col1, filter_col2 = st.columns(2)
    filtered_df = predictions_df.copy()

    with filter_col1:
        if "agreement_status" in filtered_df.columns:
            status_filter = st.selectbox(
                "Filter by Agreement Status:",
                options=["All Records", "Agreement Only", "Disagreement Only"],
                index=0,
                key="sel_filter_agreement",
            )
            if status_filter == "Agreement Only":
                filtered_df = filtered_df[filtered_df["agreement_status"] == "Agreement"]
            elif status_filter == "Disagreement Only":
                filtered_df = filtered_df[filtered_df["agreement_status"] == "Disagreement"]

    with filter_col2:
        class_col = f"{manifest.selected_algorithms[0]}_prediction" if f"{manifest.selected_algorithms[0]}_prediction" in filtered_df.columns else "predicted_class"
        if class_col in filtered_df.columns:
            unique_classes = ["All Classes"] + sorted(filtered_df[class_col].dropna().unique().tolist())
            cls_filter = st.selectbox("Filter by Primary Prediction:", options=unique_classes, index=0, key="sel_filter_class")
            if cls_filter != "All Classes":
                filtered_df = filtered_df[filtered_df[class_col] == cls_filter]

    # Pagination
    total_records = len(filtered_df)
    page_size = 25
    total_pages = max(1, (total_records + page_size - 1) // page_size)
    curr_page = st.number_input(f"Page (1 to {total_pages}, {total_records:,} matches):", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (curr_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_records)
    page_slice = filtered_df.iloc[start_idx:end_idx]

    st.dataframe(page_slice, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # SECTION G: Export and Run History
    # --------------------------------------------------------------------------
    st.markdown("---")
    section_header(
        title="Artifact Export & Provenance",
        description="Download auditable forensic CSV records and cryptographic run manifests:",
    )

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    pred_csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
    with exp_col1:
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=pred_csv_bytes,
            file_name=f"{detection_run_id}_predictions.csv",
            mime="text/csv",
            key="dl_pred_csv",
        )

    comparison_df = det_registry.load_comparison_df(detection_run_id)
    if comparison_df is not None:
        comp_csv_bytes = comparison_df.to_csv(index=False).encode("utf-8")
        with exp_col2:
            st.download_button(
                label="⬇️ Download Comparison CSV",
                data=comp_csv_bytes,
                file_name=f"{detection_run_id}_model_comparison.csv",
                mime="text/csv",
                key="dl_comp_csv",
            )

    manifest_json_bytes = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
    with exp_col3:
        st.download_button(
            label="⬇️ Download Manifest JSON",
            data=manifest_json_bytes,
            file_name=f"{detection_run_id}_manifest.json",
            mime="application/json",
            key="dl_manifest_json",
        )


def _render_detection_history(
    det_registry: DetectionRegistry,
    detection_runs: List[Dict[str, Any]],
) -> None:
    """Render the historical intrusion detection run catalog."""
    section_header(
        title="Historical Intrusion Detection Runs",
        description="Catalog of verified intrusion detection analyses committed to disk in data/analyses/:",
    )

    if not detection_runs:
        info_panel(
            title="No Prior Detection Runs Found",
            message="No intrusion detection analyses have been executed yet. Run an analysis in the Live Detection Console tab to populate this history.",
        )
        return

    table_data = []
    for r in detection_runs:
        table_data.append({
            "Run ID": r["detection_run_id"],
            "Timestamp (UTC)": r["created_at_utc"][:19].replace("T", " "),
            "Dataset": r["source_filename"],
            "Models": ", ".join(m.upper() for m in r.get("selected_algorithms", [])),
            "Flows Classified": f"{r.get('predicted_rows', 0):,}",
            "Ground Truth": "Present" if r.get("has_ground_truth") else "Absent",
            "Status": r.get("status", "Completed"),
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Allow selecting a historical run to view
    run_ids = [r["detection_run_id"] for r in detection_runs]
    sel_inspect = st.selectbox("Select Run ID to Inspect:", options=run_ids, key="sel_inspect_run")
    if st.button("🔍 Load Run into Console", key="btn_load_history"):
        st.session_state["active_detection_run_id"] = sel_inspect
        st.success(f"Loaded run '{sel_inspect}'. Switch to the '⚡ Live Detection Console' tab to explore results.")
