"""Prediction Explorer & Analyst Review Page (CP-09).

Interactive console providing:
  - Discovery and exploration of completed intrusion detection runs
  - Bounded multi-attribute search, filtering, and pagination
  - Forensic inspection of individual network traffic flows
  - Cross-model consensus and disagreement analysis without pseudo-ensembling
  - Human-in-the-loop analyst review workflow with persistent cryptographic audit trails
"""

import html
import math
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.badges import badge_html
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import error_panel, info_panel, success_panel, warning_panel
from src.explorer.disagreement_service import DisagreementService
from src.explorer.explorer_registry import EligibleRunRecord, ExplorerRegistry
from src.explorer.prediction_query import PredictionFilterParams, PredictionQueryEngine
from src.explorer.prediction_reader import PredictionReader
from src.explorer.record_inspector import RecordInspector
from src.explorer.review_models import AnalystDisposition, ReviewStatus
from src.explorer.review_service import ReviewService
from src.explorer.review_store import ReviewStore


def render() -> None:
    """Render the Prediction Explorer & Analyst Review interface."""
    page_header(
        title="Prediction Explorer & Analyst Review",
        subtitle="Forensic traffic classification analysis, model disagreement exploration, and human-in-the-loop triage",
        icon="🔍",
        badges=[("Checkpoint CP-09", "info"), ("Analyst Review Subsystem", "success")],
    )

    registry = ExplorerRegistry()
    eligible_runs = registry.list_eligible_runs()

    # --------------------------------------------------------------------------
    # Genuine Empty State: No Completed Detection Runs
    # --------------------------------------------------------------------------
    if not eligible_runs:
        info_panel(
            title="No Completed Production Detection Runs Available",
            message=(
                "No verified production detection runs were discovered in `data/analyses/`. "
                "To explore predictions and perform analyst reviews:\n\n"
                "1. **Traffic Upload:** Ingest a network flow CSV via the **Traffic Analysis** console.\n"
                "2. **Dataset Engineering:** Preprocess data via **Dataset Engineering (CP-05)**.\n"
                "3. **Model Laboratory:** Train and evaluate the three authorized models (SVM, MLP, Decision Tree).\n"
                "4. **Detection Center:** Execute an intrusion detection run on ingested traffic."
            ),
        )
        return

    # --------------------------------------------------------------------------
    # SECTION A: Detection Run Selector
    # --------------------------------------------------------------------------
    section_header(
        title="Detection Run Selection",
        description="Select a completed intrusion detection inference run to explore:",
    )

    run_labels = {r.display_label: r for r in eligible_runs}
    run_options = list(run_labels.keys())

    # Session state persistence for run selection
    curr_run_id = st.session_state.get("selected_explorer_run_id")
    default_idx = 0
    if curr_run_id:
        for idx, r in enumerate(eligible_runs):
            if r.detection_run_id == curr_run_id:
                default_idx = idx
                break

    selected_label = st.selectbox(
        "Select Detection Run:",
        options=run_options,
        index=default_idx,
        key="sel_explorer_run_label",
    )

    selected_run = run_labels[selected_label]
    st.session_state["selected_explorer_run_id"] = selected_run.detection_run_id
    run_id = selected_run.detection_run_id

    # Display Run Metadata Card
    with st.expander("Detection Run Provenance & Verification Details", expanded=False):
        prov_col1, prov_col2, prov_col3 = st.columns(3)
        with prov_col1:
            st.markdown(f"**Run ID:** `{selected_run.detection_run_id}`")
            st.markdown(f"**Timestamp (UTC):** `{selected_run.created_at_utc}`")
            st.markdown(f"**Integrity Status:** `SHA-256 Verified`")
        with prov_col2:
            st.markdown(f"**Source File:** `{selected_run.source_filename}`")
            st.markdown(f"**Source Dataset ID:** `{selected_run.source_dataset_id}`")
            st.markdown(f"**Source SHA-256:** `{selected_run.source_sha256[:16]}...`")
        with prov_col3:
            st.markdown(f"**Evaluated Models:** `{', '.join(selected_run.selected_algorithms)}`")
            st.markdown(f"**Classified Rows:** `{selected_run.predicted_rows:,}` / `{selected_run.input_rows:,}`")
            st.markdown(f"**Ground Truth:** `{'Present' if selected_run.has_ground_truth else 'None'}`")

        if selected_run.warnings:
            for w in selected_run.warnings:
                st.warning(f"Run Warning: {w}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Load Run Artifacts & live Reviews
    # --------------------------------------------------------------------------
    reader = PredictionReader()
    review_store = ReviewStore()
    review_service = ReviewService(review_store=review_store)

    try:
        preds_df = reader.load_predictions(run_id, attach_reviews=True)
        manifest = reader.load_manifest(run_id)
    except Exception as e:
        warning_panel(
            title="Artifact Loading Error",
            message=f"Failed to load detection artifacts for run '{run_id}': {e}",
        )
        return

    if preds_df.empty:
        warning_panel(title="Empty Artifact", message="The selected detection run contains 0 prediction records.")
        return

    # Compute live review summary
    total_records = len(preds_df)
    rev_summary = review_service.get_summary(run_id, total_eligible_records=total_records)
    disagreement_data = DisagreementService.analyze_consensus(preds_df, selected_run.selected_algorithms)

    # --------------------------------------------------------------------------
    # SECTION B: Prediction & Review Overview
    # --------------------------------------------------------------------------
    section_header(
        title="Prediction Telemetry & Review Progress",
        description=f"Overview of classification decisions and human analyst triage for run `{run_id}`:",
    )

    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)

    with met_col1:
        metric_card(
            label="Total Flows",
            value=f"{total_records:,}",
            subtitle="Analyzed Records",
        )

    with met_col2:
        num_models = len(selected_run.selected_algorithms)
        metric_card(
            label="Models Evaluated",
            value=f"{num_models}",
            subtitle="Authorized Estimators",
        )

    with met_col3:
        if num_models > 1:
            agree_pct = disagreement_data.get("agreement_pct", 0.0)
            metric_card(
                label="Model Agreement",
                value=f"{agree_pct:.1f}%",
                subtitle=f"{disagreement_data.get('agreement_count', 0):,} Rows Agreed",
                status_color="#22C55E" if agree_pct >= 90 else "#EAB308",
            )
        else:
            metric_card(
                label="Agreement State",
                value="N/A",
                subtitle="Single Model Inference",
            )

    with met_col4:
        if num_models > 1:
            disagree_pct = disagreement_data.get("disagreement_pct", 0.0)
            metric_card(
                label="Model Disagreement",
                value=f"{disagree_pct:.1f}%",
                subtitle=f"{disagreement_data.get('disagreement_count', 0):,} Rows Divergent",
                status_color="#EF4444" if disagree_pct > 0 else "#94A3B8",
            )
        else:
            metric_card(
                label="Disagreement",
                value="N/A",
                subtitle="Single Model Inference",
            )

    with met_col5:
        metric_card(
            label="Analyst Reviewed",
            value=f"{rev_summary.completion_percentage:.1f}%",
            subtitle=f"{rev_summary.reviewed_records:,} / {total_records:,} Flows",
            status_color="#3B82F6" if rev_summary.completion_percentage > 0 else "#94A3B8",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION C: Prediction & Review Visualizations
    # --------------------------------------------------------------------------
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.markdown("##### Predicted Class Distribution")
        # Extract class counts from predicted_class or primary model
        target_pred_col = "predicted_class"
        if target_pred_col in preds_df.columns:
            cls_counts = preds_df[target_pred_col].value_counts().reset_index()
            cls_counts.columns = ["Class", "Count"]
            fig_cls = px.bar(
                cls_counts,
                x="Class",
                y="Count",
                color="Class",
                title="Predicted Traffic Classes",
            )
            fig_cls.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0E0"},
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    with chart_col2:
        st.markdown("##### Model Consensus Breakdown")
        if num_models > 1:
            consensus_counts = pd.DataFrame([
                {"State": "Agreement", "Count": disagreement_data.get("agreement_count", 0)},
                {"State": "Disagreement", "Count": disagreement_data.get("disagreement_count", 0)},
                {"State": "Incomplete", "Count": disagreement_data.get("incomplete_count", 0)},
            ])
            # Filter zero counts
            consensus_counts = consensus_counts[consensus_counts["Count"] > 0]
            fig_con = px.pie(
                consensus_counts,
                names="State",
                values="Count",
                color="State",
                color_discrete_map={
                    "Agreement": "#22C55E",
                    "Disagreement": "#EF4444",
                    "Incomplete": "#F59E0B",
                },
                hole=0.45,
                title="Consensus vs Divergence",
            )
            fig_con.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0E0"},
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_con, use_container_width=True)
        else:
            info_panel(
                title="Single Model Run",
                message="Consensus visualization requires comparative evaluation across 2 or more models.",
            )

    with chart_col3:
        st.markdown("##### Analyst Dispositions")
        disp_df_rows = []
        for disp, count in rev_summary.disposition_counts.items():
            if count > 0:
                disp_df_rows.append({"Disposition": disp, "Count": count})
        if rev_summary.unreviewed_records > 0:
            disp_df_rows.append({"Disposition": "Unreviewed", "Count": rev_summary.unreviewed_records})

        disp_df = pd.DataFrame(disp_df_rows)
        if not disp_df.empty:
            color_map = {
                AnalystDisposition.POTENTIALLY_MALICIOUS: "#EF4444",
                AnalystDisposition.LIKELY_BENIGN: "#22C55E",
                AnalystDisposition.FALSE_POSITIVE: "#F59E0B",
                AnalystDisposition.NEEDS_FURTHER_INVESTIGATION: "#3B82F6",
                AnalystDisposition.INCONCLUSIVE: "#8B5CF6",
                "Unreviewed": "#475569",
            }
            fig_disp = px.pie(
                disp_df,
                names="Disposition",
                values="Count",
                color="Disposition",
                color_discrete_map=color_map,
                hole=0.45,
                title="Human Analyst Assessments",
            )
            fig_disp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0E0"},
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_disp, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION D: Filtering, Search & Prediction Records Table
    # --------------------------------------------------------------------------
    section_header(
        title="Prediction Records Explorer",
        description="Filter, search, and sort classified traffic flow records:",
    )

    with st.expander("Filter & Search Controls", expanded=True):
        f_row1_col1, f_row1_col2, f_row1_col3, f_row1_col4 = st.columns(4)

        # 1. Class Filter
        all_pred_classes = sorted(list(preds_df["predicted_class"].dropna().unique()))
        with f_row1_col1:
            sel_classes = st.multiselect(
                "Predicted Class:",
                options=all_pred_classes,
                default=[],
                key=f"flt_classes_{run_id}",
            )

        # 2. Model Filter
        with f_row1_col2:
            model_opts = ["All Models"] + selected_run.selected_algorithms
            sel_model_choice = st.selectbox(
                "Filter by Model:",
                options=model_opts,
                key=f"flt_model_{run_id}",
            )
            sel_model = None if sel_model_choice == "All Models" else sel_model_choice

        # 3. Agreement Filter
        with f_row1_col3:
            agree_opts = ["All States"] + list(preds_df["agreement_status"].dropna().unique())
            sel_agree_choice = st.selectbox(
                "Agreement State:",
                options=agree_opts,
                key=f"flt_agree_{run_id}",
            )
            sel_agree = None if sel_agree_choice == "All States" else [sel_agree_choice]

        # 4. Review Status Filter
        with f_row1_col4:
            rev_opts = ["All Records", ReviewStatus.UNREVIEWED, ReviewStatus.REVIEWED]
            sel_rev_choice = st.selectbox(
                "Review Status:",
                options=rev_opts,
                key=f"flt_rev_{run_id}",
            )
            sel_rev = None if sel_rev_choice == "All Records" else [sel_rev_choice]

        f_row2_col1, f_row2_col2, f_row2_col3, f_row2_col4 = st.columns(4)

        # 5. Analyst Disposition Filter
        with f_row2_col1:
            disp_opts = ["All Dispositions"] + list(AnalystDisposition.ALL_DISPOSITIONS) + ["Unreviewed"]
            sel_disp_choice = st.selectbox(
                "Analyst Disposition:",
                options=disp_opts,
                key=f"flt_disp_{run_id}",
            )
            sel_disp = None if sel_disp_choice == "All Dispositions" else [sel_disp_choice]

        # 6. Ground Truth Filter
        with f_row2_col2:
            gt_opts = {
                "All Flows": "all",
                "Has Ground Truth": "has_ground_truth",
                "Prediction Agrees with Ground Truth": "matches_prediction",
                "Prediction Differs from Ground Truth": "differs_from_prediction",
            }
            sel_gt_choice = st.selectbox(
                "Ground Truth Comparison:",
                options=list(gt_opts.keys()),
                key=f"flt_gt_{run_id}",
            )
            sel_gt = gt_opts[sel_gt_choice]

        # 7. Substring Search
        with f_row2_col3:
            search_query = st.text_input(
                "Search Record / Notes:",
                placeholder="Search row ID, attack name, notes...",
                key=f"flt_search_{run_id}",
            )

        # 8. Triage & Sorting
        with f_row2_col4:
            sort_cols = [c for c in ["source_row_index", "predicted_class", "agreement_status", "review_status"] if c in preds_df.columns]
            sort_by = st.selectbox("Sort By:", options=sort_cols, index=0, key=f"flt_sort_{run_id}")
            sort_asc = st.checkbox("Ascending", value=True, key=f"flt_sort_asc_{run_id}")
            disagree_first = st.checkbox("Disagreements First (Triage)", value=False, key=f"flt_disagree_first_{run_id}")

    # Build query params
    params = PredictionFilterParams(
        predicted_classes=sel_classes if sel_classes else None,
        selected_model=sel_model,
        agreement_statuses=sel_agree,
        review_statuses=sel_rev,
        analyst_dispositions=sel_disp,
        ground_truth_filter=sel_gt,
        search_query=search_query if search_query.strip() else None,
        sort_by=sort_by,
        sort_ascending=sort_asc,
        disagreement_first=disagree_first,
    )

    # Pagination controls
    page_size_col, page_nav_col = st.columns([1, 3])
    with page_size_col:
        page_size = st.selectbox("Page Size:", options=[10, 25, 50, 100], index=1, key=f"sel_page_size_{run_id}")

    curr_page = st.session_state.get(f"explorer_page_{run_id}", 1)

    # Execute Query
    page_df, total_matching = PredictionQueryEngine.query(
        df=preds_df,
        params=params,
        page=curr_page,
        page_size=page_size,
    )

    total_pages = max(1, math.ceil(total_matching / page_size))
    if curr_page > total_pages:
        curr_page = total_pages
        st.session_state[f"explorer_page_{run_id}"] = curr_page
        page_df, total_matching = PredictionQueryEngine.query(
            df=preds_df,
            params=params,
            page=curr_page,
            page_size=page_size,
        )

    with page_nav_col:
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns([1, 1, 2, 1, 1])
        with p_c1:
            if st.button("⏪ First", disabled=(curr_page <= 1), key=f"btn_p_first_{run_id}"):
                st.session_state[f"explorer_page_{run_id}"] = 1
                st.rerun()
        with p_c2:
            if st.button("◀ Prev", disabled=(curr_page <= 1), key=f"btn_p_prev_{run_id}"):
                st.session_state[f"explorer_page_{run_id}"] = max(1, curr_page - 1)
                st.rerun()
        with p_c3:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; color: #94A3B8;'>Page <b>{curr_page}</b> of <b>{total_pages}</b> ({total_matching:,} flows)</div>", unsafe_allow_html=True)
        with p_c4:
            if st.button("Next ▶", disabled=(curr_page >= total_pages), key=f"btn_p_next_{run_id}"):
                st.session_state[f"explorer_page_{run_id}"] = min(total_pages, curr_page + 1)
                st.rerun()
        with p_c5:
            if st.button("Last ⏩", disabled=(curr_page >= total_pages), key=f"btn_p_last_{run_id}"):
                st.session_state[f"explorer_page_{run_id}"] = total_pages
                st.rerun()

    if page_df.empty:
        warning_panel(
            title="No Matching Records",
            message="No network flows match the selected filter criteria. Try adjusting or clearing your filters.",
        )
        return

    # Display Table
    table_display_cols = ["source_row_index"]
    # Model prediction columns
    for a in selected_run.selected_algorithms:
        if f"{a}_prediction" in page_df.columns:
            table_display_cols.append(f"{a}_prediction")
    if "predicted_class" in page_df.columns and "predicted_class" not in table_display_cols:
        table_display_cols.append("predicted_class")
    if "agreement_status" in page_df.columns and num_models > 1:
        table_display_cols.append("agreement_status")
    if "ground_truth_label" in page_df.columns and selected_run.has_ground_truth:
        table_display_cols.append("ground_truth_label")
    table_display_cols.extend(["review_status", "analyst_disposition"])

    clean_table_df = page_df[[c for c in table_display_cols if c in page_df.columns]].copy()
    st.dataframe(clean_table_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION E: Individual Record Inspector & Forensic Details
    # --------------------------------------------------------------------------
    section_header(
        title="Forensic Record Inspector & Analyst Review",
        description="Select an individual network traffic record to inspect attributes, examine predictions, and record human review:",
    )

    page_row_indices = page_df["source_row_index"].tolist()
    default_row = page_row_indices[0] if page_row_indices else 0

    inspect_col1, inspect_col2 = st.columns([1, 3])
    with inspect_col1:
        selected_row_idx = st.selectbox(
            "Select Flow to Inspect (Row Index):",
            options=page_row_indices,
            index=0,
            key=f"sel_inspect_row_{run_id}",
        )

    inspector = RecordInspector(prediction_reader=reader, review_store=review_store)
    try:
        inspection = inspector.inspect_record(run_id, selected_row_idx)
    except Exception as e:
        warning_panel(title="Inspection Error", message=f"Failed to inspect row {selected_row_idx}: {e}")
        return

    # Tabs for Inspection and Review
    insp_tab1, insp_tab2, insp_tab3, insp_tab4 = st.tabs([
        "📊 Network Attributes",
        "🤖 Model Predictions & Scores",
        "🎯 Dataset Ground Truth",
        "📝 Analyst Review & History",
    ])

    # Tab 1: Network Attributes
    with insp_tab1:
        st.markdown(f"##### Flow Attributes — Row `{inspection.source_row_index:,}`")
        st.caption(inspection.network_attributes_note)

        if inspection.network_attributes_available:
            attrs = inspection.network_attributes
            # Render attributes in structured grid
            attr_cols = st.columns(4)
            items = list(attrs.items())
            chunk_size = math.ceil(len(items) / 4)
            for col_idx in range(4):
                with attr_cols[col_idx]:
                    sub_items = items[col_idx * chunk_size : (col_idx + 1) * chunk_size]
                    for k, v in sub_items:
                        st.markdown(f"**{html.escape(str(k))}:** `{html.escape(str(v))}`")
        else:
            info_panel(
                title="Traffic Attributes Unavailable",
                message=inspection.network_attributes_note,
            )

    # Tab 2: Model Predictions & Score Semantics
    with insp_tab2:
        st.markdown("##### Model Decision Transparency")
        if inspection.model_predictions:
            pred_cols = st.columns(len(inspection.model_predictions))
            for col_idx, (algo_key, pred_detail) in enumerate(inspection.model_predictions.items()):
                with pred_cols[col_idx]:
                    st.markdown(f"#### {pred_detail.algorithm_display_name}")
                    st.markdown(f"**Predicted Class:** `{pred_detail.predicted_class}`")
                    score_str = f"{pred_detail.raw_score:.4f}" if pred_detail.raw_score is not None else "N/A"
                    st.markdown(f"**Score:** `{score_str}` ({pred_detail.score_type})")
                    st.caption(f"ℹ️ {pred_detail.score_explanation}")
        else:
            st.info("No per-model predictions available.")

        if inspection.agreement_status != "Single Model":
            st.markdown("---")
            st.markdown(f"**Consensus Status:** `{inspection.agreement_status}`")
            if inspection.differing_models and inspection.differing_models != "None":
                st.markdown(f"**Divergent Models:** `{inspection.differing_models}`")

    # Tab 3: Dataset Ground Truth
    with insp_tab3:
        st.markdown("##### Dataset Ground Truth Reference")
        if inspection.has_ground_truth and inspection.ground_truth_label:
            gt_val = inspection.ground_truth_label
            st.markdown(f"**Dataset Ground-Truth Label:** `{gt_val}`")

            # Check matching against models
            matches = []
            for algo, p in inspection.model_predictions.items():
                is_match = (p.predicted_class == gt_val)
                badge_v = "success" if is_match else "error"
                matches.append(f"{p.algorithm_display_name}: {'Match' if is_match else 'Mismatch'}")

            for m in matches:
                st.markdown(f"- {m}")

            info_panel(
                title="Academic Benchmark Reference Only",
                message=(
                    "The ground-truth label originates from academic benchmark labels in the raw dataset. "
                    "It is provided strictly for validation and auditing. It is NOT an analyst-confirmed incident."
                ),
            )
        else:
            st.info("No ground-truth label was present in the source dataset for this record.")

    # Tab 4: Analyst Review & History
    with insp_tab4:
        st.markdown(f"##### Human Analyst Review — Flow Row `{inspection.source_row_index:,}`")

        # Current Status
        curr_disp = inspection.latest_review.new_disposition if inspection.latest_review else "Unreviewed"
        curr_stat = ReviewStatus.REVIEWED if inspection.latest_review else ReviewStatus.UNREVIEWED

        disp_col1, disp_col2 = st.columns(2)
        with disp_col1:
            st.markdown(f"**Current Status:** `{curr_stat}`")
        with disp_col2:
            st.markdown(f"**Current Disposition:** `{curr_disp}`")

        st.markdown("---")
        st.markdown("###### Submit / Update Analyst Assessment")

        with st.form(key=f"form_review_{run_id}_{selected_row_idx}"):
            # Authorized dispositions
            disp_index = 0
            if inspection.latest_review:
                try:
                    disp_index = AnalystDisposition.ALL_DISPOSITIONS.index(inspection.latest_review.new_disposition)
                except ValueError:
                    disp_index = 0

            new_disposition = st.radio(
                "Analyst Disposition Assessment:",
                options=list(AnalystDisposition.ALL_DISPOSITIONS),
                index=disp_index,
                help="Select your forensic evaluation for this network flow.",
            )

            prev_notes = inspection.latest_review.analyst_notes if inspection.latest_review else ""
            notes_text = st.text_area(
                "Analyst Notes (Optional, max 2000 chars):",
                value=prev_notes or "",
                max_chars=2000,
                placeholder="Enter forensic observations, triage reasoning, or investigation details...",
            )

            prev_label = inspection.latest_review.reviewer_label if inspection.latest_review else ""
            reviewer_label = st.text_input(
                "Reviewer Display Label (Optional):",
                value=prev_label or "",
                placeholder="e.g. Analyst J. Doe / Tier-2 SOC",
                help="Informational display label; does not represent verified cryptographic identity.",
            )

            submit_clicked = st.form_submit_button("💾 Save Analyst Review", use_container_width=True)

            if submit_clicked:
                try:
                    event = review_service.submit_review(
                        detection_run_id=run_id,
                        source_dataset_id=selected_run.source_dataset_id,
                        source_row_index=int(selected_row_idx),
                        disposition=new_disposition,
                        analyst_notes=notes_text,
                        reviewer_label=reviewer_label,
                    )
                    st.success(f"Analyst review recorded successfully! Event ID: `{event.event_id}`")
                    # Clear reader cache to reflect changes immediately
                    reader.clear_cache(run_id)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to record review: {e}")

        # Review Audit History Timeline
        st.markdown("---")
        st.markdown("###### Chronological Audit Trail")
        history = inspection.review_history
        if history:
            history_rows = []
            for ev in history:
                history_rows.append({
                    "Timestamp (UTC)": ev.timestamp_utc,
                    "Event ID": ev.event_id,
                    "Disposition": ev.new_disposition,
                    "Previous Disposition": ev.previous_disposition or "None",
                    "Reviewer": ev.reviewer_label or "Anonymous",
                    "Notes": ev.analyst_notes or "",
                    "Event Hash": f"{ev.event_hash[:12]}...",
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

            # Audit integrity check
            is_valid, integrity_errors = review_service.verify_integrity(run_id)
            if is_valid:
                st.caption("🔒 Review journal hash chain verified: No tampering or deletions detected.")
            else:
                st.error(f"⚠️ Audit Integrity Warning: {'; '.join(integrity_errors)}")
        else:
            st.caption("No previous review events recorded for this flow. Record is currently Unreviewed.")

    # --------------------------------------------------------------------------
    # SECTION F: Model Disagreement Analytics (when multiple models)
    # --------------------------------------------------------------------------
    if num_models > 1 and disagreement_data.get("disagreement_count", 0) > 0:
        st.markdown("---")
        section_header(
            title="Model Disagreement Explorer",
            description="Detailed analysis of classification divergence patterns across the three authorized algorithms:",
        )

        d_col1, d_col2 = st.columns([1, 1])

        with d_col1:
            st.markdown("##### Pairwise Model Agreement Matrix")
            pairwise = disagreement_data.get("pairwise_agreements", {})
            if pairwise:
                pair_rows = [{"Model Comparison": k, "Agreement %": f"{v:.1f}%"} for k, v in pairwise.items()]
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

        with d_col2:
            st.markdown("##### Top Divergence Patterns")
            top_patterns = disagreement_data.get("top_divergence_patterns", [])
            if top_patterns:
                pat_rows = [
                    {"Divergence Pattern": p["pattern"], "Flows": f"{p['count']:,}", "% Total": f"{p['percentage']:.2f}%"}
                    for p in top_patterns[:5]
                ]
                st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

        info_panel(
            title="Forensic Triage Guidance",
            message=disagreement_data.get("disclaimer", DisagreementService.DISAGREEMENT_DISCLAIMER),
        )
