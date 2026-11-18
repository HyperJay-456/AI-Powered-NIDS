"""Threat Intelligence & Attack Context Page (CP-10).

Integrates official MITRE ATT&CK Enterprise v20.1 reference knowledge with
AI-Powered NIDS classifications:
  - Official MITRE ATT&CK Attribution Notice & Trademark Disclaimer
  - Dataset attack label taxonomy and threat family categorizations
  - Conservative label-to-technique candidate mappings and human review workflows
  - Evidence-aware contextual enrichment for completed detection runs
  - Deterministic security insights and local STIX reference bundle management
"""

import html
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings
from dashboard.components.badges import badge_html
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import error_panel, info_panel, success_panel, warning_panel
from src.explorer.explorer_registry import ExplorerRegistry
from src.explorer.prediction_reader import PredictionReader
from src.threat_intelligence.attack_models import (
    AttackFamily,
    MappingStatus,
    MappingStrength,
    MITRE_ATTRIBUTION_NOTICE,
    MITRE_DISCLAIMER,
)
from src.threat_intelligence.context_service import ThreatContextService
from src.threat_intelligence.insight_engine import ThreatInsightEngine
from src.threat_intelligence.mapping_registry import MappingRegistry
from src.threat_intelligence.reference_registry import ReferenceRegistry
from src.threat_intelligence.taxonomy_service import TaxonomyService


def render() -> None:
    """Render the Threat Intelligence & Attack Context console."""
    page_header(
        title="Threat Intelligence & Attack Context",
        subtitle="MITRE ATT&CK Enterprise v20.1 knowledge integration, candidate mapping review, and evidence-aware context",
        icon="🌐",
        badges=[("Checkpoint CP-10", "info"), ("ATT&CK v20.1 Enterprise", "success")],
    )

    # --------------------------------------------------------------------------
    # Official MITRE Attribution & Disclaimer
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
<div style="background-color: #1E293B; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
    <div style="font-size: 0.875rem; color: #94A3B8;">
        <b>Attribution Notice:</b> {html.escape(MITRE_ATTRIBUTION_NOTICE)}<br>
        <span style="font-size: 0.8125rem; color: #64748B;">{html.escape(MITRE_DISCLAIMER)}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    ref_registry = ReferenceRegistry()
    mapping_registry = MappingRegistry()
    context_service = ThreatContextService(
        reference_registry=ref_registry,
        mapping_registry=mapping_registry,
    )
    insight_engine = ThreatInsightEngine(mapping_registry=mapping_registry)

    catalog = ref_registry.get_active_catalog()
    ref_metadata = ref_registry.get_metadata()

    # --------------------------------------------------------------------------
    # SECTION A: Threat Intelligence Overview
    # --------------------------------------------------------------------------
    section_header(
        title="Threat Reference Telemetry",
        description="Local MITRE ATT&CK Enterprise knowledge base status and mapping coverage:",
    )

    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)

    coverage_info = insight_engine.compute_mapping_coverage()

    with met_col1:
        ver_text = ref_metadata.version if ref_metadata else "Not Installed"
        metric_card(
            label="ATT&CK Version",
            value=ver_text,
            subtitle="Enterprise Knowledge Base",
            status_color="#22C55E" if ref_metadata else "#EF4444",
        )

    with met_col2:
        tech_cnt = len(catalog.techniques) if catalog else 0
        metric_card(
            label="ATT&CK Techniques",
            value=f"{tech_cnt:,}",
            subtitle=f"{len(catalog.tactics) if catalog else 0} Tactics",
        )

    with met_col3:
        metric_card(
            label="Dataset Attack Classes",
            value=f"{coverage_info['total_classes']}",
            subtitle="CIC-IDS-2017 Benchmark",
        )

    with met_col4:
        metric_card(
            label="Mapped Classes",
            value=f"{coverage_info['mapped_classes']}",
            subtitle=f"{coverage_info['coverage_pct']:.0f}% Coverage",
            status_color="#3B82F6",
        )

    with met_col5:
        metric_card(
            label="Approved Mappings",
            value=f"{coverage_info['approved_classes']}",
            subtitle="Human-Reviewed",
            status_color="#22C55E" if coverage_info["approved_classes"] > 0 else "#94A3B8",
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Genuine Empty State: When no ATT&CK STIX bundle is installed
    # --------------------------------------------------------------------------
    if not catalog or not ref_metadata:
        warning_panel(
            title="MITRE ATT&CK Reference Bundle Not Installed",
            message=(
                "No active MITRE ATT&CK Enterprise STIX 2.1 bundle was discovered in `data/knowledge/mitre_attack/`. "
                "To explore threat intelligence and view technique details:\n\n"
                "1. Download the official Enterprise STIX 2.1 JSON bundle (`enterprise-attack-20.1.json`).\n"
                "2. Place or import it into the local reference directory using **Section G: Reference Data Management** below.\n"
                "3. The system operates strictly offline and will never query external servers at runtime."
            ),
        )

    # --------------------------------------------------------------------------
    # Tabs for Primary Functional Subsystems
    # --------------------------------------------------------------------------
    tab_tax, tab_tech, tab_map, tab_ctx, tab_ins, tab_mgm = st.tabs([
        "📚 Attack Taxonomy",
        "🔍 ATT&CK Explorer",
        "🎯 Label-to-Technique Mappings",
        "🔬 Contextual Predictions",
        "📊 Security Insights",
        "⚙️ Reference Management",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Dataset Attack Taxonomy
    # --------------------------------------------------------------------------
    with tab_tax:
        section_header(
            title="Dataset Attack Label Taxonomy",
            description="Normalized classification taxonomy and broad threat families for CIC-IDS-2017:",
        )

        all_tax = TaxonomyService.get_all_taxonomy_items()
        all_families = [AttackFamily.OTHER] + list(AttackFamily.ALL_FAMILIES)

        fam_col, _ = st.columns([2, 3])
        with fam_col:
            selected_family = st.selectbox(
                "Filter by Attack Family:",
                options=["All Families"] + list(AttackFamily.ALL_FAMILIES),
                key="sel_tax_family",
            )

        tax_rows = []
        for item in all_tax:
            if selected_family != "All Families" and item.broad_family != selected_family:
                continue
            tax_rows.append({
                "Dataset Label": item.original_label,
                "Threat Family": item.broad_family,
                "Description": item.description,
                "Benchmark Source": item.dataset_source,
            })

        st.dataframe(pd.DataFrame(tax_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: ATT&CK Explorer
    # --------------------------------------------------------------------------
    with tab_tech:
        section_header(
            title="MITRE ATT&CK Enterprise Technique Explorer",
            description="Browse, search, and inspect tactics, techniques, and defensive mitigations:",
        )

        if catalog:
            t_search_col1, t_search_col2 = st.columns([2, 1])
            with t_search_col1:
                tech_search_q = st.text_input(
                    "Search Techniques:",
                    placeholder="Search technique ID (e.g. T1046), name, or description...",
                    key="txt_tech_search",
                )
            with t_search_col2:
                all_tacs = sorted(list(set(t.name for t in catalog.tactics.values())))
                tactic_filter = st.selectbox(
                    "Filter by Tactic:",
                    options=["All Tactics"] + all_tacs,
                    key="sel_tech_tactic",
                )

            selected_tac_shortname = None
            if tactic_filter != "All Tactics":
                for tac in catalog.tactics.values():
                    if tac.name == tactic_filter:
                        selected_tac_shortname = tac.shortname
                        break

            matching_techs = catalog.search_techniques(
                query=tech_search_q,
                tactic=selected_tac_shortname,
            )

            st.caption(f"Showing {len(matching_techs)} matching technique(s).")

            if matching_techs:
                tech_display_map = {f"{t.external_id} — {t.name}": t for t in matching_techs}
                selected_tech_label = st.selectbox(
                    "Select Technique to Inspect:",
                    options=list(tech_display_map.keys()),
                    key="sel_active_tech",
                )
                chosen_tech = tech_display_map[selected_tech_label]

                with st.expander(f"Technique Detail: {chosen_tech.external_id} — {chosen_tech.name}", expanded=True):
                    col_det1, col_det2 = st.columns([2, 1])
                    with col_det1:
                        st.markdown(f"**Technique ID:** `{chosen_tech.external_id}`")
                        st.markdown(f"**Name:** {chosen_tech.name}")
                        st.markdown(f"**Tactics:** `{', '.join(chosen_tech.tactics)}`")
                        st.markdown(f"**Platforms:** `{', '.join(chosen_tech.platforms)}`")
                        st.markdown(f"**Description:** {chosen_tech.description}")
                        if chosen_tech.detection_guidance:
                            st.markdown(f"**Detection Guidance:** {chosen_tech.detection_guidance}")
                    with col_det2:
                        if chosen_tech.url:
                            st.link_button(
                                "🌐 View on MITRE ATT&CK",
                                url=chosen_tech.url,
                                use_container_width=True,
                            )

                        # Associated Mitigations
                        associated_mits = catalog.get_mitigations_for_technique(chosen_tech.external_id)
                        st.markdown("##### Relevant Mitigations")
                        if associated_mits:
                            for mit in associated_mits:
                                st.markdown(f"- **{mit.external_id}** — {mit.name}")
                                st.caption(mit.description[:150] + "...")
                        else:
                            st.caption("No specific mitigations linked in reference data.")
        else:
            info_panel(
                title="ATT&CK Catalog Empty",
                message="Install or import the official STIX 2.1 bundle to explore techniques.",
            )

    # --------------------------------------------------------------------------
    # TAB 3: Label-to-Technique Mapping Registry
    # --------------------------------------------------------------------------
    with tab_map:
        section_header(
            title="Dataset Label to ATT&CK Technique Mapping Registry",
            description="Conservative candidate mappings between dataset classifications and ATT&CK techniques with human approval actions:",
        )

        all_mappings = mapping_registry.list_mappings()

        filter_status = st.selectbox(
            "Filter Mappings by Status:",
            options=["All Statuses"] + list(MappingStatus.ALL_STATUSES),
            key="sel_map_filter_status",
        )

        filtered_mappings = [
            m for m in all_mappings
            if filter_status == "All Statuses" or m.mapping_status == filter_status
        ]

        # Table Display
        map_table_rows = []
        for m in filtered_mappings:
            map_table_rows.append({
                "Dataset Label": m.original_dataset_label,
                "Technique ID": m.technique_id or "Unmapped",
                "Technique Name": m.technique_name,
                "Tactic": m.tactic or "None",
                "Status": m.mapping_status,
                "Alignment Strength": m.mapping_strength,
                "Rationale": m.mapping_rationale[:80] + ("..." if len(m.mapping_rationale) > 80 else ""),
            })

        st.dataframe(pd.DataFrame(map_table_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### Human Mapping Review & Approval Console")

        mapping_choices = {f"{m.original_dataset_label} -> {m.technique_id or 'Unmapped'} ({m.mapping_status})": m for m in all_mappings}
        sel_map_label = st.selectbox(
            "Select Mapping to Review / Modify:",
            options=list(mapping_choices.keys()),
            key="sel_mapping_to_review",
        )
        target_map = mapping_choices[sel_map_label]

        with st.expander(f"Review Mapping: {target_map.original_dataset_label}", expanded=True):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown(f"**Dataset Label:** `{target_map.original_dataset_label}`")
                st.markdown(f"**Mapped Technique:** `{target_map.technique_id}` ({target_map.technique_name})")
                st.markdown(f"**Tactic:** `{target_map.tactic}`")
                st.markdown(f"**Current Status:** `{target_map.mapping_status}`")
                st.markdown(f"**Alignment Strength:** `{target_map.mapping_strength}`")
                st.markdown(f"**Rationale:** {target_map.mapping_rationale}")
            with r_col2:
                st.markdown("###### Required Observable Evidence")
                for req in target_map.required_evidence:
                    st.markdown(f"- {req}")
                st.markdown("###### Missing Telemetry from NetFlow")
                for mis in target_map.missing_evidence:
                    st.markdown(f"- ⚠️ {mis}")

            st.markdown("---")
            # Human Review Actions
            act_col1, act_col2 = st.columns([2, 1])
            with act_col1:
                rev_notes = st.text_area(
                    "Analyst Review Notes (max 2000 chars):",
                    value=target_map.analyst_notes or "",
                    placeholder="Document technical justification, network evidence, or reason for status change...",
                    key=f"txt_map_notes_{target_map.mapping_id}",
                )
                rev_label = st.text_input(
                    "Reviewer Display Label (Optional):",
                    value=target_map.reviewer_label or "",
                    placeholder="e.g. Threat Intelligence Lead / SecOps",
                    key=f"txt_map_label_{target_map.mapping_id}",
                )

            with act_col2:
                st.markdown("###### Action Controls")
                if st.button("✅ Approve Mapping", key=f"btn_app_{target_map.mapping_id}", use_container_width=True):
                    mapping_registry.approve_mapping(
                        mapping_id=target_map.mapping_id,
                        reviewer_label=rev_label,
                        analyst_notes=rev_notes,
                    )
                    st.success(f"Mapping '{target_map.mapping_id}' APPROVED successfully!")
                    st.rerun()

                if st.button("🛑 Reject Mapping", key=f"btn_rej_{target_map.mapping_id}", use_container_width=True):
                    mapping_registry.reject_mapping(
                        mapping_id=target_map.mapping_id,
                        reviewer_label=rev_label,
                        analyst_notes=rev_notes,
                    )
                    st.warning(f"Mapping '{target_map.mapping_id}' REJECTED.")
                    st.rerun()

                if st.button("⚪ Mark Unmapped", key=f"btn_unmap_{target_map.mapping_id}", use_container_width=True):
                    mapping_registry.mark_unmapped(
                        mapping_id=target_map.mapping_id,
                        reviewer_label=rev_label,
                        analyst_notes=rev_notes,
                    )
                    st.info(f"Mapping '{target_map.mapping_id}' marked as UNMAPPED.")
                    st.rerun()

            # Mapping Revision History
            history_events = mapping_registry.get_history(target_map.mapping_id)
            if history_events:
                st.markdown("###### Mapping Audit Trail")
                hist_data = [
                    {
                        "Timestamp (UTC)": e.timestamp_utc,
                        "Status Transition": f"{e.previous_status} ➔ {e.new_status}",
                        "Reviewer": e.reviewer_label or "Anonymous",
                        "Notes": e.analyst_notes or "",
                        "Event Hash": f"{e.event_hash[:12]}...",
                    }
                    for e in history_events
                ]
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: Contextual Predictions
    # --------------------------------------------------------------------------
    with tab_ctx:
        section_header(
            title="Evidence-Aware Contextual Prediction Inspector",
            description="Select an actual detection run from CP-08 and enrich classified flows with ATT&CK context:",
        )

        exp_registry = ExplorerRegistry()
        eligible_runs = exp_registry.list_eligible_runs()

        if not eligible_runs:
            info_panel(
                title="No Completed Detection Runs Available",
                message="Execute an intrusion detection run in the Detection Center to inspect contextual predictions.",
            )
        else:
            run_dict = {r.display_label: r for r in eligible_runs}
            sel_run_lbl = st.selectbox(
                "Select Detection Run to Contextualize:",
                options=list(run_dict.keys()),
                key="sel_ctx_run",
            )
            active_run = run_dict[sel_run_lbl]

            p_reader = PredictionReader()
            run_preds_df = p_reader.load_predictions(active_run.detection_run_id, attach_reviews=True)

            if not run_preds_df.empty:
                row_choices = run_preds_df["source_row_index"].tolist()
                sel_row_idx = st.selectbox(
                    "Select Flow Row to Inspect:",
                    options=row_choices[:100],  # Bounded to first 100 in selector for speed
                    key="sel_ctx_row_idx",
                )

                row_item = run_preds_df[run_preds_df["source_row_index"] == sel_row_idx].iloc[0]

                target_cls = str(row_item.get("predicted_class", "UNKNOWN"))
                single_score = row_item.get("model_score")
                score_type = str(row_item.get("score_type", "score"))
                disagree = str(row_item.get("agreement_status", "Agreement"))
                gt_label = str(row_item.get("ground_truth_label")) if "ground_truth_label" in row_item else None
                analyst_disp = str(row_item.get("analyst_disposition")) if "analyst_disposition" in row_item else None

                # Generate enriched context
                enriched = context_service.enrich_prediction_record(
                    predicted_class=target_cls,
                    model_name=active_run.selected_algorithms[0] if active_run.selected_algorithms else "svm",
                    model_score=single_score,
                    score_type=score_type,
                    source_row_index=sel_row_idx,
                    flow_attributes={},
                    ground_truth_label=gt_label,
                    analyst_disposition=analyst_disp,
                    disagreement_state=disagree,
                )

                ctx_col1, ctx_col2 = st.columns(2)
                with ctx_col1:
                    st.markdown(f"#### Predicted Attack: `{enriched.predicted_class}`")
                    st.markdown(f"**Threat Family:** `{enriched.taxonomy_item.broad_family if enriched.taxonomy_item else 'Unknown'}`")
                    st.markdown(f"**Disagreement Status:** `{enriched.disagreement_state}`")
                    if enriched.ground_truth_label:
                        st.markdown(f"**Dataset Ground Truth:** `{enriched.ground_truth_label}`")
                    if enriched.analyst_disposition and enriched.analyst_disposition != "Unreviewed":
                        st.markdown(f"**Analyst Disposition:** `{enriched.analyst_disposition}`")

                    # Candidate ATT&CK mapping
                    st.markdown("---")
                    st.markdown("##### Candidate MITRE ATT&CK Association")
                    if enriched.candidate_mapping and enriched.candidate_mapping.technique_id:
                        cm = enriched.candidate_mapping
                        st.markdown(f"**Technique:** `{cm.technique_id}` — {cm.technique_name}")
                        st.markdown(f"**Tactic:** `{cm.tactic}`")
                        st.markdown(f"**Mapping Status:** `{cm.mapping_status}`")
                        st.markdown(f"**Alignment Strength:** `{cm.mapping_strength}`")
                        st.markdown(f"**Rationale:** {cm.mapping_rationale}")
                    else:
                        st.info("No candidate ATT&CK technique mapping assigned to this attack class.")

                with ctx_col2:
                    st.markdown("##### Evidence Evaluation")
                    st.markdown("###### Missing Evidence from Telemetry")
                    for mis in enriched.missing_evidence_fields:
                        st.markdown(f"- ⚠️ {mis}")

                    st.markdown("---")
                    st.markdown("##### Relevant Defensive Mitigations")
                    if enriched.relevant_mitigations:
                        for mit in enriched.relevant_mitigations:
                            st.markdown(f"- **{mit.external_id}** — {mit.name}")
                            st.caption(mit.description[:120] + "...")
                    else:
                        st.caption("No mitigations linked to this candidate mapping.")

                # Non-Attribution Disclaimer
                st.markdown("---")
                info_panel(
                    title="Forensic Non-Attribution Safeguard",
                    message=enriched.context_disclaimer,
                )

    # --------------------------------------------------------------------------
    # TAB 5: Security Insights
    # --------------------------------------------------------------------------
    with tab_ins:
        section_header(
            title="Deterministic Cybersecurity Insights",
            description="Verifiable security metrics derived from active detection runs and ATT&CK mappings:",
        )

        ins_col1, ins_col2 = st.columns(2)

        with ins_col1:
            st.markdown("##### ATT&CK Mapping Coverage")
            cov = insight_engine.compute_mapping_coverage()
            cov_df = pd.DataFrame([
                {"Category": "Approved Mappings", "Count": cov["approved_classes"]},
                {"Category": "Candidate Mappings", "Count": cov["mapped_classes"] - cov["approved_classes"]},
                {"Category": "Unmapped Classes", "Count": cov["unmapped_classes"]},
            ])
            fig_cov = px.pie(
                cov_df,
                names="Category",
                values="Count",
                color="Category",
                color_discrete_map={
                    "Approved Mappings": "#22C55E",
                    "Candidate Mappings": "#3B82F6",
                    "Unmapped Classes": "#64748B",
                },
                hole=0.45,
                title="Class Mapping Coverage",
            )
            fig_cov.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E0E0E0"})
            st.plotly_chart(fig_cov, use_container_width=True)

        with ins_col2:
            st.markdown("##### Unmapped / Inconclusive Classes")
            st.caption(
                "These classes lack sufficient flow-level evidence to associate with specific ATT&CK techniques without payload inspection:"
            )
            for u in cov["unmapped_list"]:
                st.markdown(f"- **{u}** — *No specific ATT&CK technique assigned*")

    # --------------------------------------------------------------------------
    # TAB 6: Reference Data Management
    # --------------------------------------------------------------------------
    with tab_mgm:
        section_header(
            title="MITRE ATT&CK Reference Data Management",
            description="Inspect active STIX 2.1 dataset provenance or import updated local official snapshots:",
        )

        if ref_metadata:
            prov_col1, prov_col2 = st.columns(2)
            with prov_col1:
                st.markdown(f"**Dataset Name:** `{ref_metadata.dataset_name}`")
                st.markdown(f"**Version:** `{ref_metadata.version}`")
                st.markdown(f"**Release Date:** `{ref_metadata.release_date}`")
                st.markdown(f"**Source File:** `{ref_metadata.source_file}`")
                st.markdown(f"**Validation Status:** `{ref_metadata.validation_status}`")
            with prov_col2:
                st.markdown(f"**STIX Specification:** `{ref_metadata.stix_spec_version}`")
                st.markdown(f"**Total STIX Objects:** `{ref_metadata.total_objects:,}`")
                st.markdown(f"**Techniques Parsed:** `{ref_metadata.techniques_count:,}`")
                st.markdown(f"**Tactics Parsed:** `{ref_metadata.tactics_count:,}`")
                st.markdown(f"**SHA-256:** `{ref_metadata.sha256_checksum[:20]}...`")

        st.markdown("---")
        st.markdown("##### Manual STIX 2.1 Bundle Import")
        st.caption("Import an official MITRE ATT&CK Enterprise STIX 2.1 JSON file into the local knowledge base:")

        import_path_str = st.text_input(
            "Path to Local STIX 2.1 JSON Bundle:",
            placeholder="e.g. C:/Downloads/enterprise-attack-20.1.json",
            key="txt_import_stix_path",
        )
        import_version_str = st.text_input("Version Tag:", value="v20.1", key="txt_import_stix_ver")

        if st.button("📥 Import & Validate Bundle", key="btn_import_stix"):
            if not import_path_str.strip():
                st.error("Please enter a valid file path to a STIX 2.1 JSON bundle.")
            else:
                ok, msg, new_meta = ref_registry.import_stix_bundle(
                    bundle_file_path=Path(import_path_str.strip()),
                    version_label=import_version_str.strip(),
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
