"""Reporting & Evidence Export Page (CP-11).

Provides an interactive console for generating publication-grade cybersecurity reports,
exporting structured machine-readable JSON/CSVs, configuring privacy redactions,
and packaging cryptographically verified digital evidence bundles.
"""

from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config.settings import settings
from dashboard.components.badges import badge_html
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import error_panel, info_panel, success_panel, warning_panel
from src.detection.detection_registry import DetectionRegistry
from src.reporting.report_models import (
    RedactionConfig,
    ReportClassification,
    ReportMetadata,
    ReportStatus,
    ReportType,
)
from src.reporting.report_registry import ReportRegistry
from src.reporting.report_service import ReportService
from src.utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Reporting & Evidence Export dashboard view."""
    page_header(
        title="Reporting & Evidence Export",
        subtitle="Publication-grade PDF assessment reports, machine-readable forensic exports, and verified evidence bundles",
        icon="📑",
        badges=[("Forensic PDF Engine", "info"), ("Cryptographic Verification", "success")],
    )

    registry = ReportRegistry()
    detection_registry = DetectionRegistry()
    service = ReportService()

    # --------------------------------------------------------------------------
    # Reporting Overview
    # --------------------------------------------------------------------------
    section_header(
        title="Reporting & Evidence Overview",
        description="Summary of generated cybersecurity assessment reports and verified evidence bundles.",
    )

    all_reports = registry.list_reports(include_test_fixtures=True, verify_checksums=True)
    prod_reports = [r for r in all_reports if not r.get("is_test_fixture", False)]

    total_count = len(prod_reports)
    exec_count = sum(1 for r in prod_reports if r.get("report_type") == ReportType.EXECUTIVE)
    tech_count = sum(1 for r in prod_reports if r.get("report_type") == ReportType.TECHNICAL)
    analyst_count = sum(1 for r in prod_reports if r.get("report_type") == ReportType.ANALYST_REVIEW)
    cons_count = sum(1 for r in prod_reports if r.get("report_type") == ReportType.CONSOLIDATED)
    bundles_count = sum(1 for r in prod_reports if r.get("has_bundle", False))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card(label="Total Reports", value=str(total_count), subtitle="Production reports")
    with c2:
        metric_card(label="Executive Summaries", value=str(exec_count), subtitle="Management reports")
    with c3:
        metric_card(label="Technical Reports", value=str(tech_count), subtitle="Forensic analyses")
    with c4:
        metric_card(label="Analyst Audits", value=str(analyst_count), subtitle="Human review audits")
    with c5:
        metric_card(label="Evidence Bundles", value=str(bundles_count), subtitle="Verified ZIP packages")

    st.write("")
    if total_count == 0:
        info_panel(
            title="No Production Reports Generated Yet",
            message=(
                "No cybersecurity assessment reports have been generated for verified production detection runs. "
                "Select a completed detection run in Section B below and click 'Generate Cybersecurity Report'."
            ),
        )

    # --------------------------------------------------------------------------
    # Source Detection Run Selection
    # --------------------------------------------------------------------------
    section_header(
        title="Source Detection Run Selection",
        description="Select a completed intrusion detection run to serve as the evidence baseline.",
    )

    allow_fixtures = st.checkbox(
        "Include automated test fixtures (for demonstration & evaluation)",
        value=False,
        help="Enables selection of synthetic test fixture runs when no production traffic analyses exist.",
    )

    available_runs = detection_registry.list_detection_runs(
        include_test_fixtures=allow_fixtures,
        verify_checksums=True,
    )

    if not available_runs:
        warning_panel(
            title="No Eligible Detection Runs Available",
            message=(
                "No completed detection runs were found in data/analyses/. "
                "Execute intrusion detection in Section 06 (Intrusion Detection Engine) before generating reports, "
                "or enable test fixtures above to inspect reporting templates."
            ),
        )
        selected_run_id = None
    else:
        run_options = {
            f"{r['detection_run_id']} | {r['source_filename']} ({r['predicted_rows']:,} rows)": r["detection_run_id"]
            for r in available_runs
        }
        selected_label = st.selectbox("Select Detection Run:", options=list(run_options.keys()))
        selected_run_id = run_options[selected_label]

        # Display Selected Run Snapshot
        selected_run_meta = next((r for r in available_runs if r["detection_run_id"] == selected_run_id), None)
        if selected_run_meta:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.caption("Detection Run ID")
                st.write(f"<code>{selected_run_meta['detection_run_id']}</code>", unsafe_allow_html=True)
            with m2:
                st.caption("Source File")
                st.write(f"<b>{selected_run_meta.get('source_filename', 'unknown')}</b>", unsafe_allow_html=True)
            with m3:
                st.caption("Analyzed Records")
                st.write(f"{selected_run_meta.get('predicted_rows', 0):,}")
            with m4:
                st.caption("Selected Algorithms")
                algos = selected_run_meta.get("selected_algorithms", [])
                st.write(", ".join(algos) if algos else "SVM, MLP, Decision Tree")

    # --------------------------------------------------------------------------
    # Report Configuration
    # --------------------------------------------------------------------------
    section_header(
        title="Report Format & Classification Configuration",
        description="Configure report archetype, audience scope, sensitivity labels, and custom titles.",
    )

    c_col1, c_col2, c_col3 = st.columns([1.5, 2, 1.5])
    with c_col1:
        report_type = st.selectbox(
            "Report Archetype:",
            options=list(ReportType.ALL_TYPES),
            index=0,
            help=(
                "Executive: high-level metrics and observations; "
                "Technical: per-class distributions and algorithm comparisons; "
                "Analyst: human-in-the-loop review audit trail; "
                "Consolidated: unified multi-section assessment."
            ),
        )

    with c_col2:
        custom_title = st.text_input(
            "Custom Report Title (Optional):",
            placeholder=f"e.g. Q4 Security Assessment — {selected_run_id or 'Traffic'}",
        )

    with c_col3:
        classification = st.selectbox(
            "Information Sensitivity Label:",
            options=list(ReportClassification.ALL_CLASSIFICATIONS),
            index=0,
            help="Traffic Light Protocol (TLP) or Academic marking printed in header and metadata.",
        )

    # --------------------------------------------------------------------------
    # Privacy & Evidence Configuration
    # --------------------------------------------------------------------------
    section_header(
        title="Privacy Controls & Evidence Selection",
        description="Enforce data protection, IP address masking, and formula injection safeguards.",
    )

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        redact_ips = st.checkbox(
            "Redact Host IP Addresses",
            value=True,
            help="Masks IPv4 host octets (e.g. 192.168.1.xxx) across all exported reports and CSV summaries.",
        )
    with d_col2:
        include_notes = st.checkbox(
            "Include Analyst Notes",
            value=True,
            help="Includes human analyst review comments in report audit trails (sanitized for privacy).",
        )
    with d_col3:
        package_bundle = st.checkbox(
            "Package Evidence Bundle ZIP",
            value=True,
            help="Generates a downloadable, path-traversal secure evidence_bundle_<report_id>.zip with manifest.",
        )

    st.markdown(
        """
        <div style="font-size: 0.8125rem; color: #64748B; background: #0F172A; padding: 8px 12px; border-radius: 4px; margin-top: 6px;">
            🛡️ <b>Security Safeguards:</b> All exported CSV spreadsheets are sanitized against CSV Formula Injection (DDE).
            Original source files are read-only and never modified; redaction is applied to exported copies only.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Report Generation Action
    # --------------------------------------------------------------------------
    st.write("")
    section_header(
        title="Report Generation Engine",
        description="Compile publication-grade PDF documents and cryptographically verified evidence packages.",
    )

    gen_btn = st.button(
        "🚀 Generate Cybersecurity Report",
        type="primary",
        disabled=(selected_run_id is None),
        use_container_width=True,
    )

    if gen_btn and selected_run_id:
        with st.spinner("Compiling publication-grade report, verifying checksums, and building evidence manifest..."):
            try:
                redact_cfg = RedactionConfig(
                    redact_ips=redact_ips,
                    include_analyst_notes=include_notes,
                    include_network_identifiers=False,
                )

                meta_result: ReportMetadata = service.generate_report(
                    detection_run_id=selected_run_id,
                    report_type=report_type,
                    title=custom_title if custom_title.strip() else None,
                    classification=classification,
                    redaction_config=redact_cfg,
                    allow_test_fixtures=allow_fixtures,
                    create_bundle=package_bundle,
                )

                st.session_state["latest_report_id"] = meta_result.report_id
                success_panel(
                    title=f"Report Generated Successfully: {meta_result.report_id}",
                    message=(
                        f"Document compiled: {meta_result.pdf_filename} ({meta_result.pdf_page_count} pages, "
                        f"{meta_result.pdf_size_bytes:,} bytes). "
                        f"Evidence manifest verified (SHA-256: {meta_result.pdf_sha256[:16]}...)."
                    ),
                )
            except Exception as e:
                error_panel(
                    title="Report Generation Error",
                    message=f"Failed to generate report: {str(e)}",
                )
                logger.error(f"UI report generation failed: {e}", exc_info=True)

    # --------------------------------------------------------------------------
    # Preview & Download Center
    # --------------------------------------------------------------------------
    latest_id = st.session_state.get("latest_report_id")
    if latest_id:
        st.write("")
        section_header(
            title="Report Preview & Download Center",
            description=f"Direct inspection and downloads for Report: {latest_id}",
        )

        rep_data = registry.get_report_data(latest_id)
        if rep_data:
            meta = rep_data.get("metadata", {})
            r_dir = Path(settings.reports_dir) / latest_id

            tab1, tab2, tab3, tab4 = st.tabs([
                "📄 Document Metadata",
                "📊 Traffic & Predictions",
                "🔍 Analyst Audit & ATT&CK",
                "💾 Download Artifacts",
            ])

            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Report ID:** `{meta.get('report_id')}`")
                    st.markdown(f"**Report Archetype:** `{meta.get('report_type')}`")
                    st.markdown(f"**Classification:** `{meta.get('classification')}`")
                    st.markdown(f"**Detection Run:** `{meta.get('detection_run_id')}`")
                    st.markdown(f"**Source File:** `{meta.get('source_filename')}`")
                with col_b:
                    st.markdown(f"**Generated UTC:** `{meta.get('created_at_utc')}`")
                    st.markdown(f"**Total Records:** `{meta.get('total_records_analyzed', 0):,}`")
                    st.markdown(f"**PDF Pages:** `{meta.get('pdf_page_count', 0)}`")
                    st.markdown(f"**PDF SHA-256:** `{meta.get('pdf_sha256')}`")
                    if meta.get("bundle_filename"):
                        st.markdown(f"**Evidence Bundle:** `{meta.get('bundle_filename')}`")

            with tab2:
                p_dist = rep_data.get("prediction_distribution", [])
                if p_dist:
                    df_dist = pd.DataFrame(p_dist)
                    st.dataframe(df_dist, use_container_width=True)
                else:
                    st.info("No prediction distribution records available.")

            with tab3:
                c_rev, c_attck = st.columns(2)
                with c_rev:
                    st.markdown("##### Analyst Dispositions")
                    disp_counts = rep_data.get("analyst_summary", {}).get("disposition_counts", {})
                    if disp_counts:
                        st.json(disp_counts)
                    else:
                        st.caption("No reviewed records.")
                with c_attck:
                    st.markdown("##### ATT&CK Candidate Mappings")
                    mappings = rep_data.get("attck_mappings", [])
                    if mappings:
                        st.write(f"Total Associations: **{len(mappings)}**")
                        for m in mappings[:5]:
                            st.write(f"• **{m['dataset_label']}** → `{m['technique_id']}` ({m['technique_name']}) — *{m['status']}*")
                    else:
                        st.caption("No mapped attack classes.")

            with tab4:
                st.markdown("##### Download Export Artifacts")
                d_c1, d_c2, d_c3 = st.columns(3)

                # PDF Download
                pdf_file = r_dir / meta.get("pdf_filename", "report.pdf")
                with d_c1:
                    if pdf_file.exists():
                        with open(pdf_file, "rb") as pf:
                            st.download_button(
                                label="📥 Download Report PDF",
                                data=pf.read(),
                                file_name=pdf_file.name,
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.button("PDF Not Found", disabled=True, use_container_width=True)

                # JSON Download
                json_file = r_dir / "report_data.json"
                with d_c2:
                    if json_file.exists():
                        with open(json_file, "rb") as jf:
                            st.download_button(
                                label="📥 Download report_data.json",
                                data=jf.read(),
                                file_name=f"{latest_id}_data.json",
                                mime="application/json",
                                use_container_width=True,
                            )
                    else:
                        st.button("JSON Not Found", disabled=True, use_container_width=True)

                # Evidence Bundle ZIP Download
                bundle_file = r_dir / meta.get("bundle_filename", f"evidence_bundle_{latest_id}.zip")
                with d_c3:
                    if bundle_file.exists():
                        with open(bundle_file, "rb") as zf:
                            st.download_button(
                                label="📦 Download Evidence ZIP",
                                data=zf.read(),
                                file_name=bundle_file.name,
                                mime="application/zip",
                                use_container_width=True,
                            )
                    else:
                        st.button("ZIP Not Found", disabled=True, use_container_width=True)

                # CSV Downloads Expander
                with st.expander("📥 Download Sanitized Forensic CSV Tables"):
                    exports_dir = r_dir / "exports"
                    for csv_name in [
                        "prediction_summary.csv",
                        "model_comparison.csv",
                        "analyst_review_summary.csv",
                        "attck_mapping_summary.csv",
                    ]:
                        csv_path = exports_dir / csv_name
                        if csv_path.exists():
                            with open(csv_path, "rb") as cf:
                                st.download_button(
                                    label=f"Download {csv_name}",
                                    data=cf.read(),
                                    file_name=f"{latest_id}_{csv_name}",
                                    mime="text/csv",
                                )

    # --------------------------------------------------------------------------
    # Historical Reports Catalog
    # --------------------------------------------------------------------------
    st.write("")
    section_header(
        title="Historical Reports & Cryptographic Verification",
        description="Browse, verify cryptographic checksums, and download previous reports without regeneration.",
    )

    history_reports = registry.list_reports(include_test_fixtures=allow_fixtures, verify_checksums=True)
    if not history_reports:
        st.info("No historical reports found in data/reports/.")
    else:
        for idx, rep in enumerate(history_reports[:20]):
            r_id = rep["report_id"]
            with st.expander(f"📑 {rep['title']} — {r_id} ({rep['created_at_utc'][:10]})"):
                h_c1, h_c2, h_c3, h_c4 = st.columns(4)
                with h_c1:
                    st.caption("Archetype")
                    st.write(f"<b>{rep['report_type']}</b>", unsafe_allow_html=True)
                with h_c2:
                    st.caption("Detection Run")
                    st.write(f"<code>{rep['detection_run_id']}</code>", unsafe_allow_html=True)
                with h_c3:
                    st.caption("Records Analyzed")
                    st.write(f"{rep['total_records']:,}")
                with h_c4:
                    st.caption("PDF Status")
                    badge = badge_html("Verified", "success") if rep["pdf_valid"] else badge_html("Corrupted", "error")
                    st.write(badge, unsafe_allow_html=True)

                # Integrity Verification Action
                verify_col, dl_pdf_col, dl_zip_col = st.columns(3)
                with verify_col:
                    if st.button(f"🔍 Verify Integrity", key=f"verify_{r_id}_{idx}"):
                        ver_res = registry.verify_report_integrity(r_id)
                        if ver_res["verified"]:
                            st.success("All cryptographic checksums verified (PDF, JSON, ZIP).")
                        else:
                            st.error(f"Verification failure: {', '.join(ver_res['errors'])}")

                rep_dir = Path(rep["dir_path"])
                pdf_p = rep_dir / "report.pdf"
                zip_p = rep_dir / f"evidence_bundle_{r_id}.zip"

                with dl_pdf_col:
                    if pdf_p.exists():
                        with open(pdf_p, "rb") as pf:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pf.read(),
                                file_name=f"{r_id}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{r_id}_{idx}",
                            )

                with dl_zip_col:
                    if zip_p.exists():
                        with open(zip_p, "rb") as zf:
                            st.download_button(
                                label="📦 Download ZIP",
                                data=zf.read(),
                                file_name=f"evidence_{r_id}.zip",
                                mime="application/zip",
                                key=f"dl_zip_{r_id}_{idx}",
                            )
