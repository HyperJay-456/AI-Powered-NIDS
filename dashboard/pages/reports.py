"""Reports Page: Executive summaries, audit trails, and forensic exports."""

import streamlit as st

from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import pending_panel


def render() -> None:
    """Render the Reports view."""
    page_header(
        title="Reports & Forensic Exports",
        subtitle="Executive summaries, forensic event exports, model cards, and audit logs",
        icon="📑",
        badge_text="Scheduled for CP-10",
        badge_variant="neutral",
    )

    pending_panel(
        checkpoint_id="CP-10",
        title="Reports & Exports",
        description=(
            "Provides automated report generation and export capabilities for "
            "security audits, academic evaluation, and compliance documentation."
        ),
        planned_capabilities=[
            "One-click PDF / Markdown executive threat summary export",
            "Detailed CSV export for flagged malicious flow events",
            "Model card generation summarizing hyperparameters, training dataset, and validation metrics",
            "Audit trail logger recording analyst queries and configuration changes",
        ],
    )

    section_header(
        title="Export Templates Available in CP-10",
        description="Standardized report formats supported by the platform:",
    )

    st.markdown(
        """
        1. **Executive Threat Assessment Report:** High-level summary of network traffic, anomalous spikes, and classified threat families.
        2. **Forensic Flow Event Log:** Complete row-level attributes of all flows triggering intrusion alert thresholds.
        3. **Academic Evaluation Card:** Comprehensive benchmark report detailing Accuracy, Precision, Recall, F1, ROC-AUC, and inference latency across SVM, MLP, and Decision Tree.
        """
    )
