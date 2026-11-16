"""Detection Center Page: CSV-based intrusion detection analysis and operational console."""

import streamlit as st

from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import pending_panel


def render() -> None:
    """Render the Detection Center view."""
    page_header(
        title="Intrusion Detection Center",
        subtitle="CSV-based intrusion detection analysis, alert thresholding, and forensic triage",
        icon="⚡",
        badge_text="Scheduled for CP-08",
        badge_variant="warning",
    )

    pending_panel(
        checkpoint_id="CP-08",
        title="Intrusion Detection Engine",
        description=(
            "The operational intrusion detection console will execute CSV-based intrusion detection "
            "analysis on uploaded network flow batches using pre-trained SVM, MLP, "
            "and Decision Tree estimators."
        ),
        planned_capabilities=[
            "Single-flow vector manual inspection simulator",
            "Batch network flow record classifier from uploaded CSV files",
            "Configurable probability thresholds for alert escalation",
            "Multi-tiered severity scoring (Benign, Low, Medium, High, Critical)",
            "Automated incident alert queue and mitigation recommendations",
        ],
    )

    section_header(
        title="Operational Detection Workflow",
        description="Flow telemetry pipeline from ingestion to alert generation:",
    )

    st.markdown(
        """
        ```
        [ Raw Flow Vector ] -> [ Feature Scaler ] -> [ Model Pipeline (SVM/MLP/DT) ] -> [ Confidence Threshold ] -> [ Alert Console ]
        ```
        """
    )
