"""Prediction Explorer Page: Model explainability and feature attribution."""

import streamlit as st

from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import pending_panel


def render() -> None:
    """Render the Prediction Explorer view."""
    page_header(
        title="Prediction Explorer & Explainability",
        subtitle="Feature attribution, prediction confidence distributions, and decision transparency",
        icon="🔍",
        badge_text="Scheduled for CP-09",
        badge_variant="info",
    )

    pending_panel(
        checkpoint_id="CP-09",
        title="Threat Intelligence & Prediction Explorer",
        description=(
            "Enables cybersecurity analysts to inspect individual classification decisions, "
            "evaluating why a specific network flow was categorized as an intrusion and assessing "
            "model prediction confidence."
        ),
        planned_capabilities=[
            "Flow feature attribution and discriminator importance plots",
            "Probability distribution waterfall charts across all 15 candidate classes",
            "Margin distance visualizer for Support Vector Machine decisions",
            "Decision path tracer through Decision Tree partition rules",
            "Activation layer inspection for Multilayer Perceptron representations",
        ],
    )

    section_header(
        title="Key Diagnostic Flow Features",
        description="Core statistical attributes examined during intrusion classification:",
    )

    st.markdown(
        """
        - **Temporal Dynamics:** Flow Duration, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min.
        - **Volumetric Indicators:** Total Fwd Packets, Total Backward Packets, Flow Bytes/s, Flow Packets/s.
        - **Packet Distributions:** Packet Length Mean, Packet Length Std, Packet Length Variance, Max Packet Length.
        - **Protocol Flags:** SYN Flag Count, FIN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count.
        """
    )
