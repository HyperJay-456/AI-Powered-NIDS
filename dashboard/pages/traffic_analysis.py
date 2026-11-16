"""Traffic Analysis Page: Network flow inspection and data exploration."""

import streamlit as st

from config.settings import settings
from dashboard.components.actions import styled_table
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import pending_panel


def render() -> None:
    """Render the Traffic Analysis view."""
    page_header(
        title="Network Traffic Analysis & Explorer",
        subtitle="Flow telemetry inspection, feature distributions, and packet characteristics",
        icon="📊",
        badge_text="Scheduled for CP-04",
        badge_variant="info",
    )

    pending_panel(
        checkpoint_id="CP-04",
        title="Traffic Upload & Data Explorer",
        description=(
            "This module will provide CSV-based network flow ingestion and batch analysis, "
            "schema validation against expected CIC-IDS-2017 features (awaiting validation in CP-04), and interactive "
            "tabular telemetry analysis."
        ),
        planned_capabilities=[
            "Drag-and-drop CSV flow file upload and parsing",
            "Interactive schema verification against Canadian Cyber Center specifications",
            "Multi-column filtering (Flow Duration, Total Fwd Packets, Flow Bytes/s, Flag counts)",
            "Dynamic histogram and scatter plot generators for traffic distributions",
            "Anomaly visualizer for extreme outlier detection (e.g. volumetric DDoS bursts)",
        ],
    )

    section_header(
        title="Target Traffic File Structure",
        description=f"Expected flow files in {settings.dataset.name} corpus:",
    )

    files_data = [
        {"Target File": fname, "Expected Path": f"data/raw/{fname}", "Pipeline Stage": "CP-04 / CP-05"}
        for fname in settings.dataset.raw_files
    ]
    styled_table(files_data)
