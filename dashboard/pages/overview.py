"""Overview Page: Executive system overview and architecture status."""

import streamlit as st

from config.settings import SUPPORTED_ALGORITHMS, settings
from dashboard.components.actions import styled_table
from dashboard.components.badges import render_badge
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import info_panel


def render() -> None:
    """Render the Overview page viewport."""
    page_header(
        title="System Overview & Architecture",
        subtitle="AI-Powered Network Intrusion Detection Platform (CIC-IDS-2017)",
        icon="🛡️",
        badge_text="CP-02 Active",
        badge_variant="success",
    )

    # Top KPI Metrics (Real system state, zero fabrication)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            label="Target Dataset",
            value=settings.dataset.name,
            subtitle="CIC-UNB Network Flows",
        )
    with col2:
        metric_card(
            label="Authorized Models",
            value=f"{len(SUPPORTED_ALGORITHMS)} Algorithms",
            subtitle="SVM, MLP, Decision Tree",
        )
    with col3:
        metric_card(
            label="Active Milestone",
            value="CP-02",
            subtitle="UI Design System",
            status_color="var(--color-accent-primary)",
        )
    with col4:
        metric_card(
            label="Engine Status",
            value="Idle",
            subtitle="Awaiting CP-06 Pipeline",
            status_color="var(--color-status-warning)",
        )

    # Core System Architecture Panel
    section_header(
        title="Layered System Architecture",
        description="Unified 5-tier architecture connecting raw network telemetry to real-time classification.",
    )

    st.markdown(
        """
        ```
        +---------------------------------------------------------------------------------------+
        |                               Presentation Layer (Streamlit)                          |
        |      Modern Enterprise Cybersecurity SaaS UI  •  Centralized Design Tokens            |
        +-------------------------------------------+-------------------------------------------+
                                                    |
        +-------------------------------------------v-------------------------------------------+
        |                               Intrusion Detection Engine                              |
        |               Sub-50ms Low-Latency Inference  •  Multi-class Attack Mapping           |
        +-------------------------------------------+-------------------------------------------+
                                                    |
        +-------------------------------------------v-------------------------------------------+
        |                                Model Subsystem (3 Algorithms)                         |
        |       Support Vector Machine (SVM)  •  Multilayer Perceptron (MLP)  •  Decision Tree  |
        +-------------------------------------------+-------------------------------------------+
                                                    |
        +-------------------------------------------v-------------------------------------------+
        |                           Feature Engineering & Preprocessing                         |
        |              Missing/Inf Remediation  •  Robust Scaling  •  Stratified Splits         |
        +-------------------------------------------+-------------------------------------------+
                                                    |
        +-------------------------------------------v-------------------------------------------+
        |                                   Data Asset Layer                                    |
        |               CIC-IDS-2017 Network Flow Logs  •  Canadian Cyber Center                |
        +---------------------------------------------------------------------------------------+
        ```
        """
    )

    # Repository & Environment Health
    section_header(
        title="Environment & Storage Subsystems",
        description="Status of project directories, model checkpoints, and data pipelines.",
    )

    env_records = [
        {"Subsystem": "Raw Data Repository", "Path": str(settings.raw_data_dir.relative_to(settings.base_dir)), "Status": "Configured (Awaiting CP-05)"},
        {"Subsystem": "Processed Data Repository", "Path": str(settings.processed_data_dir.relative_to(settings.base_dir)), "Status": "Configured (Awaiting CP-05)"},
        {"Subsystem": "Serialized Model Registry", "Path": str(settings.saved_models_dir.relative_to(settings.base_dir)), "Status": "Configured (Awaiting CP-06)"},
        {"Subsystem": "Runtime Logs Store", "Path": str(settings.logs_dir.relative_to(settings.base_dir)), "Status": "Active"},
    ]
    styled_table(env_records)

    info_panel(
        title="Checkpoint CP-02 Notice",
        message=(
            "The UI Design System and application shell are actively loaded. "
            "ML detection pipelines, live flow ingestion, and model training will be "
            "progressively integrated in accordance with the 12-checkpoint execution plan."
        ),
        icon="ℹ️",
    )
