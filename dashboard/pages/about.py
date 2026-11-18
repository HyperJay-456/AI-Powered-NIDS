"""About Page: Academic context, system specifications, and ethical disclaimer."""

import streamlit as st

from config.settings import settings
from dashboard.components.headers import page_header, section_header
from src import __version__


def render() -> None:
    """Render the About view."""
    page_header(
        title="About CYBER-NIDS Platform",
        subtitle="Architecture specifications, algorithmic justifications, and security compliance",
        icon="ℹ️",
        badge_text=f"v{__version__} Production",
        badge_variant="success",
    )

    section_header(
        title="Platform Mission & Objective",
        description="Bridging statistical machine learning theory with enterprise network intrusion defense.",
    )

    st.markdown(
        """
        The **CYBER-NIDS Platform** is an enterprise-grade threat detection and telemetry classification system
        engineered to evaluate the operational trade-offs between geometric margin classification, feedforward
        neural representations, and recursive binary decision trees in classifying complex network flow telemetry.

        Using the realistic **CIC-IDS-2017** benchmark corpus, the system is architected to address modern
        enterprise cybersecurity challenges, including low latency requirements, class imbalance across rare
        attack vectors, and verifiable explainability of automated classification decisions.
        """
    )

    section_header(
        title="Tri-Model Algorithmic Justifications",
        description="Why these three distinct machine learning paradigms were selected:",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="nids-card" style="height: 100%;">
                <div class="nids-card-title">Support Vector Machine</div>
                <div style="font-size: 0.8125rem; color: #818cf8; font-weight: 600; margin-bottom: 0.5rem;">LinearSVC (Margin-Based)</div>
                <div style="font-size: 0.8125rem; color: var(--bb-text-muted); line-height: 1.5;">
                    Effective in high-dimensional feature spaces with clear margins of separation. Provides strong mathematical guarantees against overfitting on flow rates.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="nids-card" style="height: 100%;">
                <div class="nids-card-title">Multilayer Perceptron</div>
                <div style="font-size: 0.8125rem; color: #38bdf8; font-weight: 600; margin-bottom: 0.5rem;">MLPClassifier (Neural Network)</div>
                <div style="font-size: 0.8125rem; color: var(--bb-text-muted); line-height: 1.5;">
                    Models complex non-linear correlations and cross-feature interactions across continuous network flow telemetry with adaptive early stopping.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="nids-card" style="height: 100%;">
                <div class="nids-card-title">Decision Tree</div>
                <div style="font-size: 0.8125rem; color: #34d399; font-weight: 600; margin-bottom: 0.5rem;">DecisionTreeClassifier (CART)</div>
                <div style="font-size: 0.8125rem; color: var(--bb-text-muted); line-height: 1.5;">
                    Deterministic hierarchical decision logic, directly interpretable split paths, and robust handling of physical network packet attribute thresholds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header(
        title="System Specifications & Release Information",
        description="Release metadata and runtime specifications for the local platform:",
    )

    spec_col1, spec_col2, spec_col3 = st.columns(3)
    with spec_col1:
        st.markdown(
            f"""
            - **Platform Version:** `v{__version__}`
            - **Release Status:** Production Release
            - **Architecture:** Local-First / Offline
            """
        )
    with spec_col2:
        st.markdown(
            """
            - **Python Environment:** Python 3.13.14
            - **Interface:** Streamlit Framework
            - **Dataset Benchmark:** CIC-IDS-2017
            """
        )
    with spec_col3:
        st.markdown(
            """
            - **Algorithms (3):** LinearSVC, MLP, DecisionTree
            - **Knowledge Base:** MITRE ATT&CK v20.1
            - **Reporting:** Publication-Grade PDF & Bundles
            """
        )

    section_header(
        title="Academic & Ethical Disclaimer",
        description="Standards of academic integrity and responsible vulnerability disclosure:",
    )

    st.markdown(
        """
        This software is developed strictly for educational, scientific, and defensive cybersecurity research.
        All network attack categories analyzed originate from sanitized, publicly released academic datasets
        (CIC-IDS-2017). The project does not contain tools for malicious exploitation, unauthorized penetration
        testing, or offensive cyber operations.
        """
    )
