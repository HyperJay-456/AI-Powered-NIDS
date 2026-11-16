"""About Page: Academic context, system specifications, and ethical disclaimer."""

import streamlit as st

from config.settings import settings
from dashboard.components.headers import page_header, section_header


def render() -> None:
    """Render the About view."""
    page_header(
        title="About the NIDS Platform",
        subtitle="Academic context, theoretical foundations, and system specifications",
        icon="ℹ️",
        badge_text="Academic / MIT License",
        badge_variant="neutral",
    )

    section_header(
        title="Project Mission & Objective",
        description="Bridging statistical machine learning theory with enterprise network intrusion defense.",
    )

    st.markdown(
        """
        The **AI-Powered Network Intrusion Detection System (NIDS)** is an advanced academic research project
        investigating the operational trade-offs between geometric margin classification, artificial neural
        representation learning, and recursive binary decision trees in identifying network intrusions.

        Using the realistic **CIC-IDS-2017** benchmark corpus, the system is architected to address modern
        enterprise cybersecurity challenges, including low latency requirements, class imbalance across rare
        attack vectors, and explainability of automated alert decisions.
        """
    )

    section_header(
        title="Theoretical Algorithmic Justification",
        description="Why these three distinct machine learning paradigms were selected:",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **Support Vector Machine (SVM)**
            - *Paradigm:* Structural Risk Minimization.
            - *Rationale:* Effective in high-dimensional feature spaces with clear margins of separation. Provides strong mathematical guarantees against overfitting.
            """
        )
    with col2:
        st.markdown(
            """
            **Multilayer Perceptron (MLP)**
            - *Paradigm:* Feedforward Neural Architecture.
            - *Rationale:* Models complex non-linear correlations and cross-feature interactions across expected CIC-IDS-2017 continuous network flow statistics (specification awaiting validation in CP-04/05).
            """
        )
    with col3:
        st.markdown(
            """
            **Decision Tree (CART)**
            - *Paradigm:* Recursive Binary Partitioning.
            - *Rationale:* Deterministic hierarchical decision logic, directly interpretable split paths, and robust handling of non-linear attribute thresholds.
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
