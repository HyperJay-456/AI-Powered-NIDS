"""Threat Intelligence Page: Attack taxonomy mapping and signature definitions."""

import streamlit as st

from config.settings import settings
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import pending_panel


def render() -> None:
    """Render the Threat Intelligence view."""
    page_header(
        title="Threat Intelligence & Attack Taxonomy",
        subtitle="Cyber threat intelligence mapping, attack signature definitions, and MITRE categorization",
        icon="🌐",
        badge_text="Scheduled for CP-09",
        badge_variant="info",
    )

    pending_panel(
        checkpoint_id="CP-09",
        title="Threat Intelligence & Prediction Explorer",
        description=(
            "Provides deep threat intelligence analytics mapping detected intrusion "
            "events against verified CIC-IDS-2017 attack taxonomies and defensive cybersecurity frameworks."
        ),
        planned_capabilities=[
            "Taxonomy mapping across Denial of Service, DDoS, Reconnaissance, Brute Force, and Web Attacks",
            "Interactive signature breakdown explaining primary flow discriminators for each threat",
            "Mitigation playbooks and defensive remediation rules",
            "Attack frequency and threat prevalence visualizers",
        ],
    )

    section_header(
        title="Verified Attack Taxonomy in CIC-IDS-2017",
        description="The 15 target traffic classifications defined in project settings:",
    )

    cols = st.columns(3)
    for idx, threat in enumerate(settings.dataset.known_attack_classes):
        with cols[idx % 3]:
            st.code(threat, language="text")
