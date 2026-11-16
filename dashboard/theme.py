"""Theme management and CSS injection for the NIDS Streamlit Application.

This module coordinates loading the centralized design tokens and injecting
the custom cybersecurity design stylesheet into the active Streamlit session.
"""

from pathlib import Path
from typing import Optional
import streamlit as st

from dashboard.design_tokens import tokens

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"
STYLESHEET_PATH: Path = ASSETS_DIR / "styles.css"


def load_stylesheet() -> str:
    """Read the stylesheet from disk.

    Returns:
        str: Raw CSS stylesheet content.
    """
    if STYLESHEET_PATH.exists():
        return STYLESHEET_PATH.read_text(encoding="utf-8")
    return ""


def inject_theme(custom_css: Optional[str] = None) -> None:
    """Inject the design tokens CSS variables and enterprise stylesheet into Streamlit.

    Args:
        custom_css: Optional additional CSS rules to append.
    """
    css_vars = tokens.to_css_variables()
    base_styles = load_stylesheet()

    combined_css = f"""
<style>
/* Design System CSS Custom Properties */
{css_vars}

/* Application Component Styles */
{base_styles}

/* Extra Inline Custom Styles */
{custom_css or ""}
</style>
"""
    st.markdown(combined_css, unsafe_allow_html=True)
