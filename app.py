"""AI-Powered Network Intrusion Detection System (NIDS) - Application Shell.

Master entry point for the Streamlit presentation layer. Initialized with
centralized design system tokens, persistent route management, and modular view routing.
"""

from pathlib import Path
import sys

# Ensure workspace root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from config.settings import settings


def main() -> None:
    """Application entry point and master presentation shell."""
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in the active environment.")
        print("Install dependencies using: pip install -r requirements.txt")
        print("Launch the dashboard using: streamlit run app.py")
        return

    # Ensure required runtime folders exist
    settings.ensure_directories()

    # Configure Streamlit page layout and title
    st.set_page_config(
        page_title=settings.ui.app_title,
        page_icon=settings.ui.app_icon,
        layout=settings.ui.layout,
        initial_sidebar_state=settings.ui.initial_sidebar_state,
    )

    # Import dashboard components after Streamlit is confirmed available
    from dashboard.navigation import render_shell
    from dashboard.theme import inject_theme

    # Inject design system tokens and custom cybersecurity stylesheet
    inject_theme()

    # Render persistent sidebar navigation and active page view
    render_shell()


if __name__ == "__main__":
    main()
