"""Navigation controller and application shell routing for the NIDS Dashboard.

Manages active routes, session-state persistence across reruns, sidebar layout,
and route-to-view delegation for the 8 functional views.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import streamlit as st

from config.settings import settings
from dashboard.pages import (
    about,
    detection_center,
    model_laboratory,
    overview,
    prediction_explorer,
    reports,
    threat_intelligence,
    traffic_analysis,
)
from dashboard.status_checker import check_model_status

SESSION_ROUTE_KEY = "nids_active_route"


@dataclass(frozen=True)
class Route:
    """Navigation route definition."""

    route_id: str
    label: str
    icon: str
    description: str
    render_fn: Callable[[], None]

    @property
    def display_name(self) -> str:
        """Formatted display string for radio navigation."""
        return f"{self.icon}  {self.label}"


# Master Route Registry (8 Authorized Routes)
ROUTES: List[Route] = [
    Route(
        route_id="overview",
        label="Overview",
        icon="🛡️",
        description="Executive telemetry & application dashboard",
        render_fn=overview.render,
    ),
    Route(
        route_id="traffic_analysis",
        label="Traffic Analysis",
        icon="📊",
        description="CSV flow upload, validation & explorer",
        render_fn=traffic_analysis.render,
    ),
    Route(
        route_id="detection_center",
        label="Detection Center",
        icon="⚡",
        description="CSV-based intrusion detection console",
        render_fn=detection_center.render,
    ),
    Route(
        route_id="model_laboratory",
        label="Model Laboratory",
        icon="🧪",
        description="SVM, MLP, and Decision Tree benchmark lab",
        render_fn=model_laboratory.render,
    ),
    Route(
        route_id="threat_intelligence",
        label="Threat Intelligence",
        icon="🌐",
        description="CIC-IDS-2017 attack taxonomy & signatures",
        render_fn=threat_intelligence.render,
    ),
    Route(
        route_id="prediction_explorer",
        label="Prediction Explorer",
        icon="🔍",
        description="Model explainability & feature attribution",
        render_fn=prediction_explorer.render,
    ),
    Route(
        route_id="reports",
        label="Reports",
        icon="📑",
        description="Executive summaries & forensic log exports",
        render_fn=reports.render,
    ),
    Route(
        route_id="about",
        label="About",
        icon="ℹ️",
        description="Academic research context & ethics disclaimer",
        render_fn=about.render,
    ),
]

ROUTE_MAP: Dict[str, Route] = {r.route_id: r for r in ROUTES}
DISPLAY_NAME_TO_ROUTE: Dict[str, Route] = {r.display_name: r for r in ROUTES}


def init_navigation_state() -> None:
    """Initialize navigation session state if not already present."""
    if SESSION_ROUTE_KEY not in st.session_state:
        st.session_state[SESSION_ROUTE_KEY] = "overview"


def get_active_route_id() -> str:
    """Retrieve the current active route identifier.

    Returns:
        str: Route ID (default: "overview").
    """
    init_navigation_state()
    return st.session_state.get(SESSION_ROUTE_KEY, "overview")


def set_active_route(route_id: str) -> None:
    """Programmatically set the active navigation route.

    Args:
        route_id: Target route identifier.
    """
    if route_id in ROUTE_MAP:
        st.session_state[SESSION_ROUTE_KEY] = route_id


def render_sidebar() -> Route:
    """Render the sidebar branding, navigation radio, and system health status.

    Returns:
        Route: The currently selected active route object.
    """
    init_navigation_state()
    current_route_id = get_active_route_id()

    # 1. Branding Header
    brand_markup = """
<div class="nids-brand">
    <div class="nids-brand-icon">🛡️</div>
    <div>
        <div class="nids-brand-title">CYBER-NIDS</div>
        <div class="nids-brand-caption"><span class="nids-pulse-dot"></span>SOC Threat Operations</div>
    </div>
</div>
"""
    st.sidebar.markdown(brand_markup, unsafe_allow_html=True)

    # 2. Route Navigation
    display_options = [r.display_name for r in ROUTES]
    current_route = ROUTE_MAP.get(current_route_id, ROUTES[0])
    current_index = display_options.index(current_route.display_name)

    selected_display = st.sidebar.radio(
        label="CONSOLE NAVIGATION",
        options=display_options,
        index=current_index,
        key="nids_navigation_radio",
        label_visibility="visible",
    )

    selected_route = DISPLAY_NAME_TO_ROUTE[selected_display]
    if selected_route.route_id != current_route_id:
        st.session_state[SESSION_ROUTE_KEY] = selected_route.route_id

    # 3. System Telemetry Widget in Sidebar
    model_stat = check_model_status()
    models_label = f"{model_stat.trained_count}/{model_stat.total_models} Calibrated" if model_stat.trained_count > 0 else "0/3 Initializing"
    models_color = "#34d399" if model_stat.trained_count > 0 else "#94a3b8"

    status_markup = f"""
<div class="nids-status-widget">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.625rem;">
        <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--bb-text-muted);">
            SYSTEM TELEMETRY
        </span>
        <span class="nids-badge nids-badge-success" style="font-size: 0.6875rem; padding: 0.15rem 0.5rem;">
            LIVE
        </span>
    </div>
    <div class="nids-status-row">
        <span class="nids-status-label">Engine</span>
        <span style="color: #ffffff; font-weight: 600;">Enterprise Defense</span>
    </div>
    <div class="nids-status-row">
        <span class="nids-status-label">Corpus Baseline</span>
        <span style="color: #818cf8; font-weight: 500;">{settings.dataset.name}</span>
    </div>
    <div class="nids-status-row">
        <span class="nids-status-label">Estimators</span>
        <span style="color: {models_color}; font-weight: 600;">{models_label}</span>
    </div>
    <div class="nids-status-row">
        <span class="nids-status-label">Inference Mode</span>
        <span style="color: #cbd5e1; font-weight: 500;">Tri-Model Consensus</span>
    </div>
</div>
"""
    st.sidebar.markdown(status_markup, unsafe_allow_html=True)

    # 4. Enterprise Console Footer
    st.sidebar.markdown(
        """
        <div style="margin-top: 2.5rem; font-size: 0.75rem; color: #64748b; text-align: center; line-height: 1.5;">
            CYBER-NIDS Operations Console<br>
            <span style="color: #475569; font-size: 0.6875rem;">BinaryNbeyond Cosmic Architecture</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_route


def render_shell() -> None:
    """Coordinate layout, sidebar navigation, and active page rendering."""
    active_route = render_sidebar()

    # Route delegation: render active page view
    active_route.render_fn()
