"""Alert, notification, and implementation status panels."""

import html
from typing import List, Optional
import streamlit as st


def _render_panel(
    variant: str,
    title: str,
    message: str,
    icon: str,
) -> None:
    """Internal helper to render a styled panel."""
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    safe_icon = html.escape(icon)

    panel_html = f"""
<div class="nids-panel nids-panel-{variant}">
    <div class="nids-panel-icon">{safe_icon}</div>
    <div class="nids-panel-content">
        <div class="nids-panel-title">{safe_title}</div>
        <div>{safe_message}</div>
    </div>
</div>
"""
    st.markdown(panel_html, unsafe_allow_html=True)


def info_panel(title: str, message: str, icon: str = "ℹ️") -> None:
    """Render an informational notice panel."""
    _render_panel("info", title, message, icon)


def warning_panel(title: str, message: str, icon: str = "⚠️") -> None:
    """Render a warning notice panel."""
    _render_panel("warning", title, message, icon)


def error_panel(title: str, message: str, icon: str = "🛑") -> None:
    """Render an error notice panel."""
    _render_panel("error", title, message, icon)


def success_panel(title: str, message: str, icon: str = "✅") -> None:
    """Render a success confirmation panel."""
    _render_panel("success", title, message, icon)


def pending_panel(
    checkpoint_id: str,
    title: str,
    description: str,
    planned_capabilities: Optional[List[str]] = None,
) -> None:
    """Render an implementation-pending banner for future checkpoints.

    Args:
        checkpoint_id: Checkpoint identifier (e.g. "CP-04").
        title: Target checkpoint title.
        description: Summary of upcoming capability.
        planned_capabilities: List of planned feature items.
    """
    safe_cp = html.escape(checkpoint_id)
    safe_title = html.escape(title)
    safe_desc = html.escape(description)

    items_html = ""
    if planned_capabilities:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in planned_capabilities)
        items_html = f"""
        <div style="margin-top: 0.75rem;">
            <div style="font-size: 0.8125rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-text-secondary); margin-bottom: 0.375rem;">Planned Capabilities:</div>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.875rem; color: var(--color-text-secondary); line-height: 1.6;">
                {items}
            </ul>
        </div>
        """

    markup = f"""
<div class="nids-panel nids-panel-pending">
    <div class="nids-panel-icon">🔒</div>
    <div class="nids-panel-content">
        <div class="nids-panel-title">Stage Guardrail Active &bull; {safe_cp}: {safe_title}</div>
        <div>{safe_desc}</div>
        {items_html}
    </div>
</div>
"""
    st.markdown(markup, unsafe_allow_html=True)
