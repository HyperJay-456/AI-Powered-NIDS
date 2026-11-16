"""Page and section headers for consistent view structure."""

import html
from typing import List, Optional, Tuple
import streamlit as st

from dashboard.components.badges import badge_html, BadgeVariant


def page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    badge_text: Optional[str] = None,
    badge_variant: BadgeVariant = "info",
    badges: Optional[List[Tuple[str, BadgeVariant]]] = None,
) -> None:
    """Render a standardized page header with optional icon and status badges.

    Args:
        title: Main page title text.
        subtitle: Optional descriptive subtitle.
        icon: Optional emoji or symbol icon.
        badge_text: Optional single text for right-aligned status badge.
        badge_variant: Variant for the single badge.
        badges: Optional list of (label, variant) tuples for multiple badges.
    """
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle) if subtitle else ""
    safe_icon = html.escape(icon) if icon else ""

    badge_elements: List[str] = []
    if badges:
        for b_text, b_var in badges:
            badge_elements.append(badge_html(label=b_text, variant=b_var))
    elif badge_text:
        badge_elements.append(badge_html(label=badge_text, variant=badge_variant))

    badge_markup = "".join(badge_elements)
    icon_markup = f'<span class="nids-header-icon">{safe_icon}</span>' if safe_icon else ""
    subtitle_markup = (
        f'<p class="nids-header-subtitle">{safe_subtitle}</p>' if safe_subtitle else ""
    )

    header_html = f"""
<div class="nids-page-header">
    <div class="nids-header-topline">
        <div class="nids-header-title-group">
            {icon_markup}
            <h1 class="nids-header-title">{safe_title}</h1>
        </div>
        <div class="nids-header-meta-group">
            {badge_markup}
        </div>
    </div>
    {subtitle_markup}
</div>
"""
    st.markdown(header_html, unsafe_allow_html=True)


def section_header(
    title: str,
    description: Optional[str] = None,
    divider: bool = True,
) -> None:
    """Render a section header with optional description and divider.

    Args:
        title: Section title.
        description: Optional brief explanatory text.
        divider: Whether to show a dividing line below.
    """
    safe_title = html.escape(title)
    safe_desc = html.escape(description) if description else ""

    desc_html = f'<p style="color: var(--color-text-secondary); margin: 0.25rem 0 0.75rem 0; font-size: 0.875rem;">{safe_desc}</p>' if safe_desc else ""
    divider_html = '<div style="height: 1px; background-color: var(--color-border-subtle); margin-bottom: 1rem;"></div>' if divider else ""

    html_content = f"""
<div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">
    <h3 style="font-size: 1.125rem; font-weight: 600; color: var(--color-text-primary); margin: 0;">{safe_title}</h3>
    {desc_html}
    {divider_html}
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)
