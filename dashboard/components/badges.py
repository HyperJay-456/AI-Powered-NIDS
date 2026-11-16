"""Status and semantic indicator badges."""

import html
from typing import Literal
import streamlit as st

BadgeVariant = Literal["success", "warning", "error", "info", "neutral"]


def badge_html(
    label: str,
    variant: BadgeVariant = "neutral",
    show_dot: bool = True,
) -> str:
    """Generate HTML string for a styled status badge.

    Args:
        label: Text to display inside the badge.
        variant: Semantic color variant.
        show_dot: Whether to display a pulsing status dot.

    Returns:
        str: Sanitized HTML string.
    """
    safe_label = html.escape(str(label))
    dot_html = '<span class="nids-badge-dot"></span>' if show_dot else ""
    return (
        f'<span class="nids-badge nids-badge-{variant}">'
        f'{dot_html}{safe_label}'
        f'</span>'
    )


def render_badge(
    label: str,
    variant: BadgeVariant = "neutral",
    show_dot: bool = True,
) -> None:
    """Render a status badge directly to the Streamlit view.

    Args:
        label: Text to display inside the badge.
        variant: Semantic color variant.
        show_dot: Whether to display a status dot.
    """
    st.markdown(
        badge_html(label=label, variant=variant, show_dot=show_dot),
        unsafe_allow_html=True,
    )
