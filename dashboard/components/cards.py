"""Metric cards and bordered container components."""

import html
from typing import Optional
import streamlit as st


def metric_card(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    status_color: Optional[str] = None,
) -> None:
    """Render a styled cybersecurity metric card.

    Args:
        label: Metric category or KPI title.
        value: Formatted metric value (e.g. "98.4%", "12,450", "Idle").
        subtitle: Optional secondary descriptor or caption.
        delta: Optional delta or trend indicator (e.g. "+1.2%").
        status_color: Optional hex or css color override for value text.
    """
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_sub = html.escape(subtitle) if subtitle else ""
    safe_delta = html.escape(delta) if delta else ""

    value_style = f' style="color: {status_color};"' if status_color else ""
    delta_markup = f'<span style="font-size: 0.8125rem; color: var(--color-status-success); margin-left: 0.5rem;">{safe_delta}</span>' if safe_delta else ""
    sub_markup = f'<div class="nids-metric-subtitle">{safe_sub}</div>' if safe_sub else ""

    card_html = f"""
<div class="nids-metric-card">
    <div class="nids-metric-label">{safe_label}</div>
    <div class="nids-metric-value"{value_style}>{safe_value}{delta_markup}</div>
    {sub_markup}
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def content_card(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    content: Optional[str] = None,
) -> None:
    """Render a standalone content container card with styled borders.

    Args:
        title: Card header title.
        subtitle: Card header subtitle.
        content: Inner body text or HTML.
    """
    safe_title = f'<div class="nids-card-title">{html.escape(title)}</div>' if title else ""
    safe_sub = f'<div class="nids-card-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    body = content or ""

    card_html = f"""
<div class="nids-card">
    {safe_title}
    {safe_sub}
    <div>{body}</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)
