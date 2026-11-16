"""Reusable UI component library for the NIDS Dashboard."""

from dashboard.components.badges import badge_html, render_badge
from dashboard.components.cards import content_card, metric_card
from dashboard.components.headers import page_header, section_header
from dashboard.components.panels import (
    error_panel,
    info_panel,
    pending_panel,
    success_panel,
    warning_panel,
)
from dashboard.components.actions import primary_button, secondary_button, styled_table

__all__ = [
    "badge_html",
    "render_badge",
    "metric_card",
    "content_card",
    "page_header",
    "section_header",
    "info_panel",
    "warning_panel",
    "error_panel",
    "success_panel",
    "pending_panel",
    "primary_button",
    "secondary_button",
    "styled_table",
]
