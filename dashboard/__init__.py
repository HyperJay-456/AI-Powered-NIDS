"""Dashboard presentation layer package."""

from dashboard.design_tokens import tokens
from dashboard.navigation import ROUTES, render_shell
from dashboard.theme import inject_theme

__all__ = ["tokens", "inject_theme", "ROUTES", "render_shell"]
