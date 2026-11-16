"""Unit tests for CP-02 UI Design System and Navigation Shell."""

import unittest
from pathlib import Path

from dashboard.design_tokens import tokens
from dashboard.navigation import ROUTE_MAP, ROUTES, Route
from dashboard.theme import load_stylesheet, STYLESHEET_PATH
from dashboard.components.badges import badge_html


class TestUIDesignSystem(unittest.TestCase):
    """Test suite for UI design tokens, theme stylesheet, and navigation."""

    def test_design_tokens_colors(self) -> None:
        """Verify centralized color tokens strictly match enterprise cybersecurity specification."""
        colors = tokens.colors
        self.assertEqual(colors.background, "#0B1020")
        self.assertEqual(colors.secondary_background, "#111827")
        self.assertEqual(colors.surface, "#172033")
        self.assertEqual(colors.surface_elevated, "#1E293B")
        self.assertEqual(colors.primary_accent, "#6385FF")
        self.assertEqual(colors.secondary_accent, "#8B5CF6")
        self.assertEqual(colors.text_primary, "#F1F5F9")
        self.assertEqual(colors.text_secondary, "#94A3B8")
        self.assertEqual(colors.border_default, "#29364B")

    def test_semantic_status_colors(self) -> None:
        """Verify semantic indicator color definitions."""
        colors = tokens.colors
        self.assertEqual(colors.status_success, "#10B981")
        self.assertEqual(colors.status_warning, "#F59E0B")
        self.assertEqual(colors.status_error, "#EF4444")
        self.assertEqual(colors.status_info, "#3B82F6")
        self.assertEqual(colors.status_neutral, "#64748B")

    def test_css_variables_export(self) -> None:
        """Verify CSS custom properties string generation."""
        css = tokens.to_css_variables()
        self.assertIn("--color-bg: #0B1020;", css)
        self.assertIn("--color-bg-secondary: #111827;", css)
        self.assertIn("--color-accent-primary: #6385FF;", css)
        self.assertIn("--color-text-primary: #F1F5F9;", css)
        self.assertIn("--color-border-default: #29364B;", css)

    def test_stylesheet_loading(self) -> None:
        """Verify custom stylesheet file exists and contains expected component classes."""
        self.assertTrue(STYLESHEET_PATH.exists(), f"Missing stylesheet at {STYLESHEET_PATH}")
        css_content = load_stylesheet()
        self.assertGreater(len(css_content), 100)
        self.assertIn(".nids-page-header", css_content)
        self.assertIn(".nids-metric-card", css_content)
        self.assertIn(".nids-badge", css_content)
        self.assertIn(".nids-panel", css_content)

    def test_navigation_routes_count_and_keys(self) -> None:
        """Verify exactly eight functional routes are configured."""
        self.assertEqual(len(ROUTES), 8)
        expected_route_ids = {
            "overview",
            "traffic_analysis",
            "detection_center",
            "model_laboratory",
            "threat_intelligence",
            "prediction_explorer",
            "reports",
            "about",
        }
        self.assertEqual(set(ROUTE_MAP.keys()), expected_route_ids)

    def test_route_properties_and_callables(self) -> None:
        """Verify each route has proper labels, icons, descriptions, and callable render functions."""
        for route in ROUTES:
            self.assertIsInstance(route, Route)
            self.assertTrue(bool(route.label))
            self.assertTrue(bool(route.icon))
            self.assertTrue(bool(route.description))
            self.assertTrue(callable(route.render_fn), f"render_fn for {route.route_id} must be callable")

    def test_badge_html_sanitization(self) -> None:
        """Verify XSS sanitization in badge rendering."""
        malicious = "<script>alert('xss')</script>"
        safe_html = badge_html(malicious, variant="error")
        self.assertNotIn("<script>", safe_html)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", safe_html)
        self.assertIn("nids-badge-error", safe_html)

    def test_streamlit_config_toml_present(self) -> None:
        """Verify .streamlit/config.toml exists with dark theme palette."""
        config_path = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
        self.assertTrue(config_path.exists(), ".streamlit/config.toml must exist")
        content = config_path.read_text(encoding="utf-8")
        self.assertIn('primaryColor = "#6385FF"', content)
        self.assertIn('backgroundColor = "#0B1020"', content)
        self.assertIn('secondaryBackgroundColor = "#111827"', content)
        self.assertIn('textColor = "#F1F5F9"', content)

    def test_page_header_multiple_badges_rendering(self) -> None:
        """Verify page_header correctly renders multiple status badges."""
        from dashboard.components.headers import page_header
        # Verify function accepts badges list without error
        badges = [("Mode: Local / Offline", "neutral"), ("CP-03 Active", "info")]
        try:
            page_header(title="Test Header", badges=badges)
        except Exception as e:
            self.fail(f"page_header raised exception with multiple badges: {e}")


if __name__ == "__main__":
    unittest.main()
